#!/usr/bin/env python3
"""
Birdwatch runtime pipeline:

Two-stage pipeline:
  - Stage 1: YOLOv8 (COCO) animal detection on full frames
  - Stage 2: ONNX species classification on detection crops (optional)

This file is intentionally short: implementation details live in `bd/`.
"""

from __future__ import annotations

import queue
import threading
import time
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# NOTE: We intentionally import ultralytics lazily inside main() so we can keep
# startup output clean on CPU-only systems (some native libs can emit warnings
# during import-time device discovery).

from bd.classifier import classify_bird, load_classifier
from bd.config import RuntimeConfig
from bd.capture import capture_loop_jpeg
from bd.env import env_bool, load_dotenv
from bd.frame_filter import is_corrupted_frame
from bd.http_server import start_detections_http_server
from bd.outputs import format_detection_summary, save_detection_outputs
from bd.db import connect as db_connect
from bd.tts import TTSSpeaker
from bd.wildlife_db import create_wildlife_table, row_from_detection, insert_wildlife
from bd.yolo_detect import detect_animals
from bd.mqtt_pub import MQTTPublisher


def main() -> None:
    # Some native deps (notably onnxruntime in some builds) can print a one-time
    # device discovery WARNING during import. We run detector CPU-only; this log
    # is just noisy on Raspberry Pi, so we silence stderr only for imports.
    saved_stderr_fd = None
    devnull_fd = None
    try:
        saved_stderr_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
        from ultralytics import YOLO  # type: ignore
    finally:
        if devnull_fd is not None:
            try:
                os.close(devnull_fd)
            except OSError:
                pass
        if saved_stderr_fd is not None:
            try:
                os.dup2(saved_stderr_fd, 2)
            except OSError:
                pass
            try:
                os.close(saved_stderr_fd)
            except OSError:
                pass

    base_dir = Path(__file__).resolve().parent
    # Allow runtime settings to come from repo-local .env (same file used by training).
    # Real environment variables still take precedence unless override=True is used.
    load_dotenv(base_dir / ".env", override=False)
    cfg = RuntimeConfig.from_env(base_dir=base_dir)

    cfg.output.detections_dir.mkdir(exist_ok=True)
    cfg.output.crops_dir.mkdir(exist_ok=True)

    http_server = None
    if env_bool("DETECTIONS_HTTP_ENABLED", True):
        host = (os.environ.get("DETECTIONS_HTTP_HOST") or "0.0.0.0").strip() or "0.0.0.0"
        port = int((os.environ.get("DETECTIONS_HTTP_PORT") or "8765").strip() or "8765")
        try:
            http_server = start_detections_http_server(
                host=host,
                port=port,
                directory=cfg.output.detections_dir,
            )
            print(f"[http] serving detections on http://{host}:{port}/ (latest: /latest.jpg)")
        except Exception as e:
            print(f"[http] start failed; continuing without HTTP server: {e}")

    # Startup summary (ordered): capture -> detector -> classifier -> tts -> db
    print(
        f"[capture] JPEG {cfg.capture.jpeg_url} every {cfg.capture.capture_interval_s:.3f}s (wall-clock aligned)"
    )

    # Detector (YOLO, CPU-only)
    try:
        model = YOLO(str(cfg.yolo_weights))
        print(f"[detector] loaded {cfg.yolo_weights.name} (cpu)")
    except Exception as e:
        print(f"[detector] load failed: {cfg.yolo_weights} ({e})")
        raise

    classifier = load_classifier(models_dir=cfg.models_dir)

    tts = TTSSpeaker(
        enabled=cfg.tts.enabled,
        base_dir=cfg.base_dir,
        piper_model=cfg.tts.piper_model,
        min_conf=cfg.tts.min_conf,
        cooldown_s=cfg.tts.cooldown_s,
        max_queue=cfg.tts.max_queue,
        preroll_ms=cfg.tts.preroll_ms,
        bird_songs_enabled=cfg.tts.bird_songs_enabled,
        bird_songs_dir=cfg.tts.bird_songs_dir,
        bird_songs_max_s=cfg.tts.bird_songs_max_s,
    )

    mqtt = MQTTPublisher(cfg.mqtt)

    # Announcement policy:
    # - "No detection" does not reset the current species (blue_jay -> none -> blue_jay is not a change)
    # - Name: speak immediately on a real species change; otherwise at most once per minute
    # - Song: play while detected at most once per 10 seconds
    same_species_repeat_s = cfg.tts.same_species_repeat_s
    song_every_s = cfg.tts.song_every_s
    current_species: str | None = None
    last_name_spoken_at: dict[str, float] = {}
    last_song_played_at: dict[str, float] = {}

    db_conn = None
    db_enabled = env_bool("DB_ENABLED", False)
    wildlife_table = None
    video_source = None
    if db_enabled:
        wildlife_table = (os.environ.get("WILDLIFE_TABLE") or "wildlife").strip() or "wildlife"
        video_source = (os.environ.get("VIDEO_SOURCE") or cfg.capture.jpeg_url).strip() or None
        try:
            db_conn = db_connect(autocommit=True)
            create_wildlife_table(db_conn, table=wildlife_table)
            print(f"[db] enabled: writing detections to table '{wildlife_table}'")
        except Exception as e:
            print(f"[db] connection/setup failed; continuing without DB logging: {e}")
            db_conn = None

    frame_queue: queue.Queue = queue.Queue(maxsize=1)  # holds (scheduled_at_s, frame, grab_ms)
    stop_event = threading.Event()

    def producer() -> None:
        capture_loop_jpeg(
            jpeg_url=cfg.capture.jpeg_url,
            timeout_s=cfg.capture.jpeg_timeout_s,
            interval_s=cfg.capture.capture_interval_s,
            stop_event=stop_event,
            frame_queue=frame_queue,
            is_corrupted_frame=is_corrupted_frame,
        )

    threading.Thread(target=producer, daemon=True).start()
    print("Running detection continuously. Press Ctrl+C to stop.\n")

    frame_count = 0
    try:
        while not stop_event.is_set():
            try:
                scheduled_at, frame, grab_ms = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            ts = datetime.fromtimestamp(scheduled_at).strftime("%H%M%S_%y%m%d")  # hhmmss_yymmdd
            grab_time = float(grab_ms) / 1000.0

            detect_start = time.time()
            detections = detect_animals(
                model,
                frame,
                confidence=cfg.detect.conf,
                padding=cfg.detect.padding,
            )
            detect_time = time.time() - detect_start

            classify_time = 0.0
            best_species: str | None = None
            best_species_conf: float | None = None
            if classifier:
                classify_start = time.time()
                session, idx_to_class, input_size = classifier
                for det in detections:
                    try:
                        species, prob = classify_bird(det["crop"], session, idx_to_class, input_size)
                        det["species"] = species
                        det["species_confidence"] = prob
                    except Exception as e:
                        print(f"[{ts}] Classification error: {e}")
                classify_time = time.time() - classify_start

                # Speak the best species for this frame (if enabled).
                best = None
                best_prob = -1.0
                for det in detections:
                    sp = det.get("species")
                    pr = det.get("species_confidence")
                    if sp and pr is not None and float(pr) > best_prob:
                        best = sp
                        best_prob = float(pr)

                if best is not None:
                    best_species = str(best)
                    best_species_conf = float(best_prob)

                if best is not None and best_prob >= tts.min_conf:
                    now = time.time()
                    changed = current_species is None or best != current_species
                    if changed:
                        current_species = best

                    speak = False
                    if changed:
                        speak = True
                    else:
                        last = last_name_spoken_at.get(best, 0.0)
                        if now - last >= same_species_repeat_s:
                            speak = True

                    play_song = False
                    if tts.bird_songs_enabled:
                        last_song = last_song_played_at.get(best, 0.0)
                        if now - last_song >= song_every_s:
                            play_song = True

                    if speak:
                        last_name_spoken_at[best] = now
                    if play_song:
                        last_song_played_at[best] = now

                    spoken = str(best).replace("_", " ").strip()
                    tts.enqueue(best, spoken, speak=speak, play_song=play_song)

            if detections:
                annotated_path = save_detection_outputs(
                    frame=frame,
                    detections=detections,
                    ts=ts,
                    detections_dir=cfg.output.detections_dir,
                    crops_dir=cfg.output.crops_dir,
                    keep_last_annotated=cfg.output.keep_last_annotated,
                )

                # MQTT publishing:
                # - event: one message per frame where detections exist
                # - state: retained "latest" payload for dashboards
                if cfg.mqtt.enabled and annotated_path is not None:
                    detected_at = datetime.fromtimestamp(float(scheduled_at), tz=timezone.utc)
                    base_url = (os.environ.get("DETECTIONS_BASE_URL") or "").strip().rstrip("/")
                    annotated_image_url = f"{base_url}/latest.jpg" if base_url else None
                    event = {
                        "event_id": f"{ts}-{int(float(scheduled_at) * 1000)}",
                        "ts": detected_at.isoformat(),
                        "epoch_s": float(scheduled_at),
                        "source": cfg.capture.jpeg_url,
                        "annotated_image_path": str(annotated_path),
                        "annotated_image_url": annotated_image_url,
                        "best_species": best_species,
                        "best_species_conf": best_species_conf,
                        "detections": [
                            {
                                "class": det.get("class"),
                                "confidence": det.get("confidence"),
                                "bbox": list(det.get("bbox") or []),
                                "species": det.get("species"),
                                "species_confidence": det.get("species_confidence"),
                                "crop_path": det.get("crop_path"),
                            }
                            for det in detections
                        ],
                    }
                    mqtt.publish_event(event)
                    mqtt.publish_state(event)

                # Optional DB logging: one row per detection.
                if db_conn is not None and annotated_path is not None:
                    detected_at = datetime.fromtimestamp(float(scheduled_at), tz=timezone.utc)
                    detector_model = cfg.yolo_weights.name
                    classifier_model = "bird_classifier.onnx" if classifier else None
                    for det in detections:
                        try:
                            row = row_from_detection(
                                detected_at=detected_at,
                                image_path=str(annotated_path),
                                det=det,
                                detector_model=detector_model,
                                classifier_model=classifier_model,
                                video_source=video_source,
                            )
                            insert_wildlife(db_conn, row, table=wildlife_table)
                        except Exception as e:
                            print(f"[{ts}] DB insert error: {e}")

                det_summary = format_detection_summary(detections)
                print(
                    f"[{ts}] Detected: {det_summary} "
                    f"(grab:{grab_time*1000:.0f}ms, detect:{detect_time*1000:.0f}ms, classify:{classify_time*1000:.0f}ms)"
                )
            else:
                print(f"[{ts}] No animals (grab:{grab_time*1000:.0f}ms, detect:{detect_time*1000:.0f}ms)")

            frame_count += 1
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        tts.stop()
        mqtt.stop()
        try:
            if http_server is not None:
                http_server.stop()
        except Exception:
            pass
        try:
            if db_conn is not None:
                db_conn.close()
        except Exception:
            pass

    print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    main()


