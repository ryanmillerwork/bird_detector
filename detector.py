#!/usr/bin/env python3
"""
Bird/animal detector using YOLOv8s.
Stage 1 of the detection pipeline.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import json
import os
import subprocess
import threading
import queue
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def load_env():
    """Load credentials from .env file."""
    env = {}
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().strip().split("\n"):
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                env[key.strip()] = val.strip()
    return env


_env = load_env()
RTSP_URL_OVERRIDE = _env.get("RTSP_URL") or None
# Default to go2rtc stream (user can override via .env)
RTSP_PATH = _env.get("RTSP_PATH", "rtsp://192.168.0.50:8554/bird_cam")
RTSP_TRANSPORT = _env.get("RTSP_TRANSPORT", "tcp")  # passed as FFmpeg option, not query param
RTSP_TIMEOUT_US = _env.get("RTSP_TIMEOUT_US", "5000000")  # FFmpeg stimeout in microseconds
RTSP_MAX_DELAY_US = _env.get("RTSP_MAX_DELAY_US", "0")   # FFmpeg max_delay (0 = disable buffering)
RTSP_BUFFER_SIZE = _env.get("RTSP_BUFFER_SIZE", "0")     # FFmpeg buffer_size bytes (0 = default)
SNAPSHOT_MODE = _env.get("SNAPSHOT_MODE", "false").lower() in ("1", "true", "yes")
try:
    RTSP_TIMEOUT_US = int(RTSP_TIMEOUT_US) if RTSP_TIMEOUT_US else None
except ValueError:
    RTSP_TIMEOUT_US = None
try:
    RTSP_MAX_DELAY_US = int(RTSP_MAX_DELAY_US) if RTSP_MAX_DELAY_US else None
except ValueError:
    RTSP_MAX_DELAY_US = None
try:
    RTSP_BUFFER_SIZE = int(RTSP_BUFFER_SIZE) if RTSP_BUFFER_SIZE else None
except ValueError:
    RTSP_BUFFER_SIZE = None


def build_rtsp_url():
    if RTSP_URL_OVERRIDE:
        return RTSP_URL_OVERRIDE
    # Allow RTSP_PATH to be a full URL; if so, use it verbatim
    if RTSP_PATH.startswith("rtsp://"):
        return RTSP_PATH
    required = ("CAMERA_USER", "CAMERA_PASS", "CAMERA_IP", "CAMERA_PORT")
    missing = [k for k in required if not _env.get(k)]
    if missing:
        raise ValueError(
            "RTSP_PATH is not a full rtsp:// URL, but camera credentials are missing.\n"
            f"Missing: {', '.join(missing)}\n"
            "Fix: set RTSP_URL to a full URL (recommended with go2rtc) or set CAMERA_USER/CAMERA_PASS/CAMERA_IP/CAMERA_PORT."
        )
    return f"rtsp://{_env['CAMERA_USER']}:{_env['CAMERA_PASS']}@{_env['CAMERA_IP']}:{_env['CAMERA_PORT']}/{RTSP_PATH}"


def set_ffmpeg_capture_options(enable_transport: bool = True, enable_timeout: bool = True, enable_buffer: bool = True):
    opts = []
    if enable_transport and RTSP_TRANSPORT:
        opts.append(f"rtsp_transport;{RTSP_TRANSPORT}")
    if enable_timeout and RTSP_TIMEOUT_US:
        opts.append(f"stimeout;{RTSP_TIMEOUT_US}")
    if enable_buffer:
        if RTSP_MAX_DELAY_US is not None:
            opts.append(f"max_delay;{RTSP_MAX_DELAY_US}")
        if RTSP_BUFFER_SIZE is not None:
            opts.append(f"buffer_size;{RTSP_BUFFER_SIZE}")
    if opts:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(opts)
    else:
        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)


def grab_frame_snapshot(rtsp_url: str, transport: str | None, timeout_us: int | None):
    """Grab a single frame using ffmpeg and return (frame, stderr_text), or (None, stderr_text) on failure."""
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-analyzeduration", "0",
        "-probesize", "32k",
    ]
    if transport:
        cmd += ["-rtsp_transport", transport, "-rtsp_flags", "prefer_tcp"]
    # Some ffmpeg builds lack -stimeout; rely on subprocess timeout instead
    if RTSP_MAX_DELAY_US is not None:
        cmd += ["-max_delay", str(RTSP_MAX_DELAY_US)]
    if RTSP_BUFFER_SIZE is not None:
        cmd += ["-buffer_size", str(RTSP_BUFFER_SIZE)]
    cmd += [
        "-i", rtsp_url,
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-",
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        err = ""
        if hasattr(e, "stderr") and e.stderr:
            err = e.stderr.decode(errors="ignore")
        return None, err
    
    data = result.stdout
    if not data:
        return None, result.stderr.decode(errors="ignore")
    img_array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return frame, result.stderr.decode(errors="ignore")

# COCO classes we care about (animals)
ANIMAL_CLASSES = {
    14: 'bird',
    15: 'cat', 
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
}

# Output / model paths
DETECTIONS_DIR = Path("detections")  # annotated full frames
CROPS_DIR = Path("crops")            # crops organized by classification
MODELS_DIR = Path(__file__).parent / "models"
CLASSIFIER_ONNX = MODELS_DIR / "bird_classifier.onnx"
CLASS_MAPPING = MODELS_DIR / "class_mapping.json"


def detect_animals(model, frame, confidence=0.4, padding=100):
    """Detect animals in frame with padded crops."""
    results = model(frame, conf=confidence, verbose=False)[0]
    h, w = frame.shape[:2]
    
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in ANIMAL_CLASSES:
            continue
        
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Add padding, clamped to image bounds
        crop_x1 = max(0, x1 - padding)
        crop_y1 = max(0, y1 - padding)
        crop_x2 = min(w, x2 + padding)
        crop_y2 = min(h, y2 + padding)
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        
        detections.append({
            'class': ANIMAL_CLASSES[cls_id],
            'confidence': conf,
            'bbox': (x1, y1, x2, y2),
            'crop': crop,
            'species': None,
            'species_confidence': None,
        })
    
    return detections


def draw_detections(frame, detections):
    """Draw bounding boxes on frame."""
    annotated = frame.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['class']} {det['confidence']:.2f}"
        if det.get('species'):
            label += f" -> {det['species']} ({det['species_confidence']:.2f})"
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return annotated


def load_classifier():
    """Load ONNX classifier and class mapping."""
    if not CLASSIFIER_ONNX.exists() or not CLASS_MAPPING.exists():
        print("Classifier not found in models/. Skipping species classification.\n")
        return None
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed. Install with `pip install onnxruntime` to enable species classification.\n")
        return None
    
    # Handle external data format (PyTorch may emit .onnx.data when model >2GB)
    data_path = Path(str(CLASSIFIER_ONNX) + ".data")
    if data_path.exists():
        # Inform user in case file is missing in deployments
        print(f"Using external data file: {data_path.name}")
    try:
        session = ort.InferenceSession(
            str(CLASSIFIER_ONNX),
            providers=["CPUExecutionProvider"],
        )
    except Exception as e:
        if "onnx.data" in str(e) or "file size" in str(e):
            print(f"Could not load classifier (missing external data file?). Expected: {data_path}")
            print(f"Error: {e}\nSkipping species classification.\n")
            return None
        raise
    
    mapping = json.loads(CLASS_MAPPING.read_text())
    idx_to_class = {int(k): v for k, v in mapping.get("idx_to_class", {}).items()}
    
    # Infer input size from ONNX graph (expects [1, 3, H, W])
    input_meta = session.get_inputs()[0]
    shape = input_meta.shape
    input_size = int(shape[2]) if len(shape) >= 3 and shape[2] is not None else 320
    
    print(f"Loaded classifier ({CLASSIFIER_ONNX.name}), input size {input_size}, {len(idx_to_class)} classes.")
    return session, idx_to_class, input_size


def classify_bird(crop, session, idx_to_class, input_size):
    """Run bird species classification on a cropped image."""
    # Preprocess: BGR -> RGB, resize, normalize
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size))
    img = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, 0)  # NCHW
    
    outputs = session.run(None, {"input": img})[0]
    # Softmax to probabilities
    logits = outputs[0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    species = idx_to_class.get(top_idx, f"class_{top_idx}")
    return species, top_prob


def is_corrupted_frame(frame, low_std_thresh=2.0, flat_row_ratio=0.3):
    """Heuristic to reject obviously corrupted frames (common with RTSP packet loss)."""
    if frame is None or frame.size == 0:
        return True
    
    # If the frame is effectively a single color, treat as corrupted
    if len(np.unique(frame)) <= 1:
        return True
    
    # Check bottom half for large runs of near-flat rows (typical when decode fails mid-frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    bottom = gray[h // 2 :]
    row_std = bottom.std(axis=1)
    flat_rows = (row_std < low_std_thresh).sum()
    if flat_rows / max(len(row_std), 1) > flat_row_ratio:
        return True
    
    return False


def main():
    """Run detection pipeline with producer/consumer frame flow."""
    DETECTIONS_DIR.mkdir(exist_ok=True)
    CROPS_DIR.mkdir(exist_ok=True)
    
    classifier = load_classifier()
    
    print("Loading YOLOv8s model...")
    model = YOLO("yolov8s.pt")
    print("Model loaded.")
    
    try:
        connect_url = build_rtsp_url()
    except ValueError as e:
        print(str(e))
        return
    if RTSP_URL_OVERRIDE or RTSP_PATH.startswith("rtsp://"):
        print("Connecting to RTSP stream... (explicit URL)")
    else:
        print(f"Connecting to RTSP stream... (transport={RTSP_TRANSPORT or 'default'}, path={RTSP_PATH})")
    
    frame_queue: queue.Queue = queue.Queue(maxsize=1)  # holds (frame, grab_ms)
    stop_event = threading.Event()
    cap_ref = {"cap": None}  # to allow cleanup in producer
    consecutive_snap_failures = 0
    last_snap_error_log = 0.0

    def producer():
        nonlocal consecutive_snap_failures, last_snap_error_log, connect_url
        if SNAPSHOT_MODE:
            print("Snapshot mode enabled: producer grabbing frames via ffmpeg as fast as possible.")
        else:
            set_ffmpeg_capture_options(enable_transport=True, enable_timeout=True, enable_buffer=True)
            cap = cv2.VideoCapture(connect_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                if RTSP_URL_OVERRIDE or RTSP_PATH.startswith("rtsp://"):
                    print("Primary RTSP connect failed (explicit URL), no fallback available.")
                    stop_event.set()
                    return
                fallback_url = build_rtsp_url()
                print("Primary RTSP connect failed, retrying without transport/timeout/buffer options...")
                cap.release()
                set_ffmpeg_capture_options(enable_transport=False, enable_timeout=False, enable_buffer=False)
                cap = cv2.VideoCapture(fallback_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                print("Failed to open stream")
                stop_event.set()
                return
            cap_ref["cap"] = cap
            print("Producer connected.")

        while not stop_event.is_set():
            if SNAPSHOT_MODE:
                snap_start = time.time()
                frame, snap_err = grab_frame_snapshot(connect_url, RTSP_TRANSPORT or "tcp", None)
                if frame is None:
                    # Keep the latest non-empty error string for logging
                    last_err = snap_err or ""
                    # Fallback without transport hints
                    frame, snap_err = grab_frame_snapshot(connect_url, None, None)
                    if frame is None and snap_err:
                        last_err = snap_err
                    if frame is None:
                        consecutive_snap_failures += 1
                        now = time.time()
                        # Rate-limit logs but surface first failure and periodic updates
                        if consecutive_snap_failures == 1 or now - last_snap_error_log > 5:
                            prefix = f"[producer] Snapshot failed {consecutive_snap_failures}x in a row"
                            if last_err:
                                first_line = last_err.strip().splitlines()[0]
                                print(f"{prefix}: {first_line}")
                            else:
                                print(prefix)
                            last_snap_error_log = now
                        # Back off to avoid hammering the camera/log spam
                        backoff = min(0.2 * consecutive_snap_failures, 5.0)
                        if backoff > 0:
                            time.sleep(backoff)
                        # Reset connection after repeated failures
                        if consecutive_snap_failures >= 10 and consecutive_snap_failures % 10 == 0:
                            reset_delay = min(2.0 + 0.1 * consecutive_snap_failures, 5.0)
                            print(f"[producer] Resetting after {consecutive_snap_failures} snapshot failures (sleep {reset_delay:.1f}s)...")
                            connect_url = build_rtsp_url()  # reload in case env changed
                            time.sleep(reset_delay)
                            last_snap_error_log = time.time()
                            # keep counting to know streak length
                        continue
                # Success path resets failure counter
                consecutive_snap_failures = 0
                grab_ms = (time.time() - snap_start) * 1000.0
            else:
                cap = cap_ref["cap"]
                if cap is None:
                    break
                read_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("[producer] Failed to grab frame, reconnecting...")
                    cap.release()
                    time.sleep(0.2)
                    set_ffmpeg_capture_options(enable_transport=True, enable_timeout=True, enable_buffer=True)
                    cap = cv2.VideoCapture(connect_url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if not cap.isOpened() and not RTSP_URL_OVERRIDE:
                        fallback_url = build_rtsp_url()
                        cap.release()
                        set_ffmpeg_capture_options(enable_transport=False, enable_timeout=False, enable_buffer=False)
                        cap = cv2.VideoCapture(fallback_url, cv2.CAP_FFMPEG)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if not cap.isOpened():
                        print("[producer] Reconnect failed.")
                        continue
                    cap_ref["cap"] = cap
                    continue
                grab_ms = (time.time() - read_start) * 1000.0

            if is_corrupted_frame(frame):
                print("[producer] Corrupted frame detected, skipping...")
                continue

            produced_at = time.time()
            try:
                if frame_queue.full():
                    frame_queue.get_nowait()
                frame_queue.put_nowait((frame, grab_ms))
            except queue.Full:
                pass

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    print("Connected.\n")
    print("Running detection continuously. Press Ctrl+C to stop.\n")
    
    frame_count = 0
    try:
        while not stop_event.is_set():
            ts = datetime.now().strftime('%H%M%S_%y%m%d')  # hhmmss_yymmdd
            
            try:
                frame, grab_ms = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            grab_time = grab_ms / 1000.0
            
            # Run detection
            detect_start = time.time()
            detections = detect_animals(model, frame)
            detect_time = time.time() - detect_start
            classify_time = 0.0
            
            # Stage 2: classify birds if classifier is available
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
            
            # Save crops per classification and annotated full frame
            if detections:
                # Annotated frame
                annotated = draw_detections(frame, detections)
                annotated_path = DETECTIONS_DIR / f"detection_{ts}.jpg"
                cv2.imwrite(str(annotated_path), annotated)
                
                # Retain only the most recent 10 annotated images
                annotated_files = sorted(
                    [p for p in DETECTIONS_DIR.glob("detection_*.jpg") if p.is_file()],
                    key=lambda p: p.stat().st_mtime
                )
                for old in annotated_files[:-10]:
                    try:
                        old.unlink()
                    except OSError:
                        pass
                
                # Crops organized by classification
                for i, det in enumerate(detections):
                    # Choose folder by classifier output; fallback to detected class
                    label = det.get("species") or det["class"]
                    safe_label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label)
                    dest_dir = CROPS_DIR / safe_label
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    filename = f"{ts}.jpg"
                    dest_path = dest_dir / filename
                    counter = 1
                    while dest_path.exists():
                        dest_path = dest_dir / f"{ts}_{counter}.jpg"
                        counter += 1
                    
                    cv2.imwrite(str(dest_path), det["crop"])
                
                def summarize(det):
                    base = f"{det['class']}({det['confidence']:.2f})"
                    if det.get("species"):
                        base += f"->{det['species']}({det['species_confidence']:.2f})"
                    return base
                
                det_summary = ", ".join(summarize(d) for d in detections)
                print(f"[{ts}] Detected: {det_summary} (grab:{grab_time*1000:.0f}ms, detect:{detect_time*1000:.0f}ms, classify:{classify_time*1000:.0f}ms)")
            else:
                print(f"[{ts}] No animals (grab:{grab_time*1000:.0f}ms, detect:{detect_time*1000:.0f}ms)")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if cap_ref.get("cap") is not None:
            cap_ref["cap"].release()
    
    print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    main()
