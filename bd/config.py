from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .env import env_bool, env_float, env_int


@dataclass(frozen=True)
class CaptureConfig:
    jpeg_url: str
    jpeg_timeout_s: float
    capture_interval_s: float


@dataclass(frozen=True)
class DetectConfig:
    conf: float
    padding: int


@dataclass(frozen=True)
class OutputConfig:
    detections_dir: Path
    crops_dir: Path
    keep_last_annotated: int


@dataclass(frozen=True)
class TTSConfig:
    enabled: bool
    piper_model: Path
    min_conf: float
    cooldown_s: float
    max_queue: int
    preroll_ms: int
    bird_songs_enabled: bool
    bird_songs_dir: Path
    bird_songs_max_s: float
    same_species_repeat_s: float
    song_every_s: float


@dataclass(frozen=True)
class RuntimeConfig:
    base_dir: Path
    capture: CaptureConfig
    detect: DetectConfig
    output: OutputConfig
    tts: TTSConfig
    models_dir: Path
    yolo_weights: Path

    @classmethod
    def from_env(cls, *, base_dir: Path) -> "RuntimeConfig":
        frigate_host = os.environ.get("FRIGATE_HOST", "192.168.0.50:5000")
        frigate_camera = os.environ.get("FRIGATE_CAMERA", "bird")
        jpeg_url = os.environ.get("JPEG_URL", f"http://{frigate_host}/api/{frigate_camera}/latest.jpg")

        capture = CaptureConfig(
            jpeg_url=jpeg_url,
            jpeg_timeout_s=env_float("JPEG_TIMEOUT_S", 3.0),
            capture_interval_s=env_float("CAPTURE_INTERVAL_S", 2.0),
        )

        detect = DetectConfig(
            conf=env_float("DETECT_CONF", 0.25),
            padding=env_int("DETECT_PADDING", 100),
        )

        output = OutputConfig(
            detections_dir=Path("detections"),
            crops_dir=Path("crops"),
            keep_last_annotated=env_int("KEEP_LAST_ANNOTATED", 10),
        )

        tts = TTSConfig(
            enabled=env_bool("TTS_ENABLED", True),
            piper_model=Path(
                os.environ.get(
                    "TTS_PIPER_MODEL",
                    str(base_dir / "tts_models" / "piper" / "en_US-libritts_r-medium.onnx"),
                )
            ),
            min_conf=env_float("TTS_MIN_CONF", 0.0),
            cooldown_s=env_float("TTS_COOLDOWN_S", 15.0),
            max_queue=env_int("TTS_MAX_QUEUE", 10),
            preroll_ms=env_int("TTS_PREROLL_MS", 650),
            bird_songs_enabled=env_bool("BIRD_SONGS_ENABLED", True),
            bird_songs_dir=Path(os.environ.get("BIRD_SONGS_DIR", str(base_dir / "bird_songs"))),
            bird_songs_max_s=env_float("BIRD_SONGS_MAX_S", 10.0),
            same_species_repeat_s=env_float("TTS_SAME_SPECIES_REPEAT_S", 60.0),
            song_every_s=env_float("BIRD_SONGS_EVERY_S", 10.0),
        )

        return cls(
            base_dir=base_dir,
            capture=capture,
            detect=detect,
            output=output,
            tts=tts,
            models_dir=base_dir / "models",
            yolo_weights=base_dir / "yolov8s.pt",
        )





