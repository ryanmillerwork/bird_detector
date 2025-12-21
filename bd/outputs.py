from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2

from .yolo_detect import draw_detections


def _safe_label(label: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(label))


def _retain_last_n(glob_dir: Path, pattern: str, *, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    files = sorted(
        [p for p in glob_dir.glob(pattern) if p.is_file()],
        key=lambda p: p.stat().st_mtime,
    )
    for old in files[:-keep_last_n]:
        try:
            old.unlink()
        except OSError:
            pass


def save_detection_outputs(
    *,
    frame,
    detections: list[dict],
    ts: str,
    detections_dir: Path,
    crops_dir: Path,
    keep_last_annotated: int = 10,
) -> Path | None:
    """
    Save annotated frame + crops.

    - Annotated full frame -> detections/detection_<ts>.jpg
    - Crops -> crops/<label>/<ts>[_N].jpg
    """
    if not detections:
        return None

    detections_dir.mkdir(exist_ok=True)
    crops_dir.mkdir(exist_ok=True)

    annotated = draw_detections(frame, detections)
    annotated_path = detections_dir / f"detection_{ts}.jpg"
    cv2.imwrite(str(annotated_path), annotated)
    _retain_last_n(detections_dir, "detection_*.jpg", keep_last_n=keep_last_annotated)

    for det in detections:
        label = det.get("species") or det["class"]
        dest_dir = crops_dir / _safe_label(label)
        dest_dir.mkdir(parents=True, exist_ok=True)

        base = dest_dir / f"{ts}.jpg"
        dest_path = base
        counter = 1
        while dest_path.exists():
            dest_path = dest_dir / f"{ts}_{counter}.jpg"
            counter += 1
        cv2.imwrite(str(dest_path), det["crop"])
        # Expose where this crop landed for downstream integrations (e.g., DB logging).
        det["crop_path"] = str(dest_path)

    # Expose where the annotated frame landed for downstream integrations (e.g., DB logging).
    return annotated_path


def format_detection_summary(detections: Iterable[dict]) -> str:
    def summarize(det: dict) -> str:
        base = f"{det['class']}({det['confidence']:.2f})"
        if det.get("species"):
            base += f"->{det['species']}({det['species_confidence']:.2f})"
        return base

    return ", ".join(summarize(d) for d in detections)


