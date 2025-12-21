from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np


Classifier = Tuple[Any, Dict[int, str], int]  # (onnx session, idx_to_class, input_size)


def load_classifier(*, models_dir: Path) -> Classifier | None:
    """Load ONNX classifier and class mapping."""
    classifier_onnx = models_dir / "bird_classifier.onnx"
    class_mapping = models_dir / "class_mapping.json"

    if not classifier_onnx.exists() or not class_mapping.exists():
        print("[classifier] not found in models/ (skipping species classification)\n")
        return None

    try:
        # Keep startup output clean on CPU-only systems.
        # Some builds of onnxruntime emit a one-time device discovery WARNING during import.
        # We still run CPU-only (providers=["CPUExecutionProvider"]), but we silence stderr
        # for the import to avoid the noisy message.
        saved_stderr_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
        try:
            import onnxruntime as ort
        finally:
            try:
                os.close(devnull_fd)
            except OSError:
                pass
            try:
                os.dup2(saved_stderr_fd, 2)
            except OSError:
                pass
            try:
                os.close(saved_stderr_fd)
            except OSError:
                pass
    except ImportError:
        print("[classifier] onnxruntime not installed (skipping species classification)\n")
        return None
    try:
        # Belt-and-suspenders: keep ORT from emitting WARNING logs even after import.
        ort.set_default_logger_severity(3)
    except Exception:
        pass

    # Handle external data format (PyTorch may emit .onnx.data when model >2GB)
    data_path = Path(str(classifier_onnx) + ".data")
    # If .onnx.data exists, ORT will automatically use it; no need to log it on startup.
    try:
        session = ort.InferenceSession(
            str(classifier_onnx),
            providers=["CPUExecutionProvider"],
        )
    except Exception as e:
        if "onnx.data" in str(e) or "file size" in str(e):
            print(f"Could not load classifier (missing external data file?). Expected: {data_path}")
            print(f"Error: {e}\nSkipping species classification.\n")
            return None
        raise

    mapping = json.loads(class_mapping.read_text())
    idx_to_class = {int(k): v for k, v in mapping.get("idx_to_class", {}).items()}

    # Infer input size from ONNX graph (expects [1, 3, H, W])
    input_meta = session.get_inputs()[0]
    shape = input_meta.shape
    input_size = int(shape[2]) if len(shape) >= 3 and shape[2] is not None else 320

    print(
        f"[classifier] loaded {classifier_onnx.name}, input {input_size}, classes={len(idx_to_class)}"
    )
    return session, idx_to_class, input_size


def classify_bird(crop, session, idx_to_class: Dict[int, str], input_size: int):
    """Run bird species classification on a cropped image."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (int(input_size), int(input_size)))
    img = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, 0)  # NCHW

    outputs = session.run(None, {"input": img})[0]
    logits = outputs[0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    species = idx_to_class.get(top_idx, f"class_{top_idx}")
    return species, top_prob



