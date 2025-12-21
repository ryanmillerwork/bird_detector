from __future__ import annotations

import cv2


# COCO classes we care about (animals)
ANIMAL_CLASSES: dict[int, str] = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}


def detect_animals(model, frame, *, confidence: float = 0.25, padding: int = 100):
    """Detect animals in frame with padded crops."""
    # Explicitly run on CPU (no auto device selection).
    results = model(frame, conf=float(confidence), verbose=False, device="cpu")[0]
    h, w = frame.shape[:2]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in ANIMAL_CLASSES:
            continue

        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Add padding, clamped to image bounds
        crop_x1 = max(0, x1 - int(padding))
        crop_y1 = max(0, y1 - int(padding))
        crop_x2 = min(w, x2 + int(padding))
        crop_y2 = min(h, y2 + int(padding))
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

        detections.append(
            {
                "class": ANIMAL_CLASSES[cls_id],
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "crop": crop,
                "species": None,
                "species_confidence": None,
            }
        )

    return detections


def draw_detections(frame, detections):
    """Draw bounding boxes on frame."""
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class']} {det['confidence']:.2f}"
        if det.get("species"):
            label += f" -> {det['species']} ({det['species_confidence']:.2f})"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    return annotated



