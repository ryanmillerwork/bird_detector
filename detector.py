#!/usr/bin/env python3
"""
Bird/animal detector using YOLOv8s.
Stage 1 of the detection pipeline.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import cv2
import time
from datetime import datetime
from pathlib import Path
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
RTSP_URL = f"rtsp://{_env['CAMERA_USER']}:{_env['CAMERA_PASS']}@{_env['CAMERA_IP']}:{_env['CAMERA_PORT']}/h264Preview_01_main"

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

# Output directories
DETECTIONS_DIR = Path("detections")
CROPS_DIR = Path("crops")


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
        })
    
    return detections


def draw_detections(frame, detections):
    """Draw bounding boxes on frame."""
    annotated = frame.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['class']} {det['confidence']:.2f}"
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return annotated


def main():
    """Run detection pipeline with persistent stream."""
    DETECTIONS_DIR.mkdir(exist_ok=True)
    CROPS_DIR.mkdir(exist_ok=True)
    
    print("Loading YOLOv8s model...")
    model = YOLO("yolov8s.pt")
    print("Model loaded.")
    
    print("Connecting to camera...")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("Failed to open stream")
        return
    
    print("Connected.\n")
    print("Running detection every 2 seconds. Press Ctrl+C to stop.\n")
    
    frame_count = 0
    try:
        while True:
            loop_start = time.time()
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Drain buffer and get fresh frame
            # Read a few frames to clear buffer
            for _ in range(3):
                ret, frame = cap.read()
            
            if not ret:
                print(f"[{ts}] Failed to grab frame, reconnecting...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            
            grab_time = time.time() - loop_start
            
            # Run detection
            detect_start = time.time()
            detections = detect_animals(model, frame)
            detect_time = time.time() - detect_start
            
            # Save if detections found
            if detections:
                annotated = draw_detections(frame, detections)
                cv2.imwrite(str(DETECTIONS_DIR / f"detection_{ts}.jpg"), annotated)
                
                for i, det in enumerate(detections):
                    cv2.imwrite(str(CROPS_DIR / f"crop_{ts}_{i}_{det['class']}.jpg"), det['crop'])
                
                det_summary = ", ".join(f"{d['class']}({d['confidence']:.2f})" for d in detections)
                print(f"[{ts}] Detected: {det_summary} (grab:{grab_time*1000:.0f}ms, detect:{detect_time*1000:.0f}ms)")
            else:
                print(f"[{ts}] No animals (grab:{grab_time*1000:.0f}ms, detect:{detect_time*1000:.0f}ms)")
            
            frame_count += 1
            
            # Wait for 2 second interval
            elapsed = time.time() - loop_start
            time.sleep(max(0, 2.0 - elapsed))
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
    
    print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    main()
