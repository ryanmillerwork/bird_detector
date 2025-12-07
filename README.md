# Bird Detector

Two-stage bird detection and classification system for Raspberry Pi 5. Captures frames from a Reolink camera via RTSP, detects animals using YOLOv8s, then classifies bird species using a fine-tuned ConvNeXt-Small model.

## Setup

```bash
# Create and activate environment
uv venv birds
source birds/bin/activate

# Install dependencies
uv pip install opencv-python ultralytics timm
```

### Camera Credentials

Copy the example config and add your camera credentials:

```bash
cp .env.example .env
# Edit .env with your camera details
```

`.env` format:
```
CAMERA_USER=admin
CAMERA_PASS=your_password
CAMERA_IP=192.168.0.100
CAMERA_PORT=554
```

This file is git-ignored to keep credentials out of the repo.

## Files

### `capture_frame.py`

Simple utility to grab a single frame from the camera. Useful for testing connectivity or grabbing sample images.

```bash
# Save with auto-generated timestamp filename
python capture_frame.py

# Save with custom filename
python capture_frame.py my_photo.jpg
```

### `detector.py`

Main detection pipeline. Connects to the RTSP stream, runs YOLOv8s every 2 seconds to detect animals (birds, cats, dogs, etc.), and saves:
- Annotated frames with bounding boxes → `detections/`
- Cropped animals (with 100px padding) → `crops/`

```bash
python detector.py
```

Press `Ctrl+C` to stop. Timing breakdown is printed for each frame showing grab time and detection time.

**Configuration** (edit at top of file):
- `ANIMAL_CLASSES` — Which COCO classes to detect
- `confidence` — Detection threshold (default 0.4)
- `padding` — Pixels to add around crops (default 100)

### `train_classifier.py`

Trains a ConvNeXt-Small classifier on labeled bird images for Stage 2 species identification.

**Data format:** Images organized in folders by class name:
```
hand_sorted/
  blue_jay/
    image1.jpg
    image2.jpg
  cardinal/
    ...
```

Classes with fewer than 5 images are automatically excluded.

```bash
cd ~/bird_detector
source birds/bin/activate
taskset -c 1-3 python train_classifier.py
```

> **Note:** `taskset -c 1-3` pins training to cores 1-3, leaving core 0 free for system tasks and SSH.

**Output** (saved to `models/`):
- `best_model.pt` — Best validation accuracy checkpoint
- `final_model.pt` — Final epoch checkpoint  
- `bird_classifier.onnx` — Optimized for inference
- `class_mapping.json` — Class name ↔ index mapping

**Configuration** (edit at top of file):
- `INPUT_SIZE` — Image size (default 320)
- `BATCH_SIZE` — Reduce if running out of RAM (default 8)
- `EPOCHS` — Training epochs (default 30)
- `MIN_SAMPLES` — Minimum images per class (default 5)

Training on Pi 5 CPU takes ~5-10 minutes per epoch.

## Pipeline Overview

```
Camera (RTSP)
     │
     ▼
┌─────────────┐
│  YOLOv8s    │  ← Stage 1: Detect animals
│  (~900ms)   │
└─────────────┘
     │
     ▼ crops
┌─────────────┐
│ ConvNeXt-S  │  ← Stage 2: Classify species
│  (~500ms)   │
└─────────────┘
     │
     ▼
  Bird ID + confidence
```

## Hardware

- Raspberry Pi 5 (4GB+ RAM recommended)
- Reolink camera with RTSP support

## Performance (Pi 5)

| Stage | Model | Time |
|-------|-------|------|
| Frame grab | — | ~50ms |
| Detection | YOLOv8s | ~900ms |
| Classification | ConvNeXt-Small | ~500ms (estimated) |
| **Total** | | **~1.5s** |

Comfortably fits in a 2-second capture interval.

