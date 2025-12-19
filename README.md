# Bird Detector

Two-stage bird detection and classification system for Raspberry Pi 5. Captures frames from a Reolink camera via RTSP, detects animals using YOLOv8s, then classifies bird species using a fine-tuned ConvNeXt-Small model.

## Setup

```bash
# Create and activate environment
uv venv birds
# Install dependencies
uv pip install opencv-python ultralytics timm onnxscript onnxruntime
```

### Camera Credentials

Copy the example config and add your camera credentials:

```bash
cp env.example .env
# Edit .env with your camera details
```

`.env` format:
```
# RTSP (recommended: go2rtc)
RTSP_URL=rtsp://192.168.0.50:8554/bird_cam

# Alternative: direct camera RTSP path + credentials
# CAMERA_USER=admin
# CAMERA_PASS=your_password
# CAMERA_IP=192.168.0.100
# CAMERA_PORT=554
# RTSP_PATH=h264Preview_01_main

# RTSP / capture tuning
RTSP_TRANSPORT=tcp                # passed to ffmpeg/opencv; tcp recommended
RTSP_TIMEOUT_US=5000000           # optional; ffmpeg stimeout
RTSP_MAX_DELAY_US=0               # optional; ffmpeg max_delay (0 disables buffering)
RTSP_BUFFER_SIZE=0                # optional; ffmpeg buffer_size
SNAPSHOT_MODE=true                # true uses ffmpeg per-frame capture; false uses persistent OpenCV RTSP

# Training config (optional)
DATA_DIR=hand_sorted
OUTPUT_DIR=models
BATCH_SIZE=16
NUM_WORKERS=4
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

Main detection pipeline. Default mode (snapshot + producer/consumer threading):
- Producer grabs frames via ffmpeg/RTSP (TCP by default), as fast as possible, dropping old frames to keep only the freshest.
- Consumer runs YOLOv8s on the latest frame, classifies crops with the ONNX classifier, and saves:
  - Annotated frames with bounding boxes → `detections/` (keeps last 10)
  - Cropped animals (with 100px padding) → `crops/<classification>/`

```bash
python detector.py
```

Press `Ctrl+C` to stop. Timing breakdown is printed for each frame showing grab, detection, and classification times.

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
python train_classifier.py
```

> **Note:** `taskset -c 1-3` pins training to cores 1-3, leaving core 0 free for system tasks and SSH.

**Output** (saved to `models/`):
- `best_model.pt` — Best validation accuracy checkpoint
- `final_model.pt` — Final epoch checkpoint  
- `bird_classifier.onnx` — Optimized for inference (updated on each best model)
- `class_mapping.json` — Class name ↔ index mapping

**Resume training:** If `best_model.pt` exists, training automatically resumes from that checkpoint.

**Configuration** (in `.env` file):
- `INPUT_SIZE` — Image size (default 320)
- `EPOCHS` — Training epochs (default 30)
- `MIN_SAMPLES` — Minimum images per class (default 5)

## Deployment to Pi 5

After training, copy these files from the training machine to the Pi:

```bash
# From training machine
scp models/bird_classifier.onnx pi@datalogger.local:~/bird_detector/models/
scp models/bird_classifier.onnx.data pi@datalogger.local:~/bird_detector/models/
scp models/class_mapping.json pi@datalogger.local:~/bird_detector/models/
```

**Required files on Pi:**
| File | Location | Purpose |
|------|----------|---------|
| `bird_classifier.onnx` | `models/` | Trained classifier (ONNX for fast inference) |
| `bird_classifier.onnx.data` (if present) | `models/` | External weight shards when ONNX export uses external data |
| `class_mapping.json` | `models/` | Maps model output indices to bird names |

The ONNX file is exported automatically whenever a new best model is saved during training, so you'll always have a usable model even if training is interrupted. If your export produced a `bird_classifier.onnx.data` file, copy it alongside `bird_classifier.onnx` or re-export with embedded weights to avoid the extra file.

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

