# Bird Detector

Two-stage bird detection and classification system for Raspberry Pi 5. Captures frames via **Frigate JPEG snapshots**, detects animals using YOLOv8s, then classifies bird species using a fine-tuned ConvNeXt-Small model.

At runtime, `birdwatch.py`:
- Saves **annotated full frames** (YOLO boxes) to `detections/`
- Saves **cropped detections** to `crops/<label>/`
- Optionally **speaks the detected species name** (Piper) and can **play a matching bird song** from `bird_songs/`
- Maintains `detections/latest.jpg` for dashboards (latest annotated frame)

## Setup

```bash
# Create and activate environment
uv venv birds
# Install dependencies
uv pip install opencv-python ultralytics timm onnxscript onnxruntime paho-mqtt
```

### Camera Credentials

Copy the example config and add your camera credentials:

```bash
cp env.example .env
# Edit .env with your camera details
```

`.env` format (used by **both** `birdwatch.py` and `train_classifier.py`):
```
# Runtime (birdwatch.py)
FRIGATE_HOST=192.168.0.50:5000
FRIGATE_CAMERA=bird
# JPEG_URL=http://192.168.0.50:5000/api/bird/latest.jpg   # optional override
CAPTURE_INTERVAL_S=2.0

# Training (train_classifier.py)
DATA_DIR=hand_sorted
OUTPUT_DIR=models
BATCH_SIZE=16
NUM_WORKERS=4
```

`birdwatch.py` loads `.env` at startup. **Real environment variables take precedence** over `.env` values.
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

### `birdwatch.py`

Main detection pipeline (producer/consumer threading):
- Producer grabs frames from **Frigate JPEG snapshots** (avoids many RTSP/H264 corruption issues).
- Capture is **wall-clock aligned** to a fixed cadence (default `CAPTURE_INTERVAL_S=2.0`).
- Consumer runs YOLOv8s on the latest frame, classifies crops with the ONNX classifier (if present), and saves:
  - Annotated frames with bounding boxes → `detections/` (keeps last 10)
  - Latest annotated frame → `detections/latest.jpg` (overwritten each detection frame)
  - Cropped animals (with 100px padding) → `crops/<classification>/`

```bash
python birdwatch.py
```

Press `Ctrl+C` to stop. Timing breakdown is printed for each frame showing grab, detection, and classification times.

**Runtime configuration** (environment variables)

- **Capture source**
  - **`FRIGATE_HOST`**: Frigate host:port (default: `192.168.0.50:5000`)
  - **`FRIGATE_CAMERA`**: Frigate camera name (default: `bird`)
  - **`JPEG_URL`**: override full URL for latest JPEG (default: `http://$FRIGATE_HOST/api/$FRIGATE_CAMERA/latest.jpg`)
  - **`JPEG_TIMEOUT_S`**: HTTP timeout for JPEG fetches (default: `3.0`)
  - **`CAPTURE_INTERVAL_S`**: seconds between grabs (default: `2.0`)

- **Detection**
  - **`DETECT_CONF`**: YOLO confidence threshold (default: `0.25`)
  - **`DETECT_PADDING`**: pixels of padding around crops (default: `100`)

- **Text-to-speech + bird songs (optional)**
  - **`TTS_ENABLED`**: `0/1` (default: `1`)
  - **`TTS_PIPER_MODEL`**: path to Piper `.onnx` model (expects a matching `.onnx.json` beside it)
  - **`TTS_MIN_CONF`**: minimum species confidence to speak/play song (default: `0.0`)
  - **`TTS_COOLDOWN_S`**: minimum seconds between repeating the same phrase (default: `15`)
  - **`TTS_PREROLL_MS`**: leading silence before speech (default: `650`)
  - **`BIRD_SONGS_ENABLED`**: `0/1` (default: `1` when TTS is enabled)
  - **`BIRD_SONGS_DIR`**: directory of audio files named like `<class>.(mp3|wav)` (default: `./bird_songs`)
  - **`BIRD_SONGS_MAX_S`**: max seconds of bird song to play (default: `10`)

- **MQTT (optional; Home Assistant integration)**
  - **`MQTT_ENABLED`**: `0/1` (default: `1`; set to `0` to disable)
  - **`MQTT_HOST`**: broker host (example: `192.168.0.84`)
  - **`MQTT_PORT`**: broker port (default: `1883`)
  - **`MQTT_USER`** / **`MQTT_PASS`**: broker credentials (optional)
  - **`MQTT_TOPIC_EVENT`**: per-detection-frame events (default: `bird_detector/event`)
  - **`MQTT_TOPIC_STATE`**: retained latest payload (default: `bird_detector/state`)
  - **`MQTT_QOS`**: publish QoS (default: `0`)
  - **`DETECTIONS_BASE_URL`**: if set, MQTT payload includes `annotated_image_url` pointing at `${DETECTIONS_BASE_URL}/latest.jpg`

Example:

```bash
# Frigate JPEG (default):
export FRIGATE_HOST="192.168.0.50:5000"
export FRIGATE_CAMERA="bird"
export CAPTURE_INTERVAL_S="2.0"

# Optional tuning
export DETECT_CONF="0.25"
export DETECT_PADDING="100"

# Optional audio
export TTS_ENABLED="1"  # set to 0 to disable
python birdwatch.py
```

**Home Assistant quick test**:
- Developer Tools → MQTT → Listen to topic: `bird_detector/#`
- Start `birdwatch.py` and you should see JSON messages on `bird_detector/event` (and a retained `bird_detector/state`).

### Serving `latest.jpg` over HTTP (for Home Assistant dashboards)

`birdwatch.py` starts the server automatically (disable with `DETECTIONS_HTTP_ENABLED=0`).

Standalone run:

```bash
python serve_detections.py
```

Then `latest.jpg` is available at:
- `http://<pi-ip>:8765/latest.jpg`

Recommended `.env`:
```
DETECTIONS_HTTP_HOST=0.0.0.0
DETECTIONS_HTTP_PORT=8765
DETECTIONS_BASE_URL=http://<pi-ip>:8765
```

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

> **Tip (recommended over SSH):** Run training inside `tmux` so it keeps running if your SSH session drops.
>
> ```bash
> tmux new -s train
> # run your training command...
> # Detach: Ctrl-b then d
> # Re-attach later: tmux attach -t train
> ```

> **No-activate option (recommended):** If you keep a training venv+script under `~/pyenvs/bird_detector/`, you can run training without activating anything:
>
> ```bash
> cd ~/pyenvs/bird_detector
> ./birds/bin/python3 train_classifier.py
> ```

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
| Classification | ConvNeXt-Small | ~1000ms (estimated) |
| **Total** | | **~1.9s** |
