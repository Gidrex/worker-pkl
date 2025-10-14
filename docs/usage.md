# Vehicle Tracking System - Usage Guide

Complete system for training and detecting parked/moving vehicles.

## Architecture Overview

```
Training Pipeline:
  Local Videos → Extract Frames → Auto-Label → Train YOLOv9 → Save to ./weights

Detection Pipeline:
  Video → YOLOv9 Detection → Vehicle Tracker → State (Parked/Moving) → GUI + JSON
```

## Installation

```bash
just sync
```

## Workflow

### Step 1: Prepare Dataset from Local Videos

Download videos:

```bash
./scripts/download_videos.sh
```

Or manually place videos in `./videos/` directory.

Run dataset preparation:

```bash
uv run scripts/prepare_dataset.py
```

**Output:**
- `./datasets/parking_vehicles/images/train/` - Training images
- `./datasets/parking_vehicles/images/val/` - Validation images
- `./datasets/parking_vehicles/labels/train/` - Training labels (YOLO format)
- `./datasets/parking_vehicles/labels/val/` - Validation labels
- `./datasets/parking_vehicles/dataset.yaml` - Dataset configuration

**What it does:**
1. Scans `./videos/` for all video files (.mp4, .avi, .mov, .mkv, .flv)
2. Extracts every 30th frame from videos
3. Auto-labels vehicles using pretrained COCO YOLOv9 model
4. Splits 80% train / 20% validation
5. Creates YOLO format dataset

### Step 2: Train Model

```bash
uv run scripts/train_model.py
```

**Training parameters:**
- Model: YOLOv9c (medium size)
- Epochs: 100
- Batch size: 8 (optimized for RTX 3060 Laptop)
- Device: GPU 0

**Output:**
- `./weights/parking_vehicles/weights/best.pt` - Best model weights
- `./weights/parking_vehicles/weights/last.pt` - Last epoch weights
- `./weights/parking_vehicles/training_summary.json` - Training metrics

### Step 3: Test Vehicle Tracking

```bash
uv run scripts/test_tracking.py
```

**Output:**
- `./output/tracked_video.mp4` - Annotated video with colored boxes
- `./results/test_tracked.json` - Frame-by-frame detection results

**Visualization:**
- Green boxes = PARKED vehicles
- Red boxes = MOVING vehicles
- Yellow boxes = UNKNOWN (not enough frames)

## Using the Python API

### Basic Detection

```python
from models import YOLOv9Detector

detector = YOLOv9Detector(
    model_path="./weights/parking_vehicles/weights/best.pt",
    options={"conf_threshold": 0.3, "device": 0}
)

results = detector.videoLocalTracked(
    video_path="./videos/test.mp4",
    output_path="./output/annotated.mp4",
    frame_skip=3,
    movement_threshold=15.0
)
```

### Real-time Visualization

```python
results = detector.videoLocalTracked(
    video_path="./videos/test.mp4",
    show_realtime=True  # Opens OpenCV window
)
```

**Controls:**
- Press `p` to pause/resume
- Press `q` to quit

## REST API

### Start API Server

```bash
just run
```

Server runs at: `http://localhost:8000`

API docs: `http://localhost:8000/docs`

## Configuration

### Model Selection

**Pretrained COCO (quick test):**
```python
model_path = "yolov9c.pt"  # Auto-downloads
```

**Custom trained (better accuracy):**
```python
model_path = "./weights/parking_vehicles/weights/best.pt"
```

### Frame Skip

```python
frame_skip = 3   # 66% faster, good for parking lots
frame_skip = 1   # Process all frames, slower
```

### Movement Threshold

```python
movement_threshold = 15.0  # Balanced (default)
```

## Troubleshooting

### GPU Out of Memory

Reduce batch size in `scripts/train_model.py:32`:
```python
batch_size=4  # Default: 8 for RTX 3060 Laptop
```

### No Vehicles Detected

Lower confidence threshold:
```python
options = {"conf_threshold": 0.2}  # Default: 0.3
```

## Supported Video Formats

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.flv`
