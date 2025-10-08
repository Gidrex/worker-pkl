# Worker PKL - Documentation

## How the System Works

### 1. **Vehicle Detection** (`models/yolov9.py`)

```python
from models import YOLOv9Detector

# Detection settings in JSON
options = {
    "conf_threshold": 0.3,      # confidence threshold
    "iou_threshold": 0.45,       # IoU threshold for NMS
    "imgsz": 640,                # input image size
    "device": 0,                 # GPU 0
    "vid_stride": 1              # process every N-th frame
}

detector = YOLOv9Detector(
    model_path="./weights/yolov9c.pt",  # model weights
    options=options
)

# Local video
results = detector.videoLocal(
    video_path="./videos/parking.mp4",
    output_path="./results/annotated.mp4"
)

# Video from S3/Beget (public URL)
results = detector.videoSource(
    video_url="https://your-bucket.beget.com/video.mp4",
    output_path="./results/remote_annotated.mp4"
)
```

**Process:**
1. Loads YOLOv9 model (yolov9c)
2. For S3 - downloads video to temporary file
3. Processes each frame, detects vehicles (car, bus, truck, motorcycle)
4. Saves results to JSON + annotated video

---

### 2. **Training** (`models/training.py`)

```python
from models import YOLOv9Trainer, create_dataset_yaml

# Create dataset.yaml
create_dataset_yaml(
    dataset_path="/path/to/dataset",
    train_path="images/train",
    val_path="images/val",
    classes=["car", "bus", "truck"],
    output_path="dataset.yaml"
)

# Training
trainer = YOLOv9Trainer(
    model_name="yolov9c.pt",
    device=0,  # GPU 0
    output_dir="./weights"
)

results = trainer.train(
    data_yaml="dataset.yaml",
    epochs=100,
    batch_size=16,
    imgsz=640
)
```

**Process:**
1. Loads base model yolov9c.pt
2. Trains on dataset (Kaggle vehicles or CNRPark)
3. Uses 1 GPU (device=0)
4. Saves:
   - `./weights/{name}/weights/best.pt` - best weights
   - `./weights/{name}/weights/last.pt` - last weights
   - `./weights/{name}/training_summary.json` - training logs

**Available Datasets:**
- `vehicles_kaggle` - vehicles from Kaggle
- `cnrpark` - parking spaces (free/occupied)
- `pklot` - alternative parking dataset

---

### 3. **Parking Space Detection** (`models/parking_detection.py`)

```python
from models import ParkingDetector

detector = ParkingDetector(
    model_path="./weights/parking_yolov9c.pt",  # trained model
    options={"conf_threshold": 0.3}
)

# Define parking zones manually (by clicking)
detector.define_parking_spaces_manual(
    image_path="parking_lot.jpg",
    output_path="parking_config.json"
)

# Or load existing configuration
detector.load_parking_spaces("parking_config.json")

# Process video
results = detector.detect_from_video(
    video_path="parking_video.mp4",
    output_path="parking_annotated.mp4"
)

# Result for each frame:
# {
#   "frame": 0,
#   "occupied_count": 12,
#   "free_count": 8,
#   "spaces": [
#     {"id": 1, "status": "occupied", "confidence": 0.95},
#     {"id": 2, "status": "free", "confidence": 0.87}
#   ]
# }
```

**Process:**
1. Loads model trained on CNRPark (free/occupied)
2. For each frame:
   - Checks each parking zone
   - Classifies: occupied or free
3. Outputs statistics: how many occupied/free

---

## Usage Workflow

### Step 1: Train Vehicle Detection Model
```bash
python -c "
from models import YOLOv9Trainer
trainer = YOLOv9Trainer(model_name='yolov9c.pt', device=0)
trainer.train(data_yaml='vehicles_dataset.yaml', epochs=100)
"
```

### Step 2: Train Parking Detection Model
```bash
python -c "
from models import YOLOv9Trainer
trainer = YOLOv9Trainer(model_name='yolov9c.pt', device=0)
trainer.train(data_yaml='cnrpark_dataset.yaml', epochs=100)
"
```

### Step 3: Use Trained Models
```python
# Vehicle detection
detector = YOLOv9Detector(model_path="./weights/train_vehicles/weights/best.pt")
detector.videoLocal("video.mp4")

# Parking detection
parking = ParkingDetector(model_path="./weights/train_parking/weights/best.pt")
parking.load_parking_spaces("config.json")
parking.detect_from_video("video.mp4")
```

---

## Key Features

- **videoSource** - works with S3 Beget via public URLs
- **videoLocal** - processes local mp4 files
- **JSON options** - all detection parameters configured via dictionary
- **1 GPU** - training.py uses device=0
- **Results** - JSON with detections + annotated video

---

## Output Structure

### Vehicle Detection Output
```json
{
  "video_source": "./videos/parking.mp4",
  "total_frames": 300,
  "detection_options": {...},
  "frames": [
    {
      "frame": 0,
      "vehicle_count": 5,
      "detections": [
        {
          "class_id": 2,
          "class_name": "car",
          "confidence": 0.87,
          "bbox": {
            "x1": 100, "y1": 200,
            "x2": 300, "y2": 400,
            "width": 200, "height": 200
          }
        }
      ]
    }
  ]
}
```

### Parking Detection Output
```json
{
  "video_source": "./videos/parking.mp4",
  "total_frames": 300,
  "total_parking_spaces": 20,
  "frames": [
    {
      "frame": 0,
      "occupied_count": 12,
      "free_count": 8,
      "total_spaces": 20,
      "spaces": [
        {
          "id": 1,
          "status": "occupied",
          "confidence": 0.95,
          "points": [[100, 100], [200, 100], [200, 200], [100, 200]]
        }
      ]
    }
  ]
}
```

---

## Dataset Links

### Vehicle Detection
- **Kaggle Car Detection**: https://www.kaggle.com/datasets/sshikamaru/car-object-detection
- **Vehicle Detection Dataset**: https://www.kaggle.com/datasets/marquis03/vehicle-detection-dataset
- **COCO Pretrained**: Included in ultralytics

### Parking Detection
- **CNRPark Dataset**: http://cnrpark.it/
  - Download: http://cnrpark.it/dataset/CNRPark-Patches-150x150.zip
- **CNRPark+EXT**: http://cnrpark.it/dataset/CNR-EXT-Patches-150x150.zip
- **PKLot Dataset**: https://www.kaggle.com/datasets/blanderbuss/parking-lot-dataset

---

## Installation & Setup

```bash
# Install dependencies
just prepare

# Run application
just run

# Code quality checks
just qa
```

---

## Requirements

- Python >= 3.13
- CUDA-capable GPU (for training)
- uv (dependency manager)
- just (command runner)

### Python Dependencies
- ultralytics >= 8.0.0 (YOLOv9)
- opencv-python >= 4.8.0 (video processing)
- numpy >= 1.24.0 (array operations)
- requests >= 2.31.0 (S3/URL downloads)
- fastapi >= 0.104.0 (API server)
- uvicorn >= 0.24.0 (ASGI server)
