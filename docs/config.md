# Configuration Reference

Complete reference for `config.json` settings.

## Structure

```json
{
  "logging": {...},
  "model": {...},
  "processing": {...},
  "tracking": {...},
  "parking": {...},
  "storage": {...},
  "api": {...}
}
```

## Logging Configuration

Controls logging behavior and output format, powered by **[Loguru](https://loguru.readthedocs.io/)**.

```json
"logging": {
  "level": "DEBUG",
  "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
  "rotation": "100 MB"
}
```

### Parameters

- **level** (string): Minimum log level to display. Corresponds to `loguru` [severity levels](https://loguru.readthedocs.io/en/stable/api/logger.html#levels).
  - Values: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
  - Default: `INFO`
  - `DEBUG`: Shows frame-by-frame timing, all detections
  - `INFO`: Processing start/end, status changes

- **format** (string): Log message format using **[Loguru syntax](https://loguru.readthedocs.io/en/stable/api/logger.html#record)**.
  - Default: `"{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}"`
  - Variables: `{time}`, `{level}`, `{name}`, `{line}`, `{message}`

- **rotation** (string): Log file rotation condition, handled by **[Loguru rotation](https://loguru.readthedocs.io/en/stable/api/logger.html#rotation)**.
  - Default: `"100 MB"`

## Model Configuration

Settings for the vehicle detection model, utilizing **[Ultralytics YOLO](https://docs.ultralytics.com/)**.

```json
"model": {
  "path": "yolo26s.pt",
  "device": "cuda:0",
  "conf_threshold": 0.3,
  "classes": [2, 3, 5, 7]
}
```

### Parameters

- **path** (string): Path or name of the YOLO model file. Loaded via `ultralytics.YOLO()`.
  - Default: `"yolo26s.pt"`
  - See [YOLO Models](https://docs.ultralytics.com/models/).
  - Supported: `yolo11x.pt`, `yolo12x.pt`, `yolo26s.pt`, custom trained models

- **device** (string): Compute device for inference, passed to **[PyTorch](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)**.
  - Default: `"cuda:0"`
  - Options: `"cpu"`, `"cuda:N"`, `"mps"` (Apple Silicon)

- **conf_threshold** (float): Minimum confidence score for detections. Maps to the `conf` argument in YOLO [predict mode](https://docs.ultralytics.com/modes/predict/#inference-arguments).
  - Default: `0.3`
  - Recommended: `0.25-0.4`

- **classes** (array[int]): List of **[COCO Dataset](https://cocodataset.org/#explore)** class IDs to filter. Maps to the `classes` argument in YOLO predict.
  - Default: `[2, 3, 5, 7]` (car, motorcycle, bus, truck)

## Processing Configuration

Settings for the frame processing pipeline implemented in `src.core.frame_processor`.

```json
"processing": {
  "frame_interval": 2,
  "batch_size": 8,
  "max_workers": 4
}
```

### Parameters

- **frame_interval** (int): Process every N-th frame to control FPS processing speed.
  - Default: `3`
  - Formula: `processed_fps = video_fps / frame_interval`

- **batch_size** (int): (Future Use) For batch inference.
  - Default: `8`

- **max_workers** (int): (Future Use) For `concurrent.futures` multiprocessing.
  - Default: `4`

## Tracking Configuration

Settings for Multi-Object Tracking (MOT), supported by **[Ultralytics Tracking](https://docs.ultralytics.com/modes/track/)**.

```json
"tracking": {
  "tracker": "botsort.yaml",
  "min_hits": 3,
  "max_age": 30
}
```

### Parameters

- **tracker** (string): The tracker configuration file or name.
  - Default: `"botsort.yaml"`
  - Options:
    - `"botsort.yaml"`: **[BotSORT](https://arxiv.org/abs/2206.14651)** (Robust, appearance-based).
    - `"bytetrack.yaml"`: **[ByteTrack](https://arxiv.org/abs/2110.06864)** (Fast, IoU-based).

- **min_hits** (int): Minimum number of consecutive detections required to initialize a track (handled by the tracker implementation).
  - Default: `3`

- **max_age** (int): Maximum number of frames to keep a track alive without detection (buffer for occlusion).
  - Default: `30`

## Parking Configuration

Parameters for the custom parking analysis logic (`src.core.parking_analyzer`), utilizing **[IoU](https://en.wikipedia.org/wiki/Jaccard_index)** and centroid distance.

```json
"parking": {
  "iou_threshold": 0.3,
  "centroid_threshold": 30.0,
  "stationary_frames": 5
}
```

### Parameters

- **iou_threshold** (float): Minimum **[Intersection over Union](https://en.wikipedia.org/wiki/Jaccard_index)** overlap between current and previous box to consider it "stationary".
  - Default: `0.3`
  - Range: `0.0` to `1.0`

- **centroid_threshold** (float): Maximum Euclidean distance (in pixels) the box center is allowed to move to be considered "stationary".
  - Default: `30.0`
  - Scale with resolution: `threshold_pixels ≈ width / 64`

- **stationary_frames** (int): Number of consecutive frames satisfying the stationary condition required to change status to `parked`.
  - Default: `5`

## Storage Configuration

Settings for persistence using **[SQLite](https://www.sqlite.org/)** and file storage.

```json
"storage": {
  "database_path": "./data/parking.db",
  "save_frames": true,
  "frames_dir": "./data/frames"
}
```

### Parameters

- **database_path** (string): Filesystem path for the SQLite database.
  - Default: `"./data/parking.db"`

- **save_frames** (boolean): Toggle saving of debug frames with drawn bounding boxes using **[OpenCV](https://opencv.org/)**.
  - Default: `true`

- **frames_dir** (string): Directory path for saving processed frames.
  - Default: `"./data/frames"`

## API Configuration

Settings for the **[FastAPI](https://fastapi.tiangolo.com/)** server.

```json
"api": {
  "enabled": true,
  "host": "0.0.0.0",
  "port": 8844
}
```

### Parameters

- **enabled** (boolean): Feature flag for the API.
  - Default: `true`

- **host** (string): Bind host for **[Uvicorn](https://www.uvicorn.org/)**.
  - Default: `"0.0.0.0"`

- **port** (int): Bind port for **[Uvicorn](https://www.uvicorn.org/)**.
  - Default: `8844`

## Example Configurations

### High Accuracy (Slow)

```json
{
  "model": {"conf_threshold": 0.4},
  "processing": {"frame_interval": 1},
  "tracking": {"min_hits": 5},
  "parking": {
    "iou_threshold": 0.7,
    "centroid_threshold": 15.0,
    "stationary_frames": 15
  }
}
```

### Balanced (Recommended)

```json
{
  "model": {"conf_threshold": 0.3},
  "processing": {"frame_interval": 12},
  "tracking": {"min_hits": 3},
  "parking": {
    "iou_threshold": 0.3,
    "centroid_threshold": 30.0,
    "stationary_frames": 5
  }
}
```

### Fast Processing (Lower Accuracy)

```json
{
  "model": {"conf_threshold": 0.25},
  "processing": {"frame_interval": 30},
  "tracking": {"min_hits": 2},
  "parking": {
    "iou_threshold": 0.2,
    "centroid_threshold": 50.0,
    "stationary_frames": 3
  }
}
```