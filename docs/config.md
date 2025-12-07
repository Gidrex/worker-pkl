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

Controls logging behavior and output format.

```json
"logging": {
  "level": "DEBUG",
  "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
  "rotation": "100 MB"
}
```

### Parameters

- **level** (string): Minimum log level to display
  - Values: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
  - Default: `INFO`
  - `DEBUG`: Shows frame-by-frame timing, all detections
  - `INFO`: Processing start/end, status changes
  - `WARNING`: Low confidence detections, tracking issues
  - `ERROR`: Critical failures

- **format** (string): Log message format using loguru syntax
  - Default: `"{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}"`
  - Variables: `{time}`, `{level}`, `{name}`, `{line}`, `{message}`

- **rotation** (string): Log file rotation size
  - Default: `"100 MB"`
  - Examples: `"10 MB"`, `"500 KB"`, `"1 GB"`

## Model Configuration

YOLO model settings for vehicle detection.

```json
"model": {
  "path": "yolo12x.pt",
  "device": "cuda:0",
  "conf_threshold": 0.3,
  "classes": [2, 3, 5, 7]
}
```

### Parameters

- **path** (string): Path or name of YOLO model
  - Default: `"yolo12x.pt"`
  - If file doesn't exist locally, ultralytics auto-downloads it
  - Supported: `yolo11x.pt`, `yolo12x.pt`, custom trained models
  - Larger models (x) = slower but more accurate
  - Smaller models (n, s, m) = faster but less accurate

- **device** (string): Compute device for inference
  - Default: `"cuda:0"`
  - `"cuda:0"`: First GPU
  - `"cuda:1"`: Second GPU
  - `"cpu"`: CPU (much slower)
  - `"mps"`: Apple M1/M2 GPU (macOS)

- **conf_threshold** (float): Minimum confidence for detections
  - Default: `0.3`
  - Range: `0.0` to `1.0`
  - Lower = more detections but more false positives
  - Higher = fewer detections but more accurate
  - Recommended: `0.25-0.4` for vehicle detection

- **classes** (array[int]): COCO class IDs to detect
  - Default: `[2, 3, 5, 7]` (car, motorcycle, bus, truck)
  - COCO IDs:
    - `2`: car
    - `3`: motorcycle
    - `5`: bus
    - `7`: truck
  - Only specified classes will be detected

## Processing Configuration

Frame processing and parallelization settings.

```json
"processing": {
  "frame_interval": 2,
  "batch_size": 8,
  "max_workers": 4
}
```

### Parameters

- **frame_interval** (int): Process every N-th frame
  - Default: `3`
  - Range: `1` to `∞`
  - Formula: `frames_per_second = video_fps / frame_interval`
  - Examples (for 60 FPS video):
    - `1` = 60 frames/sec (no skip, very slow)
    - `2` = 30 frames/sec
    - `3` = 20 frames/sec
    - `5` = 12 frames/sec
    - `12` = 5 frames/sec (recommended for 60 FPS)
    - `30` = 2 frames/sec
  - Higher = faster processing but may miss fast-moving vehicles

- **batch_size** (int): Number of frames to process in parallel (future use)
  - Default: `8`
  - Currently not implemented (reserved for future batch processing)

- **max_workers** (int): Maximum parallel workers (future use)
  - Default: `4`
  - Currently not implemented (reserved for future multiprocessing)

## Tracking Configuration

Vehicle tracking algorithm settings.

```json
"tracking": {
  "tracker": "botsort.yaml",
  "min_hits": 3,
  "max_age": 30
}
```

### Parameters

- **tracker** (string): Tracking algorithm to use
  - Default: `"botsort.yaml"`
  - Options:
    - `"bytetrack.yaml"`: Fast, simple IoU-based tracking
    - `"botsort.yaml"`: Advanced, uses appearance + motion (recommended)
  - BotSORT is more robust for occluded/crowded scenes
  - ByteTrack is faster but less accurate with ID switches

- **min_hits** (int): Minimum detections before assigning track ID
  - Default: `3`
  - Range: `1` to `∞`
  - Higher = reduces false tracks but may miss short appearances
  - Lower = more tracks but more false positives
  - Recommended: `3-5`

- **max_age** (int): Frames to keep track alive without detection
  - Default: `30`
  - Range: `1` to `∞`
  - Higher = maintains tracks longer during occlusion
  - Lower = removes disappeared vehicles faster
  - Formula: `seconds = max_age * frame_interval / video_fps`
  - Example: `30 * 12 / 60 = 6 seconds` for 60 FPS video with interval=12

## Parking Configuration

Parking detection algorithm parameters.

```json
"parking": {
  "iou_threshold": 0.3,
  "centroid_threshold": 30.0,
  "stationary_frames": 5
}
```

### Parameters

- **iou_threshold** (float): Minimum IoU overlap between consecutive frames
  - Default: `0.3`
  - Range: `0.0` to `1.0`
  - Formula: `IoU = intersection_area / union_area`
  - Higher (0.7-0.9) = very strict, vehicle must be perfectly still
  - Lower (0.2-0.4) = lenient, allows small movements
  - Recommended: `0.3` (tolerates detection variance)
  - Too high = parked vehicles marked as moving due to bbox jitter
  - Too low = moving vehicles marked as parked

- **centroid_threshold** (float): Maximum centroid movement in pixels
  - Default: `30.0`
  - Range: `0.0` to `∞`
  - Centroid = center point of bounding box `((x1+x2)/2, (y1+y2)/2)`
  - Higher = allows more movement before marking as "moving"
  - Lower = stricter parking detection
  - Recommended: `20-40` pixels for 1920x1080 video
  - Scale with resolution: `threshold_pixels ≈ width / 64`

- **stationary_frames** (int): Consecutive frames vehicle must be stationary
  - Default: `5`
  - Range: `1` to `∞`
  - Formula: `time_seconds = stationary_frames * frame_interval / video_fps`
  - Examples (60 FPS, interval=12):
    - `5` frames = 1 second
    - `10` frames = 2 seconds
    - `25` frames = 5 seconds
  - Lower = faster parking detection but more false positives
  - Higher = more accurate but slower to detect parked vehicles
  - Recommended: `5-10` frames

### Parking Detection Algorithm

Vehicle marked as **parked** when BOTH conditions met for N consecutive frames:

1. **IoU Check**: `compute_iou(current_bbox, previous_bbox) >= iou_threshold`
2. **Centroid Check**: `distance(current_centroid, previous_centroid) <= centroid_threshold`

If either condition fails, counter resets to 0.

## Storage Configuration

Database and frame saving settings.

```json
"storage": {
  "database_path": "./data/parking.db",
  "save_frames": true,
  "frames_dir": "./data/frames"
}
```

### Parameters

- **database_path** (string): Path to SQLite database file
  - Default: `"./data/parking.db"`
  - File created automatically if doesn't exist
  - Stores: videos, vehicles, positions

- **save_frames** (boolean): Save annotated frames to disk
  - Default: `true`
  - `true`: Saves frames with bounding boxes
  - `false`: No frames saved (faster, less disk usage)
  - Frames saved as: `{video_name}_frame_{frame_idx:06d}.jpg`

- **frames_dir** (string): Directory for saved frames
  - Default: `"./data/frames"`
  - Created automatically if doesn't exist
  - Warning: Can use significant disk space
  - Estimate: `~200 KB/frame × processed_frames`

## API Configuration

JSON-RPC API server settings.

```json
"api": {
  "enabled": true,
  "host": "0.0.0.0",
  "port": 8080
}
```

### Parameters

- **enabled** (boolean): Enable JSON-RPC API
  - Default: `true`
  - Currently informational (server must be started manually)

- **host** (string): Bind address for API server
  - Default: `"0.0.0.0"` (all interfaces)
  - `"0.0.0.0"`: Accessible from network
  - `"127.0.0.1"`: Localhost only

- **port** (int): TCP port for API server
  - Default: `8080`
  - Range: `1` to `65535`
  - Avoid: `80` (HTTP), `443` (HTTPS), `22` (SSH)

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

## Performance Tuning

### Increase Speed
- ↑ `frame_interval` (process fewer frames)
- ↓ `conf_threshold` (detect less strictly)
- Use `bytetrack.yaml` instead of `botsort.yaml`
- Use smaller YOLO model (yolo11n.pt instead of yolo12x.pt)
- Set `save_frames: false`

### Increase Accuracy
- ↓ `frame_interval` (process more frames)
- ↑ `conf_threshold` (detect more strictly)
- Use `botsort.yaml` for better tracking
- ↑ `stationary_frames` (stricter parking detection)
- ↑ `min_hits` (reduce false tracks)

### Reduce False Parking Detections
- ↑ `iou_threshold` (0.5-0.7)
- ↓ `centroid_threshold` (15-25)
- ↑ `stationary_frames` (10-20)

### Detect Parking Faster
- ↓ `stationary_frames` (3-5)
- ↓ `iou_threshold` (0.2-0.4)
- ↑ `centroid_threshold` (30-50)
