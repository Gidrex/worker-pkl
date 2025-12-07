# Parking Vehicle Detection System

Production-ready vehicle detection and parking analysis system using YOLO models with BotSORT tracking.

## Documentation

- [User Guide / Virtual RTSP Server](docs/usage.md)
- [Configuration](md/config.md)

## Features

- **Vehicle Detection**: YOLOv11/v12 detection (car, motorcycle, bus, truck)
- **Advanced Tracking**: BotSORT/ByteTrack for stable vehicle identification
- **Parking Analysis**: Dual-threshold detection using IoU + centroid
- **Database Storage**: SQLite for vehicle metadata and position history
- **JSON-RPC API**: Query detection results via REST API
- **Detailed Logging**: Multi-level logging with loguru
- **Frame Export**: Save annotated frames with bounding boxes

## Process Video

```bash
just run ./videos/video.avi
```

### Start API Server (Optional)

```bash
uvicorn api.server:create_app --host 0.0.0.0 --port 8080
```

### Processing Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                      main.py (CLI)                      │
└──────────────┬──────────────────────────────────────────┘
               │
               v
┌─────────────────────────────────────────────────────────┐
│               FrameProcessor                            │
│  ┌──────────────────┐  ┌────────────────┐               │
│  │ Detector+Tracker │→ │ ParkingAnalyzer│→ Database     │
│  │   (BotSORT)      │  │  (IoU+Centroid)│               │
│  └──────────────────┘  └────────────────┘               │
└─────────────────────────────────────────────────────────┘
               │
               v
┌─────────────────────────────────────────────────────────┐
│              SQLite Database                            │
│    ProcessedVideo → Vehicle → VehiclePosition           │
└─────────────────────────────────────────────────────────┘
               │
               v
┌─────────────────────────────────────────────────────────┐
│            JSON-RPC API (FastAPI)                       │
│   get_vehicles | get_vehicle_history | get_videos       │
└─────────────────────────────────────────────────────────┘
```

## Logging

Set log level in `config.json`:

```json
{
  "logging": {
    "level": "DEBUG"  // DEBUG, INFO, WARNING, ERROR, CRITICAL
  }
}
```

Example output:

```
2025-12-07 12:28:25 | INFO     | core.parking_analyzer:152 - Vehicle 260 status changed: moving → parked
2025-12-07 12:28:25 | INFO     | core.frame_processor:210 - Vehicle 260: moving → parked at frame 39720
2025-12-07 12:28:25 | DEBUG    | core.frame_processor:248 - Frame 39720: saved with boxes in 0.016s
2025-12-07 12:28:25 | DEBUG    | core.frame_processor:177 - Frame 39840: detected+tracked 33 vehicles in 0.064s
2025-12-07 12:28:25 | DEBUG    | core.frame_processor:248 - Frame 39840: saved with boxes in 0.015s
2025-12-07 12:28:25 | DEBUG    | core.frame_processor:177 - Frame 39960: detected+tracked 33 vehicles in 0.062s
```

## Development

```bash
# Run QA (format + lint) before every commit
just qa
```

## Performance

Tested on NVIDIA RTX 3060 (laptop):
- Model: pretrained YOLOv12x
- Resolution: 1920x1080
- Processing: ~15 FPS (frame_interval=2)
- Tracking: BotSORT
