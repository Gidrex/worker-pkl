# Parking Vehicle Detection System

Production-ready vehicle detection and parking analysis system using YOLO models with BotSORT tracking.

## Features

- **Vehicle Detection**: YOLOv11/v12 detection (car, motorcycle, bus, truck)
- **Advanced Tracking**: BotSORT/ByteTrack for stable vehicle identification
- **Parking Analysis**: Dual-threshold detection using IoU + centroid
- **Database Storage**: SQLite for vehicle metadata and position history
- **JSON-RPC API**: Query detection results via REST API
- **Detailed Logging**: Multi-level logging with loguru
- **Frame Export**: Save annotated frames with bounding boxes

## Quick Start

### 1. Install Dependencies

```bash
just sync
```

### 2. Initialize Configuration

```bash
uv run main.py init
```

Creates `config.json` with default settings. See [Configuration Guide](md/config.md) for details.

### 3. Process Video

```bash
just run ./videos/video.avi
```

### 4. Start API Server (Optional)

```bash
uvicorn api.server:create_app --host 0.0.0.0 --port 8080
```

## Architecture

```
worker-pkl/
├── core/               # Core processing components
│   ├── detector.py     # YOLO wrapper with tracking
│   ├── tracker.py      # Vehicle tracker
│   ├── parking_analyzer.py  # Parking detection logic
│   └── frame_processor.py   # Main processing pipeline
├── storage/            # Database layer
│   ├── models.py       # SQLAlchemy models
│   └── database.py     # Database operations
├── api/                # JSON-RPC API
│   └── server.py       # FastAPI server
├── utils/              # Utilities
│   ├── config.py       # Configuration management
│   └── logging.py      # Logging setup
├── md/                 # Documentation
│   └── config.md       # Configuration reference
├── config.json         # Configuration file
└── main.py             # CLI entry point
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
│  ┌──────────────────┐  ┌────────────────┐              │
│  │ Detector+Tracker │→ │ ParkingAnalyzer│→ Database    │
│  │   (BotSORT)      │  │  (IoU+Centroid)│              │
│  └──────────────────┘  └────────────────┘              │
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

## JSON-RPC API

### Get Vehicles

```bash
curl -X POST http://localhost:8080/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "get_vehicles",
    "params": {"video_id": 1},
    "id": 1
  }'
```

Response:
```json
{
  "jsonrpc": "2.0",
  "result": [
    {
      "id": 1,
      "track_id": 1,
      "first_seen": "2025-11-30T10:00:00",
      "last_seen": "2025-11-30T10:05:00",
      "vehicle_class": "car",
      "status": "parked"
    }
  ],
  "id": 1
}
```

### Get Vehicle History

```bash
curl -X POST http://localhost:8080/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "get_vehicle_history",
    "params": {"vehicle_id": 1},
    "id": 2
  }'
```

### List Videos

```bash
curl -X POST http://localhost:8080/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "get_videos",
    "params": {},
    "id": 3
  }'
```

## Database Schema

### ProcessedVideo
- Video metadata and processing stats

### Vehicle
- Unique tracked vehicles with status (moving/parked)
- Links to video via `video_id`
- COCO class: car, motorcycle, bus, truck

### VehiclePosition
- Position history for each vehicle
- Bounding box coordinates per frame
- Detection confidence scores

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
2025-11-30 23:15:09 | INFO     | core.detector:55 - Model loaded on device: cuda:0
2025-11-30 23:15:09 | INFO     | core.frame_processor:94 - Video info: 50016 frames, 60.00 FPS
2025-11-30 23:15:10 | DEBUG    | core.frame_processor:173 - Frame 12: detected+tracked 15 vehicles
2025-11-30 23:15:12 | INFO     | core.parking_analyzer:158 - Vehicle 5 status changed: moving → parked
```

## Development

```bash
# Run QA (format + lint) before every commit
just qa
```

## Performance

Tested on NVIDIA RTX 3060 (laptop):
- Model: YOLOv12x
- Resolution: 1920x1080
- Processing: ~15 FPS (frame_interval=2)
- Tracking: BotSORT

## Troubleshooting

### Issue: Parked vehicles marked as "moving"
- **Cause**: Detection variance causing bbox jitter
- **Fix**: Lower `iou_threshold` (0.2-0.4) and increase `centroid_threshold` (30-50)

### Issue: Moving vehicles marked as "parked"
- **Cause**: Thresholds too lenient
- **Fix**: Increase `iou_threshold` (0.5-0.7), decrease `centroid_threshold` (10-20)

### Issue: Track ID switches frequently
- **Cause**: ByteTrack struggling with occlusions
- **Fix**: Switch to `botsort.yaml` tracker

### Issue: Processing too slow
- **Fix**: Increase `frame_interval`, use smaller model (yolo11n.pt), disable frame saving
