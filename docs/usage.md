# Vehicle Tracking System - Usage Guide

## Installation

```bash
just sync
```

## Workflow

**What it does:**
1. Scans `./videos/` for all video files (.mp4, .avi, .mov, .mkv, .flv)
2. Extracts every <config value. Default: 5>-th frame from videos
3. Auto-labels vehicles using pretrained COCO YOLOv9 model
4. Splits 80% train / 20% validation
5. Creates YOLO format dataset
racked.json` - Frame-by-frame detection results

**Visualization:**
- Green boxes = PARKED vehicles
- Red boxes = MOVING vehicles
- Yellow boxes = UNKNOWN (not enough frames)

## Supported Video Formats

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.flv`
