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

## Using the Virtual RTSP Server

This project includes a built-in environment for emulating IP cameras (RTSP servers) based on `Nix` and `MediaMTX`. This allows developing and testing streaming functionality without access to a physical camera.

### Prerequisites

To use the virtual server, you need the [Nix package manager](https://nixos.org/download.html) installed.

1. Enter the virtual environment:
   ```bash
   nix develop
   ```

### Core Commands

Inside the nix flake environment, the following tools are available:

1.  **Start Camera Emulation (One-Liner):**
    ```bash
    fake-cam ./videos/video.avi
    ```
    The stream will be available at: `rtsp://localhost:8554/cam1`

2.  **Manual Setup:**
    
    *Terminal 1 (Server):*
    ```bash
    mediamtx ./mediamtx.yml
    ```
    
    *Terminal 2 (Streamer):*
    ```bash
    ffmpeg -re -stream_loop -1 -i ./videos/video.avi -c copy -f rtsp rtsp://localhost:8554/cam1
    ```

### Viewing and Usage

*   **Verify Stream (MPV):**
    ```bash
    mpv rtsp://localhost:8554/cam1
    ```

*   **Run Analyzer on Stream:**
    ```bash
    uv run src/main.py process rtsp://localhost:8554/cam1
    ```

### Configuration

The server configuration is located in `mediamtx.yml`. By default, publishing streams to any path is allowed (`paths: all:`).
