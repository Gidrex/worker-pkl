# Worker PKL

Vehicle tracking system for detecting parked/moving cars using YOLOv9 and computer vision.

## Features

- Train custom YOLOv9 models on local videos
- Track vehicles with persistent IDs across frames
- Detect vehicle state: PARKED vs MOVING
- Real-time GUI visualization (green=parked, red=moving)
- Process every 3rd frame (66% faster)
- REST API for training and detection
- Optimized for RTX 3060 Laptop (6GB VRAM)

## Quick Start


## Documentation

- [**Usage Guide**](docs/usage.md) - Complete workflow and API reference
- [**Docker Guide**](docs/docker.md) - Docker setup for RTX 3060 Laptop with resource limits

## Requirements

- [uv](https://docs.astral.sh/uv/) - Python dependency manager
- [just](https://just.systems/) - Command runner
