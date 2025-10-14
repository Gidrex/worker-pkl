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

```bash
# Install dependencies
just sync

# Place videos in ./videos/
cp /path/to/videos/*.mp4 ./videos/

# Prepare dataset from videos
uv run scripts/prepare_dataset.py

# Train model
uv run scripts/train_model.py

# Test tracking
uv run scripts/test_tracking.py
```

## Documentation

- [**Usage Guide**](docs/usage.md) - Complete workflow and API reference
- [**Docker Guide**](docs/docker.md) - Docker setup for RTX 3060 Laptop with resource limits

## Requirements

- [uv](https://docs.astral.sh/uv/) - Python dependency manager
- [just](https://just.systems/) - Command runner
- NVIDIA GPU with CUDA support (optional, CPU fallback available)

## Just Commands

- `just sync` - Install and sync dependencies
- `just run` - Start API server
- `just lint` - Lint code with ruff
- `just format` - Format code
- `just qa` - Quality assurance (format + lint)
