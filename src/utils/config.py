"""Configuration management with validation."""

import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}"
    rotation: str = "100 MB"


class ModelConfig(BaseModel):
    """YOLO model configuration."""

    path: str = "./models/yolo12x.pt"
    device: str = "cuda:0"
    conf_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    imgsz: int = Field(
        default=1280, ge=640, le=3840, description="Inference resolution"
    )
    augment: bool = Field(default=False, description="Enable Test Time Augmentation")


class ProcessingConfig(BaseModel):
    """Frame processing configuration."""

    frame_interval: int = Field(default=3, ge=1)
    batch_size: int = Field(default=8, ge=1)
    max_workers: int = Field(default=4, ge=1)


class TrackingConfig(BaseModel):
    """Vehicle tracking configuration."""

    tracker: str = Field(
        default="bytetrack.yaml",
        pattern="^(bytetrack|botsort|bytetrack\\.yaml|botsort\\.yaml)$",
    )
    min_hits: int = Field(default=3, ge=1)
    max_age: int = Field(default=30, ge=1)


class ParkingConfig(BaseModel):
    """Parking detection configuration."""

    iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    centroid_threshold: float = Field(default=15.0, ge=0.0)
    stationary_frames: int = Field(default=10, ge=1)


class StorageConfig(BaseModel):
    """Storage configuration."""

    database_path: str = "./data/parking.db"
    save_frames: bool = False
    frames_dir: str = "./data/frames"


class APIConfig(BaseModel):
    """JSON-RPC API configuration."""

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = Field(default=8844, ge=1, le=65535)


class Config(BaseModel):
    """Main configuration."""

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    parking: ParkingConfig = Field(default_factory=ParkingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    api: APIConfig = Field(default_factory=APIConfig)


def load_config(config_path: str = "config.json") -> Config:
    """Load and validate configuration from JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Validated configuration object
    """
    path = Path(config_path)

    if not path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("Creating default config.json")
        config = Config()
        save_config(config, config_path)
        return config

    with path.open() as file:
        data = json.load(file)

    return Config(**data)


def save_config(config: Config, config_path: str = "config.json") -> None:
    """Save configuration to JSON file.

    Args:
        config: Configuration object
        config_path: Path to save configuration
    """
    path = Path(config_path)

    with path.open("w") as file:
        json.dump(config.model_dump(), file, indent=2)


def main():
    """Generate default config.json."""
    config = Config()
    save_config(config)
    print("Created default config.json")


if __name__ == "__main__":
    sys.exit(main())
