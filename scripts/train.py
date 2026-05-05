import argparse
from pathlib import Path

import yaml
from loguru import logger
from ultralytics import YOLO


def train_model(
    data_path: str,
    model_name: str = "yolo26s.pt",  # Using latest state-of-the-art
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0",
    project: str = "runs/train",
    name: str = "parking_model",
):
    """
    Train the YOLO model with advanced augmentations and strategies.

    Args:
        data_path: Path to dataset.yaml
        model_name: Base model to start from (e.g., yolo26s.pt)
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image resolution
        device: CUDA device index or 'cpu'
        project: Save directory
        name: Experiment name
    """

    # Check if data file exists
    if not Path(data_path).exists():
        logger.error(f"Dataset config not found at {data_path}")
        # Create a template if it doesn't exist to help the user
        template = {
            "path": "../datasets/parking",
            "train": "images/train",
            "val": "images/val",
            "names": {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"},
        }
        with open("dataset_template.yaml", "w") as f:
            yaml.dump(template, f)
        logger.info(
            "Created 'dataset_template.yaml'. Please configure your dataset path."
        )
        return

    logger.info(f"Starting training with model {model_name}...")

    # Initialize model
    try:
        model = YOLO(model_name)
    except Exception as e:
        logger.warning(
            f"Could not load {model_name}, falling back to yolov8x.pt. Error: {e}"
        )
        model = YOLO("yolov8x.pt")

    # Hyperparameters & Augmentations
    # Note: These are passed to model.train()
    model.train(
        data=data_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        # Optimization
        optimizer="AdamW",
        cos_lr=True,  # Cosine annealing
        lr0=0.01,
        lrf=0.01,  # Final learning rate = lr0 * lrf
        warmup_epochs=3.0,
        # Augmentations (Advanced)
        mosaic=1.0,  # Mosaic augmentation (probability)
        mixup=0.15,  # Mixup augmentation (probability)
        copy_paste=0.1,  # Copy-paste augmentation
        degrees=0.0,  # Image rotation (+/- deg)
        translate=0.1,  # Image translation (+/- fraction)
        scale=0.5,  # Image scale (+/- gain)
        shear=0.0,  # Image shear (+/- deg)
        perspective=0.0,  # Image perspective (+/- fraction), range 0-0.001
        flipud=0.0,  # Image flip up-down (probability)
        fliplr=0.5,  # Image flip left-right (probability)
        # Loss Gains (can be tuned)
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # Post-processing during validation
        conf=0.25,
        iou=0.7,
        # Speed
        workers=8,
        cache=True,  # Cache images for faster training
        exist_ok=True,
        plots=True,
        save=True,
    )

    logger.success(f"Training completed. Results saved to {project}/{name}")

    # Export the best model
    success = model.export(format="onnx")
    logger.info(f"Model exported to ONNX: {success}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Parking Detection Model")
    parser.add_argument(
        "--data", type=str, default="dataset.yaml", help="Path to data.yaml"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--model", type=str, default="yolo26s.pt", help="Base model")

    args = parser.parse_args()

    train_model(
        data_path=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
    )
