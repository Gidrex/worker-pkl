"""YOLOv9 training module for vehicle and parking detection on single GPU."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ultralytics import YOLO


class YOLOv9Trainer:
    """YOLOv9 trainer for vehicle and parking spot detection."""

    DATASETS = {
        "vehicles_coco": {
            "name": "COCO Vehicle Subset",
            "description": "Pretrained on COCO dataset (cars, buses, trucks, motorcycles)",
            "classes": ["person", "bicycle", "car", "motorcycle", "bus", "truck"],
            "url": "https://github.com/ultralytics/ultralytics",
            "notes": "Use pretrained COCO weights, filter vehicle classes",
        },
        "vehicles_kaggle": {
            "name": "Kaggle Vehicle Detection Dataset",
            "description": "Vehicle detection dataset from Kaggle",
            "url": "https://www.kaggle.com/datasets/sshikamaru/car-object-detection",
            "alternative_urls": [
                "https://www.kaggle.com/datasets/marquis03/vehicle-detection-dataset",
                "https://www.kaggle.com/datasets/dataclusterlabs/vehicle-detection",
            ],
            "notes": "Download via Kaggle API: kaggle datasets download -d sshikamaru/car-object-detection",
        },
        "cnrpark": {
            "name": "CNRPark Dataset",
            "description": "CNR Park dataset for parking space detection",
            "url": "http://cnrpark.it/",
            "download_url": "http://cnrpark.it/dataset/CNRPark-Patches-150x150.zip",
            "classes": ["free", "occupied"],
            "notes": "Parking spot classification dataset with labeled patches",
        },
        "cnrpark_ext": {
            "name": "CNRPark+EXT Dataset",
            "description": "Extended CNR Park dataset with weather variations",
            "url": "http://cnrpark.it/",
            "download_url": "http://cnrpark.it/dataset/CNR-EXT-Patches-150x150.zip",
            "classes": ["free", "occupied"],
            "notes": "Extended dataset with overcast and rainy weather conditions",
        },
        "pklot": {
            "name": "PKLot Dataset",
            "description": "Parking lot occupancy dataset",
            "url": "https://www.kaggle.com/datasets/blanderbuss/parking-lot-dataset",
            "notes": "Download via Kaggle API: kaggle datasets download -d blanderbuss/parking-lot-dataset",
        },
    }

    def __init__(
        self,
        model_name: str = "yolov9c.pt",
        device: int = 0,
        output_dir: str = "./weights",
    ):
        """
        Initialize YOLOv9 trainer.

        Args:
            model_name: Base model to use (yolov9c.pt, yolov9e.pt, etc.)
            device: GPU device ID (0 for first GPU)
            output_dir: Directory to save trained weights
        """
        self.model_name = model_name
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 640,
        name: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Train YOLOv9 model on custom dataset.

        Args:
            data_yaml: Path to dataset YAML configuration file
            epochs: Number of training epochs
            batch_size: Batch size for training
            imgsz: Input image size
            name: Experiment name (optional)
            **kwargs: Additional training arguments

        Returns:
            Training results dictionary
        """
        self.model = YOLO(self.model_name)

        if name is None:
            name = f"yolov9_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"Starting training: {name}")
        print(f"Dataset: {data_yaml}")
        print(f"Device: GPU {self.device}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {imgsz}")

        training_args = {
            "data": data_yaml,
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": imgsz,
            "device": self.device,
            "name": name,
            "project": str(self.output_dir),
            "patience": 50,
            "save": True,
            "save_period": 10,
            "cache": False,
            "workers": 8,
            "optimizer": "auto",
            "verbose": True,
            "seed": 42,
            "deterministic": True,
            "single_cls": False,
            "rect": False,
            "cos_lr": False,
            "close_mosaic": 10,
            "amp": True,
            "fraction": 1.0,
            "profile": False,
            "freeze": None,
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "pose": 12.0,
            "kobj": 1.0,
            "label_smoothing": 0.0,
            "nbs": 64,
            "overlap_mask": True,
            "mask_ratio": 4,
            "dropout": 0.0,
            "val": True,
        }

        training_args.update(kwargs)

        self.model.train(**training_args)

        best_weights = self.output_dir / name / "weights" / "best.pt"
        last_weights = self.output_dir / name / "weights" / "last.pt"

        training_summary = {
            "name": name,
            "model": self.model_name,
            "dataset": data_yaml,
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "device": self.device,
            "best_weights": str(best_weights),
            "last_weights": str(last_weights),
            "timestamp": datetime.now().isoformat(),
        }

        self._save_training_summary(training_summary, name)

        print("\nTraining complete!")
        print(f"Best weights: {best_weights}")
        print(f"Last weights: {last_weights}")

        return training_summary

    def validate(self, weights_path: str, data_yaml: str, **kwargs) -> dict[str, Any]:
        """
        Validate trained model.

        Args:
            weights_path: Path to trained weights
            data_yaml: Path to dataset YAML
            **kwargs: Additional validation arguments

        Returns:
            Validation results
        """
        model = YOLO(weights_path)

        print(f"Validating model: {weights_path}")

        val_args = {
            "data": data_yaml,
            "device": self.device,
            "batch": 16,
            "imgsz": 640,
            "verbose": True,
        }
        val_args.update(kwargs)

        results = model.val(**val_args)

        return results

    def export(
        self,
        weights_path: str,
        format: str = "onnx",
        **kwargs,
    ) -> str:
        """
        Export trained model to different formats.

        Args:
            weights_path: Path to trained weights
            format: Export format (onnx, torchscript, coreml, etc.)
            **kwargs: Additional export arguments

        Returns:
            Path to exported model
        """
        model = YOLO(weights_path)

        print(f"Exporting model to {format}...")

        export_args = {
            "format": format,
            "device": self.device,
        }
        export_args.update(kwargs)

        export_path = model.export(**export_args)

        print(f"Model exported to: {export_path}")

        return export_path

    def _save_training_summary(self, summary: dict[str, Any], name: str) -> None:
        """Save training summary to JSON file."""
        summary_path = self.output_dir / name / "training_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Training summary saved to: {summary_path}")

    @staticmethod
    def print_dataset_info():
        """Print information about available datasets."""
        print("\n" + "=" * 80)
        print("AVAILABLE DATASETS FOR TRAINING")
        print("=" * 80 + "\n")

        for key, dataset in YOLOv9Trainer.DATASETS.items():
            print(f"[{key.upper()}]")
            print(f"  Name: {dataset['name']}")
            print(f"  Description: {dataset['description']}")
            print(f"  URL: {dataset['url']}")

            if "download_url" in dataset:
                print(f"  Download: {dataset['download_url']}")

            if "classes" in dataset:
                print(f"  Classes: {', '.join(dataset['classes'])}")

            if "alternative_urls" in dataset:
                print("  Alternative URLs:")
                for url in dataset["alternative_urls"]:
                    print(f"    - {url}")

            print(f"  Notes: {dataset['notes']}")
            print()


def create_dataset_yaml(
    dataset_path: str,
    train_path: str,
    val_path: str,
    classes: list,
    output_path: str = "dataset.yaml",
) -> str:
    """
    Create dataset YAML configuration file for YOLOv9 training.

    Args:
        dataset_path: Root path to dataset
        train_path: Relative path to training images
        val_path: Relative path to validation images
        classes: List of class names
        output_path: Output YAML file path

    Returns:
        Path to created YAML file
    """
    config = {
        "path": dataset_path,
        "train": train_path,
        "val": val_path,
        "names": dict(enumerate(classes)),
    }

    output_file = Path(output_path)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"path: {config['path']}\n")
        f.write(f"train: {config['train']}\n")
        f.write(f"val: {config['val']}\n")
        f.write("\nnames:\n")
        for idx, name in config["names"].items():
            f.write(f"  {idx}: {name}\n")

    print(f"Dataset YAML created: {output_file}")
    return str(output_file)


if __name__ == "__main__":
    YOLOv9Trainer.print_dataset_info()

    trainer = YOLOv9Trainer(
        model_name="yolov9c.pt",
        device=0,
        output_dir="./weights",
    )
