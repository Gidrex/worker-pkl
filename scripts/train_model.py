"""Train YOLOv9 model on prepared parking vehicle dataset."""

from pathlib import Path

from models.training import YOLOv9Trainer


def main():
    """Train model on parking vehicles dataset."""
    dataset_yaml = "./datasets/parking_vehicles/dataset.yaml"

    if not Path(dataset_yaml).exists():
        print("Error: Dataset not found!")
        print(f"Expected: {dataset_yaml}")
        print("\nRun dataset preparation first:")
        print("  uv run scripts/prepare_dataset.py")
        return

    print("=" * 80)
    print("YOLOV9 TRAINING - PARKING VEHICLES")
    print("=" * 80)

    trainer = YOLOv9Trainer(
        model_name="yolov9c.pt",
        device=0,
        output_dir="./weights",
    )

    results = trainer.train(
        data_yaml=dataset_yaml,
        epochs=100,
        batch_size=8,  # with cool GPU need to try 16
        imgsz=640,
        name="parking_vehicles",
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest weights: {results['best_weights']}")
    print(f"Last weights: {results['last_weights']}")
    print("\nTo use the trained model:")
    print("  model_path = './weights/parking_vehicles/weights/best.pt'")


if __name__ == "__main__":
    main()
