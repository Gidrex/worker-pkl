"""Training endpoints for YOLOv9 model training."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from models.training import YOLOv9Trainer

router = APIRouter()


class TrainingRequest(BaseModel):
    """Training request parameters."""

    data_yaml: str
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    name: str | None = None
    device: int = 0


class TrainingStatusResponse(BaseModel):
    """Training status response."""

    status: str
    message: str
    results: dict[str, Any] | None = None


@router.post("/start", response_model=TrainingStatusResponse)
async def start_training(request: TrainingRequest):
    """
    Start model training.

    Args:
        request: Training parameters

    Returns:
        Training status and results
    """
    if not Path(request.data_yaml).exists():
        raise HTTPException(
            status_code=404, detail=f"Dataset YAML not found: {request.data_yaml}"
        )

    try:
        trainer = YOLOv9Trainer(
            model_name="yolov9c.pt", device=request.device, output_dir="./weights"
        )

        results = trainer.train(
            data_yaml=request.data_yaml,
            epochs=request.epochs,
            batch_size=request.batch_size,
            imgsz=request.imgsz,
            name=request.name,
        )

        return TrainingStatusResponse(
            status="completed",
            message="Training completed successfully",
            results=results,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/datasets")
async def list_datasets():
    """List available datasets for training."""
    datasets_dir = Path("./datasets")

    if not datasets_dir.exists():
        return {"datasets": []}

    datasets = []
    for dataset_path in datasets_dir.iterdir():
        if dataset_path.is_dir():
            yaml_path = dataset_path / "dataset.yaml"
            if yaml_path.exists():
                datasets.append(
                    {
                        "name": dataset_path.name,
                        "path": str(dataset_path),
                        "yaml": str(yaml_path),
                    }
                )

    return {"datasets": datasets}


@router.get("/models")
async def list_trained_models():
    """List trained models."""
    weights_dir = Path("./weights")

    if not weights_dir.exists():
        return {"models": []}

    models = []
    for model_dir in weights_dir.iterdir():
        if model_dir.is_dir():
            best_weights = model_dir / "weights" / "best.pt"
            if best_weights.exists():
                models.append(
                    {
                        "name": model_dir.name,
                        "path": str(best_weights),
                        "directory": str(model_dir),
                    }
                )

    return {"models": models}
