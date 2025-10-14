"""Detection endpoints for vehicle tracking."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from models.yolov9 import YOLOv9Detector

router = APIRouter()


class DetectionRequest(BaseModel):
    """Detection request parameters."""

    video_path: str
    model_path: str = "yolov9c.pt"
    output_path: str | None = None
    frame_skip: int = 3
    movement_threshold: float = 15.0
    conf_threshold: float = 0.3


class DetectionResponse(BaseModel):
    """Detection response."""

    status: str
    message: str
    total_frames: int
    parked_total: int
    moving_total: int
    output_video: str | None = None
    results_json: str | None = None


@router.post("/track", response_model=DetectionResponse)
async def track_vehicles(request: DetectionRequest):
    """
    Track vehicles in video with parked/moving detection.

    Args:
        request: Detection parameters

    Returns:
        Detection results summary
    """
    if not Path(request.video_path).exists():
        raise HTTPException(
            status_code=404, detail=f"Video not found: {request.video_path}"
        )

    try:
        detector = YOLOv9Detector(
            model_path=request.model_path,
            options={"conf_threshold": request.conf_threshold, "device": 0},
        )

        results = detector.videoLocalTracked(
            video_path=request.video_path,
            output_path=request.output_path,
            save_results=True,
            show_realtime=False,
            frame_skip=request.frame_skip,
            movement_threshold=request.movement_threshold,
        )

        parked_total = sum(r["parked_count"] for r in results)
        moving_total = sum(r["moving_count"] for r in results)

        results_json = f"./results/{Path(request.video_path).stem}_tracked.json"

        return DetectionResponse(
            status="completed",
            message="Detection completed successfully",
            total_frames=len(results),
            parked_total=parked_total,
            moving_total=moving_total,
            output_video=request.output_path,
            results_json=results_json,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.get("/videos")
async def list_videos():
    """List available videos for detection."""
    videos_dir = Path("./videos")

    if not videos_dir.exists():
        return {"videos": []}

    videos = []
    for video_path in videos_dir.iterdir():
        if video_path.is_file() and video_path.suffix.lower() in [
            ".mp4",
            ".avi",
            ".mov",
        ]:
            videos.append({"name": video_path.name, "path": str(video_path)})

    return {"videos": videos}


@router.get("/results")
async def list_results():
    """List detection results."""
    results_dir = Path("./results")

    if not results_dir.exists():
        return {"results": []}

    results = []
    for result_path in results_dir.iterdir():
        if result_path.is_file() and result_path.suffix == ".json":
            results.append({"name": result_path.name, "path": str(result_path)})

    return {"results": results}
