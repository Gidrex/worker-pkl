"""JSON-RPC API server for parking detection system."""

from datetime import datetime
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from src.utils.config import Config, load_config
from storage.database import Database


class JSONRPCRequest(BaseModel):
    """JSON-RPC request."""

    jsonrpc: str = "2.0"
    method: str
    params: dict[str, Any] | None = None
    id: int | None = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC response."""

    jsonrpc: str = "2.0"
    result: Any | None = None
    error: dict[str, Any] | None = None
    id: int | None = None


class JSONRPCServer:
    """JSON-RPC server for database queries."""

    def __init__(self, config: Config):
        """Initialize JSON-RPC server.

        Args:
            config: Configuration object
        """
        self.config = config
        self.database = Database(config.storage.database_path)

        self.app = FastAPI(title="Parking Detection JSON-RPC API")

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.app.post("/rpc")(self.handle_rpc)
        self.app.get("/health")(self.health)

        logger.info("JSON-RPC server initialized")

    async def handle_rpc(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """Handle JSON-RPC request.

        Args:
            request: RPC request

        Returns:
            RPC response
        """
        logger.debug(f"RPC call: {request.method}")

        if request.method == "get_vehicles":
            return await self._get_vehicles(request)

        if request.method == "get_vehicle_history":
            return await self._get_vehicle_history(request)

        if request.method == "get_videos":
            return await self._get_videos(request)

        error = {
            "code": -32601,
            "message": f"Method not found: {request.method}",
        }

        return JSONRPCResponse(error=error, id=request.id)

    async def _get_vehicles(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """Get vehicles for video.

        Params:
            video_id: Video ID
            start_time: Optional start time (ISO format)
            end_time: Optional end time (ISO format)
        """
        params = request.params or {}

        video_id = params.get("video_id")
        if not video_id:
            return JSONRPCResponse(
                error={"code": -32602, "message": "Missing video_id parameter"},
                id=request.id,
            )

        start_time = None
        end_time = None

        if params.get("start_time"):
            start_time = datetime.fromisoformat(params["start_time"])

        if params.get("end_time"):
            end_time = datetime.fromisoformat(params["end_time"])

        vehicles = self.database.get_vehicles(video_id, start_time, end_time)

        result = [
            {
                "id": v.id,
                "track_id": v.track_id,
                "first_seen": v.first_seen.isoformat(),
                "last_seen": v.last_seen.isoformat(),
                "vehicle_class": v.vehicle_class,
                "status": v.status,
            }
            for v in vehicles
        ]

        return JSONRPCResponse(result=result, id=request.id)

    async def _get_vehicle_history(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """Get position history for vehicle.

        Params:
            vehicle_id: Vehicle ID
        """
        params = request.params or {}

        vehicle_id = params.get("vehicle_id")
        if not vehicle_id:
            return JSONRPCResponse(
                error={"code": -32602, "message": "Missing vehicle_id parameter"},
                id=request.id,
            )

        positions = self.database.get_vehicle_history(vehicle_id)

        result = [
            {
                "timestamp": p.timestamp.isoformat(),
                "frame_idx": p.frame_idx,
                "bbox": [p.bbox_x1, p.bbox_y1, p.bbox_x2, p.bbox_y2],
                "confidence": p.confidence,
            }
            for p in positions
        ]

        return JSONRPCResponse(result=result, id=request.id)

    async def _get_videos(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """Get all processed videos."""
        from storage.models import ProcessedVideo

        with self.database.Session() as session:
            videos = session.query(ProcessedVideo).all()

            result = [
                {
                    "id": v.id,
                    "filename": v.filename,
                    "processed_at": v.processed_at.isoformat(),
                    "total_frames": v.total_frames,
                    "processing_time": v.processing_time,
                }
                for v in videos
            ]

        return JSONRPCResponse(result=result, id=request.id)

    async def health(self):
        """Health check endpoint."""
        return {"status": "ok", "time": datetime.utcnow().isoformat()}


def create_app(config_path: str = "config.json") -> FastAPI:
    """Create FastAPI application.

    Args:
        config_path: Path to config file

    Returns:
        FastAPI app
    """
    config = load_config(config_path)
    server = JSONRPCServer(config)
    return server.app
