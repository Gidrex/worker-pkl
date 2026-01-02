"""Database operations using SQLAlchemy."""

from datetime import datetime
from pathlib import Path

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.storage.models import Base, ProcessedVideo, State, Vehicle, VehiclePosition


class Database:
    """Database manager for vehicle tracking."""

    def __init__(self, database_path: str):
        """Initialize database connection.

        Args:
            database_path: Path to SQLite database file
        """
        db_path = Path(database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(f"sqlite:///{database_path}", future=True)
        self.Session = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized: {database_path}")

    def create_video(
        self,
        filename: str,
        total_frames: int,
        processing_time: float,
    ) -> ProcessedVideo:
        """Create video record.

        Args:
            filename: Video filename
            total_frames: Total number of frames
            processing_time: Processing time in seconds

        Returns:
            Created video record
        """
        with self.Session() as session:
            video = ProcessedVideo(
                filename=filename,
                processed_at=datetime.utcnow(),
                total_frames=total_frames,
                processing_time=processing_time,
            )
            session.add(video)
            session.commit()
            session.refresh(video)
            logger.debug(f"Created video record: id={video.id}, filename={filename}")
            return video

    def update_video_stats(
        self, video_id: int, total_frames: int, processing_time: float
    ) -> None:
        """Update video processing statistics.

        Args:
            video_id: Video ID
            total_frames: Total processed frames
            processing_time: Total processing time
        """
        with self.Session() as session:
            video = session.query(ProcessedVideo).get(video_id)
            if video:
                video.total_frames = total_frames
                video.processing_time = processing_time
                session.commit()
                logger.debug(f"Updated video stats: id={video_id}")

    def save_state(
        self,
        video_id: int,
        timestamp: datetime,
        parked_count: int,
        moving_count: int,
        total_count: int,
    ) -> None:
        """Save parking state statistics.

        Args:
            video_id: Video ID
            timestamp: Current timestamp
            parked_count: Number of parked vehicles
            moving_count: Number of moving vehicles
            total_count: Total vehicles
        """
        with self.Session() as session:
            state = State(
                video_id=video_id,
                timestamp=timestamp,
                parked_count=parked_count,
                moving_count=moving_count,
                total_count=total_count,
            )
            session.add(state)
            session.commit()

    def upsert_vehicle(
        self,
        video_id: int,
        track_id: int,
        timestamp: datetime,
        status: str = "unknown",
        vehicle_class: str = "car",
    ) -> int:
        """Create or update vehicle record.

        Args:
            video_id: Video ID
            track_id: Tracker ID
            timestamp: Current timestamp (for last_seen)
            status: Current vehicle status
            vehicle_class: Vehicle class

        Returns:
            Database ID of the vehicle
        """
        with self.Session() as session:
            vehicle = (
                session.query(Vehicle)
                .filter_by(video_id=video_id, track_id=track_id)
                .first()
            )

            if vehicle:
                vehicle.last_seen = timestamp
                vehicle.status = status
            else:
                vehicle = Vehicle(
                    video_id=video_id,
                    track_id=track_id,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    status=status,
                    vehicle_class=vehicle_class,
                )
                session.add(vehicle)

            session.commit()
            session.refresh(vehicle)
            return vehicle.id

    def save_vehicle_position(
        self,
        vehicle_id: int,
        frame_idx: int,
        timestamp: datetime,
        bbox: tuple[float, float, float, float],
        confidence: float,
        status: str,
    ) -> None:
        """Save vehicle position history.

        Args:
            vehicle_id: Database vehicle ID
            frame_idx: Frame index
            timestamp: Frame timestamp
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: Detection confidence
            status: Vehicle status at this frame
        """
        with self.Session() as session:
            x1, y1, x2, y2 = bbox
            pos = VehiclePosition(
                vehicle_id=vehicle_id,
                frame_idx=frame_idx,
                timestamp=timestamp,
                bbox_x1=x1,
                bbox_y1=y1,
                bbox_x2=x2,
                bbox_y2=y2,
                confidence=confidence,
                status=status,
            )
            session.add(pos)
            session.commit()

    def get_vehicles(
        self,
        video_id: int,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[Vehicle]:
        """Get vehicles for a video with optional time filtering.

        Args:
            video_id: Video ID
            start_time: Filter by last_seen >= start_time
            end_time: Filter by first_seen <= end_time

        Returns:
            List of vehicles
        """
        with self.Session() as session:
            query = session.query(Vehicle).filter(Vehicle.video_id == video_id)

            if start_time:
                query = query.filter(Vehicle.last_seen >= start_time)
            if end_time:
                query = query.filter(Vehicle.first_seen <= end_time)

            return query.all()

    def get_vehicle_history(self, vehicle_id: int) -> list[VehiclePosition]:
        """Get position history for a vehicle.

        Args:
            vehicle_id: Vehicle ID

        Returns:
            List of positions ordered by timestamp
        """
        with self.Session() as session:
            return (
                session.query(VehiclePosition)
                .filter(VehiclePosition.vehicle_id == vehicle_id)
                .order_by(VehiclePosition.timestamp)
                .all()
            )
