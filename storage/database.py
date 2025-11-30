"""Database operations using SQLAlchemy."""

from datetime import datetime
from pathlib import Path

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from storage.models import Base, ProcessedVideo, Vehicle, VehiclePosition


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

    def create_vehicle(
        self,
        video_id: int,
        track_id: int,
        first_seen: datetime,
        last_seen: datetime,
        vehicle_class: str,
        status: str,
    ) -> Vehicle:
        """Create vehicle record.

        Args:
            video_id: Parent video ID
            track_id: Tracking ID
            first_seen: First detection timestamp
            last_seen: Last detection timestamp
            vehicle_class: Vehicle class (car/bus/truck/motorcycle)
            status: Vehicle status (moving/parked)

        Returns:
            Created vehicle record
        """
        with self.Session() as session:
            vehicle = Vehicle(
                video_id=video_id,
                track_id=track_id,
                first_seen=first_seen,
                last_seen=last_seen,
                vehicle_class=vehicle_class,
                status=status,
            )
            session.add(vehicle)
            session.commit()
            session.refresh(vehicle)
            logger.debug(
                f"Created vehicle: id={vehicle.id}, track_id={track_id}, class={vehicle_class}, status={status}"
            )
            return vehicle

    def update_vehicle_status(
        self, vehicle_id: int, status: str, last_seen: datetime
    ) -> None:
        """Update vehicle status.

        Args:
            vehicle_id: Vehicle ID
            status: New status
            last_seen: Last detection timestamp
        """
        with self.Session() as session:
            vehicle = session.query(Vehicle).get(vehicle_id)
            if not vehicle:
                logger.warning(f"Vehicle not found: id={vehicle_id}")
                return

            vehicle.status = status
            vehicle.last_seen = last_seen
            session.commit()
            logger.info(f"Vehicle {vehicle.track_id} status updated: {status}")

    def add_position(
        self,
        vehicle_id: int,
        timestamp: datetime,
        frame_idx: int,
        bbox: tuple[float, float, float, float],
        confidence: float,
    ) -> None:
        """Add vehicle position.

        Args:
            vehicle_id: Vehicle ID
            timestamp: Detection timestamp
            frame_idx: Frame index
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: Detection confidence
        """
        with self.Session() as session:
            position = VehiclePosition(
                vehicle_id=vehicle_id,
                timestamp=timestamp,
                frame_idx=frame_idx,
                bbox_x1=bbox[0],
                bbox_y1=bbox[1],
                bbox_x2=bbox[2],
                bbox_y2=bbox[3],
                confidence=confidence,
            )
            session.add(position)
            session.commit()
            logger.debug(f"Added position: vehicle_id={vehicle_id}, frame={frame_idx}")

    def get_vehicle(self, video_id: int, track_id: int) -> Vehicle | None:
        """Get vehicle by video and track ID.

        Args:
            video_id: Video ID
            track_id: Track ID

        Returns:
            Vehicle record or None
        """
        with self.Session() as session:
            return (
                session.query(Vehicle)
                .filter(Vehicle.video_id == video_id, Vehicle.track_id == track_id)
                .first()
            )

    def get_vehicles(
        self,
        video_id: int,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[Vehicle]:
        """Get vehicles for video with optional time filtering.

        Args:
            video_id: Video ID
            start_time: Start time filter
            end_time: End time filter

        Returns:
            List of vehicles
        """
        with self.Session() as session:
            query = session.query(Vehicle).filter(Vehicle.video_id == video_id)

            if start_time:
                query = query.filter(Vehicle.first_seen >= start_time)

            if end_time:
                query = query.filter(Vehicle.last_seen <= end_time)

            return query.all()

    def get_vehicle_history(self, vehicle_id: int) -> list[VehiclePosition]:
        """Get position history for vehicle.

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
