"""Database operations using SQLAlchemy."""

from datetime import datetime
from pathlib import Path

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.storage.models import Base, ProcessedVideo, State


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
