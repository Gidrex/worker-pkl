"""Database models for vehicle tracking."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class ProcessedVideo(Base):
    """Video processing metadata."""

    __tablename__ = "videos"

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False, index=True)
    processed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    total_frames = Column(Integer)
    processing_time = Column(Float)
    vehicles = relationship(
        "Vehicle", back_populates="video", cascade="all, delete-orphan"
    )


class Vehicle(Base):
    """Vehicle track metadata."""

    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)
    track_id = Column(Integer, nullable=False, index=True)
    first_seen = Column(DateTime, nullable=False, index=True)
    last_seen = Column(DateTime, nullable=False)
    vehicle_class = Column(String, nullable=False)
    status = Column(String, nullable=False)

    video = relationship("ProcessedVideo", back_populates="vehicles")
    positions = relationship(
        "VehiclePosition", back_populates="vehicle", cascade="all, delete-orphan"
    )


class VehiclePosition(Base):
    """Position history for each vehicle."""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    frame_idx = Column(Integer, nullable=False)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)

    vehicle = relationship("Vehicle", back_populates="positions")
