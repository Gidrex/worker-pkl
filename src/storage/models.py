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

    states = relationship("State", back_populates="video", cascade="all, delete-orphan")
    vehicles = relationship(
        "Vehicle", back_populates="video", cascade="all, delete-orphan"
    )


class State(Base):
    """Parking state statistics over time."""

    __tablename__ = "state"

    id = Column(Integer, primary_key=True)

    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)

    timestamp = Column(DateTime, nullable=False, index=True)

    parked_count = Column(Integer, nullable=False, default=0)

    moving_count = Column(Integer, nullable=False, default=0)

    total_count = Column(Integer, nullable=False, default=0)

    video = relationship("ProcessedVideo", back_populates="states")


class Vehicle(Base):
    """Detected vehicle metadata."""

    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)
    track_id = Column(Integer, nullable=False)

    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)

    vehicle_class = Column(String, default="car")
    status = Column(String, default="unknown")

    video = relationship("ProcessedVideo", back_populates="vehicles")
    positions = relationship(
        "VehiclePosition", back_populates="vehicle", cascade="all, delete-orphan"
    )


class VehiclePosition(Base):
    """Vehicle position history."""

    __tablename__ = "vehicle_positions"

    id = Column(Integer, primary_key=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=False, index=True)

    timestamp = Column(DateTime, nullable=False, index=True)
    frame_idx = Column(Integer, nullable=False)

    bbox_x1 = Column(Float)
    bbox_y1 = Column(Float)
    bbox_x2 = Column(Float)
    bbox_y2 = Column(Float)

    confidence = Column(Float)
    status = Column(String)

    vehicle = relationship("Vehicle", back_populates="positions")
