"""Video frame processor with parallel processing."""

import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from core.detector import VehicleDetector
from core.parking_analyzer import ParkingAnalyzer
from core.tracker import Track, VehicleTracker
from storage.database import Database
from utils.config import Config


class FrameProcessor:
    """Process video frames with detection, tracking, and parking analysis."""

    def __init__(self, config: Config, database: Database):
        """Initialize frame processor.

        Args:
            config: Configuration object
            database: Database instance
        """
        self.config = config
        self.database = database

        logger.info("Initializing frame processor components...")

        start = time.time()
        self.detector = VehicleDetector(
            model_path=config.model.path,
            device=config.model.device,
            conf_threshold=config.model.conf_threshold,
            classes=config.model.classes,
        )
        elapsed = time.time() - start
        logger.info(f"Model loaded in {elapsed:.2f}s")

        valid_trackers = ["bytetrack.yaml", "botsort.yaml"]
        self.use_builtin_tracker = config.tracking.tracker in valid_trackers
        self.tracker_type = config.tracking.tracker

        if not self.use_builtin_tracker:
            self.tracker = VehicleTracker(
                min_hits=config.tracking.min_hits,
                max_age=config.tracking.max_age,
            )

        self.parking_analyzer = ParkingAnalyzer(
            iou_threshold=config.parking.iou_threshold,
            centroid_threshold=config.parking.centroid_threshold,
            stationary_frames=config.parking.stationary_frames,
        )

        self.frame_interval = config.processing.frame_interval
        self.save_frames = config.storage.save_frames
        self.frames_dir = Path(config.storage.frames_dir)

        if self.save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

    def process_video(self, video_path: str) -> int:
        """Process video file.

        Args:
            video_path: Path to video file

        Returns:
            Video ID in database
        """
        video_file = Path(video_path)
        if not video_file.exists():
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Processing video: {video_file.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        logger.info(
            f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s"
        )

        start_time = time.time()
        processed_frames = 0
        vehicle_db_mapping = {}

        for frame_idx in range(0, total_frames, self.frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                break

            timestamp = datetime.now() + timedelta(seconds=frame_idx / fps)

            frame_start = time.time()
            self._process_frame(
                frame,
                frame_idx,
                timestamp,
                video_file.stem,
                vehicle_db_mapping,
                video_id=None,
            )
            frame_elapsed = time.time() - frame_start

            processed_frames += 1

            if processed_frames % 100 == 0:
                logger.debug(
                    f"Processed {processed_frames} frames, last frame: {frame_elapsed:.3f}s"
                )

        cap.release()

        total_time = time.time() - start_time
        logger.info(
            f"Video processing complete: {processed_frames} frames in {total_time:.2f}s"
        )

        video_id = self.database.create_video(
            filename=video_file.name,
            total_frames=processed_frames,
            processing_time=total_time,
        )

        self._save_vehicles_to_db(video_id, vehicle_db_mapping)

        logger.success(f"Video {video_file.name} processed: video_id={video_id}")

        return video_id.id

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: datetime,
        video_name: str,
        vehicle_db_mapping: dict,
        video_id: int | None,
    ) -> None:
        """Process single frame.

        Args:
            frame: Frame array
            frame_idx: Frame index
            timestamp: Frame timestamp
            video_name: Video name for frame saving
            vehicle_db_mapping: Mapping of track_id to vehicle_id
            video_id: Video database ID
        """
        if self.use_builtin_tracker:
            track_start = time.time()
            detections = self.detector.track(frame, tracker=self.tracker_type)
            track_time = time.time() - track_start

            logger.debug(
                f"Frame {frame_idx}: detected+tracked {len(detections)} vehicles in {track_time:.3f}s"
            )

            tracks = [
                Track(
                    track_id=det.track_id,
                    bbox=det.bbox,
                    class_id=det.class_id,
                    confidence=det.confidence,
                )
                for det in detections
                if det.track_id is not None
            ]
        else:
            detect_start = time.time()
            detections = self.detector.detect(frame)
            detect_time = time.time() - detect_start

            logger.debug(
                f"Frame {frame_idx}: detected {len(detections)} vehicles in {detect_time:.3f}s"
            )

            track_start = time.time()
            tracks = self.tracker.update(detections, frame_idx)
            track_time = time.time() - track_start

            logger.debug(
                f"Frame {frame_idx}: tracked {len(tracks)} vehicles in {track_time:.3f}s"
            )

        status_changes = self.parking_analyzer.update(tracks, frame_idx, timestamp)

        for track_id, old_status, new_status in status_changes:
            logger.info(
                f"Vehicle {track_id}: {old_status} → {new_status} at frame {frame_idx}"
            )

        if self.save_frames and len(tracks) > 0:
            save_start = time.time()
            self._save_frame_with_boxes(frame, tracks, frame_idx, video_name)
            save_time = time.time() - save_start
            logger.debug(f"Frame {frame_idx}: saved with boxes in {save_time:.3f}s")

    def _save_frame_with_boxes(
        self,
        frame: np.ndarray,
        tracks: list,
        frame_idx: int,
        video_name: str,
    ) -> None:
        """Save frame with bounding boxes.

        Args:
            frame: Frame array
            tracks: List of tracks
            frame_idx: Frame index
            video_name: Video name
        """
        frame_copy = frame.copy()

        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)

            state = self.parking_analyzer.get_vehicle_state(track.track_id)
            color = (0, 255, 0) if state and state.status == "parked" else (0, 0, 255)

            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{track.track_id}"
            if state:
                label += f" {state.status}"

            cv2.putText(
                frame_copy,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        output_path = self.frames_dir / f"{video_name}_frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(output_path), frame_copy)

    def _save_vehicles_to_db(self, video_id: int, vehicle_db_mapping: dict) -> None:
        """Save all vehicles to database after processing.

        Args:
            video_id: Video database ID
            vehicle_db_mapping: Mapping to populate
        """
        for track_id, state in self.parking_analyzer.vehicle_states.items():
            vehicle = self.database.create_vehicle(
                video_id=video_id,
                track_id=track_id,
                first_seen=state.first_seen,
                last_seen=state.last_seen,
                vehicle_class=self.detector.get_class_name(state.class_id),
                status=state.status,
            )

            vehicle_db_mapping[track_id] = vehicle.id

            for idx, (bbox, timestamp) in enumerate(
                zip(
                    state.bbox_history,
                    [state.first_seen] * len(state.bbox_history),
                    strict=False,
                )
            ):
                self.database.add_position(
                    vehicle_id=vehicle.id,
                    timestamp=timestamp,
                    frame_idx=idx,
                    bbox=bbox,
                    confidence=0.9,
                )

        logger.info(f"Saved {len(vehicle_db_mapping)} vehicles to database")
