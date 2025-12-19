"""Video frame processor with parallel processing."""

import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from src.core.detector import VehicleDetector
from src.core.parking_analyzer import ParkingAnalyzer
from src.core.reid import VehicleReID
from src.core.tracker import Track, VehicleTracker
from src.storage.database import Database
from src.utils.config import Config


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
        )
        elapsed = time.time() - start
        logger.info(f"Model loaded in {elapsed:.2f}s")

        # Initialize ReID
        self.reid = VehicleReID(device=config.model.device)
        self.vehicle_embeddings = {}  # track_id -> embedding (np.array)

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
        """Process video file or stream.

        Args:
            video_path: Path to video file or URL stream

        Returns:
            Video ID in database
        """
        is_url = str(video_path).lower().startswith(('rtsp://', 'http://', 'https://'))
        video_file = Path(video_path)

        if not is_url and not video_file.exists():
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video not found: {video_path}")

        video_name = "stream" if is_url else video_file.name
        if is_url:
             # Generate a timestamp-based name for streams to ensure uniqueness
             video_name = f"stream_{int(time.time())}"

        logger.info(f"Processing video source: {video_name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # For streams, total_frames might be garbage or -1
        if total_frames <= 0:
             total_frames = -1
        
        duration = total_frames / fps if fps > 0 and total_frames > 0 else 0

        logger.info(
            f"Video info: {total_frames if total_frames > 0 else 'Unknown'} frames, "
            f"{fps:.2f} FPS, {duration:.2f}s"
        )

        # Create video record immediately
        video_record = self.database.create_video(
            filename=video_name,
            total_frames=total_frames if total_frames > 0 else 0,
            processing_time=0.0,
        )
        video_id = video_record.id

        start_time = time.time()
        processed_frames = 0
        frame_idx = 0
        
        # Retry mechanism for streams
        max_retries = 50
        retry_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    if is_url:
                        retry_count += 1
                        if retry_count % 10 == 0:  # Log every 10th failure to avoid spam
                             logger.warning(f"Failed to read frame from stream (attempt {retry_count}/{max_retries})")
                        
                        if retry_count > max_retries:
                            logger.error("Max retries exceeded for stream. Exiting.")
                            break
                        
                        time.sleep(0.1)  # Wait briefly before retrying
                        continue
                    else:
                        break  # End of file
                
                # Reset retry count on successful read
                retry_count = 0
                
                # Skip frames if needed to match frame_interval
                if frame_idx % self.frame_interval != 0:
                    frame_idx += 1
                    continue

                # Estimate timestamp if not available (simple incremental)
                # For streams, we might want real wall-clock time, but keeping it consistent with file logic for now
                # If it's a file, we calculate based on frame index.
                if fps > 0:
                    timestamp = datetime.now() + timedelta(seconds=frame_idx / fps)
                else:
                    # Fallback for streams with no FPS info -> just use current time
                    timestamp = datetime.now()

                frame_start = time.time()
                self._process_frame(
                    frame,
                    frame_idx,
                    timestamp,
                    video_name,
                    video_id,
                )
                frame_elapsed = time.time() - frame_start

                processed_frames += 1
                frame_idx += 1

                if processed_frames % 100 == 0:
                    logger.debug(
                        f"Processed {processed_frames} frames, last frame: {frame_elapsed:.3f}s"
                    )

        finally:
            cap.release()
            total_time = time.time() - start_time

            # Update video stats
            self.database.update_video_stats(
                video_id=video_id,
                total_frames=processed_frames,
                processing_time=total_time,
            )

            logger.info(
                f"Video processing complete: {processed_frames} frames in {total_time:.2f}s"
            )
            logger.success(f"Video {video_name} processed: video_id={video_id}")

        return video_id

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: datetime,
        video_name: str,
        video_id: int,
    ) -> None:
        """Process single frame.

        Args:
            frame: Frame array
            frame_idx: Frame index
            timestamp: Frame timestamp
            video_name: Video name for frame saving
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

        # Calculate Frame Statistics
        parked_count = 0
        moving_count = 0

        # We only count ACTIVE vehicles visible in the current frame (from tracks)
        # However, parking_analyzer maintains state even if detection missed for a few frames.
        # But for 'how many cars now', we should look at active tracks or known states.
        # Let's use the states of currently tracked vehicles.

        active_states = []
        for track in tracks:
            state = self.parking_analyzer.get_vehicle_state(track.track_id)
            if state:
                active_states.append(state)
                if state.status == "parked":
                    parked_count += 1
                else:
                    moving_count += 1

        total_count = len(tracks)

        # Save State to DB
        self.database.save_state(
            video_id=video_id,
            timestamp=timestamp,
            parked_count=parked_count,
            moving_count=moving_count,
            total_count=total_count,
        )

        if self.save_frames and len(tracks) > 0:
            save_start = time.time()
            self._save_frame_with_boxes(frame, tracks, frame_idx, video_name, timestamp)
            save_time = time.time() - save_start
            logger.debug(f"Frame {frame_idx}: saved with boxes in {save_time:.3f}s")

    def _save_frame_with_boxes(
        self,
        frame: np.ndarray,
        tracks: list,
        frame_idx: int,
        video_name: str,
        timestamp: datetime,
    ) -> None:
        """Save frame with bounding boxes.

        Args:
            frame: Frame array
            tracks: List of tracks
            frame_idx: Frame index
            video_name: Video name
            timestamp: Frame timestamp
        """
        frame_copy = frame.copy()

        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)

            state = self.parking_analyzer.get_vehicle_state(track.track_id)
            color = (0, 255, 0) if state and state.status == "parked" else (0, 0, 255)

            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{track.track_id} {track.confidence:.2f}"
            if state:
                label += f" {state.status}"

            cv2.putText(
                frame_copy,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        timestamp_str = timestamp.strftime("%d%m%Y_%H%M%S")
        output_path = (
            self.frames_dir / f"{video_name}_{timestamp_str}_frame_{frame_idx:06d}.jpg"
        )
        cv2.imwrite(str(output_path), frame_copy)
