"""YOLO vehicle detector wrapper."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from ultralytics import YOLO


@dataclass
class Detection:
    """Detection result."""

    bbox: tuple[float, float, float, float]
    confidence: float
    track_id: int | None = None


class VehicleDetector:
    """YOLO model wrapper for vehicle detection."""

    def __init__(self, model_path: str, device: str, conf_threshold: float):
        """Initialize detector.

        Args:
            model_path: Path to YOLO model
            device: Device for inference (cuda:0, cpu, etc.)
            conf_threshold: Confidence threshold
        """
        logger.info(f"Loading YOLO model: {model_path}")

        model_file = Path(model_path)
        if not model_file.exists():
            logger.warning(
                f"Model file not found locally, ultralytics will download: {model_path}"
            )

        self.model = YOLO(model_path)
        self.device = device
        self.conf_threshold = conf_threshold

        logger.success(f"Model loaded on device: {device}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on single frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detections
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        if not results or results[0].boxes is None:
            return []

        detections = []
        boxes = results[0].boxes.cpu().numpy()

        for box in boxes:
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                )
            )

        return detections

    def track(self, frame: np.ndarray, tracker: str = "bytetrack") -> list[Detection]:
        """Run detection with tracking on single frame.

        Args:
            frame: Input frame (BGR format)
            tracker: Tracker type (bytetrack, botsort)

        Returns:
            List of detections with track IDs
        """
        results = self.model.track(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            tracker=tracker,
            verbose=False,
            persist=True,
        )

        if not results or results[0].boxes is None:
            return []

        detections = []
        boxes = results[0].boxes.cpu().numpy()

        for box in boxes:
            if not hasattr(box, "id") or box.id is None:
                continue

            track_id = int(box.id[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                )
            )
            detections[-1].track_id = track_id

        return detections
