"""Vehicle tracking using ByteTrack."""

from dataclasses import dataclass

from loguru import logger

from core.detector import Detection


@dataclass
class Track:
    """Vehicle track."""

    track_id: int
    bbox: tuple[float, float, float, float]
    class_id: int
    confidence: float


class VehicleTracker:
    """ByteTrack wrapper for vehicle tracking."""

    def __init__(self, min_hits: int = 3, max_age: int = 30):
        """Initialize tracker.

        Args:
            min_hits: Minimum hits to start a track
            max_age: Maximum age before track is deleted
        """
        self.min_hits = min_hits
        self.max_age = max_age
        self.next_id = 1
        self.tracks: dict[int, dict] = {}

        logger.info(f"Tracker initialized: min_hits={min_hits}, max_age={max_age}")

    def update(self, detections: list[Detection], frame_idx: int) -> list[Track]:
        """Update tracker with new detections.

        Args:
            detections: List of detections
            frame_idx: Current frame index

        Returns:
            List of active tracks
        """
        if not detections:
            self._age_tracks(frame_idx)
            return self._get_active_tracks()

        matched_tracks = []

        for detection in detections:
            matched = False

            for track_id, track_data in self.tracks.items():
                if track_data["class_id"] != detection.class_id:
                    continue

                if self._compute_iou(track_data["bbox"], detection.bbox) > 0.3:
                    track_data["bbox"] = detection.bbox
                    track_data["confidence"] = detection.confidence
                    track_data["last_seen"] = frame_idx
                    track_data["hits"] += 1
                    matched = True
                    matched_tracks.append(track_id)
                    break

            if not matched:
                new_track_id = self.next_id
                self.next_id += 1

                self.tracks[new_track_id] = {
                    "track_id": new_track_id,
                    "bbox": detection.bbox,
                    "class_id": detection.class_id,
                    "confidence": detection.confidence,
                    "first_seen": frame_idx,
                    "last_seen": frame_idx,
                    "hits": 1,
                }

                logger.debug(
                    f"New track created: id={new_track_id}, class={detection.class_id}"
                )

        self._age_tracks(frame_idx)

        return self._get_active_tracks()

    def _age_tracks(self, frame_idx: int) -> None:
        """Remove old tracks.

        Args:
            frame_idx: Current frame index
        """
        to_remove = []

        for track_id, track_data in self.tracks.items():
            age = frame_idx - track_data["last_seen"]

            if age > self.max_age:
                to_remove.append(track_id)
                logger.debug(f"Track {track_id} removed: age={age}")

        for track_id in to_remove:
            del self.tracks[track_id]

    def _get_active_tracks(self) -> list[Track]:
        """Get active tracks that meet minimum hits.

        Returns:
            List of active tracks
        """
        active_tracks = []

        for track_data in self.tracks.values():
            if track_data["hits"] >= self.min_hits:
                active_tracks.append(
                    Track(
                        track_id=track_data["track_id"],
                        bbox=track_data["bbox"],
                        class_id=track_data["class_id"],
                        confidence=track_data["confidence"],
                    )
                )

        return active_tracks

    @staticmethod
    def _compute_iou(bbox1: tuple, bbox2: tuple) -> float:
        """Compute IoU between two bounding boxes.

        Args:
            bbox1: First bbox (x1, y1, x2, y2)
            bbox2: Second bbox (x1, y1, x2, y2)

        Returns:
            IoU value
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)

        union_area = bbox1_area + bbox2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area
