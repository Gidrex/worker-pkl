"""Parking detection analyzer using IoU and centroid."""

from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from core.tracker import Track


@dataclass
class VehicleState:
    """Vehicle state information."""

    track_id: int
    class_id: int
    status: str
    first_seen: datetime
    last_seen: datetime
    bbox_history: list[tuple[float, float, float, float]]
    centroid_history: list[tuple[float, float]]
    stationary_count: int


class ParkingAnalyzer:
    """Analyze vehicle parking status using IoU and centroid."""

    STATUS_MOVING = "moving"
    STATUS_PARKED = "parked"

    def __init__(
        self,
        iou_threshold: float = 0.7,
        centroid_threshold: float = 15.0,
        stationary_frames: int = 10,
    ):
        """Initialize parking analyzer.

        Args:
            iou_threshold: IoU threshold for stationary detection
            centroid_threshold: Centroid distance threshold (pixels)
            stationary_frames: Frames required to mark as parked
        """
        self.iou_threshold = iou_threshold
        self.centroid_threshold = centroid_threshold
        self.stationary_frames = stationary_frames

        self.vehicle_states: dict[int, VehicleState] = {}

        logger.info(
            f"ParkingAnalyzer initialized: iou={iou_threshold}, "
            f"centroid={centroid_threshold}, frames={stationary_frames}"
        )

    def update(
        self,
        tracks: list[Track],
        frame_idx: int,
        timestamp: datetime,
    ) -> list[tuple[int, str, str]]:
        """Update vehicle states and detect parking.

        Args:
            tracks: List of active tracks
            frame_idx: Current frame index
            timestamp: Current timestamp

        Returns:
            List of status changes (track_id, old_status, new_status)
        """
        status_changes = []

        for track in tracks:
            if track.track_id not in self.vehicle_states:
                self._create_vehicle_state(track, timestamp)
                continue

            change = self._update_vehicle_state(track, timestamp)
            if change:
                status_changes.append(change)

        return status_changes

    def _create_vehicle_state(self, track: Track, timestamp: datetime) -> None:
        """Create new vehicle state.

        Args:
            track: Vehicle track
            timestamp: Current timestamp
        """
        centroid = self._compute_centroid(track.bbox)

        self.vehicle_states[track.track_id] = VehicleState(
            track_id=track.track_id,
            class_id=track.class_id,
            status=self.STATUS_MOVING,
            first_seen=timestamp,
            last_seen=timestamp,
            bbox_history=[track.bbox],
            centroid_history=[centroid],
            stationary_count=0,
        )

        logger.debug(f"Vehicle state created: track_id={track.track_id}")

    def _update_vehicle_state(
        self,
        track: Track,
        timestamp: datetime,
    ) -> tuple[int, str, str] | None:
        """Update existing vehicle state.

        Args:
            track: Vehicle track
            timestamp: Current timestamp

        Returns:
            Status change tuple if status changed, None otherwise
        """
        state = self.vehicle_states[track.track_id]
        state.last_seen = timestamp

        centroid = self._compute_centroid(track.bbox)

        is_stationary = self._is_stationary(
            track.bbox,
            centroid,
            state.bbox_history[-1],
            state.centroid_history[-1],
        )

        state.bbox_history.append(track.bbox)
        state.centroid_history.append(centroid)

        if len(state.bbox_history) > self.stationary_frames * 2:
            state.bbox_history = state.bbox_history[-self.stationary_frames * 2 :]
            state.centroid_history = state.centroid_history[
                -self.stationary_frames * 2 :
            ]

        if is_stationary:
            state.stationary_count += 1
        else:
            state.stationary_count = 0

        old_status = state.status

        if state.stationary_count >= self.stationary_frames:
            state.status = self.STATUS_PARKED
        else:
            state.status = self.STATUS_MOVING

        if old_status != state.status:
            logger.info(
                f"Vehicle {track.track_id} status changed: {old_status} → {state.status}"
            )
            return (track.track_id, old_status, state.status)

        return None

    def _is_stationary(
        self,
        bbox: tuple,
        centroid: tuple,
        prev_bbox: tuple,
        prev_centroid: tuple,
    ) -> bool:
        """Check if vehicle is stationary using IoU and centroid.

        Args:
            bbox: Current bounding box
            centroid: Current centroid
            prev_bbox: Previous bounding box
            prev_centroid: Previous centroid

        Returns:
            True if vehicle is stationary
        """
        iou = self._compute_iou(bbox, prev_bbox)
        centroid_dist = self._compute_distance(centroid, prev_centroid)

        is_stationary = (
            iou >= self.iou_threshold and centroid_dist <= self.centroid_threshold
        )

        return is_stationary

    def get_vehicle_state(self, track_id: int) -> VehicleState | None:
        """Get vehicle state by track ID.

        Args:
            track_id: Track ID

        Returns:
            Vehicle state or None
        """
        return self.vehicle_states.get(track_id)

    @staticmethod
    def _compute_centroid(bbox: tuple) -> tuple[float, float]:
        """Compute bbox centroid.

        Args:
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Centroid (x, y)
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _compute_distance(point1: tuple, point2: tuple) -> float:
        """Compute Euclidean distance between two points.

        Args:
            point1: First point (x, y)
            point2: Second point (x, y)

        Returns:
            Distance in pixels
        """
        x1, y1 = point1
        x2, y2 = point2
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

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
