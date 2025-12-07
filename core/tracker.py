"""Vehicle tracking using ByteTrack."""

from dataclasses import dataclass


@dataclass
class Track:
    """Tracked object."""

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    age: int = 0
    hits: int = 0
    time_since_update: int = 0


class VehicleTracker:
    """Simple IoU tracker for fallback."""

    def __init__(self, min_hits: int = 3, max_age: int = 30):
        """Initialize tracker.

        Args:
            min_hits: Minimum hits to start tracking
            max_age: Maximum frames to keep lost track
        """
        self.min_hits = min_hits
        self.max_age = max_age
        self.tracks: list[Track] = []
        self.frame_count = 0

    def update(self, detections: list, frame_idx: int) -> list[Track]:
        """Update tracks with new detections.

        Args:
            detections: List of Detection objects
            frame_idx: Current frame index

        Returns:
            List of active tracks
        """
        self.frame_count += 1

        # Simple IoU matching (greedy)
        # This is a placeholder for a real tracker like SORT/ByteTrack
        # We are using built-in tracker mostly, this is fallback

        # Prediction (just age increment for now)
        for track in self.tracks:
            track.time_since_update += 1

        # Matching
        if not self.tracks:
            for det in detections:
                self._create_track(det)
        else:
            # Match existing tracks to detections
            # (Simplified logic for fallback)
            matched_indices = set()

            for track in self.tracks:
                best_iou = 0
                best_det_idx = -1

                for i, det in enumerate(detections):
                    if i in matched_indices:
                        continue

                    iou = self._compute_iou(track.bbox, det.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = i

                if best_iou > 0.3:  # Threshold
                    det = detections[best_det_idx]
                    track.bbox = det.bbox
                    track.confidence = det.confidence
                    track.hits += 1
                    track.time_since_update = 0
                    matched_indices.add(best_det_idx)

            # Create new tracks for unmatched detections
            for i, det in enumerate(detections):
                if i not in matched_indices:
                    self._create_track(det)

        # Filter dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

        # Return valid tracks
        return [
            t
            for t in self.tracks
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits
        ]

    def _create_track(self, detection) -> None:
        """Create new track from detection."""
        # Simple ID generation
        new_id = len(self.tracks) + 1
        if self.tracks:
            new_id = max(t.track_id for t in self.tracks) + 1

        self.tracks.append(
            Track(
                track_id=new_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                hits=1,
                time_since_update=0,
            )
        )

    @staticmethod
    def _compute_iou(bbox1, bbox2):
        """Compute IoU."""
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
        return inter_area / union_area if union_area > 0 else 0
