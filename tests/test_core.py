from src.core.tracker import VehicleTracker
from src.utils.config import ModelConfig


def test_config_no_classes():
    """Test that model config no longer has classes field."""
    config = ModelConfig()
    assert not hasattr(config, "classes")


def test_tracker_no_classes():
    """Test that tracker works without class_id."""
    tracker = VehicleTracker(min_hits=1)

    # Simulate a detection
    class MockDetection:
        def __init__(self, bbox, confidence):
            self.bbox = bbox
            self.confidence = confidence

    detections = [
        MockDetection((0, 0, 10, 10), 0.9),
        MockDetection((100, 100, 110, 110), 0.8),
    ]

    tracks = tracker.update(detections, frame_idx=0)

    assert len(tracks) == 2
    assert not hasattr(tracks[0], "class_id")
    assert tracks[0].bbox == (0, 0, 10, 10)


def test_tracker_matching():
    """Test tracker matching without class constraints."""
    tracker = VehicleTracker(min_hits=1)

    # Frame 0
    detections1 = [MockDetection((0, 0, 10, 10), 0.9)]
    tracks1 = tracker.update(detections1, 0)
    id1 = tracks1[0].track_id

    # Frame 1 (moved slightly)
    detections2 = [MockDetection((1, 1, 11, 11), 0.95)]
    tracks2 = tracker.update(detections2, 1)

    assert len(tracks2) == 1
    assert tracks2[0].track_id == id1


class MockDetection:
    def __init__(self, bbox, confidence):
        self.bbox = bbox
        self.confidence = confidence
