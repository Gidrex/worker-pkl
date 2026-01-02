from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.frame_processor import FrameProcessor
from src.storage.database import Database
from src.storage.models import Vehicle, VehiclePosition
from src.utils.config import Config


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"

@pytest.fixture
def database(db_path):
    return Database(str(db_path))

@pytest.fixture
def config(db_path):
    c = Config()
    c.storage.database_path = str(db_path)
    c.model.path = "yolov8n.pt" # dummy
    c.storage.save_frames = False
    return c

def test_save_vehicle_data(database, config):
        # Mock detector to avoid loading model
        with patch("src.core.frame_processor.VehicleDetector"):
            processor = FrameProcessor(config, database)

        # Override detector/tracker if needed, but we can just call _process_frame
        # But _process_frame does detection.
        # Let's just mock the methods called inside _process_frame to return what we want
        # or easier: create a method that injects tracks directly or mock the detector.track/detect return.

        # Set processor.use_builtin_tracker = True to skip local tracker logic for simplicity
        processor.use_builtin_tracker = True
        processor.detector.track.return_value = [] # Default

        # Create a mock track
        mock_track = MagicMock()
        mock_track.track_id = 1
        mock_track.bbox = (10, 10, 50, 50)
        mock_track.confidence = 0.9

        # We need to simulate the result of detector.track returning objects with these attributes
        # The code expects objects with .track_id, .bbox, .confidence

        # Actually processor._process_frame calls self.detector.track()
        processor.detector.track.return_value = [mock_track]

        # Run processing on a dummy frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        timestamp = datetime.utcnow()
        video_id = 1

        # Create video first
        database.create_video("test.mp4", 100, 0.0)

        # Inject parking analyzer state to avoid it being "unknown"
        processor.parking_analyzer.vehicle_states = {} # Reset
        # We rely on update to create state, which happens inside _process_frame

        processor._process_frame(frame, 0, timestamp, "test.mp4", video_id)

        # Now verify DB content
        with database.Session() as session:
            vehicles = session.query(Vehicle).all()
            assert len(vehicles) == 1
            assert vehicles[0].track_id == 1
            assert vehicles[0].video_id == video_id

            positions = session.query(VehiclePosition).all()
            assert len(positions) == 1
            assert positions[0].vehicle_id == vehicles[0].id
            assert positions[0].frame_idx == 0
            assert positions[0].confidence == 0.9

        # Process next frame (position change)
        mock_track.bbox = (20, 20, 60, 60)
        processor._process_frame(frame, 1, timestamp, "test.mp4", video_id)

        with database.Session() as session:
            positions = session.query(VehiclePosition).all()
            assert len(positions) == 2
            assert positions[1].frame_idx == 1
            assert positions[1].bbox_x1 == 20.0
