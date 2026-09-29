"""Exercise stop paths with a fake pointer: never moves the real Windows cursor."""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np

from app import App
from gaze import CursorFilter, EyeObservation


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.app = App.__new__(App)
        self.app.active = True
        self.app.filter = CursorFilter()
        self.app.pointer = Mock()
        self.app.status = Mock()
        self.app.toggle_button = Mock()
        self.app.mapper = Mock()
        self.app.mapper.head_is_close.return_value = True
        self.app.mapper.predict.return_value = np.array([.5, .5])
        self.app.smoothing = Mock()
        self.app.smoothing.get.return_value = .15
        self.app.closed_since = self.app.missing_since = None
        self.app.calibration = None
        self.app.worker = Mock()

    def frame(self, openness=.25):
        return SimpleNamespace(observation=EyeObservation(np.zeros(2), openness, np.array([.5, .5, .3])))

    def test_blink_freezes_and_long_closure_pauses(self):
        self.app.track_frame(self.frame(.05), 10)
        self.assertTrue(self.app.active)
        self.app.track_frame(self.frame(.05), 11.3)
        self.assertFalse(self.app.active)
        self.app.pointer.move.assert_not_called()

    def test_missing_face_stops_pointer_immediately(self):
        self.app.track_frame(SimpleNamespace(observation=None), 10)
        self.app.track_frame(SimpleNamespace(observation=None), 11.1)
        self.assertFalse(self.app.active)
        self.app.pointer.move.assert_not_called()

    def test_head_movement_pauses(self):
        self.app.mapper.head_is_close.return_value = False
        self.app.track_frame(self.frame(), 10)
        self.assertFalse(self.app.active)
        self.app.pointer.move.assert_not_called()

    def test_valid_gaze_moves_fake_pointer(self):
        self.app.track_frame(self.frame(), 10)
        self.app.pointer.move.assert_called_once()

    def test_stale_frame_never_moves_pointer(self):
        self.app.pointer.pressed.return_value = False
        self.app.last_frame = None
        self.app.worker.latest.return_value = SimpleNamespace(timestamp=time.monotonic()-2, error="")
        self.app.process_tick()
        self.assertFalse(self.app.active)
        self.app.pointer.move.assert_not_called()

    def test_camera_stall_pauses(self):
        self.app.pointer.pressed.return_value = False
        self.app.worker.latest.return_value = None
        self.app.last_frame = SimpleNamespace(timestamp=time.monotonic()-2)
        self.app.process_tick()
        self.assertFalse(self.app.active)
        self.app.pointer.move.assert_not_called()


if __name__ == "__main__":
    unittest.main()
