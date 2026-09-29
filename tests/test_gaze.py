import unittest
from types import SimpleNamespace
import numpy as np

from gaze import GazeMapper, CursorFilter, extract_observation


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.targets = np.array([(x, y) for y in (.12, .5, .88) for x in (.12, .5, .88)])
        self.features = (self.targets - .5) * [.2, .12]
        self.heads = [[.5, .5, .3]] * 9

    def test_calibration_predicts_unseen_positions(self):
        mapper = GazeMapper()
        error = mapper.fit(self.features, self.targets, self.heads)
        self.assertLess(error, .03)
        for target in ([.3, .7], [.8, .2], [.5, .5]):
            feature = (np.array(target) - .5) * [.2, .12]
            np.testing.assert_allclose(mapper.predict(feature), target, atol=.025)

    def test_clamps_to_screen(self):
        mapper = GazeMapper()
        mapper.fit(self.features, self.targets, self.heads)
        output = mapper.predict([10, -10])
        self.assertTrue(np.all((output >= 0) & (output <= 1)))

    def test_rejects_no_eye_motion(self):
        with self.assertRaises(ValueError):
            GazeMapper().fit(np.zeros((9, 2)), self.targets, self.heads)

    def test_rejects_collinear_motion(self):
        f = np.column_stack([np.linspace(0, .1, 9)] * 2)
        with self.assertRaises(ValueError):
            GazeMapper().fit(f, self.targets, self.heads)

    def test_head_displacement_is_rejected(self):
        mapper = GazeMapper()
        mapper.fit(self.features, self.targets, self.heads)
        self.assertTrue(mapper.head_is_close([.51, .5, .31]))
        self.assertFalse(mapper.head_is_close([.7, .5, .3]))
        self.assertFalse(mapper.head_is_close([.5, .5, .5]))

    def test_filter_bounds_jump(self):
        filt = CursorFilter()
        filt.update([0, 0], 10)
        output = filt.update([1, 1], 10.02)
        self.assertLessEqual(np.linalg.norm(output), .040001)
        filt.reset()
        np.testing.assert_allclose(filt.update([.4, .6], 11), [.4, .6])

    def test_eye_features_ignore_translation_and_uniform_scale(self):
        points = np.zeros((478, 2))
        for offset, iris, left, right, upper, lower in (
            (100, 468, 33, 133, 159, 145), (250, 473, 362, 263, 386, 374)
        ):
            points[left], points[right] = [offset, 120], [offset+50, 120]
            points[upper], points[lower] = [offset+25, 113], [offset+25, 127]
            points[iris] = [offset+28, 122]
        def extract(p):
            return extract_observation([SimpleNamespace(x=x/640, y=y/480) for x, y in p], 640, 480)
        a, b = extract(points), extract(points * 1.2 + [30, 20])
        np.testing.assert_allclose(a.features, b.features)
        self.assertAlmostEqual(a.openness, b.openness)
        self.assertGreater(a.openness, .14)


if __name__ == "__main__":
    unittest.main()
