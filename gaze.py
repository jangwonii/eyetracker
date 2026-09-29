"""Eye-relative features and calibrated mapping. No camera or OS side effects."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class EyeObservation:
    features: np.ndarray
    openness: float
    head: np.ndarray


def extract_observation(landmarks, width, height):
    if len(landmarks) < 478:
        return None
    points = np.array([(p.x * width, p.y * height) for p in landmarks])
    features, openings = [], []
    # Corner order is consistent in image coordinates for both eyes.
    for iris, left, right, upper, lower in (
        (468, 33, 133, 159, 145), (473, 362, 263, 386, 374)
    ):
        axis = points[right] - points[left]
        eye_width = np.linalg.norm(axis)
        if eye_width < 8:
            return None
        horizontal = axis / eye_width
        vertical = np.array([-horizontal[1], horizontal[0]])
        relative = points[iris] - (points[left] + points[right]) / 2
        features.append([relative @ horizontal / eye_width,
                         relative @ vertical / eye_width])
        openings.append(abs((points[lower] - points[upper]) @ vertical) / eye_width)
    eye_center = (points[33] + points[263]) / 2
    face_width = np.linalg.norm(points[263] - points[33])
    if face_width < 30:
        return None
    head = np.array([eye_center[0] / width, eye_center[1] / height,
                     face_width / width])
    return EyeObservation(np.mean(features, axis=0), min(openings), head)


class GazeMapper:
    """Regularized quadratic mapping learned from nine screen targets."""
    def __init__(self):
        self.weights = None
        self.mean = self.scale = self.head = None
        self.validation_error = None

    @staticmethod
    def design(z):
        x, y = z[..., 0], z[..., 1]
        return np.stack([np.ones_like(x), x, y, x*y, x*x, y*y], axis=-1)

    def fit(self, features, targets, heads):
        f, t = np.asarray(features), np.asarray(targets)
        if f.shape != (9, 2) or t.shape != (9, 2) or not np.isfinite(f).all():
            raise ValueError("보정 데이터가 부족합니다. 다시 보정해 주세요.")
        self.mean, self.scale = f.mean(axis=0), f.std(axis=0)
        if np.any(self.scale < 0.006):
            raise ValueError("눈 움직임이 충분히 구분되지 않습니다. 점을 정확히 바라보며 다시 보정해 주세요.")
        z = (f - self.mean) / self.scale
        if np.linalg.svd(z, compute_uv=False)[-1] < 0.4:
            raise ValueError("가로·세로 시선이 구분되지 않습니다. 머리를 고정하고 다시 보정해 주세요.")
        a = self.design(z)
        penalty = np.diag([0, .08, .08, .3, .3, .3])
        # Hold out an entire target, rather than neighbouring frames of that target.
        errors = []
        for i in range(9):
            keep = np.arange(9) != i
            w = np.linalg.solve(a[keep].T @ a[keep] + penalty, a[keep].T @ t[keep])
            errors.append(np.linalg.norm(a[i] @ w - t[i]))
        error = float(np.mean(errors))
        if not math.isfinite(error) or error > .24:
            raise ValueError("보정 오차가 큽니다. 조명·카메라 각도를 조정하고 다시 보정해 주세요.")
        self.weights = np.linalg.solve(a.T @ a + penalty, a.T @ t)
        self.head = np.median(heads, axis=0)
        self.validation_error = error
        return error

    def predict(self, features):
        if self.weights is None:
            raise ValueError("보정이 필요합니다.")
        xy = self.design((np.asarray(features) - self.mean) / self.scale) @ self.weights
        if not np.isfinite(xy).all():
            raise ValueError("유효하지 않은 시선 좌표입니다.")
        return np.clip(xy, 0, 1)

    def head_is_close(self, head):
        if self.head is None:
            return False
        displacement = np.linalg.norm(np.asarray(head)[:2] - self.head[:2])
        return displacement < self.head[2] * .20 and .75 < head[2] / self.head[2] < 1.3


class CursorFilter:
    def __init__(self):
        self.position = None
        self.timestamp = None

    def reset(self):
        self.position = self.timestamp = None

    def update(self, target, now, smoothing=.12):
        target = np.asarray(target, dtype=float)
        if self.position is None or now - self.timestamp > .4:
            self.position = target.copy()
        else:
            dt = max(.001, now - self.timestamp)
            alpha = 1 - math.exp(-dt / max(.02, smoothing))
            delta = (target - self.position) * alpha
            # Bound sudden jumps to roughly two screen widths per second.
            length = np.linalg.norm(delta)
            self.position += delta * min(1., 2 * dt / max(length, 1e-9))
        self.timestamp = now
        return self.position.copy()
