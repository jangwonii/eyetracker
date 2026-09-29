"""Local camera worker. Only the latest result is retained; nothing is recorded."""
from dataclasses import dataclass
import queue
import threading
import time

import cv2
import mediapipe as mp

from download_model import MODEL_PATH
from gaze import extract_observation


@dataclass
class Frame:
    timestamp: float
    image: object = None
    observation: object = None
    error: str = ""


class CameraWorker(threading.Thread):
    def __init__(self, index=0):
        super().__init__(daemon=True)
        self.index = index
        self.frames = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()

    def publish(self, frame):
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        self.frames.put_nowait(frame)

    def latest(self):
        try:
            return self.frames.get_nowait()
        except queue.Empty:
            return None

    def run(self):
        capture = None
        try:
            if not MODEL_PATH.is_file():
                raise RuntimeError("모델이 없습니다. setup.ps1을 먼저 실행해 주세요.")
            # Bytes avoid native Windows path issues with Korean user directories.
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=.65,
                min_face_presence_confidence=.65,
                min_tracking_confidence=.65,
            )
            capture = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
            if not capture.isOpened():
                raise RuntimeError("카메라를 열 수 없습니다. 카메라 번호·Windows 권한·다른 앱 사용 여부를 확인해 주세요.")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capture.set(cv2.CAP_PROP_FPS, 30)
            previous_ms = -1
            failures = 0
            with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
                while not self.stop_event.is_set():
                    ok, bgr = capture.read()
                    now = time.monotonic()
                    if not ok:
                        failures += 1
                        self.publish(Frame(now, error="카메라 프레임을 읽지 못했습니다."))
                        if failures >= 20:
                            raise RuntimeError("카메라 연결이 끊겼습니다. 앱을 다시 실행해 주세요.")
                        self.stop_event.wait(.05)
                        continue
                    failures = 0
                    rgb = cv2.cvtColor(cv2.flip(bgr, 1), cv2.COLOR_BGR2RGB)
                    timestamp_ms = max(previous_ms + 1, int(now * 1000))
                    previous_ms = timestamp_ms
                    result = detector.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                    observation = None
                    if result.face_landmarks:
                        landmarks = result.face_landmarks[0]
                        h, w = rgb.shape[:2]
                        observation = extract_observation(landmarks, w, h)
                        for i in (468, 473):
                            if len(landmarks) > i:
                                p = landmarks[i]
                                cv2.circle(rgb, (int(p.x*w), int(p.y*h)), 3, (60, 240, 160), -1)
                    self.publish(Frame(now, rgb, observation))
        except Exception as exc:
            self.publish(Frame(time.monotonic(), error=str(exc)))
        finally:
            if capture is not None:
                capture.release()


def probe(index=0, seconds=8):
    """Hardware smoke check. Does not display/save images or move the pointer."""
    worker = CameraWorker(index)
    worker.start()
    frames = faces = 0
    error = ""
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            frame = worker.latest()
            if frame is not None:
                if frame.error:
                    error = frame.error
                else:
                    frames += 1
                    faces += frame.observation is not None
            time.sleep(.02)
    finally:
        worker.stop_event.set()
        worker.join(timeout=3)
    print(f"Camera frames: {frames}; face/iris detections: {faces}")
    if error:
        print(error)
    return frames > 0
