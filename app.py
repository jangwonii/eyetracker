"""Windows webcam eye cursor prototype; run start.cmd to open the control panel."""
import argparse
import ctypes
import os
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from camera import CameraWorker, probe
from gaze import CursorFilter, GazeMapper

TARGETS = [(x, y) for y in (.12, .5, .88) for x in (.12, .5, .88)]


class WindowsPointer:
    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("이 프로그램은 Windows용입니다.")
        self.user32 = ctypes.windll.user32
        try:
            self.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except (AttributeError, OSError):
            self.user32.SetProcessDPIAware()
        self.width = self.user32.GetSystemMetrics(0)
        self.height = self.user32.GetSystemMetrics(1)
        self.keys = {}

    def pressed(self, key):
        down = bool(self.user32.GetAsyncKeyState(key) & 0x8000)
        previous = self.keys.get(key, False)
        self.keys[key] = down
        return down and not previous

    def move(self, xy):
        x = round(float(xy[0]) * (self.width - 1))
        y = round(float(xy[1]) * (self.height - 1))
        if not self.user32.SetCursorPos(x, y):
            raise RuntimeError("Windows 커서를 이동할 수 없습니다.")


class App:
    def __init__(self, root, pointer, camera_index):
        self.root, self.pointer = root, pointer
        self.worker = CameraWorker(camera_index)
        self.mapper = GazeMapper()
        self.filter = CursorFilter()
        self.active = False
        self.last_frame = None
        self.closed_since = None
        self.missing_since = None
        self.calibration = None
        self.calibration_index = 0
        self.samples, self.features, self.heads = [], [], []
        self.point_started = 0
        self.closing = False

        root.title("Eye Cursor · 눈으로 커서 이동")
        root.geometry("700x720")
        root.minsize(640, 680)
        root.configure(bg="#111827")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("맑은 고딕", 11), padding=10)
        style.configure("TLabel", font=("맑은 고딕", 11), background="#111827", foreground="#e5e7eb")
        container = tk.Frame(root, bg="#111827", padx=24, pady=18)
        container.pack(fill="both", expand=True)
        ttk.Label(container, text="눈으로 커서 움직이기", font=("맑은 고딕", 21, "bold")).pack(anchor="w")
        ttk.Label(container, text="웹캠 시선 추정 · 기본 모니터 · 클릭 없음").pack(anchor="w", pady=(4, 12))
        self.preview = tk.Label(container, bg="#1f2937", text="카메라 연결 중…", fg="white")
        self.preview.pack(fill="both", expand=True)
        self.status = tk.StringVar(value="카메라 연결 중입니다. 보정 전에는 커서가 움직이지 않습니다.")
        ttk.Label(container, textvariable=self.status, wraplength=640).pack(anchor="w", pady=12)
        buttons = tk.Frame(container, bg="#111827")
        buttons.pack(fill="x")
        self.calibrate_button = ttk.Button(buttons, text="1. 시선 보정 (약 30초)", command=self.start_calibration)
        self.calibrate_button.pack(side="left", padx=(0, 8))
        self.toggle_button = ttk.Button(buttons, text="2. 커서 시작 (F8)", command=self.toggle)
        self.toggle_button.pack(side="left")
        ttk.Button(buttons, text="종료", command=self.close).pack(side="right")
        self.smoothing = tk.DoubleVar(value=.15)
        ttk.Label(container, text="커서 안정화: 왼쪽은 빠르게 / 오른쪽은 부드럽게").pack(anchor="w", pady=(14, 0))
        ttk.Scale(container, from_=.05, to=.35, variable=self.smoothing).pack(fill="x")
        ttk.Label(container, text="F8 시작·일시정지  |  Esc 즉시 정지·보정 취소\n눈을 1.2초 감으면 일시정지합니다. 머리 위치가 바뀌면 다시 보정하세요.",
                  wraplength=640).pack(anchor="w", pady=(10, 0))
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.worker.start()
        root.after(20, self.tick)

    def pause(self, message="일시정지했습니다. F8로 다시 시작합니다."):
        self.active = False
        self.filter.reset()
        self.toggle_button.configure(text="2. 커서 시작 (F8)")
        self.status.set(message)

    def toggle(self):
        if self.calibration is not None:
            return
        if self.active:
            self.pause()
            return
        if self.mapper.weights is None:
            self.status.set("먼저 ‘시선 보정’을 완료해 주세요.")
            return
        frame = self.last_frame
        if frame is None or time.monotonic() - frame.timestamp > .4 or frame.observation is None:
            self.status.set("얼굴과 눈이 카메라에 보이는지 확인해 주세요.")
            return
        if frame.observation.openness < .14:
            self.status.set("눈을 뜬 상태에서 시작해 주세요.")
            return
        if not self.mapper.head_is_close(frame.observation.head):
            self.status.set("보정 때와 머리 위치가 다릅니다. 위치를 되돌리거나 다시 보정하세요.")
            return
        self.active = True
        self.closed_since = self.missing_since = None
        self.filter.reset()
        self.toggle_button.configure(text="커서 일시정지 (F8)")
        self.status.set("커서 제어 중 · Esc 또는 눈을 1.2초 감으면 정지")

    def start_calibration(self):
        if self.calibration is not None:
            return
        frame = self.last_frame
        if frame is None or time.monotonic() - frame.timestamp > .4 or frame.observation is None:
            self.status.set("카메라에 얼굴과 두 눈이 보인 뒤 보정을 시작해 주세요.")
            return
        self.pause("보정 중입니다. 머리를 고정하고 점을 바라보세요.")
        self.mapper = GazeMapper()
        self.features, self.heads, self.samples = [], [], []
        self.calibration_index = 0
        window = tk.Toplevel(self.root)
        window.title("시선 보정")
        # Explicit primary-monitor placement keeps coordinates aligned on multi-monitor systems.
        window.overrideredirect(True)
        window.geometry(f"{self.pointer.width}x{self.pointer.height}+0+0")
        window.attributes("-topmost", True)
        self.canvas = tk.Canvas(window, bg="#0b1220", highlightthickness=0, cursor="none")
        self.canvas.pack(fill="both", expand=True)
        self.calibration = window
        window.focus_force()
        self.point_started = time.monotonic() + 2
        self.draw_target(time.monotonic())

    def cancel_calibration(self, message="보정을 취소했습니다."):
        if self.calibration is not None:
            self.calibration.destroy()
            self.calibration = None
        self.pause(message)

    def draw_target(self, now):
        self.canvas.delete("all")
        w, h = self.pointer.width, self.pointer.height
        x, y = TARGETS[self.calibration_index]
        ready = now - self.point_started >= 1.1
        color = "#34d399" if ready else "#60a5fa"
        self.canvas.create_oval(x*w-23, y*h-23, x*w+23, y*h+23, outline=color, width=3)
        self.canvas.create_oval(x*w-5, y*h-5, x*w+5, y*h+5, fill=color, outline=color)
        self.canvas.create_text(w/2, h*.035, text=f"{self.calibration_index+1} / 9   머리를 고정하고 점 중심을 바라보세요.",
                                fill="white", font=("맑은 고딕", 16))
        text = "준비 중…" if not ready else f"눈 위치 수집 중 ({len(self.samples)} 프레임)"
        self.canvas.create_text(w/2, h*.965, text=text + "   ·   Esc 취소", fill="#94a3b8", font=("맑은 고딕", 13))

    def calibration_frame(self, frame, now):
        elapsed = now - self.point_started
        observation = frame.observation
        if elapsed >= 1.1 and observation is not None and observation.openness >= .14:
            self.samples.append(observation)
        elif elapsed >= 1.1:
            # Require a continuous stable segment after blinks / tracking loss.
            self.samples = []
        if elapsed >= 2.5 and len(self.samples) >= 15:
            recent = self.samples[-45:]
            self.features.append(np.median([s.features for s in recent], axis=0))
            self.heads.extend(s.head for s in recent)
            self.calibration_index += 1
            self.samples = []
            if self.calibration_index == len(TARGETS):
                try:
                    self.mapper.fit(self.features, TARGETS, self.heads)
                    self.cancel_calibration("보정 완료! F8 또는 ‘커서 시작’을 누르세요. 정확하지 않으면 다시 보정하세요.")
                except ValueError as exc:
                    self.mapper = GazeMapper()
                    self.cancel_calibration(str(exc))
                return
            self.point_started = now
        if elapsed > 12:
            self.cancel_calibration("눈을 안정적으로 추적하지 못했습니다. 조명을 확인하고 다시 보정해 주세요.")
            return
        self.draw_target(now)

    def track_frame(self, frame, now):
        observation = frame.observation
        if observation is None:
            self.filter.reset()
            self.closed_since = None
            self.missing_since = self.missing_since or now
            self.status.set("눈 추적이 끊겨 커서를 멈췄습니다.")
            if now - self.missing_since > 1:
                self.pause("얼굴을 놓쳐 일시정지했습니다. 위치를 확인한 뒤 F8로 시작하세요.")
            return
        self.missing_since = None
        if observation.openness < .14:
            self.filter.reset()
            self.closed_since = self.closed_since or now
            if now - self.closed_since >= 1.2:
                self.pause("눈 감기를 감지해 일시정지했습니다. F8로 다시 시작합니다.")
            return
        self.closed_since = None
        if not self.mapper.head_is_close(observation.head):
            self.pause("머리 위치가 달라져 정지했습니다. 원래 위치로 돌아가거나 다시 보정하세요.")
            return
        target = self.mapper.predict(observation.features)
        xy = self.filter.update(target, now, self.smoothing.get())
        self.pointer.move(xy)
        self.status.set("커서 제어 중 · Esc 또는 눈을 1.2초 감으면 정지")

    def tick(self):
        if self.closing:
            return
        try:
            self.process_tick()
        except Exception as exc:
            self.cancel_calibration(f"오류로 정지했습니다: {exc}")
        finally:
            if not self.closing:
                self.root.after(20, self.tick)

    def process_tick(self):
        now = time.monotonic()
        if self.pointer.pressed(0x1B):
            self.cancel_calibration("Esc로 정지했습니다. F8로 다시 시작합니다.")
        elif self.pointer.pressed(0x77):
            self.toggle()
        frame = self.worker.latest()
        if frame is None:
            if self.last_frame and now - self.last_frame.timestamp > .5:
                if self.active or self.calibration is not None:
                    self.cancel_calibration("카메라 응답이 늦어 정지했습니다. 카메라 연결을 확인하세요.")
            return
        first_frame = self.last_frame is None
        self.last_frame = frame
        if frame.error:
            self.cancel_calibration(frame.error)
            return
        if now - frame.timestamp > .4:
            if self.active or self.calibration is not None:
                self.cancel_calibration("영상 처리가 지연되어 정지했습니다.")
            return
        image = Image.fromarray(frame.image)
        image.thumbnail((640, 400))
        self.photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.photo, text="")
        if first_frame:
            self.status.set("카메라가 연결되었습니다. 얼굴과 두 눈을 화면에 맞추고 ‘시선 보정’을 누르세요.")
        if self.calibration is not None:
            self.calibration_frame(frame, now)
        elif self.active:
            self.track_frame(frame, now)

    def close(self):
        self.closing = True
        self.active = False
        self.worker.stop_event.set()
        self.root.destroy()
        self.worker.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description="웹캠으로 Windows 커서 이동")
    parser.add_argument("--camera", type=int, default=0, help="카메라 번호 (기본: 0)")
    parser.add_argument("--check-camera", action="store_true", help="영상 저장·커서 이동 없이 카메라 점검")
    args = parser.parse_args()
    if args.check_camera:
        raise SystemExit(0 if probe(args.camera) else 1)
    pointer = WindowsPointer()
    root = tk.Tk()
    App(root, pointer, args.camera)
    root.mainloop()


if __name__ == "__main__":
    main()
