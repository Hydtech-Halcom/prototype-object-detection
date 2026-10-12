"""RTSP capture in QThread. Emits raw frames, simple reconnect."""

import os

import cv2
from PySide6.QtCore import QThread, Signal

# Low-latency FFmpeg RTSP options. Set before VideoCapture opens the stream.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
)


class CameraWorker(QThread):
    frame_ready = Signal(object)  # np.ndarray BGR
    status = Signal(str)

    def __init__(self, url, parent=None):
        super().__init__(parent)
        self.url = url
        self._run = True

    def stop(self):
        self._run = False
        self.wait(2000)

    def run(self):
        # ponytail: plain reconnect loop, fixed 2s backoff. Raise if needed.
        while self._run:
            self.status.emit(f"connect {self.url}")
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.status.emit("connection failed, retry in 2s")
                self.msleep(2000)
                continue
            self.status.emit("connected")
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            while self._run:
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.status.emit("frame lost, reconnecting")
                    break
                self.frame_ready.emit(frame)
                self.msleep(33)
            cap.release()
            if self._run:
                self.msleep(2000)
