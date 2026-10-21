"""Entry + PySide6 UI. Capture thread, inference worker thread, GUI renders."""

import logging
import os
import sys
import time
from pathlib import Path

if getattr(sys, "frozen", False):
    # ponytail: frozen exe has no pip metadata; ultralytics would treat deps
    # as missing and relaunch [exe, "-m", "pip", ...] -> fork bomb. Skip it:
    # everything is already bundled by PyInstaller.
    os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")

import cv2
from PySide6.QtCore import (
    Qt,
    QMutex,
    QMutexLocker,
    QThread,
    QWaitCondition,
    Signal,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.camera import CameraWorker
from app.detector import Detector
from app.monitor import RegionMonitor

RTSP_URL = "rtsp://admin:QUSVQW@192.168.1.3:554/ch1/main"
MARGIN = 40
# Inference runs in InferenceWorker (below), not the GUI thread. Render stays
# in on_frame so video is live even while the worker is busy.
# YOLO_IMGSZ set -> manual (wins). Unset -> auto tier from startup bench.
_IMGSZ_ENV = os.getenv("YOLO_IMGSZ")
IMGSZ = int(_IMGSZ_ENV) if _IMGSZ_ENV else 0  # 0 = auto
TORCH_THREADS = 2
YOLO_DEVICE = os.getenv("YOLO_DEVICE")  # None = auto (cuda:0 when available)
YOLO_HALF = os.getenv("YOLO_HALF", "0") == "1"


def pick_tier(bench_ms):
    """Map startup bench to (imgsz, stride). Stride N = infer every Nth frame."""
    if bench_ms < 50:
        return ("high", 640, 1)
    if bench_ms < 150:
        return ("med", 480, 1)
    return ("low", 320, 2)  # ponytail: stride, not motion-gate (sitter is static)

try:
    import torch

    torch.set_num_threads(TORCH_THREADS)
except Exception:
    pass

# ponytail: exe/dist has no logs/ dir; cwd may differ when launched via
# shortcut, so anchor logs to the exe dir when frozen, repo root otherwise.
_BASE = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path.cwd()
(_BASE / "logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(_BASE / "logs" / "app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("poc")


class InferenceWorker(QThread):
    """YOLO inference off the GUI thread. Latest-only slot: submit() never
    blocks; a newer frame overwrites a still-pending one (counted as dropped)."""

    result_ready = Signal(list, float)  # boxes, infer_ms
    status = Signal(str)
    tier_ready = Signal(str, int, int)  # tier, imgsz, stride

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        self._latest = None
        self._run = True
        self._warned = False
        self.dropped = 0
        self.inferred = 0

    def try_submit(self, frame):
        with QMutexLocker(self._mutex):
            if self._latest is not None:
                self.dropped += 1
            self._latest = frame
            self._cond.wakeOne()

    def stop(self):
        with QMutexLocker(self._mutex):
            self._run = False
            self._cond.wakeOne()
        self.wait(5000)

    def run(self):
        from app.detector import quick_bench

        self.status.emit("loading model...")
        t0 = time.monotonic()
        try:
            det = Detector(device=YOLO_DEVICE, half=YOLO_HALF, imgsz=IMGSZ or 480)
        except Exception as e:
            self.status.emit(f"model load failed: {e}")
            log.exception("model load failed")
            return
        log.info("model loaded in %.1fs", time.monotonic() - t0)
        self.status.emit("warming up...")
        t0 = time.monotonic()
        try:
            det.warmup()
        except Exception as e:
            # ponytail: never die silently (frozen NMS/thread issues hide here)
            self.status.emit(f"warmup failed: {e}")
            log.exception("warmup failed")
            return
        log.info("warmup done in %.1fs", time.monotonic() - t0)
        det.enable_dml()  # predictor exists now; no-op for .pt/CPU-only
        if IMGSZ:
            tier, imgsz, stride = "manual", IMGSZ, 1
        else:
            ms = quick_bench(det)  # at 480 probe size
            tier, imgsz, stride = pick_tier(ms)
            det.imgsz = imgsz
        self.tier_ready.emit(tier, det.imgsz, stride)
        self.status.emit(f"YOLO {det.backend} on {det.device} ({tier}, imgsz={det.imgsz}, stride={stride})")
        self.status.emit("YOLO ready")
        while True:
            self._mutex.lock()
            try:
                while self._latest is None and self._run:
                    self._cond.wait(self._mutex)
                if not self._run:
                    return
                frame = self._latest
                self._latest = None
            finally:
                self._mutex.unlock()
            t0 = time.monotonic()
            try:
                boxes = det.detect(frame)
            except Exception as e:  # model failed -> empty, never crash worker
                boxes = []
                if not self._warned:
                    self._warned = True
                    self.status.emit(f"detector error: {e}")
            infer_ms = (time.monotonic() - t0) * 1000
            self.inferred += 1
            self.result_ready.emit(boxes, infer_ms)


class MainWindow(QMainWindow):
    def __init__(self, rtsp_url=RTSP_URL):
        super().__init__()
        self.setWindowTitle("PoC Exam Monitor")
        self.resize(960, 700)

        self.monitor = RegionMonitor(margin=MARGIN)
        self.last_boxes = []
        self.frame_size = (0, 0)
        # Stats state (worker owns inference counters)
        self._frames_seen = 0
        self._infer_ms = 0.0
        self._last_stats_t = time.monotonic()

        self.video = QLabel("connecting...")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(640, 360)
        self.video.setStyleSheet("background: black; color: white;")

        self.status_lbl = QLabel("Status: OFF")
        self.alert_lbl = QLabel("Alert: -")
        self.alert_lbl.setStyleSheet("font-weight: bold;")

        self.btn_on = QPushButton("ON")
        self.btn_off = QPushButton("OFF")
        self.btn_on.clicked.connect(self.on_start)
        self.btn_off.clicked.connect(self.on_stop)

        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setMaximumHeight(150)

        top = QHBoxLayout()
        top.addWidget(self.btn_on)
        top.addWidget(self.btn_off)
        top.addWidget(self.status_lbl)
        top.addWidget(self.alert_lbl)

        lay = QVBoxLayout()
        lay.addWidget(self.video)
        lay.addLayout(top)
        lay.addWidget(self.log_panel)
        w = QWidget()
        w.setLayout(lay)
        self.setCentralWidget(w)

        self.worker = CameraWorker(rtsp_url)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status.connect(self.add_log)
        self.worker.start()

        # Inference in background thread from the start; UI shows immediately.
        self.stride = 1
        self._tier_seen = False
        self._up_windows = 0
        self._down_windows = 0
        self._last_dropped = 0
        self._last_inferred = 0
        self.infer_worker = InferenceWorker()
        self.infer_worker.result_ready.connect(self._on_result)
        self.infer_worker.status.connect(self.add_log)
        self.infer_worker.tier_ready.connect(self._on_tier)
        self.infer_worker.start()

    def _on_tier(self, tier, imgsz, stride):
        self.stride = stride
        self._tier_seen = True
        self.add_log(f"tier: {tier} (imgsz={imgsz}, stride={stride})")

    def add_log(self, msg):
        log.info(msg)
        self.log_panel.append(msg)

    def on_start(self):
        if not self.last_boxes:
            self.add_log("ON failed: no person in frame")
            self.status_lbl.setText("Status: OFF (no person)")
            return
        fw, fh = self.frame_size
        rects = self.monitor.lock_all(self.last_boxes, fw, fh)
        n = len(rects)
        self._last_left = 0
        self.status_lbl.setText(f"Status: Monitoring ON ({n})")
        self.alert_lbl.setText("Alert: inside")
        self.add_log(f"locked {n} region(s)")
        log.info("LOCK %s", rects)

    def on_stop(self):
        self.monitor.clear()
        self.status_lbl.setText("Status: OFF")
        self.alert_lbl.setText("Alert: -")
        self.add_log("monitoring OFF")

    def on_frame(self, frame):
        """GUI thread: render fast with last boxes, hand newest frame to worker."""
        h, w = frame.shape[:2]
        self.frame_size = (w, h)
        self._frames_seen += 1
        now = time.monotonic()

        if self._frames_seen % self.stride == 0:
            self.infer_worker.try_submit(frame)
        self._render(frame, self.last_boxes)

        # Periodic stats so slowdowns are diagnosable from logs/app.log
        if now - self._last_stats_t >= 5.0:
            log.info(
                "stats seen=%d inferred=%d dropped=%d infer_ms=%.0f stride=%d",
                self._frames_seen,
                self.infer_worker.inferred,
                self.infer_worker.dropped,
                self._infer_ms,
                self.stride,
            )
            self._adapt_stride()
            self._last_stats_t = now

    def _adapt_stride(self):
        """Hysteresis on drops/window: 1 bad window steps down (max 3),
        3 clean windows step back up. ponytail: stride only, never reload."""
        if not self._tier_seen:  # worker still in warmup; nothing to judge yet
            return
        dropped = self.infer_worker.dropped - self._last_dropped
        inferred = self.infer_worker.inferred - self._last_inferred
        self._last_dropped = self.infer_worker.dropped
        self._last_inferred = self.infer_worker.inferred
        if dropped > max(1, inferred):  # more overwritten than processed
            self._down_windows += 1
            self._up_windows = 0
        elif inferred > 0:
            self._up_windows += 1
            self._down_windows = 0
        else:
            return
        if self._down_windows >= 1 and self.stride < 3:
            self.stride += 1
            self._down_windows = 0
            log.info("stride -> %d (shedding load)", self.stride)
        elif self._up_windows >= 3 and self.stride > 1:
            self.stride -= 1
            self._up_windows = 0
            log.info("stride -> %d (headroom back)", self.stride)

    def _on_result(self, boxes, infer_ms):
        """Runs in GUI thread when the worker finishes one inference."""
        self.last_boxes = boxes
        self._infer_ms = infer_ms

        if not self.monitor.active:
            return
        inside_list = self.monitor.check_all(boxes)
        left = sum(1 for v in inside_list if not v)
        n = len(inside_list)
        if left > 0 and left != getattr(self, "_last_left", 0):
            self.add_log(f"ALERT: {left}/{n} person(s) left region")
            log.warning("ALERT %d/%d left", left, n)
        self._last_left = left
        self.alert_lbl.setText(f"Alert: LEFT ({left}/{n})" if left else "Alert: inside")
        self.alert_lbl.setStyleSheet(
            "font-weight: bold; color: red;" if left else "font-weight: bold; color: green;"
        )
        self.status_lbl.setText(
            f"Status: ALERT ({left}/{n})" if left else f"Status: Monitoring ON ({n})")

    def _render(self, frame, boxes):
        h, w = frame.shape[:2]
        for x1, y1, x2, y2, _c in boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        if self.monitor.active:
            inside_list = self.monitor.check_all(boxes)
            for i, (rect, inside) in enumerate(zip(self.monitor.regions, inside_list)):
                rx1, ry1, rx2, ry2 = rect
                color = (0, 255, 0) if inside else (0, 0, 255)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 2)
                cv2.putText(frame, f"P{i + 1}", (rx1, max(0, ry1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if not inside:
                    cv2.putText(frame, "LEFT", (rx1, ry2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.video.setPixmap(QPixmap.fromImage(img).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    def closeEvent(self, e):
        self.worker.stop()
        self.infer_worker.stop()
        super().closeEvent(e)


def _cli_url():
    """argv[1] wins unless it looks like a flag (stray spawn) -> env/default."""
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        return sys.argv[1]
    return os.getenv("RTSP_URL", RTSP_URL)


def main():
    import multiprocessing

    multiprocessing.freeze_support()  # no-op except frozen multiprocessing
    from PySide6.QtCore import QLockFile

    lock = QLockFile(str(_BASE / "ExamMonitor.lock"))
    lock.setStaleLockTime(0)
    if not lock.tryLock(100):
        print("ExamMonitor already running.")
        return
    url = _cli_url()
    app = QApplication(sys.argv)
    win = MainWindow(url)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
