"""Entry + PySide6 UI. Single capture thread, inference in GUI slot."""

import logging
import os
import sys
from pathlib import Path

import cv2
from PySide6.QtCore import Qt
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("poc")


class MainWindow(QMainWindow):
    def __init__(self, rtsp_url=RTSP_URL):
        super().__init__()
        self.setWindowTitle("PoC Exam Monitor")
        self.resize(960, 700)

        self.monitor = RegionMonitor(margin=MARGIN)
        self.detector = None  # lazy, so UI shows fast
        self.last_boxes = []
        self.frame_size = (0, 0)

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

    def add_log(self, msg):
        log.info(msg)
        self.log_panel.append(msg)

    def ensure_detector(self):
        if self.detector is None:
            self.add_log("loading YOLOv8n...")
            self.detector = Detector()
            self.add_log("YOLO ready")
        return self.detector

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
        h, w = frame.shape[:2]
        self.frame_size = (w, h)
        try:
            # ponytail: inference in GUI thread. Move to worker if laggy.
            det = self.ensure_detector()
            boxes = det.detect(frame)
        except Exception as e:  # model failed -> show raw frame, never crash
            boxes = []
            if not hasattr(self, "_det_warned"):
                self._det_warned = True
                self.add_log(f"detector error: {e}")
        self.last_boxes = boxes

        alert = False
        for x1, y1, x2, y2, _c in boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        if self.monitor.active:
            inside_list = self.monitor.check_all(boxes)
            left = sum(1 for v in inside_list if not v)
            n = len(inside_list)
            for i, (rect, inside) in enumerate(zip(self.monitor.regions, inside_list)):
                rx1, ry1, rx2, ry2 = rect
                color = (0, 255, 0) if inside else (0, 0, 255)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 2)
                cv2.putText(frame, f"P{i + 1}", (rx1, max(0, ry1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if not inside:
                    cv2.putText(frame, "LEFT", (rx1, ry2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            alert = left > 0
            if alert and left != getattr(self, "_last_left", 0):
                self.add_log(f"ALERT: {left}/{n} person(s) left region")
                log.warning("ALERT %d/%d left", left, n)
            self._last_left = left
            self.alert_lbl.setText(f"Alert: LEFT ({left}/{n})" if alert else "Alert: inside")
            self.alert_lbl.setStyleSheet(
                "font-weight: bold; color: red;" if alert else "font-weight: bold; color: green;"
            )
            self.status_lbl.setText(
                f"Status: ALERT ({left}/{n})" if alert else f"Status: Monitoring ON ({n})")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.video.setPixmap(QPixmap.fromImage(img).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, e):
        self.worker.stop()
        super().closeEvent(e)


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else os.getenv("RTSP_URL", RTSP_URL)
    app = QApplication(sys.argv)
    win = MainWindow(url)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
