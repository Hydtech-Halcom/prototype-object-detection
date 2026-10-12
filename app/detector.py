"""Minimal YOLO wrapper. Person filter (cls 0) only."""

PERSON_CLS = 0


class Detector:
    def __init__(self, model="yolov8n.pt", conf=0.4, imgsz=640, device=None, half=False):
        import torch
        from ultralytics import YOLO

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.half = bool(half) and device != "cpu"
        self.model = YOLO(model)
        self.conf = conf
        self.imgsz = imgsz

    def warmup(self, n=3):
        """Pay PTX-JIT/first-inference cost once at startup, not mid-stream."""
        import numpy as np

        blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(n):
            self.detect(blank)

    def detect(self, frame):
        """Return list (x1, y1, x2, y2, conf). Empty when no person."""
        kwargs = {}
        if self.half:
            kwargs["half"] = True  # passed only when on: ultralytics
            # warns 'half is deprecated' whenever the arg is present
        res = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz,
            device=self.device, verbose=False, **kwargs,
        )[0]
        out = []
        for b in res.boxes:
            if int(b.cls[0]) != PERSON_CLS:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            out.append((x1, y1, x2, y2, float(b.conf[0])))
        return out

    @staticmethod
    def biggest(boxes):
        """Largest person box (assumes exam subject is closest to camera)."""
        if not boxes:
            return None
        return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
