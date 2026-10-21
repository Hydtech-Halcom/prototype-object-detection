"""Minimal YOLO wrapper. Person filter (cls 0) only."""

import os
import sys
from pathlib import Path

PERSON_CLS = 0
DEFAULT_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")


def resolve_model(name=None):
    """Frozen-aware model path. Explicit name always wins. Default (None) on
    win32 prefers sibling .onnx (DML GPU when present, CPU otherwise, no
    torch-CUDA bloat); elsewhere prefers .pt (Jetson/docker path unchanged)."""
    auto = name is None
    name = name or DEFAULT_MODEL
    cands = []
    if getattr(sys, "frozen", False):
        cands += [Path(sys.executable).parent / name, Path(sys._MEIPASS) / name]  # noqa: SLF001
    cands += [Path(name), Path(__file__).resolve().parent.parent / name]
    if auto and os.name == "nt" and not str(name).endswith(".onnx"):
        onnx = Path(name).with_suffix(".onnx").name
        for p in cands:
            q = p.parent / onnx
            if q.is_file():
                return str(q)
    for p in cands:
        if p.is_file():
            return str(p)
    return name  # let ultralytics download/cache (dev loop)


def _try_enable_dml(yolo, weight):
    """Swap the ORT session to DmlExecutionProvider. Returns True when the
    session now runs on DML. CPU fallback on any failure (never crash)."""
    try:
        import onnxruntime as ort

        if "DmlExecutionProvider" not in ort.get_available_providers():
            return False
        bk = yolo.predictor.model.backend  # ONNXBackend; .pt has none
        sess = bk.session
        opts = sess.get_session_options()
        dml = ort.InferenceSession(str(weight), opts, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
        dml.run  # touch
        bk.session = dml
        try:
            yolo.predictor.model.session = dml
        except Exception:
            pass
        return "DmlExecutionProvider" in dml.get_providers()
    except Exception:
        return False


class Detector:
    def __init__(self, model=None, conf=0.4, imgsz=640, device=None, half=False):
        import torch
        from ultralytics import YOLO

        model = resolve_model(model)
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.half = bool(half) and device != "cpu"
        self.model = YOLO(model)
        self.backend = "pt" if str(model).endswith(".pt") else "onnx-cpu"
        self._weight = model
        self.conf = conf
        self.imgsz = imgsz

    def enable_dml(self):
        """Swap ORT session to DML. Call once after warmup (predictor exists).
        Returns True when DML active; CPU fallback otherwise. Never raises."""
        if not str(self._weight).endswith(".onnx"):
            return False
        if _try_enable_dml(self.model, self._weight):
            self.backend = "onnx-dml"
            return True
        return False

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


def quick_bench(det, n=3):
    """Median-ish infer ms on a blank frame. Used once at startup for tiering."""
    import time

    import numpy as np

    blank = np.zeros((det.imgsz, det.imgsz, 3), dtype=np.uint8)
    ms = []
    for _ in range(n):
        t0 = time.monotonic()
        det.detect(blank)
        ms.append((time.monotonic() - t0) * 1000)
    ms.sort()
    return ms[len(ms) // 2]
