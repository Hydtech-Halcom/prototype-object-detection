"""Minimal YOLO wrapper. Person filter (cls 0) only."""

PERSON_CLS = 0


class Detector:
    def __init__(self, model="yolov8n.pt", conf=0.4):
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.conf = conf

    def detect(self, frame):
        """Return list (x1, y1, x2, y2, conf). Empty when no person."""
        res = self.model.predict(frame, conf=self.conf, verbose=False)[0]
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
