# Module 4 - Behavioral detection: person tracking + business rules (press Q to quit)
import time
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)
RED_ZONE = (400, 0, 640, 480)                  # forbidden area: x1, y1, x2, y2
enter_time, last_seen = {}, time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    r = model.track(frame, persist=True, classes=[0], verbose=False)[0]   # persons with track ID
    ids = r.boxes.id.int().tolist() if r.boxes.id is not None else []
    state = "NORMAL"
    for pid, (x1, y1, x2, y2) in zip(ids, r.boxes.xyxy.tolist()):
        if RED_ZONE[0] < (x1 + x2) / 2 < RED_ZONE[2]:            # person center inside red zone
            seconds = time.time() - enter_time.setdefault(pid, time.time())
            state = "VIOLATION: red zone" if seconds > 3 else "WARNING: red zone"
        else:
            enter_time.pop(pid, None)
    if ids:
        last_seen = time.time()                                  # someone is at the workstation
    if time.time() - last_seen > 30:
        state = "VIOLATION: left workstation"
    out = r.plot()
    cv2.rectangle(out, RED_ZONE[:2], RED_ZONE[2:], (0, 0, 255), 3)
    cv2.putText(out, state, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Behavior Detection", out)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
