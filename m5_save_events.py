# Module 5 - Save detections to PostgreSQL once per second (press Q to quit)
# Install: pip install "psycopg[binary]"   Start the database: docker compose up -d
import time
import cv2
import psycopg
from ultralytics import YOLO

db = psycopg.connect("postgresql://ai:ai123@localhost:5432/production_ai", autocommit=True)
model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)
last = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    result = model(frame, conf=0.5, verbose=False)[0]
    if time.time() - last >= 1:                                  # save once per second
        last = time.time()
        for box in result.boxes:
            db.execute("INSERT INTO detection_event (camera_id, class_name, confidence) VALUES (%s, %s, %s)",
                       ("Camera_01", model.names[int(box.cls)], float(box.conf)))
    cv2.imshow("Saving to PostgreSQL", result.plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
