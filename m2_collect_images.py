# Module 2 - Collect dataset images: save one frame every 2 seconds (press Q to stop)
import os
import time
import cv2

SOURCE = 0          # webcam or "rtsp://..."
INTERVAL = 2        # seconds between images
TARGET = 1000       # number of images

os.makedirs("dataset", exist_ok=True)
cap = cv2.VideoCapture(SOURCE)
count, last = 0, 0

while count < TARGET:
    ok, frame = cap.read()
    if not ok:
        break
    if time.time() - last >= INTERVAL:
        count += 1
        last = time.time()
        cv2.imwrite(f"dataset/img_{count:04d}.jpg", frame)
        print("Saved", count)
    cv2.imshow("Collecting - press Q to stop", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
