# Module 1 - Basic image processing with the webcam (press S = snapshot, Q = quit)
import time
import cv2

SOURCE = 0   # 0 = laptop webcam, or an IP camera: "rtsp://user:pass@192.168.1.10:554/stream"
cap = cv2.VideoCapture(SOURCE)
prev = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    fps = 1 / (time.time() - prev); prev = time.time()           # frames per second
    small = cv2.resize(frame, (640, 360))                        # resize
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)               # grayscale
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)                 # HSV color space
    blur = cv2.GaussianBlur(gray, (5, 5), 0)                     # blur removes noise
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY) # threshold: black / white
    edges = cv2.Canny(blur, 50, 150)                             # Canny edge detection
    cv2.putText(small, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    for name, img in [("Original", small), ("HSV", hsv), ("Threshold", thresh), ("Edges", edges)]:
        cv2.imshow(name, img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        cv2.imwrite("snapshot_roi.jpg", small[90:270, 160:480])  # crop region of interest
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
