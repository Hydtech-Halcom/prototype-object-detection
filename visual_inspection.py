# AI Visual Inspection - real-time object detection with the laptop webcam
# Install : pip install ultralytics opencv-python
# Run     : python visual_inspection.py   (press Q to quit)

import cv2
from ultralytics import YOLO

# 1. Load the YOLO model (downloaded automatically on first run)
model = YOLO("yolo11n.pt")

# 2. Open the laptop webcam (0 = built-in camera)
cap = cv2.VideoCapture(0)

while True:
    # 3. Read one frame from the camera
    ok, frame = cap.read()
    if not ok:
        print("Camera not found")
        break

    # 4. Detect objects in the frame
    results = model(frame, conf=0.5, verbose=False)

    # 5. Draw bounding boxes, class names and confidence
    output = results[0].plot()
    total = len(results[0].boxes)
    cv2.putText(output, f"Objects: {total}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 6. Show the result
    cv2.imshow("AI Visual Inspection", output)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
