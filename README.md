# AI: Visual Inspection — Training Practicum

Hands-on code for the training **"Implementing an AI Vision System for Production Lines"**:
object detection and behavioral detection with Roboflow, Google Colab and YOLO, deployed
on-premise with an RTX workstation and four IP cameras.

Every exercise is one short file that runs on the **laptop webcam**. To use an IP camera
instead, change the camera source to an RTSP address, for example
`"rtsp://user:password@192.168.1.10:554/stream"`.

---

## Files

| Module | File | What it does |
|---|---|---|
| 1 | `m1_image_processing.py` | Webcam / RTSP, FPS, resize, grayscale, HSV, blur, threshold, Canny, ROI snapshot |
| 2 | `m2_collect_images.py` | Saves one image every 2 seconds into `dataset/` for Roboflow |
| 3 | `m3_train_yolo.ipynb` | Google Colab: download the Roboflow dataset, train YOLO, evaluate, download `best.pt` |
| 3 | `visual_inspection.py` | Real-time object detection on the webcam |
| 4 | `m4_behavior.py` | Person tracking + rules: red zone > 3 s, left workstation > 30 s |
| 5 | `m5_schema.sql` | The five PostgreSQL tables |
| 5 | `m5_save_events.py` | Saves detections to PostgreSQL once per second |
| 5 | `docker-compose.yml` | PostgreSQL + Grafana, restart 24/7 |
| – | `yolo11n.pt` | Pretrained YOLO model (80 everyday classes), used until you have `best.pt` |

---

## Preparation

**Participant laptop:** Intel Core i5 Gen 12, 16 GB RAM, 512 GB SSD, Windows 11 Pro
(recommended: Core i7 / Ryzen 7, 32 GB RAM, 1 TB SSD).

**Accounts:** Gmail, Google Drive, GitHub, Roboflow (Free or Pro), Docker Hub.

**Software:** Visual Studio Code, Git, Python 3.11, Docker Desktop, PostgreSQL, DBeaver,
latest Chrome or Edge.

**Python libraries:**

```
pip install ultralytics opencv-python "psycopg[binary]"
```

Each program opens a window. Press **Q** to close it.

---

## Module 1 — AI Vision Fundamentals & Image Processing (1 day)

```
python m1_image_processing.py
```

Opens four windows: Original (with FPS), HSV, Threshold and Edges.
Press **S** to save a cropped region of interest as `snapshot_roi.jpg`.

Concepts: frames, resolution and aspect ratio, color spaces (BGR, RGB, HSV, grayscale),
resize, crop, blur, thresholding, Canny edge detection, FPS, RTSP streams.

---

## Module 2 — Data Collection & Dataset Management (2 days)

### Collect images

```
python m2_collect_images.py
```

Saves one frame every 2 seconds into `dataset/` until 1,000 images (change `INTERVAL`
and `TARGET` at the top of the file). Collect in production conditions: fixed or moving
camera, all lighting situations, the real background and angle, good and defective parts.
Target: 1,000–5,000 images from all four cameras.

### Prepare the dataset in Roboflow

1. Create the workspace **ProductionAI** and an **Object Detection** project.
2. Upload the `dataset/` folder.
3. Label with boxes: **Defect, Good, Component, Helmet**.
4. Split: **Train 70% · Validation 20% · Test 10%**.
5. Augment: rotation, flip, brightness, blur, noise, crop.
6. Generate a version, export in **YOLO** format, copy the download code.

---

## Module 3 — YOLO Training & Model Optimization (1 day)

### Train in Google Colab

1. Open `m3_train_yolo.ipynb` in Google Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. In cell 3, paste your Roboflow API key, workspace and project name (from the Export page).
4. Run all cells. The last cell downloads **`best.pt`**.

Evaluation: mAP50, mAP50-95, precision, recall, IoU, false positives / false negatives.
**Target: mAP above 90%.** If it is not reached, tune epochs, image size or model size
(`yolo11n.pt` → `yolo11s.pt`), improve the data, and retrain.

### Real-time detection

```
python visual_inspection.py
```

Shows boxes, class names, confidence and an object counter. To use your own model,
copy `best.pt` into this folder and change one line:

```python
model = YOLO("best.pt")
```

Try `conf=0.25`, `0.5` and `0.8` to see the precision / recall trade-off.

---

## Module 4 — Behavioral Detection & Multi-Camera Processing (1 day)

```
python m4_behavior.py
```

Behavioral detection = YOLO person detection + tracking (stable ID) + rules.

| Rule | Condition | State |
|---|---|---|
| Area intrusion | Person center in the red zone (right side of the image) | WARNING, then VIOLATION after 3 s |
| Leaving workstation | No person in view | VIOLATION after 30 s |
| No helmet (task) | No `Helmet` box for 5 consecutive frames (needs `best.pt`) | VIOLATION |

Change `RED_ZONE = (x1, y1, x2, y2)` to move the forbidden area.

**Multi-camera:** one reader thread per camera (Camera_01–Camera_04 over RTSP), keep only
the newest frame (frame skipping), and send all frames to the GPU in one batch
(`model(list_of_frames)`).

---

## Module 5 — Database, Dashboard & Production Deployment (1 day)

### 1. Start PostgreSQL and Grafana

```
docker compose up -d
```

PostgreSQL creates the tables from `m5_schema.sql` on first start:
`camera_status`, `detection_event`, `behavior_event`, `system_alarm`, `user_login`.
`restart: always` brings the services back after a crash or reboot.

| Service | Address | Login |
|---|---|---|
| PostgreSQL | `localhost:5432`, database `production_ai` | `ai` / `ai123` |
| Grafana | http://localhost:3000 | `admin` / `admin` |

### 2. Save detections

```
python m5_save_events.py
```

Check the rows in DBeaver: `SELECT * FROM detection_event ORDER BY id DESC;`

### 3. Grafana dashboard

Add a data source: PostgreSQL, host `postgres:5432`, database `production_ai`,
user `ai`, password `ai123`, SSL disabled. Example panel:

```sql
SELECT class_name, count(*) FROM detection_event GROUP BY class_name;
```

Dashboard metrics: detection count, alarm count, violation count, camera health, GPU usage.
Optional notification service: email, Telegram or WhatsApp gateway for critical events.

### Production architecture

```
4 IP cameras (RTSP) → AI Engine (YOLO + behavior rules) → PostgreSQL → Grafana + alarms
```

In production the AI engine runs as a third Docker container on the RTX workstation.

---

## Production rollout

| Phase | Scope | Duration |
|---|---|---|
| 1 — Proof of Concept | One camera, one object model | 2–4 weeks |
| 2 — Pilot Project | Two cameras, object + behavioral detection | 4–6 weeks |
| 3 — Production Rollout | Four cameras 24/7, Grafana, database logging, alarms | 6–8 weeks |

## Final deliverables

- **AI models:** one YOLO object-detection model, one behavioral-detection model.
- **Dataset and training:** versioned Roboflow dataset with augmentation, Colab notebook,
  training script, evaluation report.
- **Production system:** RTX AI workstation, four IP cameras, PostgreSQL, Grafana, Docker.
- **Documentation:** architecture, installation guide, operation manual, maintenance
  procedure, troubleshooting guide.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Camera not found` | Close other apps using the webcam, or try `VideoCapture(1)` |
| IP camera does not open | Test the RTSP address in VLC first; check user, password and network |
| Slow detection | Use a smaller image (`imgsz=480`) or run on a machine with a GPU |
| `connection refused` in Module 5 | Start the database first: `docker compose up -d` |
