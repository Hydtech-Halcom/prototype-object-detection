# Prototype Object Detection

A minimal desktop application that demonstrates real-time object detection and region monitoring using an RTSP IP Camera and YOLO.

This project is intended as a **proof of concept (PoC)** for an exam monitoring system. Instead of implementing complex behavior analysis or attention detection models, the prototype monitors whether a detected person remains inside a predefined monitoring area.

---

# Quick Start

## Jetson Orin with GPU (primary, 25-30 fps path)

Host setup, once per boot (locks Orin to max performance):

```
./scripts/jetson-power.sh
```

Run with GPU passthrough (needs the `nvidia` container runtime on the host):

```
xhost +local:docker
docker compose -f docker-compose.yml -f docker-compose.jetson.yml up --build
```

After testing, revoke X access: `xhost -local:docker`.

## Any machine, CPU only (fallback, ~5-8 fps)

```
xhost +local:docker
docker compose up --build
```

## Camera URL

Override without editing files:

```
RTSP_URL="rtsp://user:pass@host:554/path" docker compose -f docker-compose.yml -f docker-compose.jetson.yml up --build
```

Default `RTSP_URL` lives in `docker-compose.yml` (falls back to the example IP when the env var is empty).

## Useful tuning (no rebuild needed)

| Variable | Default | Effect |
|----------|---------|--------|
| `YOLO_IMGSZ` | `640` | e.g. `480` trades resolution for inference rate |
| `YOLO_HALF` | `0` | `1` enables FP16 (measured no gain at this size) |
| `YOLO_DEVICE` | auto | `cuda:0` when a GPU is visible, else `cpu` |

Example:

```
YOLO_IMGSZ=480 docker compose -f docker-compose.yml -f docker-compose.jetson.yml up --build
```

## Health

Watch the pipeline in `logs/app.log` — every 5 s it prints:

```
stats seen=<frames from camera> inferred=<YOLO runs> dropped=<skipped> infer_ms=<ms per run>
```

Healthy GPU run: `inferred` tracks `seen` 1:1, `infer_ms` ~20-60, memory flat (~1.3 GiB). If `dropped` climbs, inference is slower than the camera rate — lower `YOLO_IMGSZ` or the camera frame rate.

Note: the frame rate can never exceed what the camera sends. If `seen` grows slower than expected (e.g. ~15/s), raise the frame rate in the camera's own encoding settings.

---

# Current System

Three threads, each with one job:

```
RTSP camera
    │
    ▼
CameraWorker (capture thread) ── raw frames, TCP + low-delay FFmpeg,
    │                              1-frame buffer, auto-reconnect
    ▼
InferenceWorker (GPU/CPU thread) ─ YOLO person detect, latest-only slot:
    │                              newest frame overwrites a pending one,
    │                              GUI never blocks on inference
    ▼
MainWindow (GUI thread) ───────── renders live video + boxes + regions,
                                   ON/OFF locking, leave-region alerts,
                                   event log panel
```

* **Capture** (`app/camera.py`): `cv2.VideoCapture` with `rtsp_transport=tcp`, `nobuffer`/`low_delay`, buffer size 1, 2 s reconnect backoff.
* **Inference** (`app/detector.py` + worker in `app/main.py`): Ultralytics YOLOv8n, person class only, `imgsz` 640 default, model warmup at startup (pays GPU JIT cost once), results delivered via Qt signal.
* **Monitoring** (`app/monitor.py`, pure logic, no Qt/cv2): ON locks one region per detected person (box + margin); each frame checks person centers against regions; leaving triggers red highlight + `LEFT` label + log + alert panel.
* **Deployment**: `Dockerfile` (portable CPU image) + `docker-compose.yml` (portable) + `docker-compose.jetson.yml` (Orin overlay: `runtime: nvidia`, GPU env). `network_mode: host` so a LAN camera (`192.168.x.x`) is directly reachable. Logs go to `logs/app.log` (bind-mounted) and the UI panel.
* **Requirements**: `requirements.txt` (CPU/portable). OpenCV is the headless build — the GUI is PySide6 only, which avoids a bundled-Qt conflict. Full history of production fixes: see `CHANGELOG.md`.

---

# Project Goal

Develop a lightweight desktop application that:

- Connects to an RTSP IP Camera.
- Detects objects using YOLO.
- Focuses primarily on detecting people.
- Allows the user to lock a monitoring region.
- Alerts when the monitored person leaves that region.
- Demonstrates the core monitoring workflow without requiring additional machine learning models.

This project prioritizes simplicity, maintainability, and fast prototyping.

---

# Technology Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python |
| Desktop UI | PySide6 |
| Computer Vision | OpenCV |
| Object Detection | Ultralytics YOLO |
| Video Source | RTSP IP Camera |
| Containerization | Docker (optional for deployment) |

---

# Main Features

## 1. Real-Time Object Detection

- Connect to an RTSP camera.
- Detect objects in real time.
- Prioritize detection of:
    - person
    - common animals (supported by YOLO COCO)

---

## 2. Region Monitoring

The user can enable monitoring mode.

When monitoring starts:

1. Detect the selected person.
2. Capture the person's current position.
3. Expand the detected bounding box with configurable margins.
4. Store this expanded rectangle as a fixed monitoring region.

Example:

```
+--------------------------------------+
|                                      |
|        Monitoring Region             |
|                                      |
|      +---------------+               |
|      |    Person     |               |
|      +---------------+               |
|                                      |
+--------------------------------------+
```

---

## 3. Leave Region Alert

While monitoring is active:

- Detect the person's position every frame.
- Calculate the center point of the detected person.
- Check whether the center remains inside the monitoring region.

If the center leaves the region:

- Display an alert.
- Highlight the monitoring region in red.
- Write an event to the log.

---

## 4. Monitoring Toggle

### OFF

- Monitoring disabled.
- Monitoring region removed.
- Alerts cleared.

### ON

- Monitoring region created.
- Monitoring starts.
- Alerts become active.

---

# User Workflow

```
Launch Application
        │
        ▼
Connect RTSP Camera
        │
        ▼
YOLO detects person
        │
        ▼
User presses ON
        │
        ▼
Create Monitoring Region
        │
        ▼
Person stays inside
        │
        ▼
Normal
```

If the person exits:

```
Monitoring
      │
      ▼
Person leaves region
      │
      ▼
Alert
      │
      ▼
Continue monitoring
```

---

# Desktop UI

The desktop application should provide:

- Live camera feed
- Bounding boxes
- Monitoring region overlay
- Monitoring ON/OFF button
- Current monitoring status
- Alert notification
- Event log panel

Example layout:

```
+------------------------------------------------------+
| Live Camera                                          |
|                                                      |
|  [ Monitoring Region ]                              |
|                                                      |
+------------------------------------------------------+

Status : Monitoring ON

Alert  : Person inside region

[ ON ]      [ OFF ]
```

When the monitored person leaves:

```
Status : ALERT

Person left monitoring region.
```

---

# Detection Logic

```
RTSP Camera
      │
      ▼
OpenCV
      │
      ▼
YOLO Detection
      │
      ▼
Calculate Person Center
      │
      ▼
Inside Monitoring Region?
      │
 ┌────┴────┐
 │         │
YES       NO
 │         │
 │      Show Alert
 │      Write Log
 │
Continue
```

---

# Scope

Core scope done. Full checklist with done/pending status: see [PLAN.md](PLAN.md).

Included in PoC: RTSP streaming, realtime YOLO person detection,
region monitoring, desktop app, event logging, ON/OFF monitoring.

---

# Design Principles

- Keep the architecture minimal.
- Avoid unnecessary dependencies.
- Avoid overengineering.
- Prefer readability over optimization.
- Focus on demonstrating the monitoring concept.
- Build components that can be extended later.

---

# Project Structure

```
prototype-object-detection/
│
├── app/
│   ├── main.py
│   ├── camera.py
│   ├── detector.py
│   └── monitor.py
│
├── logs/
│
├── scripts/
│   └── jetson-power.sh   # host: lock Orin to max performance
│
├── Dockerfile
├── docker-compose.yml          # portable (CPU)
├── docker-compose.jetson.yml   # Orin overlay (GPU passthrough)
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

Plan + status (done vs pending): see [PLAN.md](PLAN.md).

---

# How to Run

See [Quick Start](#quick-start) at the top for the recommended commands
(GPU on Jetson Orin, or CPU fallback).

## Linux (compose, recommended)

Just one command:

```
xhost +local:docker
docker compose up --build
```

Override the camera URL without editing files:

```
RTSP_URL="rtsp://user:pass@host:554/path" docker compose up --build
```

After testing, revoke X access: `xhost -local:docker`.

Notes:

- Default `RTSP_URL` lives in `docker-compose.yml` (falls back to the example IP when the env var is empty).
- `network_mode: host` so a LAN camera (`192.168.x.x`) is directly reachable, no network setup needed.
- Wayland: works on top of XWayland as-is (`QT_QPA_PLATFORM=xcb` is already the default in the image).

## Jetson Orin with GPU

Same as above but with the GPU overlay (see [Quick Start](#quick-start)):

```
./scripts/jetson-power.sh   # host, once per boot
xhost +local:docker
docker compose -f docker-compose.yml -f docker-compose.jetson.yml up --build
```

Requires the `nvidia` container runtime on the host. Check GPU visibility with:

```
docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all poc-monitor python -c "import torch; print(torch.cuda.is_available())"
```

## Manual `docker run` (without compose)

Build once: `docker build -t poc-monitor .`, then:

```
xhost +local:docker
docker run --rm \
  -e DISPLAY=$DISPLAY \
  -e RTSP_URL="rtsp://admin:QUSVQW@192.168.1.3:554/ch1/main" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  poc-monitor
```

With GPU on Jetson, add `--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all`.

Override URL via arg: `docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix poc-monitor python -m app.main "rtsp://user:pass@host:554/path"`.

## Windows

Docker Desktop cannot show Qt directly. Steps:

1. Install VcXsrv, start with "Disable access control".
2. Note host IP reachable from container (e.g. `192.168.1.2`).
3. Run:

```
docker build -t poc-monitor .
docker run --rm -e DISPLAY=192.168.1.2:0.0 -e LIBGL_ALWAYS_INDIRECT=1 -e RTSP_URL="rtsp://admin:QUSVQW@192.168.1.3:554/ch1/main" poc-monitor
```

Notes:

- Camera must be reachable from container network. `192.168.x.x` works when Docker uses bridge/NAT on same LAN. No `localhost` camera URLs.
- Headless test (no display) via YOLO CLI:

```
docker run --rm poc-monitor yolo predict model=yolov8n.pt source="rtsp://admin:QUSVQW@192.168.1.3:554/ch1/main"
```

---

See [PLAN.md](PLAN.md) for future improvements + status.
