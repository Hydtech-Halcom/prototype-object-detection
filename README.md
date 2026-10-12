# Prototype Object Detection

A minimal desktop application that demonstrates real-time object detection and region monitoring using an RTSP IP Camera and YOLO.

This project is intended as a **proof of concept (PoC)** for an exam monitoring system. Instead of implementing complex behavior analysis or attention detection models, the prototype monitors whether a detected person remains inside a predefined monitoring area.

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
├── Dockerfile
├── requirements.txt
└── README.md
```

Plan + status (done vs pending): see [PLAN.md](PLAN.md).

---

# How to Run with Docker

GUI app, so the container must forward the host display. Camera URL passes via `RTSP_URL` env or CLI arg.

Build once:

```
docker build -t poc-monitor .
```

## Linux

Allow X access, then run with display socket:

```
xhost +local:docker
docker run --rm \
  -e DISPLAY=$DISPLAY \
  -e RTSP_URL="rtsp://admin:QUSVQW@192.168.1.3:554/ch1/main" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  poc-monitor
```

Override URL via arg instead of env:

```
docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  poc-monitor python -m app.main "rtsp://user:pass@host:554/path"
```

Revoke after test: `xhost -local:docker`.

Wayland: run under XWayland (`QT_QPA_PLATFORM=xcb`, default in image) or set `QT_QPA_PLATFORM=wayland` plus Wayland socket mount.

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
