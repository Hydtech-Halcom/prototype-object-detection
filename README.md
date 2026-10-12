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

Included:

- RTSP streaming
- Real-time YOLO inference
- Person detection
- Animal detection
- Region monitoring
- Desktop application
- Event logging
- ON/OFF monitoring

Not included:

- Face recognition
- Person identification
- Multi-person tracking
- Behavior analysis
- Eye tracking
- Head pose estimation
- Gaze estimation
- Attention estimation
- Deep learning beyond YOLO
- Database
- REST API
- Cloud deployment
- User authentication

---

# Design Principles

- Keep the architecture minimal.
- Avoid unnecessary dependencies.
- Avoid overengineering.
- Prefer readability over optimization.
- Focus on demonstrating the monitoring concept.
- Build components that can be extended later.

---

# Proposed Project Structure

```
prototype-object-detection/
│
├── app/
│   ├── main.py
│   ├── ui.py
│   ├── camera.py
│   ├── detector.py
│   ├── monitor.py
│   ├── logger.py
│   └── config.py
│
├── assets/
│
├── logs/
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# Future Improvements

Potential future enhancements include:

- Multiple monitoring regions
- Multi-person monitoring
- Object tracking
- Custom object selection
- Adjustable monitoring margins
- Alarm sound configuration
- Screenshot on alert
- Alert history
- Video recording
- Web dashboard
- REST API
- MQTT integration
- AI-based focus detection
- Head pose estimation
- Eye gaze estimation
- Exam violation detection
