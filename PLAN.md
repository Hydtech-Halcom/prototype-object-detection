# Plan

PoC status vs README scope. Checked = done, unchecked = pending/skipped.

## Core PoC (done)

- [x] RTSP connect + auto-reconnect (`app/camera.py`)
- [x] Realtime YOLO person detect (`app/detector.py`, `yolov8n.pt`)
- [x] Lock monitoring regions, one per person + margin (`app/monitor.py`)
- [x] Multi-person monitoring: ON locks all persons, per-region P1..Pn check, `LEFT (k/n)` alert
- [x] Center-point inside check per frame
- [x] Leave alert: red rect + `LEFT REGION` label + log
- [x] ON/OFF toggle (OFF clears region + alert)
- [x] Desktop UI: live feed, boxes, region overlay, status, alert, log panel (`app/main.py`)
- [x] Event logging to `logs/app.log` + panel (stdlib `logging`)
- [x] `requirements.txt` minimal (opencv, ultralytics, PySide6)
- [x] `.venv` isolated env + `.gitignore`
- [x] Dockerfile + `.dockerignore` (unbuilt: no Docker daemon on dev machine)
- [x] All in-app text English

## Scope from README (partial)

- [x] RTSP streaming
- [x] Real-time YOLO inference
- [x] Person detection
- [ ] Animal detection — skipped deliberately, person-only filter for exam PoC. Add when needed (drop `cls==0` filter).
- [x] Region monitoring
- [x] Desktop application
- [x] Event logging
- [x] ON/OFF monitoring

Out of scope (not planned for PoC):

- [ ] Face recognition
- [ ] Person identification
- [ ] Multi-person tracking
- [ ] Behavior analysis
- [ ] Eye tracking / head pose / gaze / attention estimation
- [ ] Deep learning beyond YOLO
- [ ] Database / REST API / cloud deploy / user auth

## Structure (done, simplified vs original proposal)

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

Dropped from proposal: `ui.py` (merged into `main.py`), `logger.py` (stdlib),
`config.py` (top-file constants), `assets/` (empty). Restore when each earns its file.

## Future improvements

- [x] Multiple monitoring regions (one per person, CCTV)
- [x] Multi-person monitoring (no ID tracking; region OK when any person center inside)
- [ ] Object tracking (ByteTrack/SORT)
- [ ] Custom object selection
- [ ] Adjustable margins in UI (now `MARGIN` constant)
- [ ] Alarm sound config
- [ ] Screenshot on alert
- [ ] Alert history
- [ ] Video recording
- [ ] Web dashboard
- [ ] REST API
- [ ] MQTT integration
- [ ] AI focus detection / head pose / gaze / violation detection
