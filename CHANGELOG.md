# Changelog

## Unreleased - Fix frozen fork-bomb (exe opened dozens of windows)

* Root cause: ultralytics `check_requirements` saw no pip metadata in the
  bundle and ran `[sys.executable, "-m", "pip", ...]` — in the exe that is
  `ExamMonitor.exe` itself, so every child opened a new app window with URL
  `-m` and spawned again (49 processes, `connect -m` in log).
* `app/main.py`: `ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1` when frozen,
  `argv[1]` starting with `-` ignored, `freeze_support()` + `QLockFile`
  single-instance guard. Verified: 1 process, zero `connect -m`.
* Root cause of `inferred=0` in exe was NOT a hang: missing `torchvision`
  native ops in the bundle (`torchvision::nms does not exist`) killed the
  worker thread silently after `warming up...`. Fix: `--collect-all
  torchvision` in `scripts/build_exe.ps1` + try/except around load/warmup
  with status errors. Verified frozen: `warmup done`, `tier: high
  (640)`, `YOLO onnx-dml`, `infer_ms ~50`. Bisect probes removed.

## Unreleased - Adaptive Windows build (1 exe, light to high-end)

* `yolov8n.pt` -> `yolo11n.pt` default (same API, ~30% faster CPU, +2 mAP).
  `.onnx` export runs on DirectML GPU when present else CPU (no torch-CUDA
  bloat); explicit `YOLO_MODEL`/`.pt` still wins (Jetson/docker unchanged).
* `app/main.py`: startup bench auto-tier (high 640 / med 480 / low 320 +
  stride 2-3 with hysteresis on `dropped`); `YOLO_IMGSZ` set pins manual.
  No motion-gate (exam sitter is static — stride only).
* `scripts/build_exe.ps1` (new): PyInstaller `--onedir` + headless cv2 +
  bundled `yolo11n.onnx`. Verified headless: `tier: med (imgsz=480,
  stride=1)`, `YOLO onnx-dml on cpu` on Ryzen 5 5500U.

## Unreleased - GPU on Orin + worker-thread inference (25-30fps path)

### Findings (measured, not assumed)

* Host is Jetson Orin, JetPack 7.2.1, MAXN already set. Image torch
  (`2.14.1+cu130`) is a generic build whose arch list excludes `sm_87`
  (warns, PTX-JITs) — yet computes fine: 4k-matmul OK on host.
* `docker compose` had no GPU access at all (`cuda_available=False`).
  `--gpus all` is rejected by this daemon; CDI `gpu=all` fails on missing
  `/dev/dri/card0`. Working path: `runtime: nvidia` +
  `NVIDIA_VISIBLE_DEVICES/NVIDIA_DRIVER_CAPABILITIES=all`.
* With passthrough, existing image does yolov8n/640 in **~18 ms (56 fps)**
  on zeros. No Jetson-base rebuild needed — generic torch + runtime is
  enough, so `Dockerfile.jetson` was deliberately NOT built.

### Changes

* `app/detector.py`: `device` (auto `cuda:0` when available), `half`
  (FP16, default off — measured no gain at this size), `warmup(n=3)` for
  one-time JIT cost, `imgsz` param.
* `app/main.py`: new `InferenceWorker(QThread)` with latest-only slot —
  `try_submit()` never blocks the GUI, newer frames overwrite pending ones.
  `on_frame` renders fast (`Qt.FastTransformation`) with last boxes;
  `_on_result` updates boxes/monitor/alerts. `ensure_detector` removed
  (worker owns model lifecycle, loads in background). `YOLO_DEVICE`,
  `YOLO_HALF`, `YOLO_IMGSZ` env-tunable.
* `docker-compose.jetson.yml` (new overlay, base file stays CPU-portable):
  `runtime: nvidia`, NVIDIA env, `YOLO_HALF: 0`. Run:
  `docker compose -f docker-compose.yml -f docker-compose.jetson.yml up --build`
* `scripts/jetson-power.sh` (new, host-side): `nvpmodel -m 0` +
  `jetson_clocks`. Re-run after reboot.

### Measured end-to-end (live RTSP main-stream, offscreen)

* Before (CPU, GUI-thread inference): frozen GUI, 381% CPU, 12.3 GiB.
* After P0: alive, 8 fps inferred, 120% CPU, 1.2 GiB.
* After P1 worker: **15.2 fps seen / 15.2 fps inferred, 0 steady drops,
  infer 29-49 ms, ~100% CPU, 1.3 GiB flat.** Pipeline tracks source 1:1.
* Remaining gap to 25-30 fps is the SOURCE: camera main-stream delivers
  ~15 fps. Raise it in the camera's encoding settings; pipeline headroom
  (56 fps micro-benchmark) already covers 30 fps. `YOLO_IMGSZ=480` is
  available if real-content inference needs trimming at 30 fps.

## Unreleased - Fix GUI freeze after camera connects (P0 throttle)

### Problem

App connected to RTSP and loaded YOLO, then the GUI stopped responding.
`docker stats` showed `~380% CPU, 12+ GiB MEM` climbing; `logs/app.log`
stopped after `YOLO ready`. Root cause: `CameraWorker` emits up to ~30-60
fps with no queue limit while `MainWindow.on_frame` ran YOLO `predict` in
the GUI thread per frame. Producer >> consumer -> queued `ndarray` frames
ballooned memory and starved paint/click events.

### Fix

* `app/camera.py`: `CAP_PROP_BUFFERSIZE=1`, FFmpeg low-delay options
  (`rtsp_transport=tcp`, `nobuffer`, `low_delay`), emit interval 15ms -> 33ms.
* `app/main.py`: re-entrancy guard + `INFER_EVERY_MS=200` (~5 fps) decimation.
  Frames arriving while busy/too-soon reuse `last_boxes` for drawing so video
  stays live; `torch.set_num_threads(2)` caps CPU contention.
* `app/detector.py`: `predict(..., imgsz=640)` to bound per-frame cost.
* `Dockerfile`: `YOLO_CONFIG_DIR=/tmp/Ultralytics` (silences read-only
  config warning), `PIP_ROOT_USER_ACTION=ignore` (root pip is standard in
  containers; a venv adds no isolation inside Docker).
* `Dockerfile`: explicit `pip uninstall opencv-python` + reinstall headless
  layer. Needed because Docker layer caching kept the old `opencv-python`
  (with its conflicting `cv2/qt/plugins`) installed alongside headless.
* Observability: `stats seen/inferred/dropped/infer_ms` log line every 5s.

Verified with offscreen smoke test (stub detector, 0.25s/infer): 20 frames
in 0.29s, 1 inferred / 19 dropped, GUI responsive (`THROTTLE_OK`).

Rebuild and check `docker stats` stays flat plus periodic `stats` lines in
`logs/app.log`. For full-res main-stream still too slow on CPU, point
`RTSP_URL` at the camera sub-stream.

## Unreleased - Fix `libEGL.so.1` ImportError in Docker image

### Problem

`docker compose up --build` built successfully but the container crashed on start:

```text
File "/app/app/main.py", line 10, in <module>
    from PySide6.QtGui import QImage, QPixmap
ImportError: libEGL.so.1: cannot open shared object file: No such file or directory
```

Root cause: `python:3.13-slim` is minimal. The `Dockerfile` only installed
`libgl1` plus a few XCB/X11 libs, which is not enough for PySide6 >= 6.7 / Qt6.
QtGui needs `libEGL.so.1` (`libegl1`), `libOpenGL.so.0` (`libopengl0`), and for
the `xcb` platform plugin also `libxkbcommon` and `libxcb-cursor`.

### Fix (`Dockerfile`)

Extended the `apt-get install` layer (Option A - full Qt6 set):

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libegl1 libopengl0 libglib2.0-0t64 \
    libxcb1 libx11-6 libxext6 libxrender1 libxi6 libxtst6 \
    libxcb-cursor0 libxkbcommon0 libdbus-1-3 libfontconfig1 \
    libxcb-image0 libxcb-render-util0 libxcb-keysyms1 libxcb-icccm4 \
    && rm -rf /var/lib/apt/lists/*
```

* Minimal fix for this exact error would be `libegl1` + `libopengl0`.
* The rest (`libxkbcommon0`, `libxcb-cursor0`, `libxcb-image0`, etc.)
  prevents the next likely `xcb` plugin crash after EGL is fixed.
* Note: base `python:3.13-slim` is now Debian 13 (trixie), where
  `libglib2.0-0` was renamed to `libglib2.0-0t64`. The old name has no
  install candidate and would break rebuilds, so the `t64` name is used.
  Verified via `apt-cache policy` + `apt-get install --dry-run`.

No changes to `docker-compose.yml` or `app/main.py`.

## Unreleased - Fix Qt `xcb` plugin load failure (exit 133)

### Problem

After the `libEGL` fix, rebuild succeeded but the container crashed on start:

```text
qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed to load the Qt xcb platform plugin.
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/usr/local/lib/python3.13/site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized.
Available platform plugins are: minimal, eglfs, vkkhrdisplay, offscreen, xcb, minimalegl, wayland-brcm, wayland-egl, wayland, vnc, linuxfb.
exited with code 133
```

Two root causes, confirmed via `ldd` inside the built image:

1. `opencv-python` ships its own Qt5 `libqxcb.so` under
   `cv2/qt/plugins/platforms/` (only that one plugin). Since `app/main.py`
   does `import cv2` before `from PySide6...`, Qt tries cv2's `xcb` first
   and fails. The app never uses `cv2.imshow` (GUI is PySide6 `QLabel`),
   so the bundled Qt is pure conflict.
2. PySide6's own Qt6 `libqxcb.so` still had missing system libs:
   `libxcb-shape.so.0`, `libxcb-xkb.so.1`, `libSM.so.6`, `libICE.so.6`,
   `libxkbcommon-x11.so.0` all `=> not found`. The `xcb-cursor0` hint in
   the log is generic; `libxcb-cursor0` was already installed.

### Fix

`requirements.txt`: `opencv-python` -> `opencv-python-headless`
(headless ships no `cv2/qt/plugins`, so PySide6's `xcb` is used).

`Dockerfile`: added the 5 missing runtime libs (plus explicit
`libxcb-xfixes0/randr/sync/shm` which were previously only transitive):

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libegl1 libopengl0 libglib2.0-0t64 \
    libxcb1 libx11-6 libxext6 libxrender1 libxi6 libxtst6 \
    libsm6 libice6 \
    libxcb-cursor0 libxkbcommon0 libxkbcommon-x11-0 \
    libxcb-shape0 libxcb-xkb1 libxcb-xfixes0 libxcb-randr0 libxcb-sync1 libxcb-shm0 \
    libdbus-1-3 libfontconfig1 \
    libxcb-image0 libxcb-render-util0 libxcb-keysyms1 libxcb-icccm4 \
    && rm -rf /var/lib/apt/lists/*
```

Package names verified with `apt-cache policy` on `python:3.13-slim`
(Debian 13 trixie): `libsm6`, `libice6`, `libxcb-shape0`, `libxcb-xkb1`,
`libxkbcommon-x11-0` all have install candidates.

### Verify

```bash
# rebuild (expect ~10 min, torch + ultralytics)
docker compose up --build

# no cv2-bundled Qt anymore, PySide6 xcb fully linked
docker run --rm prototype-object-detection-monitor bash -c \
  "ls /usr/local/lib/python3.13/site-packages/cv2/qt/plugins 2>&1; \
   ldd /usr/local/lib/python3.13/site-packages/PySide6/Qt/plugins/platforms/libqxcb.so | grep -c 'not found'"

# headless Qt smoke (no X needed)
docker run --rm -e QT_QPA_PLATFORM=offscreen prototype-object-detection-monitor \
  python -c "import cv2; from PySide6.QtWidgets import QApplication; import sys; app=QApplication(sys.argv); print('QApplication OK', cv2.__version__)"

# full GUI (as in README.md)
xhost +local:docker
RTSP_URL="rtsp://admin:QUSVQW@192.168.1.3:554/ch1/main" docker compose up --build
xhost -local:docker
```

### Verify

```bash
# rebuild (changing the apt layer invalidates pip cache, expect ~10 min)
docker compose up --build

# fast import check, no X needed
docker run --rm prototype-object-detection-monitor \
  python -c "from PySide6.QtGui import QImage, QPixmap; print('Qt import OK')"

# shared libs present
docker run --rm prototype-object-detection-monitor \
  ldconfig -p | grep -E "EGL|OpenGL|xkbcommon|xcb-cursor"

# headless Qt smoke
docker run --rm -e QT_QPA_PLATFORM=offscreen \
  prototype-object-detection-monitor \
  python -c "from PySide6.QtWidgets import QApplication; import sys; app=QApplication(sys.argv); print('QApplication OK')"

# full GUI (as in README.md)
xhost +local:docker
RTSP_URL="rtsp://admin:QUSVQW@192.168.1.3:554/ch1/main" docker compose up --build
xhost -local:docker
```
