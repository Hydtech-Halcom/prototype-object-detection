FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 QT_QPA_PLATFORM=xcb \
    YOLO_CONFIG_DIR=/tmp/Ultralytics PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libegl1 libopengl0 libglib2.0-0t64 \
    libxcb1 libx11-6 libxext6 libxrender1 libxi6 libxtst6 \
    libsm6 libice6 \
    libxcb-cursor0 libxkbcommon0 libxkbcommon-x11-0 \
    libxcb-shape0 libxcb-xkb1 libxcb-xfixes0 libxcb-randr0 libxcb-sync1 libxcb-shm0 \
    libdbus-1-3 libfontconfig1 \
    libxcb-image0 libxcb-render-util0 libxcb-keysyms1 libxcb-icccm4 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Drop opencv-python's bundled Qt (conflicts with PySide6 xcb). Cached layers
# may still contain it from before the headless switch, so uninstall explicitly
# then restore headless files in case the uninstall removed shared cv2 files.
RUN pip uninstall -y opencv-python opencv-python-headless || true \
    && pip install --no-cache-dir "opencv-python-headless>=4.9"
RUN mkdir -p logs
COPY app/ app/

CMD ["python", "-m", "app.main"]
