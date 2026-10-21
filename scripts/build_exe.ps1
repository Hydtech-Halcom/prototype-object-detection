# Build ExamMonitor.exe (Windows, one dir). Run from repo root:
#   powershell -ExecutionPolicy Bypass -File scripts/build_exe.ps1
# Needs: .venv with requirements installed + PyInstaller (installed below).
# Output: dist/ExamMonitor/ExamMonitor.exe (zip that folder for target PCs).
$ErrorActionPreference = "Stop"

$py = ".\.venv\Scripts\python.exe"
& $py -m pip install pyinstaller
# Headless cv2 only: the full wheel ships its own Qt which fights PySide6.
& $py -m pip uninstall -y opencv-python
& $py -m pip install "opencv-python-headless>=4.9"

# Model next to the exe: export once here so target PCs never download.
$env:ULTRALYTICS_AUTO_UPDATE = "off"
& $py -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx', imgsz=640)"

& $py -m PyInstaller --noconfirm --onedir --windowed --name ExamMonitor `
  --collect-all ultralytics `
  --collect-all torch `
  --collect-all torchvision `
  --exclude-module matplotlib `
  --exclude-module polars `
  --add-data "yolo11n.onnx;." `
  -p . app/main.py

Write-Host "OK: dist/ExamMonitor/ExamMonitor.exe"
