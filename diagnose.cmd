@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.cmd first.
  exit /b 2
)
set "PY=.venv\Scripts\python.exe"
echo === Python ===
"%PY%" --version
echo === Packages ===
"%PY%" -c "import cv2,numpy,mediapipe; print('OpenCV',cv2.__version__); print('NumPy',numpy.__version__); print('MediaPipe',getattr(mediapipe,'__version__','unknown'))"
echo === Model ===
if exist "models\hand_landmarker.task" (echo OK: models\hand_landmarker.task) else (echo MISSING: models\hand_landmarker.task)
echo === Cameras (MSMF) ===
call "%~dp0run.cmd" --list-cameras --backend msmf
exit /b %ERRORLEVEL%
