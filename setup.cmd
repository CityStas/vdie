@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Intent-Based Gesture Engine - Windows setup
rem Requires Windows x64 and Python 3.11-3.13.

set "PY="
for %%V in (3.13 3.12 3.11) do (
  if not defined PY (
    py -%%V -c "import sys; print(sys.version_info[:2])" >nul 2>&1
    if not errorlevel 1 set "PY=py -%%V"
  )
)

if not defined PY (
  echo [ERROR] Python 3.11, 3.12 or 3.13 was not found.
  echo Install Python from https://www.python.org/downloads/windows/ and enable the Python Launcher.
  exit /b 1
)

echo [OK] Using %PY%

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating virtual environment...
  %PY% -m venv .venv
  if errorlevel 1 exit /b %ERRORLEVEL%
)

set "VENV=.venv\Scripts\python.exe"

echo [2/3] Installing runtime dependencies...
rem Keep this venv to one OpenCV build. opencv-python and opencv-contrib-python
rem both expose the same "cv2" module, so mixing them makes the actual runtime
rem depend on install order. This project uses no contrib APIs.
"%VENV%" -m pip uninstall -y opencv-contrib-python opencv-contrib-python-headless opencv-python-headless >nul 2>&1
rem Do not upgrade pip here: the package is intended to work behind corporate/offline
rem mirrors as long as Python packages are available from the configured index.
"%VENV%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Dependency installation failed.
  echo If your network uses a proxy/private PyPI, configure pip and run setup.cmd again.
  exit /b %ERRORLEVEL%
)

echo [3/3] Verifying bundled MediaPipe model...
if not exist "models\hand_landmarker.task" (
  echo [ERROR] models\hand_landmarker.task is missing from the distribution.
  exit /b 2
)

"%VENV%" -c "import cv2, numpy, sys; print('OpenCV', cv2.__version__); print('NumPy', numpy.__version__); print('cv2 path', cv2.__file__); sys.exit(0 if cv2.__version__.split('.')[0] == '4' else 3)"
if errorlevel 1 (
  echo [ERROR] OpenCV must resolve to the pinned 4.x build from requirements.txt.
  exit /b %ERRORLEVEL%
)
"%VENV%" -c "import mediapipe as mp; print('MediaPipe', getattr(mp, '__version__', 'unknown'))"
if errorlevel 1 (
  echo [ERROR] MediaPipe import failed. See the error above.
  exit /b %ERRORLEVEL%
)

echo.
echo Setup complete.
echo.
echo Safe camera test:  camera_test.cmd
 echo Live mouse control: camera_live.cmd
 echo Camera diagnostics: diagnose.cmd
exit /b 0
