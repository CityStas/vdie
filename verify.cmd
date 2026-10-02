@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" ( echo Run setup.cmd first. & exit /b 2 )
echo === Python / dependency smoke ===
".venv\Scripts\python.exe" --version
echo === OpenCV package hygiene ===
".venv\Scripts\python.exe" -c "import cv2,sys; print('cv2=',cv2.__version__); print('cv2=',cv2.__file__); sys.exit(0 if cv2.__version__.split('.')[0] == '4' else 3)"
if errorlevel 1 (
  echo [ERROR] cv2 is not OpenCV 4.x. Remove contrib/headless variants and run setup.cmd.
  exit /b %ERRORLEVEL%
)
echo === Unit tests (optional) ===
".venv\Scripts\python.exe" -c "import pytest" >nul 2>&1
if errorlevel 1 (
  echo [INFO] pytest is not installed; skipping unit suite.
) else (
  ".venv\Scripts\python.exe" -m pytest -q
  if errorlevel 1 exit /b %ERRORLEVEL%
)
echo === Synthetic safe smoke ===
call "%~dp0run.cmd" --source synthetic --scenario click --headless --no-control --frames 180 --quiet --diagnostic logs\verify_%RANDOM%.jsonl
if errorlevel 1 exit /b %ERRORLEVEL%
echo.
echo VERIFY OK - no real mouse/keyboard control was used.
exit /b 0
