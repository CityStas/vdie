@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" ( echo Run setup.cmd first. & exit /b 2 )
echo SAFE CAMERA TEST: tracking/overlay only, no OS input.
call "%~dp0run.cmd" --source mediapipe --backend msmf --no-control --mode touch_surface --set tracking.max_hands=2 %*
pause
