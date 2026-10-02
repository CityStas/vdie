@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" ( echo Run setup.cmd first. & exit /b 2 )
if not exist "logs" mkdir "logs"
set LOG=logs\diagnostic_safe_%RANDOM%.jsonl
echo Starting SAFE diagnostic mode: no OS mouse/keyboard control.
call "%~dp0run.cmd" --source mediapipe --backend msmf --no-control --mode touch_surface --set tracking.max_hands=2 --set bimanual.enabled=true --diagnostic "%LOG%" %*
echo.
echo Diagnostic log: %LOG%
pause
