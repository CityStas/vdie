@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" ( echo Run setup.cmd first. & exit /b 2 )
if not exist "logs" mkdir "logs"
set LOG=logs\gesture_diagnostic_%RANDOM%.jsonl
echo ============================================================
echo VIRTUAL TOUCH SURFACE + BIMANUAL CONTROL
echo Log: %LOG%
echo Activate: raise INDEX FINGER vertically for about 0.25 s
echo Move: fingertip position maps to the desktop directly
echo Second hand: short pinch = left click; hold = right click
echo Second hand + movement = drag; both pinches + deliberate spread = zoom
echo Desktop selection and canvas pan are OFF by default (opt-in in config)
echo Emergency: CLOSED FIST, or Ctrl+C
echo ============================================================
pause
call "%~dp0run.cmd" --source mediapipe --backend msmf --mode touch_surface --set tracking.max_hands=2 --set bimanual.enabled=true --diagnostic "%LOG%" %*
set RC=%ERRORLEVEL%
echo.
echo Diagnostic log saved to: %LOG%
echo Send the JSONL to ChatGPT for debugging.
pause
exit /b %RC%
