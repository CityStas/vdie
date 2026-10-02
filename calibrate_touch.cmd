@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" ( echo Run setup.cmd first. & exit /b 2 )
call "%~dp0.venv\Scripts\python.exe" "%~dp0tools\calibrate_touch.py" %*
pause
