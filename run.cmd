@echo off
rem Intent-Based Gesture Engine launcher for cmd.exe / PowerShell.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo .venv is missing. Run setup.cmd first.
  exit /b 2
)
".venv\Scripts\python.exe" "%~dp0app.py" %*
exit /b %ERRORLEVEL%
