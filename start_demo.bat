@echo off
rem Double-click to start the CompanionAI demo: the patient chat page and the carer page open in the browser.
rem Needs the one-time "Set up" steps in README.md, and LM Studio running with the model loaded (Developer -> Start Server).
rem Offline test without LM Studio:  start_demo.bat --provider mock
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo No Python environment found. Do the "Set up" steps in README.md first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" demo\server.py %*
pause
