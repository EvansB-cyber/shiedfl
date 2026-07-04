@echo off
echo Starting ShieldFL API Server...
cd /d "%~dp0"
.venv\Scripts\uvicorn.exe api:app --reload
