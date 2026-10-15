@echo off
cd /d "%~dp0\.."
uvicorn api.main:app --host 0.0.0.0 --port 8130
