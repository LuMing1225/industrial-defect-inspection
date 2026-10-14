@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [backend] Starting FastAPI on http://localhost:8562 (docs at /docs)
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8562
pause
