@echo off
chcp 65001 >nul
start "Defect-Backend" cmd /k "cd /d ""%~dp0"" && python -m uvicorn backend.app:app --host 0.0.0.0 --port 8562"
timeout /t 2 /nobreak >nul
start "Defect-Frontend" cmd /k "cd /d ""%~dp0frontend"" && python -m http.server 8522 --bind 0.0.0.0"
timeout /t 3 /nobreak >nul
start "" "http://localhost:8522"
