@echo off
chcp 65001 >nul
cd /d "%~dp0frontend"
echo [frontend] Serving static UI on http://localhost:8522
python -m http.server 8522 --bind 0.0.0.0
pause
