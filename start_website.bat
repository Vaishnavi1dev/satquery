@echo off
title SatQuery AI Website Launcher
echo ========================================================
echo       🛰️ Starting SatQuery AI Web Application
echo ========================================================
echo.

echo 1. Starting FastAPI Backend Server on http://127.0.0.1:8000 ...
start "SatQuery Backend API" cmd /k "python -m app.api.main"

echo 2. Starting Vite Frontend Server on http://localhost:5173 ...
start "SatQuery Frontend UI" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================================
echo  ✅ Both servers launched!
echo  👉 Web UI:       http://localhost:5173
echo  👉 API Docs:     http://127.0.0.1:8000/docs
echo ========================================================
