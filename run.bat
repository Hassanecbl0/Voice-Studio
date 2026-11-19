@echo off
title XTTS Voice Studio
cd /d "%~dp0"

echo ========================================================
echo             XTTS Voice Studio
echo ========================================================
echo.
if not exist ".\venv\Scripts\python.exe" (
    echo ERROR: Project Python was not found at: %~dp0venv\Scripts\python.exe
    pause
    exit /b 1
)

echo.
echo Starting Web UI... Once loaded, open your browser to:
echo    the local URL printed by Gradio (normally http://127.0.0.1:7860)
echo.
echo Press Ctrl+C to stop the server.
echo.

.\venv\Scripts\python.exe app.py

echo.
echo ========================================================
echo Server stopped.
echo ========================================================
pause
