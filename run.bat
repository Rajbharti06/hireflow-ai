@echo off
title HireFlow AI — Launching...
cd /d "%~dp0"

echo.
echo  ======================================
echo    HireFlow AI  ^|  Resume Screener
echo  ======================================
echo.

:: Check Python
where python >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Install Python 3.9+ from python.org
    pause
    exit /b 1
)

:: Create venv if missing
if not exist ".venv\Scripts\activate.bat" (
    echo  [SETUP] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Activate venv
call .venv\Scripts\activate.bat

:: Install / upgrade dependencies
echo  [SETUP] Checking dependencies...
pip install -r requirements.txt --quiet --disable-pip-version-check
if errorlevel 1 (
    echo  [ERROR] Dependency installation failed. Run: pip install -r requirements.txt
    pause
    exit /b 1
)

:: Copy .env.example to .env if .env doesn't exist yet
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo  [INFO] Created .env from .env.example — add your API keys there.
    )
)

echo.
echo  [OK] Starting HireFlow AI on http://localhost:8501
echo  [OK] Browser will open automatically.
echo  [INFO] Press Ctrl+C in this window to stop the server.
echo.

:: Open browser after a short delay (background)
start /b cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8501"

:: Launch Streamlit (config.toml handles theme/headless settings)
streamlit run app.py

pause
