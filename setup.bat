@echo off
title HireFlow AI — First-Time Setup
cd /d "%~dp0"

echo.
echo  ======================================
echo    HireFlow AI  ^|  First-Time Setup
echo  ======================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python 3.9+ is required. Download from https://python.org
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  [OK] Python %PYVER% found

echo  [SETUP] Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

echo  [SETUP] Installing dependencies (this may take 2-3 minutes)...
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo  [ERROR] Installation failed. Check your internet connection.
    pause
    exit /b 1
)

echo  [SETUP] Creating .env file...
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo  [INFO] .env created. Open it and add your API keys.
    )
)

echo.
echo  ======================================
echo    Setup complete!
echo    Double-click run.bat to start.
echo  ======================================
echo.
pause
