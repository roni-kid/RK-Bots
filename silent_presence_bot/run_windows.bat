@echo off
title SilentPresence Bot — Setup ^& Launch

echo.
echo ==========================================
echo   SilentPresence Attendance Bot
echo   GCTU Computer Engineering
echo ==========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Download from https://python.org
    pause & exit /b 1
)

:: Install dependencies
echo [1/3] Installing Python packages...
pip install flask playwright pytesseract Pillow requests --quiet --break-system-packages 2>nul || pip install flask playwright pytesseract Pillow requests --quiet

:: Install Playwright browser
echo [2/3] Installing Playwright browser (Firefox)...
python -m playwright install firefox

:: Check Tesseract
where tesseract >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WARNING] Tesseract OCR not found!
    echo   Download from: https://github.com/UB-Mannheim/tesseract/wiki
    echo   Install it, then re-run this script.
    echo   Default install path: C:\Program Files\Tesseract-OCR\tesseract.exe
    echo.
    echo   After installing, add it to PATH or the app will prompt you.
    pause
)

:: Launch
echo [3/3] Launching bot...
echo.
echo   Your browser will open automatically at http://localhost:7842
echo   Keep this window open while the bot is running.
echo.
python app.py
pause
