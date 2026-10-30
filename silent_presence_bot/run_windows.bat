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
echo [1/4] Installing Python packages (EasyOCR download ~200MB on first run)...
pip install flask playwright easyocr pytesseract Pillow numpy requests pandas openpyxl --quiet

:: Install Playwright browser
echo [2/4] Installing Playwright browser (Firefox)...
python -m playwright install firefox

:: Check Tesseract
echo [3/4] Checking for Tesseract OCR (second engine)...
where tesseract >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WARNING] Tesseract OCR not found on PATH!
    echo   The bot will still run on EasyOCR alone, but for best accuracy:
    echo   Download from: https://github.com/UB-Mannheim/tesseract/wiki
    echo   Install it, tick "Add to PATH", then re-run this script.
    echo.
) else (
    echo   Tesseract found — both OCR engines will run.
)

:: Launch
echo [4/4] Launching bot...
echo.
echo   Your browser will open automatically at http://localhost:7842
echo   Keep this window open while the bot is running.
echo.
python app.py
pause
