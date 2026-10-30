#!/data/data/com.termux/files/usr/bin/bash
# SilentPresence Bot — Android (Termux) Launcher
# Run: chmod +x run_android_termux.sh && ./run_android_termux.sh

echo ""
echo "=========================================="
echo "  SilentPresence Bot — Android (Termux)"
echo "  GCTU Computer Engineering"
echo "=========================================="
echo ""

# Update packages
echo "[1/5] Updating Termux packages..."
pkg update -y -q && pkg upgrade -y -q

# Install system dependencies
echo "[2/5] Installing system packages..."
pkg install -y python python-pip tesseract libjpeg-turbo libpng 2>/dev/null

# Try installing chromium for Termux (optional, may fail on some devices)
pkg install -y chromium 2>/dev/null && echo "✅ Chromium installed" || echo "⚠️  Chromium not available — will use patchright fallback"

# Install Python packages
echo "[3/5] Installing Python packages..."
pip install flask easyocr pytesseract Pillow numpy requests pandas openpyxl --quiet
pip install playwright --quiet || pip install patchright --quiet

# Install Playwright browsers
echo "[4/5] Installing browser automation..."
python -m playwright install firefox 2>/dev/null || \
echo "⚠️  Browser install skipped — may need manual setup"

echo "✅ EasyOCR will download its model (~200MB) on first run automatically."

# Launch
echo "[5/5] Launching bot..."
echo ""
echo "  Open Firefox on your Android device and go to:"
echo "  http://localhost:7842"
echo ""
echo "  Keep this Termux session open while running."
echo ""
python app.py
