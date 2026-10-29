# SilentPresence Attendance Bot
**GCTU Computer Engineering — Automated Attendance Marking**

Automates the manual process of marking student attendance on `mainsp.gctu.edu.gh` using OCR and browser automation.

---

## What it does

1. Reads your attendance sheet photo using OCR (Tesseract)
2. Auto-detects the **course** and **date** from the sheet
3. Opens a **real Chromium browser** and logs into the portal
4. Navigates to **Attendance → Manual Override**
5. Searches each student's name and ticks them present
6. Shows a live count of marked students vs total

---

## Setup

### Windows (Recommended — easiest)

**Step 1 — Install Python 3.11+**
Download from https://python.org — tick "Add to PATH" during install.

**Step 2 — Install Tesseract OCR**
Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Install to the default path: `C:\Program Files\Tesseract-OCR\`
- Tick "Add to PATH" during install

**Step 3 — Run the bot**
Double-click `run_windows.bat` — it installs everything and opens the app.

---

### Android (Termux)

**Step 1 — Install Termux**
From F-Droid (not Play Store): https://f-droid.org/packages/com.termux/

**Step 2 — Run the setup script**
```bash
chmod +x run_android_termux.sh
./run_android_termux.sh
```

**Step 3 — Open in browser**
Open Firefox on your Android and go to: `http://localhost:7842`

> **Note on Android:** Full browser automation (Playwright) may be limited on Android. The bot will still run the OCR and guide you through the process, but you may need to do the browser steps manually if Playwright can't launch on your device. Consider running on Windows for the fully automated experience.

---

## How to use

1. **Open** `http://localhost:7842` in your browser (auto-opens on Windows)
2. **Enter** your portal email/phone and password
3. **Drop** your attendance sheet photo onto the upload area
4. **Choose mode:**
   - **AUTO** — OCR auto-detects course and date (recommended)
   - **MANUAL** — You pick course and date from dropdowns
5. **Click "Start Automation"**
6. Watch the log and counter — the bot marks each student one by one
7. When done, the browser window stays open so you can verify

---

## Troubleshooting

| Problem | Fix |
|---|---|
| OCR can't read names | Use a well-lit, clear photo. Avoid glare or shadows. |
| Student not found | The bot will list them in red. Mark them manually. |
| Login fails | Check credentials. Try logging in manually first. |
| Browser doesn't open | Make sure Playwright is installed: `python -m playwright install chromium` |
| Tesseract error on Windows | Ensure it's installed and in PATH |
| Android: no browser | Use the Manual mode and do browser steps yourself |

---

## File structure

```
silent_presence_bot/
├── app.py                  ← Main application
├── requirements.txt        ← Python dependencies
├── run_windows.bat         ← Windows launcher
├── run_android_termux.sh   ← Android launcher
├── README.md               ← This file
└── uploads/                ← Temporary image storage (auto-created)
```

---

## Technical notes

- **Browser:** Playwright + Firefox (visible, non-headless — you can watch it work)
- **OCR:** Tesseract with Pillow preprocessing (2× upscale + grayscale for better accuracy)
- **GUI:** Flask web app — runs locally at `localhost:7842`
- **Security:** Credentials are never stored — only used in memory for the session
- **Staggered timing:** 1–1.5s delays between actions to mimic human speed

---

*Built for Aaron Baidoo Andoh Dadson — GCTU Level 20 Computer Engineering*
