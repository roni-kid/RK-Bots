# SilentPresence Attendance Bot
**GCTU Computer Engineering — Automated Attendance Marking**

Automates the manual process of marking student attendance on `mainsp.gctu.edu.gh` using dual-engine OCR, roster-assisted name matching, and browser automation.

---

## What it does

1. Reads your attendance sheet photo with **two independent OCR engines** — EasyOCR (handwriting-tuned) and Tesseract (second opinion) — and combines what each one finds
2. Matches every OCR-read name fragment against the **real class roster** (`data/BCE_GROUP_B_LIST.xlsx`), which fixes most OCR letter-scrambling errors and correctly tells apart students who share a first name (e.g. two "Elijah"s) by matching on the FULL name, not just one word
3. Auto-detects the **course** and **date** from the sheet
4. Opens a **real Firefox browser** and logs into the portal
5. Navigates to **Attendance → Manual Override**
6. Searches each confidently-matched student's official name and ticks them present
7. Shows a live count of marked students, and flags anything it couldn't confidently resolve (unmatched OCR fragments, or two readings that both claim the same student) for **your manual review** instead of guessing

### Why roster-matching matters

Handwriting OCR alone regularly misreads a name (e.g. "InSah Noble" instead of "Noble Ansah"). Rather than trusting raw OCR text, the bot treats it as a rough guess and searches for the closest real student in your official class list. Anything it isn't confident about is never silently marked — it's surfaced to you by name so you can check it in seconds instead of auditing the whole class.

---

## Setup

### Windows (Recommended — easiest)

**Step 1 — Install Python 3.11+**
Download from https://python.org — tick "Add to PATH" during install.

**Step 2 — Run the bot**
Double-click `run_windows.bat` — it installs everything (including EasyOCR) and opens the app.

> **Note:** EasyOCR downloads its handwriting model (~200 MB) automatically on the very first run. After that it's instant.

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
| OCR can't read names | Use a well-lit, clear, flat photo. Avoid glare, shadows, and blur. |
| Names flagged "unmatched" or "ambiguous" | Shown in the red review tray — check them manually; this means neither OCR engine's reading was close enough to a roster name to trust automatically. |
| Student not found on portal | The bot lists them for manual marking — the roster match worked but the portal search came up empty (check spelling matches the portal exactly). |
| Login fails / times out | The portal is client-rendered and sometimes needs a refresh — the bot now retries this automatically. If it still fails, check credentials by logging in manually first. |
| Browser doesn't open | Make sure Playwright's Firefox is installed: `python -m playwright install firefox` |
| EasyOCR slow on first run | Normal — it's downloading the model (~200MB). Subsequent runs are fast. |
| Tesseract not found | Install from the link above — the bot still works with EasyOCR alone, just slightly less accurate. |
| Roster not loading | Make sure `data/BCE_GROUP_B_LIST.xlsx` exists and matches the original column layout (`Index Number`, `Name`, `Gender`). |
| Android: no browser | Use the Manual mode and do browser steps yourself |

---

## File structure

```
silent_presence_bot/
├── app.py                        ← Main application
├── roster.py                     ← Class-roster loading + fuzzy name matching
├── data/
│   └── BCE_GROUP_B_LIST.xlsx     ← Official class list (index numbers + names)
├── requirements.txt              ← Python dependencies
├── run_windows.bat                ← Windows launcher
├── run_android_termux.sh         ← Android launcher
├── README.md                     ← This file
└── uploads/                      ← Temporary image storage (auto-created)
```

---

## Technical notes

- **Browser:** Playwright + Firefox (visible, non-headless — you can watch it work)
- **OCR:** Two independent engines run on every sheet:
  - **EasyOCR** (deep learning, handwriting-tuned) — primary engine, also provides text positions used to isolate the NAME column
  - **Tesseract** — second opinion; catches some fragments EasyOCR's detector misses
  - Both engines' output is pooled before roster matching, so only ONE engine needs to get a name right
- **Roster matching:** Every OCR fragment is scored against `data/BCE_GROUP_B_LIST.xlsx` using a blend of full-string similarity and per-word alignment (tolerant of scrambled letters and reordered words). Matches require a full-name-level score ≥0.60 and at least 6 letters of OCR signal — short or weak fragments are never guessed. Two different OCR readings that resolve to the same roster student are flagged **ambiguous** rather than silently merged or duplicated.
- **GUI:** Flask web app — runs locally at `localhost:7842`
- **Security:** Credentials are never stored — only used in memory for the session
- **Staggered timing:** delays between actions to mimic human speed

---

*Built for Aaron Baidoo Andoh Dadson — GCTU Level 20 Computer Engineering*
