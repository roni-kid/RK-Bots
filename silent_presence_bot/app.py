#!/usr/bin/env python3
"""
SilentPresence Attendance Automation Bot
Automates marking student attendance on mainsp.gctu.edu.gh
"""

import os, sys, re, json, time, base64, threading, logging
from pathlib import Path
from datetime import datetime
from io import BytesIO
from flask import Flask, render_template_string, request, jsonify, send_from_directory
from PIL import Image
import pytesseract
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ── Config ─────────────────────────────────────────────────────────────────
BASE_URL   = "https://mainsp.gctu.edu.gh"
LOGIN_URL  = f"{BASE_URL}/admin/login"
ATTEND_URL = f"{BASE_URL}/admin/attendance"
UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("SPBot")

app  = Flask(__name__)
STATE = {
    "status": "idle",          # idle | running | done | error
    "log":    [],
    "counts": {"marked": 0, "total": 0, "failed": []},
    "browser_ready": False,
    "page": None,
    "playwright": None,
    "browser": None,
}

# ── OCR helpers ────────────────────────────────────────────────────────────

COURSE_ALIASES = {
    "DIGITAL LOGIC": "ENCE 251",
    "DIGITAL LOGIC COMPUTER DESIGN": "ENCE 251",
    "ENCE 251": "ENCE 251",
    "OPERATING SYSTEMS": "ENCE 243",
    "ENCE 243": "ENCE 243",
    "SYSTEM SIMULATION": "ENCE 253",
    "MATLAB": "ENCE 253",
    "ENCE 253": "ENCE 253",
    "LINEAR ELECTRONICS": "ENEE 221",
    "ENEE 221": "ENEE 221",
    "PROGRAMMING WITH PYTHON": "ENCE 231",
    "PYTHON FOR ENGINEERS": "ENCE 231",
    "ENCE 231": "ENCE 231",
    "COMPUTER ARCHITECTURE": "ENCE 241",
    "ENCE 241": "ENCE 241",
    "ENGINEERING MATHEMATICS": "MATH 211",
    "MATH 211": "MATH 211",
    "ELECTRICITY AND MAGNETISM": "PHY 241",
    "PHY 241": "PHY 241",
}

def emit(msg, level="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = {"ts": ts, "msg": msg, "level": level}
    STATE["log"].append(entry)
    if level == "error":
        log.error(msg)
    else:
        log.info(msg)

def ocr_image(path):
    """Run Tesseract OCR on an image, return raw text."""
    img = Image.open(path)
    # Upscale for better OCR
    w, h = img.size
    img = img.resize((w * 2, h * 2), Image.LANCZOS)
    # Convert to grayscale
    img = img.convert("L")
    text = pytesseract.image_to_string(img, config="--psm 6")
    return text

def extract_course_from_text(text):
    """Try to detect course code or title from OCR text."""
    text_upper = text.upper()
    for keyword, code in COURSE_ALIASES.items():
        if keyword.upper() in text_upper:
            return code
    return None

def extract_date_from_text(text):
    """Try to extract a date from OCR text (various formats)."""
    patterns = [
        r"(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})",   # 28/10/2026 or 28-10-26
        r"(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[,\s]+(\d{4})",
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2})[,\s]+(\d{4})",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return None

def parse_student_names(text):
    """
    Parse the NAME column from the attendance sheet OCR text.
    Returns a list of full name strings.
    """
    names = []
    lines = text.splitlines()

    # Find the header row
    header_idx = None
    for i, line in enumerate(lines):
        if re.search(r"NAME.*INDEX", line, re.IGNORECASE) or \
           re.search(r"INDEX.*NAME", line, re.IGNORECASE) or \
           re.search(r"^\s*NO\s+NAME", line, re.IGNORECASE):
            header_idx = i
            break

    if header_idx is None:
        # Fallback: look for lines that look like names after the header area
        # (lines with 2+ capitalised words, no digits, not short)
        for line in lines:
            line = line.strip()
            if len(line) < 5:
                continue
            # Skip lines that are headers / metadata
            if re.search(r"(COURSE|LECTURER|DATE|CLASS REP|SIGN|INDEX|FACULTY|DEPARTMENT|UNIVERSITY|SEMESTER|SESSION|ACADEMIC|LEVEL|TITLE|CONTACT|START|FINISH)", line, re.IGNORECASE):
                continue
            # Skip lines that are mostly digits
            if re.match(r"^\d+$", line.replace(" ", "")):
                continue
            # A name: 2–5 words, mostly letters
            words = line.split()
            if 2 <= len(words) <= 6 and all(re.match(r"[A-Za-z\-']+$", w) for w in words):
                names.append(" ".join(w.upper() for w in words))
        return names

    # Parse rows below the header
    for line in lines[header_idx + 1:]:
        line = line.strip()
        if not line:
            continue
        # Remove leading row number
        line = re.sub(r"^\d+[\.\s]+", "", line)
        # Extract name: first block of words before index number (digits)
        m = re.match(r"([A-Za-z\s'\-]+?)(?:\s{2,}|\t)(\d{5,})", line)
        if m:
            name = m.group(1).strip()
            if len(name) > 3:
                names.append(name.upper())
            continue
        # Fallback: if line has 2-5 words and no numbers, treat as name
        words = line.split()
        if 2 <= len(words) <= 5 and all(re.match(r"[A-Za-z\-']+$", w) for w in words):
            names.append(" ".join(w.upper() for w in words))

    return list(dict.fromkeys(names))  # deduplicate preserving order

# ── Playwright automation ──────────────────────────────────────────────────

def run_automation(credentials, image_path, manual_course=None, manual_date=None, mode="auto"):
    """Main automation coroutine — runs in background thread."""
    STATE["status"] = "running"
    STATE["counts"] = {"marked": 0, "total": 0, "failed": []}
    STATE["log"] = []

    try:
        # ── Step 1: OCR ────────────────────────────────────────────
        emit("📄 Running OCR on attendance sheet…")
        raw_text = ocr_image(image_path)
        emit(f"OCR raw text (first 500 chars):\n{raw_text[:500]}")

        if mode == "auto":
            course_code = extract_course_from_text(raw_text)
            date_str    = extract_date_from_text(raw_text)
            if course_code:
                emit(f"✅ AUTO detected course: {course_code}")
            else:
                emit("⚠️ Could not auto-detect course — will ask you to select manually on the page.", "warn")
            if date_str:
                emit(f"✅ AUTO detected date: {date_str}")
            else:
                emit("⚠️ Could not auto-detect date — today's date will be used.", "warn")
        else:
            course_code = manual_course
            date_str    = manual_date
            emit(f"📋 MANUAL mode — Course: {course_code}  Date: {date_str}")

        student_names = parse_student_names(raw_text)
        STATE["counts"]["total"] = len(student_names)
        emit(f"👥 Detected {len(student_names)} student(s) on sheet: {student_names}")

        if not student_names:
            emit("❌ No student names found. Please check the image quality.", "error")
            STATE["status"] = "error"
            return

        # ── Step 2: Browser launch ─────────────────────────────────
        emit("🌐 Launching Firefox…")
        pw = sync_playwright().start()
        browser = pw.firefox.launch(headless=False)
        ctx  = browser.new_context(viewport={"width": 1400, "height": 900})
        page = ctx.new_page()
        STATE["playwright"] = pw
        STATE["browser"]    = browser
        STATE["page"]       = page

        # ── Step 3: Login ──────────────────────────────────────────
        emit(f"🔐 Navigating to {LOGIN_URL}…")
        page.goto(LOGIN_URL, wait_until="networkidle", timeout=30000)
        time.sleep(1)

        # Fill credentials
        email_field = page.locator("input[type='text'], input[type='email'], input[name*='email'], input[name*='user'], input[placeholder*='email' i], input[placeholder*='user' i]").first
        email_field.fill(credentials["email"])
        pw_field = page.locator("input[type='password']").first
        pw_field.fill(credentials["password"])
        emit("🔑 Credentials entered.")

        login_btn = page.locator("button:has-text('Login'), button[type='submit'], input[type='submit']").first
        login_btn.click()
        page.wait_for_url(f"{BASE_URL}/admin/**", timeout=15000)
        emit("✅ Login successful!")

        # ── Step 4: Navigate to Attendance ─────────────────────────
        emit("📌 Navigating to Attendance page…")
        page.goto(ATTEND_URL, wait_until="networkidle", timeout=20000)
        time.sleep(1.5)

        # ── Step 5: Select course if auto-detected ─────────────────
        if course_code:
            emit(f"🔍 Selecting course: {course_code}…")
            try:
                # Open the course dropdown
                course_dropdown = page.locator("select, [class*='course'], button:has-text('ENEE'), button:has-text('ENCE'), button:has-text('MATH'), button:has-text('PHY')").first
                # Try select element first
                selects = page.locator("select").all()
                for sel in selects:
                    options_text = sel.inner_text()
                    if any(c in options_text.upper() for c in ["ENEE", "ENCE", "MATH", "PHY"]):
                        sel.select_option(label=lambda l: course_code in l.upper() if l else False)
                        break
                # Also try clicking dropdown button
                dropdowns = page.locator("[class*='dropdown'], [class*='select']").all()
                for dd in dropdowns:
                    txt = dd.inner_text()
                    if "ENEE" in txt.upper() or "ENCE" in txt.upper() or "COURSE" in txt.upper():
                        dd.click()
                        time.sleep(0.5)
                        # Click the matching option
                        option = page.locator(f"li:has-text('{course_code}'), option:has-text('{course_code}'), [role='option']:has-text('{course_code}')").first
                        if option.is_visible():
                            option.click()
                            emit(f"✅ Course {course_code} selected.")
                        break
            except Exception as e:
                emit(f"⚠️ Auto-select course failed ({e}). Please select manually in the browser.", "warn")

        # Select date if detected
        if date_str:
            emit(f"📅 Attempting to set date: {date_str}…")
            try:
                date_inputs = page.locator("input[type='date']").all()
                for di in date_inputs:
                    # Try to parse and format the date
                    parsed = None
                    for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %b %Y", "%B %d, %Y"]:
                        try:
                            parsed = datetime.strptime(date_str, fmt)
                            break
                        except:
                            pass
                    if parsed:
                        di.fill(parsed.strftime("%Y-%m-%d"))
                        emit(f"✅ Date set to {parsed.strftime('%Y-%m-%d')}")
            except Exception as e:
                emit(f"⚠️ Auto-set date failed ({e}). Please set manually in the browser.", "warn")

        # Click Filter button
        try:
            filter_btn = page.locator("button:has-text('FILTER'), button:has-text('Filter')").first
            if filter_btn.is_visible():
                filter_btn.click()
                emit("🔘 Filter applied.")
                time.sleep(1.5)
        except:
            pass

        # ── Step 6: Mark each student via Manual Override ──────────
        marked = 0
        failed = []

        for idx, name in enumerate(student_names, 1):
            emit(f"[{idx}/{len(student_names)}] Marking: {name}…")
            try:
                # Click the + Manual button
                manual_btn = page.locator("button:has-text('MANUAL'), button:has-text('Manual'), a:has-text('Manual')").first
                manual_btn.wait_for(state="visible", timeout=8000)
                manual_btn.click()
                time.sleep(0.8)

                # Wait for the Modal to appear
                modal = page.locator("[role='dialog'], .modal, [class*='modal'], [class*='overlay']").first
                modal.wait_for(state="visible", timeout=8000)

                # Clear search and type student name
                search_input = modal.locator("input[type='text'], input[type='search'], input").first
                search_input.clear()
                search_input.fill(name)
                time.sleep(1.2)  # wait for search results

                # Look for the student in results
                # Try to find a checkbox or list item matching the name
                result_items = modal.locator("[class*='result'], li, label, [class*='student']").all()
                found = False

                # Strategy 1: find item containing the full name
                for item in result_items:
                    item_text = item.inner_text().upper().strip()
                    if name in item_text or all(word in item_text for word in name.split() if len(word) > 2):
                        # Make sure checkbox is checked (blue)
                        cb = item.locator("input[type='checkbox']")
                        if cb.count() > 0:
                            if not cb.is_checked():
                                cb.check()
                        else:
                            item.click()
                        emit(f"  ✓ Found and ticked: {item_text[:60]}")
                        found = True
                        break

                # Strategy 2: just check first result if search is specific enough
                if not found:
                    first_cb = modal.locator("input[type='checkbox']").first
                    if first_cb.count() > 0 and first_cb.is_visible():
                        result_label = modal.locator("label, [class*='name'], p").first
                        result_text  = result_label.inner_text().upper() if result_label.count() > 0 else ""
                        # Only tick if at least one word matches
                        name_words = [w for w in name.split() if len(w) > 2]
                        if any(w in result_text for w in name_words):
                            if not first_cb.is_checked():
                                first_cb.check()
                            emit(f"  ✓ Ticked first result: {result_text[:60]}")
                            found = True

                if not found:
                    emit(f"  ⚠️ Could not find '{name}' in results — skipping.", "warn")
                    failed.append(name)
                    # Close modal
                    cancel = modal.locator("button:has-text('CANCEL'), button:has-text('Cancel')").first
                    if cancel.is_visible():
                        cancel.click()
                    time.sleep(0.5)
                    continue

                # Click Verify Selection
                verify_btn = modal.locator("button:has-text('VERIFY'), button:has-text('Verify')").first
                verify_btn.wait_for(state="visible", timeout=5000)
                verify_btn.click()
                time.sleep(1.0)

                # Wait for success toast/notification
                try:
                    page.wait_for_selector("text=Manual override successful, text=marked present, [class*='success'], [class*='toast']", timeout=6000)
                    emit(f"  🟢 Confirmed present: {name}")
                    marked += 1
                    STATE["counts"]["marked"] = marked
                except:
                    # Toast might not match exactly; assume success if no error
                    emit(f"  🟡 Submitted (no confirmation toast detected) — {name}")
                    marked += 1
                    STATE["counts"]["marked"] = marked

                time.sleep(0.5)

            except PlaywrightTimeout as e:
                emit(f"  ❌ Timeout for '{name}': {e}", "error")
                failed.append(name)
                # Try to close any open modal
                try:
                    page.keyboard.press("Escape")
                    time.sleep(0.3)
                except:
                    pass

            except Exception as e:
                emit(f"  ❌ Error for '{name}': {e}", "error")
                failed.append(name)
                try:
                    page.keyboard.press("Escape")
                    time.sleep(0.3)
                except:
                    pass

        # ── Done ───────────────────────────────────────────────────
        STATE["counts"]["marked"] = marked
        STATE["counts"]["failed"] = failed
        total = len(student_names)
        emit(f"")
        emit(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        emit(f"✅ DONE — {marked}/{total} students marked present.")
        if failed:
            emit(f"⚠️  {len(failed)} not found: {', '.join(failed)}", "warn")
        emit(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        STATE["status"] = "done"

    except Exception as e:
        emit(f"❌ Fatal error: {e}", "error")
        STATE["status"] = "error"
        log.exception("Fatal error in automation thread")

# ── Flask routes ───────────────────────────────────────────────────────────

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SilentPresence Bot</title>
<style>
  :root {
    --navy: #0d1b2e;
    --navy2: #162336;
    --navy3: #1e2f47;
    --gold: #c9a227;
    --gold-light: #e8c547;
    --gold-pale: #f5e6a8;
    --text: #e8edf5;
    --text-dim: #8a9ab5;
    --green: #2ecc71;
    --red: #e74c3c;
    --amber: #f39c12;
    --radius: 10px;
    --font: 'Segoe UI', system-ui, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--navy); color: var(--text); font-family: var(--font); min-height: 100vh; }

  header {
    background: var(--navy2);
    border-bottom: 2px solid var(--gold);
    padding: 16px 28px;
    display: flex; align-items: center; gap: 14px;
  }
  header .logo-dot { width: 36px; height: 36px; background: var(--gold); border-radius: 8px;
    display: flex; align-items: center; justify-content: center; font-weight: 900; font-size: 18px; color: var(--navy); }
  header h1 { font-size: 20px; font-weight: 700; letter-spacing: -.3px; }
  header h1 span { color: var(--gold); }
  header small { color: var(--text-dim); font-size: 11px; letter-spacing: 2px; text-transform: uppercase; }

  main { max-width: 960px; margin: 0 auto; padding: 28px 20px; display: grid; gap: 20px; }

  .card {
    background: var(--navy2); border-radius: var(--radius);
    border: 1px solid rgba(201,162,39,.18); padding: 22px;
  }
  .card h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 1.5px; color: var(--gold); margin-bottom: 16px; }

  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  @media(max-width: 600px){ .grid2 { grid-template-columns: 1fr; } }

  label { font-size: 12px; color: var(--text-dim); text-transform: uppercase; letter-spacing: .8px; display: block; margin-bottom: 5px; }
  input[type=text], input[type=password], input[type=date], select, textarea {
    width: 100%; background: var(--navy3); border: 1px solid rgba(201,162,39,.25);
    color: var(--text); border-radius: 6px; padding: 10px 12px; font-size: 14px; outline: none;
    transition: border-color .2s;
  }
  input:focus, select:focus, textarea:focus { border-color: var(--gold); }

  .drop-zone {
    border: 2px dashed rgba(201,162,39,.35); border-radius: var(--radius);
    min-height: 130px; display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 10px; cursor: pointer; transition: border-color .2s, background .2s;
    padding: 20px; text-align: center;
  }
  .drop-zone:hover, .drop-zone.over { border-color: var(--gold); background: rgba(201,162,39,.06); }
  .drop-zone .icon { font-size: 36px; }
  .drop-zone p { color: var(--text-dim); font-size: 14px; }
  .drop-zone p strong { color: var(--gold); }
  #preview-img { max-width: 100%; max-height: 200px; border-radius: 6px; display: none; margin-top: 10px; }

  .mode-toggle { display: flex; gap: 0; background: var(--navy3); border-radius: 8px; overflow: hidden; border: 1px solid rgba(201,162,39,.2); }
  .mode-toggle button {
    flex: 1; padding: 10px; border: none; background: transparent; color: var(--text-dim);
    font-size: 13px; cursor: pointer; transition: background .2s, color .2s; font-weight: 600;
  }
  .mode-toggle button.active { background: var(--gold); color: var(--navy); }

  #manual-fields { display: none; }
  #manual-fields.show { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px; }
  @media(max-width:600px){ #manual-fields.show { grid-template-columns: 1fr; } }

  .btn {
    padding: 12px 24px; border-radius: 8px; border: none; font-size: 14px; font-weight: 700;
    cursor: pointer; transition: opacity .2s, transform .1s; letter-spacing: .5px;
  }
  .btn:active { transform: scale(.98); }
  .btn-gold { background: var(--gold); color: var(--navy); }
  .btn-gold:hover { background: var(--gold-light); }
  .btn-red { background: var(--red); color: #fff; }
  .btn:disabled { opacity: .4; cursor: not-allowed; }

  .actions { display: flex; gap: 12px; flex-wrap: wrap; }

  .counters { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
  @media(max-width:500px){ .counters { grid-template-columns: 1fr 1fr; } }
  .counter-box { background: var(--navy3); border-radius: 8px; padding: 14px; text-align: center; border: 1px solid rgba(255,255,255,.06); }
  .counter-box .num { font-size: 32px; font-weight: 800; color: var(--gold); }
  .counter-box .num.green { color: var(--green); }
  .counter-box .num.red   { color: var(--red); }
  .counter-box .lbl { font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px; margin-top: 3px; }

  .status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
    padding: 4px 12px; border-radius: 99px;
  }
  .status-badge.idle    { background: rgba(138,154,181,.15); color: var(--text-dim); }
  .status-badge.running { background: rgba(243,156,18,.15);  color: var(--amber); }
  .status-badge.done    { background: rgba(46,204,113,.15);  color: var(--green); }
  .status-badge.error   { background: rgba(231,76,60,.15);   color: var(--red); }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
  .running .dot { animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

  #log-box {
    background: var(--navy); border: 1px solid rgba(255,255,255,.07); border-radius: 8px;
    height: 320px; overflow-y: auto; padding: 12px; font-family: 'Courier New', monospace;
    font-size: 12.5px; line-height: 1.7;
  }
  .log-line { display: flex; gap: 10px; }
  .log-line .ts { color: var(--text-dim); min-width: 58px; flex-shrink: 0; }
  .log-line .msg { word-break: break-word; }
  .log-line.error .msg { color: var(--red); }
  .log-line.warn  .msg { color: var(--amber); }
  .log-line.info  .msg { color: var(--text); }

  .failed-list { margin-top: 12px; }
  .failed-list p { color: var(--text-dim); font-size: 13px; margin-bottom: 8px; }
  .failed-tag { display: inline-block; background: rgba(231,76,60,.12); color: var(--red);
    border: 1px solid rgba(231,76,60,.25); border-radius: 4px; padding: 3px 8px; font-size: 12px; margin: 3px; }

  .progress-bar { height: 6px; background: var(--navy3); border-radius: 3px; overflow: hidden; margin-top: 10px; }
  .progress-fill { height: 100%; background: var(--gold); border-radius: 3px; transition: width .5s; width: 0%; }
</style>
</head>
<body>
<header>
  <div class="logo-dot">SP</div>
  <div>
    <h1>Silent<span>Presence</span> Bot</h1>
    <small>Attendance Automation · GCTU</small>
  </div>
</header>

<main>
  <!-- Credentials -->
  <div class="card">
    <h2>🔐 Portal Credentials</h2>
    <div class="grid2">
      <div>
        <label>Email / Phone</label>
        <input type="text" id="email" placeholder="e.g. 2526401387" autocomplete="username">
      </div>
      <div>
        <label>Password</label>
        <input type="password" id="password" placeholder="••••••••••••" autocomplete="current-password">
      </div>
    </div>
  </div>

  <!-- Attendance Sheet Upload -->
  <div class="card">
    <h2>📄 Attendance Sheet</h2>
    <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
      <div class="icon">📋</div>
      <p>Drop attendance sheet image here or <strong>click to browse</strong></p>
      <p style="font-size:12px; color: var(--text-dim)">PNG · JPG · JPEG · PDF</p>
    </div>
    <input type="file" id="file-input" accept="image/*,.pdf" style="display:none">
    <img id="preview-img" alt="Preview">
  </div>

  <!-- Mode Selection -->
  <div class="card">
    <h2>⚙️ Detection Mode</h2>
    <div class="mode-toggle">
      <button class="active" id="btn-auto" onclick="setMode('auto')">🤖 AUTO — OCR detects course &amp; date</button>
      <button id="btn-manual" onclick="setMode('manual')">✍️ MANUAL — I'll specify</button>
    </div>
    <div id="manual-fields">
      <div>
        <label>Course</label>
        <select id="manual-course">
          <option value="">Select course…</option>
          <option value="ENCE 251">ENCE 251 – Digital Logic Computer Design</option>
          <option value="ENCE 243">ENCE 243 – Operating Systems</option>
          <option value="ENCE 253">ENCE 253 – System Simulation &amp; Control</option>
          <option value="ENEE 221">ENEE 221 – Linear Electronics</option>
          <option value="ENCE 231">ENCE 231 – Programming with Python</option>
          <option value="ENCE 241">ENCE 241 – Computer Architecture</option>
          <option value="MATH 211">MATH 211 – Engineering Mathematics III</option>
          <option value="PHY 241">PHY 241 – Electricity &amp; Magnetism</option>
        </select>
      </div>
      <div>
        <label>Date</label>
        <input type="date" id="manual-date">
      </div>
    </div>
  </div>

  <!-- Controls + Status -->
  <div class="card">
    <h2>🚀 Run</h2>
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
      <span id="status-badge" class="status-badge idle"><span class="dot"></span> Idle</span>
      <div class="actions">
        <button class="btn btn-gold" id="run-btn" onclick="startBot()">▶ Start Automation</button>
        <button class="btn btn-red" id="stop-btn" onclick="stopBot()" disabled>⏹ Stop</button>
      </div>
    </div>
    <div class="progress-bar" style="margin-top:14px">
      <div class="progress-fill" id="progress-fill"></div>
    </div>
  </div>

  <!-- Live Counters -->
  <div class="card">
    <h2>📊 Progress</h2>
    <div class="counters">
      <div class="counter-box">
        <div class="num green" id="cnt-marked">0</div>
        <div class="lbl">Marked Present</div>
      </div>
      <div class="counter-box">
        <div class="num" id="cnt-total">0</div>
        <div class="lbl">On Sheet</div>
      </div>
      <div class="counter-box">
        <div class="num red" id="cnt-failed">0</div>
        <div class="lbl">Not Found</div>
      </div>
    </div>
    <div class="failed-list" id="failed-list" style="display:none">
      <p>Could not find these students:</p>
      <div id="failed-tags"></div>
    </div>
  </div>

  <!-- Log -->
  <div class="card">
    <h2>📋 Activity Log</h2>
    <div id="log-box"></div>
  </div>
</main>

<script>
let mode = 'auto';
let pollTimer = null;
let lastLogLen = 0;
let uploadedFilename = null;

// ── File drop ──────────────────────────────────────────────────────────
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewImg = document.getElementById('preview-img');

['dragenter','dragover'].forEach(e => dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.add('over'); }));
['dragleave','drop'].forEach(e => dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.remove('over'); }));
dropZone.addEventListener('drop', ev => handleFile(ev.dataTransfer.files[0]));
fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));

function handleFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewImg.style.display = 'block';
  };
  reader.readAsDataURL(file);
  uploadFile(file);
}

async function uploadFile(file) {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch('/upload', { method: 'POST', body: fd });
  const data = await res.json();
  uploadedFilename = data.filename;
  dropZone.querySelector('p').textContent = `✅ Uploaded: ${file.name}`;
}

// ── Mode toggle ────────────────────────────────────────────────────────
function setMode(m) {
  mode = m;
  document.getElementById('btn-auto').className   = m === 'auto'   ? 'active' : '';
  document.getElementById('btn-manual').className = m === 'manual' ? 'active' : '';
  const mf = document.getElementById('manual-fields');
  mf.className = m === 'manual' ? 'show' : '';
}

// Set today's date as default for manual date
const today = new Date().toISOString().slice(0,10);
document.getElementById('manual-date').value = today;

// ── Bot control ────────────────────────────────────────────────────────
async function startBot() {
  const email    = document.getElementById('email').value.trim();
  const password = document.getElementById('password').value.trim();
  if (!email || !password) { alert('Please enter your portal credentials.'); return; }
  if (!uploadedFilename)   { alert('Please upload an attendance sheet image first.'); return; }

  const payload = {
    email, password,
    filename: uploadedFilename,
    mode,
    course: mode === 'manual' ? document.getElementById('manual-course').value : null,
    date:   mode === 'manual' ? document.getElementById('manual-date').value   : null,
  };

  document.getElementById('run-btn').disabled  = true;
  document.getElementById('stop-btn').disabled = false;
  setStatus('running');
  lastLogLen = 0;

  await fetch('/start', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
  pollTimer = setInterval(poll, 1000);
}

async function stopBot() {
  await fetch('/stop', { method: 'POST' });
  clearInterval(pollTimer);
  document.getElementById('run-btn').disabled  = false;
  document.getElementById('stop-btn').disabled = true;
  setStatus('idle');
}

async function poll() {
  const res  = await fetch('/state');
  const data = await res.json();

  // Update status
  setStatus(data.status);

  // Update counters
  document.getElementById('cnt-marked').textContent = data.counts.marked;
  document.getElementById('cnt-total').textContent  = data.counts.total;
  document.getElementById('cnt-failed').textContent = data.counts.failed.length;

  // Progress bar
  const pct = data.counts.total > 0 ? (data.counts.marked / data.counts.total) * 100 : 0;
  document.getElementById('progress-fill').style.width = pct + '%';

  // Append new log lines
  const logBox = document.getElementById('log-box');
  const newLines = data.log.slice(lastLogLen);
  newLines.forEach(entry => {
    const div = document.createElement('div');
    div.className = `log-line ${entry.level}`;
    div.innerHTML = `<span class="ts">${entry.ts}</span><span class="msg">${entry.msg}</span>`;
    logBox.appendChild(div);
  });
  if (newLines.length) logBox.scrollTop = logBox.scrollHeight;
  lastLogLen = data.log.length;

  // Failed list
  if (data.counts.failed.length > 0) {
    document.getElementById('failed-list').style.display = 'block';
    document.getElementById('failed-tags').innerHTML =
      data.counts.failed.map(n => `<span class="failed-tag">${n}</span>`).join('');
  }

  // Stop polling if done or error
  if (data.status === 'done' || data.status === 'error') {
    clearInterval(pollTimer);
    document.getElementById('run-btn').disabled  = false;
    document.getElementById('stop-btn').disabled = true;
  }
}

function setStatus(s) {
  const badge = document.getElementById('status-badge');
  const labels = { idle: 'Idle', running: 'Running…', done: 'Done ✓', error: 'Error' };
  badge.className = `status-badge ${s}`;
  badge.innerHTML = `<span class="dot"></span> ${labels[s] || s}`;
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no file"}), 400
    filename = f"upload_{int(time.time())}_{f.filename}"
    path = UPLOAD_DIR / filename
    f.save(path)
    return jsonify({"filename": filename, "path": str(path)})

@app.route("/start", methods=["POST"])
def start():
    if STATE["status"] == "running":
        return jsonify({"error": "already running"}), 400
    data = request.json
    credentials = {"email": data["email"], "password": data["password"]}
    image_path   = UPLOAD_DIR / data["filename"]
    mode         = data.get("mode", "auto")
    manual_course = data.get("course")
    manual_date   = data.get("date")

    t = threading.Thread(
        target=run_automation,
        args=(credentials, image_path, manual_course, manual_date, mode),
        daemon=True
    )
    t.start()
    return jsonify({"ok": True})

@app.route("/stop", methods=["POST"])
def stop():
    STATE["status"] = "idle"
    try:
        if STATE.get("browser"):
            STATE["browser"].close()
        if STATE.get("playwright"):
            STATE["playwright"].stop()
    except:
        pass
    return jsonify({"ok": True})

@app.route("/state")
def state():
    return jsonify({
        "status": STATE["status"],
        "log":    STATE["log"],
        "counts": STATE["counts"],
    })

# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import webbrowser
    port = 7842
    print(f"\n{'='*55}")
    print("  SilentPresence Attendance Bot")
    print(f"  Open in your browser: http://localhost:{port}")
    print(f"{'='*55}\n")
    # Open browser after short delay
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
