#!/usr/bin/env python3
"""
SilentPresence Attendance Automation Bot
Automates marking student attendance on mainsp.gctu.edu.gh
"""

import os, sys, re, json, time, base64, threading, logging
from pathlib import Path
from datetime import datetime
from io import BytesIO
from urllib.parse import urlparse, parse_qsl, urlencode
from flask import Flask, render_template_string, request, jsonify, send_from_directory
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from difflib import SequenceMatcher
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from roster import load_roster, match_all_ocr_fragments, token_order_similarity

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

def name_similarity(a, b):
    """
    0.0–1.0 similarity between two names, tolerant of OCR noise AND word
    order (surname-first vs surname-last, or any other shuffled order —
    the portal's own display order frequently differs from how a name was
    handwritten on the sheet, or from how the roster stores it).
    """
    return token_order_similarity(a, b)

# ── EasyOCR loader (lazy — downloaded once on first use) ───────────────────
_easyocr_reader = None
def get_ocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        emit("⬇️  Loading EasyOCR model (first run only — may take ~30s)…")
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        emit("✅ EasyOCR model ready.")
    return _easyocr_reader

# ── TrOCR loader (lazy — handwriting checkpoint, downloaded once) ──────────
# TrOCR is a transformer-based OCR model from Microsoft that reads one LINE
# of text at a time. It was trained specifically on handwritten text, so it
# handles slant, irregular spacing, and letter-shape variation that confuse
# EasyOCR and Tesseract. We use it as a RESCUE engine: only lines that
# EasyOCR returned with low confidence are cropped and re-run through TrOCR.
# This avoids the GPU cost of running TrOCR on every pixel of the sheet.
_trocr_processor = None
_trocr_model     = None
_trocr_available = False

def get_trocr():
    """Return (processor, model) or (None, None) if transformers is missing."""
    global _trocr_processor, _trocr_model, _trocr_available
    if _trocr_processor is not None:
        return _trocr_processor, _trocr_model
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        emit("⬇️  Loading TrOCR handwriting model (first run only — ~400 MB)…")
        _trocr_processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        _trocr_model.eval()
        _trocr_available = True
        emit("✅ TrOCR model ready.")
    except Exception as e:
        emit(f"⚠️  TrOCR unavailable ({e}) — continuing with EasyOCR + Tesseract only.", "warn")
        _trocr_processor = None
        _trocr_model     = None
        _trocr_available = False
    return _trocr_processor, _trocr_model

# Confidence threshold below which EasyOCR lines are sent to TrOCR for
# a second opinion. 0.5 means "less than 50% confident" — a safe floor
# for noisy handwriting without making every line pay the TrOCR cost.
TROCR_CONFIDENCE_THRESHOLD = 0.50

def ocr_line_trocr(line_crop_np):
    """
    Run TrOCR on a single line crop (numpy uint8 array, greyscale or RGB).
    Returns the recognised text string, or None if TrOCR is unavailable or fails.
    TrOCR reads the whole image as ONE line, so the caller must pass a
    tight crop of exactly one text row — which is what EasyOCR's bboxes give us.
    """
    processor, model = get_trocr()
    if processor is None:
        return None
    try:
        import torch
        from PIL import Image as PILImage

        # TrOCR expects an RGB PIL image
        if line_crop_np.ndim == 2:
            pil_img = PILImage.fromarray(line_crop_np).convert("RGB")
        else:
            pil_img = PILImage.fromarray(line_crop_np)

        # Add a few pixels of white padding — tight crops can confuse the
        # patch encoder at the image boundary.
        padded = PILImage.new("RGB", (pil_img.width + 8, pil_img.height + 8), (255, 255, 255))
        padded.paste(pil_img, (4, 4))

        pixel_values = processor(images=padded, return_tensors="pt").pixel_values

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=64,
                num_beams=4,       # beam search gives better results than greedy
                early_stopping=True,
            )

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return text if text else None
    except Exception as e:
        emit(f"⚠️  TrOCR line read failed ({e})", "warn")
        return None

def trocr_rescore_low_confidence_lines(image_np, ocr_results):
    """
    For every EasyOCR result whose confidence is below TROCR_CONFIDENCE_THRESHOLD,
    crop that bounding box from the full image and re-run TrOCR on it.
    Returns a NEW results list where low-confidence items are replaced with
    TrOCR's reading (marked with conf=0.70 as a neutral mid-confidence sentinel)
    when TrOCR returns something non-empty.

    High-confidence EasyOCR lines pass through unchanged — we don't pay
    the TrOCR cost for text EasyOCR already read well.
    """
    processor, _ = get_trocr()
    if processor is None:
        # TrOCR not available — pass through unchanged
        return ocr_results

    upgraded = []
    rescued = 0

    for r in ocr_results:
        if r["conf"] >= TROCR_CONFIDENCE_THRESHOLD:
            upgraded.append(r)
            continue

        # Crop the bounding box from the full image
        # EasyOCR bbox: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (top-left clockwise)
        bbox = r["bbox"]
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x1, x2 = max(0, int(min(xs)) - 4), min(image_np.shape[1], int(max(xs)) + 4)
        y1, y2 = max(0, int(min(ys)) - 4), min(image_np.shape[0], int(max(ys)) + 4)

        if x2 <= x1 or y2 <= y1:
            upgraded.append(r)
            continue

        crop = image_np[y1:y2, x1:x2]
        trocr_text = ocr_line_trocr(crop)

        if trocr_text and trocr_text.strip():
            emit(
                f"  🔄 TrOCR rescued [{r['conf']:.2f}→0.70]: "
                f"'{r['text']}' → '{trocr_text}'"
            )
            upgraded.append({
                "text": trocr_text,
                "bbox": r["bbox"],
                # Use 0.70 as a neutral mid-confidence sentinel so this result
                # passes the column-detection confidence filter (>0.10) but
                # downstream code can still distinguish it from a high-confidence
                # EasyOCR read (>0.80) if needed.
                "conf": 0.70,
                "source": "trocr",
            })
            rescued += 1
        else:
            # TrOCR also returned nothing useful — keep original
            upgraded.append(r)

    if rescued:
        emit(f"✅ TrOCR rescued {rescued} low-confidence line(s).")
    else:
        emit("ℹ️  TrOCR: no low-confidence lines needed rescuing.")

    return upgraded

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

# Noise words that are never student names
HEADER_NOISE = re.compile(
    r"(COURSE|LECTURER|DATE|CLASS.?REP|SIGN|INDEX|FACULTY|DEPARTMENT|"
    r"UNIVERSITY|SEMESTER|SESSION|ACADEMIC|LEVEL|TITLE|CONTACT|START|"
    r"FINISH|GHANA|COMMUNICATION|TECHNOLOGY|ENGINEERING|ATTENDANCE|"
    r"SHEET|PROGRAMME|PROG|NO\b|NAME\b|NUMBER\b|FIRST|SECOND|THIRD)",
    re.IGNORECASE,
)

def emit(msg, level="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = {"ts": ts, "msg": msg, "level": level}
    STATE["log"].append(entry)
    if level == "error":
        log.error(msg)
    else:
        log.info(msg)

def preprocess_image(path):
    """
    Enhance the image before OCR:
    - Upscale 2.5×  for more detail
    - Convert to greyscale
    - Sharpen edges (handwriting benefits from this)
    - Boost contrast so light pencil lines become darker
    Returns a numpy array (uint8) ready for EasyOCR.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    # Upscale
    img = img.resize((int(w * 2.5), int(h * 2.5)), Image.LANCZOS)
    # Greyscale
    img = img.convert("L")
    # Sharpen
    img = img.filter(ImageFilter.SHARPEN)
    img = img.filter(ImageFilter.SHARPEN)
    # Boost contrast
    img = ImageEnhance.Contrast(img).enhance(2.0)
    # Convert to numpy for EasyOCR
    return np.array(img)

def ocr_image_easyocr(path):
    """
    Run EasyOCR on the image.
    Returns (full_text: str, results: list[dict])
    Each result dict: {text, bbox, confidence}
    bbox = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]  (top-left → clockwise)
    """
    reader = get_ocr_reader()
    img_np = preprocess_image(path)
    emit("🔍 Running EasyOCR on preprocessed image…")
    raw = reader.readtext(img_np, detail=1, paragraph=False)
    # raw items: (bbox, text, confidence)
    results = []
    lines   = []
    for bbox, text, conf in raw:
        text = text.strip()
        if text:
            results.append({"text": text, "bbox": bbox, "conf": conf})
            lines.append(text)
            emit(f"  OCR [{conf:.2f}]: {text}")
    full_text = "\n".join(lines)
    return full_text, results

def ocr_image_tesseract(path):
    """
    Run Tesseract as a SECOND, independent OCR pass. Tesseract and EasyOCR
    have different failure modes (Tesseract is tuned for print but still
    catches some handwriting fragments EasyOCR's detector misses, or reads
    a noisy fragment more cleanly) — running both and merging candidates
    into one pool before roster-matching gives the best chance of a correct
    read on any given row, since we only need ONE of the two engines to get
    a name right for the roster matcher to recover the full, correct name.

    Returns a flat list of line strings (no bbox/confidence — Tesseract's
    confidence scores are unreliable on handwriting, so we don't use them
    for filtering; the roster-match score is the real quality gate).
    """
    if not _TESSERACT_AVAILABLE:
        emit("⚠️  pytesseract not installed — skipping Tesseract pass (EasyOCR only).", "warn")
        return []
    try:
        img_np = preprocess_image(path)
        img = Image.fromarray(img_np)
        emit("🔍 Running Tesseract (second OCR engine) on preprocessed image…")
        # psm 6 = assume a single uniform block of text (good for a table region)
        raw_text = pytesseract.image_to_string(img, config="--psm 6")
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        for l in lines:
            emit(f"  Tesseract: {l}")
        return lines
    except Exception as e:
        emit(f"⚠️  Tesseract pass failed ({e}) — continuing with EasyOCR only.", "warn")
        return []

def extract_course_from_text(text):
    """Detect course code from OCR text."""
    text_upper = text.upper()
    for keyword, code in COURSE_ALIASES.items():
        if keyword.upper() in text_upper:
            return code
    return None

# Tolerates common OCR letter-swaps on handwritten month abbreviations
# (t<->k, m<->n/rn, etc. are frequent confusions on cursive strokes).
MONTH_RE = (
    r"(Jan|Feb|Mar|Apr|Ma[yj]|Jun|Jul|Aug|Sep|"
    r"Oc[tk]|Nov|De[cs])[a-z]*"
)

_DATE_PATTERNS = [
    # Month-name dates are unambiguous — an index number can never produce
    # "Oct" or "4 Oct", so these are safe to trust without extra validation.
    # The day-number group tolerates 0-3 stray OCR characters right before
    # the month (e.g. "4th" misread as "Ith", "Oct4" with a stuck digit) —
    # we only need to recover ONE real digit for the day.
    rf"[^\d]{{0,3}}(\d{{1,2}})[a-z\.]{{0,4}}\s*{MONTH_RE}\.?\s*,?\s*(\d{{3,4}})",
    rf"{MONTH_RE}\s+(\d{{1,2}})[,\s]+(\d{{4}})",
    # OCR sometimes glues the day digit onto the END of the month token
    # instead of before it (e.g. "4th Oct" misread as "Oct4 ...") — this
    # pattern recovers that specific shape, still anchored on an
    # unambiguous month name.
    rf"{MONTH_RE}\s*(\d{{1,2}})\D{{0,5}}(\d{{3,4}})",
    # Pure-numeric dates (28/10/2026 style) are the risky case — a 10-digit
    # index number fragment can look like one, so these get a calendar
    # sanity check below before being accepted.
    r"(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})",
]

def _fix_ocr_year(y: str) -> int | None:
    """
    OCR sometimes mangles one digit of a 4-digit year (e.g. "2026" ->
    "3026"). Since attendance sheets only ever carry a handful of plausible
    academic years, we correct a single wrong leading/trailing digit against
    the current year rather than rejecting the whole date outright.
    """
    try:
        y_int = int(y)
    except (ValueError, TypeError):
        return None
    if len(str(y)) == 3:
        y_int += 2000 if y_int < 100 else 0
    current_year = datetime.now().year
    # Already plausible as-is
    if abs(y_int - current_year) <= 2:
        return y_int
    # Try swapping each digit for "2" (the near-universal century/decade
    # digit for any sheet dated in the 2020s) and see if that lands
    # within range of the current year — recovers "3026" -> "2026".
    y_str = str(y_int).zfill(4)
    for i in range(len(y_str)):
        candidate = int(y_str[:i] + "2" + y_str[i+1:])
        if abs(candidate - current_year) <= 2:
            return candidate
    return None

def _is_plausible_calendar_date(d, m, y):
    """Reject anything that can't be a real date — this is what stops a
    mangled index-number fragment like '64.02.266' from being accepted as
    a day/month/year."""
    try:
        d, m = int(d), int(m)
        y_fixed = _fix_ocr_year(y)
        if y_fixed is None:
            return False
        if not (1 <= d <= 31 and 1 <= m <= 12):
            return False
        datetime(y_fixed, m, d)  # raises if the day doesn't exist for that month
        return True
    except (ValueError, TypeError):
        return False

def extract_date_from_text(text):
    """
    Extract the attendance date from OCR text.

    Per the sheet's own layout, the date is written right after the literal
    label "DATE" on the "START TIME ... FINISH TIME ... DATE ..." line — so
    we search THAT substring first, ignoring the fixed university/
    department boilerplate header and the student rows entirely (neither
    can contain a real date). Only if no "DATE"-anchored match is found do
    we fall back to scanning the whole text, and even then every numeric
    (non-month-name) candidate must pass a real calendar-date sanity check
    before being accepted — this is what rejects index-number fragments
    like "64.02.266" or "252640**2.266" that LOOK like a date but aren't.
    """
    MONTH_NUM = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
                 "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12,
                 # Common OCR letter-swap variants on handwritten cursive
                 # (matches the MONTH_RE alternation above)
                 "MAJ":5, "OCK":10, "DES":12}

    def _build_clean_date(groups):
        """Turn a regex match's groups into a clean 'DD Mon YYYY' string,
        correcting the year if OCR mangled a digit. Returns None if the
        groups don't resolve to a real date."""
        # Figure out which group is the month name (if any)
        month_name = next((g for g in groups if g and g[:3].isalpha()), None)
        numeric_groups = [g for g in groups if g and g.isdigit()]

        if month_name:
            mon = MONTH_NUM.get(month_name[:3].upper())
            if mon is None or len(numeric_groups) < 2:
                return None
            day_str, year_str = numeric_groups[0], numeric_groups[-1]
            try:
                day = int(day_str)
            except ValueError:
                return None
            y_fixed = _fix_ocr_year(year_str)
            if y_fixed is None or not (1 <= day <= 31):
                return None
            try:
                datetime(y_fixed, mon, day)
            except ValueError:
                return None
            return f"{day:02d} {list(MONTH_NUM.keys())[mon-1].title()} {y_fixed}"
        else:
            if len(groups) != 3:
                return None
            d, mo, y = groups
            if not _is_plausible_calendar_date(d, mo, y):
                return None
            y_fixed = _fix_ocr_year(y)
            return f"{int(d):02d}/{int(mo):02d}/{y_fixed}"

    # ── Stage 1: anchor on the "DATE" label on the sheet ────────────────
    # Tolerant of OCR misreads of the label itself (e.g. "DATH" instead of
    # "DATE" — confirmed from a real run) by allowing the last letter to be
    # any single character rather than requiring an exact "DATE" match.
    date_label_match = re.search(r"DAT[EH]\.{0,5}\s*[:.]?\s*(.{3,25})", text, re.IGNORECASE)
    if date_label_match:
        window = date_label_match.group(1)
        for p in _DATE_PATTERNS:
            m = re.search(p, window, re.IGNORECASE)
            if m:
                clean = _build_clean_date(m.groups())
                if clean:
                    return clean

    # ── Stage 2: fallback — scan everything, but numeric hits still need
    #             to pass the calendar sanity check ─────────────────────
    for p in _DATE_PATTERNS:
        for m in re.finditer(p, text, re.IGNORECASE):
            clean = _build_clean_date(m.groups())
            if clean:
                return clean

    # ── Stage 3: month+year only, no recoverable day digit ──────────────
    # Some handwriting completely destroys the day number into letters
    # ("4th" -> "Ath"/"Ith" with zero actual digits anywhere nearby) — there
    # is genuinely no digit left to extract. Rather than silently using
    # today's date with no explanation, report just the month/year so the
    # caller can warn the user specifically that the DAY needs manual
    # confirmation, instead of pretending full confidence.
    search_window = date_label_match.group(1) if date_label_match else text
    m = re.search(MONTH_RE + r"\.?\s*,?\s*(\d{3,4})", search_window, re.IGNORECASE)
    if m:
        month_name, year_str = m.group(1), m.group(2)
        mon = MONTH_NUM.get(month_name[:3].upper())
        y_fixed = _fix_ocr_year(year_str)
        if mon and y_fixed:
            return f"?? {list(MONTH_NUM.keys())[mon-1].title()} {y_fixed} (day illegible — confirm manually)"

    return None

def is_likely_name(text):
    """
    Returns True if a text fragment looks like a (possibly OCR-mangled)
    student name fragment. Loosened for blurry handwriting: EasyOCR often
    splits one handwritten name into several ragged boxes ("InSah", "Noble",
    "0Lcpis") rather than one clean "FIRST LAST" string, so we accept single
    word fragments too and tolerate a few stray digits/symbols mixed in by
    misreads of cursive strokes — as long as most of the token is letters.
    """
    text = text.strip()
    if len(text) < 2:
        return False
    if HEADER_NOISE.search(text):
        return False
    # A real handwritten name never contains underscores, brackets, plus
    # signs, pipes, etc. — these only show up when Tesseract mashes an
    # entire jumbled table row (name + index + sign column) into one run-on
    # string, e.g. "olAmark Aayye  __ifasactoz3e+ [he". Reject outright
    # rather than let the roster fuzzy-matcher try to make sense of it.
    if re.search(r"[_+\[\]{}|<>=~^@#$%&*]", text):
        return False
    # A single table row's name is realistically well under 30 characters;
    # anything longer is almost certainly multiple columns/rows glued
    # together by a failed table-structure read.
    if len(text) > 30:
        return False
    words = text.split()
    if not (1 <= len(words) <= 6):
        return False
    for w in words:
        clean = re.sub(r"[^A-Za-z0-9]", "", w)
        if not clean:
            return False
        letters = sum(c.isalpha() for c in clean)
        # Majority-letters test tolerates a misread digit or two inside a
        # name (e.g. "0Lcpis") without accepting pure index numbers.
        if letters / len(clean) < 0.6:
            return False
    return True

def parse_student_names_from_results(results):
    """
    Use EasyOCR bbox positions to isolate the NAME column.

    Strategy:
      1. Find the header row containing "NAME" and "INDEX" — note their x-positions.
      2. The NAME column occupies x-space roughly from left-of-NAME to left-of-INDEX.
      3. All text blocks below the header row whose centre-x falls inside the NAME
         column band AND pass is_likely_name() are student names.
      4. Sort by vertical position (top-to-bottom) to preserve sheet order.
    """
    if not results:
        return []

    # Helper: centre of a bbox
    def cx(bbox): return (bbox[0][0] + bbox[2][0]) / 2
    def cy(bbox): return (bbox[0][1] + bbox[2][1]) / 2
    def top(bbox): return bbox[0][1]

    # ── Step 1: find header row ─────────────────────────────────────
    header_y    = None
    name_col_x  = None   # left edge of NAME column
    index_col_x = None   # left edge of INDEX column

    for r in results:
        t = r["text"].upper().strip()
        # Fuzzy header match: EasyOCR often misreads "NAME" as "NAML", "NAMC",
        # etc. on blurry handwriting sheets — exact \bNAME\b missed this.
        if len(t) <= 6 and name_similarity(t, "NAME") >= 0.7:
            header_y   = cy(r["bbox"])
            name_col_x = r["bbox"][0][0]   # left x of NAME cell
        if "INDEX" in t or name_similarity(t, "INDEX") >= 0.7:
            index_col_x = r["bbox"][0][0]

    # ── Step 2: define the column band ─────────────────────────────
    if name_col_x is not None and index_col_x is not None:
        col_left  = name_col_x  - 20   # small tolerance
        col_right = index_col_x + 20
    else:
        # Fallback: use left third of the image as "name column"
        all_xs = [cx(r["bbox"]) for r in results]
        col_left  = min(all_xs)
        col_right = max(all_xs) * 0.55

    emit(f"📐 Name column band: x={col_left:.0f} → {col_right:.0f}, header_y={header_y}")

    # ── Step 3: collect name candidates below header ────────────────
    candidates = []
    for r in results:
        # Must be below the header row
        if header_y is not None and top(r["bbox"]) <= header_y + 10:
            continue
        # Centre x must fall inside name column band
        x_centre = cx(r["bbox"])
        if not (col_left <= x_centre <= col_right):
            continue
        # Confidence filter — being inside the verified NAME column already
        # rules out most noise, so we can afford a lower bar here than for
        # the unpositioned fallback scan below. Blurry handwriting routinely
        # scores 0.1–0.3 even on a correctly-read name.
        if r["conf"] < 0.10:
            continue
        text = r["text"].strip()
        if is_likely_name(text):
            candidates.append((top(r["bbox"]), text.upper()))

    # ── Group fragments on the same table row into one name ──────────
    # EasyOCR sometimes splits one handwritten name across 2+ boxes
    # ("InSah" / "Noble" both sitting on the same ruled line). Merge any
    # candidates whose top-y is within ~18px of each other (same row).
    candidates.sort(key=lambda x: x[0])
    grouped = []
    for y, text in candidates:
        if grouped and abs(y - grouped[-1][0]) <= 18:
            grouped[-1] = (grouped[-1][0], grouped[-1][1] + " " + text)
        else:
            grouped.append((y, text))
    names = list(dict.fromkeys(n for _, n in grouped))

    # ── Step 4: fallback — if column detection found nothing, scan all ─
    if not names:
        emit("⚠️  Column detection found nothing — falling back to full-text name scan.", "warn")
        all_text = "\n".join(r["text"] for r in results)
        for line in all_text.splitlines():
            line = line.strip()
            if is_likely_name(line):
                names.append(line.upper())
        names = list(dict.fromkeys(names))

    return names

# ── Playwright automation ──────────────────────────────────────────────────

def select_course_on_page(page, course_code, emit):
    """
    Select `course_code` in the attendance page's course filter.

    Confirmed from the real page source (not guessed): the control is a
    genuine native <select name="lecture_id" onchange="this.form.submit()">
    inside the filter <form method="GET">. Each <option>'s visible text is
    the full course line, e.g. "ENCE 243 - OPERATING SYSTEMS". Because of
    the onchange handler, calling Playwright's select_option() on it
    submits the form automatically — no separate click or FILTER button
    press is needed or wanted; doing both would double-submit.

    Returns True once the option is selected and the resulting page
    navigation (the form's GET submit) has completed.
    """
    code_upper = course_code.upper().strip()
    code_compact = code_upper.replace(" ", "")

    def text_matches(txt):
        if not txt:
            return False
        t = txt.upper()
        return code_upper in t or code_compact in t.replace(" ", "")

    try:
        select_el = page.locator("select[name='lecture_id']").first
        if select_el.count() == 0:
            emit("❌ Could not find the course <select name='lecture_id'> on this page — "
                 "the portal's markup may have changed. Please select it manually.", "warn")
            return False

        options = select_el.locator("option").all_inner_texts()
        emit(f"🔎 Course dropdown options: {options}")
        matching = [o for o in options if text_matches(o)]
        if not matching:
            emit(f"❌ No course option matched '{course_code}' among {options}. "
                 f"Please select it manually.", "warn")
            return False

        # select_option's onchange triggers this.form.submit(), which
        # navigates the page (GET) — wait for that navigation so the
        # caller doesn't proceed before the new course's page has loaded.
        with page.expect_navigation(wait_until="load", timeout=20000):
            select_el.select_option(label=matching[0])
        emit(f"✅ Course set to '{matching[0]}'.")
        return True
    except Exception as e:
        emit(f"⚠️ Course selection failed: {e}", "warn")
        return False


def set_date_on_page(page, date_str, emit):
    """
    Select the target date in the attendance page's date filter.

    Confirmed from the real page source: the control is a native
    <select name="date" onchange="this.form.submit()">, same pattern as
    the course dropdown. However — important and worth flagging rather
    than silently assuming otherwise — the markup we were given shows
    this <select> populated with only ONE <option>, matching whatever
    date is currently active for the selected course; it is not a full
    calendar or a list of every lecture date. If that holds generally,
    the only way to reach an earlier date is a date actually present as
    an option (e.g. once a given course is selected, its own session
    dates may populate here) or via the page's URL query string
    (?date=YYYY-MM-DD) directly. This function handles both: it first
    tries to find and select a matching <option> if one exists, and
    falls back to navigating with a `date` query parameter appended to
    the current URL if no matching option is present.
    """
    parsed = None
    for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %b %Y", "%B %d, %Y", "%Y-%m-%d"]:
        try:
            parsed = datetime.strptime(date_str, fmt)
            break
        except Exception:
            pass

    if not parsed:
        emit(f"⚠️ Could not parse '{date_str}' into a real date — please set it manually.", "warn")
        return False

    iso = parsed.strftime("%Y-%m-%d")

    try:
        select_el = page.locator("select[name='date']").first
        if select_el.count() == 0:
            emit("❌ Could not find the date <select name='date'> on this page — "
                 "please set it manually.", "warn")
            return False

        option_values = select_el.locator("option").evaluate_all(
            "els => els.map(e => e.value)"
        )
        option_texts = select_el.locator("option").all_inner_texts()
        emit(f"🔎 Date dropdown options: {list(zip(option_values, option_texts))}")

        if iso in option_values:
            with page.expect_navigation(wait_until="load", timeout=20000):
                select_el.select_option(value=iso)
            emit(f"✅ Date set to {iso} via the date dropdown.")
            return True

        # No matching <option> — the dropdown may only ever contain the
        # currently active date. Fall back to driving the URL directly,
        # since the filter form is a GET with named fields (lecture_id,
        # date) that map straight onto query parameters.
        emit(f"⚠️ {iso} isn't an available option in the date dropdown "
             f"(found: {option_texts}) — trying the URL's date parameter instead.", "warn")
        current_url = page.url
        parsed_url = urlparse(current_url)
        query = dict(parse_qsl(parsed_url.query))
        query["date"] = iso
        new_url = parsed_url._replace(query=urlencode(query)).geturl()
        with page.expect_navigation(wait_until="load", timeout=20000):
            page.goto(new_url)

        # Confirm it actually took — re-read the (now reloaded) date select.
        select_el = page.locator("select[name='date']").first
        if select_el.count() > 0:
            new_values = select_el.locator("option").evaluate_all("els => els.map(e => e.value)")
            if iso in new_values:
                emit(f"✅ Date set to {iso} via URL parameter (confirmed in the reloaded dropdown).")
                return True
        emit(f"⚠️ Tried setting the date via URL parameter, but couldn't confirm {iso} took effect — "
             f"please check and set it manually if needed.", "warn")
        return False
    except Exception as e:
        emit(f"⚠️ Date selection failed: {e}", "warn")
        return False

def run_automation(credentials, image_path, manual_course=None, manual_date=None, mode="auto"):
    """Main automation coroutine — runs in background thread."""
    STATE["status"] = "running"
    STATE["counts"] = {"marked": 0, "total": 0, "failed": []}
    STATE["log"] = []

    try:
        # ── Step 1: Dual OCR — EasyOCR (primary, handwriting-tuned) +
        #           Tesseract (secondary, independent cross-check) ──────
        emit("📄 Reading attendance sheet with EasyOCR (handwriting model)…")
        raw_text, ocr_results = ocr_image_easyocr(image_path)
        emit(f"📝 EasyOCR full text:\n{raw_text[:800]}")

        # ── Step 1a: TrOCR rescue — re-read low-confidence EasyOCR lines ──
        # Any EasyOCR bbox whose confidence is below TROCR_CONFIDENCE_THRESHOLD
        # gets cropped and re-run through TrOCR's handwriting-specific model.
        # This replaces the weak EasyOCR reading in-place, so the rest of
        # the pipeline (column detection, roster matching) sees the better text
        # without knowing which engine produced it.
        emit("🔄 Running TrOCR rescue pass on low-confidence EasyOCR lines…")
        preprocessed_np = preprocess_image(image_path)  # reuse the same enhanced image
        ocr_results = trocr_rescore_low_confidence_lines(preprocessed_np, ocr_results)
        # Rebuild raw_text from the (possibly upgraded) results
        raw_text = "\n".join(r["text"] for r in ocr_results if r["text"].strip())

        tesseract_lines = ocr_image_tesseract(image_path)
        # Tesseract output has no bbox, so it can't participate in the
        # column-position parse — but its lines still go into the roster
        # candidate pool below, catching rows EasyOCR's detector skipped.
        combined_text = raw_text + "\n" + "\n".join(tesseract_lines)

        if mode == "auto":
            course_code = extract_course_from_text(combined_text)
            date_str    = extract_date_from_text(combined_text)
            if course_code:
                emit(f"✅ AUTO detected course: {course_code}")
            else:
                emit("⚠️ Could not auto-detect course — please select manually on the page.", "warn")
            if date_str:
                emit(f"✅ AUTO detected date: {date_str}")
            else:
                emit("⚠️ Could not auto-detect date — today's date will be used.", "warn")
        else:
            course_code = manual_course
            date_str    = manual_date
            emit(f"📋 MANUAL mode — Course: {course_code}  Date: {date_str}")

        # ── Step 1b: Combine OCR candidates from both engines ──────────
        easyocr_names = parse_student_names_from_results(ocr_results)
        tesseract_names = [l for l in tesseract_lines if is_likely_name(l)]
        all_candidates = list(dict.fromkeys(easyocr_names + tesseract_names))
        emit(f"👥 Combined OCR candidates from both engines ({len(all_candidates)}): {all_candidates}")

        if not all_candidates:
            emit("❌ No student name fragments found by either engine. Tips: use a flat, well-lit photo with no shadows.", "error")
            STATE["status"] = "error"
            return

        # ── Step 1c: Match every OCR fragment against the REAL class roster ──
        emit("📋 Loading class roster (BCE_GROUP_B_LIST.xlsx)…")
        roster = load_roster()
        if roster:
            emit(f"✅ Roster loaded: {len(roster)} students.")
        else:
            emit("⚠️  Roster file not found or empty — proceeding with raw OCR names (less reliable).", "warn")

        matches = match_all_ocr_fragments(all_candidates, roster) if roster else []

        confirmed  = [m for m in matches if m["matched"] and not m["ambiguous"]]
        ambiguous  = [m for m in matches if m["matched"] and m["ambiguous"]]
        unmatched  = [m for m in matches if not m["matched"]]

        # De-duplicate confirmed matches by roster index (two OCR fragments
        # might legitimately both resolve to the same already-grouped name)
        seen_idx = set()
        deduped_confirmed = []
        for m in confirmed:
            if m["index"] not in seen_idx:
                seen_idx.add(m["index"])
                deduped_confirmed.append(m)
        confirmed = deduped_confirmed

        emit(f"✅ {len(confirmed)} confidently matched to roster students.")
        for m in confirmed:
            emit(f"   {m['ocr_text']!r} → {m['raw_name']}  ({m['index']})  score={m['score']}")

        if ambiguous:
            emit(f"⚠️  {len(ambiguous)} AMBIGUOUS matches (same roster student claimed by multiple OCR reads — review needed):", "warn")
            for m in ambiguous:
                emit(f"   {m['ocr_text']!r} → {m['raw_name']} ({m['index']})  score={m['score']}", "warn")

        if unmatched:
            emit(f"⚠️  {len(unmatched)} OCR fragments could NOT be confidently matched to any roster student:", "warn")
            for m in unmatched:
                emit(f"   {m['ocr_text']!r}  (best score {m['score']})", "warn")

        # Use roster names (reliable, correctly spelled, correctly ordered)
        # for the actual portal search — NOT the raw OCR text — since the
        # roster's "raw_name" is what the portal's own student list contains.
        student_names = [m["raw_name"] for m in confirmed] if roster else all_candidates
        STATE["counts"]["total"] = len(all_candidates)
        STATE["counts"]["marked"] = 0

        if not student_names:
            emit("❌ No OCR fragment could be confidently matched to a roster student. Try a clearer photo.", "error")
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

        # The portal appears to be a client-rendered app: the URL/DOM updates
        # without firing a full "load"/"networkidle" navigation event, which is
        # why wait_for_url used to time out. Instead, poll for either the URL
        # changing away from /login OR a known post-login element showing up —
        # and if neither appears in time, force a refresh (which you confirmed
        # reveals the dashboard content) before giving up.
        logged_in = False
        for attempt in range(20):   # ~10s of polling
            time.sleep(0.5)
            if "/login" not in page.url:
                logged_in = True
                break
            if page.locator("text=Dashboard, text=SilentPresence, text=Command Center").first.is_visible():
                logged_in = True
                break

        if not logged_in:
            emit("🔄 Login didn't confirm yet — refreshing page (matches manual behaviour)…", "warn")
            page.reload(wait_until="load", timeout=20000)
            time.sleep(1.5)
            if "/login" in page.url:
                raise Exception("Still on login page after refresh — check credentials.")

        emit("✅ Login successful!")

        # ── Step 4: Navigate to Attendance ─────────────────────────
        emit("📌 Navigating to Attendance page…")
        page.goto(ATTEND_URL, wait_until="load", timeout=20000)
        time.sleep(1.5)
        # Same SPA quirk can apply here — refresh once if the ledger content
        # hasn't rendered yet (client-side data fetch didn't fire on first paint)
        try:
            page.locator("text=Attendance Ledger").first.wait_for(state="visible", timeout=3000)
        except PlaywrightTimeout:
            emit("🔄 Refreshing Attendance page to load data…", "warn")
            page.reload(wait_until="load", timeout=20000)
            time.sleep(1.5)

        # ── Step 5: Select course if auto-detected ─────────────────
        if course_code:
            emit(f"🔍 Selecting course: {course_code}…")
            select_course_on_page(page, course_code, emit)

        # Select date if detected
        if date_str and "illegible" in date_str:
            emit(f"⚠️ Could only read the month/year ({date_str}) — the day digit was fully "
                 f"obscured by handwriting. Please set the correct date manually on the page.", "warn")
        elif date_str:
            emit(f"📅 Attempting to set date: {date_str}…")
            set_date_on_page(page, date_str, emit)

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
        # The real modal (confirmed from the page's own HTML) is a single
        # element that always exists in the DOM and is toggled by removing/
        # adding the "hidden" Tailwind class — it is NOT destroyed and
        # recreated per open. That's why a modal left open by a prior failed
        # attempt can sit on top of the page (z-50, full-screen backdrop)
        # and silently block every click on the next iteration. We force it
        # closed before every single attempt, not just after failures.
        MODAL_SELECTOR = "#manual-override-modal"

        def force_close_modal(page):
            """Guarantee the modal is hidden before we try to open it again."""
            try:
                page.evaluate(
                    "document.getElementById('manual-override-modal')"
                    "?.classList.add('hidden')"
                )
            except Exception:
                pass
            try:
                page.keyboard.press("Escape")
            except Exception:
                pass
            time.sleep(0.3)

        marked = 0
        failed = []

        for idx, name in enumerate(student_names, 1):
            emit(f"[{idx}/{len(student_names)}] Marking: {name}…")

            # Always start from a known-closed state — this is the fix for
            # the "subtree intercepts pointer events" failure: a modal stuck
            # open from a previous timeout/error no longer blocks this turn.
            force_close_modal(page)

            try:
                # Click the + Manual button (opens the modal by removing
                # the "hidden" class on #manual-override-modal)
                manual_btn = page.locator("button:has-text('MANUAL'), button:has-text('Manual'), a:has-text('Manual')").first
                manual_btn.wait_for(state="visible", timeout=8000)
                manual_btn.click(timeout=5000)

                # Wait for the actual modal element to lose "hidden" —
                # more reliable than a generic visibility check, since the
                # element technically exists in the DOM the whole time.
                modal = page.locator(MODAL_SELECTOR)
                page.wait_for_function(
                    "() => { const m = document.getElementById('manual-override-modal');"
                    " return m && !m.classList.contains('hidden'); }",
                    timeout=8000,
                )

                # Clear search and type student name. The portal's own
                # search appears to be order-sensitive (it may index by
                # first-name-first, surname-first, or something in
                # between — and that ordering doesn't always match how a
                # name was handwritten on the sheet or stored in the
                # roster). So instead of a single literal query, try a
                # few reorderings of the name's tokens and stop at the
                # first one that returns any results at all.
                # Confirmed from the real page source: the search box has
                # id="student-search-input" and triggers filterStudents()
                # on keyup, which does a synchronous, client-side,
                # case-insensitive substring match against each row's
                # .search-name and .search-index text — no network
                # request, no debounce. Non-matching rows get
                # style.display='none' (they stay in the DOM, just
                # hidden), so matching must check actual visibility,
                # not just presence of text.
                search_input = modal.locator("#student-search-input")
                if search_input.count() == 0:
                    search_input = modal.locator("input[type='text'], input[type='search'], input").first

                # The roster's "Name" column often has a comma (e.g.
                # "Affadu-Mensah, Judah Ekow") but the portal's own search
                # and display never do (it shows "BRIGHT AKPENE ALBERT",
                # space-separated, no comma, and often in yet another
                # token order than either the roster or the handwriting).
                # Strip the comma first so it never reaches the search box,
                # then tokenize for the reordering attempts below.
                name_no_comma = re.sub(r",", " ", name)
                name_no_comma = re.sub(r"\s+", " ", name_no_comma).strip()
                name_tokens = [w for w in name_no_comma.split() if w]

                query_candidates = [name_no_comma]  # comma-free, as-given, always try first
                if len(name_tokens) >= 2:
                    # swap first <-> last token (covers "Surname First" vs
                    # "First Surname" and most 2-word reorderings)
                    swapped = name_tokens[:]
                    swapped[0], swapped[-1] = swapped[-1], swapped[0]
                    query_candidates.append(" ".join(swapped))
                    # The two most distinctive (longest) tokens together —
                    # e.g. a surname + a middle/given name — sidesteps
                    # ordering entirely and is usually enough to surface
                    # the right row without over-matching like a single
                    # short token could.
                    two_longest = sorted(name_tokens, key=len, reverse=True)[:2]
                    query_candidates.append(" ".join(two_longest))
                    # Fallback: just the single longest token alone.
                    query_candidates.append(max(name_tokens, key=len))
                # de-dupe while preserving order
                seen_q = set()
                query_candidates = [q for q in query_candidates if not (q in seen_q or seen_q.add(q))]

                # Every candidate row, matched or not, is one of these —
                # filterStudents() only toggles their CSS display, so we
                # must filter by is_visible(), not by presence in the DOM.
                all_rows_locator = modal.locator("label.student-checkbox-item")

                def visible_rows():
                    rows = all_rows_locator.all()
                    return [r for r in rows if r.is_visible()]

                result_items = []
                used_query = name_no_comma
                for q in query_candidates:
                    search_input.fill("")
                    search_input.fill(q)
                    # filterStudents() runs synchronously on the keyup
                    # fired by fill(), so the DOM is already updated by
                    # the time fill() returns — no polling needed. A tiny
                    # sleep only guards against any rendering lag.
                    time.sleep(0.05)
                    visible = visible_rows()
                    if visible:
                        result_items = visible
                        used_query = q
                        if q != name_no_comma:
                            emit(f"  🔁 Query '{name_no_comma}' matched nothing — retried as '{q}', got {len(visible)} visible result(s).")
                        break
                else:
                    emit(f"  ⚠️ No visible rows for any reordering of '{name_no_comma}' (tried {query_candidates}).", "warn")

                found = False

                # Strategy 1: find item containing the full name. Read
                # .search-name specifically (the exact text the page's own
                # filterStudents() matches against) rather than the whole
                # row's text, which also includes the index number.
                for item in result_items:
                    name_span = item.locator(".search-name")
                    item_name = (name_span.inner_text() if name_span.count() > 0 else item.inner_text()).upper().strip()
                    if name_no_comma.upper() in item_name or all(
                        word in item_name for word in name_tokens if len(word) > 2
                    ):
                        cb = item.locator("input[type='checkbox']")
                        if cb.count() > 0:
                            if not cb.is_checked():
                                cb.check(timeout=3000)
                        else:
                            item.click(timeout=3000)
                        emit(f"  ✓ Found and ticked: {item_name[:60]}")
                        found = True
                        break

                # Strategy 2: just check the first visible result if the
                # search narrowed things down to a small, plausible set.
                # Use result_items (already filtered to visible rows by
                # the current query) rather than re-querying the modal,
                # which would include hidden non-matching rows again.
                if not found and result_items:
                    first_item = result_items[0]
                    first_cb = first_item.locator("input[type='checkbox']")
                    name_span = first_item.locator(".search-name")
                    result_text = (name_span.inner_text() if name_span.count() > 0 else first_item.inner_text()).upper()
                    name_words = [w for w in name_tokens if len(w) > 2]
                    if first_cb.count() > 0 and any(w.upper() in result_text for w in name_words):
                        if not first_cb.is_checked():
                            first_cb.check(timeout=3000)
                        emit(f"  ✓ Ticked first visible result: {result_text[:60]}")
                        found = True

                # Strategy 3: fuzzy match — handwriting OCR is rarely pixel-perfect,
                # so pick the closest-scoring visible result instead of giving up.
                if not found:
                    best_item, best_score, best_text = None, 0.0, ""
                    for item in result_items:
                        item_text = item.inner_text().strip()
                        if not item_text:
                            continue
                        score = name_similarity(name_no_comma, item_text)
                        if score > best_score:
                            best_item, best_score, best_text = item, score, item_text
                    if best_item is not None and best_score >= 0.55:
                        cb = best_item.locator("input[type='checkbox']")
                        if cb.count() > 0:
                            if not cb.is_checked():
                                cb.check(timeout=3000)
                        else:
                            best_item.click(timeout=3000)
                        emit(f"  ✓ Fuzzy-matched ({best_score:.0%}): '{name}' → '{best_text[:50]}'", "warn")
                        found = True

                # Confirm the checkbox is actually checked before verifying —
                # directly matches your instruction to count only confirmed
                # (blue/checked) boxes, not just "we clicked something".
                if found:
                    checked_box = modal.locator("input[type='checkbox']:checked").first
                    if checked_box.count() == 0:
                        emit(f"  ⚠️ Clicked a result for '{name}' but checkbox isn't confirmed checked — treating as not found.", "warn")
                        found = False

                if not found:
                    emit(f"  ⚠️ Could not find '{name}' in results — skipping.", "warn")
                    failed.append(name)
                    cancel = modal.locator("button:has-text('CANCEL'), button:has-text('Cancel')").first
                    if cancel.is_visible():
                        cancel.click(timeout=3000)
                    force_close_modal(page)
                    continue

                # Click Verify Selection
                verify_btn = modal.locator("button:has-text('VERIFY'), button:has-text('Verify')").first
                verify_btn.wait_for(state="visible", timeout=5000)
                verify_btn.click(timeout=5000)

                # Wait for success toast/notification — this wait_for_selector
                # already blocks until either the toast appears or the timeout
                # elapses, so no separate fixed sleep is needed beforehand.
                try:
                    page.wait_for_selector("text=Manual override successful, text=marked present, [class*='success'], [class*='toast']", timeout=6000)
                    emit(f"  🟢 Confirmed present: {name}")
                    marked += 1
                    STATE["counts"]["marked"] = marked
                except Exception:
                    emit(f"  🟡 Submitted (no confirmation toast detected) — {name}")
                    marked += 1
                    STATE["counts"]["marked"] = marked

                # Make sure the modal is actually gone before looping —
                # this is what used to leave it stuck open for the NEXT
                # student's attempt. force_close_modal already waits
                # briefly for the close to take effect.
                force_close_modal(page)

            except PlaywrightTimeout as e:
                emit(f"  ❌ Timeout for '{name}': {e}", "error")
                failed.append(name)
                force_close_modal(page)

            except Exception as e:
                emit(f"  ❌ Error for '{name}': {e}", "error")
                failed.append(name)
                force_close_modal(page)

        # ── Done ───────────────────────────────────────────────────
        # "failed" in the UI's red "Not Found" tray should show every name
        # that still needs YOUR eyes: portal-search misses, plus any OCR
        # fragment the roster matcher couldn't confidently resolve, plus
        # ambiguous double-claims. Nothing gets silently dropped.
        review_needed = failed[:]  # portal-search misses from the loop above
        review_needed += [f"{m['ocr_text']} (unmatched)" for m in unmatched]
        review_needed += [f"{m['ocr_text']} → {m['raw_name']}? (ambiguous)" for m in ambiguous]

        STATE["counts"]["marked"] = marked
        STATE["counts"]["failed"] = review_needed
        total = STATE["counts"]["total"]
        emit(f"")
        emit(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        emit(f"✅ DONE — {marked}/{total} OCR-detected row(s) marked present on the portal.")
        if review_needed:
            emit(f"⚠️  {len(review_needed)} need your manual review: {', '.join(review_needed)}", "warn")
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
