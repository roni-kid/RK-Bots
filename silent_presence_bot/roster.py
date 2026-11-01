"""
roster.py — Loads the official class list (BCE_GROUP_B_LIST.xlsx) and matches
OCR-read attendance-sheet names against it.

Why this matters:
  OCR (handwriting) is inherently noisy — "InSah Noble" might really be
  "Nii Sah Nobles" on the roster. Instead of trusting raw OCR text as the
  final answer, we treat it as a *search query* and find the closest real
  student in the roster. This also solves the "duplicate first names"
  problem (multiple "Elijah"s, "Kwame"s, etc.) because we always match
  against FULL names, never just a first/last fragment alone.
"""

import re
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd

ROSTER_PATH = Path(__file__).parent / "data" / "BCE_GROUP_B_LIST.xlsx"


def _normalize_name(raw: str) -> str:
    """
    Convert a roster "Name" cell into a clean, space-separated, all-caps
    full name with no parenthetical notes (e.g. "(Class Captain)") and no
    comma (roster sometimes stores "Surname, First Middle").
    """
    raw = str(raw)
    raw = re.sub(r"\(.*?\)", "", raw).strip()          # drop "(Class Captain)"
    if "," in raw:
        surname, rest = raw.split(",", 1)
        full = f"{rest.strip()} {surname.strip()}"
    else:
        full = raw.strip()
    full = re.sub(r"\s+", " ", full)
    return full.upper()


def load_roster():
    """
    Returns a list of dicts: [{"index": "2526401387", "name": "AARON ANDOH DADSON BAIDOO", "raw_name": "..."}]
    Deduplicated by index number (the source file has at least one exact
    duplicate row).
    """
    if not ROSTER_PATH.exists():
        return []

    df = pd.read_excel(ROSTER_PATH, sheet_name=0, header=1)
    df = df.dropna(subset=["Index Number"])

    seen_index = set()
    roster = []
    for _, row in df.iterrows():
        idx = str(row["Index Number"]).strip()
        # pandas may read index numbers as float (e.g. 2526401387.0)
        if idx.endswith(".0"):
            idx = idx[:-2]
        if idx in seen_index:
            continue
        seen_index.add(idx)
        raw_name = str(row["Name"])
        roster.append({
            "index": idx,
            "name": _normalize_name(raw_name),
            "raw_name": raw_name.strip(),
            "gender": str(row.get("Gender", "")).strip(),
        })
    return roster


def _token_set(name: str) -> set:
    return set(name.upper().split())


def _best_token_alignment(query_tokens: list, name_tokens: list) -> float:
    """
    Greedy best-pairing similarity between two token lists, tolerant of
    word-order shuffling AND letter-level OCR noise within each word
    (e.g. query token "INSAH" vs roster token "ANSAH" still scores high on
    SequenceMatcher even though it's not an exact match). This recovers
    cases like "InSah Noble" -> "NOBLE ANSAH" where EasyOCR scrambled a
    name's letters but got most of them right.

    Returns the mean of each query token's best match against any name
    token, weighted by token length (longer/rarer tokens — surnames,
    unusual first names — matter more than short common ones).
    """
    if not query_tokens or not name_tokens:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0
    for qt in query_tokens:
        weight = len(qt)  # longer tokens are more distinctive, weight more
        best = max(SequenceMatcher(None, qt, nt).ratio() for nt in name_tokens)
        weighted_sum  += best * weight
        total_weight  += weight

    return weighted_sum / total_weight if total_weight else 0.0


def token_order_similarity(a: str, b: str) -> float:
    """
    0.0–1.0 similarity between two names that is tolerant of WORD ORDER,
    for comparing two already-OCR'd/typed name strings (as opposed to
    _best_full_match, which compares one OCR fragment against the whole
    roster). Portal search results and attendance-sheet handwriting
    frequently order names differently — "Aaron Andoh Dadson Baidoo" vs
    "Baidoo Aaron Dadson Andoh" are the same person, but a plain
    SequenceMatcher on the raw strings scores that very low because it's
    comparing character runs, not word identity.

    Blends:
      1. Full-string sequence similarity (catches minor letter noise when
         order happens to already match)
      2. Per-token best-alignment similarity (catches reordering and
         letter-level noise within individual words)
    token-alignment is weighted higher since word reordering is the
    dominant real-world failure mode here.
    """
    a_norm = re.sub(r"[^A-Z0-9 ]", "", a.upper()).strip()
    b_norm = re.sub(r"[^A-Z0-9 ]", "", b.upper()).strip()
    a_tokens = a_norm.split()
    b_tokens = b_norm.split()

    seq_score = SequenceMatcher(None, a_norm, b_norm).ratio()
    token_score = _best_token_alignment(a_tokens, b_tokens)

    return (0.4 * seq_score) + (0.6 * token_score)


def _best_full_match(query: str, roster: list):
    """
    Score every roster entry against the OCR query using a BLEND of:
      1. Full-string sequence similarity (handles minor OCR letter noise
         when word order and spacing are roughly preserved)
      2. Per-token best-alignment similarity (handles word reordering AND
         letter scrambling within individual words — the dominant failure
         mode for blurry handwriting, e.g. "InSah" vs "Ansah")

    Returns (best_entry, score). A single shared common short token
    ("ELIJAH") alone still won't win alone, because the token-alignment
    score averages across ALL query tokens (weighted by length) — missing
    or mismatching the other tokens drags the average down.
    """
    query_norm   = re.sub(r"[^A-Z0-9 ]", "", query.upper()).strip()
    query_tokens = query_norm.split()

    best_entry, best_score = None, 0.0
    for entry in roster:
        name = entry["name"]
        name_tokens = name.split()

        seq_score   = SequenceMatcher(None, query_norm, name).ratio()
        token_score = _best_token_alignment(query_tokens, name_tokens)

        # Token alignment carries more weight: it's the one that recovers
        # reordered/scrambled handwriting OCR, which is our dominant
        # real-world failure mode.
        score = (0.4 * seq_score) + (0.6 * token_score)

        if score > best_score:
            best_entry, best_score = entry, score

    return best_entry, best_score


def match_ocr_name_to_roster(ocr_text: str, roster: list, min_score: float = 0.60):
    """
    Given one OCR-read name fragment/line, find the most likely roster
    student. Returns dict: {"index", "name", "raw_name", "score", "matched": bool}

    min_score defaults to 0.60 — high enough that short/weak fragments
    ("Kel", "Dau") don't get silently pinned to a random student, but low
    enough to recover genuinely scrambled handwriting OCR ("InSah Noble" ->
    "NOBLE ANSAH" scores 0.72). A short query (<6 letters total) is rejected
    outright regardless of score — 2-3 letter fragments simply don't carry
    enough signal to safely identify one specific student out of 120+.
    """
    if not roster:
        return {"matched": False, "score": 0.0, "index": None, "name": None, "raw_name": None}

    clean_len = len(re.sub(r"[^A-Za-z]", "", ocr_text))
    if clean_len < 6:
        return {"matched": False, "score": 0.0, "index": None, "name": None, "raw_name": None}

    entry, score = _best_full_match(ocr_text, roster)
    if entry is None or score < min_score:
        return {"matched": False, "score": round(score, 3), "index": None, "name": None, "raw_name": None}

    return {
        "matched": True,
        "score": round(score, 3),
        "index": entry["index"],
        "name": entry["name"],
        "raw_name": entry["raw_name"],
    }


def match_all_ocr_fragments(ocr_fragments: list, roster: list, min_score: float = 0.45):
    """
    Match a list of OCR name-fragment strings (one per detected row) against
    the roster, returning one best-guess student per fragment PLUS a flag
    for ambiguous cases (two fragments mapping to the same roster student —
    a sign the OCR split one name awkwardly, or two different students
    matched the same weak fragment and need manual review).

    Returns: list of match dicts (same shape as match_ocr_name_to_roster,
    plus the original "ocr_text" and an "ambiguous" bool).
    """
    results = []
    claimed_indices = {}  # roster index -> ocr_text that claimed it first

    for frag in ocr_fragments:
        m = match_ocr_name_to_roster(frag, roster, min_score=min_score)
        m["ocr_text"] = frag
        m["ambiguous"] = False

        if m["matched"]:
            if m["index"] in claimed_indices:
                # Two different OCR fragments both think they're this student.
                # Flag BOTH as ambiguous so a human checks — never silently
                # mark the same student twice or silently drop one.
                m["ambiguous"] = True
                for prior in results:
                    if prior.get("index") == m["index"]:
                        prior["ambiguous"] = True
            else:
                claimed_indices[m["index"]] = frag

        results.append(m)

    return results


if __name__ == "__main__":
    # Quick self-test
    roster = load_roster()
    print(f"Loaded {len(roster)} students.")
    test_fragments = ["Ahwa Redtemer", "InSah Noble", "Batelg Beaeni", "Asen Oku", "kuzu Kuumi Kendt"]
    for f in test_fragments:
        m = match_ocr_name_to_roster(f, roster)
        print(f"{f!r:25} -> {m}")
