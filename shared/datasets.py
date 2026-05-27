"""
shared/datasets.py
------------------
Load and normalise result files from Task 1, 2, and 3.

Each task stores results as a JSON/TXT file with the structure:
  {
    "summary": { "<mode>": {"correct": int, "count": int}, ... },
    "results":  [ {"question": str, "gold": str, "pred": str,
                   "correct": 0|1, "mode": str, "output": str, ...}, ... ]
  }

Mode strings used in the data  (the paper calls them explicit_cot /
explicit_no_cot / neutral_strict; the codebase stores them as the
keys seen in the summary block, typically "explicit_cot", "no_cot",
"neutral").  load_results() normalises every variant to the canonical
set below so downstream code never has to worry about the difference.
"""

import json
import re
from pathlib import Path
from typing import Dict, List

# ── canonical mode names ──────────────────────────────────────────────────────
COT   = "explicit_cot"
NOCOT = "explicit_no_cot"   # normalised from "no_cot"
NEUT  = "neutral_strict"    # normalised from "neutral"

_MODE_ALIASES: Dict[str, str] = {
    "explicit_cot":    COT,
    "cot":             COT,
    "explicit_no_cot": NOCOT,
    "no_cot":          NOCOT,
    "no-cot":          NOCOT,
    "nocot":           NOCOT,
    "neutral_strict":  NEUT,
    "neutral":         NEUT,
    "neutral strict":  NEUT,
}

# ── repo root (two levels up from this file: shared/ → repo root) ────────────
REPO_ROOT = Path(__file__).parent.parent


def _normalise_mode(mode: str) -> str:
    """Return canonical mode name, raising KeyError on unknown mode."""
    key = mode.strip().lower()
    if key not in _MODE_ALIASES:
        raise KeyError(
            f"Unknown mode '{mode}'. Known aliases: {list(_MODE_ALIASES)}"
        )
    return _MODE_ALIASES[key]


def _parse_file(path: Path) -> List[dict]:
    """Read a JSON or .txt result file and return a flat list of result dicts."""
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    # Some files wrap results under a top-level "results" key; others are bare lists.
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and "results" in data:
        records = data["results"]
    else:
        raise ValueError(f"Unrecognised file structure in {path}")
    # Normalise mode field in place.
    for r in records:
        if "mode" in r:
            r["mode"] = _normalise_mode(r["mode"])
    return records


# ── Task 1 — Vanilla baseline ─────────────────────────────────────────────────

def load_task1_instruct(n: int | None = None) -> List[dict]:
    """Load Task 1 Vanilla Instruct results (all three prompt conditions).

    Args:
        n: If given, return only the first *n* questions per mode.

    Returns:
        List of result dicts with fields: question, gold, pred, correct, mode,
        output, and (if present) latency_per_token, entropy_mean, entropy_slope.
    """
    # Try the most likely filenames; extend this list if your file is named differently.
    candidates = [
        REPO_ROOT / "Task 1" / "Vanilla" / "gsm8k_qwen_instruct_results.json",
        REPO_ROOT / "Task 1" / "Vanilla" / "gsm8k_instruct_results.json",
    ]
    for p in candidates:
        if p.exists():
            records = _parse_file(p)
            return _maybe_trim(records, n)
    raise FileNotFoundError(
        f"Task 1 Instruct results not found. Looked in:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


def load_task1_thinking(n: int | None = None) -> List[dict]:
    """Load Task 1 Vanilla Thinking results."""
    candidates = [
        REPO_ROOT / "Task 1" / "Vanilla" / "gsm8k_qwen_think_results.json",
        REPO_ROOT / "Task 1" / "Vanilla" / "gsm8k_think_results.json",
    ]
    for p in candidates:
        if p.exists():
            records = _parse_file(p)
            return _maybe_trim(records, n)
    raise FileNotFoundError(
        f"Task 1 Thinking results not found. Looked in:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


# ── Task 2 — Perturbation variants ────────────────────────────────────────────

def load_task2_renamed_instruct(n: int | None = None) -> List[dict]:
    p = REPO_ROOT / "Task 2" / "Renamed" / "CTVP_renamed_gsm8k_qwen_instruct_results.json"
    return _maybe_trim(_parse_file(p), n)


def load_task2_renamed_thinking(n: int | None = None) -> List[dict]:
    p = REPO_ROOT / "Task 2" / "Renamed" / "CTVP_renamed_gsm8k_qwen_think_results.txt"
    return _maybe_trim(_parse_file(p), n)


def load_task2_shuffled_instruct(n: int | None = None) -> List[dict]:
    p = REPO_ROOT / "Task 2" / "Shuffled" / "CTVP_shuffled_gsm8k_qwen_instruct_results.json"
    return _maybe_trim(_parse_file(p), n)


def load_task2_shuffled_thinking(n: int | None = None) -> List[dict]:
    p = REPO_ROOT / "Task 2" / "Shuffled" / "CTVP_shuffled_gsm8k_qwen_think_results.txt"
    return _maybe_trim(_parse_file(p), n)


# ── Task 3 — Paraphrase variants ──────────────────────────────────────────────

def load_task3(n: int | None = None) -> List[dict]:
    """Load Task 3 paraphrase results.

    The file may be JSON or plain text — we handle both.
    Expected fields: question_id (or question), mode, correct.
    """
    candidates = [
        REPO_ROOT / "Task 3" / "Task 3 GSM8K.txt",
        REPO_ROOT / "Task 3" / "Task3_GSM8K.json",
        REPO_ROOT / "Task 3" / "task3_gsm8k_results.json",
    ]
    for p in candidates:
        if p.exists():
            records = _parse_file(p)
            return _maybe_trim(records, n)
    raise FileNotFoundError(
        f"Task 3 results not found. Looked in:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _maybe_trim(records: List[dict], n: int | None) -> List[dict]:
    """Return first n records per mode, or all if n is None."""
    if n is None:
        return records
    from collections import defaultdict
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        buckets[r.get("mode", "unknown")].append(r)
    out = []
    for mode_records in buckets.values():
        out.extend(mode_records[:n])
    return out


def group_by_mode(records: List[dict]) -> Dict[str, List[dict]]:
    """Split a flat list of results into a dict keyed by mode."""
    from collections import defaultdict
    d: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        d[r["mode"]].append(r)
    return dict(d)


def align_by_question(
    cot_records: List[dict],
    nocot_records: List[dict],
    neut_records: List[dict],
) -> List[tuple]:
    """Return list of (cot_r, nocot_r, neut_r) tuples matched by question text.

    Questions are matched on the first 80 characters of the question field so
    minor whitespace differences don't break alignment.
    """
    def key(r: dict) -> str:
        return r.get("question", "")[:80].strip()

    cot_map   = {key(r): r for r in cot_records}
    nocot_map = {key(r): r for r in nocot_records}
    neut_map  = {key(r): r for r in neut_records}

    common = set(cot_map) & set(nocot_map) & set(neut_map)
    if not common:
        raise ValueError(
            "No questions matched across the three prompt conditions. "
            "Check that all three modes are present in your results files."
        )
    return [
        (cot_map[k], nocot_map[k], neut_map[k])
        for k in sorted(common)
    ]
