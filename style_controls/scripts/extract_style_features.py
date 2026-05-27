"""
style_controls/scripts/extract_style_features.py
-------------------------------------------------
Load the raw style-control outputs produced by run_style_controls.py and
build a clean per-(question, prompt_id) feature table, then save it as a CSV.

Also loads the corresponding Task 1 Vanilla results so that the style
prompt features can be compared to the original explicit_cot, explicit_no_cot,
and neutral_strict feature values on the same questions.

Usage (from repo root):
  python style_controls/scripts/extract_style_features.py \\
      --raw_instruct  style_controls/outputs/style_instruct_raw.json \\
      --raw_thinking  style_controls/outputs/style_thinking_raw.json \\
      --out_dir       style_controls/outputs/

Outputs:
  style_controls/outputs/features_instruct.csv
  style_controls/outputs/features_thinking.csv

Each CSV row: question_id, prompt_id, group, model, + one column per feature.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.datasets import (
    load_task1_instruct,
    load_task1_thinking,
    load_task2_renamed_instruct,
    load_task2_renamed_thinking,
    load_task2_shuffled_instruct,
    load_task2_shuffled_thinking,
    load_task3,
    group_by_mode,
    COT, NOCOT, NEUT,
)
from shared.feature_extraction import (
    extract_linguistic_features,
    compute_paraphrase_consistency,
    compute_perturbation_sensitivity,
    compute_mechanistic_sensitivity,
    FEATURE_NAMES,
)


def build_feature_row(record: dict, para_records=None, pert_records=None) -> dict:
    """Merge all six features into a single flat dict for a result record."""
    row = {"question_id": record.get("question_id", record.get("question", "")[:40])}
    row["prompt_id"] = record.get("prompt_id", record.get("mode", "unknown"))
    row["group"]     = record.get("group", "baseline")
    row["model"]     = record.get("model", "unknown")
    row["correct"]   = record.get("correct", float("nan"))

    # Linguistic features (latency / entropy) — already stored in raw output.
    ling = extract_linguistic_features(record)
    row.update(ling)

    # Paraphrase consistency.
    row["paraphrase_consistency"] = (
        compute_paraphrase_consistency(record, para_records)
        if para_records else float("nan")
    )

    # Perturbation sensitivity.
    row["perturbation_sensitivity"] = (
        compute_perturbation_sensitivity(record, pert_records)
        if pert_records else float("nan")
    )

    # Mechanistic (requires intervention columns — NaN for style variants).
    row["mechanistic_sensitivity"] = compute_mechanistic_sensitivity(record)

    return row


def load_raw(path: Path) -> list:
    return json.loads(path.read_text())


def extract_features_for_model(
    raw_path: Path,
    model_key: str,
) -> pd.DataFrame:
    """Build feature DataFrame for one model's style-control raw results."""
    raw = load_raw(raw_path)

    # ── Load Task 2 & 3 for perturbation and paraphrase features ────────────
    try:
        if model_key == "instruct":
            pert_renamed  = load_task2_renamed_instruct()
            pert_shuffled = load_task2_shuffled_instruct()
        else:
            pert_renamed  = load_task2_renamed_thinking()
            pert_shuffled = load_task2_shuffled_thinking()

        task3 = load_task3()
        # Build lookup by question text prefix.
        def make_lookup(records):
            return {r.get("question", "")[:80].strip(): r for r in records}

        renamed_lookup  = make_lookup(pert_renamed)
        shuffled_lookup = make_lookup(pert_shuffled)
        task3_lookup    = make_lookup(task3)
        have_pert  = True
        have_para  = True
    except FileNotFoundError as e:
        print(f"  Warning: {e}\n  Perturbation/paraphrase features will be NaN.")
        have_pert = have_para = False
        renamed_lookup = shuffled_lookup = task3_lookup = {}

    rows = []
    for rec in raw:
        q_key = rec.get("question", "")[:80].strip()

        pert_recs = []
        if have_pert:
            if q_key in renamed_lookup:
                pert_recs.append(renamed_lookup[q_key])
            if q_key in shuffled_lookup:
                pert_recs.append(shuffled_lookup[q_key])

        para_recs = [task3_lookup[q_key]] if (have_para and q_key in task3_lookup) else None

        row = build_feature_row(rec, para_records=para_recs, pert_records=pert_recs or None)
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_instruct", default=None,
                        help="Path to style_instruct_raw.json (relative to repo root).")
    parser.add_argument("--raw_thinking", default=None,
                        help="Path to style_thinking_raw.json (relative to repo root).")
    parser.add_argument("--out_dir", default="style_controls/outputs")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_key, raw_rel in [
        ("instruct", args.raw_instruct),
        ("thinking", args.raw_thinking),
    ]:
        if raw_rel is None:
            print(f"Skipping {model_key} — no --raw_{model_key} provided.")
            continue

        raw_path = REPO_ROOT / raw_rel
        if not raw_path.exists():
            print(f"File not found: {raw_path} — skipping {model_key}.")
            continue

        print(f"\nExtracting features for {model_key}...")
        df = extract_features_for_model(raw_path, model_key)
        out_path = out_dir / f"features_{model_key}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved {len(df)} rows → {out_path}")


if __name__ == "__main__":
    main()
