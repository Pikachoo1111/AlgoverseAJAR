"""
detector_comparison/scripts/build_feature_table.py
---------------------------------------------------
Build the feature matrix needed for the detector comparison (P2.1).

For each question q and model m, we need:
  X[q] = f_neutral[q] − f_nocot[q]    (difference vector, z-scored)
  y[q] = 1 if HCDS_q > 0 else 0       (binary label: neutral closer to CoT?)

The intuition: if the difference vector X[q] predicts y[q], the features
carry signal about hidden reasoning even individually.

Usage (from repo root):
  python detector_comparison/scripts/build_feature_table.py \\
      --model instruct \\
      --out   detector_comparison/features/feature_matrix_instruct.csv

  python detector_comparison/scripts/build_feature_table.py \\
      --model thinking \\
      --out   detector_comparison/features/feature_matrix_thinking.csv

Output CSV columns:
  question_id, hcds_q, y_label,
  diff_latency_per_token, diff_entropy_mean, diff_entropy_slope,
  diff_paraphrase_consistency, diff_perturbation_sensitivity,
  diff_mechanistic_sensitivity
"""

import argparse
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
    align_by_question,
    COT, NOCOT, NEUT,
)
from shared.feature_extraction import (
    extract_all_features,
    FEATURE_NAMES,
)
from shared.hcds_utils import (
    build_feature_matrix,
    zscore_features,
    hcds_per_question,
)


def load_all_records(model_key: str):
    """Load Task 1/2/3 records for a given model.

    Returns:
        cot_records, nocot_records, neut_records — each a list of dicts
        aligned by question text (same order).
    """
    loader = load_task1_instruct if model_key == "instruct" else load_task1_thinking
    all_records = loader()
    by_mode = group_by_mode(all_records)

    cot_records   = sorted(by_mode.get(COT,  []), key=lambda r: r.get("question", "")[:80])
    nocot_records = sorted(by_mode.get(NOCOT, []), key=lambda r: r.get("question", "")[:80])
    neut_records  = sorted(by_mode.get(NEUT,  []), key=lambda r: r.get("question", "")[:80])

    return cot_records, nocot_records, neut_records


def load_perturbation_lookup(model_key: str) -> dict:
    """Build {question_key: [pert_records]} from Task 2."""
    try:
        if model_key == "instruct":
            r1 = load_task2_renamed_instruct()
            r2 = load_task2_shuffled_instruct()
        else:
            r1 = load_task2_renamed_thinking()
            r2 = load_task2_shuffled_thinking()
    except FileNotFoundError as e:
        print(f"  Warning (perturbation): {e}")
        return {}

    lookup: dict = {}
    for rec in r1 + r2:
        key = rec.get("question", "")[:80].strip()
        lookup.setdefault(key, []).append(rec)
    return lookup


def load_paraphrase_lookup() -> dict:
    """Build {question_key: [para_records]} from Task 3."""
    try:
        records = load_task3()
    except FileNotFoundError as e:
        print(f"  Warning (paraphrase): {e}")
        return {}
    lookup: dict = {}
    for rec in records:
        key = rec.get("question", "")[:80].strip()
        lookup.setdefault(key, []).append(rec)
    return lookup


def build_features(
    records: list,
    pert_lookup: dict,
    para_lookup: dict,
) -> list:
    """Extract all six features for each record."""
    feats = []
    for rec in records:
        q_key = rec.get("question", "")[:80].strip()
        pert_recs = pert_lookup.get(q_key, None)
        para_recs = para_lookup.get(q_key, None)
        feats.append(extract_all_features(rec, para_recs, pert_recs))
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["instruct", "thinking"], required=True)
    parser.add_argument("--out",   required=True,
                        help="Output CSV path (relative to repo root).")
    args = parser.parse_args()

    print(f"Building feature matrix for: {args.model}")

    # ── Load records ──────────────────────────────────────────────────────────
    cot_rec, nocot_rec, neut_rec = load_all_records(args.model)
    N = min(len(cot_rec), len(nocot_rec), len(neut_rec))
    cot_rec, nocot_rec, neut_rec = cot_rec[:N], nocot_rec[:N], neut_rec[:N]
    print(f"  {N} aligned questions loaded from Task 1.")

    pert_lookup = load_perturbation_lookup(args.model)
    para_lookup = load_paraphrase_lookup()
    print(f"  {len(pert_lookup)} questions have perturbation records.")
    print(f"  {len(para_lookup)} questions have paraphrase records.")

    # ── Extract features ──────────────────────────────────────────────────────
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress proxy warnings in batch
        cot_feats   = build_features(cot_rec,   pert_lookup, para_lookup)
        nocot_feats = build_features(nocot_rec, pert_lookup, para_lookup)
        neut_feats  = build_features(neut_rec,  pert_lookup, para_lookup)

    # ── Build matrices and z-score ────────────────────────────────────────────
    cot_mat, nocot_mat, neut_mat = build_feature_matrix(
        cot_feats, nocot_feats, neut_feats
    )
    all_mat = np.vstack([cot_mat, nocot_mat, neut_mat])
    all_z   = zscore_features(all_mat)
    cot_z   = all_z[:N]
    nocot_z = all_z[N:2*N]
    neut_z  = all_z[2*N:]

    # ── Per-question HCDS and labels ──────────────────────────────────────────
    hcds_q = np.array([
        hcds_per_question(neut_z[i], nocot_z[i], cot_z[i])
        for i in range(N)
    ])
    y_label = (hcds_q > 0).astype(float)

    # ── Difference vectors X = f_neutral − f_nocot ────────────────────────────
    diff_z = neut_z - nocot_z   # shape (N, 6)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    rows = []
    for i in range(N):
        row = {
            "question_id":  i,
            "question":     neut_rec[i].get("question", "")[:80],
            "hcds_q":       round(float(hcds_q[i]), 6) if not np.isnan(hcds_q[i]) else float("nan"),
            "y_label":      int(y_label[i]) if not np.isnan(hcds_q[i]) else float("nan"),
            "model":        args.model,
        }
        for j, fname in enumerate(FEATURE_NAMES):
            row[f"diff_{fname}"] = (
                round(float(diff_z[i, j]), 6)
                if not np.isnan(diff_z[i, j]) else float("nan")
            )
            # Also store raw (z-scored) condition values for reference.
            row[f"cot_{fname}"]   = round(float(cot_z[i, j]),   6) if not np.isnan(cot_z[i, j])   else float("nan")
            row[f"nocot_{fname}"] = round(float(nocot_z[i, j]), 6) if not np.isnan(nocot_z[i, j]) else float("nan")
            row[f"neut_{fname}"]  = round(float(neut_z[i, j]),  6) if not np.isnan(neut_z[i, j])  else float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")

    # Quick summary.
    valid = y_label[~np.isnan(hcds_q)]
    n_pos = int((valid > 0).sum())
    n_neg = int((valid == 0).sum())
    print(f"  Label balance: {n_pos} positive (HCDS>0), {n_neg} negative (HCDS≤0)")
    hcds_valid = hcds_q[~np.isnan(hcds_q)]
    print(f"  Mean HCDS: {np.mean(hcds_valid):.4f}  (std {np.std(hcds_valid):.4f})")


if __name__ == "__main__":
    main()
