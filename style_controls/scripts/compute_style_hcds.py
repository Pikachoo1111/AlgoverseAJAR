"""
style_controls/scripts/compute_style_hcds.py
--------------------------------------------
Compute HCDS for each of the 6 style-control prompt variants and compare
to the original neutral_strict HCDS from Task 1 Vanilla results.

For each style variant v and model m:
  HCDS_v  = HCDS computed using style prompt v as the "neutral" condition,
             with the same Task 1 explicit_cot / explicit_no_cot poles.

Usage (from repo root):
  python style_controls/scripts/compute_style_hcds.py \\
      --features_instruct style_controls/outputs/features_instruct.csv \\
      --features_thinking style_controls/outputs/features_thinking.csv \\
      --out               style_controls/outputs/style_hcds_results.json

Output JSON structure:
  {
    "instruct": {
      "neutral_strict":  { "hcds_mean": ..., "hcds_ci": [...], "pvalue": ... },
      "cot_style_1":     { ... },
      ...
    },
    "thinking": { ... }
  }
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
    group_by_mode,
    COT, NOCOT, NEUT,
)
from shared.feature_extraction import (
    extract_all_features,
    FEATURE_NAMES,
)
from shared.hcds_utils import compute_hcds, print_hcds_summary


def records_to_feature_list(records: list) -> list:
    """Convert a list of result records to a list of feature dicts."""
    return [extract_all_features(r) for r in records]


def load_task1_features(model_key: str):
    """Return (cot_feats, nocot_feats) from Task 1 Vanilla for a model."""
    loader = load_task1_instruct if model_key == "instruct" else load_task1_thinking
    all_records = loader()
    by_mode = group_by_mode(all_records)

    if COT not in by_mode or NOCOT not in by_mode:
        raise KeyError(
            f"Expected modes {COT} and {NOCOT} in Task 1 data. "
            f"Found: {list(by_mode.keys())}"
        )

    cot_records   = sorted(by_mode[COT],   key=lambda r: r.get("question", "")[:80])
    nocot_records = sorted(by_mode[NOCOT], key=lambda r: r.get("question", "")[:80])

    return (
        records_to_feature_list(cot_records),
        records_to_feature_list(nocot_records),
        cot_records,   # for question alignment
    )


def features_from_csv(csv_path: Path, prompt_id: str) -> tuple:
    """Load features for a specific prompt_id from a feature CSV.

    Returns (feature_list, question_ids) aligned to the questions.
    """
    df = pd.read_csv(csv_path)
    subset = df[df["prompt_id"] == prompt_id].copy()
    if len(subset) == 0:
        raise ValueError(f"prompt_id '{prompt_id}' not found in {csv_path}.")
    subset = subset.sort_values("question_id").reset_index(drop=True)
    feat_list = [
        {col: float(row[col]) for col in FEATURE_NAMES if col in row.index}
        for _, row in subset.iterrows()
    ]
    return feat_list, subset["question_id"].tolist()


def compute_for_all_variants(
    model_key: str,
    features_csv: Path,
) -> dict:
    """Compute HCDS for all 6 style variants + original neutral_strict."""
    print(f"\n── {model_key.upper()} ──────────────────────────────────────")

    # Task 1 CoT and no-CoT poles.
    cot_feats, nocot_feats, cot_records = load_task1_features(model_key)

    # Original neutral_strict HCDS from Task 1.
    loader = load_task1_instruct if model_key == "instruct" else load_task1_thinking
    all_records = loader()
    by_mode = group_by_mode(all_records)
    neut_records  = sorted(by_mode[NEUT], key=lambda r: r.get("question", "")[:80])
    neut_feats    = records_to_feature_list(neut_records)

    # Align lengths (use the minimum common set).
    N = min(len(cot_feats), len(nocot_feats), len(neut_feats))
    cot_feats   = cot_feats[:N]
    nocot_feats = nocot_feats[:N]
    neut_feats  = neut_feats[:N]

    results = {}

    # Baseline: original neutral_strict HCDS.
    baseline = compute_hcds(cot_feats, nocot_feats, neut_feats)
    print_hcds_summary("neutral_strict (baseline)", baseline)
    results["neutral_strict"] = {
        "hcds_mean": baseline["hcds_mean"],
        "hcds_ci":   list(baseline["hcds_ci"]),
        "pvalue":    baseline["pvalue"],
        "n_valid":   baseline["n_valid"],
    }

    # Load feature CSV for style variants.
    if not features_csv.exists():
        print(f"  Feature CSV not found: {features_csv}")
        print("  Run extract_style_features.py first.")
        return results

    df = pd.read_csv(features_csv)
    prompt_ids = df["prompt_id"].unique().tolist()

    for pid in prompt_ids:
        try:
            style_feats, _ = features_from_csv(features_csv, pid)
            style_feats = style_feats[:N]
            # Pad with NaN features if fewer questions.
            while len(style_feats) < N:
                style_feats.append({f: float("nan") for f in FEATURE_NAMES})

            hcds_result = compute_hcds(cot_feats, nocot_feats, style_feats)
            print_hcds_summary(pid, hcds_result)
            results[pid] = {
                "hcds_mean": hcds_result["hcds_mean"],
                "hcds_ci":   list(hcds_result["hcds_ci"]),
                "pvalue":    hcds_result["pvalue"],
                "n_valid":   hcds_result["n_valid"],
            }
        except Exception as e:
            print(f"  ERROR for {pid}: {e}")
            results[pid] = {"error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_instruct",
                        default="style_controls/outputs/features_instruct.csv")
    parser.add_argument("--features_thinking",
                        default="style_controls/outputs/features_thinking.csv")
    parser.add_argument("--out",
                        default="style_controls/outputs/style_hcds_results.json")
    args = parser.parse_args()

    all_results = {}

    for model_key, feat_rel in [
        ("instruct", args.features_instruct),
        ("thinking", args.features_thinking),
    ]:
        feat_path = REPO_ROOT / feat_rel
        all_results[model_key] = compute_for_all_variants(model_key, feat_path)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
