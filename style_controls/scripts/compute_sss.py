"""
style_controls/scripts/compute_sss.py
--------------------------------------
Compute the Style Shift Score (SSS) for every style variant and model.

SSS = |HCDS_variant − HCDS_neutral| / |HCDS_neutral|

SUCCESS CRITERION (PM plan P1.5):
  SSS < 0.20 for ALL 6 style variants on BOTH models.
  → HCDS shift attributable to style manipulation is < 20%.

Usage (from repo root):
  python style_controls/scripts/compute_sss.py \\
      --results style_controls/outputs/style_hcds_results.json \\
      --out     style_controls/tables/sss_table.csv

Output:
  • Console table with pass/fail column.
  • CSV saved to style_controls/tables/sss_table.csv.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.hcds_utils import style_shift_score

PROMPT_GROUPS = {
    "cot_style_1":     "cot_style_no_reasoning",
    "cot_style_2":     "cot_style_no_reasoning",
    "cot_style_3":     "cot_style_no_reasoning",
    "nocot_style_1":   "nocot_style_reasoning_allowed",
    "nocot_style_2":   "nocot_style_reasoning_allowed",
    "nocot_style_3":   "nocot_style_reasoning_allowed",
}

THRESHOLD = 0.20   # PM plan P1.5 success criterion


def compute_sss_table(results: dict) -> pd.DataFrame:
    """Build SSS table from the style_hcds_results.json structure.

    Args:
        results: dict loaded from style_hcds_results.json.

    Returns:
        DataFrame with columns:
          model, prompt_id, group, hcds_baseline, hcds_variant, sss, pass
    """
    rows = []
    for model_key, model_results in results.items():
        if "neutral_strict" not in model_results:
            print(f"  Warning: neutral_strict missing for {model_key} — skipping.")
            continue

        baseline_hcds = model_results["neutral_strict"]["hcds_mean"]

        for pid, res in model_results.items():
            if pid == "neutral_strict":
                continue
            if "error" in res:
                rows.append({
                    "model": model_key,
                    "prompt_id": pid,
                    "group": PROMPT_GROUPS.get(pid, "unknown"),
                    "hcds_baseline": baseline_hcds,
                    "hcds_variant": float("nan"),
                    "sss": float("nan"),
                    "pass": False,
                    "note": res.get("error", "error"),
                })
                continue

            variant_hcds = res["hcds_mean"]
            sss = style_shift_score(baseline_hcds, variant_hcds)
            rows.append({
                "model":          model_key,
                "prompt_id":      pid,
                "group":          PROMPT_GROUPS.get(pid, "unknown"),
                "hcds_baseline":  round(baseline_hcds, 4),
                "hcds_variant":   round(variant_hcds, 4),
                "sss":            round(sss, 4) if not np.isnan(sss) else float("nan"),
                "pass":           bool(sss < THRESHOLD) if not np.isnan(sss) else False,
            })

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    """Print a formatted console summary."""
    print("\n" + "="*80)
    print(f"  STYLE SHIFT SCORE (SSS)  —  threshold = {THRESHOLD*100:.0f}%")
    print("="*80)
    print(f"{'Model':<12} {'Prompt':<22} {'Group':<32} {'HCDS_base':>10} {'HCDS_var':>10} {'SSS':>7} {'Pass?':>6}")
    print("-"*80)
    for _, row in df.iterrows():
        flag = "✓" if row["pass"] else "✗"
        print(
            f"{row['model']:<12} {row['prompt_id']:<22} {row['group']:<32} "
            f"{row['hcds_baseline']:>10.4f} {row['hcds_variant']:>10.4f} "
            f"{row['sss']:>7.3f} {flag:>6}"
        )
    print("-"*80)

    # Overall result per model.
    for model, gdf in df.groupby("model"):
        n_total  = len(gdf)
        n_pass   = gdf["pass"].sum()
        all_pass = n_pass == n_total
        verdict  = "PASS ✓" if all_pass else "FAIL ✗"
        print(f"  {model}: {n_pass}/{n_total} variants below {THRESHOLD*100:.0f}% threshold → {verdict}")

    overall = df["pass"].all()
    print(f"\nOverall P1.5 criterion: {'PASSED ✓' if overall else 'FAILED ✗'}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",
                        default="style_controls/outputs/style_hcds_results.json")
    parser.add_argument("--out",
                        default="style_controls/tables/sss_table.csv")
    args = parser.parse_args()

    results_path = REPO_ROOT / args.results
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run compute_style_hcds.py first.")
        sys.exit(1)

    results = json.loads(results_path.read_text())
    df = compute_sss_table(results)

    print_summary(df)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nTable saved → {out_path}")


if __name__ == "__main__":
    main()
