"""
detector_comparison/scripts/normalize_features.py
--------------------------------------------------
Re-check and report feature coverage, impute missing values, and save
a clean, analysis-ready version of the feature matrix.

This is a lightweight data-quality step between build_feature_table.py and
train_detectors.py.

Usage (from repo root):
  python detector_comparison/scripts/normalize_features.py \\
      --instruct detector_comparison/features/feature_matrix_instruct.csv \\
      --thinking detector_comparison/features/feature_matrix_thinking.csv \\
      --out_dir  detector_comparison/features/

Outputs:
  feature_matrix_instruct_clean.csv
  feature_matrix_thinking_clean.csv
  feature_coverage_report.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.feature_extraction import FEATURE_NAMES

DIFF_COLS = [f"diff_{f}" for f in FEATURE_NAMES]


def report_coverage(df: pd.DataFrame, model_label: str) -> str:
    """Return a text coverage report for the feature matrix."""
    lines = [f"\n── Feature coverage: {model_label} (N={len(df)}) ──"]
    for col in DIFF_COLS:
        if col not in df.columns:
            lines.append(f"  {col:<40s}  MISSING from CSV")
            continue
        n_nan = df[col].isna().sum()
        n_ok  = len(df) - n_nan
        pct   = 100 * n_ok / len(df)
        lines.append(f"  {col:<40s}  {n_ok:>4d}/{len(df)} ({pct:5.1f}%)")
    lines.append(f"  {'y_label':<40s}  {df['y_label'].notna().sum():>4d}/{len(df)} labels valid")
    return "\n".join(lines)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute NaN feature values with the column median.

    Rows where y_label is NaN are dropped (HCDS was undefined for those q's).
    """
    df = df.dropna(subset=["y_label"]).copy()
    df["y_label"] = df["y_label"].astype(int)

    for col in DIFF_COLS:
        if col not in df.columns:
            # Feature column missing entirely — add as zeros with NaN flag.
            df[col] = 0.0
            df[f"{col}_imputed"] = True
            continue
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            df[f"{col}_imputed"] = (df[col] == median_val) & (n_nan > 0)
        else:
            df[f"{col}_imputed"] = False

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruct",
                        default="detector_comparison/features/feature_matrix_instruct.csv")
    parser.add_argument("--thinking",
                        default="detector_comparison/features/feature_matrix_thinking.csv")
    parser.add_argument("--out_dir",
                        default="detector_comparison/features")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_lines = []

    for model_key, rel_path in [("instruct", args.instruct), ("thinking", args.thinking)]:
        path = REPO_ROOT / rel_path
        if not path.exists():
            print(f"File not found: {path} — skipping {model_key}.")
            continue

        df = pd.read_csv(path)
        report_lines.append(report_coverage(df, model_key))

        df_clean = impute_missing(df)
        out_path = out_dir / f"feature_matrix_{model_key}_clean.csv"
        df_clean.to_csv(out_path, index=False)
        print(f"  {model_key}: {len(df_clean)} clean rows → {out_path}")

    # Write coverage report.
    report_path = out_dir / "feature_coverage_report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"\nCoverage report → {report_path}")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
