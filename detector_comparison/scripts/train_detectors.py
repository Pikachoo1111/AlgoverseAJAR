"""
detector_comparison/scripts/train_detectors.py
-----------------------------------------------
Train and evaluate all 7 detectors from PM plan P2.1.

Detectors:
  1. entropy_only         LogisticRegression on [diff_entropy_mean, diff_entropy_slope]
  2. latency_only         LogisticRegression on [diff_latency_per_token]
  3. length_only          LogisticRegression on [diff_latency_per_token] as proxy
                          (swap for a real token-count column if available)
  4. mechanistic_only     LogisticRegression on [diff_mechanistic_sensitivity]
  5. logistic_all_6       LogisticRegression on all 6 diff features
  6. random_forest        RandomForestClassifier(100 trees) on all 6 diff features
  7. full_hcds            Rule-based: hcds_q > 0  (no training needed)

Evaluation: 5-fold stratified CV, seed 17.
Metrics: Accuracy, AUROC, F1, Calibration error (Brier score).

Also runs P2.2 incremental feature addition waterfall.

Usage (from repo root):
  python detector_comparison/scripts/train_detectors.py \\
      --instruct detector_comparison/features/feature_matrix_instruct_clean.csv \\
      --thinking detector_comparison/features/feature_matrix_thinking_clean.csv \\
      --out_dir  detector_comparison/tables/

Outputs:
  detector_comparison/tables/detector_comparison_instruct.csv
  detector_comparison/tables/detector_comparison_thinking.csv
  detector_comparison/tables/waterfall_instruct.csv
  detector_comparison/tables/waterfall_thinking.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.feature_extraction import FEATURE_NAMES

DIFF_COLS = [f"diff_{f}" for f in FEATURE_NAMES]
SEED = 17
N_SPLITS = 5
N_TREES = 100

# ── Feature subsets per detector ─────────────────────────────────────────────

def get_detector_specs(df: pd.DataFrame) -> list:
    """Return list of (name, X_array, model_or_tag) tuples.

    model_or_tag can be:
      - a sklearn estimator instance (will be fit/predict)
      - "hcds" (special: use hcds_q directly as score)
    """
    available = [c for c in DIFF_COLS if c in df.columns]
    X_all = df[available].values

    def col(fname):
        c = f"diff_{fname}"
        return df[[c]].values if c in df.columns else np.zeros((len(df), 1))

    specs = [
        ("entropy_only",
         np.hstack([col("entropy_mean"), col("entropy_slope")]),
         LogisticRegression(max_iter=1000, random_state=SEED)),

        ("latency_only",
         col("latency_per_token"),
         LogisticRegression(max_iter=1000, random_state=SEED)),

        ("length_only",
         col("latency_per_token"),   # proxy; swap for token-count diff if available
         LogisticRegression(max_iter=1000, random_state=SEED)),

        ("mechanistic_only",
         col("mechanistic_sensitivity"),
         LogisticRegression(max_iter=1000, random_state=SEED)),

        ("logistic_all_6",
         X_all,
         LogisticRegression(max_iter=1000, random_state=SEED)),

        ("random_forest",
         X_all,
         RandomForestClassifier(n_estimators=N_TREES, random_state=SEED)),

        ("full_hcds",
         df[["hcds_q"]].values,
         "hcds"),
    ]
    return specs


# ── Cross-validation evaluation ───────────────────────────────────────────────

def evaluate_detectors(
    df: pd.DataFrame,
    model_label: str,
) -> pd.DataFrame:
    """Run 5-fold CV for all detectors and return a results DataFrame."""
    y = df["y_label"].values.astype(int)
    specs = get_detector_specs(df)

    rows = []
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for name, X, estimator in specs:
        accs, aurocs, f1s, briers = [], [], [], []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # Replace NaN with 0 for training (already imputed in normalize step,
            # but guard here as well).
            X_tr = np.nan_to_num(X_tr, nan=0.0)
            X_te = np.nan_to_num(X_te, nan=0.0)

            if estimator == "hcds":
                # Rule: hcds_q > 0 → predict 1.
                scores = X_te[:, 0]        # raw HCDS_q
                preds  = (scores > 0).astype(int)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    estimator.fit(X_tr, y_tr)
                preds  = estimator.predict(X_te)
                scores = estimator.predict_proba(X_te)[:, 1]

            accs.append(  accuracy_score(y_te, preds))
            f1s.append(   f1_score(y_te, preds, zero_division=0))
            briers.append(brier_score_loss(y_te, scores))

            # AUROC requires both classes present in test fold.
            if len(np.unique(y_te)) == 2:
                aurocs.append(roc_auc_score(y_te, scores))
            else:
                aurocs.append(float("nan"))

        rows.append({
            "detector":          name,
            "model":             model_label,
            "accuracy":          round(float(np.mean(accs)),   4),
            "auroc":             round(float(np.nanmean(aurocs)), 4),
            "f1":                round(float(np.mean(f1s)),    4),
            "calibration_error": round(float(np.mean(briers)), 4),
            "accuracy_std":      round(float(np.std(accs)),    4),
            "auroc_std":         round(float(np.nanstd(aurocs)), 4),
        })

    results = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    return results


# ── P2.2 Incremental waterfall ────────────────────────────────────────────────

WATERFALL_ORDER = [
    ["diff_latency_per_token"],
    ["diff_latency_per_token", "diff_entropy_mean", "diff_entropy_slope"],
    ["diff_latency_per_token", "diff_entropy_mean", "diff_entropy_slope",
     "diff_paraphrase_consistency"],
    ["diff_latency_per_token", "diff_entropy_mean", "diff_entropy_slope",
     "diff_paraphrase_consistency", "diff_perturbation_sensitivity"],
    ["diff_latency_per_token", "diff_entropy_mean", "diff_entropy_slope",
     "diff_paraphrase_consistency", "diff_perturbation_sensitivity",
     "diff_mechanistic_sensitivity"],
]

WATERFALL_LABELS = [
    "latency",
    "+entropy",
    "+paraphrase",
    "+perturbation",
    "+mechanistic",
]


def waterfall_analysis(df: pd.DataFrame, model_label: str) -> pd.DataFrame:
    """Sequential feature addition — P2.2."""
    y = df["y_label"].values.astype(int)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rows = []

    for label, feature_set in zip(WATERFALL_LABELS, WATERFALL_ORDER):
        cols = [c for c in feature_set if c in df.columns]
        if not cols:
            continue
        X = np.nan_to_num(df[cols].values, nan=0.0)

        aurocs = []
        for train_idx, test_idx in kf.split(X, y):
            clf = LogisticRegression(max_iter=1000, random_state=SEED)
            clf.fit(X[train_idx], y[train_idx])
            scores = clf.predict_proba(X[test_idx])[:, 1]
            if len(np.unique(y[test_idx])) == 2:
                aurocs.append(roc_auc_score(y[test_idx], scores))

        rows.append({
            "step":    label,
            "model":   model_label,
            "features": ", ".join(c.replace("diff_", "") for c in cols),
            "auroc":    round(float(np.nanmean(aurocs)), 4) if aurocs else float("nan"),
            "auroc_std": round(float(np.nanstd(aurocs)), 4) if aurocs else float("nan"),
        })

    return pd.DataFrame(rows)


# ── Pretty-print table ────────────────────────────────────────────────────────

def print_results(df: pd.DataFrame, title: str) -> None:
    print(f"\n{'='*70}\n  {title}\n{'='*70}")
    print(
        f"{'Detector':<22} {'Acc':>7} {'AUROC':>7} {'F1':>7} {'Brier':>7}"
    )
    print("-"*70)
    for _, row in df.iterrows():
        print(
            f"{row['detector']:<22} "
            f"{row['accuracy']:>7.4f} "
            f"{row['auroc']:>7.4f} "
            f"{row['f1']:>7.4f} "
            f"{row['calibration_error']:>7.4f}"
        )
    print("="*70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruct",
                        default="detector_comparison/features/feature_matrix_instruct_clean.csv")
    parser.add_argument("--thinking",
                        default="detector_comparison/features/feature_matrix_thinking_clean.csv")
    parser.add_argument("--out_dir",
                        default="detector_comparison/tables")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_key, rel_path in [("instruct", args.instruct), ("thinking", args.thinking)]:
        path = REPO_ROOT / rel_path
        if not path.exists():
            print(f"File not found: {path} — skipping {model_key}.")
            continue

        df = pd.read_csv(path)
        # Drop rows with no label.
        df = df.dropna(subset=["y_label", "hcds_q"])
        print(f"\nLoaded {len(df)} rows for {model_key}.")

        # ── Detector comparison ──────────────────────────────────────────────
        results = evaluate_detectors(df, model_key)
        print_results(results, f"Detector Comparison — {model_key.upper()}")
        out_path = out_dir / f"detector_comparison_{model_key}.csv"
        results.to_csv(out_path, index=False)
        print(f"Saved → {out_path}")

        # ── Waterfall ────────────────────────────────────────────────────────
        wf = waterfall_analysis(df, model_key)
        print(f"\nWaterfall (P2.2) — {model_key.upper()}:")
        print(wf.to_string(index=False))
        wf_path = out_dir / f"waterfall_{model_key}.csv"
        wf.to_csv(wf_path, index=False)
        print(f"Saved → {wf_path}")

        # P2.1 success criterion check.
        top2 = results.head(2)["detector"].tolist()
        if "full_hcds" in top2:
            print(f"\n✓ SUCCESS: full_hcds is among the top-2 detectors for {model_key}.")
        else:
            print(f"\n⚠ Note: full_hcds not in top-2 for {model_key}. Top-2: {top2}")


if __name__ == "__main__":
    main()
