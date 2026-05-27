"""
detector_comparison/scripts/plot_detector_results.py
------------------------------------------------------
Generate the two figures for the detector comparison section (P2.1 / P2.2):

  Figure A: Grouped bar chart comparing all 7 detectors across Instruct and
            Thinking on AUROC and Accuracy (2×2 panel).

  Figure B: Waterfall plot showing marginal AUROC gain from sequential
            feature addition (P2.2).

Usage (from repo root):
  python detector_comparison/scripts/plot_detector_results.py \\
      --instruct_det detector_comparison/tables/detector_comparison_instruct.csv \\
      --thinking_det detector_comparison/tables/detector_comparison_thinking.csv \\
      --instruct_wf  detector_comparison/tables/waterfall_instruct.csv \\
      --thinking_wf  detector_comparison/tables/waterfall_thinking.csv \\
      --out_dir      detector_comparison/plots/

Outputs:
  detector_comparison/plots/detector_comparison_figure.pdf
  detector_comparison/plots/waterfall_figure.pdf
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

DETECTOR_ORDER = [
    "full_hcds",
    "random_forest",
    "logistic_all_6",
    "entropy_only",
    "latency_only",
    "length_only",
    "mechanistic_only",
]

DETECTOR_LABELS = {
    "full_hcds":       "Full HCDS\n(rule)",
    "random_forest":   "Random\nForest",
    "logistic_all_6":  "Logistic\n(all 6)",
    "entropy_only":    "Entropy\nonly",
    "latency_only":    "Latency\nonly",
    "length_only":     "Length\nonly",
    "mechanistic_only":"Mechanistic\nonly",
}

MODEL_COLORS = {
    "instruct": "#1565C0",
    "thinking": "#2E7D32",
}


# ── Figure A: Detector comparison bar chart ───────────────────────────────────

def plot_detector_comparison(
    df_instruct: pd.DataFrame,
    df_thinking: pd.DataFrame,
    out_path: Path,
) -> None:
    """2×2 panel: top row = AUROC, bottom row = Accuracy; left = Instruct, right = Thinking."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    metrics = [("auroc", "AUROC"), ("accuracy", "Accuracy")]
    model_dfs = [("instruct", df_instruct), ("thinking", df_thinking)]

    for row, (metric, metric_label) in enumerate(metrics):
        for col, (model_key, df) in enumerate(model_dfs):
            ax = axes[row, col]
            if df is None or len(df) == 0:
                ax.text(0.5, 0.5, f"No data: {model_key}", ha="center",
                        va="center", transform=ax.transAxes)
                continue

            # Order bars by DETECTOR_ORDER; put any extras at the end.
            ordered = [d for d in DETECTOR_ORDER if d in df["detector"].values]
            extras  = [d for d in df["detector"].values if d not in ordered]
            ordered += extras

            vals  = []
            stds  = []
            xlabs = []
            for det in ordered:
                row_data = df[df["detector"] == det]
                if len(row_data) == 0:
                    continue
                vals.append(float(row_data[metric].iloc[0]))
                stds.append(float(row_data.get(f"{metric}_std", pd.Series([0])).iloc[0]))
                xlabs.append(DETECTOR_LABELS.get(det, det))

            xs = np.arange(len(vals))
            color = MODEL_COLORS[model_key]
            bars = ax.bar(xs, vals, width=0.6, color=color, alpha=0.8,
                          yerr=stds, error_kw={"capsize": 4, "linewidth": 1.2},
                          zorder=3)

            # Highlight full_hcds bar.
            hcds_idx = [i for i, d in enumerate(ordered) if d == "full_hcds"]
            for idx in hcds_idx:
                bars[idx].set_edgecolor("gold")
                bars[idx].set_linewidth(2.5)

            ax.set_xticks(xs)
            ax.set_xticklabels(xlabs, fontsize=7)
            ax.set_ylabel(metric_label, fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.set_title(
                f"Qwen3-4B-{model_key.capitalize()} — {metric_label}",
                fontsize=9, fontweight="bold"
            )
            ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
            ax.grid(axis="y", alpha=0.3, zorder=0)

            # Annotate values on bars.
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=6.5)

    fig.suptitle(
        "Detector Comparison (P2.1) — 5-fold CV\n"
        "Gold outline = Full HCDS (success if among top-2 by AUROC)",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure A saved → {out_path}")
    plt.close()


# ── Figure B: Waterfall / incremental feature gain ────────────────────────────

def plot_waterfall(
    df_instruct: pd.DataFrame,
    df_thinking: pd.DataFrame,
    out_path: Path,
) -> None:
    """Side-by-side waterfall AUROC for sequential feature addition (P2.2)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, (model_key, df) in zip(axes, [("instruct", df_instruct), ("thinking", df_thinking)]):
        if df is None or len(df) == 0:
            ax.text(0.5, 0.5, f"No data: {model_key}", ha="center",
                    va="center", transform=ax.transAxes)
            continue

        steps  = df["step"].tolist()
        aurocs = df["auroc"].tolist()
        stds   = df.get("auroc_std", pd.Series([0]*len(df))).tolist()
        xs = np.arange(len(steps))

        # Bars coloured by marginal gain.
        base = aurocs[0]
        gains = [aurocs[0]] + [aurocs[i] - aurocs[i-1] for i in range(1, len(aurocs))]
        cumsum = np.cumsum(gains)

        color = MODEL_COLORS[model_key]
        for i in range(len(xs)):
            gain = gains[i]
            bar_color = color if gain >= 0 else "#E53935"
            ax.bar(xs[i], aurocs[i], width=0.6, color=bar_color, alpha=0.8,
                   bottom=0, zorder=3,
                   yerr=[[stds[i]], [stds[i]]],
                   error_kw={"capsize": 4, "linewidth": 1.2})
            gain_str = f"+{gain:.3f}" if gain >= 0 else f"{gain:.3f}"
            ax.text(xs[i], aurocs[i] + 0.01, gain_str,
                    ha="center", va="bottom", fontsize=7.5,
                    color="green" if gain >= 0 else "red")

        ax.set_xticks(xs)
        ax.set_xticklabels(steps, fontsize=8)
        ax.set_ylabel("AUROC", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"Qwen3-4B-{model_key.capitalize()} — Incremental Feature Gain",
                     fontsize=9, fontweight="bold")
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle(
        "Waterfall Analysis (P2.2) — Sequential Feature Addition to Logistic Detector\n"
        "Annotations show marginal AUROC change at each step",
        fontsize=10,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure B saved → {out_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def load_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    print(f"  Not found: {path}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruct_det",
                        default="detector_comparison/tables/detector_comparison_instruct.csv")
    parser.add_argument("--thinking_det",
                        default="detector_comparison/tables/detector_comparison_thinking.csv")
    parser.add_argument("--instruct_wf",
                        default="detector_comparison/tables/waterfall_instruct.csv")
    parser.add_argument("--thinking_wf",
                        default="detector_comparison/tables/waterfall_thinking.csv")
    parser.add_argument("--out_dir",
                        default="detector_comparison/plots")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_inst_det  = load_csv(REPO_ROOT / args.instruct_det)
    df_think_det = load_csv(REPO_ROOT / args.thinking_det)
    df_inst_wf   = load_csv(REPO_ROOT / args.instruct_wf)
    df_think_wf  = load_csv(REPO_ROOT / args.thinking_wf)

    if df_inst_det is not None or df_think_det is not None:
        plot_detector_comparison(
            df_instruct=df_inst_det,
            df_thinking=df_think_det,
            out_path=out_dir / "detector_comparison_figure.pdf",
        )

    if df_inst_wf is not None or df_think_wf is not None:
        plot_waterfall(
            df_instruct=df_inst_wf,
            df_thinking=df_think_wf,
            out_path=out_dir / "waterfall_figure.pdf",
        )


if __name__ == "__main__":
    main()
