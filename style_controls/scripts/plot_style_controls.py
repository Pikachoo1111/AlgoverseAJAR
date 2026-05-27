"""
style_controls/scripts/plot_style_controls.py
----------------------------------------------
Generate the Style Controls figure for the paper.

Creates two subplots (Instruct / Thinking) showing HCDS bars for:
  - The original neutral_strict baseline (grey)
  - 3 CoT-style / no-reasoning variants (blue family)
  - 3 no-CoT-style / reasoning-allowed variants (orange family)

The 20% threshold band is drawn around the baseline to show pass/fail visually.

Usage (from repo root):
  python style_controls/scripts/plot_style_controls.py \\
      --results style_controls/outputs/style_hcds_results.json \\
      --sss_table style_controls/tables/sss_table.csv \\
      --out style_controls/plots/style_controls_figure.pdf

Requires: matplotlib, pandas, numpy.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

THRESHOLD = 0.20

COLORS = {
    "neutral_strict":           "#888888",
    "cot_style_no_reasoning":   "#2196F3",   # blue
    "nocot_style_reasoning_allowed": "#FF9800",  # orange
}

PROMPT_GROUPS = {
    "neutral_strict":  "neutral_strict",
    "cot_style_1":     "cot_style_no_reasoning",
    "cot_style_2":     "cot_style_no_reasoning",
    "cot_style_3":     "cot_style_no_reasoning",
    "nocot_style_1":   "nocot_style_reasoning_allowed",
    "nocot_style_2":   "nocot_style_reasoning_allowed",
    "nocot_style_3":   "nocot_style_reasoning_allowed",
}

PROMPT_ORDER = [
    "neutral_strict",
    "cot_style_1", "cot_style_2", "cot_style_3",
    "nocot_style_1", "nocot_style_2", "nocot_style_3",
]

PRETTY_LABELS = {
    "neutral_strict":  "neutral\n(baseline)",
    "cot_style_1":     "CoT-style\n(no reason) 1",
    "cot_style_2":     "CoT-style\n(no reason) 2",
    "cot_style_3":     "CoT-style\n(no reason) 3",
    "nocot_style_1":   "no-CoT-style\n(reason ok) 1",
    "nocot_style_2":   "no-CoT-style\n(reason ok) 2",
    "nocot_style_3":   "no-CoT-style\n(reason ok) 3",
}


def plot_model(ax, model_results: dict, model_label: str):
    """Draw a single model's bar chart on ax."""
    baseline_hcds = model_results.get("neutral_strict", {}).get("hcds_mean", 0.0)
    threshold_band = abs(baseline_hcds) * THRESHOLD

    # Draw 20% threshold band.
    ax.axhspan(
        baseline_hcds - threshold_band,
        baseline_hcds + threshold_band,
        alpha=0.12, color="grey",
        label=f"±{THRESHOLD*100:.0f}% band",
    )
    ax.axhline(baseline_hcds, color="grey", linewidth=1.0, linestyle="--", alpha=0.7)

    xs, ys, cis, colors, labels = [], [], [], [], []

    for i, pid in enumerate(PROMPT_ORDER):
        if pid not in model_results:
            continue
        res = model_results[pid]
        if "hcds_mean" not in res:
            continue

        hcds_m = res["hcds_mean"]
        ci = res.get("hcds_ci", [hcds_m, hcds_m])
        err = [[hcds_m - ci[0]], [ci[1] - hcds_m]]

        group = PROMPT_GROUPS.get(pid, "unknown")
        color = COLORS.get(group, "#888888")

        xs.append(i)
        ys.append(hcds_m)
        cis.append(err)
        colors.append(color)
        labels.append(PRETTY_LABELS.get(pid, pid))

    for x, y, err, c in zip(xs, ys, cis, colors):
        ax.bar(x, y, width=0.6, color=c, alpha=0.8,
               yerr=[[abs(err[0][0])], [abs(err[1][0])]],
               error_kw={"ecolor": "black", "capsize": 4, "linewidth": 1.5},
               zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("HCDS", fontsize=9)
    ax.set_title(f"Qwen3-4B-{model_label}", fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3, zorder=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",
                        default="style_controls/outputs/style_hcds_results.json")
    parser.add_argument("--sss_table",
                        default="style_controls/tables/sss_table.csv")
    parser.add_argument("--out",
                        default="style_controls/plots/style_controls_figure.pdf")
    args = parser.parse_args()

    results_path = REPO_ROOT / args.results
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run compute_style_hcds.py first.")
        sys.exit(1)

    results = json.loads(results_path.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, (model_key, label) in zip(axes, [("instruct", "Instruct"), ("thinking", "Thinking")]):
        if model_key in results:
            plot_model(ax, results[model_key], label)
        else:
            ax.text(0.5, 0.5, f"No data for {model_key}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Qwen3-4B-{label}", fontsize=10)

    # Shared legend.
    legend_elements = [
        mpatches.Patch(color=COLORS["neutral_strict"],                 label="Baseline (neutral_strict)"),
        mpatches.Patch(color=COLORS["cot_style_no_reasoning"],         label="CoT-style wording, reasoning FORBIDDEN"),
        mpatches.Patch(color=COLORS["nocot_style_reasoning_allowed"],  label="no-CoT-style wording, reasoning ALLOWED"),
        mpatches.Patch(color="grey", alpha=0.3,                        label=f"±{THRESHOLD*100:.0f}% threshold band"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))

    fig.suptitle(
        "Style Control Experiment (P1.5): HCDS stability under prompt-style manipulation\n"
        "Bars within the grey band → shift < 20% (success criterion met)",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out_path}")
    plt.close()


if __name__ == "__main__":
    main()