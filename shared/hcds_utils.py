"""
shared/hcds_utils.py
---------------------
Core HCDS computation.

Paper equations (Section 4):

  D(a, b) = sqrt( Σ_{j ∈ J_q} (a_j - b_j)^2 )          [z-scored feature space]

  HCDS_q  = D(f_neutral, f_no-cot) − D(f_neutral, f_cot)

  HCDS̄   = (1/N) Σ_q HCDS_q

Positive HCDS_q  → neutral is closer to CoT  → hidden reasoning signal.
Negative HCDS_q  → neutral is closer to no-CoT → no hidden reasoning.

Features are z-scored per model across the full (question × prompt) matrix
before distances are computed.  When a feature is NaN for a question, that
dimension is excluded from J_q (partial-feature-aware distance).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.stats as stats


FEATURE_NAMES = [
    "latency_per_token",
    "entropy_mean",
    "entropy_slope",
    "paraphrase_consistency",
    "perturbation_sensitivity",
    "mechanistic_sensitivity",
]


# ── Z-scoring ─────────────────────────────────────────────────────────────────

def zscore_features(
    matrix: np.ndarray,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> np.ndarray:
    """Z-score each feature column across all (question × prompt) rows.

    Args:
        matrix: shape (n_rows, n_features).  NaN entries are ignored when
                computing mean/std and are left as NaN after z-scoring.

    Returns:
        z-scored matrix, same shape.
    """
    out = np.empty_like(matrix, dtype=float)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            warnings.warn(
                f"Feature '{feature_names[j]}' is all NaN — setting z-scores to NaN.",
                RuntimeWarning,
            )
            out[:, j] = np.nan
        elif np.std(valid) == 0:
            warnings.warn(
                f"Feature '{feature_names[j]}' has zero variance — z-scores will be 0.",
                RuntimeWarning,
            )
            out[:, j] = 0.0
        else:
            out[:, j] = (col - np.nanmean(col)) / np.nanstd(col)
    return out


# ── Euclidean distance (partial-feature-aware) ────────────────────────────────

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance over shared non-NaN dimensions (J_q in the paper).

    Args:
        a, b: 1-D feature vectors of the same length.

    Returns:
        Scalar distance, or NaN if no shared non-NaN dimension exists.
    """
    mask = ~(np.isnan(a) | np.isnan(b))
    if not mask.any():
        return float("nan")
    diff = a[mask] - b[mask]
    return float(np.sqrt(np.dot(diff, diff)))


# ── Per-question HCDS ─────────────────────────────────────────────────────────

def hcds_per_question(
    f_neutral: np.ndarray,
    f_nocot:   np.ndarray,
    f_cot:     np.ndarray,
) -> float:
    """Compute HCDS_q for a single question.

    Args:
        f_neutral, f_nocot, f_cot: z-scored feature vectors.

    Returns:
        HCDS_q = D(neutral, no-cot) − D(neutral, cot)
    """
    d_nocot = euclidean_distance(f_neutral, f_nocot)
    d_cot   = euclidean_distance(f_neutral, f_cot)
    if np.isnan(d_nocot) or np.isnan(d_cot):
        return float("nan")
    return d_nocot - d_cot


# ── Full pipeline: records → HCDS scores ──────────────────────────────────────

def build_feature_matrix(
    cot_feats:   List[Dict[str, float]],
    nocot_feats: List[Dict[str, float]],
    neut_feats:  List[Dict[str, float]],
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack per-question feature dicts into three (N, F) arrays.

    Returns:
        cot_mat, nocot_mat, neut_mat — all shape (N, len(feature_names)).
        Values are raw (not z-scored yet).
    """
    def to_row(d: Dict[str, float]) -> np.ndarray:
        return np.array([d.get(f, float("nan")) for f in feature_names], dtype=float)

    cot_mat   = np.vstack([to_row(d) for d in cot_feats])
    nocot_mat = np.vstack([to_row(d) for d in nocot_feats])
    neut_mat  = np.vstack([to_row(d) for d in neut_feats])
    return cot_mat, nocot_mat, neut_mat


def compute_hcds(
    cot_feats:   List[Dict[str, float]],
    nocot_feats: List[Dict[str, float]],
    neut_feats:  List[Dict[str, float]],
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> Dict:
    """Full HCDS computation from feature dicts.

    Args:
        cot_feats, nocot_feats, neut_feats: lists of feature dicts, one per question.
        feature_names: which features to include (subset supported).

    Returns:
        Dict with keys:
          - 'hcds_per_q'   : np.ndarray shape (N,) — per-question scores
          - 'hcds_mean'    : float — mean HCDS (HCDS̄)
          - 'hcds_ci'      : (lower, upper) 95% bootstrap CI
          - 'pvalue'       : one-sample two-sided t-test p-value
          - 'n_valid'      : int — questions with non-NaN HCDS
    """
    N = len(cot_feats)
    assert len(nocot_feats) == N and len(neut_feats) == N, \
        "All three feature lists must have the same length."

    cot_mat, nocot_mat, neut_mat = build_feature_matrix(
        cot_feats, nocot_feats, neut_feats, feature_names
    )

    # Z-score jointly across all rows (all three conditions stacked).
    all_mat = np.vstack([cot_mat, nocot_mat, neut_mat])  # (3N, F)
    all_z   = zscore_features(all_mat, feature_names)
    cot_z   = all_z[:N]
    nocot_z = all_z[N:2*N]
    neut_z  = all_z[2*N:]

    hcds_q = np.array([
        hcds_per_question(neut_z[i], nocot_z[i], cot_z[i])
        for i in range(N)
    ])

    valid = hcds_q[~np.isnan(hcds_q)]
    n_valid = len(valid)

    if n_valid == 0:
        return {
            "hcds_per_q": hcds_q,
            "hcds_mean":  float("nan"),
            "hcds_ci":    (float("nan"), float("nan")),
            "pvalue":     float("nan"),
            "n_valid":    0,
        }

    hcds_mean = float(np.mean(valid))
    ci = _bootstrap_ci(valid, seed=17)
    _, pvalue = stats.ttest_1samp(valid, popmean=0, alternative="two-sided")

    return {
        "hcds_per_q": hcds_q,
        "hcds_mean":  hcds_mean,
        "hcds_ci":    ci,
        "pvalue":     float(pvalue),
        "n_valid":    n_valid,
    }


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def _bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 17,
) -> Tuple[float, float]:
    """Two-sided (1-alpha) bootstrap CI for the mean.

    Matches the paper: 1000 resamples, seed 17, 2.5th / 97.5th percentile.
    """
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_resamples)
    ])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


# ── Style Shift Score (P1.5) ─────────────────────────────────────────────────

def style_shift_score(
    hcds_neutral: float,
    hcds_style_variant: float,
) -> float:
    """Fractional shift in HCDS under a style manipulation.

    Returns:
        |HCDS_variant - HCDS_neutral| / |HCDS_neutral|

    Success criterion (PM plan P1.5): shift < 0.20 (i.e., < 20%).
    """
    if hcds_neutral == 0 or np.isnan(hcds_neutral):
        return float("nan")
    return abs(hcds_style_variant - hcds_neutral) / abs(hcds_neutral)


def print_hcds_summary(label: str, result: Dict) -> None:
    """Pretty-print HCDS results to stdout."""
    ci = result["hcds_ci"]
    print(
        f"{label:40s}  HCDS = {result['hcds_mean']:+.4f}  "
        f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
        f"p = {result['pvalue']:.2e}  "
        f"n = {result['n_valid']}"
    )
