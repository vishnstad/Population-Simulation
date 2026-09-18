"""
Distributional evaluation metrics (module M9).

Means alone are banned here. Every headline number is a property of a predicted
*distribution* against a true one, because the claim being tested is about
distributional faithfulness, not about getting a subgroup average roughly right.

The metrics, and what each one kills if it fails
-----------------------------------------------
``w1``
    Wasserstein-1 on the normalised ordinal scale. The primary metric. Comparable
    with the silicon-sampling literature, which uses the same family.
``js``
    Jensen-Shannon divergence, for nominal scales where distance between adjacent
    codes is meaningless.
``variance_ratio``
    Predicted SD / true SD, within cluster. Persona sampling collapses this well
    below 1; the target band is [0.8, 1.2]. On its own it is gameable by
    predicting a uniform histogram, so it is never reported without W1.
``between_cluster_sd_ratio``
    Predicted between-cluster SD / true between-cluster SD. **The direct detector
    for between-cluster collapse.** If an RLHF-tuned model emits nearly the same
    histogram for every subgroup, W1 can still look acceptable while this sits
    near zero -- and the entire premise of cluster conditioning is dead. This is
    the number to watch in the early-warning run before committing a full budget.
``rank_correlation_rho``
    Spearman correlation between predicted and true cluster means. Does the model
    order subgroups correctly, even if the levels are off?
``ece``
    Expected calibration error over pooled (cluster x item x option) cells.
``coverage``
    Fraction of true values falling inside the predicted 90% intervals.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# pairwise distances
# ----------------------------------------------------------------------
def wasserstein_1_distance(
    p: Sequence[float], q: Sequence[float], support: Optional[np.ndarray] = None
) -> float:
    """
    Exact 1-Wasserstein distance between two discrete distributions on an ordinal
    scale normalised to [0, 1].

    W1 = sum_k |F_p(x_k) - F_q(x_k)| * (x_{k+1} - x_k)
    """
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    if len(p_arr) != len(q_arr):
        raise ValueError(f"scale length mismatch: {len(p_arr)} vs {len(q_arr)}")
    if support is None:
        support = np.linspace(0.0, 1.0, len(p_arr))
    cdf_p = np.cumsum(p_arr)[:-1]
    cdf_q = np.cumsum(q_arr)[:-1]
    return float(np.sum(np.abs(cdf_p - cdf_q) * np.diff(support)))


def jensen_shannon_div(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence in [0, 1] (base-2 distance, squared)."""
    p_arr = np.maximum(np.asarray(p, dtype=np.float64), 1e-12)
    q_arr = np.maximum(np.asarray(q, dtype=np.float64), 1e-12)
    p_arr /= p_arr.sum()
    q_arr /= q_arr.sum()
    return float(jensenshannon(p_arr, q_arr, base=2.0) ** 2)


def hellinger_distance(p: Sequence[float], q: Sequence[float]) -> float:
    """Hellinger distance -- used to rank which segments diverge most from the whole."""
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p_arr) - np.sqrt(q_arr)) ** 2)))


# ----------------------------------------------------------------------
# distribution moments
# ----------------------------------------------------------------------
def _support(n: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def compute_distribution_mean(probs: Sequence[float], support: Optional[np.ndarray] = None) -> float:
    p = np.asarray(probs, dtype=np.float64)
    s = _support(len(p)) if support is None else support
    return float(np.sum(p * s))


def compute_distribution_sd(probs: Sequence[float], support: Optional[np.ndarray] = None) -> float:
    p = np.asarray(probs, dtype=np.float64)
    s = _support(len(p)) if support is None else support
    mean = float(np.sum(p * s))
    var = float(np.sum(p * (s - mean) ** 2))
    return float(np.sqrt(max(var, 0.0)))


# ----------------------------------------------------------------------
# calibration and coverage
# ----------------------------------------------------------------------
def expected_calibration_error(
    predicted: Sequence[Sequence[float]],
    truth: Sequence[Sequence[float]],
    n_bins: int = 10,
) -> float:
    """
    ECE over pooled (cell x option) cells.

    Every predicted probability is binned; within a bin we compare the mean
    predicted probability with the mean true proportion. A well-calibrated system
    says 30% when the thing happens 30% of the time.
    """
    preds, trues = [], []
    for p, t in zip(predicted, truth):
        preds.extend(np.asarray(p, dtype=float).tolist())
        trues.extend(np.asarray(t, dtype=float).tolist())
    if not preds:
        return float("nan")

    preds_arr = np.asarray(preds)
    trues_arr = np.asarray(trues)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, total = 0.0, len(preds_arr)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (preds_arr >= lo) & (preds_arr < hi if hi < 1.0 else preds_arr <= hi)
        if not in_bin.any():
            continue
        ece += (in_bin.sum() / total) * abs(preds_arr[in_bin].mean() - trues_arr[in_bin].mean())
    return float(ece)


def interval_coverage(
    truth: Sequence[Sequence[float]],
    lo: Sequence[Sequence[float]],
    hi: Sequence[Sequence[float]],
) -> float:
    """Fraction of true option proportions inside their predicted interval."""
    inside = total = 0
    for t, l, h in zip(truth, lo, hi):
        for tv, lv, hv in zip(t, l, h):
            total += 1
            if lv <= tv <= hv:
                inside += 1
    return float(inside / total) if total else float("nan")


# ----------------------------------------------------------------------
# the headline evaluation
# ----------------------------------------------------------------------
def evaluate_cluster_predictions(
    predicted_dists: Sequence[Sequence[float]],
    true_dists: Sequence[Sequence[float]],
    cluster_weights: Optional[Sequence[float]] = None,
    scale_type: str = "ordinal",
) -> Dict[str, float]:
    """
    Score a set of per-cluster predictions for ONE item against the truth.

    ``cluster_weights`` should be population shares so the aggregate is
    population-weighted rather than treating a 0.2% cluster like a 12% one.
    """
    k = len(predicted_dists)
    if k == 0 or k != len(true_dists):
        raise ValueError(f"need matching non-empty prediction/truth lists, got {k} and {len(true_dists)}")

    if cluster_weights is None:
        w = np.ones(k) / k
    else:
        w = np.asarray(cluster_weights, dtype=np.float64)
        w = w / w.sum()

    w1s, jss, p_sds, t_sds, p_means, t_means = [], [], [], [], [], []
    for pred, true in zip(predicted_dists, true_dists):
        w1s.append(wasserstein_1_distance(pred, true))
        jss.append(jensen_shannon_div(pred, true))
        p_sds.append(compute_distribution_sd(pred))
        t_sds.append(compute_distribution_sd(true))
        p_means.append(compute_distribution_mean(pred))
        t_means.append(compute_distribution_mean(true))

    p_means_arr = np.asarray(p_means)
    t_means_arr = np.asarray(t_means)

    # Between-cluster spread: the weighted SD of cluster means. If the model emits
    # the same histogram everywhere this goes to zero while W1 may look fine.
    def _weighted_sd(x: np.ndarray) -> float:
        mu = float(np.sum(x * w))
        return float(np.sqrt(max(np.sum(w * (x - mu) ** 2), 0.0)))

    pred_between = _weighted_sd(p_means_arr)
    true_between = _weighted_sd(t_means_arr)

    if k > 2 and np.std(p_means_arr) > 1e-12 and np.std(t_means_arr) > 1e-12:
        rho, _ = spearmanr(p_means_arr, t_means_arr)
        rho_val = float(rho) if not np.isnan(rho) else 0.0
    else:
        rho_val = float("nan")

    mean_pred_sd = float(np.sum(np.asarray(p_sds) * w))
    mean_true_sd = float(np.sum(np.asarray(t_sds) * w))

    return {
        "weighted_w1": round(float(np.sum(np.asarray(w1s) * w)), 5),
        "macro_w1": round(float(np.mean(w1s)), 5),
        "weighted_js": round(float(np.sum(np.asarray(jss) * w)), 5),
        "variance_ratio": round(mean_pred_sd / max(mean_true_sd, 1e-9), 4),
        "pred_within_sd": round(mean_pred_sd, 4),
        "true_within_sd": round(mean_true_sd, 4),
        "between_cluster_sd_pred": round(pred_between, 5),
        "between_cluster_sd_true": round(true_between, 5),
        "between_cluster_sd_ratio": round(pred_between / max(true_between, 1e-9), 4),
        "rank_correlation_rho": round(rho_val, 4) if not np.isnan(rho_val) else float("nan"),
        "n_clusters": k,
    }


def aggregate_item_metrics(
    per_item: Sequence[Dict[str, float]],
    item_weights: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """
    Average per-item metric dicts into a headline row.

    Ratios are averaged rather than recomputed from summed numerators, so one
    high-variance item cannot dominate the reported ratio.
    """
    if not per_item:
        return {}
    w = (
        np.ones(len(per_item)) / len(per_item)
        if item_weights is None
        else np.asarray(item_weights, dtype=float) / np.sum(item_weights)
    )
    keys = [k for k in per_item[0] if k != "n_clusters"]
    out: Dict[str, float] = {}
    for key in keys:
        vals = np.asarray([m.get(key, np.nan) for m in per_item], dtype=float)
        ok = ~np.isnan(vals)
        out[key] = round(float(np.sum(vals[ok] * w[ok]) / np.sum(w[ok])), 5) if ok.any() else float("nan")
    out["n_items"] = len(per_item)
    return out


def improvement_vs_baseline(model_w1: float, baseline_w1: float) -> float:
    """Percent reduction in W1 relative to a baseline. Positive means better."""
    if baseline_w1 <= 0:
        return float("nan")
    return round(100.0 * (baseline_w1 - model_w1) / baseline_w1, 2)
