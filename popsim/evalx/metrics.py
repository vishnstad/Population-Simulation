"""Distributional metrics (spec §5.3). Means alone are banned."""

from __future__ import annotations

import numpy as np

__all__ = [
    "between_cluster_sd", "between_cluster_spearman", "cluster_means", "cluster_sds",
    "ece", "interval_coverage", "js_divergence", "normalized_positions",
    "ordering_spearman", "variance_ratio", "w1",
]


def normalized_positions(k: int) -> np.ndarray:
    """Option positions on the 0-1 scale, so scales of different length compare."""
    if k < 2:
        return np.zeros(max(k, 1))
    return np.linspace(0.0, 1.0, k)


def w1(p: np.ndarray, q: np.ndarray) -> float:
    """Wasserstein-1 between two histograms on the normalized ordinal scale.

    Santurkar et al. use the same family, which is what makes the numbers here
    comparable to published work. For an ordinal scale with k options mapped to
    0, 1/(k-1), ..., 1 the earth-mover distance is the area between the CDFs:

        W1 = (1 / (k - 1)) * sum_i |CDF_p(i) - CDF_q(i)|,  i = 0 .. k-2
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError(f"histogram shapes differ: {p.shape} vs {q.shape}")
    k = p.size
    if k < 2:
        return 0.0
    ps, qs = p.sum(), q.sum()
    if ps <= 0 or qs <= 0:
        return float("nan")
    cp = np.cumsum(p / ps)[:-1]
    cq = np.cumsum(q / qs)[:-1]
    return float(np.abs(cp - cq).sum() / (k - 1))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence, for nominal scales where order is meaningless."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def variance_ratio(pred_sd: float, true_sd: float) -> float:
    """Predicted SD / true SD. Bisbee-style collapse shows as a value well under 1."""
    return float("nan") if true_sd <= 0 else float(pred_sd / true_sd)


# --------------------------------------------------------------------------
# Between-cluster fidelity, calibration and coverage (spec §5.3).
#
# W1 alone cannot tell a system that gets every cluster right from one that gets
# the item's level right and every cluster identical — the second is F2 and is
# the way this whole method family fails quietly. These are the metrics that
# separate them, and §5.3's rule is that they are reported together, never
# a headline W1 on its own.
# --------------------------------------------------------------------------

def _mean_pos(h: np.ndarray) -> float:
    h = np.asarray(h, dtype=float)
    s = h.sum()
    if s <= 0:
        return float("nan")
    return float((h / s) @ normalized_positions(h.size))


def _sd_pos(h: np.ndarray) -> float:
    h = np.asarray(h, dtype=float)
    s = h.sum()
    if s <= 0:
        return float("nan")
    p = h / s
    x = normalized_positions(p.size)
    m = p @ x
    return float(np.sqrt(max(float(p @ (x - m) ** 2), 0.0)))


def cluster_means(hists) -> np.ndarray:
    return np.asarray([_mean_pos(h) for h in hists], dtype=float)


def cluster_sds(hists) -> np.ndarray:
    return np.asarray([_sd_pos(h) for h in hists], dtype=float)


def between_cluster_spearman(preds, truths) -> tuple[float, float]:
    """Does the system order the subgroups correctly? (rho, p).

    The direct F2 detector: a hedged model that emits one answer per item
    regardless of the card scores rho ~ 0 however good its W1 is.
    """
    from scipy.stats import spearmanr

    pm, tm = cluster_means(preds), cluster_means(truths)
    good = np.isfinite(pm) & np.isfinite(tm)
    if good.sum() < 3 or np.ptp(pm[good]) == 0 or np.ptp(tm[good]) == 0:
        return float("nan"), float("nan")
    rho, p = spearmanr(pm[good], tm[good])
    return float(rho), float(p)


def between_cluster_sd(hists, weights=None) -> float:
    """Weighted SD of the cluster means. Predicted vs true detects F2 in size."""
    m = cluster_means(hists)
    good = np.isfinite(m)
    if good.sum() < 2:
        return float("nan")
    m = m[good]
    if weights is None:
        return float(m.std())
    w = np.asarray(weights, dtype=float)[good]
    w = w / w.sum()
    mu = float(w @ m)
    return float(np.sqrt(max(float(w @ (m - mu) ** 2), 0.0)))


def ece(pred_cells, true_cells, weights=None, n_bins: int = 10) -> float:
    """Expected calibration error over (cluster x item x option) cells.

    Pool every predicted proportion against the empirical one it was predicting,
    bin by predicted value, and average |mean predicted - mean empirical| over
    bins weighted by bin mass. A system can have good W1 and still be
    systematically over-confident in the tails; this is where that shows.
    """
    p = np.asarray(pred_cells, dtype=float).ravel()
    q = np.asarray(true_cells, dtype=float).ravel()
    w = (np.ones_like(p) if weights is None
         else np.asarray(weights, dtype=float).ravel())
    good = np.isfinite(p) & np.isfinite(q) & (w > 0)
    p, q, w = p[good], q[good], w[good]
    if p.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    total = w.sum()
    out = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        wb = w[m].sum()
        out += (wb / total) * abs(float(np.average(p[m], weights=w[m]))
                                  - float(np.average(q[m], weights=w[m])))
    return float(out)


def interval_coverage(lo, hi, truth) -> float:
    """Fraction of true values inside their interval. Target = the nominal level."""
    lo = np.asarray(lo, dtype=float).ravel()
    hi = np.asarray(hi, dtype=float).ravel()
    t = np.asarray(truth, dtype=float).ravel()
    good = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(t)
    if good.sum() == 0:
        return float("nan")
    return float(((t[good] >= lo[good]) & (t[good] <= hi[good])).mean())


def ordering_spearman(pred_by_item: dict, true_by_item: dict) -> tuple[float, float]:
    """Checklist 5.2a — the tolerance battery's ordering, within one cluster.

        "A model that reproduces 'atheist > Communist > racist' tolerance
         ordering *within* each education band is doing something a national
         marginal cannot."

    Both dicts map item_id -> histogram for the SAME cluster. Returns the
    Spearman correlation between the predicted and true orderings of those items.
    """
    from scipy.stats import spearmanr

    keys = sorted(set(pred_by_item) & set(true_by_item))
    if len(keys) < 3:
        return float("nan"), float("nan")
    pm = np.asarray([_mean_pos(pred_by_item[k]) for k in keys])
    tm = np.asarray([_mean_pos(true_by_item[k]) for k in keys])
    good = np.isfinite(pm) & np.isfinite(tm)
    if good.sum() < 3 or np.ptp(pm[good]) == 0 or np.ptp(tm[good]) == 0:
        return float("nan"), float("nan")
    rho, p = spearmanr(pm[good], tm[good])
    return float(rho), float(p)
