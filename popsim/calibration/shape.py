"""Histogram shape arithmetic for the calibration layer.

Everything downstream of elicitation manipulates ordinal histograms through
their **cumulative** form, for one reason: W1 on the normalized ordinal scale is
the sum of absolute CDF differences (see ``evalx.metrics.w1``). Working in CDF
space means the thing being adjusted and the thing being scored are the same
object, so an adjustment that looks small is small in the metric too.

Three operations, and nothing else:

``cdf`` / ``from_cdf``
    The k-1 interior cumulative values, and back. ``from_cdf`` projects onto the
    monotone non-decreasing range [0, 1] before differencing, so a perturbed CDF
    can never produce a negative bin. That projection is the only place a
    deviation is allowed to be clipped, and it is why the clipping is written
    once here rather than at each call site.

``temper``
    Power-tempering: ``h ** lam`` renormalized. ``lam > 1`` sharpens (lower SD),
    ``lam < 1`` flattens (higher SD). The spec calls for this in M5 step 2 and
    fixes lam by 1-D root-finding on the achieved SD, which is what ``temper_to_sd``
    does. It is the variance knob that does not move the mean much, which is what
    makes it usable after the level has already been set.

``sd_pos`` / ``mean_pos``
    Mean and SD of the histogram read as a distribution over option positions
    0, 1/(k-1), ..., 1. Same scale as W1, so "SD 0.45" and "W1 0.09" are
    commensurable numbers.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "cdf",
    "from_cdf",
    "mean_pos",
    "positions",
    "sd_pos",
    "temper",
    "temper_to_sd",
]


def positions(k: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, k) if k >= 2 else np.zeros(max(k, 1))


def _normed(h) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    s = h.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("histogram does not sum to anything positive")
    return h / s


def cdf(h) -> np.ndarray:
    """The k-1 interior cumulative values. The last one is 1 by construction."""
    return np.cumsum(_normed(h))[:-1]


def from_cdf(c) -> np.ndarray:
    """Back to a histogram, projecting onto monotone [0, 1] first."""
    c = np.maximum.accumulate(np.clip(np.asarray(c, dtype=float), 0.0, 1.0))
    h = np.clip(np.diff(np.concatenate([[0.0], c, [1.0]])), 0.0, None)
    s = h.sum()
    return h / s if s > 0 else np.full(h.size, 1.0 / h.size)


def mean_pos(h) -> float:
    p = _normed(h)
    return float(p @ positions(p.size))


def sd_pos(h) -> float:
    p = _normed(h)
    x = positions(p.size)
    m = p @ x
    return float(np.sqrt(max(float(p @ (x - m) ** 2), 0.0)))


def temper(h, lam: float) -> np.ndarray:
    """``h ** lam`` renormalized. lam > 1 sharpens, lam < 1 flattens."""
    p = _normed(h)
    if not np.isfinite(lam) or lam <= 0:
        return p
    # Scaled by the mode before powering: p ** 1000 underflows every bin to
    # zero for any p < 1, which silently returned the histogram unchanged and
    # made the sharp end of the bisection range indistinguishable from a no-op.
    base = np.clip(p, 1e-12, None)
    q = np.power(base / base.max(), float(lam))
    s = q.sum()
    return q / s if s > 0 and np.isfinite(s) else p


#: Log-spaced exponents the SD search evaluates. Fixed and checked in, so the
#: calibrated output of a run is reproducible to the last digit.
_LAM_GRID = np.unique(np.concatenate([
    np.geomspace(1e-3, 1.0, 61), [1.0], np.geomspace(1.0, 1e3, 61)
]))


def temper_to_sd(h, target_sd: float, *, tol: float = 1e-6, max_iter: int = 40) -> np.ndarray:
    """Find lam so that ``sd_pos(temper(h, lam))`` is as close to ``target_sd`` as
    power-tempering can get.

    SD is **not** monotone in lam, and assuming it is produces silent, large
    errors. lam -> 0 gives the uniform histogram, lam = 1 gives ``h``, lam -> inf
    gives a point mass at the mode — but a histogram piled at both ends of the
    scale has a higher SD than uniform, and sharpening a symmetric bimodal one
    *raises* its SD toward 0.5 instead of lowering it. The Layer 3 oracle gate
    caught both cases: a monotone bisection returned the uniform histogram for
    ``homosex`` and failed the round-trip by 0.31 W1.

    So the search is a fixed grid plus a bisection inside whichever adjacent pair
    brackets the target. Where no lam reaches the target — a symmetric bimodal
    histogram cannot be made narrow by tempering at all — the closest achievable
    is returned. That is a real limitation of the mechanism the spec chose (M5
    step 2), not a numerical failure, and the variance-ratio metric reports the
    SD actually achieved rather than the one asked for.
    """
    p = _normed(h)
    if p.size < 2 or not np.isfinite(target_sd) or target_sd <= 0:
        return p
    if abs(sd_pos(p) - target_sd) <= tol:
        return p

    sds = np.asarray([sd_pos(temper(p, lam)) for lam in _LAM_GRID])
    j = int(np.argmin(np.abs(sds - target_sd)))
    best_lam, best_err = float(_LAM_GRID[j]), float(abs(sds[j] - target_sd))
    if best_err <= tol:
        return temper(p, best_lam)

    for a in (j - 1, j):
        if a < 0 or a + 1 >= _LAM_GRID.size:
            continue
        lo, hi = float(_LAM_GRID[a]), float(_LAM_GRID[a + 1])
        s_lo, s_hi = float(sds[a]), float(sds[a + 1])
        if (s_lo - target_sd) * (s_hi - target_sd) > 0:
            continue
        for _ in range(max_iter):
            mid = float(np.sqrt(lo * hi))
            s = sd_pos(temper(p, mid))
            if abs(s - target_sd) < tol:
                return temper(p, mid)
            if (s_lo - target_sd) * (s - target_sd) <= 0:
                hi, s_hi = mid, s
            else:
                lo, s_lo = mid, s
        return temper(p, float(np.sqrt(lo * hi)))
    return temper(p, best_lam)
