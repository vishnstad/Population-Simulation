"""
Weighted cluster statistics -- the `ClusterStats` contract of B17 §3.3 M2.

Everything here is survey-weighted. Unweighted cross-tabs are a bug, not a
shortcut: the spec makes weighting a hard contract ("all cross-tabs must use
`weight`").
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def kish_neff(w: np.ndarray) -> float:
    """Kish effective sample size: (sum w)^2 / sum(w^2).

    This is the number the n_eff >= 30 scoring gate in §5.2 refers to. With
    unequal weights it is strictly below the raw cell count, often well below,
    which is why gating on raw n would be too permissive.
    """
    w = np.asarray(w, float)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return 0.0
    s = w.sum()
    return float(s * s / np.square(w).sum())


def weighted_hist(codes: np.ndarray, weights: np.ndarray, scale_codes: list[int]) -> np.ndarray:
    """Weighted response histogram over a fixed, ordered code list.

    Returns a vector summing to 1, or all-NaN when the cell has no valid
    responses (never a silent zero vector -- downstream code must be able to
    tell 'no data' from 'nobody chose anything').
    """
    codes = np.asarray(codes, float)
    weights = np.asarray(weights, float)
    ok = np.isfinite(codes) & np.isfinite(weights) & (weights > 0)
    h = np.zeros(len(scale_codes), float)
    if not ok.any():
        return np.full(len(scale_codes), np.nan)
    idx = {c: i for i, c in enumerate(scale_codes)}
    for c, w in zip(codes[ok], weights[ok]):
        j = idx.get(int(c))
        if j is not None:
            h[j] += w
    tot = h.sum()
    if tot <= 0:
        return np.full(len(scale_codes), np.nan)
    return h / tot


def hist_moments(hist: np.ndarray, scale_codes: list[int]) -> tuple[float, float]:
    """Mean and SD of an ordinal histogram, on the raw code scale."""
    if hist is None or not np.isfinite(hist).all():
        return (np.nan, np.nan)
    x = np.asarray(scale_codes, float)
    m = float((hist * x).sum())
    v = float((hist * (x - m) ** 2).sum())
    return m, float(np.sqrt(max(v, 0.0)))


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance between two histograms. NaN-safe: cells with no data
    on either side are dropped before comparison, and a pair with no shared
    support returns 1.0 (maximally distant) rather than NaN, so the greedy
    merge in partition.py always has a usable number."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = np.isfinite(p) & np.isfinite(q)
    if not m.any():
        return 1.0
    p, q = p[m], q[m]
    sp, sq = p.sum(), q.sum()
    if sp <= 0 or sq <= 0:
        return 1.0
    p, q = p / sp, q / sq
    return float(np.sqrt(0.5 * np.square(np.sqrt(p) - np.sqrt(q)).sum()))


# ---------------------------------------------------------------------------
# ClusterStats builder
# ---------------------------------------------------------------------------

def build_cluster_stats(
    table: pd.DataFrame,
    cluster_col: str,
    items: dict[str, list[int]],
    *,
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Weighted histogram, n_eff, mean and SD for every (cluster x item) cell.

    `items` maps item_id -> ordered list of valid response codes (the scale).
    """
    w_all = table[weight_col].to_numpy(float)
    cids = pd.Index(sorted(table[cluster_col].dropna().unique()))
    cidx = pd.Series(np.arange(len(cids)), index=cids)
    ci = table[cluster_col].map(cidx).to_numpy(dtype="float64", na_value=np.nan)
    K = len(cids)

    frames = []
    for item, scale in items.items():
        y = table[f"item_{item}"].to_numpy(dtype="float64", na_value=np.nan)
        code_pos = {c: j for j, c in enumerate(scale)}
        C = len(scale)

        ok = np.isfinite(y) & np.isfinite(ci) & np.isfinite(w_all) & (w_all > 0)
        # map response codes to their slot on the scale; unmapped -> drop
        pos = np.full(y.shape, -1.0)
        for c, j in code_pos.items():
            pos[y == c] = j
        ok &= pos >= 0

        c_ok = ci[ok].astype(int)
        p_ok = pos[ok].astype(int)
        w_ok = w_all[ok]

        flat = c_ok * C + p_ok
        wsum = np.bincount(flat, weights=w_ok, minlength=K * C).reshape(K, C)
        tot = wsum.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            hist = wsum / tot[:, None]
        hist[tot <= 0] = np.nan

        n_raw = np.bincount(c_ok, minlength=K)
        w_c = np.bincount(c_ok, weights=w_ok, minlength=K)
        w2_c = np.bincount(c_ok, weights=w_ok ** 2, minlength=K)
        with np.errstate(invalid="ignore", divide="ignore"):
            n_eff = np.where(w2_c > 0, w_c ** 2 / w2_c, 0.0)

        x = np.asarray(scale, float)
        mean = (hist * x).sum(axis=1)
        sd = np.sqrt(np.clip((hist * (x - mean[:, None]) ** 2).sum(axis=1), 0, None))

        frames.append(pd.DataFrame({
            "cluster_id": cids.to_numpy(),
            "item_id": item,
            "hist": list(hist),
            "n_raw": n_raw.astype(int),
            "n_eff": n_eff,
            "w_sum": w_c,
            "mean": mean,
            "sd": sd,
        }))

    return pd.concat(frames, ignore_index=True)


def population_stats(
    table: pd.DataFrame,
    items: dict[str, list[int]],
    *,
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Level-0 (whole population) weighted marginals -- the B0a oracle
    baseline of §5.4 and the reference for the variance decomposition."""
    w = table[weight_col].to_numpy(float)
    recs = []
    for item, scale in items.items():
        col = table[f"item_{item}"].to_numpy(dtype="float64", na_value=np.nan)
        ok = np.isfinite(col)
        h = weighted_hist(col, w, scale)
        m, sd = hist_moments(h, scale)
        recs.append({
            "item_id": item, "hist": h.tolist(), "n_raw": int(ok.sum()),
            "n_eff": kish_neff(w[ok]), "mean": m, "sd": sd,
        })
    return pd.DataFrame.from_records(recs)
