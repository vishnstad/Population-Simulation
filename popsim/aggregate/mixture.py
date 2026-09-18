"""M6 — aggregation and uncertainty (checklist 5.3).

    "population distribution = sum_c w_c * hist_cal(c), with w_c = census-raked
     population shares — cluster weights are raked (iterative proportional
     fitting) to **external census margins** (ACS for US) so composition matches
     the real population, not the survey sample."

Why raking is not optional here
-------------------------------
The bed pools seven GSS waves. Its composition is the composition of *survey
respondents over 2010-2022*, not of US adults in 2024: education has risen
several points over that span and the age structure has moved. Rolling the
cluster answers up with survey weights therefore answers "what would the pooled
GSS sample say", which is not the question anyone asks.

The leaves are conjunctions of the three partition axes, so raking has something
to rake *to*: every leaf maps to a set of (age_band, degree, sex) cells, and ACS
2024 gives the population share of each. IPF finds the leaf weights whose implied
marginals match the ACS marginals on all three axes at once.

The margins must stay on the same category scale as the partition axes, which is
why `degree_scale_fingerprint` is in the config and checked here: a GSS adapter
that changed its education coding would make raking silently meaningless rather
than loudly wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["RakeResult", "aggregate", "bootstrap_population", "rake_weights"]


def _key(v) -> str:
    """One spelling for a category value, whatever dtype it arrived in.

    `degree` is an Int8 extension column, so `.to_numpy()` hands back floats and
    `astype(str)` turns 3 into "3.0", which matches nothing in a margins table
    that spells it 3. Raking then silently found no cells on the education axis —
    the one axis it exists to correct — and fell back to survey weights. So every
    comparison goes through here.
    """
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() else repr(float(v))
    return str(v)


@dataclass
class RakeResult:
    weights: dict[str, float]
    converged: bool
    n_iter: int
    max_margin_gap: float
    fallback: bool = False
    notes: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"raking: {'converged' if self.converged else 'DID NOT CONVERGE'} in "
            f"{self.n_iter} iterations, worst margin gap {self.max_margin_gap:.4f}"
            + ("  — fell back to survey weights" if self.fallback else "")
        )


def rake_weights(
    bed,
    margins: pd.DataFrame,
    *,
    axes: tuple[str, ...] = ("age_band", "degree", "sex"),
    max_iter: int = 200,
    tol: float = 1e-6,
) -> RakeResult:
    """Cluster weights whose implied marginals match the census margins.

    Iterative proportional fitting over one axis at a time. Each leaf carries the
    survey-weighted composition of its own members on each axis, so a leaf that
    spans two education bands contributes to both — the leaves are conjunctions
    of value *sets*, not single cells, and treating them as single cells is the
    quiet way raking stops meaning anything.
    """
    frame = bed.frame
    assign = bed.assign.astype("object").fillna("")
    w0 = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0).to_numpy()
    leaves = [c for c in bed.leaves_by_share]
    idx = {c: i for i, c in enumerate(leaves)}
    cl = assign.to_numpy()

    # composition[axis] is (n_leaves x n_levels), rows summing to 1
    comp: dict[str, tuple[list, np.ndarray]] = {}
    targets: dict[str, np.ndarray] = {}
    for axis in axes:
        if axis not in frame.columns or axis not in margins.columns:
            continue
        levels = sorted({_key(v) for v in margins[axis].dropna().unique().tolist()})
        M = np.zeros((len(leaves), len(levels)))
        col = np.asarray([_key(v) for v in frame[axis].to_numpy()], dtype=object)
        for j, lv in enumerate(levels):
            m = (col == lv)
            if not m.any():
                continue
            s = pd.Series(w0[m]).groupby(pd.Series(cl[m])).sum()
            for cid, v in s.items():
                if cid in idx:
                    M[idx[cid], j] += float(v)
        rows = M.sum(axis=1, keepdims=True)
        M = np.divide(M, rows, out=np.zeros_like(M), where=rows > 0)
        comp[axis] = (levels, M)
        mk = margins.assign(_k=[_key(v) for v in margins[axis]])
        t = mk.groupby("_k", observed=True)["pop"].sum().reindex(levels).fillna(0.0)
        tv = t.to_numpy(dtype=float)
        targets[axis] = tv / tv.sum() if tv.sum() > 0 else tv

    base = np.asarray([bed.cluster_weight.get(c, 0.0) for c in leaves], dtype=float)
    if base.sum() <= 0 or not comp:
        return RakeResult({c: float(v) for c, v in zip(leaves, base, strict=True)},
                          False, 0, float("nan"), fallback=True,
                          notes={"reason": "no usable margins or no survey weight"})
    w = base / base.sum()

    gap = float("nan")
    for it in range(1, max_iter + 1):
        gap = 0.0
        for axis, (_levels, M) in comp.items():
            got = w @ M
            want = targets[axis]
            ratio = np.divide(want, got, out=np.ones_like(want), where=got > 1e-12)
            w = w * (M @ ratio)
            s = w.sum()
            if s <= 0:
                return RakeResult(
                    {c: float(v) for c, v in zip(leaves, base / base.sum(), strict=True)},
                    False, it, float("nan"), fallback=True,
                    notes={"reason": "IPF collapsed to zero mass"})
            w = w / s
            gap = max(gap, float(np.abs(w @ M - want).max()))
        if gap < tol:
            return RakeResult({c: float(v) for c, v in zip(leaves, w, strict=True)},
                              True, it, gap,
                              notes={"axes": list(comp), "base": "survey weights"})
    # Capped, not failed: a sparse margin can oscillate in the last decimal for
    # ever without the weights being unusable. The gap is reported either way.
    return RakeResult({c: float(v) for c, v in zip(leaves, w, strict=True)},
                      False, max_iter, gap,
                      notes={"axes": list(comp),
                             "note": "iteration cap reached; weights returned with "
                                     "the achieved gap reported (spec §M6)"})


def aggregate(dists: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Population (or segment) histogram: the weighted mixture of cluster answers."""
    cids = [c for c in dists if weights.get(c, 0.0) > 0]
    if not cids:
        return np.asarray([])
    w = np.asarray([weights[c] for c in cids], dtype=float)
    w = w / w.sum()
    return w @ np.vstack([np.asarray(dists[c], dtype=float) for c in cids])


def bootstrap_population(
    dists_by_draw: list[dict[str, np.ndarray]],
    weights: dict[str, float],
    *,
    n_boot: int = 500,
    alpha: float = 0.10,
    seed: int = 17,
    resample_clusters: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """90 % intervals over (a) the elicitation ensemble and (b) cluster resampling.

    Both sources, as §M6 specifies, and they answer different questions: the
    ensemble captures how much the answer moves when the model is asked again,
    the cluster resample captures how much it moves when the partition happens to
    contain slightly different people.
    """
    rng = np.random.default_rng(seed)
    cids = [c for c in weights if weights[c] > 0 and any(c in d for d in dists_by_draw)]
    if not cids or not dists_by_draw:
        return np.asarray([]), np.asarray([]), np.asarray([])
    draws = []
    for _ in range(n_boot):
        d = dists_by_draw[rng.integers(0, len(dists_by_draw))]
        pick = (rng.choice(cids, size=len(cids), replace=True)
                if resample_clusters else cids)
        w = np.asarray([weights[c] for c in pick], dtype=float)
        hs = [np.asarray(d[c], dtype=float) for c in pick if c in d]
        if not hs:
            continue
        w = w[: len(hs)]
        draws.append((w / w.sum()) @ np.vstack(hs))
    if not draws:
        return np.asarray([]), np.asarray([]), np.asarray([])
    A = np.vstack(draws)
    return A.mean(axis=0), np.quantile(A, alpha / 2, axis=0), np.quantile(A, 1 - alpha / 2, axis=0)
