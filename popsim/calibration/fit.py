"""M5 — the calibration layer (checklist 4.1-4.6). **The core novel component.**

    "learn the map from raw LLM histograms to faithful cluster distributions,
     using anchors as supervision, and transfer it to targets."

What the map has to be, and why it is not the spec's
----------------------------------------------------
The spec's M5 is three steps applied to the raw histogram: isotonic on the CDF,
then a variance multiplier, then ensemble averaging. That design assumes the raw
histogram is roughly in the right place and needs its shape repaired.

Measured on 960 records from two model sizes (``scripts/forensics.py``,
``PREREGISTRATION.md`` §8), it is not. The error decomposes into two parts that
behave completely differently:

* **level** — where the model puts the item overall. Badly wrong and *shared
  across clusters*: on ``spkcom`` the 14B puts the group mean at 0.718 against a
  true 0.293; on ``homosex``, 0.241 against 0.591.
* **structure** — where the model puts each cluster relative to the others.
  Largely right: between-cluster Spearman +0.597 macro on the 14B, 8 of 10 items
  significant.

A calibration map fitted on anchors cannot repair the level, because the level
error is *per item* and a target item has no truth to fit against. Pooling
isotonic across items removes the average offset and leaves each item's own
offset untouched — which is why the naive design scores worse than copying the
national marginal.

So the layer separates the two rather than trying to fit through them:

    pred_cdf(c) = level_cdf  +  s · ( raw_cdf(c) − Σ_c w_c · raw_cdf(c) )

``level_cdf`` carries the item's level, ``s`` scales the model's subgroup
deviation, and ``s`` is the **only** thing fitted on anchors. One scalar per
scale length, fitted out of fold, is a quantity that transfers; a per-item
offset is not.

Two modes, both reported (§8.6)
-------------------------------
``observed_level``
    ``level_cdf`` is the true weighted national marginal. This is the
    small-area-estimation framing — a national topline is cheap, a
    subgroup-powered one is not — and it is the comparison that is *fair*,
    because the B0a baseline is handed exactly the same national marginal.
    ``s = 0`` reduces to B0a identically, so the deviation term is its own
    ablation.

``predicted_level``
    ``level_cdf`` comes from a population-level LLM call, isotonically corrected
    on the anchors' population-level pairs. This is what a genuinely unasked
    scenario question has to use, since it has no national marginal either. It is
    the number the M10 honesty box quotes, and it is never merged with the other.

What is deliberately *not* fitted
---------------------------------
No per-item ``s`` for targets: there is no target truth to fit it on, and
reaching for one is the same invalid move as picking an anchor by closeness to
the target's level (§5.3 of the pre-registration). Per-topic ``s`` is allowed
only where that topic has anchors, and the frozen split leaves religion, crime,
family and politics without any — those fall back to the global value, which is
a limitation of the split and is reported as one.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..evalx.metrics import w1
from .shape import cdf, from_cdf, sd_pos, temper_to_sd

__all__ = [
    "CalibrationPair",
    "Calibrator",
    "apply_calibrator",
    "deviation_reference",
    "fit_calibrator",
    "predict_cluster",
    "project_rows",
]

#: Partial-pooling weight for the per-scale-length scale: a scale length with
#: `m` anchor items gets `m/(m + tau)` of its own fit and the rest of the global
#: one. tau = 3 makes a single-anchor scale length a quarter its own.
SCALE_POOL_TAU = 3.0

#: Coarser grid for model selection, where the cost is (ranks x schemes x folds)
#: scale searches rather than one.
S_GRID_CV = np.round(np.arange(0.0, 2.401, 0.05), 3)

#: The grid ``s`` is searched on. W1 is piecewise linear in ``s`` before the
#: monotone projection and merely continuous after it, so a grid beats a
#: gradient method and is reproducible to the last digit.
S_GRID = np.round(np.arange(0.0, 3.001, 0.01), 3)


@dataclass
class CalibrationPair:
    """One cross-fitted (raw prediction, known truth) pair for an anchor cell."""

    cluster_id: str
    item_id: str
    raw: list[float]
    truth: list[float]
    weight: float
    n_eff: float

    @property
    def k(self) -> int:
        return len(self.truth)


@dataclass
class Calibrator:
    version: str = "b17-m5-v2-level-structure"
    mode: str = "observed_level"
    #: Deviation scale, per scale length k. The key is the number of options.
    scale_by_k: dict[int, float] = field(default_factory=dict)
    scale_global: float = 1.0
    #: How the scale is allowed to vary. Chosen out of fold on the anchors.
    scheme: str = "global"
    #: Per-topic scale where the topic has anchors to fit it on.
    scale_by_topic: dict[str, float] = field(default_factory=dict)
    #: Per-cluster scale, shrunk toward ``scale_global`` by n_eff/(n_eff+tau).
    #: The hierarchy doing partial pooling (spec §2.2). Ablation arm, not primary.
    scale_by_cluster: dict[str, float] = field(default_factory=dict)
    #: Within-cluster variance restoration, sd_true = alpha + beta * sd_pred.
    #: Fitted always; USED only if it improves out-of-fold anchor W1. On the
    #: level/structure design it usually does not — see `fit_calibrator`.
    var_alpha: float = 0.0
    var_beta: float = 1.0
    variance_restoration: bool = False
    #: The subgroup subspace: the top-r principal directions of the anchor
    #: deviation matrix, in `subspace_clusters` order. r is chosen on anchors.
    subspace_clusters: list[str] = field(default_factory=list)
    subspace_u: list[list[float]] = field(default_factory=list)
    subspace_rank: int = 0
    tau_pooling: float = 100.0
    #: SD of the out-of-fold anchor residual at each interior CDF position, per
    #: scale length. This is what makes an interval honest: the ensemble spread
    #: measures how much the model wobbles when asked again, which on three draws
    #: is small and is NOT the error that matters. The residual measures how far
    #: the finished prediction sat from the truth on questions where the truth is
    #: known, which is the error that matters, and it is the only part of it that
    #: can be estimated without a target's truth.
    resid_sd_by_k: dict[int, list[float]] = field(default_factory=dict)
    #: Isotonic map on population-level CDF values, for ``predicted_level`` mode.
    #: Stored as (x, y) knots so the calibrator round-trips through JSON.
    level_iso: dict[str, list[float]] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- scales
    def scale_for(self, k: int, *, topic: str | None = None,
                  cluster_id: str | None = None, scheme: str | None = None) -> float:
        scheme = scheme or self.scheme
        if scheme == "unit":
            return 1.0
        if scheme == "zero":
            return 0.0
        if scheme == "by_cluster" and cluster_id in self.scale_by_cluster:
            return self.scale_by_cluster[cluster_id]
        if scheme == "by_topic" and topic in self.scale_by_topic:
            return self.scale_by_topic[topic]
        return self.scale_by_k.get(k, self.scale_global)

    def project(self, dev_by_cluster: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Drop the part of the model's deviation that no real subgroup has.

        Across the anchor items whose per-cluster truths are observed, the
        cluster-level deviations do not point in arbitrary directions: they live
        in a low-dimensional subspace, and on this bed two directions already
        carry 68% of the variance. A deviation measured from three ensemble
        draws is mostly noise, and noise is isotropic — so the component outside
        that subspace cannot be subgroup structure and is discarded.

        The subspace is built from ANCHOR truths only. No target truth touches
        it, and the rank is chosen by out-of-fold anchor W1, so this is the
        hierarchy doing statistical work rather than a hyperparameter.

        Clusters absent from this item sit at zero deviation while the
        projection runs, which is the same thing as saying the only evidence
        about them is that they are average.
        """
        if not self.subspace_rank or not self.subspace_u:
            return dev_by_cluster
        U = np.asarray(self.subspace_u, dtype=float)[:, : self.subspace_rank]
        pos = {c: i for i, c in enumerate(self.subspace_clusters)}
        present = [c for c in dev_by_cluster if c in pos]
        if len(present) <= self.subspace_rank:
            # Fewer observations than basis directions: the projection is the
            # identity and claiming otherwise would just be a rename.
            return dev_by_cluster
        rows = np.asarray([pos[c] for c in present])
        D = np.vstack([np.asarray(dev_by_cluster[c], dtype=float) for c in present])
        P = project_rows(U, rows, D)
        out = dict(dev_by_cluster)
        for i, c in enumerate(present):
            out[c] = P[i]
        return out

    def residual_sd(self, k: int) -> np.ndarray | None:
        v = self.resid_sd_by_k.get(k) or self.resid_sd_by_k.get(str(k))
        return np.asarray(v, dtype=float) if v else None

    def apply_level_iso(self, c: np.ndarray) -> np.ndarray:
        x, y = self.level_iso.get("x"), self.level_iso.get("y")
        if not x or not y:
            return np.asarray(c, dtype=float)
        return np.interp(np.asarray(c, dtype=float), np.asarray(x), np.asarray(y))

    # ------------------------------------------------------------------- io
    def to_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["scale_by_k"] = {str(k): v for k, v in self.scale_by_k.items()}
        d["resid_sd_by_k"] = {str(k): v for k, v in self.resid_sd_by_k.items()}
        p.write_text(json.dumps(d, indent=2, default=float))
        return p

    @classmethod
    def from_json(cls, path: str | Path) -> Calibrator:
        d = json.loads(Path(path).read_text())
        d["scale_by_k"] = {int(k): float(v) for k, v in d.get("scale_by_k", {}).items()}
        d["resid_sd_by_k"] = {int(k): list(v)
                              for k, v in d.get("resid_sd_by_k", {}).items()}
        return cls(**d)

    def summary(self) -> str:
        by_k = ", ".join(f"k={k}: {v:.2f}" for k, v in sorted(self.scale_by_k.items()))
        return "\n".join([
            f"Calibrator {self.version}  mode={self.mode}",
            f"  deviation scale   global {self.scale_global:.2f}   [{by_k}]",
            (f"  variance restore  sd_true = {self.var_alpha:.4f} + "
             f"{self.var_beta:.4f} * sd_pred"
             + ("" if self.variance_restoration else "   (disabled)")),
            (f"  subspace          rank {self.subspace_rank} of "
             f"{len(self.subspace_clusters)} clusters"
             if self.subspace_rank else "  subspace          none (full space)"),
            ("  residual sd       "
             + ", ".join(f"k={k}: {np.mean(v):.4f}"
                         for k, v in sorted(self.resid_sd_by_k.items()))
             if self.resid_sd_by_k else "  residual sd       not estimated"),
            (f"  per-cluster s     {len(self.scale_by_cluster)} clusters, "
             f"tau={self.tau_pooling:g}" if self.scale_by_cluster else
             "  per-cluster s     not fitted"),
            (f"  level isotonic    {len(self.level_iso.get('x', []))} knots"
             if self.level_iso else "  level isotonic    none (observed level)"),
            *(f"  note  {k}: {v}" for k, v in self.notes.items()),
        ])


# ------------------------------------------------------------------- helpers

def project_rows(u: np.ndarray, rows: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Project ``d`` onto the subspace spanned by ``u``'s columns, restricted to ``rows``.

    The basis is built over all 56 leaves, but any given item is scored on the
    subset whose truth clears ``truth_min_neff`` — and a 16-cluster Gate-3-scope
    run uses fewer still. Zero-padding the absent clusters and projecting in the
    full space is wrong in a way that is easy to miss: a zero is not "no
    information", it is the assertion that those clusters sit exactly at the
    population mean, and the projection spends its budget honouring that claim.
    Measured cost on the 14B at 16 clusters: 0.0769 -> 0.0944 macro W1, i.e. the
    projection turned from a gain into a loss.

    The right object is the span of the basis vectors *restricted to the present
    rows*, which needs re-orthonormalising because a restriction of an
    orthonormal set is not orthonormal. QR does that.
    """
    up = np.asarray(u, dtype=float)[rows]
    if up.size == 0 or up.shape[1] == 0:
        return d
    q, _ = np.linalg.qr(up)
    # Drop directions the restriction collapsed: with fewer rows than columns,
    # QR returns a basis of the row space and the extra columns are noise.
    q = q[:, : min(q.shape[1], up.shape[0])]
    return q @ (q.T @ d)


def deviation_reference(raws: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """The population-weighted mean of the model's own CDFs for one item.

    Subtracting this is what makes the term a *deviation*: whatever constant
    offset the model applied to this item leaves with it, and what remains is
    only how the model moved each cluster relative to the others.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(w.size, 1.0 / w.size)
    return w @ np.vstack([cdf(r) for r in raws])


def predict_cluster(level_c: np.ndarray, dev: np.ndarray, s: float,
                    cal: Calibrator | None = None) -> np.ndarray:
    """One cluster's calibrated histogram from the level and its deviation."""
    h = from_cdf(np.asarray(level_c, dtype=float) + float(s) * np.asarray(dev, dtype=float))
    if cal is not None and cal.variance_restoration:
        target = cal.var_alpha + cal.var_beta * sd_pos(h)
        if np.isfinite(target) and target > 0:
            h = temper_to_sd(h, target)
    return h


def _truth_cdfs(d: dict) -> np.ndarray:
    if "_tc" not in d:
        d["_tc"] = np.vstack([cdf(t) for t in d["truth"]])
    return d["_tc"]


def _curve_for_item(d: dict, key: str, grid: np.ndarray) -> np.ndarray:
    """W1 per cell for every ``s`` at once: (len(grid), n_cells).

    The whole model-selection layer is nested grid searches — ranks x schemes x
    folds x |grid| — and the scalar version of this spent minutes on what is one
    broadcast. The monotone projection vectorises too: clip, then a cumulative
    maximum along the option axis, which is exactly what ``from_cdf`` does before
    differencing.
    """
    dev = np.asarray(d[key], dtype=float)               # (n, k-1)
    if dev.ndim == 1:
        dev = dev[None, :]
    cN = np.asarray(d["level_cdf"], dtype=float)[None, None, :]
    P = cN + grid[:, None, None] * dev[None, :, :]      # (S, n, k-1)
    np.clip(P, 0.0, 1.0, out=P)
    np.maximum.accumulate(P, axis=2, out=P)
    T = _truth_cdfs(d)[None, :, :]
    k = d["k"]
    return np.abs(P - T).sum(axis=2) / (k - 1)


def _score_scale(items: dict[str, dict], s: float, *, key: str = "dev",
                 temper_fn=None) -> float:
    """Weighted mean W1 over every pair, at one value of ``s``. Lower is better."""
    if temper_fn is None:
        num = den = 0.0
        g = np.asarray([float(s)])
        for d in items.values():
            w = np.asarray(d["w"], dtype=float)
            num += float(_curve_for_item(d, key, g)[0] @ w)
            den += float(w.sum())
        return num / den if den > 0 else float("nan")
    num = den = 0.0
    for d in items.values():
        cN = d["level_cdf"]
        for dev, truth, w in zip(d[key], d["truth"], d["w"], strict=True):
            num += w * w1(temper_fn(from_cdf(cN + s * dev)), truth)
            den += w
    return num / den if den > 0 else float("nan")


def _best_scale(items: dict[str, dict], *, key: str = "dev",
                temper_fn=None, grid=S_GRID) -> tuple[float, dict[float, float]]:
    grid = np.asarray(grid, dtype=float)
    if temper_fn is not None:
        curve = {float(x): _score_scale(items, float(x), key=key, temper_fn=temper_fn)
                 for x in grid}
    else:
        num = np.zeros(grid.size)
        den = 0.0
        for d in items.values():
            w = np.asarray(d["w"], dtype=float)
            num += _curve_for_item(d, key, grid) @ w
            den += float(w.sum())
        vals = num / den if den > 0 else np.full(grid.size, np.nan)
        curve = {float(x): float(v) for x, v in zip(grid, vals, strict=True)}
    finite = {k: v for k, v in curve.items() if np.isfinite(v)}
    if not finite:
        return 1.0, curve
    return min(finite, key=finite.get), curve


def _fit_scheme(items: dict[str, dict], key: str, scheme: str,
                grid=S_GRID_CV) -> dict:
    """Parameters for one scheme, fitted on the items given."""
    g, _ = _best_scale(items, key=key, grid=grid)
    out = {"scheme": scheme, "global": g}
    if scheme == "by_k":
        for k in sorted({d["k"] for d in items.values()}):
            sub = {i: d for i, d in items.items() if d["k"] == k}
            s_hat = _best_scale(sub, key=key, grid=grid)[0]
            lam = len(sub) / (len(sub) + SCALE_POOL_TAU)
            out.setdefault("by_k", {})[k] = lam * s_hat + (1 - lam) * out["global"]
    elif scheme == "by_topic":
        by: dict[str, dict] = {}
        for i, d in items.items():
            by.setdefault(d.get("topic", "other"), {})[i] = d
        for t, sub in by.items():
            if len(sub) >= 2:
                out.setdefault("by_topic", {})[t] = _best_scale(sub, key=key, grid=grid)[0]
    return out


def _scheme_scale(params: dict, d: dict, cid: str) -> float:
    if params["scheme"] == "by_k":
        return params.get("by_k", {}).get(d["k"], params["global"])
    if params["scheme"] == "by_topic":
        return params.get("by_topic", {}).get(d.get("topic", "other"), params["global"])
    return params["global"]


def _score_with(items: dict[str, dict], key: str, params: dict) -> float:
    num = den = 0.0
    for d in items.values():
        cN = d["level_cdf"]
        for cid, dev, truth, w in zip(d["cluster_ids"], d[key], d["truth"], d["w"],
                                      strict=True):
            num += w * w1(from_cdf(cN + _scheme_scale(params, d, cid) * dev), truth)
            den += w
    return num / den if den > 0 else float("nan")


def _cv_score(items: dict[str, dict], key: str, scheme: str,
              folds: dict[str, int]) -> float:
    """Out-of-fold anchor W1 for one (deviation, scheme) combination.

    In-sample anchor W1 cannot choose between a global scale and a per-scale-length
    one: the per-k fit has more parameters and always looks better, and on this
    split some scale lengths carry one or two anchors, so "better" is memorised.
    The cross-fit folds already exist for exactly this reason, so selection uses
    them: fit on two folds, score on the third.
    """
    fs = sorted({folds.get(i, 0) for i in items})
    num = den = 0.0
    for f in fs:
        tr = {i: d for i, d in items.items() if folds.get(i, 0) != f}
        te = {i: d for i, d in items.items() if folds.get(i, 0) == f}
        if len(tr) < 2 or not te:
            continue
        params = _fit_scheme(tr, key, scheme)
        v = _score_with(te, key, params)
        w = sum(float(np.sum(d["w"])) for d in te.values())
        if np.isfinite(v):
            num += v * w
            den += w
    return num / den if den > 0 else float("nan")


def _group(pairs: list[CalibrationPair], national: dict[str, np.ndarray],
           topics: dict[str, str] | None = None) -> dict[str, dict]:
    """Per-item deviation geometry, computed once and reused for every ``s``."""
    by_item: dict[str, list[CalibrationPair]] = {}
    for p in pairs:
        by_item.setdefault(p.item_id, []).append(p)

    out: dict[str, dict] = {}
    for item_id, ps in by_item.items():
        if len(ps) < 2 or item_id not in national:
            continue
        raws = [np.asarray(p.raw, dtype=float) for p in ps]
        w = np.asarray([p.weight for p in ps], dtype=float)
        ref = deviation_reference(raws, w)
        out[item_id] = {
            "cluster_ids": [p.cluster_id for p in ps],
            "level_cdf": cdf(national[item_id]),
            "dev": [cdf(r) - ref for r in raws],
            "truth": [np.asarray(p.truth, dtype=float) for p in ps],
            "w": w,
            "n_eff": np.asarray([p.n_eff for p in ps], dtype=float),
            "k": ps[0].k,
            "topic": (topics or {}).get(item_id, "other"),
        }
    return out


# ----------------------------------------------------------------------- fit

def fit_calibrator(
    pairs: list[CalibrationPair],
    national: dict[str, np.ndarray],
    *,
    topics: dict[str, str] | None = None,
    mode: str = "observed_level",
    variance_restoration: bool | None = None,
    fit_per_cluster: bool = True,
    tau: float = 100.0,
    level_pairs: list[tuple[np.ndarray, np.ndarray]] | None = None,
    basis: tuple[list[str], np.ndarray] | None = None,
    ranks: Sequence[int] = (0, 1, 2, 3, 4, 6, 8, 12),
    folds: dict[str, int] | None = None,
    # §8.5 of PREREGISTRATION.md, written before any of these numbers existed,
    # declares the candidate set: "One global `s` per scale-length is the primary
    # parameterisation", with per-cluster shrinkage as the hierarchical variant
    # and per-topic allowed "only where that topic has anchors". The frozen split
    # leaves religion, crime, family and politics with no anchors at all, so a
    # per-topic scale is fitted on the topics that can least afford it — it is
    # scored out of fold and reported, but it is not a candidate for the primary
    # arm. Adding it here after seeing a target number would be choosing the
    # parameterisation on results.
    schemes: Sequence[str] = ("global", "by_k"),
) -> Calibrator:
    """Fit the layer on cross-fitted anchor pairs. No target truth is touched.

    ``pairs`` are (raw prediction, known truth) for anchor cells, produced from
    cards built from folds other than the anchor's own — the condition targets
    face. ``national`` maps anchor item -> its true weighted national marginal.
    ``level_pairs`` are (population-level LLM prediction, true national marginal)
    for the anchors, used only in ``predicted_level`` mode.
    """
    items = _group(pairs, national, topics)
    if not items:
        raise ValueError("no usable anchor pairs — nothing to fit")

    # ---- the subgroup subspace, and its rank, chosen out of fold ---------
    folds = folds or {}
    sub_clusters: list[str] = []
    sub_u: np.ndarray | None = None
    best_rank = 0
    rank_curve: dict[int, float] = {}
    if basis is not None and basis[1].size:
        sub_clusters, A = list(basis[0]), np.asarray(basis[1], dtype=float)
        sub_u = np.linalg.svd(A, full_matrices=False)[0]
        pos = {c: i for i, c in enumerate(sub_clusters)}
        for r in ranks:
            if r > sub_u.shape[1]:
                continue
            key = f"dev_r{r}"
            for d in items.values():
                if r == 0:
                    d[key] = d["dev"]
                    continue
                U = sub_u[:, :r]
                present = [i for i, cid in enumerate(d["cluster_ids"]) if cid in pos]
                if len(present) <= r:
                    d[key] = d["dev"]
                    continue
                rows = np.asarray([pos[d["cluster_ids"][i]] for i in present])
                D = np.vstack([d["dev"][i] for i in present])
                P = project_rows(U, rows, D)
                out = list(d["dev"])
                for j, i in enumerate(present):
                    out[i] = P[j]
                d[key] = out
            rank_curve[r] = _cv_score(items, key, "global", folds)
        finite = {k: v for k, v in rank_curve.items() if np.isfinite(v)}
        if finite:
            best_rank = min(finite, key=finite.get)

    dev_key = f"dev_r{best_rank}" if best_rank else "dev"

    # ---- how the scale varies, also chosen out of fold -------------------
    scheme_curve = {sc: _cv_score(items, dev_key, sc, folds) for sc in schemes}
    finite = {k: v for k, v in scheme_curve.items() if np.isfinite(v)}
    best_scheme = min(finite, key=finite.get) if finite else "global"

    s_global, curve = _best_scale(items, key=dev_key)

    by_k: dict[int, float] = {}
    if best_scheme == "by_k":
        # Shrunk toward the global scale by how many anchor ITEMS carry that
        # scale length, with the same n/(n + tau) form the cluster-level pooling
        # uses. A scale length with one or two anchors cannot identify its own
        # scale, and the out-of-fold CV cannot protect against it because the
        # folds are over items, not within a scale length: at K = 18 the k = 2
        # subset fitted s = 2.62 on a handful of anchors and the targets paid
        # 0.0639 -> 0.0809 for it. This is a sample-size guard, applied
        # uniformly and before any target is scored.
        for k in sorted({d["k"] for d in items.values()}):
            sub = {i: d for i, d in items.items() if d["k"] == k}
            s_hat, _ = _best_scale(sub, key=dev_key)
            lam = len(sub) / (len(sub) + SCALE_POOL_TAU)
            by_k[k] = float(lam * s_hat + (1 - lam) * s_global)

    by_topic: dict[str, float] = {}
    if topics and best_scheme == "by_topic":
        seen: dict[str, dict] = {}
        for i, d in items.items():
            seen.setdefault(topics.get(i, "other"), {})[i] = d
        for t, sub in seen.items():
            # One anchor item cannot identify a scale; it identifies that item.
            if len(sub) >= 2:
                by_topic[t], _ = _best_scale(sub, key=dev_key)

    by_cluster: dict[str, float] = {}
    if fit_per_cluster:
        per: dict[str, dict] = {}
        neff: dict[str, float] = {}
        for i, d in items.items():
            for j, cid in enumerate(d["cluster_ids"]):
                e = per.setdefault(cid, {})
                e[i] = {"level_cdf": d["level_cdf"], "dev": [d[dev_key][j]],
                        "truth": [d["truth"][j]], "w": np.array([d["w"][j]]),
                        "k": d["k"], "topic": d.get("topic", "other")}
                neff[cid] = neff.get(cid, 0.0) + float(d["n_eff"][j])
        for cid, sub in per.items():
            if len(sub) < 3:
                continue
            s_hat, _ = _best_scale(sub)
            lam = neff.get(cid, 0.0) / (neff.get(cid, 0.0) + tau)
            by_cluster[cid] = float(lam * s_hat + (1 - lam) * s_global)

    # Within-cluster variance restoration, fitted at the chosen global scale on
    # the same pairs: sd_true = alpha + beta * sd_pred, weighted least squares.
    #
    # Fitted always, USED only if it lowers anchor W1. On the level/structure
    # design it usually does not: the prediction inherits the national
    # histogram's shape, so its within-cluster SD is already right (variance
    # ratio 1.000 measured on the 8B arm with no restoration at all). F1 —
    # within-cluster variance collapse, the failure this whole method family is
    # known for — does not arise when the shape is not generated token by token,
    # so the step that repairs it has nothing to repair and only distorts. That
    # is §5.8's "calibration kill: report and drop", decided on anchors.
    alpha, beta = 0.0, 1.0
    if variance_restoration is not False:
        xs, ys, ws = [], [], []
        for d in items.values():
            s = by_k.get(d["k"], s_global)
            for dev, truth, w in zip(d[dev_key], d["truth"], d["w"], strict=True):
                h = from_cdf(d["level_cdf"] + s * dev)
                xs.append(sd_pos(h)); ys.append(sd_pos(truth)); ws.append(w)
        x = np.asarray(xs); y = np.asarray(ys); w = np.asarray(ws)
        good = np.isfinite(x) & np.isfinite(y) & (w > 0)
        if good.sum() >= 10 and np.ptp(x[good]) > 1e-9:
            X = np.vstack([np.ones(good.sum()), x[good]]).T
            W = np.diag(w[good])
            coef = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y[good], rcond=None)[0]
            alpha, beta = float(coef[0]), float(coef[1])

    use_variance = bool(variance_restoration)
    w1_plain = _score_scale(items, s_global, key=dev_key)
    w1_tempered = float("nan")
    if variance_restoration is None:
        def _t(h, a=alpha, b=beta):
            tgt = a + b * sd_pos(h)
            return temper_to_sd(h, tgt) if np.isfinite(tgt) and tgt > 0 else h
        w1_tempered = _score_scale(items, s_global, key=dev_key, temper_fn=_t)
        use_variance = bool(np.isfinite(w1_tempered) and w1_tempered < w1_plain)

    # Out-of-fold anchor residuals, per scale length and CDF position.
    resid: dict[int, list[float]] = {}
    by_len: dict[int, list[np.ndarray]] = {}
    for d in items.values():
        k = d["k"]
        sc = by_k.get(k, s_global)
        cN = d["level_cdf"]
        for dev, truth in zip(d[dev_key], d["truth"], strict=True):
            pred = cdf(from_cdf(cN + sc * dev))
            by_len.setdefault(k, []).append(cdf(truth) - pred)
    for k, rows in by_len.items():
        if len(rows) >= 8:
            resid[k] = np.vstack(rows).std(axis=0).tolist()

    level_iso: dict[str, list[float]] = {}
    if mode == "predicted_level" and level_pairs:
        from sklearn.isotonic import IsotonicRegression
        xs = np.concatenate([cdf(p) for p, _ in level_pairs])
        ys = np.concatenate([cdf(t) for _, t in level_pairs])
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(xs, ys)
        grid = np.linspace(0.0, 1.0, 101)
        level_iso = {"x": grid.tolist(), "y": iso.predict(grid).tolist()}

    cal = Calibrator(
        mode=mode, scale_by_k=by_k, scale_global=float(s_global),
        scale_by_topic=by_topic, scale_by_cluster=by_cluster,
        scheme=best_scheme,
        var_alpha=alpha, var_beta=beta,
        variance_restoration=use_variance, tau_pooling=tau,
        level_iso=level_iso,
        subspace_clusters=sub_clusters,
        subspace_u=(sub_u[:, : max(best_rank, 1)].tolist()
                    if sub_u is not None and best_rank else []),
        subspace_rank=int(best_rank),
        resid_sd_by_k=resid,
    )
    cal.notes = {
        "n_pairs": len(pairs),
        "n_anchor_items": len(items),
        "anchor_w1_at_s0": round(curve[0.0], 5),
        "anchor_w1_at_s1": round(curve.get(1.0, float("nan")), 5),
        "anchor_w1_at_best": round(curve[s_global], 5),
        "topics_with_own_scale": sorted(by_topic),
        "subspace_rank_curve_oof": {str(k): round(float(v), 5)
                                    for k, v in rank_curve.items()},
        "scheme_curve_oof": {k: round(float(v), 5) for k, v in scheme_curve.items()},
        "scheme_chosen": best_scheme,
        "variance_restoration_anchor_w1": {
            "off": round(w1_plain, 5),
            "on": (None if not np.isfinite(w1_tempered) else round(w1_tempered, 5)),
            "chosen": "on" if use_variance else "off",
        },
    }
    return cal


# --------------------------------------------------------------------- apply

def apply_calibrator(
    cal: Calibrator,
    raw_by_cluster: dict[str, np.ndarray],
    level_hist: np.ndarray,
    weights: dict[str, float],
    *,
    topic: str | None = None,
    scheme: str | None = None,
) -> dict[str, np.ndarray]:
    """Calibrated histograms for every cluster of one item.

    ``level_hist`` is the item's level — the true national marginal in
    ``observed_level`` mode, the population-level LLM call in the other.
    """
    cids = [c for c in raw_by_cluster if raw_by_cluster[c] is not None]
    if not cids:
        return {}
    raws = [np.asarray(raw_by_cluster[c], dtype=float) for c in cids]
    w = np.asarray([float(weights.get(c, 0.0)) for c in cids], dtype=float)
    ref = deviation_reference(raws, w)
    level_c = cdf(level_hist)
    if cal.mode == "predicted_level":
        level_c = cal.apply_level_iso(level_c)
    k = len(level_hist)
    devs = cal.project({cid: cdf(r) - ref for cid, r in zip(cids, raws, strict=True)})
    out: dict[str, np.ndarray] = {}
    for cid in cids:
        s = cal.scale_for(k, topic=topic, cluster_id=cid, scheme=scheme)
        out[cid] = predict_cluster(level_c, devs[cid], s, cal)
    return out
