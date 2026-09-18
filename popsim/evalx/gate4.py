"""Layer 3 — is the calibration layer arithmetically sound? *(gate)*

    "**The oracle pass-through.** Feed the *true* per-cluster histogram in where
     the LLM histogram goes. Calibrated output must come back ~= truth.

     *Catches:* off-by-one option ordering, misaligned scale codes, sign errors
     in the variance tempering, isotonic fits on the wrong axis. These masquerade
     as 'the model is bad at this item' and can burn weeks. Ten lines of code."

**Gate 4:** oracle round-trips within 0.005 W1.

Why this gate is unusually sharp on the level/structure design
--------------------------------------------------------------
The deviation reference is the population-weighted mean of the inputs' CDFs. Feed
the true per-cluster histograms in and that reference is, identically, the CDF of
the national marginal — because the national marginal *is* the weighted mixture
of the cluster histograms over the same respondents. So

    pred_cdf(c) = cdf(national) + s * ( cdf(truth_c) - cdf(national) )

is exactly ``cdf(truth_c)`` at s = 1 and exactly ``cdf(national)`` — B0a — at
s = 0. The gate therefore checks three things at once, all of which have an
analytic answer this code must reproduce:

1. fitting on oracle anchors must return **s = 1**, or the scale fit is broken;
2. applying at s = 1 must return the truth to numerical precision, or the option
   ordering, the scale codes or the CDF arithmetic is wrong;
3. applying at s = 0 must equal B0a exactly, or the deviation reference is not
   the mixture it is claimed to be.

A run where any of those drifts is a bug in this layer and not a property of any
model, which is the whole point of putting it before the verdict.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..calibration.fit import CalibrationPair, apply_calibrator, fit_calibrator
from .baselines import b0a_national_oracle
from .metrics import w1

__all__ = ["OracleResult", "run_oracle_passthrough"]


@dataclass
class OracleResult:
    tolerance: float
    fitted_scale: float
    scale_by_k: dict[int, float] = field(default_factory=dict)
    var_alpha: float = float("nan")
    var_beta: float = float("nan")
    max_w1_at_s1: float = float("nan")
    mean_w1_at_s1: float = float("nan")
    max_w1_s0_vs_b0a: float = float("nan")
    worst_cell: tuple[str, str] | None = None
    n_cells: int = 0
    n_items: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(
            np.isfinite(self.max_w1_at_s1)
            and self.max_w1_at_s1 <= self.tolerance
            and np.isfinite(self.max_w1_s0_vs_b0a)
            and self.max_w1_s0_vs_b0a <= 1e-9
            and abs(self.fitted_scale - 1.0) <= 0.02
        )

    def summary(self) -> str:
        return "\n".join([
            "Layer 3 — the oracle pass-through",
            f"  scope            {self.n_items} items x {self.n_cells} cells",
            f"  fitted scale s   {self.fitted_scale:.4f}   (must be 1.00 +/- 0.02)",
            f"  variance fit     sd_true = {self.var_alpha:+.5f} + {self.var_beta:.5f} * sd_pred",
            (f"  round-trip W1    mean {self.mean_w1_at_s1:.2e}   "
             f"max {self.max_w1_at_s1:.2e}   (tolerance {self.tolerance})"),
            f"  s = 0 vs B0a     max {self.max_w1_s0_vs_b0a:.2e}   (must be 0)",
            f"  worst cell       {self.worst_cell}",
            f"  => {'GREEN' if self.ok else 'RED'}",
        ])

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ok"] = self.ok
        return d


def run_oracle_passthrough(
    bed,
    *,
    anchor_items: list[str],
    target_items: list[str],
    cluster_ids: list[str],
    tolerance: float = 0.005,
    min_neff: float = 30.0,
    out_dir: str | Path | None = None,
) -> OracleResult:
    """Fit on oracle anchors, apply to oracle targets, and insist on identity."""
    pairs: list[CalibrationPair] = []
    national: dict[str, np.ndarray] = {}
    for item_id in anchor_items:
        for cid in cluster_ids:
            truth = bed.stats.hist(cid, item_id)
            if truth is None or truth.sum() <= 0 or bed.cell_neff(cid, item_id) < min_neff:
                continue
            t = [float(x) for x in np.asarray(truth, dtype=float)]
            pairs.append(CalibrationPair(
                cluster_id=cid, item_id=item_id, raw=t, truth=t,
                weight=bed.cell_weight(cid, item_id),
                n_eff=bed.cell_neff(cid, item_id)))
    if not pairs:
        raise ValueError("no oracle anchor cells — check min_neff and the cluster list")
    for item_id in {p.item_id for p in pairs}:
        ps = [p for p in pairs if p.item_id == item_id]
        w = np.asarray([p.weight for p in ps]); w = w / w.sum()
        national[item_id] = w @ np.vstack([np.asarray(p.truth, dtype=float) for p in ps])

    cal = fit_calibrator(pairs, national, topics=bed.topics,
                         variance_restoration=True, fit_per_cluster=False)

    worst = -1.0
    worst_cell = None
    vals: list[float] = []
    worst_b0a = 0.0
    n_cells = n_items = 0
    for item_id in target_items:
        cids = [c for c in cluster_ids
                if bed.stats.hist(c, item_id) is not None
                and bed.stats.hist(c, item_id).sum() > 0
                and bed.cell_neff(c, item_id) >= min_neff]
        if len(cids) < 3:
            continue
        n_items += 1
        truth = {c: np.asarray(bed.stats.hist(c, item_id), dtype=float) for c in cids}
        wt = {c: bed.cell_weight(c, item_id) for c in cids}
        nat = bed.level_hist(item_id, cids)
        got = apply_calibrator(cal, truth, nat, wt,
                               topic=bed.topics.get(item_id), scheme="unit")
        zero = apply_calibrator(cal, truth, nat, wt,
                                topic=bed.topics.get(item_id), scheme="zero")
        b0a = b0a_national_oracle(nat, cids)
        for c in cids:
            n_cells += 1
            d = w1(got[c], truth[c])
            vals.append(d)
            if d > worst:
                worst, worst_cell = d, (item_id, c)
            worst_b0a = max(worst_b0a, w1(zero[c], b0a[c]))

    res = OracleResult(
        tolerance=tolerance, fitted_scale=cal.scale_global,
        scale_by_k=dict(cal.scale_by_k), var_alpha=cal.var_alpha, var_beta=cal.var_beta,
        max_w1_at_s1=float(worst), mean_w1_at_s1=float(np.mean(vals)) if vals else float("nan"),
        max_w1_s0_vs_b0a=float(worst_b0a), worst_cell=worst_cell,
        n_cells=n_cells, n_items=n_items,
        notes={
            "anchor_items": len(national), "anchor_cells": len(pairs),
            "variance_restoration": "on — with oracle input the fitted line is the "
                                    "identity, so tempering is a no-op and any drift "
                                    "here is a sign error in temper_to_sd",
        },
    )
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "gate4_report.json").write_text(json.dumps(res.to_dict(), indent=2, default=str))
        (out / "gate4_summary.txt").write_text(res.summary())
    return res
