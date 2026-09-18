"""M2 — per-cluster sufficient statistics (checklist 2.2).

``ClusterStats`` is the ground truth for everything downstream: the histograms
the LLM is scored against, the anchor histograms that go into the stat card, and
the input to the noise floor.

Two things it records that the spec does not
--------------------------------------------
* **Kish effective sample size** per (cluster, item), not raw n. The bed pools
  seven waves with weights rescaled to equal per-wave mass, so weights vary
  enough that 200 respondents can carry the information of 140. Scoring a cell
  by its raw n would overstate how much truth is there.
* **Answered n per (cluster, item)**, because GSS ballot rotation means an item
  reaches only a third to a half of any wave. START_HERE is explicit: *reason
  about min_cell from per-item answered n, never sample n.* A 300-person cluster
  can hold 40 answers on a rotated item.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["ClusterStats", "compute_cluster_stats", "kish_n_eff", "weighted_histogram"]


def kish_n_eff(w: np.ndarray) -> float:
    """Kish's effective sample size: (sum w)^2 / sum(w^2)."""
    w = np.asarray(w, dtype=float)
    w = w[w > 0]
    if w.size == 0:
        return 0.0
    s = w.sum()
    return float(s * s / np.square(w).sum())


def weighted_histogram(values: np.ndarray, weights: np.ndarray, codes: list[int]) -> np.ndarray:
    """Weighted proportion in each code, in code order. Sums to 1 (or all zeros)."""
    out = np.zeros(len(codes), dtype=float)
    tot = 0.0
    for i, c in enumerate(codes):
        m = values == c
        out[i] = weights[m].sum()
        tot += out[i]
    return out / tot if tot > 0 else out


@dataclass
class ClusterStats:
    """Long-format table: one row per (cluster_id, item_id)."""

    frame: pd.DataFrame
    codes: dict[str, list[int]]
    notes: dict[str, Any] = field(default_factory=dict)

    def hist(self, cluster_id: str, item_id: str) -> np.ndarray | None:
        r = self.frame[(self.frame.cluster_id == cluster_id) & (self.frame.item_id == item_id)]
        return np.asarray(r.iloc[0]["hist"], dtype=float) if len(r) else None

    def for_item(self, item_id: str) -> pd.DataFrame:
        return self.frame[self.frame.item_id == item_id]

    def scorable(self, min_n_eff: float = 30.0) -> pd.DataFrame:
        """Cells where the truth is worth scoring against (spec §5.2)."""
        return self.frame[self.frame.n_eff >= min_n_eff]

    def to_parquet(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_parquet(p, index=False)
        return p


def compute_cluster_stats(
    frame: pd.DataFrame,
    cluster_of: pd.Series,
    items: list[str],
    codes: dict[str, list[int]],
    *,
    weight_col: str = "weight",
) -> ClusterStats:
    w_all = pd.to_numeric(frame[weight_col], errors="coerce").fillna(0.0).to_numpy()
    rows: list[dict[str, Any]] = []

    for cid, idx in frame.groupby(cluster_of.values, sort=True).groups.items():
        pos = frame.index.get_indexer(idx)
        for item in items:
            col = frame[f"item_{item}"].to_numpy()[pos]
            w = w_all[pos]
            cs = codes[item]
            answered = np.isin(col, cs)
            n_ans = int(answered.sum())
            if n_ans == 0:
                continue
            vals, ww = col[answered], w[answered]
            h = weighted_histogram(vals, ww, cs)
            # Mean and SD on the item's own normalized 0-1 scale, so a 7-point
            # item and a binary one are comparable.
            lo, hi = min(cs), max(cs)
            pos_scale = (np.asarray(cs, dtype=float) - lo) / (hi - lo)
            mean = float((h * pos_scale).sum())
            sd = float(np.sqrt(max(0.0, (h * (pos_scale - mean) ** 2).sum())))
            rows.append({
                "cluster_id": cid,
                "item_id": item,
                "hist": h.tolist(),
                "n_answered": n_ans,
                "n_eff": kish_n_eff(ww),
                "mean": mean,
                "sd": sd,
                "weight_sum": float(ww.sum()),
            })

    out = pd.DataFrame(rows)
    return ClusterStats(
        frame=out,
        codes={k: list(v) for k, v in codes.items()},
        notes={
            "n_clusters": int(cluster_of.nunique()),
            "n_items": len(items),
            "n_cells": len(out),
            "median_n_answered": float(out.n_answered.median()) if len(out) else 0.0,
            "median_n_eff": float(out.n_eff.median()) if len(out) else 0.0,
        },
    )
