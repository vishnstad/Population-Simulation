"""Partial pooling toward the parent node (checklist 2.5, tau = 100).

This is the hierarchy doing statistical work rather than being an org chart.
Spec §2.2 is explicit that the tree earns its place through three jobs, and this
is the first: leaf clusters with thin data get shrunk toward their parent, which
is what makes K = 56 viable on a survey where a rotated item reaches only a
third of any wave.

    hist_pooled = lam * hist_leaf + (1 - lam) * hist_parent,
    lam = n_eff / (n_eff + tau)

A cell with n_eff = 100 sits halfway to its parent at tau = 100; a cell with
n_eff = 400 keeps 80% of its own shape. Cells with almost no data inherit the
parent's, which is the honest answer for them.

The parent histogram is computed from the parent node's own respondents — every
respondent under it, not an average of its children's histograms. Averaging
child histograms would weight a 60-person leaf equally with a 1,200-person one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .partition import ClusterTree
from .stats import ClusterStats

__all__ = ["pool_toward_parent", "shrinkage_weight"]


def shrinkage_weight(n_eff: float, tau: float = 100.0) -> float:
    """How much of its own shape a cell keeps."""
    if n_eff <= 0:
        return 0.0
    return float(n_eff / (n_eff + tau))


def pool_toward_parent(
    stats: ClusterStats,
    tree: ClusterTree,
    parent_stats: ClusterStats,
    *,
    tau: float = 100.0,
) -> ClusterStats:
    """Shrink each leaf histogram toward its parent node's histogram."""
    parent_lookup = {
        (r.cluster_id, r.item_id): np.asarray(r.hist, dtype=float)
        for r in parent_stats.frame.itertuples()
    }

    rows = []
    for r in stats.frame.itertuples():
        node = tree.nodes.get(r.cluster_id)
        parent_id = node.parent if node else None
        leaf = np.asarray(r.hist, dtype=float)
        lam = shrinkage_weight(float(r.n_eff), tau)
        parent_hist = parent_lookup.get((parent_id, r.item_id)) if parent_id else None

        if parent_hist is None or parent_hist.sum() <= 0:
            pooled, lam_used = leaf, 1.0
        else:
            pooled = lam * leaf + (1.0 - lam) * parent_hist
            lam_used = lam
        s = pooled.sum()
        pooled = pooled / s if s > 0 else pooled

        d = r._asdict()
        d.pop("Index", None)
        d["hist_unpooled"] = list(leaf)
        d["hist"] = pooled.tolist()
        d["shrinkage_lambda"] = lam_used
        d["parent_id"] = parent_id
        rows.append(d)

    out = pd.DataFrame(rows)
    notes = dict(stats.notes)
    notes.update({
        "tau_pooling": tau,
        "median_shrinkage_lambda": float(out.shrinkage_lambda.median()) if len(out) else 0.0,
        "cells_mostly_parent": int((out.shrinkage_lambda < 0.5).sum()) if len(out) else 0,
    })
    return ClusterStats(frame=out, codes=stats.codes, notes=notes)
