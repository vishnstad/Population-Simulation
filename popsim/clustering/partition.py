"""M2 — the cluster tree (checklist 2.1).

    "**Replace build-then-merge.** Partition targets K directly — supervised
     tree with a min-leaf constraint, splitting on anchor-response variance.
     *Why:* the spec merges on Hellinger distance between histograms built from
     1.6-person cells. Undefined, not a tuning issue."

The point of going top-down
---------------------------
The spec builds the full demographic cross first and then merges cells that came
out too small. That needs a distance between histograms estimated from a handful
of people, which is not a noisy measurement — it is undefined. A cell with 1.6
weighted respondents has no histogram.

Top-down never creates such a cell. Every split is checked *before* it is made:
if either side would fall under ``min_cell``, the split is not available. So the
merge criterion is not improved, it is eliminated — there is nothing to merge.

The split criterion
-------------------
At each node, consider every admissible split of every axis, and take the one
that most reduces within-group variance in *responses* — the standard
regression-tree criterion, applied to the pooled item matrix rather than to a
single outcome. Concretely, for each item the node's respondents are scored on
the item's normalized 0-1 scale, and a split's gain is the weighted drop in
total within-group variance summed over items, normalized per item so that a
7-point scale does not outvote a binary one.

Interpretability is a constraint, not a preference (spec §2.2 rejects embedding
clustering because reviewers cannot replicate an uninterpretable cluster). So
ordered axes (``age_band``, ``degree``) split only at contiguous cut points, and
every leaf is describable as a conjunction of ranges.

A note on what the partition is allowed to see
----------------------------------------------
Splitting on responses to items that later become *targets* would tune the
partition to the thing being predicted. With only three axes and 60 possible
cells the tree has very little freedom — the min-cell constraint does most of
the work — so the effect is small, but "small" is not "argued". ``split_items``
therefore exists to restrict the criterion, and
``compare_to_full_cross`` reports how far the supervised tree actually departs
from the plain demographic cross, so the claim can be checked rather than
asserted.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["ClusterNode", "ClusterTree", "build_cluster_tree", "compare_to_full_cross"]

#: Axes whose categories have an order, so splits must be contiguous cuts.
ORDERED_AXES: dict[str, list[str]] = {
    "age_band": ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    "age3": ["18-34", "35-54", "55+"],
    "degree": [0, 1, 2, 3, 4],          # GSS degree scale, ordered by attainment
    "edu3": ["no_college", "some_college", "degree"],
}


@dataclass
class ClusterNode:
    cluster_id: str
    level: int
    parent: str | None
    definition: dict[str, Any]          # axis -> value or list of values
    definition_text: str
    n: int                              # unweighted respondents
    n_weighted: float
    pop_share: float
    children: list[str] = field(default_factory=list)
    split_axis: str | None = None
    split_gain: float | None = None

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass
class ClusterTree:
    nodes: dict[str, ClusterNode]
    root: str
    axes: list[str]
    min_cell: int
    k_target: int
    split_items: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def leaves(self) -> list[ClusterNode]:
        return [n for n in self.nodes.values() if n.is_leaf]

    @property
    def k(self) -> int:
        return len(self.leaves())

    def parent_of(self, cluster_id: str) -> ClusterNode | None:
        p = self.nodes[cluster_id].parent
        return self.nodes[p] if p else None

    def assign(self, frame: pd.DataFrame) -> pd.Series:
        """Map each row to its leaf cluster_id."""
        out = pd.Series(pd.NA, index=frame.index, dtype="object")
        for leaf in self.leaves():
            mask = pd.Series(True, index=frame.index)
            for axis, allowed in leaf.definition.items():
                mask &= frame[axis].isin(allowed)
            out[mask & out.isna()] = leaf.cluster_id
        return out

    def to_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "root": self.root, "axes": self.axes, "min_cell": self.min_cell,
            "k_target": self.k_target, "k": self.k,
            "split_items": self.split_items, "notes": self.notes,
            "nodes": {k: asdict(v) for k, v in self.nodes.items()},
        }, indent=2, default=str))
        return p


def _normalized_scores(frame: pd.DataFrame, items: Sequence[str],
                       codes: dict[str, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Respondents x items, each item mapped to 0-1 on its own scale.

    Returns (scores, observed-mask). Missing entries are excluded per item
    rather than imputed: listwise deletion across 149 items would empty the
    table, and imputing an attitude to the mean invents the very between-group
    signal the split is supposed to find.
    """
    n, m = len(frame), len(items)
    scores = np.zeros((n, m), dtype=np.float64)
    seen = np.zeros((n, m), dtype=bool)
    for j, item in enumerate(items):
        col = pd.to_numeric(frame.get(f"item_{item}"), errors="coerce")
        if col is None:
            continue
        cs = codes.get(item) or []
        if len(cs) < 2:
            continue
        lo, hi = min(cs), max(cs)
        ok = col.notna() & (col >= lo) & (col <= hi)
        scores[ok.to_numpy(), j] = ((col[ok] - lo) / (hi - lo)).to_numpy()
        seen[ok.to_numpy(), j] = True
    return scores, seen


def _within_variance(scores: np.ndarray, seen: np.ndarray, w: np.ndarray) -> float:
    """Weighted within-group variance, averaged over items that were observed."""
    if not seen.any():
        return 0.0
    ww = w[:, None] * seen
    tot = ww.sum(axis=0)
    ok = tot > 0
    if not ok.any():
        return 0.0
    mean = (ww * scores).sum(axis=0)[ok] / tot[ok]
    var = (ww[:, ok] * (scores[:, ok] - mean) ** 2).sum(axis=0) / tot[ok]
    return float(np.nanmean(var))


def _candidate_splits(frame: pd.DataFrame, axis: str) -> Iterable[tuple[list, list]]:
    present = [v for v in frame[axis].dropna().unique()]
    if len(present) < 2:
        return
    order = ORDERED_AXES.get(axis)
    if order is not None:
        ordered = [v for v in order if v in present]
        for i in range(1, len(ordered)):
            yield ordered[:i], ordered[i:]
    else:
        # Unordered axis with few levels (sex): every one-vs-rest split.
        vals = sorted(present, key=str)
        if len(vals) == 2:
            yield [vals[0]], [vals[1]]
        else:
            for v in vals:
                yield [v], [x for x in vals if x != v]


def build_cluster_tree(
    frame: pd.DataFrame,
    *,
    axes: Sequence[str],
    item_codes: dict[str, list[int]],
    split_items: Sequence[str] | None = None,
    min_cell: int = 60,
    k_target: int = 56,
    weight_col: str = "weight",
) -> ClusterTree:
    """Grow the partition top-down, never creating a cell below ``min_cell``."""
    items = list(split_items if split_items is not None else item_codes.keys())
    scores, seen = _normalized_scores(frame, items, item_codes)
    w = pd.to_numeric(frame[weight_col], errors="coerce").fillna(0.0).to_numpy()
    total_w = float(w.sum())

    nodes: dict[str, ClusterNode] = {}
    root = ClusterNode(
        cluster_id="all", level=0, parent=None, definition={},
        definition_text="US adults 18+", n=len(frame), n_weighted=total_w, pop_share=1.0,
    )
    nodes[root.cluster_id] = root

    # (gain, node_id, axis, left_values, right_values, left_idx, right_idx)
    open_nodes: dict[str, np.ndarray] = {root.cluster_id: np.arange(len(frame))}

    def best_split(idx: np.ndarray):
        sub = frame.iloc[idx]
        base = _within_variance(scores[idx], seen[idx], w[idx])
        best = None
        for axis in axes:
            for left_vals, right_vals in _candidate_splits(sub, axis):
                lmask = sub[axis].isin(left_vals).to_numpy()
                li, ri = idx[lmask], idx[~lmask]
                if len(li) < min_cell or len(ri) < min_cell:
                    continue
                wl, wr = w[li].sum(), w[ri].sum()
                if wl <= 0 or wr <= 0:
                    continue
                after = (
                    wl * _within_variance(scores[li], seen[li], w[li])
                    + wr * _within_variance(scores[ri], seen[ri], w[ri])
                ) / (wl + wr)
                gain = base - after
                if best is None or gain > best[0]:
                    best = (gain, axis, list(left_vals), list(right_vals), li, ri)
        return best

    while len(open_nodes) + sum(1 for n in nodes.values() if n.is_leaf and n.cluster_id not in open_nodes) < k_target:
        scored = [(best_split(idx), nid) for nid, idx in open_nodes.items()]
        scored = [(b, nid) for b, nid in scored if b is not None]
        if not scored:
            break
        scored.sort(key=lambda t: -t[0][0])
        (gain, axis, left_vals, right_vals, li, ri), nid = scored[0]

        parent = nodes[nid]
        parent.split_axis, parent.split_gain = axis, float(gain)
        for side, vals, sub_idx in (("L", left_vals, li), ("R", right_vals, ri)):
            defn = dict(parent.definition)
            defn[axis] = vals
            cid = f"{nid}|{axis}={'_'.join(map(str, vals))}"
            child = ClusterNode(
                cluster_id=cid, level=parent.level + 1, parent=nid,
                definition=defn, definition_text=_describe(defn),
                n=len(sub_idx), n_weighted=float(w[sub_idx].sum()),
                pop_share=float(w[sub_idx].sum() / total_w),
            )
            nodes[cid] = child
            parent.children.append(cid)
            open_nodes[cid] = sub_idx
        del open_nodes[nid]

    full_cross = int(np.prod([frame[a].nunique() for a in axes]))
    return ClusterTree(
        nodes=nodes, root=root.cluster_id, axes=list(axes),
        min_cell=min_cell, k_target=k_target, split_items=items,
        notes={
            "n_respondents": len(frame),
            "full_cross_cells": full_cross,
            "min_leaf_n": int(min(n.n for n in nodes.values() if n.is_leaf)),
            "criterion": "weighted within-group variance on normalized item scores",
            "n_split_items": len(items),
        },
    )


def _describe(defn: dict[str, Any]) -> str:
    parts = []
    for axis, vals in defn.items():
        vals = list(vals)
        if len(vals) == 1:
            parts.append(f"{axis} {vals[0]}")
        else:
            parts.append(f"{axis} {vals[0]}..{vals[-1]}" if axis in ORDERED_AXES
                         else f"{axis} in {vals}")
    return ", ".join(parts) if parts else "US adults 18+"


def compare_to_full_cross(tree: ClusterTree, frame: pd.DataFrame) -> dict[str, Any]:
    """How far does the supervised tree depart from the plain demographic cross?

    If the answer is "barely", the worry that splitting on responses tunes the
    partition to the targets is bounded by measurement rather than argued away.
    Reported as the adjusted Rand index between the two assignments.
    """
    from sklearn.metrics import adjusted_rand_score

    assigned = tree.assign(frame)
    cross = frame[tree.axes].astype(str).agg("|".join, axis=1)
    ok = assigned.notna()
    return {
        "k_supervised": tree.k,
        "k_full_cross": int(cross[ok].nunique()),
        "adjusted_rand": float(adjusted_rand_score(cross[ok], assigned[ok].astype(str))),
        "n_assigned": int(ok.sum()),
    }
