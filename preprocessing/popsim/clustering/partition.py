"""
M2 -- constrained interpretable cross-partitioning by LATTICE COARSENING.

Implements B17 §3.3 M2. Deliberately NOT k-means / k-prototypes: the spec
rejects embedding clustering because a cluster that cannot be written down as
a conjunction of demographic constraints cannot be rendered into a stat card
or replicated by a reviewer.

    Level 0 : whole population
    Level 1 : region x urban   (coarsened the same way if a node is too thin)
    Level 2 : within each L1 node, the cross product of
              age_band x education x income_band x sex

WHY LATTICE COARSENING RATHER THAN GREEDY CELL MERGING
------------------------------------------------------
The obvious implementation of "merge any cell below min_cell into its nearest
sibling" destroys the thing that makes this partition worth having. Two
failure modes, both observed on GSS 2024:

  1. Widening each axis in turn eventually produces a cell meaning "anyone at
     all in this region". That is not an interpretable segment, and two such
     cells have byte-identical definitions -- they collide into one
     cluster_id and silently fuse in every downstream groupby.
  2. Restricting merges to cells differing on exactly one axis avoids (1) but
     strands almost everything: after a few merges no cell has a partner that
     matches on all remaining axes.

So we coarsen the LATTICE instead of the cells. Each axis carries an ordered
list of level-groups; a merge combines two level-groups **of a single axis**,
and the cells are always the full cross product of the current groups. This
guarantees:

  * every leaf is a conjunction of value-sets -- "women, 35-54, secondary or
    some-college, income q3-q5, urban South" -- readable by construction;
  * leaf definitions are mutually exclusive and exhaustive within the node,
    so ids cannot collide;
  * ordinal axes only ever merge ADJACENT levels, so no leaf spans
    "18-24 or 65+";
  * the procedure always makes progress and terminates.

The merge chosen at each step is the one that removes the most under-sized
cells, breaking ties by Hellinger distance between the two level-groups'
profile-item histograms (closest pair merges first).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from itertools import product

import numpy as np
import pandas as pd

from .stats import hellinger, kish_neff


# ---------------------------------------------------------------------------
# Axis metadata
# ---------------------------------------------------------------------------

# Ordinal axes may only merge ADJACENT levels, in this order.
ORDINAL_AXES: dict[str, list] = {
    "age_band": ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    "education": ["none", "primary", "secondary", "higher_secondary", "tertiary"],
    "income_band": ["q1", "q2", "q3", "q4", "q5"],
    "srcbelt": ["lg_central_city", "md_central_city", "lg_suburb", "md_suburb",
                "other_urban", "other_rural"],
}


def _axis_order(axis: str, present) -> list:
    o = ORDINAL_AXES.get(axis)
    if o:
        return [v for v in o if v in present] + \
               sorted((v for v in present if v not in o), key=str)
    return sorted(present, key=str)


# ---------------------------------------------------------------------------

@dataclass
class ClusterNode:
    cluster_id: str
    level: int
    parent: str | None
    definition: dict           # field -> value or sorted list of values
    n_raw: int
    n_eff: float
    n_weighted: float
    pop_share: float
    definition_text: str = ""
    below_min_cell: bool = False


_PRETTY = {
    "region": {"northeast": "the Northeast", "midwest": "the Midwest",
               "south": "the South", "west": "the West"},
    "urban": {True: "urban/suburban", False: "rural"},
    "sex": {"male": "men", "female": "women"},
    "education": {"none": "no formal schooling", "primary": "less than high school",
                  "secondary": "high-school educated",
                  "higher_secondary": "some college / associate degree",
                  "tertiary": "college graduates"},
    "income_band": {"q1": "bottom income quintile", "q2": "2nd income quintile",
                    "q3": "middle income quintile", "q4": "4th income quintile",
                    "q5": "top income quintile"},
}

_SHORT = {
    "education": {"none": "no schooling", "primary": "less than HS",
                  "secondary": "high school", "higher_secondary": "some college",
                  "tertiary": "college+"},
    "srcbelt": {"lg_central_city": "large city", "md_central_city": "mid city",
                "lg_suburb": "large suburb", "md_suburb": "mid suburb",
                "other_urban": "small town", "other_rural": "rural"},
}

_ALL = {
    "age_band": "any age", "education": "any education level",
    "income_band": "any income", "sex": "either sex",
    "region": "any region", "urban": "urban or rural",
}


def _pretty(axis: str, value, n_levels_total: int | None = None) -> str:
    vals = sorted(value, key=str) if isinstance(value, (list, tuple, set)) else [value]
    if n_levels_total and len(vals) == n_levels_total:
        return _ALL.get(axis, f"any {axis.replace('_', ' ')}")
    if len(vals) == 1:
        m = _PRETTY.get(axis)
        if m and vals[0] in m:
            return m[vals[0]]
        return f"aged {vals[0]}" if axis == "age_band" else str(vals[0])
    # ordinal groups are contiguous bands by construction -- render as a range
    order = ORDINAL_AXES.get(axis)
    if order:
        vals = sorted(vals, key=order.index)
        if axis == "age_band":
            lo = vals[0].split("-")[0]
            return f"aged {lo}+" if "+" in vals[-1] else \
                   f"aged {lo}-{vals[-1].split('-')[-1]}"
        if axis == "income_band":
            return f"income {vals[0]}–{vals[-1]}"
        short = _SHORT.get(axis, {})
        return f"{short.get(vals[0], vals[0])} to {short.get(vals[-1], vals[-1])}"
    m = _PRETTY.get(axis, {})
    return " or ".join(m.get(v, str(v)) for v in vals)


def _definition_text(definition: dict, totals: dict) -> str:
    order = ["sex", "age_band", "education", "income_band", "urban", "region"]
    bits = []
    for f in order:
        if f in definition:
            bits.append(_pretty(f, definition[f], totals.get(f)))
    for f in definition:
        if f not in order and not f.startswith("_"):
            bits.append(_pretty(f, definition[f], totals.get(f)))
    if definition.get("_note"):
        bits.append(f"[{definition['_note']}]")
    return ", ".join(bits) if bits else "the whole adult population"


# ---------------------------------------------------------------------------
# Profile machinery for the merge distance
# ---------------------------------------------------------------------------

class _Profiler:
    """Weighted response counts over the profile items, kept as one
    (n_items x max_codes) matrix so merges are matrix additions."""

    def __init__(self, table: pd.DataFrame, anchor_items: dict[str, list[int]],
                 weight: np.ndarray):
        self.items = list(anchor_items)
        self.scales = [anchor_items[i] for i in self.items]
        self.C = max((len(s) for s in self.scales), default=1)
        self.n_items = len(self.items)
        self.w = weight
        self.slots = np.full((self.n_items, len(table)), -1, dtype=np.int32)
        for k, (it, sc) in enumerate(zip(self.items, self.scales)):
            y = table[f"item_{it}"].to_numpy(dtype="float64", na_value=np.nan)
            for j, code in enumerate(sc):
                self.slots[k, y == code] = j

    def counts(self, rows) -> np.ndarray:
        m = np.zeros((self.n_items, self.C))
        rows = np.asarray(rows, dtype=int)
        if self.n_items == 0 or rows.size == 0:
            return m
        w = self.w[rows]
        for k in range(self.n_items):
            s = self.slots[k, rows]
            ok = s >= 0
            if ok.any():
                m[k] = np.bincount(s[ok], weights=w[ok], minlength=self.C)
        return m

    @staticmethod
    def distance(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0:
            return 0.0
        ds, used = 0.0, 0
        for k in range(a.shape[0]):
            if a[k].sum() > 0 and b[k].sum() > 0:
                ds += hellinger(a[k], b[k])
                used += 1
        return ds / used if used else 1.0


# ---------------------------------------------------------------------------
# The coarsening itself
# ---------------------------------------------------------------------------

def coarsen_axis(rows: np.ndarray, values: np.ndarray, order: list,
                 min_cell: int, prof: _Profiler, ordinal: bool) -> list[list]:
    """Group one axis's levels until every group holds >= min_cell rows.

    Ordinal axes merge ADJACENT groups only, so a group is always a contiguous
    band ("aged 35-54"), never "18-24 or 65+". Nominal axes merge the nearest
    pair by profile-item Hellinger distance.
    """
    present = [v for v in order if (values == v).any()]
    if not present:
        return []
    groups = [[v] for v in present]
    idx_of = {v: np.flatnonzero(values == v) for v in present}
    cnt = {v: len(idx_of[v]) for v in present}
    prof_of = {v: prof.counts(rows[idx_of[v]]) for v in present}

    def gsize(g):
        return sum(cnt[v] for v in g)

    def gprof(g):
        acc = np.zeros_like(prof_of[present[0]])
        for v in g:
            acc = acc + prof_of[v]
        return acc

    while len(groups) > 1 and min(gsize(g) for g in groups) < min_cell:
        i = min(range(len(groups)), key=lambda k: gsize(groups[k]))
        if ordinal:
            cand = [k for k in (i - 1, i + 1) if 0 <= k < len(groups)]
        else:
            cand = [k for k in range(len(groups)) if k != i]
        j = min(cand, key=lambda k: (prof.distance(gprof(groups[i]), gprof(groups[k])),
                                     gsize(groups[k])))
        lo, hi = min(i, j), max(i, j)
        fused = groups[lo] + groups[hi]
        groups[lo] = sorted(fused, key=order.index)
        del groups[hi]
    return groups


def recursive_partition(rows: np.ndarray, axes: list[str],
                        table: pd.DataFrame, min_cell: int,
                        prof: _Profiler) -> list[tuple[dict, np.ndarray]]:
    """Split `rows` by `axes` in order, refining only where the data supports it.

    A branch stops descending once it cannot be split into two parts that both
    clear min_cell; the remaining axes are then recorded as unconstrained
    ("any education level"). This is §2.2's adaptive refinement: dense branches
    end up finely cut, thin ones stay coarse, and every leaf is still a plain
    conjunction of demographic constraints.
    """
    if not axes:
        return [({}, rows)]
    a = axes[0]
    col = table[f"demo_{a}"].to_numpy()[rows]
    order = _axis_order(a, set(v for v in col if not pd.isna(v)))
    groups = coarsen_axis(rows, col, order, min_cell, prof, a in ORDINAL_AXES)

    if len(groups) <= 1:
        # axis cannot discriminate here -- carry it as unconstrained
        allv = order if order else []
        out = []
        for frag, r in recursive_partition(rows, axes[1:], table, min_cell, prof):
            out.append(({a: allv if len(allv) > 1 else (allv[0] if allv else None),
                         **frag}, r))
        return out

    out = []
    for g in groups:
        mask = np.isin(col, g)
        sub = rows[mask]
        for frag, r in recursive_partition(sub, axes[1:], table, min_cell, prof):
            out.append(({a: g if len(g) > 1 else g[0], **frag}, r))
    return out


def _coarsen(level_of: dict[str, np.ndarray], axes: list[str],
             levels: dict[str, list], rows: np.ndarray,
             min_cell: int, prof: _Profiler) -> tuple[dict, dict]:
    """Coarsen the axis lattice until every non-empty cell reaches min_cell.

    level_of[axis] is a per-respondent integer index into levels[axis].
    Returns (groups, cells) where groups[axis] is a list of lists of level
    values and cells maps a tuple of group indices -> row array.
    """
    # groups[axis] : ordered list of level-groups, each a list of level values
    groups = {a: [[v] for v in levels[a]] for a in axes}

    def gmap(a, g_axis):
        m = {}
        for gi, grp in enumerate(g_axis):
            for v in grp:
                m[levels[a].index(v)] = gi
        return m

    def build_cells(gs):
        # per-axis lookup vector: level index -> group index, applied by fancy
        # indexing so a trial merge costs one pass over the node, not a loop
        cols = []
        for a in axes:
            m = gmap(a, gs[a])
            lut = np.array([m[i] for i in range(len(levels[a]))], dtype=np.int64)
            cols.append(lut[np.asarray(level_of[a], dtype=np.int64)])
        keys = np.stack(cols, axis=1)
        cells: dict[tuple, list[int]] = {}
        for r, k in zip(rows.tolist(), map(tuple, keys.tolist())):
            cells.setdefault(k, []).append(int(r))
        return cells

    def merged(gs, a, i, j):
        """copy of gs with groups i and j of axis a fused (i < j)"""
        out = {ax: [list(g) for g in gs[ax]] for ax in axes}
        fused = out[a][i] + out[a][j]
        order = levels[a]
        out[a][i] = sorted(fused, key=order.index)
        del out[a][j]
        return out

    # profile counts per (axis, level), for the merge-distance tie-break
    lvl_counts = {
        a: [prof.counts(rows[np.asarray(level_of[a]) == i])
            for i in range(len(levels[a]))]
        for a in axes
    }

    def group_counts(a, grp):
        acc = np.zeros_like(lvl_counts[a][0])
        for v in grp:
            acc = acc + lvl_counts[a][levels[a].index(v)]
        return acc

    cells = build_cells(groups)
    while True:
        bad = sum(1 for v in cells.values() if len(v) < min_cell)
        if bad == 0 or all(len(groups[a]) == 1 for a in axes):
            break

        best = None
        for a in axes:
            g = groups[a]
            if len(g) < 2:
                continue
            # ordinal axes merge ADJACENT level-groups only
            pairs = ([(i, i + 1) for i in range(len(g) - 1)] if a in ORDINAL_AXES
                     else [(i, j) for i in range(len(g)) for j in range(i + 1, len(g))])
            for i, j in pairs:
                trial = merged(groups, a, i, j)
                tc = build_cells(trial)
                n_bad = sum(1 for v in tc.values() if len(v) < min_cell)
                dist = prof.distance(group_counts(a, g[i]), group_counts(a, g[j]))
                key = (n_bad, dist, a, i, j)
                if best is None or key[:2] < best[0][:2]:
                    best = (key, trial, tc)

        if best is None:
            break
        _, groups, cells = best

    return groups, cells


# ---------------------------------------------------------------------------

def build_cluster_tree(
    table: pd.DataFrame,
    *,
    l1_axes: list[str],
    l2_axes: list[str],
    anchor_items: dict[str, list[int]],
    min_cell: int = 40,
    weight_col: str = "weight",
    max_leaves: int | None = None,
) -> tuple[pd.Series, list[ClusterNode]]:
    """Returns (leaf assignment per respondent, list of ClusterNode)."""

    n_total = len(table)
    w = table[weight_col].to_numpy(float)
    w_total = float(w.sum())
    prof = _Profiler(table, anchor_items, w)

    totals = {a: table[f"demo_{a}"].dropna().nunique() for a in l1_axes + l2_axes}

    def _mk(cid, level, parent, definition, idx, below=False):
        idx = np.asarray(idx, dtype=int)
        ww = w[idx]
        return ClusterNode(
            cluster_id=cid, level=level, parent=parent, definition=definition,
            n_raw=len(idx), n_eff=kish_neff(ww), n_weighted=float(ww.sum()),
            pop_share=float(ww.sum() / w_total) if w_total else 0.0,
            definition_text=_definition_text(definition, totals),
            below_min_cell=below,
        )

    def _cid(prefix, defn, axis_list):
        parts = []
        for a in axis_list:
            v = defn[a]
            parts.append(",".join(map(str, sorted(v, key=str)))
                         if isinstance(v, list) else str(v))
        return prefix + ":".join(parts)

    nodes: list[ClusterNode] = [_mk("pop", 0, None, {}, np.arange(n_total))]
    assign = pd.Series(index=table.index, dtype="object")
    global_unclassified: list[int] = []

    # ---- level 1 ----------------------------------------------------------
    l1_complete = table[[f"demo_{a}" for a in l1_axes]].notna().all(axis=1).to_numpy()
    l1_rows = np.flatnonzero(l1_complete)
    l1_parts = recursive_partition(l1_rows, l1_axes, table, min_cell, prof)

    sub_complete = table[[f"demo_{a}" for a in l2_axes]].notna().all(axis=1).to_numpy()

    for defn1, idx in sorted(l1_parts, key=lambda t: str(t[0])):
        idx = np.asarray(idx)
        l1_id = _cid("l1:", defn1, l1_axes)
        nodes.append(_mk(l1_id, 1, "pop", dict(defn1), idx))

        # ---- level 2 within this L1 node ---------------------------------
        rows2 = idx[sub_complete[idx]]
        leftover = idx[~sub_complete[idx]].tolist()

        if len(rows2):
            for frag, rws in recursive_partition(rows2, l2_axes, table,
                                                 min_cell, prof):
                defn = {**defn1, **frag}
                leaf_id = _cid("gss:", defn, l1_axes + l2_axes)
                nodes.append(_mk(leaf_id, 2, l1_id, defn, np.asarray(rws),
                                 below=len(rws) < min_cell))
                assign.iloc[np.asarray(rws)] = leaf_id

        if leftover:
            # Missing at least one L2 axis -- overwhelmingly CONINC (income)
            # non-response. Keeps its region/urban identity when big enough.
            if len(leftover) >= min_cell:
                cid_u = l1_id + ":unclassified"
                defn = {**defn1, "_note": "missing >=1 level-2 partition axis"}
                nodes.append(_mk(cid_u, 2, l1_id, defn, np.asarray(leftover)))
                assign.iloc[leftover] = cid_u
            else:
                global_unclassified.extend(leftover)

    orphan = sorted(set(list(assign.isna().to_numpy().nonzero()[0]) + global_unclassified))
    if orphan:
        nodes.append(_mk("gss:unclassified", 2, "pop",
                         {"_note": "missing >=1 partition axis"},
                         np.asarray(orphan), below=len(orphan) < min_cell))
        assign.iloc[orphan] = "gss:unclassified"

    # ---- contract: leaf ids must be unique --------------------------------
    leaf_ids = [n.cluster_id for n in nodes if n.level == 2]
    if len(set(leaf_ids)) != len(leaf_ids):
        dup = [i for i in set(leaf_ids) if leaf_ids.count(i) > 1][:3]
        raise ValueError(f"duplicate leaf cluster_id(s): {dup}")

    if max_leaves and len(leaf_ids) > max_leaves:
        raise ValueError(
            f"{len(leaf_ids)} leaves exceeds max_leaves={max_leaves}; raise min_cell")

    return assign, nodes


def tree_to_json(nodes: list[ClusterNode]) -> str:
    return json.dumps([asdict(n) for n in nodes], indent=1, default=str)
