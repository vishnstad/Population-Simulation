"""
Segment scoping (module M6).

Turns a demographic query -- "the South", "urban women aged 25-44", "the whole
population" -- into a set of cluster weights that sum to one, so any downstream
aggregation can be run over an arbitrary slice of the population.

Why fractional membership
-------------------------
Cluster definitions are conjunctions over demographic axes, and after the
recursive coarsening in M2 a leaf may *span* several levels of an axis (a sparse
branch stops early and carries the rest of the axes unconstrained). So a leaf is
not simply in or out of a segment like "women aged 25-34": part of it is in.

Treating such a leaf as fully in over-counts the segment; dropping it under-counts.
Both distort the answer, and for coarse partitions the distortion is large.

Instead membership is computed exactly from the individual table:

    membership(leaf, query) = (weight of respondents in leaf satisfying query)
                              / (total weight of respondents in leaf)

The effective weight of a leaf in the segment is then its population weight times
its membership, renormalised across the segment. For a query on a level-1 axis
such as region this reduces to a clean 0/1 selection, which is the common case;
for anything finer it stays exact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SegmentQuery = Mapping[str, Union[str, bool, Sequence[str]]]

# Query keys -> individual-table columns.
FIELD_COLUMNS = {
    "region": "demo_region",
    "urban": "demo_urban",
    "srcbelt": "demo_srcbelt",
    "age_band": "demo_age_band",
    "sex": "demo_sex",
    "education": "demo_education",
    "income_band": "demo_income_band",
    "race": "demo_race",
    "religion": "demo_religion",
    "marital": "demo_marital",
    "employment": "demo_employment",
}


@dataclass
class ResolvedSegment:
    """A segment expressed as weights over leaf clusters."""

    query: Dict[str, Any]
    label: str
    cluster_ids: List[str]
    weights: np.ndarray          # normalised, sums to 1 over cluster_ids
    membership: Dict[str, float]  # per-cluster fraction of the leaf inside the segment
    pop_share: float              # share of the national population this segment covers
    n_respondents: int            # unweighted survey respondents behind it

    def __len__(self) -> int:
        return len(self.cluster_ids)

    def summary(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "query": self.query,
            "n_clusters": len(self.cluster_ids),
            "pop_share": round(self.pop_share, 4),
            "n_respondents": self.n_respondents,
            "n_partial_clusters": sum(1 for m in self.membership.values() if 0.0 < m < 0.999),
        }


class SegmentResolver:
    """
    Resolves segment queries against the individual table and cluster tree.

    Parameters
    ----------
    individual_table_path
        ``individual_table.parquet`` from preprocessing; carries per-respondent
        weights, demographics and leaf assignment.
    granularity
        ``coarse`` uses ``cluster_id_coarse``, ``fine`` uses ``cluster_id``.
    base_weights
        Optional cluster weights to start from -- pass the census-raked weights
        from :mod:`M6_aggregation.raking` to have segments inherit raking.
        Defaults to the survey population shares.
    """

    def __init__(
        self,
        individual_table_path: Path,
        granularity: str = "coarse",
        base_weights: Optional[Mapping[str, float]] = None,
    ):
        self.individuals = pd.read_parquet(individual_table_path)
        self.cluster_col = "cluster_id_coarse" if granularity == "coarse" else "cluster_id"
        if self.cluster_col not in self.individuals.columns:
            raise ValueError(
                f"{individual_table_path.name} has no column {self.cluster_col!r}; "
                "re-run preprocessing/run_pipeline.py"
            )

        self._leaf_weight = (
            self.individuals.groupby(self.cluster_col, observed=True)["weight"].sum()
        )
        self._leaf_n = self.individuals.groupby(self.cluster_col, observed=True).size()
        self._total_weight = float(self.individuals["weight"].sum())
        self.base_weights = dict(base_weights) if base_weights else None

    # ------------------------------------------------------------------
    def available_values(self, field: str) -> List[str]:
        """Distinct values of a queryable demographic field, for CLI help and the app."""
        col = FIELD_COLUMNS.get(field, field)
        if col not in self.individuals.columns:
            raise KeyError(
                f"unknown field {field!r}; queryable fields: {', '.join(sorted(FIELD_COLUMNS))}"
            )
        return sorted(str(v) for v in self.individuals[col].dropna().unique())

    def describe_query(self, query: SegmentQuery) -> str:
        if not query:
            return "the whole adult population"
        parts = []
        for field, value in query.items():
            vals = [value] if isinstance(value, (str, bool)) else list(value)
            parts.append(f"{field.replace('_', ' ')} = {'/'.join(str(v) for v in vals)}")
        return "; ".join(parts)

    # ------------------------------------------------------------------
    def _mask(self, query: SegmentQuery) -> pd.Series:
        mask = pd.Series(True, index=self.individuals.index)
        for field, value in query.items():
            col = FIELD_COLUMNS.get(field, field)
            if col not in self.individuals.columns:
                raise KeyError(
                    f"unknown segment field {field!r}; queryable fields: "
                    f"{', '.join(sorted(FIELD_COLUMNS))}"
                )
            wanted = {str(value)} if isinstance(value, (str, bool)) else {str(v) for v in value}
            mask &= self.individuals[col].astype(str).isin(wanted)
        return mask

    def resolve(
        self,
        query: Optional[SegmentQuery] = None,
        restrict_to: Optional[Sequence[str]] = None,
        min_membership: float = 1e-6,
    ) -> ResolvedSegment:
        """
        Resolve ``query`` into normalised cluster weights.

        ``restrict_to`` limits the result to clusters we actually hold predictions
        for, so a segment never silently omits mass by including a cluster the
        elicitation run skipped. The renormalisation happens after restriction and
        the retained population share is reported, so any omission is visible.
        """
        query = dict(query or {})
        mask = self._mask(query)
        sub = self.individuals[mask]

        seg_weight = sub.groupby(self.cluster_col, observed=True)["weight"].sum()
        seg_n = sub.groupby(self.cluster_col, observed=True).size()

        membership: Dict[str, float] = {}
        for cid, total in self._leaf_weight.items():
            inside = float(seg_weight.get(cid, 0.0))
            membership[str(cid)] = (inside / float(total)) if total > 0 else 0.0

        candidates = [c for c, m in membership.items() if m > min_membership]
        if restrict_to is not None:
            allowed = set(restrict_to)
            dropped = [c for c in candidates if c not in allowed]
            if dropped:
                lost = sum(float(self._leaf_weight.get(c, 0.0)) * membership[c] for c in dropped)
                logger.warning(
                    "%d in-segment clusters have no prediction; %.1f%% of the segment "
                    "is excluded from the estimate",
                    len(dropped), 100.0 * lost / max(float(seg_weight.sum()), 1e-9),
                )
            candidates = [c for c in candidates if c in allowed]

        if not candidates:
            raise ValueError(
                f"segment '{self.describe_query(query)}' matches no cluster with a prediction"
            )

        raw = []
        for cid in candidates:
            base = (
                self.base_weights.get(cid, 0.0)
                if self.base_weights is not None
                else float(self._leaf_weight.get(cid, 0.0)) / self._total_weight
            )
            raw.append(base * membership[cid])
        raw_arr = np.asarray(raw, dtype=float)
        if raw_arr.sum() <= 0:
            raise ValueError(f"segment '{self.describe_query(query)}' has zero weight")

        return ResolvedSegment(
            query=query,
            label=self.describe_query(query),
            cluster_ids=candidates,
            weights=raw_arr / raw_arr.sum(),
            membership={c: membership[c] for c in candidates},
            pop_share=float(seg_weight.sum() / self._total_weight),
            n_respondents=int(sum(int(seg_n.get(c, 0)) for c in candidates)),
        )

    # ------------------------------------------------------------------
    def observed_histogram(
        self, item_id: str, query: Optional[SegmentQuery] = None
    ) -> Optional[List[float]]:
        """
        The true weighted histogram for ``item_id`` over a segment, straight from
        microdata.

        This is the oracle router's answer for questions the survey already asks,
        and the ground truth the benchmark scores against. No LLM involved.
        """
        col = f"item_{item_id}"
        if col not in self.individuals.columns:
            return None
        sub = self.individuals[self._mask(dict(query or {}))]
        valid = sub[sub[col].notna() & (sub[col] >= 0)]
        if valid.empty:
            return None
        codes = valid[col].astype(int)
        weights = valid["weight"].astype(float)
        counts = weights.groupby(codes).sum()
        full = np.zeros(int(codes.max()) + 1, dtype=float)
        for code, w in counts.items():
            full[int(code)] = w
        full = full[1:] if full[0] == 0 and len(full) > 1 else full
        total = full.sum()
        return [float(x) for x in (full / total)] if total > 0 else None

    def segment_n_eff(self, query: Optional[SegmentQuery] = None) -> float:
        """Kish effective sample size for a segment -- how much the estimate can bear."""
        sub = self.individuals[self._mask(dict(query or {}))]
        w = sub["weight"].to_numpy(dtype=float)
        return float(w.sum() ** 2 / np.sum(w**2)) if len(w) and np.sum(w**2) > 0 else 0.0


def aggregate_segment(
    cluster_dists: Mapping[str, Sequence[float]],
    segment: ResolvedSegment,
) -> np.ndarray:
    """Mixture over a segment: sum_c w_c * hist_c, with w from the resolved segment."""
    weights, dists = [], []
    for cid, w in zip(segment.cluster_ids, segment.weights):
        hist = cluster_dists.get(cid)
        if hist is None:
            continue
        weights.append(w)
        dists.append(list(hist))
    if not dists:
        raise ValueError("no cluster distributions available for this segment")

    lengths = {len(d) for d in dists}
    if len(lengths) != 1:
        raise ValueError(f"cluster histograms disagree on scale length: {sorted(lengths)}")

    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    mixed = (np.asarray(dists, dtype=float) * w[:, None]).sum(axis=0)
    return mixed / mixed.sum()
