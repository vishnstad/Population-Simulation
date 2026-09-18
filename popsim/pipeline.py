"""The bed, the partition, the split and the cards — built once, shared by every command.

``cmd_gate3`` grew its own copy of this sequence, and a second copy in Phase 4
would be the quiet way two commands start scoring against different partitions.
Everything here is deterministic given the config: the same YAML produces the
same clusters, the same folds and the same cards, byte for byte.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .agents.statcard import StatCard, battery_near_duplicates, make_statcard
from .calibration.crossfit import CrossfitPlan, make_crossfit_plan
from .clustering.partition import build_cluster_tree
from .clustering.stats import compute_cluster_stats, weighted_histogram
from .config import Config
from .data.adapters.gss import load_gss
from .data.pool import POST_FREEZE_DOCTRINE_FLAGS
from .evalx.split import load_effective_split

log = logging.getLogger(__name__)

__all__ = ["Bed", "build_bed"]


@dataclass
class Bed:
    cfg: Config
    codebook: dict[str, dict]
    split: dict[str, Any]
    amendments: list[dict]
    table: Any
    tree: Any
    assign: pd.Series
    stats: Any
    plan: CrossfitPlan
    near_dupes: dict[str, set[str]]
    notes: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- shortcuts
    @property
    def frame(self) -> pd.DataFrame:
        return self.table.frame

    @cached_property
    def codes(self) -> dict[str, list[int]]:
        return {i: self.codebook[i]["codes"] for i in self.codebook}

    @cached_property
    def topics(self) -> dict[str, str]:
        return {i: self.codebook[i].get("topic", "other") for i in self.codebook}

    @cached_property
    def weights(self) -> np.ndarray:
        return pd.to_numeric(self.frame["weight"], errors="coerce").fillna(0.0).to_numpy()

    @cached_property
    def leaves_by_share(self) -> list[str]:
        return [n.cluster_id for n in sorted(self.tree.leaves(), key=lambda n: -n.pop_share)]

    @cached_property
    def cluster_weight(self) -> dict[str, float]:
        """Survey weight mass per cluster — the aggregation weight before raking."""
        w = self.frame.groupby(self.assign)["weight"].sum()
        return {str(k): float(v) for k, v in w.items()}

    @cached_property
    def raked_weight(self) -> dict[str, float]:
        """Cluster weights raked to ACS 2024, falling back to survey weights.

        The bed pools 2010-2022, so its composition is that of survey respondents
        over that span, not of US adults now. Any population rollup that skips
        this answers "what would the pooled GSS sample say", which is not the
        question. The fallback is logged, never silent (spec §M6).
        """
        from .aggregate.mixture import rake_weights

        try:
            margins = pd.read_parquet(self.cfg.data_path(self.cfg["aggregation.margins"]))
        except Exception as exc:  # noqa: BLE001 - a missing margins file is not fatal
            log.warning("raking margins unreadable (%s); using survey weights", exc)
            self.notes["raking"] = f"unavailable: {exc}"
            return dict(self.cluster_weight)
        res = rake_weights(self, margins,
                           axes=tuple(self.cfg["partition.axes"]),
                           max_iter=int(self.cfg["aggregation.rake_max_iter"]),
                           tol=float(self.cfg["aggregation.rake_tol"]))
        self.notes["raking"] = res.summary()
        if res.fallback:
            log.warning("%s", res.summary())
            return dict(self.cluster_weight)
        return res.weights

    def national(self, item_id: str) -> np.ndarray:
        cs = self.codes[item_id]
        col = self.frame[f"item_{item_id}"].to_numpy()
        ok = np.isin(col, cs)
        return weighted_histogram(col[ok], self.weights[ok], cs)

    def level_hist(self, item_id: str, cluster_ids) -> np.ndarray:
        """The item's level: the weighted mixture of the scored cells' truths.

        Not quite ``national()``. The deviation reference the calibrator
        subtracts is the population-weighted mean of the *inputs'* CDFs over the
        clusters in scope, and the CDF is linear in the histogram, so the only
        level that makes ``s = 0`` reproduce the no-conditioning baseline
        identically — and ``s = 1`` return the truth identically when the truth
        is fed in — is the mixture over those same cells.

        The Layer 3 oracle gate is what surfaced this: with a 16-cluster subset,
        ``national()`` is a mixture over all 56 leaves plus the respondents in
        none of them, so the reference and the level were different objects and
        the round-trip drifted by 0.04 W1.

        It is also the *stronger* baseline, which is the right direction: the
        best constant predictor of a set of cells is the weighted average of
        those cells, not of a wider population. B0a is given this same object, so
        the comparison stays about subgroup structure and nothing else, and
        ``national()`` is still reported beside it as ``B0a_national``.
        """
        hs, ws = [], []
        for c in cluster_ids:
            h = self.stats.hist(c, item_id)
            w = self.cell_weight(c, item_id)
            if h is not None and h.sum() > 0 and w > 0:
                hs.append(np.asarray(h, dtype=float)); ws.append(w)
        if not hs:
            return self.national(item_id)
        w = np.asarray(ws); w = w / w.sum()
        return w @ np.vstack(hs)

    def item(self, item_id: str) -> dict:
        return dict(self.codebook[item_id]) | {"item_id": item_id}

    def cell_weight(self, cluster_id: str, item_id: str) -> float:
        row = self.stats.frame[(self.stats.frame.cluster_id == cluster_id)
                               & (self.stats.frame.item_id == item_id)]
        return 0.0 if row.empty else float(row.iloc[0]["weight_sum"])

    def cell_neff(self, cluster_id: str, item_id: str) -> float:
        row = self.stats.frame[(self.stats.frame.cluster_id == cluster_id)
                               & (self.stats.frame.item_id == item_id)]
        return 0.0 if row.empty else float(row.iloc[0]["n_eff"])

    def scorable(self, item_id: str, cluster_ids: list[str], min_neff: float) -> list[str]:
        """Clusters whose truth for this item is solid enough to score against.

        ``truth_min_neff`` (default 30) is the spec's crude proxy for the Layer 1
        noise floor; both are applied, the floor at item level in the split and
        this at cell level.
        """
        out = []
        for c in cluster_ids:
            h = self.stats.hist(c, item_id)
            if h is not None and h.sum() > 0 and self.cell_neff(c, item_id) >= min_neff:
                out.append(c)
        return out

    # ----------------------------------------------------------------- items
    def ranked_targets(self, exclude_doctrine: bool = True) -> list[str]:
        ts = sorted(self.split["targets"], key=lambda i: -self.split["snr"].get(i, 0.0))
        if exclude_doctrine:
            ts = [i for i in ts if i not in POST_FREEZE_DOCTRINE_FLAGS]
        return ts

    def clusters(self, n: int = 0) -> list[str]:
        leaves = self.leaves_by_share
        if n and n < len(leaves):
            idx = np.linspace(0, len(leaves) - 1, n).round().astype(int)
            return [leaves[i] for i in sorted(set(idx))]
        return leaves

    # ------------------------------------------------------- the population
    @cached_property
    def population_stats(self):
        """``ClusterStats`` for one pseudo-cluster containing everybody.

        The B0b baseline and the ``predicted_level`` calibration mode both need
        the model's estimate of the item's *national* distribution, and the
        honest way to get it is the same prompt the clusters get, with the
        population in the cluster's place: same card template, same anchors, same
        response contract. Anything else would compare a cluster answer with an
        answer produced under different conditions and attribute the difference
        to conditioning.
        """
        everyone = pd.Series(["all"] * len(self.frame), index=self.frame.index)
        return compute_cluster_stats(self.frame, everyone, sorted(self.codebook), self.codes)

    def population_card(self, item_id: str) -> StatCard:
        card = make_statcard(
            cluster_id="all", tree=self.tree, stats=self.population_stats,
            frame=self.frame,
            cluster_of=pd.Series(["all"] * len(self.frame), index=self.frame.index),
            target_item=item_id, codebook=self.codebook,
            anchor_pool=self.plan.anchors_for(item_id), near_duplicates=self.near_dupes,
            anchor_fold=self.plan.fold_of.get(item_id, self.plan.target_fold.get(item_id)),
            anchors_per_card=self.cfg["elicitation.anchors_per_card"],
            fold_of=self.plan.fold_of,
        )
        card.definition_text = "All US adults (the whole population, not a subgroup)"
        card.pop_share = 1.0
        return card

    # ----------------------------------------------------------------- cards
    def card(self, item_id: str, cluster_id: str, *,
             anchors_per_card: int | None = None) -> StatCard:
        return make_statcard(
            cluster_id=cluster_id, tree=self.tree, stats=self.stats, frame=self.frame,
            cluster_of=self.assign, target_item=item_id, codebook=self.codebook,
            anchor_pool=self.plan.anchors_for(item_id), near_duplicates=self.near_dupes,
            anchor_fold=self.plan.fold_of.get(item_id, self.plan.target_fold.get(item_id)),
            anchors_per_card=(self.cfg["elicitation.anchors_per_card"]
                              if anchors_per_card is None else anchors_per_card),
            fold_of=self.plan.fold_of,
        )


def _bed_fingerprint(cfg: Config) -> str:
    """Everything that can change the bed, the partition, the split or the folds."""
    import hashlib

    parts = [
        str(cfg["bed.waves"]), str(cfg["bed.exclude_waves"]), str(cfg["bed.weight_var"]),
        str(cfg["partition.axes"]), str(cfg["partition.min_cell"]),
        str(cfg["partition.k_target"]), str(cfg["calibration.crossfit_folds"]),
        str(cfg["item_split.freeze_path"]), str(cfg.get("item_split.use_amendments", False)),
        str(cfg["seed"]),
    ]
    for rel in ("codebooks/gss_items.yaml", cfg["item_split.freeze_path"]):
        f = cfg.repo_root / rel
        parts.append(f"{rel}:{f.stat().st_mtime_ns if f.exists() else 0}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def build_bed(cfg: Config, *, verbose: bool = False, cache: bool = True) -> Bed:
    """Build the bed, or load it from the fingerprinted cache.

    Building costs ~9 s, which is nothing once but is most of a window when six
    sharded elicitation processes start up inside a 165-second budget. The
    fingerprint covers every config value and every input file that can change
    what comes out, so a stale cache is not reachable by editing the config: a
    changed fingerprint is a different file.
    """
    import pickle

    fp = _bed_fingerprint(cfg)
    cache_path = cfg.repo_root / cfg["paths.runs_root"] / "_bed_cache" / f"bed.{fp}.pkl"
    if cache and cache_path.exists():
        try:
            bed = pickle.loads(cache_path.read_bytes())
            if verbose:
                print(f"  bed      cached ({fp})  {len(bed.tree.leaves())} leaves, "
                      f"{len(bed.split['targets'])} targets / "
                      f"{len(bed.split['anchors'])} anchors")
            return bed
        except Exception as exc:  # noqa: BLE001 - a corrupt cache must never block a run
            log.debug("bed cache unreadable (%s); rebuilding", exc)
    bed = _build_bed_uncached(cfg, verbose=verbose)
    if cache:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(bed))
            tmp.replace(cache_path)
        except Exception as exc:  # noqa: BLE001 - caching is best effort
            log.debug("could not write the bed cache: %s", exc)
    return bed


def _build_bed_uncached(cfg: Config, *, verbose: bool = False) -> Bed:
    codebook = yaml.safe_load(
        (cfg.repo_root / cfg["paths.codebooks"] / "gss_items.yaml").read_text()
    )["items"]
    # Amendment 1 is RETIRED by default. PREREGISTRATION.md §7a records it as an
    # error in application, not a tested rule: `make_crossfit_plan` derives folds
    # from the anchor list, so adding five anchors re-derived every fold and put
    # `nataid` — the spending battery's outlier — back onto the spending cards.
    # Every piece of evidence the Phase 4 design rests on was measured on the
    # un-amended split, and leaving the amendment on would confound the fold
    # structure with everything else. It stays on disk and is available as an
    # ablation arm via `item_split.use_amendments: true`.
    amd_paths = None if cfg.get("item_split.use_amendments", False) else []
    split, amendments = load_effective_split(
        cfg.repo_root / cfg["item_split.freeze_path"], amendments=amd_paths
    )
    all_items = sorted(codebook)
    codes = {i: codebook[i]["codes"] for i in all_items}
    table = load_gss(cfg.bed_file, all_items, waves=cfg["bed.waves"])
    tree = build_cluster_tree(
        table.frame, axes=cfg["partition.axes"], item_codes=codes,
        min_cell=cfg["partition.min_cell"], k_target=cfg["partition.k_target"],
    )
    assign = tree.assign(table.frame)
    stats = compute_cluster_stats(table.frame, assign, all_items, codes)
    plan = make_crossfit_plan(
        split["anchors"], split["targets"],
        {i: codebook[i].get("topic", "other") for i in all_items},
        n_folds=cfg["calibration.crossfit_folds"],
    )
    if verbose:
        print(f"  bed      {table.frame.shape[0]} respondents, waves {cfg['bed.waves']}")
        print(f"  clusters {len(tree.leaves())} leaves")
        print(f"  split    {len(split['targets'])} targets / {len(split['anchors'])} anchors"
              + (f"  (+{len(amendments)} amendment(s))" if amendments else ""))
    return Bed(cfg=cfg, codebook=codebook, split=split, amendments=amendments,
               table=table, tree=tree, assign=assign, stats=stats, plan=plan,
               near_dupes=battery_near_duplicates(all_items))
