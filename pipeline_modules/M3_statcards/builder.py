"""
Stat card builder (module M3).

A stat card is the conditioning context handed to the LLM for one cluster. It
carries the cluster's demographic definition, its share of the population, extra
demographic marginals not implied by the definition, and its *observed* response
histograms on a topic-diverse set of anchor items.

The agent is never asked to role-play a person. It is shown what this group
actually answered on questions we have data for, and asked what the same group
would answer on a question we do not.

Leakage contract
----------------
A card built to elicit item ``Y`` must not contain:

1. ``Y`` itself;
2. any near-duplicate of ``Y`` (cosine similarity above a threshold on item text);
3. any anchor assigned to ``Y``'s cross-fit fold.

Rule 3 is what makes the calibration layer honest: anchor predictions used to fit
the calibrator are produced under exactly the conditions target items face, so the
learned map is a transfer map rather than a memorisation of anchors the card
already displayed. All three rules are enforced here and asserted in
``verify_card_contract``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import jinja2
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Demographic fields worth reporting on a card. Fields that appear in the
# cluster definition itself are dropped automatically (they carry no extra
# information -- the definition already pins them).
MARGINAL_FIELDS = ["race", "religion", "marital", "employment", "age_band", "education"]


@dataclass
class AnchorItemView:
    item_id: str
    topic: str
    text: str
    options: List[str]
    cluster_hist: List[float]
    national_hist: List[float]
    n_eff: float


@dataclass
class StatCard:
    cluster_id: str
    level: int
    parent: Optional[str]
    definition_prose: str
    definition: Dict[str, Any]
    pop_share: float
    n_raw: int
    n_eff: float
    demo_marginals: Dict[str, Dict[str, float]]
    anchor_items: List[AnchorItemView]
    excluded_items: List[str]
    crossfit_fold: Optional[int]
    rendered_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "cluster_id": self.cluster_id,
            "level": self.level,
            "parent": self.parent,
            "definition_prose": self.definition_prose,
            "definition": self.definition,
            "pop_share": self.pop_share,
            "n_raw": self.n_raw,
            "n_eff": self.n_eff,
            "demo_marginals": self.demo_marginals,
            "anchor_items": [a.__dict__ for a in self.anchor_items],
            "excluded_items": self.excluded_items,
            "crossfit_fold": self.crossfit_fold,
        }
        return d


class StatCardBuilder:
    """
    Builds stat cards from the M1/M2 preprocessing outputs.

    Parameters
    ----------
    tree_path, stats_path, codebook_path, split_path, pop_stats_path
        Artefacts written by ``preprocessing/run_pipeline.py``.
    individual_table_path
        Optional. Enables per-cluster demographic marginals and exact segment
        membership. Strongly recommended -- without it cards carry only the
        definition and anchors.
    duplicate_threshold
        Cosine similarity on TF-IDF item text above which two items are treated
        as near-duplicates and never co-occur on a card.
    """

    def __init__(
        self,
        tree_path: Path,
        stats_path: Path,
        codebook_path: Path,
        split_path: Path,
        pop_stats_path: Optional[Path] = None,
        individual_table_path: Optional[Path] = None,
        template_path: Optional[Path] = None,
        duplicate_threshold: float = 0.60,
        granularity: str = "coarse",
    ):
        self.tree_path = Path(tree_path)
        self.stats_path = Path(stats_path)
        self.codebook_path = Path(codebook_path)
        self.split_path = Path(split_path)
        self.pop_stats_path = Path(pop_stats_path) if pop_stats_path else None
        self.individual_table_path = (
            Path(individual_table_path) if individual_table_path else None
        )
        self.duplicate_threshold = duplicate_threshold
        self.granularity = granularity

        self._load_data()
        self._build_duplicate_index()
        self._init_jinja(template_path)

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def _load_data(self) -> None:
        with open(self.tree_path, "r", encoding="utf-8") as f:
            self.nodes = json.load(f)
        self.nodes_by_id = {n["cluster_id"]: n for n in self.nodes}
        self.leaf_ids = [n["cluster_id"] for n in self.nodes if n["level"] == 2]

        self.stats_df = pd.read_parquet(self.stats_path)
        self.stats_dict = {
            (r.cluster_id, r.item_id): r for r in self.stats_df.itertuples()
        }

        with open(self.codebook_path, "r", encoding="utf-8") as f:
            self.codebook = {item["item_id"]: item for item in yaml.safe_load(f)}

        self.split_df = pd.read_csv(self.split_path)
        self.split_dict = {r["item_id"]: r.to_dict() for _, r in self.split_df.iterrows()}

        if self.pop_stats_path and self.pop_stats_path.exists():
            self.pop_stats_df = pd.read_parquet(self.pop_stats_path)
            self.pop_dict = {r.item_id: r for r in self.pop_stats_df.itertuples()}
        else:
            self.pop_stats_df = pd.DataFrame()
            self.pop_dict = {}

        self.individuals: Optional[pd.DataFrame] = None
        self._cluster_col = "cluster_id_coarse" if self.granularity == "coarse" else "cluster_id"
        if self.individual_table_path and self.individual_table_path.exists():
            self.individuals = pd.read_parquet(self.individual_table_path)
            if self._cluster_col not in self.individuals.columns:
                logger.warning(
                    "individual table has no column %s; demographic marginals disabled",
                    self._cluster_col,
                )
                self.individuals = None

    def _build_duplicate_index(self) -> None:
        """TF-IDF index over item wording, used for near-duplicate exclusion."""
        self._dup_ids = list(self.codebook.keys())
        texts = [self.codebook[i].get("text", i) or i for i in self._dup_ids]
        self._dup_pos = {iid: n for n, iid in enumerate(self._dup_ids)}
        try:
            self._dup_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            self._dup_mat = self._dup_vec.fit_transform(texts)
        except ValueError:  # e.g. every text is a stop word
            self._dup_vec, self._dup_mat = None, None

    def _init_jinja(self, template_path: Optional[Path]) -> None:
        if template_path is None:
            template_path = Path(__file__).resolve().parent / "templates" / "statcard.jinja"
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(Path(template_path).parent)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.template = env.get_template(Path(template_path).name)

    # ------------------------------------------------------------------
    # leakage contract
    # ------------------------------------------------------------------
    def near_duplicates(self, item_id: str, threshold: Optional[float] = None) -> Set[str]:
        """Items whose wording is close enough to ``item_id`` to leak the answer."""
        if self._dup_mat is None or item_id not in self._dup_pos:
            return set()
        thr = self.duplicate_threshold if threshold is None else threshold
        row = self._dup_mat[self._dup_pos[item_id]]
        sims = cosine_similarity(row, self._dup_mat)[0]
        return {
            self._dup_ids[j]
            for j in np.flatnonzero(sims >= thr)
            if self._dup_ids[j] != item_id
        }

    def eligible_anchors(
        self,
        target_item_id: Optional[str],
        split_regime: str = "standard",
        crossfit_plan: Optional[Dict[str, int]] = None,
        extra_exclude: Optional[Set[str]] = None,
    ) -> tuple[List[str], List[str]]:
        """
        Anchors that may legally appear on a card built to elicit ``target_item_id``.

        Returns ``(eligible, excluded)`` so the exclusion is auditable rather than
        implicit.
        """
        role_col = f"role_{split_regime}"
        if role_col not in self.split_df.columns:
            role_col = "role"

        excluded: Set[str] = set(extra_exclude or set())
        if target_item_id:
            excluded.add(target_item_id)
            excluded |= self.near_duplicates(target_item_id)
            if crossfit_plan and target_item_id in crossfit_plan:
                fold = crossfit_plan[target_item_id]
                excluded |= {i for i, f in crossfit_plan.items() if f == fold}

        eligible = [
            iid
            for iid, row in self.split_dict.items()
            if row.get(role_col) == "anchor" and iid not in excluded
        ]
        return eligible, sorted(excluded)

    # ------------------------------------------------------------------
    # anchor selection
    # ------------------------------------------------------------------
    def select_topic_diverse_anchors(
        self,
        cluster_id: str,
        eligible_anchors: Sequence[str],
        max_anchors: int = 12,
    ) -> List[str]:
        """
        Greedy max-coverage over topic tags.

        Round-robins across topics so a card spans as many distinct topics as it
        can before taking a second item from any one topic. Within a topic, items
        are ordered by the cluster's effective sample size on that item, so the
        histograms shown are the least noisy available. Fully deterministic.
        """
        by_topic: Dict[str, List[str]] = {}
        for iid in eligible_anchors:
            key = (cluster_id, iid)
            if key not in self.stats_dict or iid not in self.codebook:
                continue
            topic = self.codebook[iid].get("topic", "general")
            by_topic.setdefault(topic, []).append(iid)

        for topic in by_topic:
            by_topic[topic].sort(
                key=lambda i: (-float(self.stats_dict[(cluster_id, i)].n_eff), i)
            )

        selected: List[str] = []
        topics = sorted(by_topic)
        rnd = 0
        while len(selected) < max_anchors and any(by_topic.values()):
            progressed = False
            for topic in topics:
                if len(selected) >= max_anchors:
                    break
                bucket = by_topic[topic]
                if len(bucket) > rnd:
                    selected.append(bucket[rnd])
                    progressed = True
            if not progressed:
                break
            rnd += 1
        return selected[:max_anchors]

    # ------------------------------------------------------------------
    # demographic marginals
    # ------------------------------------------------------------------
    def cluster_demo_marginals(self, cluster_id: str, definition: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Weighted marginals for demographic fields the cluster definition does not pin.

        A definition already states e.g. region and sex; repeating them adds
        nothing. Race, religion, marital status and employment are what actually
        differentiate two clusters with the same definition axes.
        """
        if self.individuals is None:
            return {}
        sub = self.individuals[self.individuals[self._cluster_col] == cluster_id]
        if sub.empty:
            return {}

        pinned = {k for k, v in definition.items() if not isinstance(v, list) or len(v) == 1}
        out: Dict[str, Dict[str, float]] = {}
        w = sub["weight"].to_numpy(dtype=float)
        total = float(w.sum())
        if total <= 0:
            return {}

        for field in MARGINAL_FIELDS:
            if field in pinned:
                continue
            col = f"demo_{field}"
            if col not in sub.columns:
                continue
            grouped = sub.groupby(col, observed=True)["weight"].sum() / total
            grouped = grouped[grouped >= 0.01].sort_values(ascending=False)
            if len(grouped) > 1:
                out[field] = {str(k): round(float(v), 3) for k, v in grouped.items()}
        return out

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------
    def build_stat_card(
        self,
        cluster_id: str,
        target_item_id: Optional[str] = None,
        split_regime: str = "standard",
        crossfit_plan: Optional[Dict[str, int]] = None,
        max_anchors: int = 12,
        excluded_items: Optional[Set[str]] = None,
    ) -> StatCard:
        node = self.nodes_by_id.get(cluster_id)
        if node is None:
            raise ValueError(f"cluster_id {cluster_id!r} not in {self.tree_path.name}")

        eligible, excluded = self.eligible_anchors(
            target_item_id=target_item_id,
            split_regime=split_regime,
            crossfit_plan=crossfit_plan,
            extra_exclude=excluded_items,
        )
        selected = self.select_topic_diverse_anchors(cluster_id, eligible, max_anchors)

        anchor_views: List[AnchorItemView] = []
        for iid in selected:
            meta = self.codebook[iid]
            stat = self.stats_dict[(cluster_id, iid)]
            pop_row = self.pop_dict.get(iid)
            national = (
                [float(x) for x in pop_row.hist]
                if pop_row is not None and hasattr(pop_row, "hist")
                else []
            )
            anchor_views.append(
                AnchorItemView(
                    item_id=iid,
                    topic=meta.get("topic", "general"),
                    text=meta.get("text", iid),
                    options=list(meta.get("scale", {}).get("labels", [])),
                    cluster_hist=[float(x) for x in stat.hist],
                    national_hist=national,
                    n_eff=float(stat.n_eff),
                )
            )

        definition = node.get("definition", {}) or {}
        card = StatCard(
            cluster_id=cluster_id,
            level=int(node["level"]),
            parent=node.get("parent"),
            definition_prose=node.get("definition_text", ""),
            definition=definition,
            pop_share=float(node.get("pop_share", 0.0)),
            n_raw=int(node.get("n_raw", 0)),
            n_eff=float(node.get("n_eff", 0.0)),
            demo_marginals=self.cluster_demo_marginals(cluster_id, definition),
            anchor_items=anchor_views,
            excluded_items=excluded,
            crossfit_fold=(crossfit_plan or {}).get(target_item_id) if target_item_id else None,
        )
        card.rendered_prompt = self.template.render(card=card)
        return card

    # ------------------------------------------------------------------
    # contract check
    # ------------------------------------------------------------------
    def verify_card_contract(
        self,
        card: StatCard,
        target_item_id: str,
        crossfit_plan: Optional[Dict[str, int]] = None,
    ) -> None:
        """Raise if the card violates the leakage contract. Called by the elicitor."""
        shown = {a.item_id for a in card.anchor_items}

        if target_item_id in shown:
            raise AssertionError(f"card for {target_item_id} shows the target item itself")

        dups = self.near_duplicates(target_item_id) & shown
        if dups:
            raise AssertionError(
                f"card for {target_item_id} shows near-duplicate(s): {sorted(dups)}"
            )

        if crossfit_plan and target_item_id in crossfit_plan:
            fold = crossfit_plan[target_item_id]
            leaked = {i for i in shown if crossfit_plan.get(i) == fold}
            if leaked:
                raise AssertionError(
                    f"card for {target_item_id} (fold {fold}) shows same-fold anchor(s): "
                    f"{sorted(leaked)}"
                )

        text = card.rendered_prompt
        if target_item_id and f"[{target_item_id}]" in text:
            raise AssertionError(f"target id {target_item_id} leaked into rendered prompt")
