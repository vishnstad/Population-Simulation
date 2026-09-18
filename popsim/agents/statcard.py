"""M3 — stat cards (checklist 3.1).

    "`statcard.jinja`. Geography as a **marginal line** ("58 % South, 31 %
     urban"), not a constraint. Leaves are disjunctions — render them honestly."

The card is the agent's entire conditioning context, so two things about it
decide whether the project works at all.

**Geography is a marginal, never a constraint.** Measured: putting geography in
the partition drops the usable item bank from 74 to 35. So a cluster is *not*
"women in the South" — it is women, of whom 58% happen to live in the South.
Writing the first when the second is true would be a false constraint, and the
model would condition on a region the cluster does not actually have. The same
applies to mode, which is nearly collinear with wave in this bed and matters
because NORC flags several pool items as mode-sensitive.

**Disjunctive leaves are rendered as disjunctions.** The tree merges cells, so a
leaf can be "degree 3 or 4" or "age 18-24 or 25-34". Rendering that as a single
tidy category would tell the model something false about who it is describing.
``definition_text`` therefore says "with a bachelor's or graduate degree", not
"college-educated", and the card carries the within-leaf split so the model can
see how lopsided the disjunction is.

The contract (§3.4)
-------------------
A card built to elicit item Y must contain no Y, no near-duplicate of Y, and no
anchor from Y's cross-fit fold. :func:`make_statcard` enforces all three and
raises rather than quietly dropping, because a card that silently lost its
exclusions is a leak that shows up as a suspiciously good score.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "AnchorItem",
    "CardContractError",
    "StatCard",
    "battery_near_duplicates",
    "make_statcard",
    "render_card",
]

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

#: Demographic fields rendered as marginals. Order is the order they appear.
MARGINAL_FIELDS = ("region", "urban", "race", "mode")

MARGINAL_LABELS = {
    "region": "census division",
    "urban": "urban/rural",
    "race": "race",
    "mode": "interview mode",
}

AXIS_PHRASES = {
    "age_band": "aged {}",
    "degree": "{}",
    "sex": "{}",
    "age3": "aged {}",
    "edu3": "{}",
}

DEGREE_WORDS = {
    0: "less than high school", 1: "high school", 2: "an associate degree",
    3: "a bachelor's degree", 4: "a graduate degree",
}


#: Tolerance-battery groups. spkath / colath / libath are the SAME proposition
#: ("should an anti-religionist be tolerated?") put in three venues — speak,
#: teach, keep a book in the library. Showing two of them on a card built to
#: elicit the third hands the model most of the answer, which is the §5.6
#: leakage objection arriving through the front door rather than pretraining.
#:
#: The nat* spending battery is deliberately NOT treated this way: foreign aid
#: and highway spending are different questions that happen to share a stem, and
#: excluding them from each other's cards would throw away real conditioning.
#: The proper similarity router (M7, checklist 6.1) replaces this later; until
#: then this covers the one family where the risk is structural rather than a
#: matter of degree.
TOLERANCE_GROUPS = ("ath", "rac", "com", "mil", "homo", "mslm")
TOLERANCE_VENUES = ("spk", "col", "lib")


def battery_near_duplicates(items: Sequence[str]) -> dict[str, set[str]]:
    """Map each tolerance-battery item to its siblings in the same group."""
    groups: dict[str, set[str]] = {}
    for item in items:
        for venue in TOLERANCE_VENUES:
            if item.startswith(venue) and item[len(venue):] in TOLERANCE_GROUPS:
                groups.setdefault(item[len(venue):], set()).add(item)
    out: dict[str, set[str]] = {}
    for members in groups.values():
        for m in members:
            out[m] = members - {m}
    return out


class CardContractError(RuntimeError):
    """A card violates the §3.4 exclusion contract."""


@dataclass
class AnchorItem:
    item_id: str
    text: str
    labels: list[str]
    hist: list[float]
    topic: str

    def as_percentages(self) -> list[int]:
        raw = np.asarray(self.hist, dtype=float) * 100
        # Largest-remainder rounding, so the printed numbers sum to exactly 100.
        floors = np.floor(raw).astype(int)
        short = 100 - floors.sum()
        if short > 0:
            for i in np.argsort(-(raw - floors))[:short]:
                floors[i] += 1
        return floors.tolist()


@dataclass
class StatCard:
    cluster_id: str
    definition_text: str
    pop_share: float
    n_respondents: int
    demo_marginals: dict[str, dict[str, float]]
    anchor_items: list[AnchorItem]
    anchor_fold: int | None
    excluded: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _describe_leaf(definition: dict[str, Any]) -> str:
    """Render a leaf definition in plain English, disjunctions intact."""
    parts: list[str] = []
    sex = definition.get("sex")
    age = definition.get("age_band") or definition.get("age3")
    deg = definition.get("degree")
    edu = definition.get("edu3")

    if sex:
        vals = list(sex)
        noun = "adults" if len(vals) > 1 else ("men" if vals[0] == "male" else "women")
    else:
        noun = "adults"
    parts.append(noun.capitalize())

    if age:
        vals = list(age)
        parts.append(f"aged {vals[0]}" if len(vals) == 1 else f"aged {vals[0]} to {vals[-1]}")

    if deg is not None:
        vals = [int(v) for v in deg]
        words = [DEGREE_WORDS.get(v, str(v)) for v in vals]
        if len(words) == 1:
            parts.append(f"whose highest qualification is {words[0]}")
        else:
            joined = " or ".join([", ".join(words[:-1]), words[-1]]) if len(words) > 2 \
                else " or ".join(words)
            parts.append(f"whose highest qualification is {joined}")
    elif edu:
        parts.append(" or ".join(map(str, edu)))

    return ", ".join(parts)


def _marginals_for(frame: pd.DataFrame, weight_col: str = "weight") -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    w = pd.to_numeric(frame[weight_col], errors="coerce").fillna(0.0)
    for fieldname in MARGINAL_FIELDS:
        if fieldname not in frame.columns:
            continue
        col = frame[fieldname]
        if col.dtype == bool or str(col.dtype) == "boolean":
            col = col.map({True: "urban", False: "rural"})
        ok = col.notna() & (w > 0)
        if not ok.any():
            continue
        shares = w[ok].groupby(col[ok]).sum()
        shares = shares / shares.sum()
        out[fieldname] = {str(k): float(v) for k, v in shares.sort_values(ascending=False).items()}
    return out


def _pick_anchors(
    candidates: Sequence[str],
    topics: dict[str, str],
    cap: int,
) -> list[str]:
    """Max-coverage over topic tags, then fill.

    Spec §M3 caps the card at 12 anchors and selects by topic diversity. A card
    of twelve civil-liberties items would tell the model a great deal about one
    corner of the respondent's views and nothing about the rest.
    """
    chosen: list[str] = []
    seen_topics: set[str] = set()
    remaining = list(candidates)

    while remaining and len(chosen) < cap:
        fresh = [i for i in remaining if topics.get(i, "other") not in seen_topics]
        pick = min(fresh) if fresh else min(remaining)
        chosen.append(pick)
        seen_topics.add(topics.get(pick, "other"))
        remaining.remove(pick)
        if not fresh and len(seen_topics) >= len({topics.get(i, "other") for i in candidates}):
            seen_topics.clear()  # start a second pass over topics
    return chosen


def make_statcard(
    *,
    cluster_id: str,
    tree,
    stats,
    frame: pd.DataFrame,
    cluster_of: pd.Series,
    target_item: str,
    codebook: dict[str, dict],
    anchor_pool: Sequence[str],
    near_duplicates: dict[str, set[str]] | None = None,
    anchor_fold: int | None = None,
    anchors_per_card: int = 12,
    fold_of: dict[str, int] | None = None,
) -> StatCard:
    """Build one card, enforcing the §3.4 contract."""
    node = tree.nodes[cluster_id]
    near_duplicates = near_duplicates or {}

    banned = {target_item} | set(near_duplicates.get(target_item, ()))
    usable = [a for a in anchor_pool if a not in banned]
    if not usable and anchors_per_card:
        raise CardContractError(
            f"no anchors left for {target_item!r} after exclusions; the card would "
            f"carry no conditioning information at all"
        )

    topics = {i: codebook[i].get("topic", "other") for i in usable}
    # anchors_per_card = 0 is the §5.5 "no-anchor stat card" ablation: the
    # demographic description only, which is the Argyle regime lifted to the
    # cluster level. It is the arm that says whether the anchors are doing the
    # work or the demographics alone are.
    picked = _pick_anchors(usable, topics, anchors_per_card) if anchors_per_card else []

    anchors: list[AnchorItem] = []
    for a in picked:
        h = stats.hist(cluster_id, a)
        if h is None or not np.isfinite(h).all() or h.sum() <= 0:
            continue
        anchors.append(AnchorItem(
            item_id=a, text=codebook[a]["text"], labels=list(codebook[a]["labels"]),
            hist=[float(x) for x in h], topic=topics.get(a, "other"),
        ))

    # cluster_of holds pd.NA for the handful of respondents outside every leaf;
    # a plain == comparison on that dtype raises rather than returning False.
    members = frame[(cluster_of.fillna("") == cluster_id).to_numpy()]
    card = StatCard(
        cluster_id=cluster_id,
        definition_text=_describe_leaf(node.definition),
        pop_share=float(node.pop_share),
        n_respondents=int(node.n),
        demo_marginals=_marginals_for(members),
        anchor_items=anchors,
        anchor_fold=anchor_fold,
        excluded=sorted(banned),
        notes={"n_anchor_candidates": len(usable), "anchors_used": len(anchors)},
    )
    assert_card_contract(card, target_item, near_duplicates, fold_of=fold_of,
                         allow_empty=(anchors_per_card == 0))
    return card


def assert_card_contract(
    card: StatCard,
    target_item: str,
    near_duplicates: dict[str, set[str]] | None = None,
    fold_of: dict[str, int] | None = None,
    allow_empty: bool = False,
) -> None:
    """§3.4: no Y, no near-duplicate of Y, no same-fold anchor.

    The third clause was claimed here and checked nowhere, which is how the
    whole spending battery came to be conditioned on ``nataid``. It needs
    ``fold_of`` to check, so callers that have the plan must pass it.
    """
    near_duplicates = near_duplicates or {}
    ids = {a.item_id for a in card.anchor_items}
    if target_item in ids:
        raise CardContractError(f"card for {target_item!r} contains the target item itself")
    dupes = ids & set(near_duplicates.get(target_item, ()))
    if dupes:
        raise CardContractError(
            f"card for {target_item!r} contains near-duplicates {sorted(dupes)}"
        )
    if not card.anchor_items and not allow_empty:
        # An empty card is a bug everywhere except in the §5.5 no-anchor
        # ablation, which exists precisely to measure what the demographics
        # alone are worth. That arm has to ask for it explicitly.
        raise CardContractError(f"card for {target_item!r} has no anchors")
    if fold_of is not None and card.anchor_fold is not None:
        same_fold = sorted(i for i in ids if fold_of.get(i) == card.anchor_fold)
        if same_fold:
            raise CardContractError(
                f"card for {target_item!r} carries anchors from its own excluded "
                f"fold {card.anchor_fold}: {same_fold}. That fold is chosen as the "
                f"one densest in same-topic anchors, so these are the anchors most "
                f"likely to be near duplicates of the target"
            )


def render_card(card: StatCard, *, template: str = "statcard.jinja") -> str:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    env = Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True,
    )
    env.filters["pct"] = lambda x: f"{round(float(x) * 100)}%"
    return env.get_template(template).render(card=card).strip()
