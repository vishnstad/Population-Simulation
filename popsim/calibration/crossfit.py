"""Cross-fit plan (checklist 4.1, needed by the §3.4 card contract).

    "Anchors are split into F = 3 folds. For fold f: elicit every anchor item in
     f using stat cards built from folds != f. This yields, for every anchor, a
     (raw prediction, ground-truth cross-tab) pair produced under exactly the
     conditions targets face."

Why this exists before Phase 4
------------------------------
The calibrator is fitted in Phase 4, but the *plan* is needed in Phase 3,
because §3.4's contract says a card for item Y must contain no anchor from Y's
own fold. Without folds there is nothing for that test to check, and the
calibration layer would later be fitted on pairs produced under conditions
targets never face — which is the quiet way a cross-fitted estimate stops being
cross-fitted.

Stratified by topic so that no fold is missing a whole subject area; an anchor
fold that happens to contain every religion item makes the calibration map for
religion targets fit on nothing like them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

__all__ = ["CrossfitPlan", "make_crossfit_plan"]


@dataclass
class CrossfitPlan:
    n_folds: int
    fold_of: dict[str, int]          # anchor item -> fold index
    target_fold: dict[str, int]      # target item -> the fold it is treated as
    stratify_by: str = "topic"
    notes: dict = field(default_factory=dict)

    def anchors_in_fold(self, f: int) -> list[str]:
        return sorted(i for i, k in self.fold_of.items() if k == f)

    def anchors_excluding_fold(self, f: int) -> list[str]:
        return sorted(i for i, k in self.fold_of.items() if k != f)

    def anchors_for(self, item_id: str) -> list[str]:
        """Anchors usable on a card built to elicit ``item_id``.

        Every anchor outside the item's own fold, for an anchor and a target
        alike. For a target that fold is ``target_fold``, which is chosen in
        ``make_crossfit_plan`` as the fold densest in same-topic anchors —
        precisely so that excluding it takes out the anchors most likely to be
        near duplicates of the target.

        This used to return all anchors for a target, which left ``target_fold``
        computed, recorded on every card, and enforced nowhere. Checklist 3.4
        names the rule ("no same-fold anchor") and ``assert_card_contract``'s own
        docstring claimed it. What it cost is concrete: ``natroad``'s excluded
        fold is 0, ``nataid`` is in fold 0, and ``_pick_anchors`` breaks ties
        alphabetically — so the one spending anchor on every spending card was
        the alphabetically first member of the fold that should have been gone.
        ``nataid`` is also the battery's extreme outlier, 57.7% "too much"
        against a battery median near 10%, and the 7B tracked it at r = +0.85 on
        ``natroad`` and +0.90 on ``natsoc`` while correlating negatively with the
        truth. The model was reading the card too well, not ignoring it.
        """
        fold = self.fold_of.get(item_id, self.target_fold.get(item_id))
        if fold is None:
            return sorted(self.fold_of)
        return self.anchors_excluding_fold(fold)

    def to_yaml(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(asdict(self), sort_keys=False, allow_unicode=True))
        return p

    @classmethod
    def from_yaml(cls, path: str | Path) -> CrossfitPlan:
        return cls(**yaml.safe_load(Path(path).read_text()))


def make_crossfit_plan(
    anchors: list[str],
    targets: list[str],
    topics: dict[str, str],
    *,
    n_folds: int = 3,
    seed: int = 17,
) -> CrossfitPlan:
    """Assign anchors to folds, round-robin within each topic."""
    import random

    rng = random.Random(seed)
    by_topic: dict[str, list[str]] = {}
    for a in sorted(anchors):
        by_topic.setdefault(topics.get(a, "other"), []).append(a)

    fold_of: dict[str, int] = {}
    for topic in sorted(by_topic):
        members = by_topic[topic][:]
        rng.shuffle(members)
        # Offset the round-robin per topic so small topics do not all pile into
        # fold 0.
        offset = rng.randrange(n_folds)
        for i, item in enumerate(members):
            fold_of[item] = (i + offset) % n_folds

    # A target is assigned the fold whose anchors it must not see. Topic-matched,
    # so a target's excluded fold is the one most likely to contain a near
    # duplicate of it.
    target_fold: dict[str, int] = {}
    for t in sorted(targets):
        same_topic = [a for a in anchors if topics.get(a) == topics.get(t)]
        if same_topic:
            counts = {f: 0 for f in range(n_folds)}
            for a in same_topic:
                counts[fold_of[a]] += 1
            target_fold[t] = max(counts, key=lambda f: (counts[f], -f))
        else:
            target_fold[t] = rng.randrange(n_folds)

    sizes = {f: sum(1 for v in fold_of.values() if v == f) for f in range(n_folds)}
    return CrossfitPlan(
        n_folds=n_folds, fold_of=fold_of, target_fold=target_fold,
        notes={
            "n_anchors": len(anchors), "n_targets": len(targets),
            "fold_sizes": sizes, "seed": seed,
            "topics_covered_per_fold": {
                f: len({topics.get(i, "other") for i, k in fold_of.items() if k == f})
                for f in range(n_folds)
            },
        },
    )
