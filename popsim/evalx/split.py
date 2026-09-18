"""The anchor/target split (checklist 2.4).

    "Item split: 40 targets / 34 anchors, SNR-gated, topic-stratified. Flag
     famous vs leakage-resistant. **Persist the split — it must never be
     regenerated.**"

Why the ratio is flipped from the spec
--------------------------------------
Spec §5.1 assigns 60% anchors / 40% targets. Measured, that leaves 29 targets,
and the §1.4 claim needs at least 40. Anchors need not be high-SNR because they
are *context* — they are shown to the model, never scored — so the scarce
resource is targets, and the ratio goes the other way.

Selection, in order
-------------------
1. **Ineligible items are removed** — ``role: sanity`` (income restated),
   ``role: excluded`` (below the SNR gate by measurement), and anything whose
   instrument wording is not verified. That last one applies to *both* roles:
   an anchor's question text is rendered into the stat card, so an anchor with
   truncated wording corrupts the prompt exactly like a target would.
2. **The §1.5 leakage-resistant set is seeded into targets first.** Headline
   numbers are reported both ways — all targets, and this subset — and that only
   works if the subset is inside the target set.
3. **Targets are filled to 40**, topic-stratified: take the highest-SNR unused
   item from the least-represented topic, repeatedly. This stops one battery
   from owning the target set, which matters because the tolerance battery alone
   could supply 18.
4. **Anchors are taken next**, topic-stratified over what remains, preferring
   topic coverage over SNR since they are context.
5. **Low-SNR items become anchor-only context**, which is the one place a
   below-gate item is still useful.

Freezing
--------
The split is written once and then treated as read-only. Re-deriving it after
seeing results — even with identical code — is how a held-out set stops being
held out. ``load_or_create`` refuses to overwrite an existing file.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["ItemSplit", "SplitFrozenError", "load_or_create", "make_split"]


class SplitFrozenError(RuntimeError):
    """The split exists on disk and may not be regenerated."""


@dataclass
class ItemSplit:
    targets: list[str]
    anchors: list[str]
    anchor_only_low_snr: list[str]
    ineligible: dict[str, str]
    leakage_resistant: list[str]
    famous: list[str]
    topics: dict[str, str]
    snr: dict[str, float]
    created_utc: str = ""
    notes: dict[str, Any] = field(default_factory=dict)

    def role_of(self, item_id: str) -> str:
        if item_id in self.targets:
            return "target"
        if item_id in self.anchors or item_id in self.anchor_only_low_snr:
            return "anchor"
        return "excluded"

    def to_yaml(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(asdict(self), sort_keys=False, allow_unicode=True, width=100))
        return p

    @classmethod
    def from_yaml(cls, path: str | Path) -> ItemSplit:
        return cls(**yaml.safe_load(Path(path).read_text()))


def _stratified_take(
    pool: list[str], topics: dict[str, str], snr: dict[str, float],
    n: int, already: list[str],
) -> list[str]:
    """Repeatedly take the best unused item from the least-represented topic."""
    chosen: list[str] = []
    counts: dict[str, int] = {}
    for it in already:
        counts[topics.get(it, "other")] = counts.get(topics.get(it, "other"), 0) + 1
    remaining = [i for i in pool if i not in already]

    while len(chosen) < n and remaining:
        by_topic: dict[str, list[str]] = {}
        for it in remaining:
            by_topic.setdefault(topics.get(it, "other"), []).append(it)
        # Least-represented topic; ties broken by the topic's best available SNR.
        topic = min(
            by_topic,
            key=lambda t: (counts.get(t, 0), -max(snr.get(i, 0.0) for i in by_topic[t]), t),
        )
        pick = max(by_topic[topic], key=lambda i: (snr.get(i, 0.0), i))
        chosen.append(pick)
        remaining.remove(pick)
        counts[topic] = counts.get(topic, 0) + 1
    return chosen


def make_split(
    *,
    snr: dict[str, float],
    topics: dict[str, str],
    roles: dict[str, str],
    wording_ok: dict[str, bool],
    leakage_resistant: set[str],
    famous: set[str],
    n_targets: int = 40,
    n_anchors: int = 34,
    n_anchor_only_low_snr: int = 14,
    min_snr: float = 1.5,
) -> ItemSplit:
    ineligible: dict[str, str] = {}
    eligible: list[str] = []

    for item in sorted(snr):
        role = roles.get(item, "unassigned")
        if role == "sanity":
            ineligible[item] = "role: sanity — income restated; may never reach a headline table"
            continue
        if role == "excluded":
            ineligible[item] = "role: excluded — measured below the SNR gate"
            continue
        if not wording_ok.get(item, False):
            ineligible[item] = (
                "instrument wording not verified — an anchor's text is rendered into "
                "the stat card, so this blocks both roles, not just the target role"
            )
            continue
        eligible.append(item)

    passing = [i for i in eligible if snr[i] >= min_snr]
    failing = [i for i in eligible if snr[i] < min_snr]

    # 2. seed targets with the leakage-resistant set
    seeded = [i for i in passing if i in leakage_resistant]
    seeded.sort(key=lambda i: (-snr[i], i))
    targets = seeded[:n_targets]

    # 3. fill to n_targets, topic-stratified
    if len(targets) < n_targets:
        targets += _stratified_take(passing, topics, snr, n_targets - len(targets), targets)

    # 4. anchors from what remains
    anchors = _stratified_take(passing, topics, snr, n_anchors, targets)

    # 5. low-SNR items as anchor-only context
    low = sorted(failing, key=lambda i: (-snr[i], i))[:n_anchor_only_low_snr]

    return ItemSplit(
        targets=sorted(targets),
        anchors=sorted(anchors),
        anchor_only_low_snr=sorted(low),
        ineligible=ineligible,
        leakage_resistant=sorted(set(targets) & leakage_resistant),
        famous=sorted(set(targets) & famous),
        topics={i: topics.get(i, "other") for i in sorted(set(targets) | set(anchors) | set(low))},
        snr={i: round(float(snr[i]), 4) for i in sorted(set(targets) | set(anchors) | set(low))},
        created_utc=_dt.datetime.now(_dt.UTC).isoformat(),
        notes={
            "n_eligible": len(eligible),
            "n_passing_snr_gate": len(passing),
            "n_ineligible": len(ineligible),
            "min_snr": min_snr,
            "target_topics": len({topics.get(i, "other") for i in targets}),
            "anchor_topics": len({topics.get(i, "other") for i in anchors}),
            "spec_said": "60% anchors / 40% targets, which leaves 29 targets — short of the >= 40 the §1.4 claim needs",
        },
    )


def load_or_create(path: str | Path, **kwargs) -> ItemSplit:
    """Load the frozen split, or create and freeze it if it does not exist.

    Never overwrites. A split regenerated after results are known is not a
    held-out set, however identical the code that produced it.
    """
    p = Path(path)
    if p.exists():
        return ItemSplit.from_yaml(p)
    split = make_split(**kwargs)
    split.to_yaml(p)
    return split


def assert_frozen(path: str | Path, split: ItemSplit) -> None:
    """Raise if ``split`` disagrees with what is on disk."""
    p = Path(path)
    if not p.exists():
        raise SplitFrozenError(f"no frozen split at {p}")
    on_disk = ItemSplit.from_yaml(p)
    for field_name in ("targets", "anchors", "anchor_only_low_snr"):
        a, b = getattr(on_disk, field_name), getattr(split, field_name)
        if a != b:
            raise SplitFrozenError(
                f"{field_name} differs from the frozen split at {p}.\n"
                f"  frozen: {json.dumps(a)[:200]}\n"
                f"  now   : {json.dumps(b)[:200]}\n"
                f"The split is frozen deliberately — regenerating it after seeing "
                f"results means the targets were not held out."
            )


# --------------------------------------------------------------- amendments
#
# 2.4: "Persist the split — it must never be regenerated." It is not. An
# amendment is a separate file layered over the frozen one at load time, so the
# frozen artifact stays byte-identical, the amendment is auditable on its own,
# and every consumer sees the same effective split rather than one assembled by
# hand at each call site.

def load_effective_split(
    frozen_path: str | Path,
    *,
    amendments: Sequence[str | Path] | None = None,
) -> tuple[dict, list[dict]]:
    """Load the frozen split plus any amendments beside it.

    Returns ``(effective_split, applied)`` where ``applied`` is one provenance
    record per amendment, for the run snapshot. If ``amendments`` is None, every
    ``item_split.amendment_*.yaml`` next to the frozen file is applied in
    numeric order.

    An amendment may only ADD anchors, and only items that are neither targets
    nor already anchors. A target can never become an anchor: that would change
    what the claim is measured on, which is the thing the freeze protects.
    """
    frozen_path = Path(frozen_path)
    split = yaml.safe_load(frozen_path.read_text())

    if amendments is None:
        found = sorted(
            frozen_path.parent.glob("item_split.amendment_*.yaml"),
            key=lambda p: int("".join(c for c in p.stem.rsplit("_", 1)[-1] if c.isdigit()) or 0),
        )
    else:
        found = [Path(a) for a in amendments]

    targets = set(split["targets"])
    anchors = list(split["anchors"])
    applied: list[dict] = []

    for path in found:
        amd = yaml.safe_load(path.read_text())
        added = list(amd.get("add_anchors") or [])
        for item in added:
            if item in targets:
                raise ValueError(
                    f"{path.name} promotes {item!r}, which is a TARGET. An amendment "
                    f"may not reassign a target — that changes what the claim is "
                    f"measured on, which is exactly what the 2.4 freeze protects."
                )
            if item in anchors:
                raise ValueError(f"{path.name} adds {item!r}, already an anchor")
            anchors.append(item)
        if set(amd.get("remove_anchors") or ()):
            raise ValueError(
                f"{path.name} tries to remove anchors. Amendments are additive only: "
                f"removing one would change the cards every earlier run was scored on."
            )
        applied.append({
            "file": path.name,
            "amendment": amd.get("amendment"),
            "date": amd.get("date"),
            "rule": (amd.get("rule") or "").strip(),
            "added_anchors": added,
            "cannot_serve": sorted(amd.get("cannot_serve") or {}),
        })

    split["anchors"] = sorted(anchors)
    split["_amendments"] = applied
    return split, applied
