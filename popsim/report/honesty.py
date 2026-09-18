"""The honesty box (checklist 6.5, spec §M10). **No report renders without it.**

    "a mandatory **honesty box**: 'This is a simulated estimate. On the *k* most
     similar validated questions, this system's error was W1 = x (baseline y).'"

A scenario answer has no ground truth and never will (§1.6). The only honest
thing that can be said about its accuracy is how the same system did on the
*measured* questions most like it — so that sentence travels with every simulated
answer, and this module is the only way to produce it.

"Most similar" uses the same TF-IDF router that decides observed-vs-simulated, on
purpose: if a question is close enough to a scored item for its number to be
informative, it is close enough that the router nearly routed it to the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["HonestyBox", "build_honesty_box"]


@dataclass
class HonestyBox:
    provenance: str                     # "observed" | "simulated"
    nearest: list[dict[str, Any]] = field(default_factory=list)
    mean_w1: float = float("nan")
    mean_baseline: float = float("nan")
    noise_floor: float = float("nan")
    model: str = ""
    n_validated: int = 0
    note: str = ""

    def sentence(self) -> str:
        if self.provenance == "observed":
            return ("This answer is **observed**, not simulated: the question is "
                    "already in the survey, so it is the weighted cross-tab and "
                    "no model was asked.")
        if not self.nearest:
            return ("This is a **simulated estimate**, and no validated question "
                    "is close enough to it to say how accurate the system is on "
                    "questions like this. Treat the numbers as illustrative.")
        names = ", ".join(f"`{n['item_id']}`" for n in self.nearest)
        n = len(self.nearest)
        which = ("the one validated question closest to it" if n == 1
                 else f"the {n} validated questions closest to it")
        return (
            f"This is a **simulated estimate** — there is no ground truth for it "
            f"and there never will be. On {which} ({names}), this system's "
            f"measured error was "
            f"**W1 = {self.mean_w1:.4f}**, against a national-marginal baseline of "
            f"{self.mean_baseline:.4f} and a floor of {self.noise_floor:.4f} set by "
            f"noise in the survey data itself. Model: {self.model or 'unspecified'}."
        )


def build_honesty_box(
    question: str,
    router,
    layer4_report: dict[str, Any] | None,
    *,
    provenance: str = "simulated",
    k: int = 3,
    min_similarity: float = 0.18,
) -> HonestyBox:
    """The nearest *scored* items and what the system's error was on them.

    ``min_similarity`` is 0.18 because that is the lowest similarity any
    hand-labeled duplicate reaches (``codebooks/router_duplicate_pairs.yaml``).
    Below it the nearest codebook item is not about the same subject, and
    quoting its error would be worse than saying nothing: an accuracy figure
    attached to an unrelated question is exactly the demo-ware §F7 warns about.

    The labeled set's duplicate and non-duplicate similarity distributions
    overlap heavily — the non-duplicates were written to be *about the same
    subject* as their item — so TF-IDF cannot separate "the same question" from
    "the same subject". For the honesty box that is fine: same-subject is what
    makes a validated number informative. For the router it is the reason
    precision is held above recall.
    """
    if provenance == "observed":
        return HonestyBox(provenance="observed")
    if not layer4_report:
        return HonestyBox(provenance="simulated",
                          note="no Layer 4 report on disk — run `popsim layer4`")

    scored = {r["item_id"]: r for r in layer4_report.get("per_item", [])}
    d = router.route(question, top_k=max(k * 6, 12))
    ranked = ([(d.matched_item, d.similarity)] if d.matched_item else []) + list(d.runners_up)
    nearest = []
    for item_id, sim in ranked:
        if item_id in scored and sim >= min_similarity and len(nearest) < k:
            r = scored[item_id]
            w = r.get("w1") or {}
            nearest.append({
                "item_id": item_id, "similarity": float(sim),
                "w1": float(r.get("w1_system", w.get("system", float("nan")))),
                "baseline": float(r.get("w1_B0a", w.get("B0a", float("nan")))),
                "text": r.get("topic", ""),
            })
    if not nearest:
        return HonestyBox(provenance="simulated", model=layer4_report.get("model", ""),
                          n_validated=len(scored),
                          note="no scored item is textually close to this question")
    return HonestyBox(
        provenance="simulated", nearest=nearest,
        mean_w1=sum(n["w1"] for n in nearest) / len(nearest),
        mean_baseline=sum(n["baseline"] for n in nearest) / len(nearest),
        noise_floor=float(layer4_report.get("noise_floor", float("nan"))),
        model=layer4_report.get("model", ""), n_validated=len(scored),
    )
