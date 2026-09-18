"""M7 — the scenario schema and compiler (checklist 6.3).

A scenario is turned into the same ``Item`` structure a survey question has, so
one code path serves both and a synthetic question cannot accidentally be scored
as if it had ground truth. Everything compiled here carries
``provenance: simulated`` for the rest of its life.

§1.6, said out loud in the type system rather than only in the writeup: product
and policy scenarios have no ground truth and never will. The held-out item score
is the honest proxy, and the honesty box is where that is said to the reader.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Scenario", "compile_scenario", "load_scenario"]


@dataclass
class Scenario:
    scenario_id: str
    decision_question: str
    context: str = ""
    labels: list[str] = field(default_factory=list)
    scale_type: str = "ordinal"
    segments_of_interest: list[dict] = field(default_factory=list)
    country: str = "US"
    topic: str = "other"

    def __post_init__(self) -> None:
        if len(self.context.split()) > 500:
            raise ValueError("scenario context is capped at 500 words (spec §M7)")
        if len(self.labels) < 2:
            raise ValueError("a response scale needs at least two options")


def load_scenario(path: str | Path) -> Scenario:
    return Scenario(**yaml.safe_load(Path(path).read_text()))


def compile_scenario(sc: Scenario) -> dict[str, Any]:
    """Wrap a scenario in the ``Item`` shape, marked synthetic and simulated."""
    text = sc.decision_question.strip()
    if sc.context.strip():
        text = f"{sc.context.strip()}\n\n{text}"
    return {
        "item_id": f"scenario:{sc.scenario_id}",
        "text": text,
        "labels": list(sc.labels),
        "codes": list(range(1, len(sc.labels) + 1)),
        "scale": {"type": sc.scale_type, "labels": list(sc.labels),
                  "codes": list(range(1, len(sc.labels) + 1))},
        "topic": sc.topic,
        "synthetic": True,
        "provenance": "simulated",
        "scenario": asdict(sc),
    }
