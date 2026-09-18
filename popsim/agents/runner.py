"""Resumable batch elicitation (checklist 0.4c).

    "**Checkpoint after every single call.** A 37k-call run at ~1k/day takes
     weeks — the runner must survive being stopped and resumed indefinitely."

The constraint is sharper than the checklist knew. Both shells this project is
driven from — the cloud container and the desktop Linux VM — **freeze between
tool calls**, so a background job does not progress and a run cannot simply be
left going. Every long run is therefore a sequence of short windows, and the
runner has to be restartable at a cell boundary with nothing lost and nothing
double-counted.

Two layers of persistence, and they do different jobs:

* ``llm/cache.py`` keys on model+prompt hash, so a repeated *call* is free. That
  makes a naive re-run cheap in quota but not in wall time: 12k cached calls
  still cost 12k prompt builds and parses.
* This store keys on (item, cluster, paraphrase, repeat) and is appended to as
  the run goes, so a resumed run skips the cell entirely.

One JSONL file per item. Item-major order means a stopped run leaves whole items
finished rather than every item part-done, which is what makes a partial store
usable for a partial evaluation.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .elicit import Paraphrase, RawElicitation, elicit_distribution

__all__ = ["ElicitationStore", "RunSlice", "plan_cells", "run_elicitation"]

POPULATION_CLUSTER = "all"


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


@dataclass
class ElicitationStore:
    """Append-only record store, one JSONL per item, keyed by model and profile."""

    root: Path
    model: str
    profile: str

    @property
    def dir(self) -> Path:
        return Path(self.root) / _slug(self.model) / _slug(self.profile)

    def path(self, item_id: str) -> Path:
        return self.dir / f"{_slug(item_id)}.jsonl"

    def done(self, item_id: str) -> set[tuple[str, int, int]]:
        p = self.path(item_id)
        if not p.exists():
            return set()
        out: set[tuple[str, int, int]] = set()
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.add((r["cluster_id"], int(r["paraphrase_id"]), int(r["repeat_id"])))
        return out

    def append(self, records: Sequence[RawElicitation]) -> None:
        if not records:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        by_item: dict[str, list[str]] = {}
        for r in records:
            by_item.setdefault(r.item_id, []).append(json.dumps(asdict(r)))
        for item_id, lines in by_item.items():
            with self.path(item_id).open("a") as fh:
                fh.write("\n".join(lines) + "\n")

    def load(self, items: Sequence[str] | None = None) -> pd.DataFrame:
        paths = ([self.path(i) for i in items] if items
                 else sorted(self.dir.glob("*.jsonl")) if self.dir.exists() else [])
        rows: list[dict] = []
        for p in paths:
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=[f.name for f in RawElicitation.__dataclass_fields__.values()]
        )

    def counts(self) -> dict[str, int]:
        if not self.dir.exists():
            return {}
        return {p.stem: sum(1 for line in p.read_text().splitlines() if line.strip())
                for p in sorted(self.dir.glob("*.jsonl"))}


@dataclass
class RunSlice:
    """What one window of a chunked run actually did."""

    cells_planned: int = 0
    cells_done_before: int = 0
    cells_this_slice: int = 0
    calls_this_slice: int = 0
    seconds: float = 0.0
    finished: bool = False
    stopped_reason: str = ""
    failure_rates: dict[str, float] = field(default_factory=dict)
    per_item_remaining: dict[str, int] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        left = sum(self.per_item_remaining.values())
        return "\n".join([
            f"  planned      {self.cells_planned} cells",
            f"  already done {self.cells_done_before}",
            (f"  this slice   {self.cells_this_slice} cells / "
             f"{self.calls_this_slice} calls in {self.seconds:.0f}s"),
            (f"  remaining    {left} cells"
             + ("  -> COMPLETE" if self.finished else f"  ({self.stopped_reason})")),
        ])


def plan_cells(items: Sequence[str], cluster_ids: Sequence[str]) -> list[tuple[str, str]]:
    """Item-major, so a stopped run leaves whole items finished."""
    return [(i, c) for i in items for c in cluster_ids]


def run_elicitation(
    bed,
    client,
    *,
    items: Sequence[str],
    cluster_ids: Sequence[str],
    paraphrases: Sequence[Paraphrase],
    n_paraphrase: int,
    n_repeat: int,
    store: ElicitationStore,
    temperature: float = 0.7,
    time_budget_s: float | None = None,
    population: bool = False,
    anchors_per_card: int | None = None,
    progress: bool = True,
    max_consecutive_failures: int = 30,
) -> RunSlice:
    """Fill the store for ``items x cluster_ids``, within a wall-clock budget.

    ``population=True`` elicits the item's *national* distribution instead, from
    the population card — one cell per item, cluster id ``all``. That is the B0b
    baseline and the ``predicted_level`` calibrator's level input.
    """
    t0 = time.monotonic()
    cells = ([(i, POPULATION_CLUSTER) for i in items] if population
             else plan_cells(items, cluster_ids))
    sl = RunSlice(cells_planned=len(cells))

    done_by_item = {i: store.done(i) for i in dict.fromkeys(i for i, _ in cells)}
    want = n_paraphrase * n_repeat
    remaining: list[tuple[str, str]] = []
    for item_id, cid in cells:
        have = sum(1 for (c, _p, _r) in done_by_item[item_id] if c == cid)
        if have >= want:
            sl.cells_done_before += 1
        else:
            remaining.append((item_id, cid))
    sl.per_item_remaining = {}
    for i, _c in remaining:
        sl.per_item_remaining[i] = sl.per_item_remaining.get(i, 0) + 1

    consecutive = 0
    fail_counts: dict[str, int] = {}
    n_records = 0
    last_item = None

    for item_id, cid in remaining:
        if time_budget_s is not None and time.monotonic() - t0 > time_budget_s:
            sl.stopped_reason = f"time budget {time_budget_s:.0f}s reached"
            break
        item = bed.item(item_id)
        card = (bed.population_card(item_id) if population
                else bed.card(item_id, cid, anchors_per_card=anchors_per_card))
        recs = elicit_distribution(
            card=card, item=item, client=client, paraphrases=paraphrases,
            n_paraphrase=n_paraphrase, n_repeat=n_repeat, temperature=temperature,
        )
        for r in recs:
            r.cluster_id = cid
            n_records += 1
            if r.ok:
                consecutive = 0
            else:
                consecutive += 1
                fail_counts[r.failure or "?"] = fail_counts.get(r.failure or "?", 0) + 1
        store.append(recs)
        sl.cells_this_slice += 1
        sl.calls_this_slice += len(recs)
        sl.per_item_remaining[item_id] -= 1
        if sl.per_item_remaining[item_id] <= 0:
            sl.per_item_remaining.pop(item_id, None)
        if progress and item_id != last_item:
            print(f"  {item_id}", flush=True)
            last_item = item_id
        if consecutive >= max_consecutive_failures:
            sl.stopped_reason = (
                f"{consecutive} calls failed in a row "
                f"({sorted(fail_counts.items())}) - a whole run of failures is not "
                f"a result; the store keeps what succeeded, so fix the provider "
                f"and resume"
            )
            break
    else:
        sl.finished = not sl.per_item_remaining
        sl.stopped_reason = "" if sl.finished else "cells skipped"

    sl.seconds = time.monotonic() - t0
    sl.failure_rates = ({k: v / n_records for k, v in fail_counts.items()}
                        if n_records else {})
    sl.notes = {"n_records": n_records, "model": store.model, "profile": store.profile}
    return sl
