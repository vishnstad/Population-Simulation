"""M4 — distributional elicitation (checklist 3.3).

    "`elicit.py` — structured JSON out, 3 paraphrases x 3 repeats, temp 0.7,
     retry <= 2, log refusals."

The call itself is small. What surrounds it is not, because every failure mode
here is one that looks like a result if it is not recorded.

* **Malformed JSON** is retried at most twice, then marked failed. The failure
  *rate* is a reported metric (spec §M4), so a model that needs three attempts
  to emit valid JSON must not look identical to one that gets it right first
  time.
* **Refusals** are counted separately from parse failures. An item refused by
  the model on sensitive content is missing data concentrated exactly where
  groups differ most; §M4 excludes items above 10% refusal and reports the
  exclusion. Folding refusals into "malformed" would hide that.
* **Histograms that do not sum to 1** are renormalized and logged. Silently
  renormalizing without a record would make a model that emits [50, 40, 30]
  indistinguishable from one that emits [45, 36, 27].
* **A histogram of the wrong length is never padded.** It is a failure. Padding
  would silently assign zero probability to a real option.
* **A positional answer is a failure, not a histogram.** The response contract is
  a JSON object keyed by the option *label*, not an array. Gate 3's smoke run is
  why: `natroad`'s verbatim instrument text is "(... are we spending too much,
  too little, or about the right amount on) Highways and bridges", which
  enumerates the options in a different order from the `codes`/`labels` order the
  truth histogram is built in. Asked for an array, the model answered in the
  *text's* order — W1 0.349 against the truth, 0.033 against the truth reversed.
  That is not a model capability finding, it is our prompt handing the model an
  ambiguity and then scoring it as if there were none. 19 items in the codebook
  have this conflict, including the whole `nat*` spending battery and `polviews`.
  Keying on the label removes the ambiguity rather than papering over it, and a
  positional reply is now counted as `positional_response` so the rate at which
  models ignore the contract is a reported number instead of a silent reordering.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..agents.statcard import StatCard, render_card
from ..llm.client import ContextWindowExceeded, LLMClient

__all__ = [
    "FAILURE_KINDS",
    "REFUSAL_MARKERS",
    "ElicitationBatch",
    "Paraphrase",
    "RawElicitation",
    "build_prompt",
    "elicit_distribution",
    "load_paraphrases",
    "parse_histogram",
]

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

#: Phrases that mark a refusal rather than a malformed answer. Kept deliberately
#: narrow: a model that merely hedges in prose before emitting JSON is not
#: refusing, and counting it as one would inflate the reported refusal rate.
REFUSAL_MARKERS = (
    "i can't", "i cannot", "i won't", "i will not", "i'm not able",
    "i am not able", "as an ai", "i don't feel comfortable",
    "i do not feel comfortable", "unable to provide", "cannot provide",
    "not appropriate", "i'm sorry, but",
)

#: A ``percentages`` payload carrying either an object (the contract) or an array
#: (a contract violation we still need to *recognise*, so it can be counted).
_JSON_RE = re.compile(
    r"\{\s*\"percentages\"\s*:\s*(?:\{[^{}]*\}|\[[^\]]*\])\s*\}", re.DOTALL
)

#: Every way a cell can fail, in the order they are reported.
FAILURE_KINDS = (
    "malformed_json", "refusal", "wrong_length", "positional_response",
    "label_mismatch", "context", "provider",
)


def _norm_label(s: str) -> str:
    """Labels are matched on this, so trivial differences do not fail a call."""
    return re.sub(r"\s+", " ", str(s)).strip().lower().rstrip(".")


@dataclass
class Paraphrase:
    id: int
    name: str
    preamble: str
    instruction: str


@dataclass
class RawElicitation:
    cluster_id: str
    item_id: str
    model: str
    provider: str
    paraphrase_id: int
    repeat_id: int
    hist: list[float] | None
    ok: bool
    failure: str | None = None       # malformed_json | refusal | wrong_length | context | provider
    renormalized: bool = False
    raw_sum: float | None = None
    attempts: int = 1
    cached: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    rationale: str = ""
    permuted_from: str | None = None  # set by the permutation test

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ElicitationBatch:
    records: list[RawElicitation] = field(default_factory=list)

    def add(self, r: RawElicitation) -> None:
        self.records.append(r)

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame([r.to_dict() for r in self.records])

    def failure_rates(self) -> dict[str, float]:
        n = len(self.records) or 1
        out: dict[str, float] = {}
        for kind in FAILURE_KINDS:
            out[kind] = sum(1 for r in self.records if r.failure == kind) / n
        out["ok"] = sum(1 for r in self.records if r.ok) / n
        out["renormalized"] = sum(1 for r in self.records if r.renormalized) / n
        return out

    def refusal_rate_by_item(self) -> dict[str, float]:
        by_item: dict[str, list[bool]] = {}
        for r in self.records:
            by_item.setdefault(r.item_id, []).append(r.failure == "refusal")
        return {k: sum(v) / len(v) for k, v in by_item.items()}

    def items_over_refusal_threshold(self, threshold: float = 0.10) -> list[str]:
        """Spec §M4: items above 10% refusal are excluded, and it is reported."""
        return sorted(k for k, v in self.refusal_rate_by_item().items() if v > threshold)


def load_paraphrases(path: str | Path | None = None) -> list[Paraphrase]:
    p = Path(path) if path else PROMPTS_DIR / "paraphrases" / "paraphrases.yaml"
    raw = yaml.safe_load(p.read_text())
    out = [Paraphrase(**r) for r in raw]
    if len({x.id for x in out}) != len(out):
        raise ValueError(f"duplicate paraphrase ids in {p}")
    return out


def build_prompt(card: StatCard, item: dict, paraphrase: Paraphrase) -> str:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    env = Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True,
    )
    env.filters["pct"] = lambda x: f"{round(float(x) * 100)}%"
    return env.get_template("elicit_hist.jinja").render(
        card_text=render_card(card), item=item, paraphrase=paraphrase,
    ).strip()


def parse_histogram(
    text: str, labels: Sequence[str]
) -> tuple[list[float] | None, str | None, float | None]:
    """Return (histogram summing to 1, failure kind, raw sum before normalizing).

    ``labels`` is the item's option labels in ``codes`` order — the order the
    truth histogram is built in. The returned histogram is always in that order,
    whatever order the model listed its keys in.
    """
    labels = list(labels)
    n_options = len(labels)
    low = text.strip().lower()
    if any(m in low[:400] for m in REFUSAL_MARKERS) and "percentages" not in low:
        return None, "refusal", None

    payload = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if m:
            try:
                payload = json.loads(m.group(0))
            except json.JSONDecodeError:
                payload = None
    if not isinstance(payload, dict) or "percentages" not in payload:
        return None, "malformed_json", None

    vals = payload["percentages"]

    if isinstance(vals, list):
        # Recognised, and refused. An array cannot say which option each number
        # belongs to, and for the 19 items whose wording enumerates its options
        # in a different order from `labels` the model demonstrably answers in
        # the wording's order. Accepting it would reintroduce exactly the bug
        # this contract exists to remove, on the items where it bites hardest.
        return None, "positional_response", None

    if not isinstance(vals, dict):
        return None, "malformed_json", None

    by_norm: dict[str, float] = {}
    for k, v in vals.items():
        try:
            by_norm[_norm_label(k)] = float(v)
        except (TypeError, ValueError):
            return None, "malformed_json", None

    wanted = [_norm_label(x) for x in labels]
    if len(by_norm) != n_options or any(w not in by_norm for w in wanted):
        # Not padded and not guessed at. A missing key would assign zero
        # probability to a real option; an unrecognised one means the model
        # answered a different question than the one whose truth we will score
        # it against.
        return None, "label_mismatch", None

    nums = [by_norm[w] for w in wanted]
    if any(v < 0 for v in nums):
        return None, "malformed_json", None

    total = float(sum(nums))
    if total <= 0:
        return None, "malformed_json", total
    return [v / total for v in nums], None, total


def elicit_distribution(
    *,
    card: StatCard,
    item: dict,
    client: LLMClient,
    paraphrases: Sequence[Paraphrase],
    n_paraphrase: int,
    n_repeat: int,
    temperature: float = 0.7,
    max_retries: int = 2,
    max_tokens: int = 400,
    permuted_from: str | None = None,
) -> list[RawElicitation]:
    """One (cluster, item) cell: n_paraphrase x n_repeat draws."""
    if n_paraphrase > len(paraphrases):
        raise ValueError(
            f"n_paraphrase={n_paraphrase} but only {len(paraphrases)} are checked in. "
            f"Paraphrases are hand-written on purpose (3.2) — add one to the YAML "
            f"rather than generating it."
        )

    labels = list(item["labels"])
    out: list[RawElicitation] = []

    for p_idx in range(n_paraphrase):
        paraphrase = paraphrases[p_idx]
        prompt = build_prompt(card, item, paraphrase)
        for r_idx in range(n_repeat):
            rec = RawElicitation(
                cluster_id=card.cluster_id, item_id=item["item_id"],
                model="", provider="", paraphrase_id=paraphrase.id, repeat_id=r_idx,
                hist=None, ok=False, permuted_from=permuted_from,
            )
            last_kind = None
            for attempt in range(max_retries + 1):
                try:
                    resp = client.complete(
                        prompt, temperature=temperature, max_tokens=max_tokens,
                        paraphrase_id=paraphrase.id,
                        # The retry must be its own draw, or a cache hit would
                        # replay the same malformed text forever.
                        repeat_id=r_idx * 100 + attempt,
                        json_mode=True,
                    )
                except ContextWindowExceeded as exc:
                    rec.failure, rec.rationale = "context", str(exc)[:200]
                    break
                except Exception as exc:  # noqa: BLE001 - see below
                    # Deliberately broad. A run is thousands of calls over days,
                    # and every provider-side failure mode (quota exhausted,
                    # transport reset, a provider returning an unexpected shape)
                    # must become a recorded failed cell rather than kill the
                    # run. The kind is kept in `rationale` so the distribution
                    # of causes is recoverable from the parquet afterwards.
                    rec.failure = "provider"
                    rec.rationale = f"{type(exc).__name__}: {exc}"[:200]
                    break

                rec.model, rec.provider = resp.model, resp.provider
                rec.cached = resp.cached
                rec.prompt_tokens += resp.prompt_tokens
                rec.completion_tokens += resp.completion_tokens
                rec.attempts = attempt + 1

                hist, kind, raw_sum = parse_histogram(resp.text, labels)
                if hist is not None:
                    rec.hist, rec.ok = hist, True
                    rec.raw_sum = raw_sum
                    rec.renormalized = raw_sum is not None and abs(raw_sum - 100.0) > 0.5
                    break
                last_kind = kind
                if kind == "refusal":
                    break  # retrying a refusal just spends quota
            if not rec.ok and rec.failure is None:
                rec.failure = last_kind or "malformed_json"
            out.append(rec)
    return out


def aggregate_histograms(records: Sequence[RawElicitation]) -> tuple[np.ndarray | None, float]:
    """Mean histogram over successful draws, plus the ensemble spread.

    The spread is retained as ``elicit_sd`` for M6's uncertainty. Note the spec
    averages *after* isotonic recalibration (M5 step 3); this raw mean is for
    diagnostics and for the permutation test, which has no calibrator yet.
    """
    good = [np.asarray(r.hist, dtype=float) for r in records if r.ok and r.hist]
    if not good:
        return None, float("nan")
    stack = np.vstack(good)
    return stack.mean(axis=0), float(stack.std(axis=0).mean())
