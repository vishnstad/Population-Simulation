"""Layer 2 — is the model reading the stat card at all? *(gate)*

    "**The permutation test.** Shuffle which stat card goes to which cluster.
     Re-run on 10 items. If W1 does not degrade sharply, the model is ignoring
     the conditioning and producing one hedged answer per item — **F2, the
     failure mode §7.2 calls the most likely one.**

     *Expected:* permuted W1 should land at or above the B0a baseline (~0.070).
     If permuted and real W1 are within noise of each other, the project's core
     premise is not working and you have ten weeks to respond, not two. **This
     is the single most valuable test in the document.**"

What the arms mean
------------------
* **real** — cluster c is scored against the card built for cluster c.
* **permuted** — cluster c is scored against a card built for some *other*
  cluster, via a derangement (no cluster keeps its own card), so any apparent
  conditioning cannot come from a fixed point.
* **B0a** — the national marginal copied to every cluster. No conditioning at
  all, and an unfairly strong baseline because it uses the *true* marginal.

Reading the result
------------------
The test asks one question: does the card carry information the model uses?

    real < permuted        the model is reading the card. How much better than
                           B0a is a separate question, for Gate 5.
    real ~= permuted       F2. The model is emitting one hedged answer per item
                           regardless of who it is describing.
    permuted << B0a        suspicious rather than good — a permuted card should
                           not beat "no information at all" by much, and if it
                           does, the items are being answered from item identity
                           alone, which is leakage (§5.6).

One caveat worth carrying into the writeup
------------------------------------------
A failed permutation test on a 7B local model is not the same evidence as a
failed one on a frontier model. §7.2's escalation ladder is explicit — frontier
model, richer cards, more anchors, *in that order* — so a red gate here means
re-run on the best available model before concluding anything about the premise.
The report records which model produced the numbers for exactly this reason.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..agents.elicit import (
    ElicitationBatch,
    Paraphrase,
    aggregate_histograms,
    elicit_distribution,
)
from ..agents.statcard import StatCard
from ..clustering.stats import weighted_histogram
from .metrics import w1

log = logging.getLogger(__name__)

__all__ = [
    "PermutationResult",
    "PreflightFailed",
    "derangement",
    "derangement_null",
    "run_permutation_test",
]


def derangement(items: Sequence[str], rng: np.random.Generator) -> dict[str, str]:
    """A permutation with no fixed point.

    A plain shuffle leaves about one cluster in e holding its own card, and
    those are exactly the cases that would make the permuted arm look better
    than it is.
    """
    items = list(items)
    n = len(items)
    if n < 2:
        raise ValueError("a derangement needs at least two clusters")
    for _ in range(1000):
        perm = list(rng.permutation(n))
        if all(perm[i] != i for i in range(n)):
            return {items[i]: items[perm[i]] for i in range(n)}
    raise RuntimeError("could not build a derangement")


def derangement_null(
    preds: Sequence[np.ndarray],
    truths: Sequence[np.ndarray],
    weights: Sequence[float],
    *,
    n_draws: int = 2000,
    seed: int = 0,
) -> tuple[float, float, int]:
    """The permuted arm's sampling distribution, for free.

    The permuted arm as specified is *one* derangement. That is one draw from a
    distribution with 8! orderings behind it even at the smoke scope, and the
    draw matters: on the 2026-09-16 smoke run seed 17 returned 0.2750 where the
    expected null was 0.2840, which is 0.009 of apparent degradation handed over
    by the seed alone — a third of the noise floor the gate is compared against.

    Reassigning already-collected histograms costs nothing, so the whole
    distribution is available at no quota. Returns (mean, p, n_draws_used),
    where p is the share of derangements scoring at or below the real arm —
    a small p means real beat almost every reassignment.
    """
    P = [np.asarray(x, dtype=float) for x in preds]
    T = [np.asarray(x, dtype=float) for x in truths]
    w = np.asarray(weights, dtype=float)
    n = len(P)
    if n < 2 or w.sum() <= 0:
        return float("nan"), float("nan"), 0
    w = w / w.sum()
    real = float(sum(w[i] * w1(P[i], T[i]) for i in range(n)))

    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_draws * 2):
        if len(vals) >= n_draws:
            break
        perm = rng.permutation(n)
        if np.any(perm == np.arange(n)):
            continue
        vals.append(float(sum(w[i] * w1(P[perm[i]], T[i]) for i in range(n))))
    if not vals:
        return float("nan"), float("nan"), 0
    arr = np.asarray(vals)
    return float(arr.mean()), float((arr <= real).mean()), int(arr.size)


@dataclass
class PermutationResult:
    model: str
    provider: str
    items: list[str]
    n_clusters: int
    n_calls: int
    real_w1: float
    permuted_w1: float
    baseline_w1: float
    noise_floor: float
    per_item: dict[str, dict[str, float]] = field(default_factory=dict)
    #: The permuted arm's expected value over many derangements rather than the
    #: one the seed happened to draw. A diagnostic, not the gate: the gate stays
    #: on the seeded real-vs-permuted gap, as frozen.
    null_mean_w1: float = float("nan")
    null_p_value: float = float("nan")
    n_derangements: int = 0
    failure_rates: dict[str, float] = field(default_factory=dict)
    refusal_over_threshold: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def degradation(self) -> float:
        """How much worse the permuted arm is, as a fraction of real W1."""
        if not np.isfinite(self.real_w1) or self.real_w1 <= 0:
            return float("nan")
        return (self.permuted_w1 - self.real_w1) / self.real_w1

    @property
    def ok(self) -> bool:
        """Gate 3 asks one question: is the model reading the card?

        So the gate is the real-vs-permuted gap, measured against the truth's
        own noise floor — a gap smaller than the uncertainty in the numbers
        being scored against is not a measurement.

        The checklist also expects permuted W1 to land at or above B0a, and with
        exact per-cluster truths it does: a random other cluster scores 0.140
        against 0.106 for the national marginal, measured on this bed. But that
        is an *expectation*, not the gate, because a model with collapsed
        variance (F1 — the documented failure of this whole method family)
        pushes its histograms toward uniform, and a flattened histogram can sit
        closer to a cluster's truth than another cluster's sharp one does.
        Failing a model for that would be failing it for the bug the calibration
        layer exists to repair, on the gate whose job is to detect a different
        bug entirely. It is reported as a diagnostic instead.
        """
        if not (np.isfinite(self.real_w1) and np.isfinite(self.permuted_w1)):
            return False
        return (self.permuted_w1 - self.real_w1) > self.noise_floor

    @property
    def permuted_beats_baseline(self) -> bool:
        return bool(
            np.isfinite(self.permuted_w1)
            and np.isfinite(self.baseline_w1)
            and self.permuted_w1 >= self.baseline_w1
        )

    def verdict(self) -> str:
        if not np.isfinite(self.real_w1):
            return "NO DATA — every call failed; check `popsim doctor`"
        gap = self.permuted_w1 - self.real_w1
        if gap <= self.noise_floor:
            return (
                "F2 — permuted scores within the truth's own noise of real. The model "
                "is not reading the stat card. Escalate per §7.2 (frontier model, then "
                "richer cards, then more anchors) BEFORE Phase 4."
            )
        return "The model is reading the card. Real is measurably better than permuted."

    def diagnostics(self) -> list[str]:
        """Things worth knowing that are not the gate."""
        out: list[str] = []
        if np.isfinite(self.permuted_w1) and not self.permuted_beats_baseline:
            out.append(
                f"permuted W1 {self.permuted_w1:.4f} is BELOW the B0a baseline "
                f"{self.baseline_w1:.4f}. Two readings: the predictions may be flattened "
                f"toward uniform (F1 variance collapse — check the variance ratio before "
                f"reading anything else into it), or the model may be answering partly "
                f"from item identity rather than the card (§5.6 leakage). The leakage "
                f"probe distinguishes them."
            )
        if np.isfinite(self.real_w1) and self.real_w1 > self.baseline_w1:
            out.append(
                f"real W1 {self.real_w1:.4f} is worse than the no-conditioning baseline "
                f"{self.baseline_w1:.4f} — raw predictions are currently worse than simply "
                f"copying the national marginal. If the gate above is GREEN this is what "
                f"the calibration layer (Phase 4) exists to fix and is not a Gate 3 "
                f"failure; if it is RED, it is the same finding said twice."
            )
        if np.isfinite(self.null_mean_w1) and np.isfinite(self.real_w1):
            seeded = self.permuted_w1 - self.real_w1
            expected = self.null_mean_w1 - self.real_w1
            out.append(
                f"permuted arm over {self.n_derangements} derangements: mean W1 "
                f"{self.null_mean_w1:.4f} (the seeded draw gave {self.permuted_w1:.4f}). "
                f"Gap against the expected null is {expected:+.4f} vs {seeded:+.4f} for "
                f"the seeded one; real beat {1 - self.null_p_value:.0%} of reassignments "
                f"(p = {self.null_p_value:.3f}). The gate is on the seeded gap, as "
                f"frozen — this says how much of that gap the seed chose."
            )
        if self.failure_rates.get("positional_response", 0) > 0:
            out.append(
                f"{self.failure_rates['positional_response']:.1%} of calls answered with "
                f"a positional array instead of the label-keyed object. Those are "
                f"recorded failures, not reordered histograms — but a high rate means the "
                f"model is fighting the contract and the prompt needs another look."
            )
        if self.failure_rates.get("label_mismatch", 0) > 0.05:
            out.append(
                f"{self.failure_rates['label_mismatch']:.1%} of calls returned option "
                f"names that do not match the item's labels; check the rendered options "
                f"before reading anything into the W1 numbers"
            )
        if self.refusal_over_threshold:
            out.append(
                f"items above the 10% refusal threshold, excluded per §M4: "
                f"{self.refusal_over_threshold}"
            )
        if self.failure_rates.get("ok", 1.0) < 0.9:
            out.append(
                f"only {self.failure_rates.get('ok', 0):.0%} of calls returned a usable "
                f"histogram; everything above rests on the rest being missing at random"
            )
        return out

    def summary(self) -> str:
        return "\n".join([
            "Layer 2 — the permutation test",
            f"  model        {self.model} ({self.provider})",
            (f"  scope        {len(self.items)} items x {self.n_clusters} clusters, "
             f"{self.n_calls} calls"),
            "",
            f"  real      W1 {self.real_w1:.4f}",
            f"  permuted  W1 {self.permuted_w1:.4f}   ({self.degradation:+.1%} vs real)",
            f"  B0a       W1 {self.baseline_w1:.4f}   (national marginal, no conditioning)",
            (f"  null mean W1 {self.null_mean_w1:.4f}   (permuted over "
             f"{self.n_derangements} derangements, p {self.null_p_value:.3f})"),
            f"  noise floor  {self.noise_floor:.4f}   (uncertainty in the truth itself)",
            "",
            (f"  JSON ok      {self.failure_rates.get('ok', 0):.1%}"
             f"   refusals {self.failure_rates.get('refusal', 0):.1%}"
             f"   malformed {self.failure_rates.get('malformed_json', 0):.1%}"),
            f"  => {'GREEN' if self.ok else 'RED'}: {self.verdict()}",
            *(f"  ! {d}" for d in self.diagnostics()),
        ])

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update({
            "ok": self.ok,
            "degradation": self.degradation,
            "verdict": self.verdict(),
            "diagnostics": self.diagnostics(),
            "permuted_beats_baseline": self.permuted_beats_baseline,
        })
        return d


class PreflightFailed(RuntimeError):
    """The provider path is broken, discovered on call 1 instead of call 960.

    Three separate setup faults were each found only by completing a whole run
    and reading 960 recorded failures afterwards:

    * the response contract expected a positional array (16 Sep, run 1);
    * nothing loaded `.env`, so every request carried an empty bearer token and
      came back 401 (16 Sep, attempt 2 at the frontier arm);
    * a per-second throttle was read as the day's allowance being gone, burning
      the only active provider on call 1 (16 Sep, attempt 3).

    Each of those is visible in the *first* response. The elicitation loop is
    deliberately forgiving — a run of thousands of calls over days must survive a
    provider having a bad afternoon, so a failed cell is recorded and the run
    continues — but that same tolerance turns a broken setup into a full run of
    nothing. So: one real call first, and a circuit breaker after that.
    """


def _preflight(
    *, item: dict, card, client, paraphrases, temperature: float
) -> None:
    """Make one real elicitation call and insist it comes back usable."""
    recs = elicit_distribution(
        card=card, item=item, client=client, paraphrases=paraphrases,
        n_paraphrase=1, n_repeat=1, temperature=temperature,
    )
    rec = recs[0]
    if rec.ok:
        log.info(
            "preflight ok: %s/%s answered %s in %d attempt(s)",
            rec.provider, rec.model, item["item_id"], rec.attempts,
        )
        return
    raise PreflightFailed(
        f"the first call failed ({rec.failure}) and the run was stopped before "
        f"spending the rest.\n"
        f"  item      {item['item_id']}\n"
        f"  provider  {rec.provider or '(none reached)'}\n"
        f"  model     {rec.model or '(none)'}\n"
        f"  detail    {rec.rationale or '(none recorded)'}\n"
        f"Run `popsim doctor` (add --set to name the provider) before retrying. "
        f"A cached first call cannot fail, so this is the live path."
    )


def run_permutation_test(
    *,
    items: Sequence[str],
    cluster_ids: Sequence[str],
    cards: dict[tuple[str, str], StatCard],   # (item_id, cluster_id)
    codebook: dict[str, dict],
    stats,
    frame: pd.DataFrame,
    cluster_of: pd.Series,
    codes: dict[str, list[int]],
    client,
    paraphrases: Sequence[Paraphrase],
    n_paraphrase: int = 1,
    n_repeat: int = 3,
    temperature: float = 0.7,
    noise_floor: float = 0.0273,
    seed: int = 17,
    n_null_draws: int = 2000,
    preflight: bool = True,
    max_consecutive_failures: int = 12,
    out_dir: str | Path | None = None,
    progress: bool = True,
) -> PermutationResult:
    rng = np.random.default_rng(seed)
    swap = derangement(cluster_ids, rng)

    if preflight and items and cluster_ids:
        first_item = dict(codebook[items[0]]) | {"item_id": items[0]}
        _preflight(
            item=first_item, card=cards[(items[0], cluster_ids[0])],
            client=client, paraphrases=paraphrases, temperature=temperature,
        )

    batch = ElicitationBatch()
    consecutive_failures = 0
    weights = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0).to_numpy()

    per_item: dict[str, dict[str, float]] = {}
    model = provider = ""

    for item_id in items:
        item = dict(codebook[item_id]) | {"item_id": item_id}
        cs = codes[item_id]

        col = frame[f"item_{item_id}"].to_numpy()
        ok = np.isin(col, cs)
        national = weighted_histogram(col[ok], weights[ok], cs)

        acc: dict[str, list[float]] = {"real": [], "permuted": [], "baseline": []}
        wts: list[float] = []
        # Kept so the permuted arm's whole sampling distribution can be
        # computed by reassignment afterwards, at no quota cost.
        real_preds: list[np.ndarray] = []
        real_truths: list[np.ndarray] = []
        null_wts: list[float] = []

        for cid in cluster_ids:
            truth = stats.hist(cid, item_id)
            if truth is None or truth.sum() <= 0:
                continue
            row = stats.frame[
                (stats.frame.cluster_id == cid) & (stats.frame.item_id == item_id)
            ]
            if row.empty or float(row.iloc[0]["n_eff"]) <= 0:
                continue
            wt = float(row.iloc[0]["weight_sum"])

            for arm, card_owner in (("real", cid), ("permuted", swap[cid])):
                recs = elicit_distribution(
                    card=cards[(item_id, card_owner)], item=item, client=client,
                    paraphrases=paraphrases, n_paraphrase=n_paraphrase,
                    n_repeat=n_repeat, temperature=temperature,
                    permuted_from=(card_owner if arm == "permuted" else None),
                )
                for r in recs:
                    r.cluster_id = cid       # scored against cid's truth either way
                    batch.add(r)
                    model = model or r.model
                    provider = provider or r.provider
                    # A run of thousands must tolerate individual failures, but a
                    # long unbroken streak is a broken setup, not a bad afternoon,
                    # and grinding out the rest buys nothing.
                    if r.ok:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            if out_dir:
                                out = Path(out_dir)
                                out.mkdir(parents=True, exist_ok=True)
                                batch.to_frame().to_parquet(
                                    out / "permutation_raw.parquet", index=False
                                )
                            raise PreflightFailed(
                                f"{consecutive_failures} calls failed in a row "
                                f"(last: {r.failure} — {r.rationale or 'no detail'}). "
                                f"Stopped after {len(batch.records)} of the planned "
                                f"calls rather than recording the rest as failures; "
                                f"what was collected is in permutation_raw.parquet. "
                                f"A whole run of failures is not a result."
                            )
                mean, _sd = aggregate_histograms(recs)
                if mean is not None:
                    acc[arm].append(w1(mean, truth))
                    if arm == "real":
                        real_preds.append(mean)
                        real_truths.append(np.asarray(truth, dtype=float))
                        null_wts.append(wt)
                else:
                    acc[arm].append(float("nan"))

            acc["baseline"].append(w1(national, truth))
            wts.append(wt)

        if wts:
            w = np.asarray(wts)
            per_item[item_id] = {
                arm: float(np.nansum(w * np.asarray(v)) / np.nansum(w[~np.isnan(v)]))
                if np.any(~np.isnan(v)) else float("nan")
                for arm, v in acc.items()
            }
            if len(real_preds) >= 2:
                nm, np_, nd = derangement_null(
                    real_preds, real_truths, null_wts, n_draws=n_null_draws, seed=seed
                )
                per_item[item_id].update(
                    null_mean=nm, null_p=np_, n_derangements=nd
                )
        if progress:
            got = per_item.get(item_id, {})
            print(f"  {item_id:10s} real {got.get('real', float('nan')):.4f}  "
                  f"permuted {got.get('permuted', float('nan')):.4f}  "
                  f"B0a {got.get('baseline', float('nan')):.4f}", flush=True)

    def _mean(arm: str) -> float:
        vals = [v[arm] for v in per_item.values() if np.isfinite(v.get(arm, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    result = PermutationResult(
        model=model, provider=provider, items=list(items),
        n_clusters=len(cluster_ids), n_calls=len(batch.records),
        real_w1=_mean("real"), permuted_w1=_mean("permuted"),
        baseline_w1=_mean("baseline"), noise_floor=noise_floor,
        null_mean_w1=_mean("null_mean"),
        null_p_value=_mean("null_p"),
        n_derangements=int(
            min((v["n_derangements"] for v in per_item.values()
                 if "n_derangements" in v), default=0)
        ),
        per_item=per_item, failure_rates=batch.failure_rates(),
        refusal_over_threshold=batch.items_over_refusal_threshold(),
        notes={
            "derangement_seed": seed,
            "n_paraphrase": n_paraphrase, "n_repeat": n_repeat,
            "temperature": temperature,
            "escalation": "a red gate on a small local model is not evidence about the "
                          "premise until §7.2's ladder has been tried on a stronger one",
        },
    )

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        batch.to_frame().to_parquet(out / "permutation_raw.parquet", index=False)
        (out / "gate3_report.json").write_text(json.dumps(result.to_dict(), indent=2, default=str))
        (out / "gate3_summary.txt").write_text(result.summary())
    return result
