"""Layer 5 / §5.6 — is the win real, or memorized?

    "(1) **Direct probe:** ask each model for the actual cross-tab of each target
     item by name/wording ('In the 2022 GSS, what fraction of college-educated
     women said…'); items where the probe is accurate within W1 < 0.05 are
     flagged `leaky` and all headline metrics are reported with and without them."

The probe is the opposite of the elicitation prompt in every way that matters.
Elicitation never names a dataset, never names a year, and asks the model to
*reason* from a group's other answers. The probe names GSS, names the pooled
waves, and asks the model to *recall* a published cross-tab. If a model can do
the second, the first is not evidence of transfer for that item.

Two readings of a miss, and they are not the same
-------------------------------------------------
A model that cannot recall a cross-tab may still have absorbed the item's rough
shape from the literature, so a clean probe is a *necessary* condition for a
clean result, never a sufficient one. That is why §5.6 pairs it with the
leakage-resistant subset and the adversarial split rather than resting on it, and
why leakage is discussed as a limitation regardless of what this returns.

What "leaky" costs
------------------
Nothing is dropped silently. Flagged items stay in the run and every headline
number is reported twice — all targets, and all targets minus the flagged ones.
If the two agree, §5.6 stops being a weakness and becomes a strength.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..agents.elicit import parse_histogram
from .metrics import w1

__all__ = ["LeakageReport", "ProbeResult", "build_probe_prompt", "run_leakage_probe"]

#: The threshold §5.6 names. An item recalled this accurately is flagged.
LEAKY_W1 = 0.05


def build_probe_prompt(item: dict, scope_text: str, waves: list[int]) -> str:
    """Name the dataset, the waves and the subgroup, and ask for recall.

    Deliberately makes recall as easy as possible: the verbatim instrument text,
    the exact option labels, the survey's name and its waves. A probe that made
    recall hard would understate leakage, which is the failure direction that
    flatters the result.
    """
    opts = "\n".join(f'  - "{label}"' for label in item["labels"])
    yrs = ", ".join(str(w) for w in waves)
    return (
        f"You are being asked to recall published survey results, not to estimate.\n\n"
        f"In the US General Social Survey (GSS), waves {yrs}, respondents were asked:\n\n"
        f"  \"{item['text']}\"\n\n"
        f"Answer options:\n{opts}\n\n"
        f"Among {scope_text}, what percentage gave each answer in that survey?\n"
        f"Report the published weighted cross-tab as best you recall it. If you do "
        f"not recall it, give your best recollection anyway.\n\n"
        f'Reply with JSON only, keyed by the exact option text above:\n'
        f'{{"percentages": {{"<option text>": <number>, ...}}}}'
    )


@dataclass
class ProbeResult:
    item_id: str
    scope: str
    cluster_id: str
    w1: float
    #: What copying the national marginal onto this subgroup would have scored.
    #: The probe only demonstrates cross-tab recall if it beats this.
    w1_b0a: float
    ok: bool
    failure: str | None = None
    pred: list[float] | None = None
    truth: list[float] | None = None


@dataclass
class LeakageReport:
    model: str
    n_items: int
    n_scopes: int
    threshold: float = LEAKY_W1
    #: Mean recall W1 over the SUBGROUP scopes — the cross-tab question.
    per_item: dict[str, float] = field(default_factory=dict)
    #: Recall of the national topline alone, which is not the leakage that
    #: matters: every GSS topline is published and a model knowing one says
    #: nothing about whether it knows the breakdown.
    per_item_population: dict[str, float] = field(default_factory=dict)
    #: What B0a scores on the same subgroups, per item.
    per_item_b0a: dict[str, float] = field(default_factory=dict)
    leaky: list[str] = field(default_factory=list)
    leaky_by_threshold_only: list[str] = field(default_factory=list)
    probes: list[ProbeResult] = field(default_factory=list)
    failure_rates: dict[str, float] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        vals = [v for v in self.per_item.values() if np.isfinite(v)]
        return "\n".join([
            "Layer 5 — the direct leakage probe (§5.6)",
            f"  model       {self.model}",
            f"  scope       {self.n_items} items x {self.n_scopes} subgroups",
            (f"  subgroup recall W1   median {np.median(vals):.4f}"
             f"   best {min(vals):.4f}" if vals else "  no usable probes"),
            (f"  topline recall W1    median "
             f"{np.median([v for v in self.per_item_population.values() if np.isfinite(v)]):.4f}"
             if self.per_item_population else "  topline recall       none"),
            (f"  under the bare {self.threshold} threshold: "
             f"{len(self.leaky_by_threshold_only)} item(s)"),
            (f"  FLAGGED     {len(self.leaky)} item(s) — recalled within "
             f"{self.threshold} AND better than copying the topline (B0a): "
             f"{self.leaky or 'none'}"),
            f"  JSON ok     {self.failure_rates.get('ok', 0):.1%}",
            "",
            ("  A clean probe is necessary, not sufficient: a model can absorb "
             "an item's shape from the literature without recalling its "
             "cross-tab. Headline numbers are reported with and without the "
             "flagged items either way."),
        ])

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["probes"] = [asdict(p) for p in self.probes]
        return d


def run_leakage_probe(
    bed,
    client,
    *,
    items: list[str],
    scopes: list[tuple[str, str]] | None = None,
    n_repeat: int = 3,
    temperature: float = 0.7,
    out_dir: str | Path | None = None,
    progress: bool = True,
) -> LeakageReport:
    """Ask for the cross-tab by name; score the answer against the real one.

    ``scopes`` are (description, cluster_id) pairs. The description is what the
    prompt says; the cluster id is whose truth it is scored against. The default
    set is the population plus the largest leaves, described in the same plain
    words the stat card uses so a model that knows the cross-tab can match it.
    """
    if scopes is None:
        scopes = [("all US adults", "all")]
        for cid in bed.leaves_by_share[:3]:
            scopes.append((bed.card(items[0], cid).definition_text.rstrip("."), cid))

    waves = list(bed.cfg["bed.waves"])
    probes: list[ProbeResult] = []
    fails: dict[str, int] = {}
    n = 0
    model_seen = ""
    for item_id in items:
        item = bed.item(item_id)
        for scope_text, cid in scopes:
            truth = (bed.national(item_id) if cid == "all"
                     else bed.stats.hist(cid, item_id))
            if truth is None or np.asarray(truth).sum() <= 0:
                continue
            b0a = (0.0 if cid == "all"
                   else w1(bed.national(item_id), np.asarray(truth, dtype=float)))
            prompt = build_probe_prompt(item, scope_text, waves)
            hists = []
            for r in range(n_repeat):
                n += 1
                try:
                    resp = client.complete(
                        prompt, temperature=temperature, max_tokens=400,
                        paraphrase_id=900, repeat_id=r, json_mode=True)
                except Exception as exc:  # noqa: BLE001 - a failed probe is data
                    fails["provider"] = fails.get("provider", 0) + 1
                    probes.append(ProbeResult(item_id, scope_text, cid, float("nan"),
                                              b0a, False, f"{type(exc).__name__}"))
                    continue
                model_seen = model_seen or resp.model
                h, kind, _ = parse_histogram(resp.text, item["labels"])
                if h is None:
                    fails[kind or "malformed_json"] = fails.get(kind or "malformed_json", 0) + 1
                    probes.append(ProbeResult(item_id, scope_text, cid, float("nan"),
                                              b0a, False, kind))
                else:
                    hists.append(np.asarray(h, dtype=float))
            if hists:
                pred = np.mean(np.vstack(hists), axis=0)
                d = w1(pred, np.asarray(truth, dtype=float))
                probes.append(ProbeResult(
                    item_id, scope_text, cid, float(d), b0a, True,
                    pred=[float(x) for x in pred],
                    truth=[float(x) for x in np.asarray(truth, dtype=float)]))
        if progress:
            got = [p.w1 for p in probes
                   if p.item_id == item_id and p.ok and p.cluster_id != "all"]
            print(f"  {item_id:10s} recall W1 "
                  f"{np.mean(got) if got else float('nan'):.4f}", flush=True)

    # Two quantities, and only one of them is the leakage §5.6 is about.
    #
    # Every GSS topline is published, so a model reciting one tells you nothing
    # about whether it knows the *breakdown* — and the breakdown is the only
    # thing the system is claiming to predict. So the flag needs the probe to do
    # two things at once: recall a subgroup within the threshold, AND do it
    # better than simply copying the topline onto that subgroup would. A probe
    # that answers every subgroup with the national marginal is not leaking; it
    # is reproducing B0a, which is the baseline, not the claim.
    #
    # Averaging over subgroups rather than taking the best is deliberate too. On
    # a binary item W1 is |delta p|, so with four subgroups probed, one landing
    # inside 0.05 by chance is not unlikely, and a `min` rule flagged 22 of 39
    # items on the 8B arm largely on that.
    per_item: dict[str, float] = {}
    per_item_pop: dict[str, float] = {}
    per_item_b0a: dict[str, float] = {}
    for item_id in items:
        sub = [p for p in probes if p.item_id == item_id and p.ok and p.cluster_id != "all"]
        pop = [p.w1 for p in probes if p.item_id == item_id and p.ok and p.cluster_id == "all"]
        if sub:
            per_item[item_id] = float(np.mean([p.w1 for p in sub]))
            per_item_b0a[item_id] = float(np.mean([p.w1_b0a for p in sub]))
        if pop:
            per_item_pop[item_id] = float(np.mean(pop))

    rep = LeakageReport(
        model=model_seen or getattr(client, "last_model", "") or "",
        n_items=len(items), n_scopes=len(scopes),
        per_item=per_item,
        per_item_population=per_item_pop,
        per_item_b0a=per_item_b0a,
        leaky=sorted(i for i, v in per_item.items()
                     if v < LEAKY_W1 and v < per_item_b0a.get(i, float("inf"))),
        leaky_by_threshold_only=sorted(i for i, v in per_item.items() if v < LEAKY_W1),
        probes=probes,
        failure_rates={**{k: v / max(n, 1) for k, v in fails.items()},
                       "ok": sum(1 for p in probes if p.ok) / max(len(probes), 1)},
        notes={"waves": waves, "n_repeat": n_repeat,
               "scopes": [s for s, _ in scopes],
               "aggregation": "mean over SUBGROUP scopes; the population scope is "
                              "reported separately as topline recall",
               "flag_rule": "recall < 0.05 AND better than copying the topline "
                            "(B0a) onto the same subgroups"},
    )
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "leakage_report.json").write_text(json.dumps(rep.to_dict(), indent=2, default=str))
        (out / "leakage_summary.txt").write_text(rep.summary())
    return rep
