#!/usr/bin/env python
"""Run Gate 3 against models whose behaviour is known, before spending inference.

STATUS.md records that this exercise is what changed the gate — the first
version also required permuted W1 >= B0a, and that failed the *oracle*, because
adding noise to the oracle's histograms flattens them toward uniform and a
flattened histogram can sit closer to a cluster's truth than another cluster's
sharp one. That is F1, the variance collapse the calibration layer exists to
repair, so failing a model for it on the gate whose job is to detect F2 was the
wrong answer to the wrong question.

The exercise was not checked in, which meant the reasoning behind the gate's
shape could not be re-run. It is checked in now, with two fakes added for the
response contract:

    oracle      the true histogram of whoever the card describes    -> GREEN
    hedger      one fixed answer per item, ignores the card         -> RED, F2
    refuser     refuses everything                                  -> RED, no data
    positional  right answer, but as a JSON array                   -> RED, all calls
                                                                       recorded as
                                                                       positional_response
    textorder   right answer, keyed, but listed in the order the
                question wording names the options                 -> GREEN, scored
                                                                       correctly

The last two are the regression guard for the natroad finding. `textorder` is
the behaviour the 7B actually showed on natroad; under the old positional
contract it scored W1 0.349 against a truth it was 0.033 away from.

Usage:  python scripts/gate3_dryrun.py [--n-items 4] [--n-clusters 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from popsim.agents.elicit import build_prompt, load_paraphrases
from popsim.agents.statcard import (
    battery_near_duplicates,
    make_statcard,
)
from popsim.calibration.crossfit import make_crossfit_plan
from popsim.clustering.partition import build_cluster_tree
from popsim.clustering.stats import compute_cluster_stats
from popsim.config import load_config
from popsim.data.adapters.gss import load_gss
from popsim.data.pool import POST_FREEZE_DOCTRINE_FLAGS
from popsim.evalx.gate3 import run_permutation_test
from popsim.llm.client import LLMResponse

ROOT = Path(__file__).resolve().parent.parent


class FakeClient:
    """Answers from a known rule, keyed on the exact prompt it is handed.

    The prompt -> (item, cluster) map is built by rendering every combination up
    front, so the fake never has to guess which card it is looking at and a
    change to the template cannot silently break the mapping.
    """

    def __init__(self, behaviour: str, lookup: dict[str, tuple[str, str]], truths, codebook):
        self.behaviour = behaviour
        self.lookup = lookup
        self.truths = truths
        self.codebook = codebook
        self.rng = np.random.default_rng(3)
        self.unmatched = 0

    def complete(self, prompt: str, **kw) -> LLMResponse:
        key = (self.behaviour, "fake")
        if self.behaviour == "refuser":
            return LLMResponse(text="I cannot provide estimates about this group.",
                               model=key[0], provider=key[1])

        hit = self.lookup.get(prompt.strip())
        if hit is None:
            self.unmatched += 1
            return LLMResponse(text="{}", model=key[0], provider=key[1])
        item_id, cluster_id = hit
        labels = list(self.codebook[item_id]["labels"])

        if self.behaviour == "hedger":
            # one fixed answer per item, whoever the card describes
            h = np.full(len(labels), 1.0 / len(labels))
        else:
            h = np.asarray(self.truths[(cluster_id, item_id)], dtype=float)
            h = h / h.sum()
            h = np.clip(h + self.rng.normal(0, 0.03, h.size), 1e-6, None)
            h = h / h.sum()
        pct = [round(float(x) * 100, 2) for x in h]

        if self.behaviour == "positional":
            return LLMResponse(text=json.dumps({"percentages": pct}),
                               model=key[0], provider=key[1])
        if self.behaviour == "textorder":
            # keyed, but listed in the order the question wording names them —
            # the natroad behaviour. Keys must make this harmless.
            pairs = list(zip(labels, pct))[::-1]
            return LLMResponse(text=json.dumps({"percentages": dict(pairs)}),
                               model=key[0], provider=key[1])
        return LLMResponse(text=json.dumps({"percentages": dict(zip(labels, pct))}),
                           model=key[0], provider=key[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-items", type=int, default=4)
    ap.add_argument("--n-clusters", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(str(ROOT / "configs/gss_main.yaml"))
    codebook = yaml.safe_load(
        (ROOT / cfg["paths.codebooks"] / "gss_items.yaml").read_text()
    )["items"]
    split = yaml.safe_load((ROOT / cfg["item_split.freeze_path"]).read_text())
    all_items = sorted(codebook)
    codes = {i: codebook[i]["codes"] for i in all_items}

    table = load_gss(cfg.bed_file, all_items, waves=cfg["bed.waves"])
    tree = build_cluster_tree(
        table.frame, axes=cfg["partition.axes"], item_codes=codes,
        min_cell=cfg["partition.min_cell"], k_target=cfg["partition.k_target"],
    )
    assign = tree.assign(table.frame)
    stats = compute_cluster_stats(table.frame, assign, all_items, codes)
    plan = make_crossfit_plan(
        split["anchors"], split["targets"],
        {i: codebook[i]["topic"] for i in all_items},
        n_folds=cfg["calibration.crossfit_folds"],
    )

    ranked = [i for i in sorted(split["targets"], key=lambda i: -split["snr"].get(i, 0.0))
              if i not in POST_FREEZE_DOCTRINE_FLAGS]
    items = ranked[: args.n_items]
    leaves = sorted(tree.leaves(), key=lambda n: -n.pop_share)
    idx = np.linspace(0, len(leaves) - 1, min(args.n_clusters, len(leaves)))
    leaves = [leaves[i] for i in sorted(set(idx.round().astype(int)))]
    cluster_ids = [n.cluster_id for n in leaves]

    near_dupes = battery_near_duplicates(all_items)
    cards, truths, lookup = {}, {}, {}
    paraphrases = load_paraphrases()
    for item_id in items:
        for cid in cluster_ids:
            card = make_statcard(
                cluster_id=cid, tree=tree, stats=stats, frame=table.frame,
                cluster_of=assign, target_item=item_id, codebook=codebook,
                anchor_pool=plan.anchors_for(item_id), near_duplicates=near_dupes,
                anchor_fold=plan.target_fold.get(item_id),
                anchors_per_card=cfg["elicitation.anchors_per_card"],
                fold_of=plan.fold_of,
            )
            cards[(item_id, cid)] = card
            h = stats.hist(cid, item_id)
            truths[(cid, item_id)] = h if h is not None else np.ones(len(codes[item_id]))
            item = dict(codebook[item_id]) | {"item_id": item_id}
            for p in paraphrases:
                lookup[build_prompt(card, item, p).strip()] = (item_id, cid)

    print(f"{len(items)} items x {len(cluster_ids)} clusters   items={items}\n")
    rows = []
    for behaviour in ("oracle", "hedger", "refuser", "positional", "textorder"):
        client = FakeClient(behaviour, lookup, truths, codebook)
        res = run_permutation_test(
            items=items, cluster_ids=cluster_ids, cards=cards, codebook=codebook,
            stats=stats, frame=table.frame, cluster_of=assign, codes=codes,
            client=client, paraphrases=paraphrases,
            n_paraphrase=1, n_repeat=3,
            temperature=cfg["elicitation.temperature"],
            noise_floor=cfg["evaluation.pass_marks"]["all_targets"]["noise_floor"],
            seed=cfg["seed"], out_dir=None, progress=False,
            # The `refuser` and `positional` fakes exist to prove the gate
            # reports NO DATA, so they must be allowed to fail every call.
            preflight=False, max_consecutive_failures=10_000,
        )
        fr = res.failure_rates
        rows.append((behaviour, res.ok, res.real_w1, res.permuted_w1, res.null_mean_w1,
                     res.baseline_w1, fr.get("ok", 0), fr.get("positional_response", 0),
                     client.unmatched))
        print(f"{behaviour:11s} {'GREEN' if res.ok else 'RED  '}  "
              f"real {res.real_w1:.4f}  perm {res.permuted_w1:.4f}  "
              f"null {res.null_mean_w1:.4f}  B0a {res.baseline_w1:.4f}  "
              f"ok {fr.get('ok', 0):.0%}  positional {fr.get('positional_response', 0):.0%}"
              + (f"  UNMATCHED {client.unmatched}" if client.unmatched else ""))

    print()
    expect = {"oracle": True, "hedger": False, "refuser": False,
              "positional": False, "textorder": True}
    bad = [b for b, ok, *_ in rows if ok != expect[b]]
    unmatched = [b for b, *_rest in rows if _rest[-1]]
    if bad:
        print(f"UNEXPECTED VERDICTS: {bad}")
    if unmatched:
        print(f"PROMPT LOOKUP MISSED for: {unmatched} — the template changed shape")
    if not bad and not unmatched:
        print("all five fakes behaved as the gate's design says they should")
    return 1 if (bad or unmatched) else 0


if __name__ == "__main__":
    raise SystemExit(main())
