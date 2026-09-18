"""
Stage 3: simulate the opinion distribution of a region or any demographic segment.

    # what the survey already knows -- served from microdata, no LLM, no cost
    python simulate_region.py --region south --item natheal

    # a question the survey never asked -- routed to the simulator
    python simulate_region.py --region south \
        --question "Do you favor or oppose federal subsidies for rooftop solar?" \
        --options "Strongly favor,Favor,Oppose,Strongly oppose"

    # finer segments
    python simulate_region.py --segment region=west,sex=female,age_band=25-34 --item natheal
    python simulate_region.py --list-regions

The oracle router decides
-------------------------
Every incoming question is compared against the harmonised codebook first. If the
survey already asks it, the answer is the exact weighted cross-tab, labelled
``observed`` -- no LLM call is made, and none should be, because a cross-tab is
not merely cheaper than a model, it is correct. Only genuinely unseen questions
reach the elicitation path, labelled ``simulated``.

This is the architectural answer to the obvious objection ("you built an
expensive way to reproduce a cross-tab you already had"): by construction the
model is only ever asked the questions the data cannot answer.

Every simulated answer carries an honesty box quoting the system's measured
out-of-sample error on the most similar validated items. It is read from the
benchmark output; if no benchmark has been run, the report says so instead of
inventing a number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

CODES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODES_DIR))

from shared.paths import load_config, load_dotenv  # noqa: E402
from shared.llm_client import is_mock  # noqa: E402
from M3_statcards.builder import StatCardBuilder  # noqa: E402
from M4_elicitation.elicit import DistributionElicitor  # noqa: E402
from M5_calibration.crossfit import CrossFittedCalibrator, load_crossfit_plan  # noqa: E402
from M6_aggregation.bootstrap import BootstrapAggregator  # noqa: E402
from M6_aggregation.scope import SegmentResolver, aggregate_segment  # noqa: E402
from M7_router.router import OracleRouter  # noqa: E402
from M9_evaluation.metrics import hellinger_distance  # noqa: E402


def parse_segment(text: Optional[str], region: Optional[str]) -> Dict[str, str]:
    """Parse ``region=south,sex=female`` into a query dict."""
    query: Dict[str, str] = {}
    if region:
        query["region"] = region.lower()
    if text:
        for part in text.split(","):
            if "=" not in part:
                raise SystemExit(f"bad --segment term {part!r}; expected field=value")
            k, v = part.split("=", 1)
            query[k.strip()] = v.strip()
    return query


def fmt_dist(labels: Sequence[str], probs: Sequence[float],
             ci: Optional[Sequence[Tuple[float, float]]] = None) -> str:
    width = max((len(str(l)) for l in labels), default=10)
    lines = []
    for n, (label, p) in enumerate(zip(labels, probs)):
        bar = "#" * int(round(p * 40))
        line = f"    {str(label):<{width}}  {p * 100:5.1f}%  {bar}"
        if ci is not None and n < len(ci):
            line += f"   [{ci[n][0] * 100:.1f}, {ci[n][1] * 100:.1f}]"
        lines.append(line)
    return "\n".join(lines)


def honesty_box(out_dir: Path, split_regime: str, topic: str) -> str:
    """
    Report measured out-of-sample error, or say plainly that none exists.

    A simulated estimate with no error bar attached is worse than no estimate,
    because it invites a decision it cannot support.
    """
    path = out_dir / f"benchmark_summary_{split_regime}.json"
    if not path.exists():
        return (
            "  HONESTY BOX\n"
            "    No benchmark has been run for this configuration, so this system's\n"
            "    out-of-sample error is UNKNOWN. Treat the numbers above as an\n"
            "    untested hypothesis, not an estimate.\n"
            "    Run:  python benchmark.py"
        )
    summary = json.loads(path.read_text(encoding="utf-8"))
    head = {h["system"]: h for h in summary.get("headline", [])}
    ours = next((v for k, v in head.items() if "ours" in k), None)
    b0a = next((v for k, v in head.items() if k.startswith("B0a")), None)
    if not ours:
        return "  HONESTY BOX\n    Benchmark file present but unreadable."

    lines = [
        "  HONESTY BOX",
        "    This is a SIMULATED estimate, not a measurement.",
        f"    On {summary['n_items_scored']} held-out questions from GSS 2024 "
        f"({summary['split_regime']} split),",
        f"    this system's mean error was W1 = {ours['W1']:.4f}"
        + (f"  (national-marginal baseline: {b0a['W1']:.4f})." if b0a else "."),
        f"    Within-cluster variance ratio {ours['var_ratio']:.2f} "
        f"(1.00 = correct dispersion).",
        f"    Between-cluster spread ratio {ours['between_SD_ratio']:.2f} "
        f"(near 0 = subgroups indistinguishable).",
    ]
    if summary.get("is_mock_run"):
        lines.append("    *** MOCK RUN: these error figures are noise. ***")
    verdict = summary.get("verdict", {})
    if "w1_improvement_vs_b0a_pct" in verdict:
        gain = verdict["w1_improvement_vs_b0a_pct"]
        lines.append(
            f"    Improvement over 'every subgroup answers like the country': {gain:+.1f}%."
        )
        if gain < 20:
            lines.append(
                "    This is BELOW the pre-registered 20% threshold - the system has not\n"
                "    demonstrated that cluster conditioning adds real subgroup signal."
            )
    lines.append(f"    No validated result exists for the topic '{topic}' specifically.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Simulate the opinion distribution of a region or segment"
    )
    ap.add_argument("--config", default=None)
    ap.add_argument("--region", default=None, help="northeast | midwest | south | west")
    ap.add_argument("--segment", default=None, help="field=value,field=value")
    ap.add_argument("--item", default=None, help="an existing codebook item id")
    ap.add_argument("--question", default=None, help="free-text question")
    ap.add_argument("--options", default=None, help="comma-separated response options")
    ap.add_argument("--list-regions", action="store_true")
    ap.add_argument("--list-questions", action="store_true")
    ap.add_argument("--top-segments", type=int, default=5)
    ap.add_argument("--model", default=None)
    ap.add_argument("--json-out", default=None, help="also write the answer as JSON here")
    args = ap.parse_args()

    load_dotenv()
    cfg = load_config(args.config)
    out_dir = cfg.out_dir

    builder = StatCardBuilder(
        tree_path=cfg.tree_path,
        stats_path=cfg.stats_path,
        codebook_path=cfg.codebook_path,
        split_path=cfg.split_path,
        pop_stats_path=cfg.pop_stats_path,
        individual_table_path=cfg.individual_table_path,
        granularity=cfg.granularity,
    )
    resolver = SegmentResolver(cfg.individual_table_path, granularity=cfg.granularity)

    if args.list_regions:
        print("Queryable fields and their values:\n")
        for field in ["region", "urban", "age_band", "sex", "education", "income_band", "race"]:
            try:
                print(f"  {field:<12} {', '.join(resolver.available_values(field))}")
            except KeyError:
                pass
        return 0

    if args.list_questions:
        rows = [
            {"item_id": i, "topic": m.get("topic", ""), "text": (m.get("text") or "")[:70]}
            for i, m in builder.codebook.items()
        ]
        print(pd.DataFrame(rows).head(60).to_string(index=False))
        print(f"\n({len(rows)} items total)")
        return 0

    if not args.item and not args.question:
        ap.error("give --item <id> or --question <text> (or --list-questions)")

    query = parse_segment(args.segment, args.region)
    segment = resolver.resolve(query)

    print("=" * 78)
    print(" B17 POPULATION SIMULATION")
    print("=" * 78)
    print(f"  Segment      : {segment.label}")
    print(f"  Covers       : {segment.pop_share * 100:.1f}% of US adults, "
          f"{segment.n_respondents} survey respondents")
    print(f"  Clusters     : {len(segment)} "
          f"({segment.summary()['n_partial_clusters']} partially inside the segment)")
    print(f"  Effective n  : {resolver.segment_n_eff(query):.0f}")

    # ---- route -----------------------------------------------------------
    codebook_list = list(builder.codebook.values())
    router = OracleRouter(codebook=codebook_list, threshold=0.85)

    if args.item:
        item_id, decision, sim = args.item, "OBSERVED", 1.0
        meta = builder.codebook.get(item_id)
        if meta is None:
            print(f"\nERROR: item {item_id!r} not in the codebook. Try --list-questions")
            return 1
        question_text = meta.get("text", item_id)
        labels = meta.get("scale", {}).get("labels", [])
    else:
        question_text = args.question
        route = router.route(question_text)
        decision, sim = route.decision, route.similarity_score
        item_id = route.matched_item_id if decision == "OBSERVED" else None
        if decision == "OBSERVED":
            meta = builder.codebook[item_id]
            labels = meta.get("scale", {}).get("labels", [])
        else:
            if not args.options:
                ap.error("--options is required for a question the survey does not ask")
            labels = [o.strip() for o in args.options.split(",") if o.strip()]

    print(f"\n  Question     : \"{question_text}\"")
    print(f"  Router       : {decision}" + (f"  (cosine {sim:.3f} vs '{item_id}')" if item_id else ""))

    # ---- observed path: exact cross-tab ---------------------------------
    if decision == "OBSERVED":
        hist = resolver.observed_histogram(item_id, query)
        if hist is None:
            print("\n  No responses for this item in this segment.")
            return 1
        if len(hist) != len(labels):
            labels = [f"code {n + 1}" for n in range(len(hist))]

        print("\n  ANSWER  [observed - exact weighted cross-tab from GSS 2024 microdata]")
        print(fmt_dist(labels, hist))
        print("\n  No LLM call was made. This is a measurement, not an estimate.")
        print("  Zero tokens, zero dollars.")

        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps(
                    {"provenance": "observed", "segment": segment.summary(),
                     "item_id": item_id, "labels": labels, "hist": hist},
                    indent=2), encoding="utf-8")
        return 0

    # ---- simulated path --------------------------------------------------
    cal_path = out_dir / "calibrated_target_distributions.parquet"
    plan_path = out_dir / "crossfit_plan.json"
    if not (cal_path.exists() and plan_path.exists()):
        print("\nERROR: no calibrated model found for this run.")
        print("Run:  python run_elicitation.py --stage all   &&   python benchmark.py")
        return 1

    model = args.model or cfg.llm.get("model", "claude-haiku-4-5")
    print(f"  Model        : {model}" + ("   [MOCK - output is noise]" if is_mock(model) else ""))

    # Refit the calibrator from the stored anchor pairs (cheap, no LLM calls).
    from benchmark import build_calibration_pairs  # noqa: E402

    anchor_cells = pd.read_parquet(out_dir / "cells_anchor_elicitation.parquet")
    plan = load_crossfit_plan(plan_path)
    pairs = build_calibration_pairs(anchor_cells, builder, plan)
    parent_of = {n["cluster_id"]: n["parent"] for n in builder.nodes if n.get("parent")}
    calibrator = CrossFittedCalibrator(tau=float(cfg.calibration.get("tau", 100.0)))
    calibrator.fit(pairs, parent_of=parent_of)

    synthetic = router._compile_synthetic_item(question_text, labels, "custom_scenario")
    builder.codebook[synthetic["item_id"]] = synthetic

    elicitor = DistributionElicitor(
        model=model,
        cache_path=cfg.cache_path,
        budget_path=cfg.budget_path,
        max_usd=float(cfg.llm.get("max_usd", 75.0)),
        temperature=float(cfg.llm.get("temperature", 0.7)),
    )

    print(f"\n  Eliciting {len(segment)} cluster agents "
          f"x {cfg.elicitation.get('n_paraphrases', 3) * cfg.elicitation.get('n_repeats', 3)} draws ...")

    cluster_dists: Dict[str, List[float]] = {}
    ensemble_members: Dict[str, List[List[float]]] = {}
    for cid in segment.cluster_ids:
        card = builder.build_stat_card(
            cluster_id=cid,
            target_item_id=synthetic["item_id"],
            split_regime=cfg.split_regime,
            max_anchors=cfg.elicitation.get("max_anchors_per_card", 12),
        )
        res = elicitor.elicit_cell(
            card=card,
            item_id=synthetic["item_id"],
            topic="custom_scenario",
            question_text=question_text,
            options=labels,
            stage="scenario",
            n_paraphrases=cfg.elicitation.get("n_paraphrases", 3),
            n_repeats=cfg.elicitation.get("n_repeats", 3),
        )
        if res.draws:
            cluster_dists[cid] = calibrator.calibrate(cid, res.mean_probabilities)
            ensemble_members[cid] = [calibrator.calibrate(cid, d) for d in res.draws]

    if not cluster_dists:
        print("  No cluster produced a usable answer.")
        return 1

    segment = resolver.resolve(query, restrict_to=list(cluster_dists))
    answer = aggregate_segment(cluster_dists, segment)

    # Bootstrap over ensemble members and cluster resampling.
    agg = BootstrapAggregator(
        n_boot=int(cfg.aggregation.get("n_boot", 500)),
        alpha=float(cfg.aggregation.get("alpha", 0.10)),
    )
    rng = np.random.RandomState(20260818)
    boots = []
    for _ in range(int(cfg.aggregation.get("n_boot", 500))):
        idx = rng.randint(0, len(segment.cluster_ids), len(segment.cluster_ids))
        picked, weights = [], []
        for j in idx:
            cid = segment.cluster_ids[j]
            members = ensemble_members[cid]
            picked.append(members[rng.randint(0, len(members))])
            weights.append(segment.weights[j])
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        boots.append((np.asarray(picked, dtype=float) * w[:, None]).sum(axis=0))
    boot_arr = np.asarray(boots)
    lo = np.percentile(boot_arr, 5, axis=0)
    hi = np.percentile(boot_arr, 95, axis=0)

    print(f"\n  ANSWER  [simulated - {segment.label}]")
    print(fmt_dist(labels, answer, list(zip(lo, hi))))

    # Which subgroups diverge most from the segment as a whole?
    div = sorted(
        ((hellinger_distance(cluster_dists[c], answer), c) for c in segment.cluster_ids),
        reverse=True,
    )[: args.top_segments]
    print(f"\n  MOST DIVERGENT SUBGROUPS (Hellinger distance from the {segment.label} average)")
    for dist, cid in div:
        node = builder.nodes_by_id.get(cid, {})
        top = int(np.argmax(cluster_dists[cid]))
        print(f"    {dist:.3f}  {node.get('definition_text', cid)[:58]:<58} "
              f"-> {labels[top]} {cluster_dists[cid][top] * 100:.0f}%")

    print()
    print(honesty_box(out_dir, cfg.split_regime, "custom_scenario"))

    spent = elicitor.budget.get_summary()
    print(f"\n  Cost of this scenario: ${spent['total_usd_spent']:.4f} cumulative "
          f"({spent['live_calls']} live / {spent['cached_calls']} cached calls)")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "provenance": "simulated",
                    "question": question_text,
                    "segment": segment.summary(),
                    "labels": labels,
                    "hist": [float(x) for x in answer],
                    "ci90": [[float(a), float(b)] for a, b in zip(lo, hi)],
                    "model": model,
                    "is_mock": is_mock(model),
                },
                indent=2), encoding="utf-8")
        print(f"  JSON -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
