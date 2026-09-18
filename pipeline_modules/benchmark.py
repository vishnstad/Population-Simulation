"""
Stage 2 of the pipeline: fit the calibrator and benchmark against the baselines.

    python benchmark.py                    # main run, config defaults
    python benchmark.py --baselines b0a b0b b1 b3 b4 b2   # include persona sampling
    python benchmark.py --split adversarial               # whole topics held out

What it does
------------
1. Loads the out-of-fold anchor elicitations and pairs each with its true
   histogram from the microdata.
2. Fits the cross-fitted calibrator (isotonic CDF + hierarchical variance
   restoration) on those pairs.
3. Applies it to the held-out target elicitations.
4. Scores the calibrated predictions and every baseline on the scoreable cells,
   and writes a results table.

Scoring rule
------------
A (cluster x item) cell is scored only if its Kish effective sample size clears
``evaluation.truth_min_neff``. Below that the "truth" is itself noise and a good
prediction would be punished for matching reality rather than the sample. An item
is scored only if at least ``min_clusters_per_item`` of its cells survive -- with
fewer, between-cluster statistics are meaningless.

Both filters shrink the evaluable set considerably. That shrinkage is reported,
not hidden: it is the real constraint a 3,309-respondent survey imposes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

CODES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODES_DIR))

from shared.paths import load_config, load_dotenv  # noqa: E402
from shared.llm_client import is_mock  # noqa: E402
from M3_statcards.builder import StatCardBuilder  # noqa: E402
from M5_calibration.crossfit import (  # noqa: E402
    CalibrationPair,
    CrossFittedCalibrator,
    load_crossfit_plan,
)
from M9_evaluation import baselines as B  # noqa: E402
from M9_evaluation.metrics import (  # noqa: E402
    aggregate_item_metrics,
    evaluate_cluster_predictions,
    expected_calibration_error,
    improvement_vs_baseline,
)


# ----------------------------------------------------------------------
def build_calibration_pairs(
    cells: pd.DataFrame, builder: StatCardBuilder, plan: Dict[str, int]
) -> List[CalibrationPair]:
    """Match each out-of-fold anchor prediction to its true cluster histogram."""
    pairs: List[CalibrationPair] = []
    for row in cells.itertuples():
        stat = builder.stats_dict.get((row.cluster_id, row.item_id))
        if stat is None:
            continue
        truth = [float(x) for x in stat.hist]
        raw = list(row.mean_probabilities)
        if len(truth) != len(raw):
            continue
        pairs.append(
            CalibrationPair(
                cluster_id=row.cluster_id,
                item_id=row.item_id,
                fold=plan.get(row.item_id, -1),
                raw_probs=raw,
                true_probs=truth,
                n_eff=float(stat.n_eff),
            )
        )
    return pairs


def scoreable_cells(
    builder: StatCardBuilder, cluster_ids: Sequence[str], item_id: str, min_neff: float
) -> List[str]:
    """Clusters whose truth for this item is precise enough to score against."""
    out = []
    for cid in cluster_ids:
        stat = builder.stats_dict.get((cid, item_id))
        if stat is not None and float(stat.n_eff) >= min_neff:
            out.append(cid)
    return out


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="B17 stage 2: calibrate and benchmark")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default=None, help="override evaluation.split_regime")
    ap.add_argument("--baselines", nargs="*", default=None,
                    help="subset of b0a b0b b1 b2 b3 b4")
    ap.add_argument("--min-neff", type=float, default=None)
    args = ap.parse_args()

    load_dotenv()
    cfg = load_config(args.config)
    out_dir = cfg.out_dir
    split_regime = args.split or cfg.split_regime
    min_neff = args.min_neff if args.min_neff is not None else cfg.evaluation.get("truth_min_neff", 30)
    min_clusters = cfg.evaluation.get("min_clusters_per_item", 5)
    want = [b.lower() for b in (args.baselines or cfg.evaluation.get("baselines", ["b0a", "b1", "b4"]))]

    anchor_path = out_dir / "cells_anchor_elicitation.parquet"
    target_path = out_dir / "cells_target_elicitation.parquet"
    for p in (anchor_path, target_path):
        if not p.exists():
            print(f"ERROR: {p.name} not found in {out_dir}.")
            print("Run:  python run_elicitation.py --stage all")
            return 1

    manifest = {}
    mf = out_dir / "elicitation_manifest.json"
    if mf.exists():
        manifest = json.loads(mf.read_text(encoding="utf-8"))
    mock_run = bool(manifest.get("is_mock", False)) or is_mock(manifest.get("model", ""))

    print("=" * 78)
    print(" B17 BENCHMARK  |  run:", cfg.run_name, "|  split:", split_regime)
    print("=" * 78)
    if mock_run:
        print("\n  *** MOCK RUN - the elicitations are random draws. ***")
        print("  *** Every number below is noise and must not be reported as a result. ***\n")

    builder = StatCardBuilder(
        tree_path=cfg.tree_path,
        stats_path=cfg.stats_path,
        codebook_path=cfg.codebook_path,
        split_path=cfg.split_path,
        pop_stats_path=cfg.pop_stats_path,
        individual_table_path=cfg.individual_table_path,
        granularity=cfg.granularity,
    )
    individuals = builder.individuals
    cluster_col = builder._cluster_col

    anchor_cells = pd.read_parquet(anchor_path)
    target_cells = pd.read_parquet(target_path)
    plan = load_crossfit_plan(out_dir / "crossfit_plan.json")

    # ---- 1. fit the calibrator on out-of-fold anchor pairs ---------------
    pairs = build_calibration_pairs(anchor_cells, builder, plan)
    parent_of = {n["cluster_id"]: n["parent"] for n in builder.nodes if n.get("parent")}
    calibrator = CrossFittedCalibrator(
        tau=float(cfg.calibration.get("tau", 100.0)),
        n_folds=int(cfg.calibration.get("n_folds", 3)),
        min_pairs_per_cluster=int(cfg.calibration.get("min_pairs_per_cluster", 5)),
    )
    report = calibrator.fit(pairs, parent_of=parent_of)
    calibrator.save(out_dir / "calibrator.json")

    print(f"\n[1/3] Calibrator fitted on {report['n_pairs']} out-of-fold anchor pairs")
    print(f"      across {report['n_clusters']} clusters "
          f"({report['n_clusters_pooled_to_parent']} shrunk toward parent)")
    print(f"      raw variance ratio BEFORE calibration: "
          f"{report['raw_variance_ratio_before_calibration']}  (1.0 = correct dispersion)")

    # ---- 2. apply to targets --------------------------------------------
    all_clusters = builder.leaf_ids
    pop_weight = {n["cluster_id"]: float(n.get("pop_share", 0.0)) for n in builder.nodes}
    anchor_ids = sorted({p.item_id for p in pairs})
    b1 = B.NearestAnchorBaseline(builder.codebook, anchor_ids)

    target_items = sorted(target_cells["item_id"].unique())
    per_item_rows: List[Dict[str, Any]] = []
    metric_bags: Dict[str, List[Dict[str, float]]] = {}
    calibrated_records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    b2_client = None
    if "b2" in want:
        from shared.llm_client import get_llm_client
        from shared.cache import PromptCache
        from shared.budget_guard import BudgetGuard

        b2_client = get_llm_client(
            model=manifest.get("model", cfg.llm.get("model")),
            cache=PromptCache(db_path=cfg.cache_path),
            budget_guard=BudgetGuard(state_file=cfg.budget_path, max_usd=float(cfg.llm.get("max_usd", 75.0))),
        )

    print(f"\n[2/3] Scoring {len(target_items)} held-out target items "
          f"(n_eff >= {min_neff}, >= {min_clusters} clusters per item)")

    for item_id in target_items:
        meta = builder.codebook.get(item_id, {})
        labels = meta.get("scale", {}).get("labels", [])
        scale_len = len(labels)
        sub = target_cells[target_cells["item_id"] == item_id]
        have = {r.cluster_id: list(r.mean_probabilities) for r in sub.itertuples()}

        cids = [c for c in scoreable_cells(builder, all_clusters, item_id, min_neff) if c in have]
        if len(cids) < min_clusters:
            skipped.append({"item_id": item_id, "n_scoreable": len(cids), "reason": "too few scoreable clusters"})
            continue

        truth = [[float(x) for x in builder.stats_dict[(c, item_id)].hist] for c in cids]
        if any(len(t) != scale_len for t in truth):
            skipped.append({"item_id": item_id, "n_scoreable": len(cids), "reason": "scale length mismatch"})
            continue

        weights = np.asarray([pop_weight.get(c, 0.0) for c in cids], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(cids))

        raw = [have[c] for c in cids]
        calibrated = [calibrator.calibrate(c, have[c]) for c in cids]

        for c, r, k in zip(cids, raw, calibrated):
            calibrated_records.append(
                {"cluster_id": c, "item_id": item_id,
                 "raw_probabilities": r, "calibrated_probabilities": k}
            )

        preds: Dict[str, List[List[float]]] = {"model": calibrated}

        pop_row = builder.pop_dict.get(item_id)
        pop_hist = [float(x) for x in pop_row.hist] if pop_row is not None else None

        if "b0a" in want and pop_hist and len(pop_hist) == scale_len:
            preds["b0a"] = B.baseline_b0a_oracle_marginal(pop_hist, len(cids))
        if "b0b" in want:
            # Fair version: the population-level histogram the agents themselves
            # imply, i.e. the population-weighted mixture of raw cluster outputs,
            # copied to every cluster. No per-cluster signal survives.
            mix = np.average(np.asarray(raw, dtype=float), axis=0, weights=weights)
            preds["b0b"] = B.baseline_b0b_predicted_marginal(list(mix / mix.sum()), len(cids))
        if "b1" in want:
            preds["b1"] = b1.predict(item_id, cids, builder.stats_dict, scale_len,
                                     pop_hist or [1.0 / scale_len] * scale_len)
        if "b3" in want and individuals is not None:
            b3 = B.baseline_b3_supervised_skyline(individuals, cids, cluster_col, item_id, scale_len)
            if b3 is not None:
                preds["b3"] = b3
        if "b4" in want:
            preds["b4"] = B.baseline_b4_uncalibrated(raw)
        if "b2" in want and b2_client is not None and individuals is not None:
            res = B.baseline_b2_persona_sampling(
                individuals, cids, cluster_col, item_id,
                meta.get("text", item_id), labels, b2_client,
                n_per_cluster=int(cfg.evaluation.get("b2_n_per_cluster", 20)),
            )
            if res.cluster_dists:
                preds["b2"] = [res.cluster_dists.get(c, pop_hist or [1.0 / scale_len] * scale_len)
                               for c in cids]

        row: Dict[str, Any] = {
            "item_id": item_id,
            "topic": meta.get("topic", ""),
            "scale_len": scale_len,
            "n_clusters_scored": len(cids),
        }
        for name, dists in preds.items():
            m = evaluate_cluster_predictions(dists, truth, weights)
            metric_bags.setdefault(name, []).append(m)
            row[f"{name}_w1"] = m["weighted_w1"]
            row[f"{name}_vr"] = m["variance_ratio"]
            row[f"{name}_bcsd"] = m["between_cluster_sd_ratio"]
            row[f"{name}_rho"] = m["rank_correlation_rho"]
        if "b0a" in preds:
            row["improvement_vs_b0a_pct"] = improvement_vs_baseline(
                row["model_w1"], row["b0a_w1"]
            )
        row["ece"] = round(expected_calibration_error(preds["model"], truth), 5)
        per_item_rows.append(row)

    if not per_item_rows:
        print("\n  No item cleared the scoring gate.")
        print(f"  Relax evaluation.truth_min_neff (currently {min_neff}) or "
              "evaluation.min_clusters_per_item, or use a coarser tree.")
        return 1

    per_item = pd.DataFrame(per_item_rows)
    per_item.to_csv(out_dir / f"benchmark_per_item_{split_regime}.csv", index=False)
    pd.DataFrame(calibrated_records).to_parquet(
        out_dir / "calibrated_target_distributions.parquet", index=False
    )

    # ---- 3. headline table ----------------------------------------------
    labels_map = {
        "model": "Calibrated cluster agent (ours)",
        "b0a": "B0a  national marginal, oracle",
        "b0b": "B0b  national marginal, predicted",
        "b1": "B1   nearest-anchor heuristic",
        "b2": "B2   per-individual persona sampling",
        "b3": "B3   supervised skyline (uses labels)",
        "b4": "B4   uncalibrated agent (ablation)",
    }
    order = ["model", "b0a", "b0b", "b1", "b2", "b3", "b4"]

    headline: List[Dict[str, Any]] = []
    for name in order:
        if name not in metric_bags:
            continue
        agg = aggregate_item_metrics(metric_bags[name])
        headline.append(
            {
                "system": labels_map[name],
                "W1": agg["weighted_w1"],
                "var_ratio": agg["variance_ratio"],
                "between_SD_ratio": agg["between_cluster_sd_ratio"],
                "rank_rho": agg["rank_correlation_rho"],
            }
        )
    head_df = pd.DataFrame(headline)

    b0a_w1 = next((h["W1"] for h in headline if h["system"].startswith("B0a")), None)
    model_w1 = headline[0]["W1"]
    model_bcsd = headline[0]["between_SD_ratio"]

    print(f"\n[3/3] Results  ({len(per_item)} items scored, "
          f"{len(skipped)} skipped for thin ground truth)")
    print("-" * 78)
    print(head_df.to_string(index=False))
    print("-" * 78)
    print("  W1               lower is better; Wasserstein-1 to the true cluster histograms")
    print("  var_ratio        predicted SD / true SD within cluster; target band [0.8, 1.2]")
    print("  between_SD_ratio predicted / true spread ACROSS clusters; near 0 = collapse")
    print("  rank_rho         Spearman correlation of cluster means; does it order subgroups")

    verdict: Dict[str, Any] = {}
    if b0a_w1:
        gain = improvement_vs_baseline(model_w1, b0a_w1)
        verdict["w1_improvement_vs_b0a_pct"] = gain
        verdict["claim_i_target_pct"] = 20.0
        verdict["claim_i_met"] = bool(gain >= 20.0)
        print(f"\n  Claim (i): W1 vs the oracle national marginal -> {gain:+.1f}%  "
              f"(spec requires >= +20%)  {'PASS' if gain >= 20 else 'FAIL'}")
    vr = headline[0]["var_ratio"]
    verdict["variance_ratio"] = vr
    verdict["claim_ii_met"] = bool(0.8 <= vr <= 1.2)
    print(f"  Claim (ii): variance ratio {vr:.3f} in [0.8, 1.2]?  "
          f"{'PASS' if 0.8 <= vr <= 1.2 else 'FAIL'}")

    print(f"\n  F2 check: between-cluster SD ratio = {model_bcsd:.3f}")
    if model_bcsd < 0.5:
        print("  WARNING: below 0.5. The model is emitting near-identical histograms for")
        print("  different subgroups (between-cluster collapse). Calibration cannot invent")
        print("  signal the base model refuses to produce. Escalate the elicitation model or")
        print("  enrich the anchor cards BEFORE spending a full budget.")

    summary = {
        "run_name": cfg.run_name,
        "split_regime": split_regime,
        "granularity": cfg.granularity,
        "model": manifest.get("model"),
        "is_mock_run": mock_run,
        "n_items_scored": len(per_item),
        "n_items_skipped": len(skipped),
        "min_neff": min_neff,
        "calibrator": report,
        "headline": headline,
        "verdict": verdict,
        "skipped_items": skipped[:50],
    }
    with open(out_dir / f"benchmark_summary_{split_regime}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  per-item CSV -> {out_dir / f'benchmark_per_item_{split_regime}.csv'}")
    print(f"  summary JSON -> {out_dir / f'benchmark_summary_{split_regime}.json'}")
    if mock_run:
        print("\n  Reminder: this was a MOCK run. Re-run with a real model before quoting anything.")
    print("\n  Next:  python simulate_region.py --region south --list-questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
