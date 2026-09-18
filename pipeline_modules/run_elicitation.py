"""
Stage 1 of the pipeline: elicit raw distributions from the LLM.

This is the only script that spends money, and the only slow one. It is
resumable: interrupt it, re-run the same command, and it continues from where it
stopped. Completed calls live in a SQLite cache and are never paid for twice.

    python run_elicitation.py --stage all              # anchors then targets
    python run_elicitation.py --stage anchors          # calibrator training data
    python run_elicitation.py --stage targets          # held-out evaluation set
    python run_elicitation.py --stage all --dry-run    # cost estimate, no calls
    python run_elicitation.py --stage all --model mock-model   # free wiring test

Two stages, and the difference between them is the whole point
--------------------------------------------------------------
``anchors``
    Items whose true histograms we know. Each anchor is elicited from a stat card
    built WITHOUT its own cross-fit fold, so the prediction is made under exactly
    the conditions a held-out target faces. The resulting (prediction, truth)
    pairs are the only valid training data for the calibrator.

``targets``
    Held-out items. Never appear on any card. These are what the benchmark scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

CODES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODES_DIR))

from shared.paths import load_config, load_dotenv  # noqa: E402
from shared.llm_client import is_mock, model_info  # noqa: E402
from M3_statcards.builder import StatCardBuilder  # noqa: E402
from M4_elicitation.elicit import DistributionElicitor, aggregate_draws  # noqa: E402
from M5_calibration.crossfit import make_crossfit_plan, save_crossfit_plan  # noqa: E402


def select_items(builder: StatCardBuilder, cfg, role: str, limit: int) -> List[str]:
    """
    Pick which items to spend budget on.

    Targets are chosen by descending debiased heterogeneity (``h_icc``): the items
    where clusters genuinely differ. That is deliberate, not cherry-picking -- the
    falsifiable claim is explicitly about top-heterogeneity items, because on an
    item where every subgroup answers identically the national marginal is
    unbeatable by construction and nothing is learned either way.
    """
    regime = cfg.split_regime
    role_col = f"role_{regime}" if f"role_{regime}" in builder.split_df.columns else "role"
    het_col = "h_icc_coarse" if cfg.granularity == "coarse" else "h_icc"
    if het_col not in builder.split_df.columns:
        het_col = "h_icc"

    df = builder.split_df[builder.split_df[role_col] == role].copy()

    # Only keep items with a usable ordinal scale and cluster statistics present.
    keep = []
    for iid in df["item_id"]:
        meta = builder.codebook.get(iid)
        if not meta:
            continue
        labels = meta.get("scale", {}).get("labels", [])
        if 2 <= len(labels) <= 7:
            keep.append(iid)
    df = df[df["item_id"].isin(keep)]

    if cfg.elicitation.get("target_selection", "heterogeneity") == "heterogeneity":
        df = df.sort_values(het_col, ascending=False)
    else:
        df = df.sample(frac=1.0, random_state=20260818)
    return df["item_id"].head(limit).tolist()


def estimate_cost(n_cells: int, n_draws: int, model: str, anchors_per_card: int) -> Dict[str, float]:
    """Rough pre-flight cost estimate so nobody starts a run blind."""
    # ~130 tokens per rendered anchor block, plus prompt scaffolding.
    in_tokens = 350 + anchors_per_card * 130
    out_tokens = 120
    calls = n_cells * n_draws
    info = model_info(model)
    return {
        "calls": calls,
        "est_input_tokens": calls * in_tokens,
        "est_output_tokens": calls * out_tokens,
        "est_usd": round(
            calls * (in_tokens / 1e6 * info["in"] + out_tokens / 1e6 * info["out"]), 2
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="B17 stage 1: LLM distributional elicitation")
    ap.add_argument("--config", default=None, help="run config YAML (default: config/run_gss2024.yaml)")
    ap.add_argument("--stage", choices=["anchors", "targets", "all"], default="all")
    ap.add_argument("--model", default=None, help="override llm.model from the config")
    ap.add_argument("--clusters", type=int, default=None, help="cap cluster count (smoke tests)")
    ap.add_argument("--items", type=int, default=None, help="cap item count (smoke tests)")
    ap.add_argument("--dry-run", action="store_true", help="print the cost estimate and exit")
    ap.add_argument("--no-resume", action="store_true", help="ignore existing output and start over")
    args = ap.parse_args()

    load_dotenv()
    cfg = load_config(args.config)
    model = args.model or cfg.llm.get("model", "claude-haiku-4-5")
    out_dir = cfg.out_dir

    print("=" * 72)
    print(" B17 ELICITATION  |  run:", cfg.run_name)
    print("=" * 72)
    print(f"  model        : {model}" + ("   [MOCK - produces noise, not results]" if is_mock(model) else ""))
    print(f"  tree         : {cfg.tree_path.name} ({cfg.granularity})")
    print(f"  split regime : {cfg.split_regime}")
    print(f"  output       : {out_dir}")

    builder = StatCardBuilder(
        tree_path=cfg.tree_path,
        stats_path=cfg.stats_path,
        codebook_path=cfg.codebook_path,
        split_path=cfg.split_path,
        pop_stats_path=cfg.pop_stats_path,
        individual_table_path=cfg.individual_table_path,
        duplicate_threshold=cfg.elicitation.get("duplicate_threshold", 0.60),
        granularity=cfg.granularity,
    )

    clusters = builder.leaf_ids[: args.clusters] if args.clusters else builder.leaf_ids
    n_anchor = args.items or cfg.elicitation.get("n_anchor_items", 36)
    n_target = args.items or cfg.elicitation.get("n_target_items", 40)
    anchors = select_items(builder, cfg, "anchor", n_anchor)
    targets = select_items(builder, cfg, "target", n_target)

    print(f"  clusters     : {len(clusters)}")
    print(f"  anchor items : {len(anchors)}  (calibrator training)")
    print(f"  target items : {len(targets)}  (held out, scored)")

    # ---- cross-fit plan -------------------------------------------------
    topics = {i: builder.codebook[i].get("topic", "general") for i in anchors}
    plan = make_crossfit_plan(
        anchors,
        item_topics=topics,
        n_folds=cfg.calibration.get("n_folds", 3),
        seed=cfg.calibration.get("fold_seed", 20260818),
    )
    save_crossfit_plan(plan, out_dir / "crossfit_plan.json")
    fold_sizes = pd.Series(list(plan.values())).value_counts().sort_index().to_dict()
    print(f"  crossfit     : {cfg.calibration.get('n_folds', 3)} folds, sizes {fold_sizes}")

    n_draws = cfg.elicitation.get("n_paraphrases", 3) * cfg.elicitation.get("n_repeats", 3)
    max_anchors = cfg.elicitation.get("max_anchors_per_card", 12)

    plans = []
    if args.stage in ("anchors", "all"):
        plans.append(("anchor", anchors, out_dir / "raw_anchor_elicitation.parquet"))
    if args.stage in ("targets", "all"):
        plans.append(("target", targets, out_dir / "raw_target_elicitation.parquet"))

    total_cells = sum(len(clusters) * len(items) for _, items, _ in plans)
    est = estimate_cost(total_cells, n_draws, model, max_anchors)
    print("\n  ESTIMATE")
    print(f"    cells        : {total_cells}  x {n_draws} draws = {est['calls']:,} calls")
    print(f"    tokens       : ~{est['est_input_tokens']:,} in / ~{est['est_output_tokens']:,} out")
    print(f"    cost         : ~${est['est_usd']}  (cap ${cfg.llm.get('max_usd', 75.0)})")
    print("    (cached calls from earlier runs are free and are skipped)")

    if args.dry_run:
        print("\n  --dry-run: no calls made.")
        return 0

    elicitor = DistributionElicitor(
        model=model,
        cache_path=cfg.cache_path,
        budget_path=cfg.budget_path,
        max_usd=float(cfg.llm.get("max_usd", 75.0)),
        temperature=float(cfg.llm.get("temperature", 0.7)),
    )

    for stage, items, path in plans:
        print(f"\n[{stage}s] ->  {path.name}")
        raw = elicitor.elicit_batch(
            card_builder=builder,
            cluster_ids=clusters,
            item_ids=items,
            stage=stage,
            split_regime=cfg.split_regime,
            crossfit_plan=plan,
            n_paraphrases=cfg.elicitation.get("n_paraphrases", 3),
            n_repeats=cfg.elicitation.get("n_repeats", 3),
            max_anchors=max_anchors,
            output_parquet_path=path,
            resume=not args.no_resume,
        )
        agg = aggregate_draws(raw)
        agg_path = path.with_name(path.stem.replace("raw_", "cells_") + ".parquet")
        agg.to_parquet(agg_path, index=False)
        print(f"  {len(raw)} draws over {len(agg)} cells -> {agg_path.name}")
        if not agg.empty:
            print(f"  mean ensemble SD: {agg['ensemble_sd'].mean():.4f} "
                  f"(0.0 would mean the ensemble is not sampling)")

    summary = elicitor.budget.get_summary()
    manifest = {
        "run_name": cfg.run_name,
        "model": model,
        "is_mock": is_mock(model),
        "granularity": cfg.granularity,
        "split_regime": cfg.split_regime,
        "n_clusters": len(clusters),
        "anchor_items": anchors,
        "target_items": targets,
        "crossfit_folds": cfg.calibration.get("n_folds", 3),
        "n_paraphrases": cfg.elicitation.get("n_paraphrases", 3),
        "n_repeats": cfg.elicitation.get("n_repeats", 3),
        "budget": summary,
    }
    with open(out_dir / "elicitation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "-" * 72)
    print(f"  spent ${summary['total_usd_spent']:.2f} of ${summary['max_usd']:.2f}")
    print(f"  {summary['live_calls']} live calls, {summary['cached_calls']} served from cache")
    print(f"  manifest -> {out_dir / 'elicitation_manifest.json'}")
    print("\n  Next:  python benchmark.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
