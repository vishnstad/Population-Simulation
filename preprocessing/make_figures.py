#!/usr/bin/env python3
"""Render every figure from a completed run directory.

    python make_figures.py configs/gss_main.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from popsim.report import plots


def load_run(cfg_path: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    run = Path(cfg["output"]["dir"])
    d = {
        "cfg": cfg,
        "run": run,
        "table": pd.read_parquet(run / "individual_table.parquet"),
        "cstats": pd.read_parquet(run / "cluster_stats.parquet"),
        "pstats": pd.read_parquet(run / "population_stats.parquet"),
        "split": pd.read_csv(run / "item_split.csv"),
        "sweep": pd.read_csv(run / "k_sweep.csv"),
        "nodes": json.loads((run / "cluster_tree.json").read_text()),
        "manifest": json.loads((run / "manifest.json").read_text()),
    }
    cb = yaml.safe_load((run / "item_codebook.yaml").read_text())
    d["codebook"] = {c["item_id"]: c for c in cb}
    d["topic_of"] = {c["item_id"]: c["topic"] for c in cb}
    return d


def main(cfg_path: str):
    d = load_run(cfg_path)
    figdir = Path(d["cfg"]["output"]["figures"])
    figdir.mkdir(parents=True, exist_ok=True)
    min_neff = d["cfg"]["scoring"]["min_neff"]
    min_cell = d["manifest"].get("min_cell") \
        or d["manifest"]["config"]["clustering"].get("min_cell")
    K = d["manifest"]["K"]
    chosen_pct = 100 * (d["cstats"].n_eff >= min_neff).mean()

    made = []

    plots.fig_k_vs_scoreable(d["sweep"], K, chosen_pct, min_neff,
                             figdir / "fig01_k_vs_scoreable.png",
                             min_cell=min_cell)
    made.append("fig01_k_vs_scoreable.png")

    plots.fig_leaf_sizes(d["nodes"], min_neff, min_cell,
                         figdir / "fig02_leaf_sizes.png")
    made.append("fig02_leaf_sizes.png")

    plots.fig_heterogeneity_bias(d["split"], figdir / "fig03_heterogeneity_bias.png")
    made.append("fig03_heterogeneity_bias.png")

    plots.fig_top_items(d["split"], d["codebook"], 25,
                        figdir / "fig04_top_heterogeneity_items.png")
    made.append("fig04_top_heterogeneity_items.png")

    plots.fig_coverage(d["codebook"], figdir / "fig05_item_coverage.png")
    made.append("fig05_item_coverage.png")

    plots.fig_split(d["split"], d["manifest"]["adversarial_target_topics"],
                    figdir / "fig06_anchor_target_split.png")
    made.append("fig06_anchor_target_split.png")

    coarse_path = d["run"] / "cluster_stats_coarse.parquet"
    cs_for_spread = pd.read_parquet(coarse_path) if coarse_path.exists() else d["cstats"]
    kc = d["manifest"].get("K_coarse")
    plots.fig_between_cluster(cs_for_spread, d["pstats"], d["split"], d["codebook"],
                              min_neff, figdir / "fig07_between_cluster_spread.png",
                              k_label=f", coarse partition K = {kc}" if kc else "")
    made.append("fig07_between_cluster_spread.png")

    plots.fig_weighting(d["table"], figdir / "fig08_weighting_effect.png")
    made.append("fig08_weighting_effect.png")

    tree_c_path = d["run"] / "cluster_tree_coarse.json"
    nodes_c = json.loads(tree_c_path.read_text()) if tree_c_path.exists() else None
    plots.fig_availability(d["cstats"], d["nodes"], d["topic_of"], min_neff,
                           figdir / "fig09_availability.png",
                           cstats_coarse=cs_for_spread if coarse_path.exists() else None,
                           nodes_coarse=nodes_c, k_main=K, k_coarse=kc)
    made.append("fig09_availability.png")

    for m in made:
        print("wrote", figdir / m)
    return made


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/gss_main.yaml")
