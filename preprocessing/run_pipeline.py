#!/usr/bin/env python3
"""
B17 preprocessing driver: raw GSS microdata -> M1 -> M2 -> item split.

    python run_pipeline.py configs/gss_main.yaml

Writes, under `output.dir`:
    individual_table.parquet   IndividualTable (§3.3 M1)
    cluster_tree.json          ClusterTree     (§3.3 M2)
    cluster_stats.parquet      ClusterStats    (§3.3 M2)
    population_stats.parquet   level-0 marginals / B0a oracle
    item_codebook.yaml         Item records    (§3.3 M1)
    item_split.csv             heterogeneity + anchor/target roles (§5.1)
    k_sweep.csv                cells clearing n_eff>=30 at each K (§5.5/§5.7)
    manifest.json              config snapshot + audit log + contract checks
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from popsim.data.adapters.gss import load_gss
from popsim.clustering.partition import build_cluster_tree, tree_to_json
from popsim.clustering.stats import build_cluster_stats, population_stats, kish_neff
from popsim.evalx.split import item_heterogeneity, stratified_split, adversarial_split


def main(cfg_path: str) -> dict:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    log: list[str] = []

    def say(m):
        print(m, flush=True)
        log.append(m)

    # ---------------- M1 ---------------------------------------------------
    d = cfg["data"]
    res = load_gss(
        d["path"], d["recodes"], wave=d.get("wave"),
        min_item_coverage=d["min_item_coverage"],
        min_scale=d["min_scale"], max_scale=d["max_scale"],
        strict_guards=d["strict_guards"],
    )
    for m in res.audit["log"]:
        say("M1: " + m)

    tax = yaml.safe_load(Path(d["taxonomy"]).read_text())
    excluded = {m for lst in tax["excluded"].values() for m in lst}
    topic_of = {m: t for t, lst in tax["topics"].items() for m in lst}

    keep_items = [i for i in res.item_cols if i not in excluded and i in topic_of]
    say(f"M1: taxonomy retains {len(keep_items)} attitudinal/behavioral items "
        f"({len(res.item_cols) - len(keep_items)} dropped as background/admin/quiz)")

    scales = {i: sorted(res.value_labels[i]) for i in keep_items}
    table = res.table[
        ["person_id", "dataset", "wave", "weight", "psu", "strata"]
        + res.demo_cols + [f"item_{i}" for i in keep_items]
    ].copy()

    # ---------------- M2 ---------------------------------------------------
    c = cfg["clustering"]
    # merge-distance profile: the highest-coverage items, spread across topics
    cov = {i: float(table[f"item_{i}"].notna().mean()) for i in keep_items}
    by_topic: dict[str, list[str]] = {}
    for i in sorted(keep_items, key=lambda x: -cov[x]):
        by_topic.setdefault(topic_of[i], []).append(i)
    profile: list[str] = []
    while len(profile) < c["anchor_profile_items"]:
        added = False
        for t in sorted(by_topic):
            if by_topic[t] and len(profile) < c["anchor_profile_items"]:
                profile.append(by_topic[t].pop(0)); added = True
        if not added:
            break
    say(f"M2: merge-distance profile items = {profile}")
    profile_scales = {i: scales[i] for i in profile}

    # min_cell: explicit, or searched so that K lands in the target band
    min_cell = c.get("min_cell")
    if min_cell is None:
        lo, hi = c["target_k"]
        trials = []
        for mc in c["min_cell_search"]:
            a_, n_ = build_cluster_tree(
                table, l1_axes=c["l1_axes"], l2_axes=c["l2_axes"],
                anchor_items=profile_scales, min_cell=mc,
            )
            k_ = sum(1 for n in n_ if n.level == 2)
            trials.append((mc, k_))
            if lo <= k_ <= hi:
                min_cell = mc
                break
        if min_cell is None:
            min_cell = min(trials, key=lambda t: min(abs(t[1] - lo), abs(t[1] - hi)))[0]
        k_best = max(k for _, k in trials)
        reachable = k_best >= lo
        say(f"M2: min_cell search {trials} -> chose min_cell={min_cell} "
            f"for target_k={c['target_k']}")
        if not reachable:
            say(f"M2: NOTE target K in {c['target_k']} is UNREACHABLE with an "
                f"interpretable cross-product partition on this sample — the "
                f"finest lattice that keeps every cell at min_cell yields "
                f"K={k_best}. Reaching K≈150 would require abandoning the "
                f"conjunction form or accepting cells below min_cell.")
        cfg.setdefault("_findings", {})["k_target_reachable"] = bool(reachable)
        cfg["_findings"]["k_max_interpretable"] = int(k_best)

    assign, nodes = build_cluster_tree(
        table, l1_axes=c["l1_axes"], l2_axes=c["l2_axes"],
        anchor_items=profile_scales, min_cell=min_cell,
        max_leaves=c.get("max_leaves"),
    )
    c = {**c, "min_cell_achieved": min_cell}
    table["cluster_id"] = assign.values
    leaves = [n for n in nodes if n.level == 2]
    say(f"M2: K = {len(leaves)} leaves at min_cell={min_cell} "
        f"(L1 nodes = {sum(1 for n in nodes if n.level == 1)})")
    say(f"M2: leaf raw-n  min/median/max = "
        f"{min(n.n_raw for n in leaves)}/{int(np.median([n.n_raw for n in leaves]))}/"
        f"{max(n.n_raw for n in leaves)}")
    say(f"M2: leaf n_eff  min/median/max = "
        f"{min(n.n_eff for n in leaves):.1f}/{np.median([n.n_eff for n in leaves]):.1f}/"
        f"{max(n.n_eff for n in leaves):.1f}")

    cstats = build_cluster_stats(table, "cluster_id", scales)
    pstats = population_stats(table, scales)

    # ---------------- item split (§5.1) ------------------------------------
    s = cfg["split"]
    het = item_heterogeneity(table, "cluster_id", scales)
    split = stratified_split(het, topic_of, anchor_frac=s["anchor_frac"],
                             seed=s["seed"], het_col=s["heterogeneity_col"])
    split = adversarial_split(split, anchor_frac=s["anchor_frac"], seed=s["seed"])
    adv_topics = split.attrs.get("adversarial_target_topics", [])
    say(f"split: standard  anchors={int((split.role_standard=='anchor').sum())} "
        f"targets={int((split.role_standard=='target').sum())}")
    say(f"split: adversarial held-out topics = {adv_topics} "
        f"(targets={int((split.role_adversarial=='target').sum())})")

    # scoring-eligible cells (§5.2)
    min_neff = cfg["scoring"]["min_neff"]
    elig = cstats["n_eff"] >= min_neff
    say(f"scoring: {int(elig.sum()):,} of {len(cstats):,} (cluster x item) cells "
        f"clear n_eff >= {min_neff} ({100*elig.mean():.1f}%)")

    # ---------------- K sweep (§5.5 / §5.7) --------------------------------
    sweep = []
    for mc in cfg["k_sweep"]["min_cells"]:
        a2, n2 = build_cluster_tree(
            table, l1_axes=c["l1_axes"], l2_axes=c["l2_axes"],
            anchor_items=profile_scales, min_cell=mc,
        )
        lv = [n for n in n2 if n.level == 2]
        t2 = table.copy(); t2["cid2"] = a2.values
        cs2 = build_cluster_stats(t2, "cid2", scales)
        sweep.append({
            "min_cell": mc, "K": len(lv),
            "median_leaf_n": float(np.median([n.n_raw for n in lv])),
            "median_leaf_neff": float(np.median([n.n_eff for n in lv])),
            "cells_total": len(cs2),
            "cells_scoreable": int((cs2["n_eff"] >= min_neff).sum()),
            "pct_scoreable": float(100 * (cs2["n_eff"] >= min_neff).mean()),
        })
        say(f"K-sweep: min_cell={mc:>3} -> K={len(lv):>3}  "
            f"scoreable cells {sweep[-1]['pct_scoreable']:.1f}%")
    sweep = pd.DataFrame(sweep)

    # ---------------- scoreable ("coarse") partition ------------------------
    # The main partition hits the spec's K target but leaves almost no cell
    # above the n_eff gate. We therefore ALSO emit the finest partition at
    # which a majority of cells are actually scoreable -- this is the one on
    # which §1.4's claim can be evaluated at all, and the two are meant to be
    # reported side by side.
    ok_rows = sweep[sweep["pct_scoreable"] >= cfg["scoring"]["coarse_min_pct"]]
    coarse_mc = int(ok_rows["min_cell"].min()) if len(ok_rows) \
        else int(sweep.loc[sweep["pct_scoreable"].idxmax(), "min_cell"])
    a_c, n_c = build_cluster_tree(
        table, l1_axes=c["l1_axes"], l2_axes=c["l2_axes"],
        anchor_items=profile_scales, min_cell=coarse_mc,
    )
    table["cluster_id_coarse"] = a_c.values
    cstats_coarse = build_cluster_stats(table, "cluster_id_coarse", scales)
    k_coarse = sum(1 for n in n_c if n.level == 2)
    say(f"coarse: min_cell={coarse_mc} -> K={k_coarse}, "
        f"{100*(cstats_coarse['n_eff'] >= min_neff).mean():.1f}% of cells scoreable")

    het_coarse = item_heterogeneity(table, "cluster_id_coarse", scales)
    het_coarse = het_coarse.rename(columns={"h_raw": "h_raw_coarse",
                                            "h_icc": "h_icc_coarse"})
    split = split.merge(het_coarse[["item_id", "h_raw_coarse", "h_icc_coarse"]],
                        on="item_id", how="left")

    # ---------------- item codebook ----------------------------------------
    het_ix = split.set_index("item_id")
    codebook = []
    for i in keep_items:
        row = het_ix.loc[i]
        codebook.append({
            "item_id": i, "dataset": "gss", "wave": res.table["wave"].iloc[0],
            "text": res.var_labels.get(i, ""),
            "scale": {
                "type": "binary" if len(scales[i]) == 2 else "ordinal_or_nominal",
                "codes": [int(x) for x in scales[i]],
                "labels": [res.value_labels[i][x] for x in scales[i]],
            },
            "topic": topic_of[i],
            "coverage": round(cov[i], 4),
            "heterogeneity": None if pd.isna(row["h_icc"]) else round(float(row["h_icc"]), 5),
            "heterogeneity_raw": None if pd.isna(row["h_raw"]) else round(float(row["h_raw"]), 5),
            "role": {"standard": row["role_standard"], "adversarial": row["role_adversarial"]},
        })

    # ---------------- contract checks (§3.3 failure modes) ------------------
    checks = {}
    sums = np.array([
        float(np.sum(h)) for h in cstats["hist"]
        if h is not None and np.isfinite(np.asarray(h, float)).all()
    ])
    checks["histograms_sum_to_1"] = bool(np.allclose(sums, 1.0)) if len(sums) else None
    checks["n_populated_cells"] = int(len(sums))
    strand = [n for n in leaves
              if n.n_raw < min_cell and "unclassified" not in n.cluster_id]
    checks["all_leaves_meet_min_cell"] = len(strand) == 0
    checks["n_leaves_below_min_cell"] = len(strand)
    checks["leaf_ids_unique"] = len({n.cluster_id for n in leaves}) == len(leaves)
    if strand:
        say(f"warn: {len(strand)} leaf(s) below min_cell with no conformable "
            f"sibling to merge into (smallest n = {min(n.n_raw for n in strand)})")
    checks["weights_all_positive"] = bool((table["weight"] > 0).all())
    checks["leaf_pop_share_sums_to_1"] = bool(
        abs(sum(n.pop_share for n in leaves) - 1.0) < 1e-6
    )
    checks["no_item_in_both_roles"] = bool(
        not ((split.role_standard == "anchor") & (split.role_adversarial == "anchor")
             & (split.role_standard == "target")).any()
    )
    checks["every_respondent_assigned"] = bool(table["cluster_id"].notna().all())
    for k, v in checks.items():
        say(f"check: {k} = {v}")

    # ---------------- write -------------------------------------------------
    table.to_parquet(out / "individual_table.parquet", index=False)
    (out / "cluster_tree.json").write_text(tree_to_json(nodes))
    cstats.to_parquet(out / "cluster_stats.parquet", index=False)
    cstats_coarse.to_parquet(out / "cluster_stats_coarse.parquet", index=False)
    (out / "cluster_tree_coarse.json").write_text(tree_to_json(n_c))
    pstats.to_parquet(out / "population_stats.parquet", index=False)
    (out / "item_codebook.yaml").write_text(yaml.safe_dump(codebook, sort_keys=False, width=200))
    split.to_csv(out / "item_split.csv", index=False)
    sweep.to_csv(out / "k_sweep.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "config": cfg, "m1_audit": res.audit, "log": log,
        "checks": checks, "K": len(leaves), "min_cell": min_cell,
        "K_coarse": k_coarse, "min_cell_coarse": coarse_mc,
        "n_items": len(keep_items),
        "adversarial_target_topics": adv_topics,
    }, indent=1, default=str))
    say(f"wrote -> {out}")

    return {"table": table, "nodes": nodes, "cstats": cstats, "pstats": pstats,
            "split": split, "sweep": sweep, "scales": scales, "topic_of": topic_of,
            "codebook": codebook, "cfg": cfg, "res": res}


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/gss_main.yaml")
