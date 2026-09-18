"""Free simulation of the deviation-form prediction, from histograms already on disk.

The forensics run says the model's error splits cleanly:

  * LEVEL     — huge (spkcom: predicted mean 0.72 against a true 0.29)
  * STRUCTURE — real (between-cluster Spearman +0.60 macro on the 14B, 8/10 items
                significant), which is the opposite of F2

B0a is handed the true national marginal and is called "unfair-strong" in the
spec for exactly that reason. This probe asks what happens when the SAME level
information is given to the system, so the only thing being compared is whether
the model adds subgroup structure on top of it:

    cdf_pred(c) = cdf_national + s * ( cdf_model(c) - weighted_mean_c cdf_model(c) )

s = 0 is B0a exactly. s > 0 adds the model's subgroup deviation. No target truth
is used except to report the oracle-optimal s as a ceiling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from popsim.clustering.partition import build_cluster_tree
from popsim.clustering.stats import compute_cluster_stats, weighted_histogram
from popsim.config import load_config
from popsim.data.adapters.gss import load_gss
from popsim.evalx.metrics import normalized_positions, w1

SCALES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def cdf(h):
    h = np.asarray(h, float)
    return np.cumsum(h / h.sum())[:-1]


def from_cdf(c):
    c = np.maximum.accumulate(np.clip(np.asarray(c, float), 0.0, 1.0))
    h = np.diff(np.concatenate([[0.0], c, [1.0]]))
    h = np.clip(h, 0.0, None)
    return h / h.sum() if h.sum() > 0 else h


def sd_pos(h):
    p = np.asarray(h, float); p = p / p.sum()
    x = normalized_positions(p.size); m = p @ x
    return float(np.sqrt(max(p @ (x - m) ** 2, 0.0)))


def main(run_dirs):
    cfg = load_config("configs/gss_main.yaml")
    cb = yaml.safe_load(Path("codebooks/gss_items.yaml").read_text())["items"]
    all_items = sorted(cb); codes = {i: cb[i]["codes"] for i in all_items}
    table = load_gss(cfg.bed_file, all_items, waves=cfg["bed.waves"])
    tree = build_cluster_tree(table.frame, axes=cfg["partition.axes"], item_codes=codes,
                              min_cell=cfg["partition.min_cell"], k_target=cfg["partition.k_target"])
    assign = tree.assign(table.frame)
    stats = compute_cluster_stats(table.frame, assign, all_items, codes)
    wcol = pd.to_numeric(table.frame["weight"], errors="coerce").fillna(0.0).to_numpy()

    for rd in run_dirs:
        raw = pd.read_parquet(Path(rd) / "permutation_raw.parquet")
        raw = raw[raw.ok & raw.permuted_from.isna()]
        print("=" * 104)
        print(f"{rd}  model={raw['model'].iloc[0]}")
        print("=" * 104)
        print(f"{'item':10s} " + " ".join(f"s={s:<5.1f}" for s in SCALES)
              + f" {'s*':>5s} {'W1@s*':>7s} {'vr':>5s}")
        per_item = {}
        for item_id, g in raw.groupby("item_id"):
            cs = codes[item_id]
            col = table.frame[f"item_{item_id}"].to_numpy(); ok = np.isin(col, cs)
            national = weighted_histogram(col[ok], wcol[ok], cs)
            P, T, W = [], [], []
            for cid, gg in g.groupby("cluster_id"):
                truth = stats.hist(cid, item_id)
                if truth is None or truth.sum() <= 0:
                    continue
                row = stats.frame[(stats.frame.cluster_id == cid) & (stats.frame.item_id == item_id)]
                if row.empty or float(row.iloc[0]["n_eff"]) <= 0:
                    continue
                pr = np.mean(np.vstack([np.asarray(h, float) for h in gg["hist"]]), axis=0)
                P.append(pr / pr.sum()); T.append(np.asarray(truth, float))
                W.append(float(row.iloc[0]["weight_sum"]))
            if len(P) < 3:
                continue
            W = np.asarray(W); W = W / W.sum()
            CP = np.vstack([cdf(p) for p in P]); cbar = W @ CP
            cN = cdf(national); delta = CP - cbar
            scores = {}
            for s in SCALES + list(np.arange(0.1, 3.01, 0.1)):
                pred = [from_cdf(cN + s * delta[i]) for i in range(len(P))]
                scores[round(float(s), 2)] = float(sum(W[i] * w1(pred[i], T[i]) for i in range(len(P))))
            sstar = min(scores, key=scores.get)
            predstar = [from_cdf(cN + sstar * delta[i]) for i in range(len(P))]
            vr = float(np.mean([sd_pos(predstar[i]) / sd_pos(T[i]) for i in range(len(P))]))
            per_item[item_id] = {"scores": scores, "sstar": sstar, "vr": vr, "W": W, "n": len(P)}
            print(f"{item_id:10s} " + " ".join(f"{scores[s]:.4f}" for s in SCALES)
                  + f" {sstar:5.1f} {scores[sstar]:7.4f} {vr:5.2f}")
        print("-" * 104)
        macro = {s: float(np.mean([v["scores"][s] for v in per_item.values()])) for s in SCALES}
        all_s = sorted({round(float(s), 2) for s in np.arange(0.1, 3.01, 0.1)})
        macro_all = {s: float(np.mean([v["scores"][s] for v in per_item.values()])) for s in all_s}
        gstar = min(macro_all, key=macro_all.get)
        print(f"{'MACRO':10s} " + " ".join(f"{macro[s]:.4f}" for s in SCALES)
              + f"   best GLOBAL s={gstar:.1f} -> {macro_all[gstar]:.4f}")
        b0a = macro[0.0]
        print(f"  B0a (s=0) = {b0a:.4f}   pass mark at -20% = {b0a*0.8:.4f}   "
              f"noise floor 0.0273")
        for s in [0.5, 1.0, 1.5, gstar]:
            v = macro_all.get(round(s, 2), macro.get(s))
            print(f"   s={s:<4.1f} macro W1 {v:.4f}  ({(v/b0a-1)*100:+.1f}% vs B0a)  "
                  f"{'PASS' if v <= b0a*0.8 else 'fail'}")
        print("  per-item s*: " + ", ".join(f"{k}={v['sstar']:.1f}" for k, v in per_item.items()))
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
