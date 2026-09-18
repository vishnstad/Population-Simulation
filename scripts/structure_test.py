"""Gate 3, asked about the thing Gate 3 is named for — subgroup conditioning.

The frozen Gate 3 statistic is the pooled real-vs-permuted W1 gap. That number
carries the model's item-level bias in both arms, where it cancels out of the
difference but sets the scale of both terms. On the 14B arm the level error is
~0.20 W1 and the whole between-cluster spread of the truth is ~0.12, so the gap
the gate reads is a small difference between two large, mostly-level numbers.

This script asks the same question on the level-matched predictions, where the
only thing left is subgroup structure:

    pred(c)      = national_cdf + s * (model_cdf(c) - mean_c model_cdf(c))
    permuted(c)  = national_cdf + s * (model_cdf(c') - mean_c model_cdf(c))

Two nulls:
  * CARD-PERMUTED — the model was actually shown another cluster's card. The
    strongest form: it tests the elicitation, not the arithmetic.
  * DERANGEMENT   — reassign the real arm's own deviations. Free, and gives a
    p-value over many draws instead of one.
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
from popsim.evalx.metrics import w1

S = 1.0
NDRAW = 2000


def cdf(h):
    h = np.asarray(h, float)
    return np.cumsum(h / h.sum())[:-1]


def from_cdf(c):
    c = np.maximum.accumulate(np.clip(np.asarray(c, float), 0.0, 1.0))
    h = np.clip(np.diff(np.concatenate([[0.0], c, [1.0]])), 0.0, None)
    return h / h.sum() if h.sum() > 0 else h


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
    rng = np.random.default_rng(cfg["seed"])

    for rd in run_dirs:
        allraw = pd.read_parquet(Path(rd) / "permutation_raw.parquet")
        allraw = allraw[allraw.ok]
        print("=" * 92)
        print(f"{rd}  model={allraw['model'].iloc[0]}   deviation scale s={S}")
        print("=" * 92)
        print(f"{'item':10s} {'B0a':>7s} {'real':>7s} {'cardperm':>8s} {'nullmean':>8s} "
              f"{'p_null':>7s} {'gap_card':>8s}")
        agg = {k: [] for k in ("b0a", "real", "perm", "nullmean", "p")}
        for item_id, g in allraw.groupby("item_id"):
            cs = codes[item_id]
            col = table.frame[f"item_{item_id}"].to_numpy(); ok = np.isin(col, cs)
            national = weighted_histogram(col[ok], wcol[ok], cs)
            cN = cdf(national)
            cids, T, W, Preal, Pperm = [], [], [], [], []
            for cid, gg in g.groupby("cluster_id"):
                truth = stats.hist(cid, item_id)
                if truth is None or truth.sum() <= 0:
                    continue
                row = stats.frame[(stats.frame.cluster_id == cid) & (stats.frame.item_id == item_id)]
                if row.empty or float(row.iloc[0]["n_eff"]) <= 0:
                    continue
                r = gg[gg.permuted_from.isna()]; p = gg[gg.permuted_from.notna()]
                if r.empty or p.empty:
                    continue
                hr = np.mean(np.vstack([np.asarray(h, float) for h in r["hist"]]), axis=0)
                hp = np.mean(np.vstack([np.asarray(h, float) for h in p["hist"]]), axis=0)
                cids.append(cid); T.append(np.asarray(truth, float))
                W.append(float(row.iloc[0]["weight_sum"]))
                Preal.append(hr / hr.sum()); Pperm.append(hp / hp.sum())
            if len(T) < 3:
                continue
            W = np.asarray(W); W = W / W.sum()
            CR = np.vstack([cdf(x) for x in Preal]); CP = np.vstack([cdf(x) for x in Pperm])
            dR = CR - (W @ CR); dP = CP - (W @ CP)
            def score(D, W=W, cN=cN, T=T):
                return float(sum(W[i] * w1(from_cdf(cN + S * D[i]), T[i])
                                 for i in range(len(T))))
            b0a = float(sum(W[i] * w1(national, T[i]) for i in range(len(T))))
            real, perm = score(dR), score(dP)
            vals = []
            n = len(T)
            while len(vals) < NDRAW:
                pm = rng.permutation(n)
                if np.any(pm == np.arange(n)):
                    continue
                vals.append(score(dR[pm]))
            v = np.asarray(vals)
            pval = float((v <= real).mean())
            agg["b0a"].append(b0a); agg["real"].append(real); agg["perm"].append(perm)
            agg["nullmean"].append(float(v.mean())); agg["p"].append(pval)
            print(f"{item_id:10s} {b0a:7.4f} {real:7.4f} {perm:8.4f} {v.mean():8.4f} "
                  f"{pval:7.3f} {perm-real:+8.4f}")
        m = {k: float(np.mean(x)) for k, x in agg.items()}
        sig = sum(1 for x in agg["p"] if x < 0.05)
        print("-" * 92)
        print(f"{'MACRO':10s} {m['b0a']:7.4f} {m['real']:7.4f} {m['perm']:8.4f} "
              f"{m['nullmean']:8.4f} {'':7s} {m['perm']-m['real']:+8.4f}")
        print(f"  items with p < 0.05 against the derangement null: {sig}/{len(agg['p'])}")
        print(f"  card-permuted gap   {m['perm']-m['real']:+.4f}")
        print(f"  derangement gap     {m['nullmean']-m['real']:+.4f}")
        print(f"  real vs B0a         {(m['real']/m['b0a']-1)*100:+.1f}%")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
