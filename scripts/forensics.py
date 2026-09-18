"""Forensic decomposition of an existing Gate 3 run — no new LLM calls.

The Gate 3 statistic (pooled real-vs-permuted W1) confounds two things:

  * LEVEL   — how far the model's item-wide prediction sits from the item's
              true population level. Identical in both arms, so it cancels out
              of the gap but dominates the W1 that is reported.
  * STRUCTURE — whether the card moves the prediction toward *this* cluster.
              This is the only thing the gap can see, and it is small because
              real between-cluster spread is small.

Phase 4's calibration layer exists to repair LEVEL. So the question that decides
whether the project can work is not the raw gap, it is: *after the level is
repaired, is there structure left?* That is measurable for free from histograms
already on disk.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from popsim.clustering.partition import build_cluster_tree
from popsim.clustering.stats import compute_cluster_stats, weighted_histogram
from popsim.config import load_config
from popsim.data.adapters.gss import load_gss
from popsim.evalx.metrics import normalized_positions, w1


def mean_pos(h: np.ndarray) -> float:
    h = np.asarray(h, float)
    if h.sum() <= 0:
        return float("nan")
    return float((h / h.sum()) @ normalized_positions(h.size))


def sd_pos(h: np.ndarray) -> float:
    h = np.asarray(h, float)
    if h.sum() <= 0:
        return float("nan")
    p = h / h.sum()
    x = normalized_positions(h.size)
    m = p @ x
    return float(np.sqrt(max(p @ (x - m) ** 2, 0.0)))


def cdf(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, float)
    return np.cumsum(h / h.sum())[:-1]


def from_cdf(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, float), 0.0, 1.0)
    c = np.maximum.accumulate(c)
    full = np.concatenate([[0.0], c, [1.0]])
    h = np.diff(full)
    h = np.clip(h, 0.0, None)
    return h / h.sum() if h.sum() > 0 else h


def main(run_dirs: list[str]) -> None:
    cfg = load_config("configs/gss_main.yaml")
    cb = yaml.safe_load(Path("codebooks/gss_items.yaml").read_text())["items"]
    all_items = sorted(cb)
    codes = {i: cb[i]["codes"] for i in all_items}
    table = load_gss(cfg.bed_file, all_items, waves=cfg["bed.waves"])
    tree = build_cluster_tree(
        table.frame, axes=cfg["partition.axes"], item_codes=codes,
        min_cell=cfg["partition.min_cell"], k_target=cfg["partition.k_target"],
    )
    assign = tree.assign(table.frame)
    stats = compute_cluster_stats(table.frame, assign, all_items, codes)
    wcol = pd.to_numeric(table.frame["weight"], errors="coerce").fillna(0.0).to_numpy()

    for rd in run_dirs:
        raw = pd.read_parquet(Path(rd) / "permutation_raw.parquet")
        raw = raw[raw.ok & raw.permuted_from.isna()]
        model = raw["model"].iloc[0]
        print("=" * 100)
        print(f"{rd}   model={model}   n_real_records={len(raw)}")
        print("=" * 100)
        hdr = (f"{'item':10s} {'k':>2s} {'B0a':>7s} {'raw':>7s} {'isoLVL':>7s} "
               f"{'trueM':>6s} {'predM':>6s} {'trueSD':>6s} {'predSD':>6s} "
               f"{'btwT':>6s} {'btwP':>6s} {'rho':>6s} {'p':>6s}")
        print(hdr)
        rows = []
        for item_id, g in raw.groupby("item_id"):
            cs = codes[item_id]
            col = table.frame[f"item_{item_id}"].to_numpy()
            ok = np.isin(col, cs)
            national = weighted_histogram(col[ok], wcol[ok], cs)
            P, T, W = [], [], []
            for cid, gg in g.groupby("cluster_id"):
                truth = stats.hist(cid, item_id)
                if truth is None or truth.sum() <= 0:
                    continue
                row = stats.frame[(stats.frame.cluster_id == cid)
                                  & (stats.frame.item_id == item_id)]
                if row.empty or float(row.iloc[0]["n_eff"]) <= 0:
                    continue
                pred = np.mean(np.vstack([np.asarray(h, float) for h in gg["hist"]]), axis=0)
                pred = pred / pred.sum()
                P.append(pred); T.append(np.asarray(truth, float))
                W.append(float(row.iloc[0]["weight_sum"]))
            if len(P) < 3:
                continue
            W = np.asarray(W); W = W / W.sum()
            k = len(cs)
            w1_raw = float(sum(W[i] * w1(P[i], T[i]) for i in range(len(P))))
            w1_b0a = float(sum(W[i] * w1(national, T[i]) for i in range(len(P))))
            # ORACLE level repair: one monotone map per item, fitted on the pooled
            # (predicted cdf value -> true cdf value) pairs across clusters. Uses
            # target truth, so it is a CEILING for what an anchor-fitted isotonic
            # layer could reach on this item, not an achievable score.
            xs = np.concatenate([cdf(p) for p in P])
            ys = np.concatenate([cdf(t) for t in T])
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(xs, ys)
            w1_iso = float(sum(W[i] * w1(from_cdf(iso.predict(cdf(P[i]))), T[i])
                               for i in range(len(P))))
            tm = np.array([mean_pos(t) for t in T]); pm = np.array([mean_pos(p) for p in P])
            tsd = np.array([sd_pos(t) for t in T]); psd = np.array([sd_pos(p) for p in P])
            rho, pv = spearmanr(pm, tm)
            rows.append({"item": item_id, "k": k, "b0a": w1_b0a, "raw": w1_raw,
                         "iso": w1_iso, "rho": rho, "p": pv})
            print(f"{item_id:10s} {k:2d} {w1_b0a:7.4f} {w1_raw:7.4f} {w1_iso:7.4f} "
                  f"{tm@W:6.3f} {pm@W:6.3f} {tsd@W:6.3f} {psd@W:6.3f} "
                  f"{tm.std():6.3f} {pm.std():6.3f} {rho:6.3f} {pv:6.3f}")
        d = pd.DataFrame(rows)
        print("-" * 100)
        print(f"{'MACRO':10s} {'':2s} {d.b0a.mean():7.4f} {d.raw.mean():7.4f} "
              f"{d.iso.mean():7.4f}   rho_mean={d.rho.mean():+.3f}  "
              f"items rho>0: {(d.rho > 0).sum()}/{len(d)}  "
              f"items rho sig(p<.05,+): {((d.p < .05) & (d.rho > 0)).sum()}/{len(d)}")
        print(f"  iso-level beats B0a on {(d.iso < d.b0a).sum()}/{len(d)} items")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
