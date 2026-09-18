"""Does projecting the model's deviation onto the observed subgroup subspace help?

The deviation the model supplies is measured with 3 ensemble draws, so most of
its variance across clusters is elicitation noise. But real subgroup variation is
not free to point anywhere: across the 34 anchor items whose per-cluster truths
we *do* have, the cluster-level deviations live in a low-dimensional subspace —
age and education gradients, mostly. Any component of the model's deviation
outside that subspace cannot be real subgroup structure.

So: build the subspace from ANCHOR truths (observed data, never a target), and
project the model's target deviation onto it before scaling.

    U  = top-r principal directions of the anchor deviation matrix (clusters x anchor CDF columns)
    d~ = U U^T d_raw
    pred_cdf(c) = level_cdf + s * d~(c)

No target truth is used to build U or to choose r. This probe reports the
oracle-best r as a ceiling AND the r that anchor cross-validation would pick,
which is the one the system may use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from popsim.agents.runner import ElicitationStore
from popsim.calibration.shape import cdf, from_cdf
from popsim.config import load_config
from popsim.evalx.harness import collect_cells
from popsim.evalx.metrics import w1
from popsim.pipeline import build_bed

RS = [0, 1, 2, 3, 4, 6, 8, 12, 56]
SS = np.round(np.arange(0.0, 3.01, 0.05), 2)


def anchor_basis(bed, cluster_ids, anchors, min_neff, *, verbose=True):
    """Columns are (anchor item, interior CDF position); rows are clusters."""
    cols, used = [], []
    for a in anchors:
        hs = []
        ok = True
        for c in cluster_ids:
            h = bed.stats.hist(c, a)
            if h is None or h.sum() <= 0 or bed.cell_neff(c, a) < min_neff:
                ok = False
                break
            hs.append(cdf(h))
        if ok and hs:
            M = np.vstack(hs)                      # C x (k-1)
            cols.append(M - M.mean(axis=0))
            used.append(a)
    if verbose:
        print(f"basis from {len(used)}/{len(anchors)} anchors at min_neff={min_neff}")
    return np.hstack(cols) if cols else np.zeros((len(cluster_ids), 0))


def main():
    cfg = load_config("configs/gss_main.yaml")
    bed = build_bed(cfg)
    model = sys.argv[1] if len(sys.argv) > 1 else "ministral-8b-2512"
    profile = sys.argv[2] if len(sys.argv) > 2 else "permutation"
    st = ElicitationStore(root=cfg.repo_root / "runs/_elicit_store",
                          model=model, profile=profile)
    min_neff = float(cfg["evaluation.truth_min_neff"])
    anchors = sorted(bed.split["anchors"])
    targets = bed.ranked_targets()
    cells = collect_cells(st.load(targets))
    acells = collect_cells(st.load(anchors))
    clusters = bed.clusters(0)

    basis_neff = float(sys.argv[3]) if len(sys.argv) > 3 else min_neff
    A = anchor_basis(bed, clusters, anchors, basis_neff)
    print(f"anchor basis {A.shape}  (clusters x anchor CDF columns)")
    U_full, sv, _ = np.linalg.svd(A, full_matrices=False)
    print("variance explained by the first 8 directions: "
          + " ".join(f"{v:.2f}" for v in np.cumsum(sv**2 / (sv**2).sum())[:8]))

    pos = {c: i for i, c in enumerate(clusters)}

    def geometry(cell_src, items):
        """Per-item geometry, computed once: the deviation matrix, the level,
        the truths and the weights. Everything the (r, s) sweep needs."""
        out = []
        for item_id in items:
            cids = [c for c in clusters
                    if (item_id, c) in cell_src
                    and bed.stats.hist(c, item_id) is not None
                    and bed.stats.hist(c, item_id).sum() > 0
                    and bed.cell_neff(c, item_id) >= min_neff]
            if len(cids) < 3:
                continue
            raw = [cell_src[(item_id, c)].mean(axis=0) for c in cids]
            if len({len(x) for x in raw}) != 1:
                continue
            wt = np.asarray([bed.cell_weight(c, item_id) for c in cids])
            C = np.vstack([cdf(x) for x in raw])
            D = C - (wt / wt.sum()) @ C
            full = np.zeros((len(clusters), D.shape[1]))
            for j, c in enumerate(cids):
                full[pos[c]] = D[j]
            out.append({
                "rows": np.asarray([pos[c] for c in cids]),
                "full": full, "wt": wt,
                "lvl": cdf(bed.level_hist(item_id, cids)),
                "truth": [np.asarray(bed.stats.hist(c, item_id), float) for c in cids],
            })
        return out

    def score(geo, r, s):
        U = U_full[:, :r] if r else None
        num = den = 0.0
        for g in geo:
            D = (U @ (U.T @ g["full"]))[g["rows"]] if U is not None else g["full"][g["rows"]]
            for j in range(len(g["wt"])):
                num += g["wt"][j] * w1(from_cdf(g["lvl"] + s * D[j]), g["truth"][j])
                den += g["wt"][j]
        return num / den if den else float("nan")

    geo_a, geo_t = geometry(acells, anchors), geometry(cells, targets)
    print(f"geometry: {len(geo_a)} anchor items, {len(geo_t)} target items")
    print(f"\n{'r':>4s} {'s*anchor':>9s} {'anchorW1':>9s} {'targetW1@s*':>12s} "
          f"{'s*target':>9s} {'targetW1 best':>14s}")
    for r in RS:
        a_curve = {s: score(geo_a, r, s) for s in SS}
        s_a = min(a_curve, key=a_curve.get)
        t_curve = {s: score(geo_t, r, s) for s in SS}
        s_t = min(t_curve, key=t_curve.get)
        print(f"{r:4d} {s_a:9.2f} {a_curve[s_a]:9.4f} {score(geo_t, r, s_a):12.4f} "
              f"{s_t:9.2f} {t_curve[s_t]:14.4f}", flush=True)


if __name__ == "__main__":
    main()
