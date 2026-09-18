"""Re-derive the headline number without the harness, and insist the two agree.

The whole result rests on one number computed by one code path, and the two bugs
this project has already hit — the deviation reference and the level being
different objects, and the projection zero-padding absent clusters — were both
silent: they produced a plausible number, not an error. So the number is
recomputed here from first principles, in code that shares nothing with
``evalx/harness.py`` except the stored calibrator and the bed:

  * truths and weights read straight from ``ClusterStats``
  * the level built by hand as the weighted mixture of the scored cells
  * the deviation, the projection and the scale applied by hand from the
    calibrator's own JSON
  * W1 written out as the sum of absolute CDF differences

A disagreement past 1e-9 means one of the two is wrong, and it does not matter
which — the number is not reportable until they agree.

    python scripts/verify_headline.py <model> [profile]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from popsim.agents.runner import ElicitationStore
from popsim.config import load_config
from popsim.evalx.harness import collect_cells, fit_from_store
from popsim.pipeline import build_bed


def w1_by_hand(p, q) -> float:
    p = np.asarray(p, float); q = np.asarray(q, float)
    k = p.size
    cp = np.cumsum(p / p.sum())[:-1]
    cq = np.cumsum(q / q.sum())[:-1]
    return float(np.abs(cp - cq).sum() / (k - 1))


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "ministral-8b-2512"
    profile = sys.argv[2] if len(sys.argv) > 2 else "permutation"
    cfg = load_config("configs/gss_main.yaml")
    bed = build_bed(cfg)
    store = ElicitationStore(root=cfg.repo_root / "runs/_elicit_store",
                             model=model, profile=profile)
    min_neff = float(cfg["evaluation.truth_min_neff"])
    anchors = sorted(bed.split["anchors"])
    targets = bed.ranked_targets()
    cal, _ = fit_from_store(bed, collect_cells(store.load(anchors)), min_neff=min_neff)
    cells = collect_cells(store.load(targets))
    clusters = bed.clusters(0)

    U = (np.asarray(cal.subspace_u, float)[:, : cal.subspace_rank]
         if cal.subspace_rank else None)
    pos = {c: i for i, c in enumerate(cal.subspace_clusters)}

    per_item, per_item_b0a = {}, {}
    for item_id in targets:
        cs = [c for c in clusters
              if (item_id, c) in cells
              and bed.stats.hist(c, item_id) is not None
              and bed.stats.hist(c, item_id).sum() > 0
              and bed.cell_neff(c, item_id) >= min_neff]
        if len(cs) < 3:
            continue
        k = len(bed.codes[item_id])
        truth = [np.asarray(bed.stats.hist(c, item_id), float) for c in cs]
        wt = np.asarray([bed.cell_weight(c, item_id) for c in cs], float)
        wn = wt / wt.sum()
        raw = [cells[(item_id, c)].mean(axis=0) for c in cs]
        if any(r.size != k for r in raw):
            continue

        # level = weighted mixture of the scored cells' truths, by hand
        level = wn @ np.vstack(truth)
        # deviation = each cluster's cdf minus the weighted mean of the cdfs
        C = np.vstack([np.cumsum(r / r.sum())[:-1] for r in raw])
        D = C - wn @ C
        if U is not None and len(cs) > cal.subspace_rank:
            rows = np.asarray([pos[c] for c in cs if c in pos])
            if rows.size == len(cs):
                q, _ = np.linalg.qr(U[rows])
                q = q[:, : min(q.shape[1], rows.size)]
                D = q @ (q.T @ D)
        s = cal.scale_for(k, topic=bed.topics.get(item_id))
        cl = np.cumsum(level / level.sum())[:-1]

        w1s, b0as = [], []
        for j in range(len(cs)):
            c = np.maximum.accumulate(np.clip(cl + s * D[j], 0.0, 1.0))
            h = np.clip(np.diff(np.concatenate([[0.0], c, [1.0]])), 0.0, None)
            h = h / h.sum()
            if cal.variance_restoration:
                raise SystemExit("this check assumes variance restoration is off")
            w1s.append(w1_by_hand(h, truth[j]))
            b0as.append(w1_by_hand(level, truth[j]))
        per_item[item_id] = float(np.asarray(w1s) @ wn)
        per_item_b0a[item_id] = float(np.asarray(b0as) @ wn)

    mine = float(np.mean(list(per_item.values())))
    mine_b0a = float(np.mean(list(per_item_b0a.values())))

    # The MOST RECENT matching report, by file mtime — not the one with the most
    # items. Several reports share a model tag and an item count, and comparing
    # against an arbitrary one of them compares against whichever code version
    # wrote it, which is how this check first "failed" against a run that predated
    # the projection fix.
    cands = []
    for path in Path("runs").glob("*/layer4_report.json"):
        try:
            r = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if (r.get("model") == model and r.get("profile", "permutation") == profile
                and not r.get("ensemble_limit")):
            cands.append((path.stat().st_mtime, path, r))
    if not cands:
        print(f"no harness report for {model}/{profile} to compare against")
        return 2
    _, path, rep = max(cands, key=lambda x: x[0])
    print(f"comparing against {path.parent.name}")
    t = rep["verdicts"]["all_targets"]

    print(f"model {model}  profile {profile}  items {len(per_item)}")
    print(f"  independent   system {mine:.6f}   B0a {mine_b0a:.6f}")
    print(f"  harness       system {t['achieved']:.6f}   B0a {t['baseline_b0a']:.6f}")
    d1, d2 = abs(mine - t["achieved"]), abs(mine_b0a - t["baseline_b0a"])
    print(f"  difference    {d1:.2e} / {d2:.2e}")
    ok = d1 < 1e-9 and d2 < 1e-9
    print("  => AGREE" if ok else "  => DISAGREE — the number is not reportable")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
