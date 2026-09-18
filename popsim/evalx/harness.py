"""M9 — the evaluation harness, and Layer 4: does it beat the baseline? *(the actual claim)*

    "Pre-registered pass marks, measured at K ~ 56 on pooled 2010-2022 ...
     Write these into `configs/gss_main.yaml` **before** the first elicitation
     run. A pass mark chosen after seeing results is not a pass mark."

The pass marks in the config have not moved since they were written and are read
from it here rather than restated, so there is one copy and it is the one the
run snapshot records.

The honest reporting rule is enforced structurally: every number this module
emits travels as a triple — **achieved W1, the B0a baseline, the noise floor** —
because "W1 = 0.052" means nothing and "0.052 against a baseline of 0.070 and a
floor of 0.027" is a sentence a reviewer can check.

Heterogeneity, for the top-quartile subset
------------------------------------------
Defined here as the cluster-weighted W1 between each cluster's truth and the
national marginal — which is B0a's own score on that item. It is the direct
measure of "do the clusters genuinely differ on this question", it uses only
truth and never a prediction, so selecting the top quartile by it is not
selection on results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..calibration.fit import (
    CalibrationPair,
    Calibrator,
    apply_calibrator,
    deviation_reference,
    fit_calibrator,
)
from ..calibration.shape import cdf, from_cdf, sd_pos
from .baselines import (
    b0a_national_oracle,
    b0b_national_predicted,
    b1_nearest_anchor,
    b3_supervised_skyline,
    b4_uncalibrated,
)
from .metrics import (
    between_cluster_sd,
    between_cluster_spearman,
    ece,
    interval_coverage,
    ordering_spearman,
    w1,
)

__all__ = ["EvalReport", "ItemResult", "collect_cells", "evaluate_targets", "fit_from_store"]

#: Arms scored side by side. "system" is the method; everything else is a
#: baseline or an ablation of it.
ARMS = ("system", "B0a", "B0a_national", "B0b", "B1", "B3", "B4",
        "system_s1", "system_percluster", "system_bytopic", "system_nosubspace",
        "system_adversarial")


def collect_cells(df: pd.DataFrame, *, max_members: int = 0
                  ) -> dict[tuple[str, str], np.ndarray]:
    """(item, cluster) -> the ensemble's member histograms, stacked.

    ``max_members`` truncates each cell's ensemble, which is how the §7.3
    ensemble-size sweep runs for free: the 3-draw store already contains the
    1-draw run. Truncation takes the FIRST members in store order, which is
    (paraphrase, repeat) order, so the 1-member arm is the first draw of the
    first paraphrase — the run a 1x1 profile would actually have made, not a
    lucky pick from three.
    """
    if df.empty:
        return {}
    ok = df[df["ok"].astype(bool)]
    if "paraphrase_id" in ok.columns and "repeat_id" in ok.columns:
        ok = ok.sort_values(["item_id", "cluster_id", "paraphrase_id", "repeat_id"])
    out: dict[tuple[str, str], np.ndarray] = {}
    for (item_id, cid), g in ok.groupby(["item_id", "cluster_id"]):
        hs = [np.asarray(h, dtype=float) for h in g["hist"] if h is not None]
        hs = [h for h in hs if h.size and np.isfinite(h).all() and h.sum() > 0]
        if max_members:
            hs = hs[:max_members]
        if hs:
            out[(item_id, cid)] = np.vstack([h / h.sum() for h in hs])
    return out


def fit_adversarial(
    bed, anchor_cells, target_topics, *, min_neff: float = 30.0
) -> dict[str, Calibrator]:
    """One calibrator per target topic, fitted with that topic's anchors removed.

    Checklist 5.1 / spec §M5: "an ablation with adversarial topic-disjoint splits
    to measure transfer honestly." The standard split stratifies anchors and
    targets by topic, so a civil-liberties target is calibrated partly on
    civil-liberties anchors. This arm asks what is left when it is not — which is
    the situation a genuinely novel scenario question is always in, since no
    anchor shares its subject.
    """
    out: dict[str, Calibrator] = {}
    for topic in sorted(set(target_topics.values())):
        kept = {k: v for k, v in anchor_cells.items()
                if bed.topics.get(k[0], "other") != topic}
        n_items = len({k[0] for k in kept})
        if n_items < 4:
            continue
        try:
            cal, _ = fit_from_store(bed, kept, min_neff=min_neff)
        except ValueError:
            continue
        out[topic] = cal
    return out


def anchor_deviation_basis(
    bed, cluster_ids: list[str], anchors: list[str], min_neff: float
) -> tuple[list[str], np.ndarray]:
    """Where real subgroup variation actually points, from anchor truths alone.

    One column per (anchor item, interior CDF position), centred across
    clusters. An anchor contributes only if every cluster in ``cluster_ids`` has
    a usable cell for it, so the columns are comparable.

    ``min_neff`` here is deliberately looser than the scoring threshold: this
    matrix is never scored against, it only has to point in the right
    direction, and dropping a whole anchor because one small cluster is thin
    costs a direction for no gain in honesty.
    """
    cols: list[np.ndarray] = []
    used: list[str] = []
    for a in anchors:
        hs = []
        for c in cluster_ids:
            h = bed.stats.hist(c, a)
            if h is None or h.sum() <= 0 or bed.cell_neff(c, a) < min_neff:
                hs = []
                break
            hs.append(cdf(h))
        if hs:
            M = np.vstack(hs)
            cols.append(M - M.mean(axis=0))
            used.append(a)
    A = np.hstack(cols) if cols else np.zeros((len(cluster_ids), 0))
    return list(cluster_ids), A


def fit_from_store(
    bed,
    anchor_cells: dict[tuple[str, str], np.ndarray],
    *,
    mode: str = "observed_level",
    min_neff: float = 30.0,
    variance_restoration: bool | None = None,
    fit_per_cluster: bool = True,
    level_cells: dict[tuple[str, str], np.ndarray] | None = None,
    use_subspace: bool = True,
) -> tuple[Calibrator, dict[str, Any]]:
    """Build the cross-fitted anchor pairs and fit the layer on them."""
    pairs: list[CalibrationPair] = []
    national: dict[str, np.ndarray] = {}
    for (item_id, cid), stack in anchor_cells.items():
        truth = bed.stats.hist(cid, item_id)
        if truth is None or truth.sum() <= 0:
            continue
        if bed.cell_neff(cid, item_id) < min_neff:
            continue
        raw = stack.mean(axis=0)
        if raw.size != len(truth):
            continue
        pairs.append(CalibrationPair(
            cluster_id=cid, item_id=item_id, raw=[float(x) for x in raw],
            truth=[float(x) for x in np.asarray(truth, dtype=float)],
            weight=bed.cell_weight(cid, item_id), n_eff=bed.cell_neff(cid, item_id),
        ))

    # The level for every anchor item is the mixture over exactly the cells that
    # ended up in the fit — the same object `Bed.level_hist` builds, and the only
    # one for which s = 0 is the no-conditioning baseline identically.
    by_item: dict[str, list[CalibrationPair]] = {}
    for pr in pairs:
        by_item.setdefault(pr.item_id, []).append(pr)
    for item_id, ps in by_item.items():
        w = np.asarray([p.weight for p in ps], dtype=float)
        w = w / w.sum() if w.sum() > 0 else np.full(w.size, 1.0 / w.size)
        national[item_id] = w @ np.vstack([np.asarray(p.truth, dtype=float) for p in ps])

    level_pairs = None
    if mode == "predicted_level" and level_cells:
        level_pairs = []
        for (item_id, cid), stack in level_cells.items():
            if cid != "all" or item_id not in national:
                continue
            level_pairs.append((stack.mean(axis=0), national[item_id]))

    basis = None
    if use_subspace:
        basis = anchor_deviation_basis(
            bed, bed.clusters(0), sorted(bed.split["anchors"]),
            float(bed.cfg.get("calibration.basis_min_neff", 8)))
        if basis[1].shape[1] == 0:
            basis = None

    cal = fit_calibrator(
        pairs, national, topics=bed.topics, mode=mode,
        variance_restoration=variance_restoration,
        fit_per_cluster=fit_per_cluster,
        tau=float(bed.cfg["partition.tau_pooling"]),
        level_pairs=level_pairs, basis=basis, folds=dict(bed.plan.fold_of),
        schemes=tuple(bed.cfg.get("calibration.scale_schemes", ("global", "by_k"))),
        ranks=tuple(bed.cfg.get("calibration.subspace_ranks", (0, 1, 2, 3, 4, 6, 8, 12))),
    )
    diag = {
        "n_anchor_pairs": len(pairs),
        "anchor_items": sorted(national),
        "n_level_pairs": len(level_pairs or []),
        "basis_shape": list(basis[1].shape) if basis else None,
    }
    return cal, diag


@dataclass
class ItemResult:
    item_id: str
    topic: str
    k: int
    n_clusters: int
    heterogeneity: float
    noise_floor: float
    w1: dict[str, float] = field(default_factory=dict)
    rho: float = float("nan")
    rho_p: float = float("nan")
    rho_raw: float = float("nan")
    btw_sd_true: float = float("nan")
    btw_sd_pred: float = float("nan")
    var_ratio: float = float("nan")
    ece: float = float("nan")
    coverage90: float = float("nan")
    scale_used: float = float("nan")
    leakage_resistant: bool = False
    famous: bool = False


@dataclass
class EvalReport:
    model: str
    mode: str
    #: Which store this scored. Two arms can share a model tag — the no-anchor
    #: ablation is the same model with a different card — so the report keys on
    #: (model, profile), not on the model alone.
    profile: str = "permutation"
    ensemble_limit: int = 0
    n_items: int = 0
    n_clusters: int = 0
    noise_floor: float = float("nan")
    per_item: list[ItemResult] = field(default_factory=list)
    macro: dict[str, dict[str, float]] = field(default_factory=dict)
    verdicts: dict[str, dict[str, Any]] = field(default_factory=dict)
    calibrator: dict[str, Any] = field(default_factory=dict)
    population: dict[str, Any] = field(default_factory=dict)
    battery: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.verdicts.get("all_targets", {}).get("pass"))

    def summary(self) -> str:
        lines = [
            "Layer 4 — the held-out-item verdict",
            f"  model   {self.model}   (store profile {self.profile}"
            + (f", ensemble capped at {self.ensemble_limit}" if self.ensemble_limit else "")
            + ")",
            f"  mode    {self.mode}",
            f"  scope   {self.n_items} target items x {self.n_clusters} clusters",
            "",
            (f"  {'subset':20s} {'achieved':>9s} {'B0a here':>9s} {'B0a pre':>8s} "
             f"{'mark':>7s} {'floor':>7s}   absolute   relative"),
        ]
        for name, v in self.verdicts.items():
            lines.append(
                f"  {name:20s} {v['achieved']:9.4f} {v['baseline_b0a']:9.4f} "
                f"{v['baseline_b0a_preregistered']:8.4f} {v['pass_w1']:7.4f} "
                f"{v['noise_floor']:7.4f}   "
                f"{'PASS' if v['pass'] else 'fail':4s}       "
                f"{'PASS' if v['beats_measured_baseline_by_20pct'] else 'fail':4s}"
                f"   ({v['vs_baseline_pct']:+.1f}% vs B0a, "
                f"{v['signal_captured_pct']:.0f}% of the available signal)"
            )
        lines += [
            "",
            "  two verdicts, both reported (PREREGISTRATION.md §8.8):",
            "    absolute — against the pass mark frozen in the config",
            "    relative — §1.4(i)'s own words, 20% below the B0a measured on THIS scope",
        ]
        lines += ["", "  arms, macro W1 over all scored targets:"]
        for arm, val in sorted(self.macro.get("all_targets", {}).items(),
                               key=lambda kv: (np.isnan(kv[1]), kv[1])):
            if np.isfinite(val):
                lines.append(f"    {arm:20s} {val:.4f}")
        b = self.battery
        if b.get("rho") is not None and np.isfinite(b.get("rho", np.nan)):
            lines += ["", (f"  tolerance-battery ordering rho {b['rho']:+.3f} over "
                           f"{b.get('n_clusters', 0)} clusters, "
                           f"{b.get('n_items', 0)} items")]
        p = self.population
        if p:
            vr = p.get("variance_ratio", float("nan"))
            lines += ["", (f"  population variance ratio {vr:.3f} "
                           f"(band {p.get('band')}) -> "
                           f"{'PASS' if p.get('pass') else 'fail'}"),
                      f"  {p.get('weights', '')}"]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ok"] = self.ok
        return d


def _weighted(vals: list[float], wts: list[float]) -> float:
    v = np.asarray(vals, dtype=float)
    w = np.asarray(wts, dtype=float)
    good = np.isfinite(v) & (w > 0)
    if not good.any():
        return float("nan")
    return float(np.average(v[good], weights=w[good]))


def _bootstrap_intervals(stack_by_cluster, level_c, ref, s, n_boot, alpha, rng,
                         resid_sd=None):
    """90% intervals per (cluster, option), from two sources that are not the same.

    **Ensemble spread** — resample the elicitation draws, recompute the
    deviation, recompute the calibrated histogram. This measures how much the
    answer moves when the model is asked again, and on three draws it is small.

    **Anchor residual** — the SD of (truth − prediction) measured out of fold on
    the anchors, at each CDF position. This measures how far the *finished*
    prediction sat from the truth on questions where the truth is known, which is
    the error that actually matters, and it is the only part of it estimable
    without a target's truth.

    Using the first alone is what §M6 calls an honest interval and is not: the
    8B arm's 90 % intervals covered **15.7 %** of true values that way. Adding the
    residual is not a widening fudge — it is the missing term, and it is fitted on
    anchors like everything else.

    The within-cluster variance tempering is not re-applied inside the bootstrap:
    it is a deterministic function of the histogram, so it shifts every draw the
    same way.
    """
    cids = list(stack_by_cluster)
    draws = {c: [] for c in cids}
    for _ in range(n_boot):
        means = {}
        for c in cids:
            st = stack_by_cluster[c]
            idx = rng.integers(0, st.shape[0], st.shape[0])
            means[c] = st[idx].mean(axis=0)
        for c in cids:
            base = level_c + s * (cdf(means[c]) - ref)
            if resid_sd is not None:
                base = base + rng.normal(0.0, resid_sd)
            draws[c].append(from_cdf(base))
    lo, hi = {}, {}
    for c in cids:
        arr = np.vstack(draws[c])
        lo[c] = np.quantile(arr, alpha / 2, axis=0)
        hi[c] = np.quantile(arr, 1 - alpha / 2, axis=0)
    return lo, hi


def evaluate_targets(
    bed,
    target_cells: dict[tuple[str, str], np.ndarray],
    cal: Calibrator,
    *,
    items: list[str],
    cluster_ids: list[str],
    model: str,
    level_cells: dict[tuple[str, str], np.ndarray] | None = None,
    n_boot: int = 200,
    with_b3: bool = True,
    leaky_items: list[str] | None = None,
    adversarial: dict[str, Calibrator] | None = None,
    profile: str = "permutation",
    ensemble_limit: int = 0,
    out_dir: str | Path | None = None,
) -> EvalReport:
    cfg = bed.cfg
    min_neff = float(cfg["evaluation.truth_min_neff"])
    marks = cfg["evaluation.pass_marks"]
    floor = float(marks["all_targets"]["noise_floor"])
    alpha = float(cfg["aggregation.ci_alpha"])
    rng = np.random.default_rng(int(cfg["seed"]))
    leak_set = set(cfg["evaluation.leakage_resistant_items"])
    flagged = set(leaky_items or ())
    famous = set(bed.split.get("famous", []))

    per_item: list[ItemResult] = []
    # Kept for the population rollup and the battery-ordering metric.
    sys_by_item: dict[str, dict[str, np.ndarray]] = {}
    truth_by_item: dict[str, dict[str, np.ndarray]] = {}

    for item_id in items:
        cids = [c for c in cluster_ids if (item_id, c) in target_cells]
        cids = [c for c in cids
                if bed.stats.hist(c, item_id) is not None
                and bed.stats.hist(c, item_id).sum() > 0
                and bed.cell_neff(c, item_id) >= min_neff]
        if len(cids) < 3:
            continue
        k = len(bed.codes[item_id])
        truth = {c: np.asarray(bed.stats.hist(c, item_id), dtype=float) for c in cids}
        wt = {c: bed.cell_weight(c, item_id) for c in cids}
        stacks = {c: target_cells[(item_id, c)] for c in cids}
        if any(stacks[c].shape[1] != k for c in cids):
            continue
        raw = {c: stacks[c].mean(axis=0) for c in cids}
        # The level, and B0a, are the same object: the weighted mixture of the
        # scored cells. See Bed.level_hist.
        national = bed.level_hist(item_id, cids)
        topic = bed.topics.get(item_id, "other")

        arms: dict[str, dict[str, np.ndarray]] = {}
        arms["B0a"] = b0a_national_oracle(national, cids)
        arms["B0a_national"] = b0a_national_oracle(bed.national(item_id), cids)
        arms["B4"] = b4_uncalibrated(raw)
        arms["system"] = apply_calibrator(cal, raw, national, wt, topic=topic)
        arms["system_s1"] = apply_calibrator(
            cal, raw, national, wt, topic=topic, scheme="unit")
        if cal.scale_by_cluster:
            arms["system_percluster"] = apply_calibrator(
                cal, raw, national, wt, topic=topic, scheme="by_cluster")
        if cal.scale_by_topic:
            arms["system_bytopic"] = apply_calibrator(
                cal, raw, national, wt, topic=topic, scheme="by_topic")
        if cal.subspace_rank:
            flat = Calibrator(**{**asdict(cal), "subspace_rank": 0})
            arms["system_nosubspace"] = apply_calibrator(
                flat, raw, national, wt, topic=topic)
        if adversarial and topic in adversarial:
            # §5.1's adversarial regime: this target's whole topic was held out
            # of the anchors the calibrator saw, so nothing it learned came from
            # a question about the same subject.
            arms["system_adversarial"] = apply_calibrator(
                adversarial[topic], raw, national, wt, topic=topic)
        b1, _b1_src = b1_nearest_anchor(
            item_id, bed.plan.anchors_for(item_id), bed.codebook, bed.stats, cids)
        if b1:
            arms["B1"] = b1
        if level_cells and (item_id, "all") in level_cells:
            pred_nat = level_cells[(item_id, "all")].mean(axis=0)
            if pred_nat.size == k:
                arms["B0b"] = b0b_national_predicted(pred_nat, cids)
        if with_b3:
            b3 = b3_supervised_skyline(
                bed.frame, bed.assign, item_id, bed.codes[item_id], cids)
            if b3:
                arms["B3"] = b3

        res = ItemResult(
            item_id=item_id, topic=topic, k=k, n_clusters=len(cids),
            heterogeneity=_weighted([w1(national, truth[c]) for c in cids],
                                    [wt[c] for c in cids]),
            noise_floor=floor,
            leakage_resistant=item_id in leak_set, famous=item_id in famous,
            scale_used=cal.scale_for(k, topic=topic),
        )
        for arm, pred in arms.items():
            common = [c for c in cids if c in pred]
            res.w1[arm] = _weighted([w1(pred[c], truth[c]) for c in common],
                                    [wt[c] for c in common])

        sys_pred = arms["system"]
        common = [c for c in cids if c in sys_pred]
        res.rho, res.rho_p = between_cluster_spearman(
            [sys_pred[c] for c in common], [truth[c] for c in common])
        res.rho_raw, _ = between_cluster_spearman(
            [raw[c] for c in common], [truth[c] for c in common])
        res.btw_sd_true = between_cluster_sd([truth[c] for c in common],
                                             [wt[c] for c in common])
        res.btw_sd_pred = between_cluster_sd([sys_pred[c] for c in common],
                                             [wt[c] for c in common])
        sd_t = np.asarray([sd_pos(truth[c]) for c in common])
        sd_p = np.asarray([sd_pos(sys_pred[c]) for c in common])
        good = np.isfinite(sd_t) & (sd_t > 0)
        res.var_ratio = float(np.mean(sd_p[good] / sd_t[good])) if good.any() else float("nan")
        res.ece = ece(np.vstack([sys_pred[c] for c in common]),
                      np.vstack([truth[c] for c in common]),
                      np.repeat([wt[c] for c in common], k))

        if n_boot:
            ref = deviation_reference([raw[c] for c in cids],
                                      np.asarray([wt[c] for c in cids]))
            s = cal.scale_for(k, topic=topic)
            lo, hi = _bootstrap_intervals(stacks, cdf(national), ref, s, n_boot,
                                          alpha, rng, resid_sd=cal.residual_sd(k))
            res.coverage90 = interval_coverage(
                np.concatenate([lo[c] for c in common]),
                np.concatenate([hi[c] for c in common]),
                np.concatenate([truth[c] for c in common]))

        per_item.append(res)
        sys_by_item[item_id] = sys_pred
        truth_by_item[item_id] = truth

    # ------------------------------------------------------------- subsets
    def subset(name: str) -> list[ItemResult]:
        if name == "all_targets":
            return per_item
        if name == "leakage_resistant":
            return [r for r in per_item if r.leakage_resistant]
        if name == "minus_leaky":
            return [r for r in per_item if r.item_id not in flagged]
        if name == "top_quartile_het":
            if not per_item:
                return []
            cut = float(np.quantile([r.heterogeneity for r in per_item], 0.75))
            return [r for r in per_item if r.heterogeneity >= cut]
        return []

    macro: dict[str, dict[str, float]] = {}
    verdicts: dict[str, dict[str, Any]] = {}
    subsets = ["all_targets", "top_quartile_het", "leakage_resistant"]
    if flagged:
        subsets.append("minus_leaky")
    for name in subsets:
        rs = subset(name)
        if not rs:
            continue
        macro[name] = {
            arm: float(np.nanmean([r.w1.get(arm, np.nan) for r in rs]))
            for arm in ARMS
        }
        # A subset with no pre-registered mark of its own is judged against the
        # all-targets mark, and the fact is recorded rather than implied.
        mark = marks.get(name, marks["all_targets"])
        achieved = macro[name]["system"]
        # The baseline reported is the one measured on THIS scope, beside the
        # pre-registered figure. A pass is judged against the pre-registered
        # mark, never against a baseline recomputed after the fact.
        b0a_here = macro[name]["B0a"]
        nf = float(mark["noise_floor"])
        headroom = max(b0a_here - nf, 1e-9)
        verdicts[name] = {
            "achieved": achieved,
            "pass_w1": float(mark["pass_w1"]),
            "baseline_b0a_preregistered": float(mark["baseline_w1"]),
            "baseline_b0a": b0a_here,
            "noise_floor": nf,
            "pass": bool(np.isfinite(achieved) and achieved <= float(mark["pass_w1"])),
            "beats_measured_baseline_by_20pct": bool(
                np.isfinite(achieved) and achieved <= 0.8 * b0a_here),
            "vs_baseline_pct": float((achieved / b0a_here - 1) * 100)
            if b0a_here > 0 else float("nan"),
            "signal_captured_pct": float((b0a_here - achieved) / headroom * 100),
            "n_items": len(rs),
        }

    # -------------------------------------------- population variance ratio
    pop: dict[str, Any] = {}
    if per_item:
        # Raked to ACS 2024, not survey-weighted: the population claim in
        # §1.4(ii) is about US adults, not about the pooled GSS sample.
        w = bed.raked_weight
        ratios = []
        for item_id, pred in sys_by_item.items():
            cs = [c for c in pred if c in truth_by_item[item_id]]
            ws = np.asarray([w.get(c, 0.0) for c in cs], dtype=float)
            if ws.sum() <= 0:
                continue
            ws = ws / ws.sum()
            hp = ws @ np.vstack([pred[c] for c in cs])
            ht = ws @ np.vstack([truth_by_item[item_id][c] for c in cs])
            if sd_pos(ht) > 0:
                ratios.append(sd_pos(hp) / sd_pos(ht))
        band = cfg["evaluation.variance_ratio_band"]
        vr = float(np.mean(ratios)) if ratios else float("nan")
        pop = {"variance_ratio": vr, "band": list(band),
               "pass": bool(np.isfinite(vr) and band[0] <= vr <= band[1]),
               "n_items": len(ratios),
               "weights": bed.notes.get("raking", "survey weights")}

    # ------------------------------------------- tolerance-battery ordering
    battery: dict[str, Any] = {}
    fam = [i for i in sys_by_item if i[:3] in ("spk", "col", "lib")]
    if len(fam) >= 3:
        rhos = []
        shared = set.intersection(*[set(sys_by_item[i]) for i in fam])
        for c in sorted(shared):
            rho, _ = ordering_spearman(
                {i: sys_by_item[i][c] for i in fam},
                {i: truth_by_item[i][c] for i in fam})
            if np.isfinite(rho):
                rhos.append(rho)
        battery = {"items": sorted(fam), "n_items": len(fam),
                   "n_clusters": len(rhos),
                   "rho": float(np.mean(rhos)) if rhos else float("nan"),
                   "excluded": ["colrac (SNR 0.64)", "librac (1.21)", "colmil (1.48)"]}

    report = EvalReport(
        model=model, mode=cal.mode, profile=profile, ensemble_limit=ensemble_limit,
        n_items=len(per_item), n_clusters=len(cluster_ids),
        noise_floor=floor, per_item=per_item, macro=macro, verdicts=verdicts,
        calibrator=json.loads(json.dumps(asdict(cal), default=float)),
        population=pop, battery=battery,
        notes={"n_boot": n_boot, "min_neff": min_neff,
               "leaky_items_excluded": sorted(flagged),
               "arms": [a for a in ARMS],
               "heterogeneity": "cluster-weighted W1 from each cluster's truth to "
                                "the national marginal; truth only, no prediction"},
    )
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "layer4_report.json").write_text(
            json.dumps(report.to_dict(), indent=2, default=str))
        (out / "layer4_summary.txt").write_text(report.summary())
        pd.DataFrame([asdict(r) | {f"w1_{k}": v for k, v in r.w1.items()}
                      for r in per_item]).drop(columns=["w1"]).to_csv(
            out / "layer4_per_item.csv", index=False)
    return report
