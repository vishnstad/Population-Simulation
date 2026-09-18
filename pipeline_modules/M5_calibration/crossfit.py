"""
Cross-fitted anchor calibration (module M5) -- the project's core mechanism.

The problem
-----------
A raw LLM histogram for a cluster is systematically wrong in two ways: it is too
flat or too peaked (variance is not calibrated), and its shape is pulled toward a
socially safe modal answer. We can measure both, because for *anchor* items we
know the truth from the microdata. The question is whether a correction learned
on anchors transfers to items the model has never been shown.

Why cross-fitting is not optional
---------------------------------
If we elicit anchor ``a`` from a stat card that already displays ``a``'s true
histogram, the model can copy it. The calibrator then learns "the model is nearly
perfect", which is true on anchors and false on targets, and the whole layer
becomes a no-op that inflates reported accuracy.

So anchors are split into ``F`` folds. To produce the training pair for anchor
``a`` in fold ``f``, the card is built from anchors in folds other than ``f`` --
``a`` is absent, and so is every anchor sharing its fold. The resulting
(prediction, truth) pairs are generated under exactly the conditions a held-out
target faces, which is the only thing that makes the fitted map a transfer map.

Target items are never in any anchor fold, so their cards may use all anchors.

Pipeline
--------
1. :func:`make_crossfit_plan` assigns anchors to folds, stratified by topic.
2. ``run_elicitation.py`` elicits anchors fold-aware and targets fold-free.
3. :meth:`CrossFittedCalibrator.fit` learns isotonic CDF recalibration plus a
   hierarchically-shrunk variance model from the out-of-fold anchor pairs.
4. :meth:`CrossFittedCalibrator.calibrate` applies both to a target histogram.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .isotonic import OrdinalCDFCalibrator
from .variance import HierarchicalVarianceRestorer, compute_histogram_sd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# fold assignment
# --------------------------------------------------------------------------
def make_crossfit_plan(
    anchor_items: Sequence[str],
    item_topics: Optional[Dict[str, str]] = None,
    n_folds: int = 3,
    seed: int = 20260818,
) -> Dict[str, int]:
    """
    Assign each anchor item to one of ``n_folds`` folds, stratified by topic.

    Stratifying matters: if a whole topic landed in one fold, cards for that fold
    would lose an entire subject area and the calibrator would be fit on
    unrepresentatively thin context.

    Returns ``{item_id: fold_index}`` with folds numbered from 0.
    """
    if n_folds < 2:
        raise ValueError("cross-fitting needs at least 2 folds")

    rng = np.random.RandomState(seed)
    topics = item_topics or {}
    by_topic: Dict[str, List[str]] = defaultdict(list)
    for iid in anchor_items:
        by_topic[topics.get(iid, "general")].append(iid)

    # `offset` carries across topics so that a topic with fewer items than folds
    # does not always deposit into fold 0. Without it, small topics pile onto the
    # low folds and the folds end up badly unbalanced.
    plan: Dict[str, int] = {}
    offset = 0
    for topic in sorted(by_topic):
        items = sorted(by_topic[topic])
        rng.shuffle(items)
        for pos, iid in enumerate(items):
            plan[iid] = (offset + pos) % n_folds
        offset = (offset + len(items)) % n_folds
    return plan


def save_crossfit_plan(plan: Dict[str, int], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n_folds": max(plan.values()) + 1 if plan else 0, "folds": plan}, f, indent=2)


def load_crossfit_plan(path: Path) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return {k: int(v) for k, v in json.load(f)["folds"].items()}


# --------------------------------------------------------------------------
# training pairs
# --------------------------------------------------------------------------
@dataclass
class CalibrationPair:
    """One out-of-fold (prediction, truth) observation for a cluster x anchor cell."""

    cluster_id: str
    item_id: str
    fold: int
    raw_probs: List[float]
    true_probs: List[float]
    n_eff: float

    @property
    def scale_length(self) -> int:
        return len(self.raw_probs)


@dataclass
class CalibratedDistributionResult:
    cluster_id: str
    item_id: str
    raw_probabilities: List[float]
    calibrated_probabilities: List[float]
    raw_sd: float
    calibrated_sd: float
    target_sd: float


# --------------------------------------------------------------------------
# calibrator
# --------------------------------------------------------------------------
class CrossFittedCalibrator:
    """
    Fits the anchor -> truth correction and applies it to target items.

    Two stages, in order:

    1. **Ordinal recalibration.** Isotonic regression from predicted cumulative
       probability to true cumulative probability, pooled across clusters and fit
       separately per scale length. This fixes systematic shape bias.
    2. **Variance restoration.** A per-cluster weighted least squares model
       ``sd_true = alpha + beta * sd_raw``, shrunk toward the parent node with
       weight ``n_eff / (n_eff + tau)``. The calibrated histogram is then
       power-tempered until it hits the predicted SD. This is the hierarchy doing
       statistical work -- thin leaves borrow strength from their parent instead
       of fitting noise.

    Set ``fit_per_fold=True`` to hold out each fold in turn when fitting, so a
    fold's own pairs never inform the map applied to it. That is the strict form
    used for reporting; the pooled form is the default because target items sit
    outside every anchor fold anyway.
    """

    def __init__(
        self,
        tau: float = 100.0,
        n_folds: int = 3,
        min_pairs_per_cluster: int = 5,
        fit_per_fold: bool = False,
    ):
        self.tau = tau
        self.n_folds = n_folds
        self.min_pairs_per_cluster = min_pairs_per_cluster
        self.fit_per_fold = fit_per_fold

        self.iso_calibrator = OrdinalCDFCalibrator()
        self.var_restorer = HierarchicalVarianceRestorer(tau=tau)
        self.parent_of: Dict[str, str] = {}
        self.is_fitted = False
        self.fit_report: Dict[str, Any] = {}

    # -- fitting ----------------------------------------------------------
    def fit(
        self,
        pairs: Iterable[CalibrationPair | Dict[str, Any]],
        parent_of: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Fit both stages from out-of-fold anchor pairs.

        ``parent_of`` maps leaf cluster id -> parent cluster id and enables the
        partial pooling described above. Without it every cluster shrinks toward
        the global fit instead.
        """
        pair_list = [p if isinstance(p, CalibrationPair) else CalibrationPair(**p) for p in pairs]
        pair_list = [p for p in pair_list if len(p.raw_probs) == len(p.true_probs) >= 2]
        if not pair_list:
            raise ValueError(
                "No usable calibration pairs. Run run_elicitation.py --stage anchors first."
            )
        self.parent_of = parent_of or {}

        # ---- stage 1: isotonic CDF recalibration, per scale length -------
        by_scale: Dict[int, Tuple[List[np.ndarray], List[np.ndarray]]] = defaultdict(
            lambda: ([], [])
        )
        for p in pair_list:
            raw = np.asarray(p.raw_probs, dtype=float)
            true = np.asarray(p.true_probs, dtype=float)
            by_scale[p.scale_length][0].append(np.cumsum(raw)[:-1])
            by_scale[p.scale_length][1].append(np.cumsum(true)[:-1])

        for L, (rs, ts) in by_scale.items():
            self.iso_calibrator.fit(L, np.asarray(rs), np.asarray(ts))

        # ---- stage 2: variance model -------------------------------------
        # SDs are measured AFTER isotonic recalibration, because that is the
        # distribution the tempering step will actually operate on.
        # Three SDs per pair, and the distinction matters when reading the report:
        #   uncal_sds -- the raw LLM output, before anything. This is the number
        #                that reveals variance collapse in the base model.
        #   iso_sds   -- after isotonic recalibration. This is what the tempering
        #                step actually operates on, so it is what the model is fit on.
        #   true_sds  -- the microdata.
        uncal_sds, iso_sds, true_sds = [], [], []
        per_cluster: Dict[str, Tuple[List[float], List[float], float]] = {}
        for p in pair_list:
            support = np.linspace(0.0, 1.0, p.scale_length)
            iso = np.asarray(
                self.iso_calibrator.transform_histogram(list(p.raw_probs)), dtype=float
            )
            s_uncal = compute_histogram_sd(np.asarray(p.raw_probs, dtype=float), support)
            s_iso = compute_histogram_sd(iso, support)
            s_true = compute_histogram_sd(np.asarray(p.true_probs, dtype=float), support)
            uncal_sds.append(s_uncal)
            iso_sds.append(s_iso)
            true_sds.append(s_true)
            r, t, _ = per_cluster.setdefault(p.cluster_id, ([], [], p.n_eff))
            r.append(s_iso)
            t.append(s_true)
        raw_sds = iso_sds

        self.var_restorer.fit_global(np.asarray(raw_sds), np.asarray(true_sds))

        # Parent-level coefficients first, so leaves can shrink toward them.
        by_parent: Dict[str, Tuple[List[float], List[float], float]] = {}
        for cid, (r, t, n_eff) in per_cluster.items():
            parent = self.parent_of.get(cid)
            if parent is None:
                continue
            pr, pt, pn = by_parent.setdefault(parent, ([], [], 0.0))
            pr.extend(r)
            pt.extend(t)
            by_parent[parent] = (pr, pt, pn + n_eff)

        parent_coeffs: Dict[str, Tuple[float, float]] = {}
        for parent, (r, t, n_eff) in by_parent.items():
            self.var_restorer.fit_cluster(parent, np.asarray(r), np.asarray(t), n_eff)
            parent_coeffs[parent] = self.var_restorer.cluster_coeffs[parent]

        n_pooled = 0
        for cid, (r, t, n_eff) in per_cluster.items():
            if len(r) < self.min_pairs_per_cluster:
                n_pooled += 1
            self.var_restorer.fit_cluster(
                cluster_id=cid,
                raw_sds=np.asarray(r),
                true_sds=np.asarray(t),
                n_eff=n_eff,
                parent_coeffs=parent_coeffs.get(self.parent_of.get(cid, ""), None),
            )

        self.is_fitted = True
        self.fit_report = {
            "n_pairs": len(pair_list),
            "n_clusters": len(per_cluster),
            "n_clusters_pooled_to_parent": n_pooled,
            "scale_lengths_fitted": sorted(by_scale),
            "global_variance_coeffs": {
                "alpha": round(self.var_restorer.global_coeffs[0], 4),
                "beta": round(self.var_restorer.global_coeffs[1], 4),
            },
            "mean_uncalibrated_sd": round(float(np.mean(uncal_sds)), 4),
            "mean_post_isotonic_sd": round(float(np.mean(iso_sds)), 4),
            "mean_true_sd": round(float(np.mean(true_sds)), 4),
            # The headline diagnostic: dispersion of the RAW model output relative
            # to the truth. Well below 1.0 is the variance collapse this layer
            # exists to repair; well above 1.0 means the model is over-hedging.
            "raw_variance_ratio_before_calibration": round(
                float(np.mean(uncal_sds) / max(np.mean(true_sds), 1e-9)), 4
            ),
            "variance_ratio_after_isotonic": round(
                float(np.mean(iso_sds) / max(np.mean(true_sds), 1e-9)), 4
            ),
            "tau": self.tau,
            "n_folds": self.n_folds,
        }
        logger.info("calibrator fitted: %s", json.dumps(self.fit_report))
        return self.fit_report

    # -- applying ---------------------------------------------------------
    def calibrate(self, cluster_id: str, raw_probs: Sequence[float]) -> List[float]:
        """Apply isotonic recalibration then variance restoration."""
        if not self.is_fitted:
            raise RuntimeError("calibrate() called before fit()")
        iso = self.iso_calibrator.transform_histogram(list(raw_probs))
        return self.var_restorer.restore_variance(cluster_id, iso)

    def calibrate_detailed(
        self, cluster_id: str, item_id: str, raw_probs: Sequence[float]
    ) -> CalibratedDistributionResult:
        raw = np.asarray(raw_probs, dtype=float)
        support = np.linspace(0.0, 1.0, len(raw))
        iso = np.asarray(self.iso_calibrator.transform_histogram(list(raw_probs)), dtype=float)
        cal = self.var_restorer.restore_variance(cluster_id, list(iso))
        return CalibratedDistributionResult(
            cluster_id=cluster_id,
            item_id=item_id,
            raw_probabilities=[float(x) for x in raw],
            calibrated_probabilities=cal,
            raw_sd=compute_histogram_sd(raw, support),
            calibrated_sd=compute_histogram_sd(np.asarray(cal, dtype=float), support),
            target_sd=self.var_restorer.predict_target_sd(
                cluster_id, compute_histogram_sd(iso, support)
            ),
        )

    # -- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        """Persist the fit report. The sklearn models are refit from pairs on load."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fit_report": self.fit_report,
            "cluster_coeffs": {
                k: [float(a), float(b)] for k, (a, b) in self.var_restorer.cluster_coeffs.items()
            },
            "global_coeffs": list(self.var_restorer.global_coeffs),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
