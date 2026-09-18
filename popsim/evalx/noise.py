"""Layer 1 — is the ground truth itself trustworthy? *(gate)*

    "For every (cluster, item) cell, estimate sampling noise in the **truth** by
     split-half resampling. Gate on SNR >= 1.5."

    *Catches:* scoring cells where truth is noise, which produces a confident
    wrong verdict in either direction. Absent from the spec — `truth_min_neff:
    30` is a crude proxy for it.

This module is the checklist's own addition (2.3) and it decides which items can
carry a result at all.

The measurement
---------------
**Noise.** Split each cell's respondents at random into two halves, build a
weighted histogram from each, and take W1 between them. Two half-size samples
differ by more than a full-size sample differs from the truth, by a factor of
about sqrt(2) under the usual independent-halves argument, so the half-vs-half
distance is divided by sqrt(2) to get the noise in a full-size cell. Averaged
over ``n_repeats`` random splits and over cells, cluster-weighted.

**Signal.** The cluster-weighted W1 between each cluster's true histogram and
the national marginal. This is exactly baseline B0a — "every cluster answers
like the average" — so the signal being measured is the thing the project has
to beat, not some other notion of spread.

**SNR = signal / noise**, and an item is scorable at SNR >= 1.5. Below that the
between-cluster differences an LLM is asked to reproduce are smaller than the
uncertainty in the numbers it is scored against, and a good result and a bad one
are indistinguishable.

Reading the output
------------------
An item with high SNR is one where clusters genuinely differ *and* the truth is
measured precisely enough to tell. Low SNR has two very different causes that
the split tells apart: a low signal means the item does not vary across these
clusters (tolerance toward racists, per §1.4 — a real finding, not a defect); a
high noise means the cells are too thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..clustering.stats import weighted_histogram
from .metrics import w1

__all__ = ["ItemNoise", "NoiseReport", "measure_noise_floor"]

#: Half-vs-half distance overstates full-sample error by a factor of 2.
#:
#: Let the estimator have SD sigma_n at sample size n, with sigma_n ~ 1/sqrt(n).
#: Each half has size n/2, so each half's SD is sigma_n * sqrt(2). The two halves
#: are independent, so the SD of their difference is sqrt(2) * sigma_n * sqrt(2)
#: = 2 * sigma_n. The quantity wanted is the error of a FULL-size sample against
#: the truth, which is sigma_n — hence divide by 2.
#:
#: This matters more than it looks. Using sqrt(2) instead put the mean noise
#: floor at 0.0375 against the checklist's measured 0.0273, a ratio of 1.37 —
#: almost exactly the sqrt(2) the wrong constant introduces. The two constants
#: move SNR by 40% and the item count with it.
SPLIT_HALF_CORRECTION = 2.0


@dataclass
class ItemNoise:
    item_id: str
    signal_w1: float          # cluster-weighted W1 from the national marginal (B0a)
    noise_w1: float           # split-half estimate of sampling noise in the truth
    snr: float
    n_cells: int              # cells contributing
    n_cells_dropped: int      # cells too thin to split
    median_n_answered: float
    scale_length: int

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class NoiseReport:
    items: dict[str, ItemNoise]
    min_snr: float
    n_repeats: int
    seed: int
    notes: dict[str, Any] = field(default_factory=dict)

    def passing(self) -> list[str]:
        return sorted(i for i, v in self.items.items() if v.snr >= self.min_snr)

    def failing(self) -> list[str]:
        return sorted(i for i, v in self.items.items() if v.snr < self.min_snr)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame([v.to_dict() for v in self.items.values()])
        return df.sort_values("snr", ascending=False).reset_index(drop=True)

    def to_parquet(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_parquet(p, index=False)
        return p

    def summary(self) -> str:
        df = self.to_frame()
        lines = [
            f"Layer 1 — split-half noise floor, {self.n_repeats} repeats, seed {self.seed}",
            f"  items measured : {len(df)}",
            f"  clearing SNR >= {self.min_snr}: {len(self.passing())}",
            f"  mean noise W1  : {df.noise_w1.mean():.4f}",
            f"  median signal  : {df.signal_w1.median():.4f}",
        ]
        return "\n".join(lines)


def _national_marginal(frame: pd.DataFrame, item: str, codes: list[int],
                       weights: np.ndarray) -> np.ndarray:
    col = frame[f"item_{item}"].to_numpy()
    ok = np.isin(col, codes)
    return weighted_histogram(col[ok], weights[ok], codes)


def measure_noise_floor(
    frame: pd.DataFrame,
    cluster_of: pd.Series,
    items: list[str],
    codes: dict[str, list[int]],
    *,
    min_snr: float = 1.5,
    n_repeats: int = 40,
    min_cell_answered: int = 30,
    seed: int = 17,
    correction: float = SPLIT_HALF_CORRECTION,
    weight_col: str = "weight",
) -> NoiseReport:
    rng = np.random.default_rng(seed)
    weights = pd.to_numeric(frame[weight_col], errors="coerce").fillna(0.0).to_numpy()
    # 37 of 18,772 respondents fall outside every leaf (a demographic combination
    # the tree never reached). They are excluded rather than pooled into a
    # catch-all cluster, which would be a cluster with no definition to render
    # into a stat card.
    clusters = cluster_of.astype("object").where(cluster_of.notna(), None).to_numpy()
    order = {
        c: np.flatnonzero(clusters == c)
        for c in pd.unique(pd.Series(clusters).dropna())
    }

    out: dict[str, ItemNoise] = {}
    for item in items:
        cs = codes[item]
        col = frame[f"item_{item}"].to_numpy()
        nat = _national_marginal(frame, item, cs, weights)
        if nat.sum() <= 0:
            continue

        sig_num = sig_den = 0.0
        noise_num = noise_den = 0.0
        n_cells = n_dropped = 0
        answered_ns: list[int] = []

        for pos in order.values():
            sub = pos[np.isin(col[pos], cs)]
            n = sub.size
            if n == 0:
                continue
            w = weights[sub]
            v = col[sub]
            truth = weighted_histogram(v, w, cs)
            wt = float(w.sum())

            # Signal: how far this cluster sits from the national marginal.
            sig_num += wt * w1(truth, nat)
            sig_den += wt

            if n < min_cell_answered:
                n_dropped += 1
                continue
            answered_ns.append(n)
            n_cells += 1

            # Noise: two independent halves of the same cell.
            acc = 0.0
            for _ in range(n_repeats):
                perm = rng.permutation(n)
                a, b = perm[: n // 2], perm[n // 2:]
                ha = weighted_histogram(v[a], w[a], cs)
                hb = weighted_histogram(v[b], w[b], cs)
                if ha.sum() <= 0 or hb.sum() <= 0:
                    continue
                acc += w1(ha, hb)
            noise_num += wt * (acc / n_repeats) / correction
            noise_den += wt

        if sig_den <= 0 or noise_den <= 0 or n_cells == 0:
            continue
        signal = sig_num / sig_den
        noise = noise_num / noise_den
        out[item] = ItemNoise(
            item_id=item,
            signal_w1=float(signal),
            noise_w1=float(noise),
            snr=float(signal / noise) if noise > 0 else float("inf"),
            n_cells=n_cells,
            n_cells_dropped=n_dropped,
            median_n_answered=float(np.median(answered_ns)) if answered_ns else 0.0,
            scale_length=len(cs),
        )

    return NoiseReport(
        items=out, min_snr=min_snr, n_repeats=n_repeats, seed=seed,
        notes={
            "min_cell_answered": min_cell_answered,
            "n_clusters": len(order),
            "split_half_correction": correction,
            "signal_definition": "cluster-weighted W1 from the national marginal (= baseline B0a)",
        },
    )
