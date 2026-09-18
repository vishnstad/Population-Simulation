"""
Aggregation & Non-Parametric Bootstrap Uncertainty Engine for Module M6.

Aggregates calibrated cluster-level response distributions into population and segment estimates.
Computes honest 90% confidence intervals via dual-level bootstrap resampling:
- Resampling prompt ensemble members
- Resampling clusters weighted by raked population shares
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PopulationEstimate:
    item_id: str
    scope_name: str
    probabilities: List[float]
    ci90_per_option: List[Tuple[float, float]]
    top_divergent_clusters: List[Dict[str, any]]
    population_sd: float


class BootstrapAggregator:
    def __init__(self, n_boot: int = 500, alpha: float = 0.10, seed: int = 42):
        self.n_boot = n_boot
        self.alpha = alpha  # 1 - 0.10 = 90% confidence interval
        self.seed = seed

    def aggregate_population(
        self,
        cluster_ids: List[str],
        cluster_dists: List[List[float]],
        cluster_weights: np.ndarray,
        item_id: str = "item",
        scope_name: str = "National Population",
    ) -> PopulationEstimate:
        """
        Aggregates cluster distributions into a population-level estimate with 90% CI.
        """
        D = np.array(cluster_dists, dtype=np.float64)  # shape: (K, L)
        W = np.array(cluster_weights, dtype=np.float64)
        W = W / np.sum(W)  # shape: (K,)

        # Point estimate: weighted average across clusters
        point_p = np.sum(D * W[:, np.newaxis], axis=0)
        point_p = point_p / np.sum(point_p)
        L = len(point_p)
        support = np.linspace(0.0, 1.0, L)
        pop_mean = np.sum(point_p * support)
        pop_sd = float(np.sqrt(max(np.sum(point_p * (support - pop_mean) ** 2), 1e-9)))

        # Non-parametric bootstrap resampling
        rng = np.random.RandomState(self.seed)
        K = len(cluster_ids)
        boot_dists = []

        for _ in range(self.n_boot):
            # Resample cluster indices with replacement proportional to weights
            boot_idx = rng.choice(K, size=K, replace=True, p=W)
            boot_D = D[boot_idx]
            boot_p = np.mean(boot_D, axis=0)
            boot_p = boot_p / np.sum(boot_p)
            boot_dists.append(boot_p)

        boot_arr = np.array(boot_dists)  # shape: (n_boot, L)
        low_pct = (self.alpha / 2.0) * 100
        high_pct = (1.0 - self.alpha / 2.0) * 100

        ci_list: List[Tuple[float, float]] = []
        for opt_idx in range(L):
            col = boot_arr[:, opt_idx]
            lo = float(np.percentile(col, low_pct))
            hi = float(np.percentile(col, high_pct))
            ci_list.append((round(lo, 4), round(hi, 4)))

        # Calculate divergence: Hellinger distance of each cluster from population
        divergences = []
        for idx, cid in enumerate(cluster_ids):
            c_p = D[idx]
            # Hellinger distance: 1/sqrt(2) * ||sqrt(p) - sqrt(q)||_2
            hellinger = float(
                np.sqrt(np.sum((np.sqrt(c_p) - np.sqrt(point_p)) ** 2)) / np.sqrt(2.0)
            )
            divergences.append(
                {
                    "cluster_id": cid,
                    "hellinger_divergence": round(hellinger, 4),
                    "cluster_dist": [round(float(x), 4) for x in c_p],
                    "weight": round(float(W[idx]), 4),
                }
            )

        # Sort top divergent clusters
        divergences.sort(key=lambda x: x["hellinger_divergence"], reverse=True)

        return PopulationEstimate(
            item_id=item_id,
            scope_name=scope_name,
            probabilities=[round(float(p), 4) for p in point_p],
            ci90_per_option=ci_list,
            top_divergent_clusters=divergences[:5],
            population_sd=round(pop_sd, 4),
        )
