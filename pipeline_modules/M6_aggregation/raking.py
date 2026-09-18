"""
Census Raking Engine (Iterative Proportional Fitting) for Module M6.

Calibrates cluster population weights against external Census marginal distributions (e.g. ACS PUMS for US, Census 2011 for India)
to eliminate survey non-response sampling bias.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class CensusRaker:
    def __init__(self, max_iter: int = 100, conv_tol: float = 1e-5):
        self.max_iter = max_iter
        self.conv_tol = conv_tol

    def rake_cluster_weights(
        self,
        cluster_definitions: List[Dict[str, any]],
        initial_weights: np.ndarray,
        marginal_targets: Dict[str, Dict[str, float]],
    ) -> np.ndarray:
        """
        cluster_definitions: List of cluster metadata dicts with demographic keys (e.g. 'region', 'sex')
        initial_weights: Initial weights w_c from survey microdata
        marginal_targets: Dict mapping demographic dimension -> {category_value: target_population_share}
        """
        weights = np.array(initial_weights, dtype=np.float64)
        weights = weights / np.sum(weights)  # Normalize to sum 1.0

        for it in range(self.max_iter):
            max_delta = 0.0

            for dim_name, target_dist in marginal_targets.items():
                # Compute current marginal sum for each category
                category_sums: Dict[str, float] = {}
                for idx, c_def in enumerate(cluster_definitions):
                    val = c_def.get(dim_name)
                    if isinstance(val, list):
                        # Handle multi-level coarsened categories by distributing weight
                        for v in val:
                            category_sums[v] = category_sums.get(v, 0.0) + weights[idx] / len(val)
                    elif val is not None:
                        val_str = str(val)
                        category_sums[val_str] = category_sums.get(val_str, 0.0) + weights[idx]

                # Update weights for each cluster
                for idx, c_def in enumerate(cluster_definitions):
                    val = c_def.get(dim_name)
                    if val is None:
                        continue
                    if isinstance(val, list):
                        # Average adjustment ratio across levels
                        ratios = [
                            target_dist.get(v, category_sums.get(v, 1.0)) / max(category_sums.get(v, 1e-8), 1e-8)
                            for v in val if v in target_dist
                        ]
                        factor = float(np.mean(ratios)) if ratios else 1.0
                    else:
                        val_str = str(val)
                        target_val = target_dist.get(val_str)
                        current_val = category_sums.get(val_str, 1e-8)
                        factor = (target_val / current_val) if target_val is not None else 1.0

                    old_w = weights[idx]
                    weights[idx] *= factor
                    max_delta = max(max_delta, abs(weights[idx] - old_w))

                # Re-normalize after each dimension update
                weights = weights / np.sum(weights)

            if max_delta < self.conv_tol:
                logger.info(f"Raking converged after {it+1} iterations (max_delta={max_delta:.6e}).")
                break

        return weights
