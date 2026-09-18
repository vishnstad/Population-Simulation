"""
Hierarchical Variance Restoration Engine for Module M5.

Models true cluster-level standard deviation from raw standard deviations using Weighted Least Squares (WLS)
with hierarchical shrinkage (partial pooling) toward parent nodes.
Restores variance by power-tempering probability distributions to hit the target standard deviation.
"""

import numpy as np
from scipy.optimize import root_scalar
from typing import Dict, List, Optional, Tuple


def compute_histogram_sd(probs: np.ndarray, support: Optional[np.ndarray] = None) -> float:
    """
    Computes the standard deviation of an ordinal probability distribution over support {0, 1/(L-1), ..., 1}.
    """
    L = len(probs)
    if support is None:
        support = np.linspace(0.0, 1.0, L)
    mean = np.sum(probs * support)
    var = np.sum(probs * (support - mean) ** 2)
    return float(np.sqrt(max(var, 1e-9)))


def temper_histogram(probs: np.ndarray, gamma: float) -> np.ndarray:
    """
    Power-tempers probability distribution: p_i^gamma / sum(p_j^gamma).
    gamma > 1 sharpens distribution (decreases SD), gamma < 1 flattens distribution (increases SD).
    """
    safe_p = np.maximum(probs, 1e-8)
    unnorm = np.power(safe_p, gamma)
    return unnorm / np.sum(unnorm)


class HierarchicalVarianceRestorer:
    def __init__(self, tau: float = 100.0):
        self.tau = tau  # Shrinkage prior weight
        # Stores (alpha, beta) regression coefficients per cluster_id
        self.cluster_coeffs: Dict[str, Tuple[float, float]] = {}
        self.global_coeffs: Tuple[float, float] = (0.05, 0.90)  # Default prior

    def fit_global(self, raw_sds: np.ndarray, true_sds: np.ndarray, weights: Optional[np.ndarray] = None):
        """Fits global fallback linear model: true_sd = alpha + beta * raw_sd."""
        if len(raw_sds) < 3:
            return
        if weights is None:
            weights = np.ones_like(raw_sds)

        X = np.vstack([np.ones_like(raw_sds), raw_sds]).T
        W = np.diag(weights / np.sum(weights))
        try:
            # WLS solution: (X^T W X)^(-1) X^T W y
            beta_hat = np.linalg.solve(X.T @ W @ X, X.T @ W @ true_sds)
            self.global_coeffs = (float(beta_hat[0]), float(beta_hat[1]))
        except np.linalg.LinAlgError:
            self.global_coeffs = (0.05, 0.90)

    def fit_cluster(
        self,
        cluster_id: str,
        raw_sds: np.ndarray,
        true_sds: np.ndarray,
        n_eff: float,
        parent_coeffs: Optional[Tuple[float, float]] = None,
    ):
        """
        Fits per-cluster variance coefficients and applies hierarchical shrinkage toward parent.
        """
        prior = parent_coeffs if parent_coeffs is not None else self.global_coeffs

        if len(raw_sds) < 3:
            self.cluster_coeffs[cluster_id] = prior
            return

        X = np.vstack([np.ones_like(raw_sds), raw_sds]).T
        try:
            # Ordinary least squares for cluster
            beta_cluster = np.linalg.lstsq(X, true_sds, rcond=None)[0]
            alpha_raw, beta_raw = float(beta_cluster[0]), float(beta_cluster[1])
        except Exception:
            alpha_raw, beta_raw = prior

        # Hierarchical Bayesian shrinkage weight: lambda = n_eff / (n_eff + tau)
        lam = n_eff / (n_eff + self.tau)
        alpha_shrunk = lam * alpha_raw + (1.0 - lam) * prior[0]
        beta_shrunk = lam * beta_raw + (1.0 - lam) * prior[1]

        self.cluster_coeffs[cluster_id] = (alpha_shrunk, beta_shrunk)

    def predict_target_sd(self, cluster_id: str, raw_sd: float) -> float:
        alpha, beta = self.cluster_coeffs.get(cluster_id, self.global_coeffs)
        pred_sd = alpha + beta * raw_sd
        # Bound predicted SD between realistic boundaries [0.05, 0.50]
        return float(np.clip(pred_sd, 0.05, 0.50))

    def restore_variance(self, cluster_id: str, probs: List[float]) -> List[float]:
        """
        Adjusts histogram dispersion via power-tempering to match target predicted standard deviation.
        """
        p_arr = np.array(probs, dtype=np.float64)
        L = len(p_arr)
        support = np.linspace(0.0, 1.0, L)
        current_sd = compute_histogram_sd(p_arr, support)
        target_sd = self.predict_target_sd(cluster_id, current_sd)

        # Objective function: compute_sd(temper(p, gamma)) - target_sd = 0
        def objective(gamma):
            tempered = temper_histogram(p_arr, gamma)
            return compute_histogram_sd(tempered, support) - target_sd

        # Check bounds
        sd_low_gamma = compute_histogram_sd(temper_histogram(p_arr, 0.01), support)
        sd_high_gamma = compute_histogram_sd(temper_histogram(p_arr, 10.0), support)

        if target_sd >= sd_low_gamma:
            final_p = temper_histogram(p_arr, 0.01)
        elif target_sd <= sd_high_gamma:
            final_p = temper_histogram(p_arr, 10.0)
        else:
            try:
                res = root_scalar(objective, bracket=[0.01, 10.0], method="brentq")
                gamma_opt = res.root if res.converged else 1.0
            except Exception:
                gamma_opt = 1.0
            final_p = temper_histogram(p_arr, gamma_opt)

        return [round(float(p), 6) for p in final_p]
