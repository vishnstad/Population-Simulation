"""
Isotonic Recalibration Engine for Module M5.

Performs monotonic non-decreasing probability recalibration on Cumulative Distribution Functions (CDFs)
for ordinal response scales (lengths L in 2..7), mapping raw LLM cumulative probabilities to true empirical CDFs.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from typing import Dict, List, Optional, Tuple


class OrdinalCDFCalibrator:
    """
    Fits and applies isotonic regression on cumulative probabilities per scale length L.
    """

    def __init__(self):
        # Dictionary mapping scale_length L -> IsotonicRegression model
        self.models: Dict[int, IsotonicRegression] = {}

    def fit(self, scale_length: int, raw_cdfs: np.ndarray, true_cdfs: np.ndarray):
        """
        raw_cdfs: shape (N, L-1) - cumulative probabilities for cuts 1..L-1
        true_cdfs: shape (N, L-1) - true empirical cumulative probabilities
        """
        x = raw_cdfs.flatten()
        y = true_cdfs.flatten()

        # Enforce boundary anchors: 0 -> 0, 1 -> 1
        x = np.concatenate([[0.0], x, [1.0]])
        y = np.concatenate([[0.0], y, [1.0]])

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        iso.fit(x, y)
        self.models[scale_length] = iso

    def transform_histogram(self, raw_probs: List[float]) -> List[float]:
        """
        Transforms a raw probability histogram into an isotonically recalibrated histogram.
        """
        L = len(raw_probs)
        if L not in self.models:
            return raw_probs  # No recalibration model fit yet for this scale length

        iso = self.models[L]
        # Compute raw CDF
        raw_cdf = np.cumsum(raw_probs)[:-1]  # cuts 1..L-1
        # Apply isotonic transformation
        cal_cdf = iso.predict(raw_cdf)

        # Enforce monotonicity
        cal_cdf = np.maximum.accumulate(cal_cdf)
        cal_cdf = np.clip(cal_cdf, 0.0, 1.0)

        # Recover probability mass from calibrated CDF
        full_cdf = np.concatenate([[0.0], cal_cdf, [1.0]])
        cal_probs = np.diff(full_cdf)

        # Numerical cleanup: non-negative and exact unit sum
        cal_probs = np.maximum(cal_probs, 1e-6)
        cal_probs = cal_probs / np.sum(cal_probs)
        return [round(float(p), 6) for p in cal_probs]
