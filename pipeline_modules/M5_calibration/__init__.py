"""M5 -- cross-fitted anchor calibration: the project's core mechanism."""

from .crossfit import (
    CalibrationPair,
    CrossFittedCalibrator,
    load_crossfit_plan,
    make_crossfit_plan,
    save_crossfit_plan,
)
from .isotonic import OrdinalCDFCalibrator
from .variance import HierarchicalVarianceRestorer, compute_histogram_sd, temper_histogram

__all__ = [
    "CalibrationPair",
    "CrossFittedCalibrator",
    "make_crossfit_plan",
    "save_crossfit_plan",
    "load_crossfit_plan",
    "OrdinalCDFCalibrator",
    "HierarchicalVarianceRestorer",
    "compute_histogram_sd",
    "temper_histogram",
]
