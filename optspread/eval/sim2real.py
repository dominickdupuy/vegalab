"""Sim-to-real gap diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def feature_coverage(real: NDArray[np.float64], synthetic: NDArray[np.float64]) -> dict[str, float]:
    """Share of real feature values inside synthetic min/max ranges per column."""
    if real.ndim != 2 or synthetic.ndim != 2 or real.shape[1] != synthetic.shape[1]:
        raise ValueError("real and synthetic must be rank-2 with same feature count")
    lows = synthetic.min(axis=0)
    highs = synthetic.max(axis=0)
    inside = (real >= lows) & (real <= highs)
    return {f"feature_{i}": float(inside[:, i].mean()) for i in range(real.shape[1])}
