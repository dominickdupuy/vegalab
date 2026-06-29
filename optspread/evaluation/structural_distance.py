"""Structural-distance diagnostics for held-out generators."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def wasserstein_1d(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Simple equal-weight 1D Wasserstein distance."""
    if a.size == 0 or b.size == 0:
        return 0.0
    n = min(a.size, b.size)
    qa = np.quantile(a, np.linspace(0.0, 1.0, n))
    qb = np.quantile(b, np.linspace(0.0, 1.0, n))
    return float(np.mean(np.abs(qa - qb)))
