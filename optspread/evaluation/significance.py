"""Bootstrap confidence intervals for metric differences."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bootstrap_mean_difference_ci(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    *,
    n_boot: int = 2_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    if a.shape != b.shape:
        raise ValueError("paired bootstrap requires equal shapes")
    if a.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    return (
        float(np.percentile(diffs, 100 * alpha / 2)),
        float(np.percentile(diffs, 100 * (1.0 - alpha / 2))),
    )
