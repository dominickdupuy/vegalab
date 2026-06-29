"""Broad regime-feature coverage sampler."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def latin_hypercube(n: int, dim: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Unit-cube Latin hypercube sample for broad map coverage."""
    result = np.zeros((n, dim), dtype=np.float64)
    for j in range(dim):
        perm = rng.permutation(n)
        result[:, j] = (perm + rng.random(n)) / n
    return result
