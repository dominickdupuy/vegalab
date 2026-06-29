"""Permutation-importance fallback used alongside SHAP-style reporting."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def permutation_importance(
    score_fn: Callable[[NDArray[np.float64]], float],
    x: NDArray[np.float64],
    *,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Feature importance as score drop after permuting each column."""
    rng = np.random.default_rng(seed)
    base = score_fn(x)
    scores = np.zeros(x.shape[1], dtype=np.float64)
    for j in range(x.shape[1]):
        shuffled = x.copy()
        shuffled[:, j] = rng.permutation(shuffled[:, j])
        scores[j] = base - score_fn(shuffled)
    return scores
