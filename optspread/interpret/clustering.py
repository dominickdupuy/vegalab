"""Small deterministic k-means implementation for regime archetypes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def kmeans(
    x: NDArray[np.float64],
    *,
    k: int,
    n_iter: int = 50,
    seed: int = 0,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    if x.ndim != 2 or k <= 0 or k > x.shape[0]:
        raise ValueError("invalid kmeans inputs")
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(x.shape[0], size=k, replace=False)].copy()
    labels = np.zeros(x.shape[0], dtype=np.int64)
    for _ in range(n_iter):
        distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1).astype(np.int64)
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                centers[i] = x[mask].mean(axis=0)
    return labels, centers
