"""Minimal VIPER-style weighted decision stump/tree utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class DecisionStump:
    feature: int
    threshold: float
    left_action: int
    right_action: int

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.int64]:
        return np.where(
            x[:, self.feature] <= self.threshold, self.left_action, self.right_action
        ).astype(np.int64)


def fit_weighted_stump(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    weights: NDArray[np.float64] | None = None,
) -> DecisionStump:
    """Fit a shallow interpretable policy surrogate."""
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("invalid stump inputs")
    w = np.ones_like(y, dtype=np.float64) if weights is None else weights.astype(np.float64)
    best: tuple[float, DecisionStump] | None = None
    for feature in range(x.shape[1]):
        for threshold in np.unique(x[:, feature]):
            left = x[:, feature] <= threshold
            if not np.any(left) or np.all(left):
                continue
            left_action = _weighted_mode(y[left], w[left])
            right_action = _weighted_mode(y[~left], w[~left])
            pred = np.where(left, left_action, right_action)
            err = float(w[pred != y].sum())
            stump = DecisionStump(feature, float(threshold), left_action, right_action)
            if best is None or err < best[0]:
                best = (err, stump)
    if best is None:
        action = _weighted_mode(y, w)
        return DecisionStump(0, float("inf"), action, action)
    return best[1]


def _weighted_mode(y: NDArray[np.int64], w: NDArray[np.float64]) -> int:
    scores: dict[int, float] = {}
    for action, weight in zip(y, w, strict=True):
        scores[int(action)] = scores.get(int(action), 0.0) + float(weight)
    return max(scores.items(), key=lambda item: item[1])[0]
