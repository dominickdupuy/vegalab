"""Policy-surrogate fidelity metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def action_agreement(y_true: NDArray[np.int64], y_pred: NDArray[np.int64]) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("action arrays must match")
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def value_regret(
    oracle_values: NDArray[np.float64],
    surrogate_actions: NDArray[np.int64],
) -> float:
    if oracle_values.ndim != 2 or oracle_values.shape[0] != surrogate_actions.size:
        raise ValueError("invalid value-regret inputs")
    oracle_best = oracle_values.max(axis=1)
    chosen = oracle_values[np.arange(surrogate_actions.size), surrogate_actions]
    return float(np.mean(oracle_best - chosen))
