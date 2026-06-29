"""Behavioral statistics for per-wave policy validation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from optspread.actions.library import ACTION_LIBRARY
from optspread.actions.margin_class import MarginClass


def credit_structure_indicator(actions: NDArray[np.int64]) -> NDArray[np.float64]:
    """1 for short-premium structures, 0 otherwise."""
    credit_names = {
        "BullPutSpreadTemplate",
        "BearCallSpreadTemplate",
        "IronCondorTemplate",
        "IronButterflyTemplate",
        "ShortStrangleTemplate",
        "ShortStraddleTemplate",
    }
    return np.asarray(
        [
            1.0 if ACTION_LIBRARY[int(action)].template.__class__.__name__ in credit_names else 0.0
            for action in actions
        ],
        dtype=np.float64,
    )


def defined_risk_indicator(actions: NDArray[np.int64]) -> NDArray[np.float64]:
    """1 for structurally defined-risk actions, 0 otherwise."""
    return np.asarray(
        [
            1.0
            if ACTION_LIBRARY[int(action)].template.margin_class() is MarginClass.DEFINED_RISK
            else 0.0
            for action in actions
        ],
        dtype=np.float64,
    )


def safe_correlation(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Pearson correlation with zero-variance guard."""
    if x.size != y.size or x.size < 2:
        return 0.0
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
