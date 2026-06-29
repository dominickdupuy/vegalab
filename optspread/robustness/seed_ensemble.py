"""Aggregate headline metrics across seeds/folds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class EnsembleSummary:
    mean: float
    std: float
    n: int


def summarize(values: NDArray[np.float64]) -> EnsembleSummary:
    if values.size == 0:
        return EnsembleSummary(0.0, 0.0, 0)
    return EnsembleSummary(
        float(values.mean()),
        float(values.std(ddof=1)) if values.size > 1 else 0.0,
        int(values.size),
    )
