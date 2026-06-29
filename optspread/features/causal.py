"""Small helpers enforcing trailing-window, point-in-time feature calculations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def trailing_window(values: Sequence[float], window: int) -> NDArray[np.float64]:
    """Return only values available at or before the current step."""
    if window <= 0:
        raise ValueError("window must be positive")
    return np.asarray(list(values)[-window:], dtype=np.float64)


def realized_vol(log_returns: Sequence[float], *, window: int, trading_days_per_year: int) -> float:
    """Annualized realized vol from trailing log returns only."""
    vals = trailing_window(log_returns, window)
    if vals.size < 2:
        return 0.0
    return float(vals.std(ddof=1) * np.sqrt(trading_days_per_year))
