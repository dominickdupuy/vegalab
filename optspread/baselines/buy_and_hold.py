"""Buy-and-hold SPX baseline."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def buy_and_hold_returns(spots: NDArray[np.float64]) -> NDArray[np.float64]:
    if spots.size < 2:
        return np.asarray([], dtype=np.float64)
    return np.diff(spots) / spots[:-1]
