"""Regime-to-structure map assembly."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class RegimeCell:
    cell_id: int
    modal_action: int
    mean_return: float
    count: int


def build_regime_cells(
    labels: NDArray[np.int64],
    actions: NDArray[np.int64],
    returns: NDArray[np.float64],
) -> list[RegimeCell]:
    cells: list[RegimeCell] = []
    for label in sorted(set(int(x) for x in labels)):
        mask = labels == label
        modal = Counter(int(a) for a in actions[mask]).most_common(1)[0][0]
        cells.append(RegimeCell(label, modal, float(returns[mask].mean()), int(mask.sum())))
    return cells
