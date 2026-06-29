"""Rollout logging for distillation datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class RolloutDataset:
    observations: NDArray[np.float32]
    actions: NDArray[np.int64]
    values: NDArray[np.float64]


def append_rollouts(parts: list[RolloutDataset]) -> RolloutDataset:
    if not parts:
        return RolloutDataset(
            np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.int64), np.empty(0)
        )
    return RolloutDataset(
        observations=np.concatenate([p.observations for p in parts], axis=0),
        actions=np.concatenate([p.actions for p in parts], axis=0),
        values=np.concatenate([p.values for p in parts], axis=0),
    )
