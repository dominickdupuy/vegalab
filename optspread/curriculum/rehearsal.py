"""Anti-forgetting helpers for curriculum training."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mix_wave_ids(
    current_wave: int,
    earlier_waves: list[int],
    *,
    rehearsal_fraction: float,
    n_episodes: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Sample a curriculum episode schedule with earlier-wave rehearsal."""
    if not 0.0 <= rehearsal_fraction <= 1.0:
        raise ValueError("rehearsal_fraction must be in [0,1]")
    schedule = np.full(n_episodes, current_wave, dtype=np.int64)
    if not earlier_waves or rehearsal_fraction == 0.0:
        return schedule
    mask = rng.random(n_episodes) < rehearsal_fraction
    schedule[mask] = rng.choice(np.asarray(earlier_waves, dtype=np.int64), size=int(mask.sum()))
    return schedule
