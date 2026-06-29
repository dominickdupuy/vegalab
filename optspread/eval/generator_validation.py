"""Generator-validation gates that run before agent training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from optspread.market.generator import PriceGenerator


@dataclass(frozen=True, slots=True)
class GeneratorValidationResult:
    passed: bool
    statistic: float
    threshold: float
    reason: str


def validate_wave1_vrp(
    make_generator: Callable[[], PriceGenerator],
    *,
    episodes: int = 16,
    threshold: float = 0.0005,
    seed: int = 40_000,
) -> GeneratorValidationResult:
    """GV_1: mean IV^2 - realized^2 is positive."""
    vrps: list[float] = []
    for i in range(episodes):
        gen = make_generator()
        snapshot = gen.reset(np.random.default_rng(seed + i))
        while not gen.done:
            vrps.append(float(snapshot.regime_features["vrp"]))
            snapshot = gen.step()
        vrps.append(float(snapshot.regime_features["vrp"]))
    mean_vrp = float(np.mean(vrps)) if vrps else 0.0
    passed = mean_vrp > threshold
    return GeneratorValidationResult(
        passed=passed,
        statistic=mean_vrp,
        threshold=threshold,
        reason=(
            f"mean VRP {mean_vrp:.6f} > {threshold:.6f}"
            if passed
            else f"mean VRP {mean_vrp:.6f} <= {threshold:.6f}"
        ),
    )
