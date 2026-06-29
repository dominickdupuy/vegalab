"""CVaR alpha frontier helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AlphaPoint:
    alpha: float
    mean_return: float
    cvar: float


def tail_improves_as_alpha_decreases(points: list[AlphaPoint]) -> bool:
    """Check a coarse frontier: lower alpha should not have worse CVaR."""
    ordered = sorted(points, key=lambda p: p.alpha)
    return all(ordered[i].cvar >= ordered[i + 1].cvar for i in range(len(ordered) - 1))
