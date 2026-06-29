"""Held-out-generator generalization decisions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GeneralizationResult:
    passed: bool
    degradation: float
    reason: str


def graceful_degradation(
    train_metric: float,
    heldout_metric: float,
    *,
    max_relative_drop: float = 0.5,
) -> GeneralizationResult:
    """Pass if held-out metric does not collapse relative to in-family metric."""
    denom = max(abs(train_metric), 1e-12)
    drop = (train_metric - heldout_metric) / denom
    passed = drop <= max_relative_drop
    return GeneralizationResult(
        passed=passed,
        degradation=float(drop),
        reason=f"relative drop {drop:.3f} <= {max_relative_drop:.3f}"
        if passed
        else f"relative drop {drop:.3f} > {max_relative_drop:.3f}",
    )
