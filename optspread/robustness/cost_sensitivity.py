"""Cost-sensitivity and break-even helpers."""

from __future__ import annotations


def break_even_cost_multiple(multiples: list[float], metrics: list[float]) -> float | None:
    """First cost multiple where metric is non-positive."""
    if len(multiples) != len(metrics):
        raise ValueError("multiples and metrics lengths must match")
    for multiple, metric in zip(multiples, metrics, strict=True):
        if metric <= 0.0:
            return multiple
    return None
