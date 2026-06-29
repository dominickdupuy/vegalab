"""Probabilistic and deflated Sharpe approximations."""

from __future__ import annotations

from math import erf, sqrt


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_returns: int,
) -> float:
    """Approximate probability that observed Sharpe exceeds a benchmark."""
    if n_returns <= 1:
        return 0.5
    z = (observed_sharpe - benchmark_sharpe) * sqrt(n_returns - 1)
    return normal_cdf(z)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_returns: int,
    n_trials: int,
    benchmark_sharpe: float = 0.0,
) -> float:
    """Conservative Sharpe significance after multiple trials."""
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    trial_penalty = sqrt(2.0) * 0.1 * sqrt(max(n_trials - 1, 0))
    return probabilistic_sharpe_ratio(
        observed_sharpe,
        benchmark_sharpe + trial_penalty,
        n_returns,
    )
