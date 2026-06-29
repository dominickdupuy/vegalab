"""Monte-Carlo pricing oracle used only in tests, never in the hot path."""

from __future__ import annotations

import numpy as np


def black_scholes_mc_price(
    right: str,
    *,
    spot: float,
    strike: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_paths: int,
    seed: int,
) -> float:
    """Monte-Carlo European option price for cross-checking analytic/COS paths."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    terminal = spot * np.exp((r - q - 0.5 * sigma * sigma) * T + sigma * np.sqrt(T) * z)
    if right == "C":
        payoff = np.maximum(terminal - strike, 0.0)
    elif right == "P":
        payoff = np.maximum(strike - terminal, 0.0)
    else:
        raise ValueError("right must be 'C' or 'P'")
    return float(np.exp(-r * T) * payoff.mean())
