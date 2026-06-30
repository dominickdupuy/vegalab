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


def heston_mc_price(
    right: str,
    *,
    spot: float,
    strike: float,
    r: float,
    q: float,
    T: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    v0: float,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> tuple[float, float]:
    """Full-truncation Euler Heston MC price and standard error."""
    if n_paths <= 1:
        raise ValueError("n_paths must be greater than 1")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if strike <= 0.0 or spot <= 0.0:
        raise ValueError("spot and strike must be positive")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    if theta < 0.0 or v0 < 0.0:
        raise ValueError("theta and v0 must be non-negative")
    if sigma_v < 0.0:
        raise ValueError("sigma_v must be non-negative")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be in [-1, 1]")

    normalized_right = right.upper()
    if normalized_right not in {"C", "P"}:
        raise ValueError("right must be 'C' or 'P'")
    if T <= 0.0:
        intrinsic = max(spot - strike, 0.0) if normalized_right == "C" else max(strike - spot, 0.0)
        return intrinsic, 0.0

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    rho_complement = np.sqrt(max(1.0 - rho * rho, 0.0))
    log_spot = np.full(n_paths, np.log(spot), dtype=np.float64)
    variance = np.full(n_paths, v0, dtype=np.float64)

    for _ in range(n_steps):
        z_v = rng.standard_normal(n_paths)
        z_s_independent = rng.standard_normal(n_paths)
        variance_pos = np.maximum(variance, 0.0)
        sqrt_variance = np.sqrt(variance_pos)
        z_s = rho * z_v + rho_complement * z_s_independent
        log_spot += (r - q - 0.5 * variance_pos) * dt + sqrt_variance * sqrt_dt * z_s
        variance += kappa * (theta - variance_pos) * dt + sigma_v * sqrt_variance * sqrt_dt * z_v

    terminal = np.exp(log_spot)
    if normalized_right == "C":
        payoff = np.maximum(terminal - strike, 0.0)
    else:
        payoff = np.maximum(strike - terminal, 0.0)
    discounted_payoff = np.exp(-r * T) * payoff
    price = float(discounted_payoff.mean())
    std_error = float(discounted_payoff.std(ddof=1) / np.sqrt(n_paths))
    return price, std_error
