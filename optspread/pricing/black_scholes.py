"""Black-Scholes-Merton pricing and greeks.

Pure, vectorized, side-effect-free functions. Every input may be a scalar or a
NumPy array; broadcasting follows NumPy rules. These are the oracle-grade
numerics the rest of the system (and the tests) lean on, so they must handle the
degenerate ``T -> 0`` and zero-vol limits without NaN/inf.

Conventions
-----------
- ``right`` is ``"C"`` (call) or ``"P"`` (put).
- ``r`` continuous risk-free rate, ``q`` continuous dividend/borrow yield.
- ``sigma`` annualized volatility, ``T`` time to expiry in years.
- Prices/greeks are per-share (multiply by the contract multiplier elsewhere).
- ``theta`` is per-year (divide by 365 for per-calendar-day).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

_FloatArr = NDArray[np.float64]


def _d1_d2(
    S: _FloatArr, K: _FloatArr, r: _FloatArr, q: _FloatArr, sigma: _FloatArr, T: _FloatArr
) -> tuple[_FloatArr, _FloatArr]:
    """Compute d1, d2 with a safe denominator (sigma*sqrt(T))."""
    vol = sigma * np.sqrt(T)
    # Guard the degenerate denominator; the callers special-case T<=0 / vol<=0.
    safe_vol = np.where(vol > 0.0, vol, 1.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / safe_vol
    d2 = d1 - vol
    return d1, d2


def _intrinsic(right: str, S: _FloatArr, K: _FloatArr) -> _FloatArr:
    if right == "C":
        return np.maximum(S - K, 0.0)
    if right == "P":
        return np.maximum(K - S, 0.0)
    raise ValueError(f"right must be 'C' or 'P', got {right!r}")


def _as_arrays(*xs: ArrayLike) -> tuple[_FloatArr, ...]:
    return tuple(np.asarray(x, dtype=np.float64) for x in xs)


def bs_price(
    right: str,
    S: ArrayLike,
    K: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    T: ArrayLike,
) -> _FloatArr:
    """Black-Scholes-Merton option price.

    At ``T <= 0`` (or ``sigma <= 0``) the value collapses to discounted
    intrinsic, returned without NaN.
    """
    S, K, r, q, sigma, T = _as_arrays(S, K, r, q, sigma, T)
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if right == "C":
        live = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    elif right == "P":
        live = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    else:
        raise ValueError(f"right must be 'C' or 'P', got {right!r}")
    degenerate = (T <= 0.0) | (sigma <= 0.0)
    out = np.where(degenerate, _intrinsic(right, S, K), live)
    return np.asarray(out, dtype=np.float64)


def bs_delta(
    right: str,
    S: ArrayLike,
    K: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    T: ArrayLike,
) -> _FloatArr:
    """Spot delta dPrice/dS. Call in [0,1], put in [-1,0]."""
    S, K, r, q, sigma, T = _as_arrays(S, K, r, q, sigma, T)
    d1, _ = _d1_d2(S, K, r, q, sigma, T)
    disc_q = np.exp(-q * T)
    if right == "C":
        live = disc_q * norm.cdf(d1)
        deg = np.where(S > K, 1.0, np.where(S < K, 0.0, 0.5))
    elif right == "P":
        live = -disc_q * norm.cdf(-d1)
        deg = np.where(S < K, -1.0, np.where(S > K, 0.0, -0.5))
    else:
        raise ValueError(f"right must be 'C' or 'P', got {right!r}")
    degenerate = (T <= 0.0) | (sigma <= 0.0)
    return np.asarray(np.where(degenerate, deg, live), dtype=np.float64)


def bs_gamma(
    S: ArrayLike,
    K: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    T: ArrayLike,
) -> _FloatArr:
    """Gamma d2Price/dS2 (identical for calls and puts)."""
    S, K, r, q, sigma, T = _as_arrays(S, K, r, q, sigma, T)
    d1, _ = _d1_d2(S, K, r, q, sigma, T)
    disc_q = np.exp(-q * T)
    vol = sigma * np.sqrt(T)
    safe = (T > 0.0) & (sigma > 0.0)
    live = disc_q * norm.pdf(d1) / np.where(safe, S * vol, 1.0)
    return np.asarray(np.where(safe, live, 0.0), dtype=np.float64)


def bs_vega(
    S: ArrayLike,
    K: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    T: ArrayLike,
) -> _FloatArr:
    """Vega dPrice/dsigma per unit vol (not per 1%)."""
    S, K, r, q, sigma, T = _as_arrays(S, K, r, q, sigma, T)
    d1, _ = _d1_d2(S, K, r, q, sigma, T)
    disc_q = np.exp(-q * T)
    safe = (T > 0.0) & (sigma > 0.0)
    live = S * disc_q * norm.pdf(d1) * np.sqrt(T)
    return np.asarray(np.where(safe, live, 0.0), dtype=np.float64)


def bs_theta(
    right: str,
    S: ArrayLike,
    K: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    T: ArrayLike,
) -> _FloatArr:
    """Theta dPrice/dT_calendar (per-year). Negative for most long options."""
    S, K, r, q, sigma, T = _as_arrays(S, K, r, q, sigma, T)
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    vol = sigma * np.sqrt(T)
    safe = (T > 0.0) & (sigma > 0.0)
    term1 = -S * disc_q * norm.pdf(d1) * sigma / (2.0 * np.where(safe, np.sqrt(T), 1.0))
    if right == "C":
        live = term1 - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
    elif right == "P":
        live = term1 + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
    else:
        raise ValueError(f"right must be 'C' or 'P', got {right!r}")
    del vol  # documented intermediate; theta uses term1 directly
    return np.asarray(np.where(safe, live, 0.0), dtype=np.float64)
