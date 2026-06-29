"""Solve for the strike that produces a target option delta.

Used to place short legs at a delta bucket (0.10 / 0.16 / 0.25 / 0.40). Pure and
deterministic: a Brent root-find on ``|delta(K)| - target`` over a bracket in K.
"""

from __future__ import annotations

from scipy.optimize import brentq

from optspread.pricing.black_scholes import bs_delta


def strike_from_delta(
    right: str,
    target_delta: float,
    S: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    *,
    bracket_lo: float = 1e-4,
    bracket_hi: float = 10.0,
    xtol: float = 1e-8,
) -> float:
    """Return the strike whose |delta| equals ``target_delta``.

    ``target_delta`` is given as a positive magnitude in (0, 1); the sign is
    implied by ``right`` ("C" -> positive delta, "P" -> negative delta). The
    bracket is expressed as a multiple of spot, so it scales with any underlying.
    """
    if not 0.0 < target_delta < 1.0:
        raise ValueError(f"target_delta must be in (0,1), got {target_delta}")
    if right not in ("C", "P"):
        raise ValueError(f"right must be 'C' or 'P', got {right!r}")
    if T <= 0.0 or sigma <= 0.0:
        raise ValueError("strike_from_delta requires T > 0 and sigma > 0")

    def objective(K: float) -> float:
        d = float(bs_delta(right, S, K, r, q, sigma, T))
        return abs(d) - target_delta

    lo = bracket_lo * S
    hi = bracket_hi * S
    # |delta| is monotone in K: deep-ITM -> 1, deep-OTM -> 0. For a call,
    # |delta| decreases in K; for a put, |delta| increases in K. Either way the
    # objective changes sign across [lo, hi], so brentq is well-posed.
    f_lo = objective(lo)
    f_hi = objective(hi)
    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"target delta {target_delta} not bracketed in "
            f"[{lo:.4f}, {hi:.4f}] (f_lo={f_lo:.4f}, f_hi={f_hi:.4f})"
        )
    root: float = brentq(objective, lo, hi, xtol=xtol)
    return root
