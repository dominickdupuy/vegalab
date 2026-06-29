"""ChainSnapshot: the option chain visible at one date.

Frozen value object. Holds the strike grid, the implied-vol surface (one IV per
(strike, expiry)), the available expiries (in years to expiry), and the
spot/rates. In Wave-0 the IV surface is flat at the GBM sigma (fair IV), but the
*schema* already supports a full per-(strike,expiry) surface so richer
generators slot in without touching the env.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optspread.instruments.leg import OptionLeg, Right
from optspread.pricing.black_scholes import bs_price

_F = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ChainSnapshot:
    """Immutable chain at one timestamp.

    Parameters
    ----------
    strikes : sorted 1-D array of available strikes.
    ivs : 2-D array shape (n_expiries, n_strikes) of implied vols.
    expiries : 1-D array of times-to-expiry in years, ascending (the tenors
        available to OPEN at this date).
    spot, r, q : scalars at this date.
    t : integer day index (for diagnostics / no-look-ahead assertions).
    trading_days_per_year : conversion used to decay a held leg's absolute
        expiry day into a time-to-expiry in years.
    """

    strikes: _F
    ivs: _F
    expiries: _F
    spot: float
    r: float
    q: float
    t: int
    trading_days_per_year: int = 252

    def __post_init__(self) -> None:
        if self.ivs.shape != (self.expiries.shape[0], self.strikes.shape[0]):
            raise ValueError(
                f"ivs shape {self.ivs.shape} != (n_expiries, n_strikes)="
                f"({self.expiries.shape[0]}, {self.strikes.shape[0]})"
            )

    @property
    def n_expiries(self) -> int:
        return int(self.expiries.shape[0])

    def nearest_strike_index(self, target: float) -> int:
        """Index of the grid strike closest to ``target``."""
        return int(np.argmin(np.abs(self.strikes - target)))

    def nearest_strike(self, target: float) -> float:
        return float(self.strikes[self.nearest_strike_index(target)])

    def iv_at(self, expiry_idx: int, strike: float) -> float:
        """Implied vol at the grid strike nearest ``strike`` for an expiry."""
        si = self.nearest_strike_index(strike)
        return float(self.ivs[expiry_idx, si])

    def price(self, right: Right, strike: float, expiry_idx: int) -> float:
        """Per-share BS price of a single option on this chain."""
        sigma = self.iv_at(expiry_idx, strike)
        T = float(self.expiries[expiry_idx])
        return float(bs_price(right, self.spot, strike, self.r, self.q, sigma, T))

    def expiry_day_for(self, expiry_idx: int) -> int:
        """Absolute day index on which the openable ``expiry_idx`` tenor expires."""
        tenor_days = round(float(self.expiries[expiry_idx]) * self.trading_days_per_year)
        return self.t + int(tenor_days)

    def time_to_expiry(self, leg: OptionLeg) -> float:
        """Years to expiry for ``leg`` at THIS date (decays as the path advances).

        A leg with an absolute ``expiry_day`` ages: ``T = (expiry_day - t)/tdy``,
        clamped at 0 (a leg held to/through expiry settles at intrinsic). A leg
        with ``expiry_day == -1`` keeps the chain's fixed tenor for its index.
        """
        if leg.expiry_day < 0:
            return float(self.expiries[leg.expiry_idx])
        return max(leg.expiry_day - self.t, 0) / self.trading_days_per_year

    def price_leg(self, leg: OptionLeg) -> float:
        """Per-share price of the option underlying ``leg`` (sign-agnostic).

        Uses the leg's decaying time-to-expiry so held positions earn theta. The
        fair-IV surface is flat, so the IV is read at the leg's strike.
        """
        T = self.time_to_expiry(leg)
        si = self.nearest_strike_index(leg.strike)
        sigma = float(self.ivs[min(leg.expiry_idx, self.n_expiries - 1), si])
        return float(bs_price(leg.right, self.spot, leg.strike, self.r, self.q, sigma, T))
