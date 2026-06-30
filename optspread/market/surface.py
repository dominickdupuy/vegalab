"""Standardized implied-volatility surface.

Phase 4's sim-to-real bridge is that synthetic generators and OptionMetrics both
produce the same object: a surface indexed by delta and constant maturity. The
tradeable chain is derived from this object, so later real-data ingestion should
not require environment changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optspread.instruments.chain import ChainSnapshot
from optspread.pricing.black_scholes import bs_delta

_F = NDArray[np.float64]

DEFAULT_DELTA_GRID: _F = np.asarray([0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90])
DEFAULT_MATURITY_GRID_DAYS: _F = np.asarray([10.0, 21.0, 42.0, 63.0, 126.0, 252.0, 504.0])


@dataclass(frozen=True, slots=True)
class IVSurface:
    """Implied volatility on a standardized delta x maturity grid."""

    deltas: _F
    maturity_days: _F
    ivs: _F
    spot: float
    r: float
    q: float
    t: int
    trading_days_per_year: int = 252

    def __post_init__(self) -> None:
        if self.ivs.shape != (self.maturity_days.shape[0], self.deltas.shape[0]):
            raise ValueError(
                f"ivs shape {self.ivs.shape} != "
                f"({self.maturity_days.shape[0]}, {self.deltas.shape[0]})"
            )
        if not np.all(np.diff(self.deltas) > 0):
            raise ValueError("deltas must be strictly increasing")
        if not np.all(np.diff(self.maturity_days) > 0):
            raise ValueError("maturity_days must be strictly increasing")
        if np.any(self.ivs <= 0.0):
            raise ValueError("all implied vols must be positive")

    @classmethod
    def flat(
        cls,
        *,
        sigma: float,
        spot: float,
        r: float,
        q: float,
        t: int,
        trading_days_per_year: int = 252,
        deltas: _F = DEFAULT_DELTA_GRID,
        maturity_days: _F = DEFAULT_MATURITY_GRID_DAYS,
    ) -> IVSurface:
        ivs = np.full((maturity_days.shape[0], deltas.shape[0]), sigma, dtype=np.float64)
        return cls(
            deltas=deltas.astype(np.float64),
            maturity_days=maturity_days.astype(np.float64),
            ivs=ivs,
            spot=spot,
            r=r,
            q=q,
            t=t,
            trading_days_per_year=trading_days_per_year,
        )

    def iv_at_delta_maturity(self, delta: float, maturity_days: float) -> float:
        """Bilinear interpolation in delta and maturity-days space."""
        maturity_slice = np.asarray(
            [
                np.interp(delta, self.deltas, self.ivs[maturity_idx])
                for maturity_idx in range(self.maturity_days.shape[0])
            ],
            dtype=np.float64,
        )
        return float(np.interp(maturity_days, self.maturity_days, maturity_slice))

    def iv_for_strike(self, strike: float, maturity_days: float) -> float:
        """Interpolate IV for a strike by mapping strike to approximate call delta."""
        maturity_years = max(maturity_days / self.trading_days_per_year, 1e-8)
        atm_sigma = self.iv_at_delta_maturity(0.50, maturity_days)
        delta = abs(
            float(bs_delta("C", self.spot, strike, self.r, self.q, atm_sigma, maturity_years))
        )
        delta = float(np.clip(delta, self.deltas[0], self.deltas[-1]))
        return self.iv_at_delta_maturity(delta, maturity_days)

    def to_chain(
        self,
        *,
        expiry_days: tuple[int, ...],
        n_strikes_each_side: int,
        strike_spacing_pct: float,
    ) -> ChainSnapshot:
        """Derive the tradeable strike/expiry chain from this surface."""
        step = self.spot * strike_spacing_pct
        lo = self.spot - n_strikes_each_side * step
        strikes = (lo + np.arange(2 * n_strikes_each_side + 1) * step).astype(np.float64)
        expiries = np.asarray(expiry_days, dtype=np.float64) / self.trading_days_per_year
        if bool(np.all(self.ivs == self.ivs[0, 0])):
            ivs = np.full((expiries.shape[0], strikes.shape[0]), float(self.ivs[0, 0]))
        else:
            ivs = np.asarray(
                [
                    [self.iv_for_strike(float(strike), float(days)) for strike in strikes]
                    for days in expiry_days
                ],
                dtype=np.float64,
            )
        return ChainSnapshot(
            strikes=strikes,
            ivs=ivs,
            expiries=expiries,
            spot=self.spot,
            r=self.r,
            q=self.q,
            t=self.t,
            trading_days_per_year=self.trading_days_per_year,
        )
