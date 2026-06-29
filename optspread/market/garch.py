"""Held-out GARCH-style generator for Phase 6 smoke/generalization tests."""

from __future__ import annotations

import numpy as np

from optspread.config import GBMConfig
from optspread.features.regime_features import build_regime_features
from optspread.market.snapshot import MarketSnapshot
from optspread.market.surface import IVSurface


class GARCHGenerator:
    """Simple GARCH(1,1) physical path with surface IV from current variance."""

    def __init__(
        self,
        config: GBMConfig,
        *,
        omega: float = 0.000002,
        alpha: float = 0.08,
        beta: float = 0.90,
        iv_premium: float = 0.02,
    ) -> None:
        self.config = config
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.iv_premium = iv_premium
        self._rng: np.random.Generator | None = None
        self._spot = config.spot0
        self._day = 0
        self._var = (config.sigma / np.sqrt(config.trading_days_per_year)) ** 2
        self._log_returns: list[float] = []
        self._iv_history: list[float] = []

    def reset(self, rng: np.random.Generator) -> MarketSnapshot:
        self._rng = rng
        self._spot = self.config.spot0
        self._day = 0
        self._var = (self.config.sigma / np.sqrt(self.config.trading_days_per_year)) ** 2
        self._log_returns = []
        self._iv_history = []
        return self._snapshot()

    def step(self) -> MarketSnapshot:
        if self._rng is None:
            raise RuntimeError("step before reset")
        z = float(self._rng.standard_normal())
        log_ret = -0.5 * self._var + np.sqrt(self._var) * z
        self._spot *= float(np.exp(log_ret))
        self._var = self.omega + self.alpha * log_ret * log_ret + self.beta * self._var
        self._log_returns.append(log_ret)
        self._day += 1
        return self._snapshot()

    @property
    def done(self) -> bool:
        return self._day >= self.config.n_days

    def _snapshot(self) -> MarketSnapshot:
        ann_vol = float(np.sqrt(self._var * self.config.trading_days_per_year))
        surface = IVSurface.flat(
            sigma=max(ann_vol + self.iv_premium, 0.01),
            spot=float(self._spot),
            r=self.config.r,
            q=self.config.q,
            t=self._day,
            trading_days_per_year=self.config.trading_days_per_year,
        )
        atm = surface.iv_at_delta_maturity(0.50, float(surface.maturity_days[0]))
        self._iv_history.append(atm)
        chain = surface.to_chain(
            expiry_days=self.config.expiry_days,
            n_strikes_each_side=self.config.n_strikes_each_side,
            strike_spacing_pct=self.config.strike_spacing_pct,
        )
        return MarketSnapshot(
            chain=chain,
            t=self._day,
            regime_features=build_regime_features(
                surface=surface,
                log_returns=self._log_returns,
                iv_history=self._iv_history,
            ),
            surface=surface,
        )
