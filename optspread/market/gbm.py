"""Fair-IV geometric-Brownian-motion generator (Wave-0).

Zero drift, constant vol ``sigma``. The entire option chain is priced at the SAME
``sigma`` that drives the underlying path (fair implied vol), so there is no
variance-risk premium and — before costs — every structure has ~zero expectancy.
This is the economic sanity baseline (see CLAUDE.md): always-on credit agents must
read ~0 P&L with no costs and < 0 with costs.

Two expiries are always exposed (constant calendar-day tenors) so calendar spreads
are buildable. Regime features are emitted on the canonical schema; in Wave-0 most
are degenerate constants, with a genuine trailing-momentum estimate.
"""

from __future__ import annotations

import numpy as np

from optspread.config import GBMConfig
from optspread.features.regime_features import build_regime_features
from optspread.market.snapshot import MarketSnapshot
from optspread.market.surface import IVSurface


class GBMGenerator:
    """Constant-vol, zero-drift GBM with fair IV. Implements ``PriceGenerator``."""

    def __init__(self, config: GBMConfig) -> None:
        self.config = config
        self._rng: np.random.Generator | None = None
        self._spot: float = config.spot0
        self._day: int = 0
        self._log_returns: list[float] = []
        self._iv_history: list[float] = []

    # -- PriceGenerator protocol ------------------------------------------- #

    def reset(self, rng: np.random.Generator) -> MarketSnapshot:
        self._rng = rng
        self._spot = self.config.spot0
        self._day = 0
        self._log_returns = []
        self._iv_history = []
        return self._snapshot()

    def step(self) -> MarketSnapshot:
        """Advance one trading day. The ONLY place the path moves."""
        if self._rng is None:
            raise RuntimeError("step() called before reset()")
        if self.done:
            raise RuntimeError("step() called after horizon reached")
        dt = 1.0 / self.config.trading_days_per_year
        sigma = self.config.sigma
        # Zero drift in the real measure; the -0.5 sigma^2 keeps E[S] flat.
        z = float(self._rng.standard_normal())
        log_ret = -0.5 * sigma * sigma * dt + sigma * np.sqrt(dt) * z
        self._spot *= float(np.exp(log_ret))
        self._log_returns.append(log_ret)
        self._day += 1
        return self._snapshot()

    @property
    def done(self) -> bool:
        return self._day >= self.config.n_days

    # -- internals --------------------------------------------------------- #

    def _surface(self) -> IVSurface:
        cfg = self.config
        return IVSurface.flat(
            sigma=cfg.sigma,
            spot=float(self._spot),
            r=cfg.r,
            q=cfg.q,
            t=self._day,
            trading_days_per_year=cfg.trading_days_per_year,
        )

    def _snapshot(self) -> MarketSnapshot:
        cfg = self.config
        surface = self._surface()
        atm = surface.iv_at_delta_maturity(0.50, float(surface.maturity_days[0]))
        self._iv_history.append(atm)
        chain = surface.to_chain(
            expiry_days=cfg.expiry_days,
            n_strikes_each_side=cfg.n_strikes_each_side,
            strike_spacing_pct=cfg.strike_spacing_pct,
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
