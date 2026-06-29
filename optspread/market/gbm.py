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
from optspread.instruments.chain import ChainSnapshot
from optspread.market.snapshot import MarketSnapshot


class GBMGenerator:
    """Constant-vol, zero-drift GBM with fair IV. Implements ``PriceGenerator``."""

    def __init__(self, config: GBMConfig) -> None:
        self.config = config
        self._rng: np.random.Generator | None = None
        self._spot: float = config.spot0
        self._day: int = 0
        self._log_returns: list[float] = []

    # -- PriceGenerator protocol ------------------------------------------- #

    def reset(self, rng: np.random.Generator) -> MarketSnapshot:
        self._rng = rng
        self._spot = self.config.spot0
        self._day = 0
        self._log_returns = []
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

    def _build_chain(self) -> ChainSnapshot:
        cfg = self.config
        step = self._spot * cfg.strike_spacing_pct
        n = cfg.n_strikes_each_side
        lo = self._spot - n * step
        strikes = lo + np.arange(2 * n + 1) * step
        # Fair IV: flat surface at sigma across all strikes and expiries.
        expiries = np.array(cfg.expiry_days, dtype=np.float64) / cfg.trading_days_per_year
        ivs = np.full((expiries.shape[0], strikes.shape[0]), cfg.sigma, dtype=np.float64)
        return ChainSnapshot(
            strikes=strikes.astype(np.float64),
            ivs=ivs,
            expiries=expiries,
            spot=float(self._spot),
            r=cfg.r,
            q=cfg.q,
            t=self._day,
            trading_days_per_year=cfg.trading_days_per_year,
        )

    def _regime_features(self) -> dict[str, float]:
        cfg = self.config
        # Trailing momentum: standardized mean of recent daily log returns.
        window = min(len(self._log_returns), 21)
        if window > 0:
            recent = np.array(self._log_returns[-window:])
            dt = 1.0 / cfg.trading_days_per_year
            daily_sigma = cfg.sigma * np.sqrt(dt)
            momentum = float(np.mean(recent) / daily_sigma) if daily_sigma > 0 else 0.0
            realized = float(np.std(recent) / np.sqrt(dt)) if window > 1 else cfg.sigma
        else:
            momentum = 0.0
            realized = cfg.sigma
        # Fair IV => VRP ~ 0, flat term structure, IV-rank undefined (use 0.5).
        return {
            "trailing_momentum": momentum,
            "realized_vol": realized,
            "vrp": cfg.sigma**2 - realized**2,
            "iv_rank": 0.5,
            "term_slope": 0.0,
        }

    def _snapshot(self) -> MarketSnapshot:
        return MarketSnapshot(
            chain=self._build_chain(),
            t=self._day,
            regime_features=self._regime_features(),
        )
