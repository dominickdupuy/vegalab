"""RealDataReplay: historical surface/path as a drop-in generator."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from optspread.config import GBMConfig
from optspread.data.optionmetrics_loader import SurfaceRow
from optspread.features.regime_features import build_regime_features
from optspread.market.snapshot import MarketSnapshot


class RealDataReplay:
    """Replay historical surfaces through the same generator protocol as synthetic."""

    def __init__(
        self,
        rows: list[SurfaceRow],
        config: GBMConfig | None = None,
        *,
        warmup_rows: int = 0,
    ) -> None:
        if len(rows) < 2:
            raise ValueError("RealDataReplay needs at least two rows")
        if warmup_rows < 0:
            raise ValueError("warmup_rows must be non-negative")
        if warmup_rows >= len(rows) - 1:
            raise ValueError("warmup_rows must leave at least one replay step")
        self.rows = rows
        self.warmup_rows = warmup_rows
        self.config = config or GBMConfig(n_days=len(rows) - 1 - warmup_rows)
        self._idx = 0
        self._log_returns: list[float] = []
        self._iv_history: list[float] = []

    def reset(self, rng: np.random.Generator) -> MarketSnapshot:
        self._idx = 0
        self._log_returns = []
        self._iv_history = []
        if self.warmup_rows > 0:
            self._seed_warmup()
        return self._snapshot()

    def step(self) -> MarketSnapshot:
        if self.done:
            raise RuntimeError("step() called after replay horizon")
        prev_spot = self.rows[self._idx].spot
        self._idx += 1
        spot = self.rows[self._idx].spot
        self._log_returns.append(float(np.log(spot / prev_spot)))
        return self._snapshot()

    @property
    def done(self) -> bool:
        replay_steps = self._idx - self.warmup_rows
        max_steps = min(len(self.rows) - 1 - self.warmup_rows, self.config.n_days)
        return replay_steps >= max_steps

    def _snapshot(self) -> MarketSnapshot:
        row = self.rows[self._idx]
        trade_t = self._idx - self.warmup_rows
        surface = replace(row.surface, t=trade_t) if self.warmup_rows > 0 else row.surface
        atm = surface.iv_at_delta_maturity(0.50, float(surface.maturity_days[0]))
        self._iv_history.append(atm)
        chain = surface.to_chain(
            expiry_days=self.config.expiry_days,
            n_strikes_each_side=self.config.n_strikes_each_side,
            strike_spacing_pct=self.config.strike_spacing_pct,
        )
        return MarketSnapshot(
            chain=chain,
            t=trade_t if self.warmup_rows > 0 else self._idx,
            regime_features=build_regime_features(
                surface=surface,
                log_returns=self._log_returns,
                iv_history=self._iv_history,
            ),
            surface=surface,
        )

    def _seed_warmup(self) -> None:
        """Accrue causal lead-in history without exposing those rows as decisions."""
        for idx in range(self.warmup_rows):
            surface = self.rows[idx].surface
            atm = surface.iv_at_delta_maturity(0.50, float(surface.maturity_days[0]))
            self._iv_history.append(atm)
            if idx > 0:
                prev_spot = self.rows[idx - 1].spot
                spot = self.rows[idx].spot
                self._log_returns.append(float(np.log(spot / prev_spot)))
        self._idx = self.warmup_rows
