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
        sample_window: int | None = None,
    ) -> None:
        if len(rows) < 2:
            raise ValueError("RealDataReplay needs at least two rows")
        if warmup_rows < 0:
            raise ValueError("warmup_rows must be non-negative")
        if warmup_rows >= len(rows) - 1:
            raise ValueError("warmup_rows must leave at least one replay step")
        if sample_window is not None:
            if sample_window < 2:
                raise ValueError("sample_window must include at least two tradeable rows")
            if warmup_rows + sample_window > len(rows):
                raise ValueError("sample_window plus warmup_rows exceeds available rows")
        self.rows = rows
        self.warmup_rows = warmup_rows
        self.sample_window = sample_window
        default_n_days = (
            sample_window - 1 if sample_window is not None else len(rows) - 1 - warmup_rows
        )
        self.config = config or GBMConfig(n_days=default_n_days)
        self._episode_rows = rows
        self._idx = 0
        self._log_returns: list[float] = []
        self._iv_history: list[float] = []
        self._rebase_surface_time = warmup_rows > 0

    def reset(self, rng: np.random.Generator) -> MarketSnapshot:
        self._select_episode_rows(rng)
        self._idx = 0
        self._log_returns = []
        self._iv_history = []
        if self.warmup_rows > 0:
            self._seed_warmup()
        return self._snapshot()

    def step(self) -> MarketSnapshot:
        if self.done:
            raise RuntimeError("step() called after replay horizon")
        prev_spot = self._episode_rows[self._idx].spot
        self._idx += 1
        spot = self._episode_rows[self._idx].spot
        self._log_returns.append(float(np.log(spot / prev_spot)))
        return self._snapshot()

    @property
    def done(self) -> bool:
        replay_steps = self._idx - self.warmup_rows
        max_steps = min(len(self._episode_rows) - 1 - self.warmup_rows, self.config.n_days)
        return replay_steps >= max_steps

    def _snapshot(self) -> MarketSnapshot:
        row = self._episode_rows[self._idx]
        trade_t = self._idx - self.warmup_rows
        surface = replace(row.surface, t=trade_t) if self._rebase_surface_time else row.surface
        atm = surface.iv_at_delta_maturity(0.50, float(surface.maturity_days[0]))
        self._iv_history.append(atm)
        chain = surface.to_chain(
            expiry_days=self.config.expiry_days,
            n_strikes_each_side=self.config.n_strikes_each_side,
            strike_spacing_pct=self.config.strike_spacing_pct,
        )
        return MarketSnapshot(
            chain=chain,
            t=trade_t if self._rebase_surface_time else self._idx,
            regime_features=build_regime_features(
                surface=surface,
                log_returns=self._log_returns,
                iv_history=self._iv_history,
            ),
            surface=surface,
        )

    def _select_episode_rows(self, rng: np.random.Generator) -> None:
        """Select the deterministic full replay or a causal random real-data window."""
        if self.sample_window is None:
            self._episode_rows = self.rows
            self._rebase_surface_time = self.warmup_rows > 0
            return

        latest_start = len(self.rows) - self.sample_window
        start = int(rng.integers(self.warmup_rows, latest_start + 1))
        window_start = start - self.warmup_rows
        window_end = start + self.sample_window
        self._episode_rows = self.rows[window_start:window_end]
        self._rebase_surface_time = True

    def _seed_warmup(self) -> None:
        """Accrue causal lead-in history without exposing those rows as decisions."""
        for idx in range(self.warmup_rows):
            surface = self._episode_rows[idx].surface
            atm = surface.iv_at_delta_maturity(0.50, float(surface.maturity_days[0]))
            self._iv_history.append(atm)
            if idx > 0:
                prev_spot = self._episode_rows[idx - 1].spot
                spot = self._episode_rows[idx].spot
                self._log_returns.append(float(np.log(spot / prev_spot)))
        self._idx = self.warmup_rows
