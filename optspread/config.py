"""Pydantic v2 configuration models for every injectable component.

These are deliberately flat and serializable: configs are the single source of
truth a future phase will edit to swap the GBM generator for Heston, recalibrate
costs to OptionMetrics quotes, or re-weight the reward. No behaviour lives here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GBMConfig(BaseModel):
    """Wave-0 geometric-Brownian-motion generator with FAIR implied vol.

    The whole chain is priced at the same ``sigma`` that drives the path, so by
    construction there is no volatility-risk premium and every structure has
    ~zero expectancy before costs.
    """

    sigma: float = Field(default=0.20, gt=0.0)
    spot0: float = Field(default=5000.0, gt=0.0)
    n_days: int = Field(default=63, gt=0)  # ~one trading quarter
    r: float = Field(default=0.05, ge=0.0)
    q: float = Field(default=0.0, ge=0.0)
    trading_days_per_year: int = Field(default=252, gt=0)
    # Two expiries are always present so calendar spreads can be built. Values
    # are calendar-day tenors measured from each step's "today".
    expiry_days: tuple[int, int] = (21, 42)
    # Strike grid: number of strikes each side of spot and spacing in % of spot.
    n_strikes_each_side: int = Field(default=40, gt=0)
    strike_spacing_pct: float = Field(default=0.01, gt=0.0)
    # Wing offset (in delta) for spread/condor long protection legs.
    wing_width_delta: float = Field(default=0.05, gt=0.0)


class EnvConfig(BaseModel):
    seed: int = 42
    episode_length: int = Field(default=63, gt=0)
    multiplier: float = Field(default=100.0, gt=0.0)  # contract multiplier
    contracts_per_trade: int = Field(default=1, gt=0)
    recent_returns_window: int = Field(default=64, gt=0)
    # Starting account equity. Large enough that Phase 1 never rejects a trade
    # for buying power; used to normalise margin/P&L in the observation.
    initial_cash: float = Field(default=100_000.0, gt=0.0)


class CostConfig(BaseModel):
    """Quoted-spread transaction cost. Half-spread is paid per leg, per side."""

    half_spread_bps: float = Field(default=5.0, ge=0.0)
    min_cost_per_leg: float = Field(default=0.10, ge=0.0)
    # Quoted spread widens for deeper-OTM options; this scales that widening.
    otm_widening: float = Field(default=2.0, ge=0.0)
    multiplier: float = Field(default=100.0, gt=0.0)


class MarginConfig(BaseModel):
    multiplier: float = Field(default=100.0, gt=0.0)
    # Reg-T style naked requirement: fraction of underlying notional.
    naked_margin_rate: float = Field(default=0.20, gt=0.0)
    naked_floor_rate: float = Field(default=0.10, gt=0.0)
    defined_risk_multiplier: float = Field(default=1.0, gt=0.0)


class RewardConfig(BaseModel):
    mtm_weight: float = 1.0
    margin_normalized_weight: float = 0.0
    sharpe_weight: float = 0.0
    sortino_weight: float = 0.0
    cvar_weight: float = 0.0
    eta: float = Field(default=0.01, gt=0.0)  # EMA rate for differential Sharpe/Sortino
    cvar_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    cvar_threshold: float = -0.02
    margin_floor: float = Field(default=1.0, gt=0.0)
    pnl_scale: float = Field(default=1.0, gt=0.0)  # divides raw $ pnl before EMA stats
