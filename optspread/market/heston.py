"""Wave-2 Heston stochastic-volatility generator."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from optspread.config import GBMConfig
from optspread.features.regime_features import build_regime_features
from optspread.market.pricing.cos_pricer import COSPricer
from optspread.market.priors import HestonPriors, ParamSampler, SampledParams
from optspread.market.snapshot import MarketSnapshot
from optspread.market.surface import DEFAULT_DELTA_GRID, DEFAULT_MATURITY_GRID_DAYS, IVSurface
from optspread.pricing.black_scholes import bs_delta, bs_price

_F = NDArray[np.float64]

_MIN_VARIANCE = 1.0e-8
_MIN_IV = 0.02
_MAX_IV = 2.00
_STRIKE_GRID_SIZE = 41
_STRIKE_WIDTH_MULT = 5.0
_MAX_LOG_MONEYNESS_WIDTH = 1.25


class HestonGenerator:
    """Physical Heston path with a richer Q-measure IV surface."""

    def __init__(
        self,
        config: GBMConfig,
        *,
        params: SampledParams | Mapping[str, float] | None = None,
        sampler: ParamSampler | None = None,
        warmup_days: int = 21,
        vrp_theta_mult: float = 1.3,
    ) -> None:
        if params is not None and sampler is not None:
            raise ValueError("pass either params or sampler, not both")
        if warmup_days < 0:
            raise ValueError("warmup_days must be non-negative")
        if vrp_theta_mult <= 1.0:
            raise ValueError("vrp_theta_mult must be > 1 so Q variance exceeds P variance")

        self.config = config
        self.sampler = sampler
        self.warmup_days = warmup_days
        self.vrp_theta_mult = vrp_theta_mult
        self._pricer = COSPricer()
        self._base_params = _coerce_params(
            params
            if params is not None
            else {
                "kappa": 4.0,
                "theta": config.sigma * config.sigma,
                "sigma_v": 0.5,
                "rho": -0.6,
                "v0": config.sigma * config.sigma,
            }
        )
        self._params = self._base_params
        self._rng: np.random.Generator | None = None
        self._spot = config.spot0
        self._variance = self._params.values["v0"]
        self._day = 0
        self._log_returns: list[float] = []
        self._iv_history: list[float] = []
        self._surface_cache: IVSurface | None = None
        self._surface_cache_day: int | None = None
        self._surface_cache_spot = self._spot
        self._surface_cache_variance = self._variance

    @classmethod
    def randomized(
        cls,
        config: GBMConfig,
        priors: HestonPriors | None = None,
    ) -> HestonGenerator:
        return cls(config, sampler=ParamSampler(priors or HestonPriors()))

    @property
    def current_params(self) -> dict[str, float]:
        return dict(self._params.values)

    def reset(self, rng: np.random.Generator) -> MarketSnapshot:
        self._rng = rng
        if self.sampler is not None:
            self._params = _coerce_params(self.sampler.sample(rng))
        else:
            self._params = self._base_params
        self._spot = self.config.spot0
        self._variance = max(self._params.values["v0"], 0.0)
        self._day = 0
        self._log_returns = []
        self._iv_history = []
        self._clear_surface_cache()
        self._run_warmup()
        return self._snapshot()

    def step(self) -> MarketSnapshot:
        if self._rng is None:
            raise RuntimeError("step() called before reset()")
        if self.done:
            raise RuntimeError("step() called after horizon reached")
        self._advance_path()
        self._day += 1
        self._clear_surface_cache()
        return self._snapshot()

    @property
    def done(self) -> bool:
        return self._day >= self.config.n_days

    def _run_warmup(self) -> None:
        if self._rng is None or self.warmup_days <= 0:
            return
        for _ in range(self.warmup_days):
            self._advance_path()
        self._clear_surface_cache()

    def _advance_path(self) -> None:
        if self._rng is None:
            raise RuntimeError("path advance before reset")
        dt = 1.0 / self.config.trading_days_per_year
        sqrt_dt = float(np.sqrt(dt))
        v_pos = max(self._variance, 0.0)
        rho = self._rho

        z_spot = float(self._rng.standard_normal())
        z_independent = float(self._rng.standard_normal())
        z_variance = rho * z_spot + float(np.sqrt(max(1.0 - rho * rho, 0.0))) * z_independent

        log_ret = -0.5 * v_pos * dt + float(np.sqrt(v_pos)) * sqrt_dt * z_spot
        self._spot *= float(np.exp(log_ret))
        self._log_returns.append(log_ret)

        variance_next = (
            self._variance
            + self._kappa * (self._theta - v_pos) * dt
            + self._sigma_v * float(np.sqrt(v_pos)) * sqrt_dt * z_variance
        )
        self._variance = max(float(variance_next), 0.0)

    @property
    def _kappa(self) -> float:
        return self._params.values["kappa"]

    @property
    def _theta(self) -> float:
        return self._params.values["theta"]

    @property
    def _theta_q(self) -> float:
        return self._theta * self.vrp_theta_mult

    @property
    def _sigma_v(self) -> float:
        return self._params.values["sigma_v"]

    @property
    def _rho(self) -> float:
        return self._params.values["rho"]

    def _surface(self) -> IVSurface:
        if (
            self._surface_cache is not None
            and self._surface_cache_day == self._day
            and self._surface_cache_spot == self._spot
            and self._surface_cache_variance == self._variance
        ):
            return self._surface_cache

        ivs = self._price_standard_iv_grid()
        surface = IVSurface(
            deltas=DEFAULT_DELTA_GRID.astype(np.float64),
            maturity_days=DEFAULT_MATURITY_GRID_DAYS.astype(np.float64),
            ivs=ivs,
            spot=float(self._spot),
            r=self.config.r,
            q=self.config.q,
            t=self._day,
            trading_days_per_year=self.config.trading_days_per_year,
        )
        self._surface_cache = surface
        self._surface_cache_day = self._day
        self._surface_cache_spot = self._spot
        self._surface_cache_variance = self._variance
        return surface

    def _clear_surface_cache(self) -> None:
        self._surface_cache = None
        self._surface_cache_day = None

    def _price_standard_iv_grid(self) -> _F:
        rows = [
            self._price_maturity_iv_row(float(days) / self.config.trading_days_per_year)
            for days in DEFAULT_MATURITY_GRID_DAYS
        ]
        return np.asarray(rows, dtype=np.float64)

    def _price_maturity_iv_row(self, T: float) -> _F:
        strikes = self._strike_grid(T)
        prices = self._heston_call_prices(strikes=strikes, T=T)
        all_ivs = _implied_call_vols(
            prices=prices,
            spot=float(self._spot),
            strikes=strikes,
            r=self.config.r,
            q=self.config.q,
            T=T,
        )
        all_deltas = bs_delta("C", self._spot, strikes, self.config.r, self.config.q, all_ivs, T)
        strike_ivs: list[float] = []
        strike_deltas: list[float] = []
        for delta, iv in zip(all_deltas, all_ivs, strict=True):
            if np.isfinite(iv):
                delta_float = float(delta)
                if np.isfinite(delta_float) and 0.0 < delta_float < 1.0:
                    strike_ivs.append(float(iv))
                    strike_deltas.append(delta_float)

        if len(strike_ivs) < 2:
            fallback = float(np.sqrt(max(self._theta_q, _MIN_VARIANCE)))
            return np.full(DEFAULT_DELTA_GRID.shape, np.clip(fallback, _MIN_IV, _MAX_IV))

        deltas = np.asarray(strike_deltas, dtype=np.float64)
        ivs = np.asarray(strike_ivs, dtype=np.float64)
        order = np.argsort(deltas)
        deltas = deltas[order]
        ivs = ivs[order]
        unique_deltas, unique_indices = np.unique(deltas, return_index=True)
        unique_ivs = ivs[unique_indices]
        interpolated = np.interp(DEFAULT_DELTA_GRID, unique_deltas, unique_ivs)
        return np.clip(interpolated, _MIN_IV, _MAX_IV).astype(np.float64)

    def _strike_grid(self, T: float) -> _F:
        forward = self._spot * float(np.exp((self.config.r - self.config.q) * T))
        sigma_span = float(np.sqrt(max(self._variance, self._theta_q, _MIN_VARIANCE)))
        width = _STRIKE_WIDTH_MULT * sigma_span * float(np.sqrt(max(T, 1.0e-8)))
        width = float(np.clip(width, 0.15, _MAX_LOG_MONEYNESS_WIDTH))
        log_moneyness = np.linspace(-width, width, _STRIKE_GRID_SIZE, dtype=np.float64)
        return np.asarray(forward * np.exp(log_moneyness), dtype=np.float64)

    def _heston_call_prices(self, *, strikes: _F, T: float) -> _F:
        try:
            return self._pricer.heston_call_prices(
                spot=float(self._spot),
                strikes=strikes,
                r=self.config.r,
                q=self.config.q,
                T=T,
                kappa=self._kappa,
                theta=self._theta_q,
                sigma_v=self._sigma_v,
                rho=self._rho,
                v0=max(self._variance, 0.0),
            )
        except (FloatingPointError, OverflowError, ValueError):
            return np.full(strikes.shape, np.nan, dtype=np.float64)

    def _snapshot(self) -> MarketSnapshot:
        surface = self._surface()
        atm = surface.iv_at_delta_maturity(0.50, float(surface.maturity_days[0]))
        self._iv_history.append(atm)
        chain = surface.to_chain(
            expiry_days=self.config.expiry_days,
            n_strikes_each_side=self.config.n_strikes_each_side,
            strike_spacing_pct=self.config.strike_spacing_pct,
        )
        features = build_regime_features(
            surface=surface,
            log_returns=self._log_returns,
            iv_history=self._iv_history,
        )
        return MarketSnapshot(
            chain=chain,
            t=self._day,
            regime_features=features,
            surface=surface,
        )


def _coerce_params(params: SampledParams | Mapping[str, float]) -> SampledParams:
    values = dict(params.values if isinstance(params, SampledParams) else params)
    required = ("kappa", "theta", "sigma_v", "rho", "v0")
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"Heston params missing keys: {missing}")
    clean = {name: float(values[name]) for name in required}
    if clean["kappa"] <= 0.0:
        raise ValueError("kappa must be positive")
    if clean["theta"] <= 0.0:
        raise ValueError("theta must be positive")
    if clean["sigma_v"] < 0.0:
        raise ValueError("sigma_v must be non-negative")
    if not -1.0 < clean["rho"] < 0.0:
        raise ValueError("rho must be negative and greater than -1")
    if clean["v0"] < 0.0:
        raise ValueError("v0 must be non-negative")
    return SampledParams(clean)


def _implied_call_vols(
    *,
    prices: _F,
    spot: float,
    strikes: _F,
    r: float,
    q: float,
    T: float,
) -> _F:
    if T <= 0.0:
        return np.full(strikes.shape, np.nan, dtype=np.float64)

    finite = np.isfinite(prices)
    lower = np.maximum(spot * np.exp(-q * T) - strikes * np.exp(-r * T), 0.0)
    upper = spot * np.exp(-q * T)
    target = np.clip(prices, lower, upper)
    low_prices = bs_price("C", spot, strikes, r, q, _MIN_IV, T)
    high_prices = bs_price("C", spot, strikes, r, q, _MAX_IV, T)

    out = np.full(strikes.shape, np.nan, dtype=np.float64)
    low_mask = finite & (target <= low_prices)
    high_mask = finite & (target >= high_prices)
    active = finite & ~low_mask & ~high_mask
    out[low_mask] = _MIN_IV
    out[high_mask] = _MAX_IV
    if not bool(np.any(active)):
        return out

    lo = np.full(strikes.shape, _MIN_IV, dtype=np.float64)
    hi = np.full(strikes.shape, _MAX_IV, dtype=np.float64)
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        mid_prices = bs_price("C", spot, strikes, r, q, mid, T)
        move_lo = active & (mid_prices < target)
        move_hi = active & ~move_lo
        lo[move_lo] = mid[move_lo]
        hi[move_hi] = mid[move_hi]
    out[active] = np.clip(0.5 * (lo[active] + hi[active]), _MIN_IV, _MAX_IV)
    return out
