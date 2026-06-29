"""Wave-1 GBM generator with variance-risk premium."""

from __future__ import annotations

import numpy as np

from optspread.config import GBMConfig
from optspread.features.regime_features import build_regime_features
from optspread.market.priors import GBMVRPPriors, ParamSampler, SampledParams
from optspread.market.snapshot import MarketSnapshot
from optspread.market.surface import IVSurface


class GBMVRPGenerator:
    """Physical GBM path, risk-neutral surface priced richer by a vol premium."""

    def __init__(
        self,
        config: GBMConfig,
        *,
        sigma: float | None = None,
        vrp_vol_premium: float = 0.04,
        sampler: ParamSampler | None = None,
    ) -> None:
        self.config = config
        self.base_sigma = config.sigma if sigma is None else sigma
        self.base_vrp_vol_premium = vrp_vol_premium
        self.sampler = sampler
        self._rng: np.random.Generator | None = None
        self._spot = config.spot0
        self._day = 0
        self._log_returns: list[float] = []
        self._iv_history: list[float] = []
        self._params = SampledParams(
            {"sigma": self.base_sigma, "vrp_vol_premium": self.base_vrp_vol_premium}
        )

    @classmethod
    def randomized(cls, config: GBMConfig, priors: GBMVRPPriors | None = None) -> GBMVRPGenerator:
        return cls(config, sampler=ParamSampler(priors or GBMVRPPriors()))

    @property
    def current_params(self) -> dict[str, float]:
        return dict(self._params.values)

    def reset(self, rng: np.random.Generator) -> MarketSnapshot:
        self._rng = rng
        if self.sampler is not None:
            self._params = self.sampler.sample(rng)
        self._spot = self.config.spot0
        self._day = 0
        self._log_returns = []
        self._iv_history = []
        return self._snapshot()

    def step(self) -> MarketSnapshot:
        if self._rng is None:
            raise RuntimeError("step() called before reset()")
        if self.done:
            raise RuntimeError("step() called after horizon reached")
        dt = 1.0 / self.config.trading_days_per_year
        sigma = self._physical_sigma
        z = float(self._rng.standard_normal())
        log_ret = -0.5 * sigma * sigma * dt + sigma * np.sqrt(dt) * z
        self._spot *= float(np.exp(log_ret))
        self._log_returns.append(log_ret)
        self._day += 1
        return self._snapshot()

    @property
    def done(self) -> bool:
        return self._day >= self.config.n_days

    @property
    def _physical_sigma(self) -> float:
        return self._params.values["sigma"]

    @property
    def _implied_sigma(self) -> float:
        return self._params.values["sigma"] + self._params.values["vrp_vol_premium"]

    def _surface(self) -> IVSurface:
        return IVSurface.flat(
            sigma=self._implied_sigma,
            spot=float(self._spot),
            r=self.config.r,
            q=self.config.q,
            t=self._day,
            trading_days_per_year=self.config.trading_days_per_year,
        )

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
        if not self._log_returns:
            features["realized_vol"] = self._physical_sigma
            features["vrp"] = self._implied_sigma**2 - self._physical_sigma**2
        return MarketSnapshot(
            chain=chain,
            t=self._day,
            regime_features=features,
            surface=surface,
        )
