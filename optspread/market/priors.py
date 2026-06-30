"""Domain-randomization priors and parameter sampler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field


class UniformPrior(BaseModel):
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        if self.high < self.low:
            raise ValueError("uniform prior high must be >= low")
        return float(rng.uniform(self.low, self.high))


class GBMVRPPriors(BaseModel):
    sigma: UniformPrior = Field(default_factory=lambda: UniformPrior(low=0.12, high=0.30))
    # VRP-premium prior spans ZERO so Wave 1 presents a VARIABLE-SIGN VRP regime:
    # negative-premium episodes (implied < realized) make selling premium a losing
    # trade, so the agent must CONDITION on VRP -- sell when rich, refrain when
    # cheap. This is what makes corr(credit_indicator, vrp) > 0 (BV_1) genuinely
    # optimal rather than indiscriminate selling. The magnitude is exaggerated for
    # TEACHABILITY: a realistic premium (~0.02-0.08, episode Sharpe ~0.6) is too
    # weak/noisy versus the zero-variance FLAT action for PPO or IQN to learn
    # within budget (both collapse to flat); a strong edge is learned in ~40k
    # steps. Realistic VRP magnitudes return at Phase 5 on real OptionMetrics data.
    # Mean premium stays positive (GV_1: mean VRP > 0). implied_sigma = sigma +
    # premium stays > 0 since sigma >= 0.12 and premium >= -0.04.
    vrp_vol_premium: UniformPrior = Field(
        default_factory=lambda: UniformPrior(low=-0.04, high=0.18)
    )


class HestonPriors(BaseModel):
    """Wave-2 stochastic-volatility priors.

    ``v0_theta_mult`` is converted by ``ParamSampler`` into ``v0 = theta * mult``
    so the latent starting variance is drawn near the sampled physical
    long-run variance.
    """

    kappa: UniformPrior = Field(default_factory=lambda: UniformPrior(low=1.0, high=10.0))
    theta: UniformPrior = Field(default_factory=lambda: UniformPrior(low=0.02, high=0.09))
    sigma_v: UniformPrior = Field(default_factory=lambda: UniformPrior(low=0.2, high=1.0))
    rho: UniformPrior = Field(default_factory=lambda: UniformPrior(low=-0.9, high=-0.3))
    v0_theta_mult: UniformPrior = Field(default_factory=lambda: UniformPrior(low=0.80, high=1.20))


@dataclass(frozen=True, slots=True)
class SampledParams:
    values: dict[str, float]


class ParamSampler:
    """Draws generator parameters per episode from configured priors."""

    def __init__(self, priors: BaseModel) -> None:
        self.priors = priors

    def sample(self, rng: np.random.Generator) -> SampledParams:
        values: dict[str, float] = {}
        for name, prior in self.priors:
            if isinstance(prior, UniformPrior):
                values[name] = prior.sample(rng)
        if "theta" in values and "v0_theta_mult" in values:
            values["v0"] = values["theta"] * values.pop("v0_theta_mult")
        return SampledParams(values)
