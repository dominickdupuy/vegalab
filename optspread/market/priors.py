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
    # VRP-premium prior is deliberately exaggerated for TEACHABILITY. Wave 1's job
    # is to teach "sell premium when implied vol is rich"; a realistic premium
    # (~0.02-0.08 vol points, episode Sharpe ~0.6) is too weak/noisy versus the
    # zero-variance FLAT action for PPO or IQN to learn within budget (both
    # collapse to flat). A 0.25 premium is learned cleanly in 40k steps. The range
    # below spans the learnability threshold so the agent learns a CONDITIONAL
    # policy (sell more when the observed VRP feature is higher), which BV_1
    # measures as corr(credit_indicator, vrp) > 0. Realistic VRP magnitudes return
    # at Phase 5 on real OptionMetrics data.
    vrp_vol_premium: UniformPrior = Field(default_factory=lambda: UniformPrior(low=0.06, high=0.20))


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
        return SampledParams(values)
