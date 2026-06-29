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
    vrp_vol_premium: UniformPrior = Field(default_factory=lambda: UniformPrior(low=0.02, high=0.08))


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
