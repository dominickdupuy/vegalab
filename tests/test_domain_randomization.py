"""ParamSampler and hidden-internals checks."""

from __future__ import annotations

import numpy as np

from optspread.config import GBMConfig
from optspread.market.gbm_vrp import GBMVRPGenerator
from optspread.market.priors import GBMVRPPriors, ParamSampler, UniformPrior


def test_param_sampler_resamples_per_episode() -> None:
    sampler = ParamSampler(
        GBMVRPPriors(
            sigma=UniformPrior(low=0.10, high=0.11),
            vrp_vol_premium=UniformPrior(low=0.03, high=0.04),
        )
    )
    rng = np.random.default_rng(0)
    a = sampler.sample(rng).values
    b = sampler.sample(rng).values
    assert a != b
    assert 0.10 <= a["sigma"] <= 0.11
    assert 0.03 <= b["vrp_vol_premium"] <= 0.04


def test_generator_internals_not_in_observation_features() -> None:
    gen = GBMVRPGenerator.randomized(GBMConfig(n_days=3))
    snapshot = gen.reset(np.random.default_rng(1))
    assert "sigma" not in snapshot.regime_features
    assert "vrp_vol_premium" not in snapshot.regime_features
    assert snapshot.surface is not None
    assert gen.current_params
