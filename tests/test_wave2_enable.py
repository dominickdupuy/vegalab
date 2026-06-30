"""Wave-2 Heston enablement through the shared training/eval harness."""

from __future__ import annotations

import numpy as np

from optspread.agents.base import RandomAgent
from optspread.config import GBMConfig
from optspread.curriculum.waves import wave2_spec
from optspread.envs.observation import ObservationBuilder
from optspread.eval.rollout import RolloutTrace, collect_rollout_trace, wave2_ivrank_statistic
from optspread.training.curriculum_factory import wave_factory


def test_wave2_factory_env_resets_and_steps_with_expected_obs_width() -> None:
    factory = wave_factory(2, episode_length=2)
    expected_dim = ObservationBuilder(factory.bundle.env).dim

    assert factory.obs_dim == expected_dim

    env = factory.make()
    try:
        obs, _info = env.reset(seed=123)
        assert obs.shape == (factory.obs_dim,)

        next_obs, _reward, _terminated, _truncated, _info = env.step(0)
        assert next_obs.shape == (factory.obs_dim,)
    finally:
        env.close()


def test_wave2_rollout_trace_and_ivrank_statistic_do_not_raise() -> None:
    factory = wave_factory(2, episode_length=3)

    trace = collect_rollout_trace(RandomAgent(seed=7), factory, (456,), deterministic=False)
    statistic = wave2_ivrank_statistic(trace)

    assert isinstance(trace, RolloutTrace)
    assert trace.actions.size > 0
    assert trace.features.shape[0] == trace.actions.size
    assert np.isfinite(statistic) or np.isnan(statistic)


def test_wave2_heston_regime_features_exclude_hidden_params() -> None:
    generator = wave2_spec(GBMConfig(n_days=2)).make_generator()
    snapshot = generator.reset(np.random.default_rng(789))

    hidden_params = {"kappa", "theta", "sigma_v", "rho", "v0"}
    assert hidden_params.isdisjoint(snapshot.regime_features)
