from __future__ import annotations

import numpy as np

from optspread.agents.base import FlatAgent
from optspread.config import EnvConfig, GBMConfig
from optspread.envs.builder import EnvBundle
from optspread.eval.rollout import RolloutTrace, collect_rollout_trace, wave1_credit_vrp_statistic
from optspread.market.gbm_vrp import GBMVRPGenerator
from optspread.market.snapshot import REGIME_FEATURE_KEYS
from optspread.training.env_factory import EnvFactory


def test_wave1_credit_vrp_statistic_is_positive_for_registered_behavior() -> None:
    vrp = np.linspace(-0.02, 0.02, 20, dtype=np.float64)
    features = np.zeros((vrp.size, len(REGIME_FEATURE_KEYS)), dtype=np.float64)
    features[:, REGIME_FEATURE_KEYS.index("vrp")] = vrp
    actions = np.where(vrp > 0.0, 11, 0).astype(np.int64)

    trace = RolloutTrace(actions=actions, features=features)

    assert wave1_credit_vrp_statistic(trace) > 0.8


def test_collect_rollout_trace_records_decision_time_observables() -> None:
    cfg = GBMConfig(n_days=5)
    factory = EnvFactory(
        EnvBundle(
            env=EnvConfig(episode_length=5),
            gbm=cfg,
            generator_factory=lambda: GBMVRPGenerator(cfg, sigma=0.16, vrp_vol_premium=0.04),
        )
    )

    trace = collect_rollout_trace(FlatAgent(), factory, (123,), deterministic=True)

    assert trace.feature_names == REGIME_FEATURE_KEYS
    assert trace.actions.tolist() == [0, 0, 0, 0, 0]
    assert np.isfinite(trace.feature("vrp")).all()
