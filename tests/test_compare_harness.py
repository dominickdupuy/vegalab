"""Distributional agents run through the same Evaluator/MetricSuite as PPO."""

from __future__ import annotations

import math

from optspread.agents.distributional.config import QRDQNConfig
from optspread.agents.distributional.qrdqn_agent import QRDQNAgent
from optspread.config import EnvConfig
from optspread.envs.builder import EnvBundle
from optspread.eval.evaluator import Evaluator
from optspread.eval.metrics import MetricSuite
from optspread.training.env_factory import EnvFactory


def test_qrdqn_agent_runs_through_shared_evaluator() -> None:
    factory = EnvFactory(EnvBundle(env=EnvConfig(episode_length=5)))
    agent = QRDQNAgent(
        factory.obs_dim,
        factory.n_actions,
        QRDQNConfig(n_quantiles=16, hidden_sizes=(16,), norm_obs=False),
    )
    report = Evaluator(factory, eval_seeds=(91_001, 91_002), metrics=MetricSuite()).run(
        agent, deterministic=True
    )
    assert report.episode_returns.shape == (2,)
    assert report.per_step_returns.size == 10
    assert math.isfinite(report.mean_pnl)
    assert abs(sum(report.action_frequencies.values()) - 1.0) < 1e-9
