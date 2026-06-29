"""A non-learning agent runs end-to-end through the shared eval harness.

This is the build-order step-3 smoke test: it proves the EnvFactory + Evaluator +
MetricSuite spine works on its own, before any network exists, so a learning agent
later only has to satisfy the same ``Agent`` protocol.
"""

from __future__ import annotations

import math

from optspread.agents.base import RandomAgent
from optspread.config import EnvConfig
from optspread.envs.builder import EnvBundle
from optspread.eval.evaluator import Evaluator
from optspread.eval.metrics import MetricSuite
from optspread.training.env_factory import EnvFactory


def test_random_agent_runs_through_eval_harness() -> None:
    factory = EnvFactory(EnvBundle(env=EnvConfig(episode_length=8)))
    ev = Evaluator(factory, eval_seeds=(70_001, 70_002, 70_003, 70_004), metrics=MetricSuite())
    report = ev.run(RandomAgent(seed=0), deterministic=False)

    # The report is well-formed and finite.
    assert report.episode_returns.shape == (4,)
    assert report.per_step_returns.size == 4 * 8
    assert math.isfinite(report.mean_pnl)
    assert math.isfinite(report.sharpe)
    # Action frequencies form a distribution over the library.
    assert abs(sum(report.action_frequencies.values()) - 1.0) < 1e-9
    # A churning random agent trades at least once.
    assert report.turnover > 0.0
