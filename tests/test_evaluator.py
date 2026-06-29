"""Evaluator: determinism, disjoint-seed enforcement, FLAT baseline behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from optspread.agents.base import FlatAgent
from optspread.config import EnvConfig
from optspread.envs.builder import EnvBundle
from optspread.eval.evaluator import Evaluator, assert_disjoint
from optspread.eval.metrics import MetricSuite
from optspread.training.env_factory import EnvFactory

EVAL_SEEDS = (90_001, 90_002, 90_003)


def _factory() -> EnvFactory:
    return EnvFactory(EnvBundle(env=EnvConfig(episode_length=10)))


def test_eval_is_deterministic() -> None:
    ev = Evaluator(_factory(), EVAL_SEEDS, MetricSuite())
    a = ev.run(FlatAgent(), deterministic=True)
    b = ev.run(FlatAgent(), deterministic=True)
    assert np.array_equal(a.per_step_returns, b.per_step_returns)
    assert a.mean_pnl == b.mean_pnl
    assert a.pnl_ci == b.pnl_ci


def test_disjoint_seed_enforcement() -> None:
    with pytest.raises(ValueError, match="overlap"):
        Evaluator(_factory(), EVAL_SEEDS, MetricSuite(), train_seeds=(1, 2, 90_002))
    # The free helper is the same check the constructor uses.
    with pytest.raises(ValueError):
        assert_disjoint([1, 2, 3], [3, 4])
    assert_disjoint([1, 2, 3], [4, 5])  # disjoint -> no raise


def test_flat_agent_holds_no_risk() -> None:
    ev = Evaluator(_factory(), EVAL_SEEDS, MetricSuite())
    report = ev.run(FlatAgent(), deterministic=True)
    # FLAT never trades, never holds a position: zero P&L every step, full FLAT freq.
    assert report.flat_frequency == 1.0
    assert report.turnover == 0.0
    assert np.allclose(report.per_step_returns, 0.0)
    assert report.mean_pnl == pytest.approx(0.0)
