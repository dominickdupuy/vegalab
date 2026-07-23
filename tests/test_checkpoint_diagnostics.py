"""C0 checkpoint diagnostics: quantile spread + decision-gap-by-signal."""

from __future__ import annotations

import math

from optspread.agents.distributional.config import IQNConfig
from optspread.agents.distributional.iqn_agent import IQNAgent
from optspread.agents.distributional.risk import RiskMeasure
from optspread.eval.checkpoint_diagnostics import (
    collect_observations,
    diagnose_checkpoint,
    format_report,
)
from optspread.training.curriculum_factory import wave_factory


def _agent_and_obs() -> tuple[IQNAgent, object]:
    factory = wave_factory(0)
    config = IQNConfig(seed=11, hidden_sizes=(32,), cvar_alpha=0.2)
    agent = IQNAgent(factory.obs_dim, factory.n_actions, config, risk_measure=RiskMeasure.cvar(0.2))
    obs = collect_observations(factory, agent, episodes=2, seed_start=41_000)
    return agent, obs


def test_diagnostics_shapes_and_finiteness() -> None:
    agent, obs = _agent_and_obs()
    diag = diagnose_checkpoint(agent, obs)  # type: ignore[arg-type]  # obs is ndarray
    assert diag.n_states > 0
    assert math.isfinite(diag.spread_mean)
    assert math.isfinite(diag.value_scale)
    assert diag.spread_to_value_ratio >= 0.0
    for stats in (diag.mean_scoring, diag.cvar_scoring):
        assert math.isfinite(stats.gap_all)
        assert 0.0 <= stats.pct_states_trade_preferred <= 100.0
    assert diag.cvar_alpha == 0.2


def test_diagnostics_serialize_and_format() -> None:
    agent, obs = _agent_and_obs()
    diag = diagnose_checkpoint(agent, obs)  # type: ignore[arg-type]  # obs is ndarray
    payload = diag.to_dict()
    assert payload["n_states"] == diag.n_states
    assert "mean_scoring" in payload and "cvar_scoring" in payload
    text = format_report("TEST", diag)
    assert "quantile spread" in text
    assert "gap Q(best trade)-Q(FLAT)" in text
