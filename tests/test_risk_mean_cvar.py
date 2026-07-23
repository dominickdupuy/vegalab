"""Spectral Mean-CVaR risk measure vs analytic oracles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from optspread.agents.distributional.config import IQNConfig
from optspread.agents.distributional.iqn_agent import IQNAgent
from optspread.agents.distributional.risk import RiskMeasure

VALUES = np.arange(1.0, 11.0)  # 1..10: mean 5.5
TAUS = np.arange(0.05, 1.0, 0.10)  # 0.05..0.95; taus <= 0.2 pick values 1,2 -> cvar 1.5


def test_mean_cvar_blend_matches_analytic() -> None:
    assert RiskMeasure.cvar(0.2).from_quantiles(VALUES, TAUS) == pytest.approx(1.5)
    assert RiskMeasure.mean().from_quantiles(VALUES, TAUS) == pytest.approx(5.5)
    blend = RiskMeasure.mean_cvar(0.2, 0.5).from_quantiles(VALUES, TAUS)
    assert blend == pytest.approx(0.5 * 5.5 + 0.5 * 1.5)


def test_mean_cvar_endpoints_reduce_to_mean_and_cvar() -> None:
    assert RiskMeasure.mean_cvar(0.2, 1.0).from_quantiles(VALUES, TAUS) == pytest.approx(5.5)
    assert RiskMeasure.mean_cvar(0.2, 0.0).from_quantiles(VALUES, TAUS) == pytest.approx(1.5)


def test_mean_cvar_without_taus_uses_sorted_tail() -> None:
    rng = np.random.default_rng(3)
    shuffled = rng.permutation(VALUES)
    assert RiskMeasure.mean_cvar(0.2, 0.5).from_quantiles(shuffled) == pytest.approx(3.5)


def test_mean_cvar_from_samples() -> None:
    assert RiskMeasure.mean_cvar(0.2, 0.5).from_samples(VALUES) == pytest.approx(3.5)


def test_torch_parity_with_numpy() -> None:
    measure = RiskMeasure.mean_cvar(0.2, 0.7)
    expected = measure.from_quantiles(VALUES, TAUS)
    got = measure.torch_from_quantiles(
        torch.as_tensor(VALUES, dtype=torch.float64), torch.as_tensor(TAUS, dtype=torch.float64)
    )
    assert float(got) == pytest.approx(float(expected))


def test_mean_cvar_validation() -> None:
    with pytest.raises(ValueError):
        RiskMeasure.mean_cvar(0.2, 1.5)
    with pytest.raises(ValueError):
        RiskMeasure.mean_cvar(0.0, 0.5)


def test_iqn_mixture_tau_sampling_realizes_the_blend() -> None:
    config = IQNConfig(seed=5, hidden_sizes=(16,), cvar_alpha=0.2)
    agent = IQNAgent(4, 3, config, risk_measure=RiskMeasure.mean_cvar(0.2, 0.5))
    torch.manual_seed(0)
    taus = agent.sample_taus(200, 100, agent.risk_measure)
    frac_above_alpha = float((taus > 0.2).float().mean())
    # full-range picks happen w.p. 0.5 and land above alpha w.p. 0.8 -> 0.40
    assert frac_above_alpha == pytest.approx(0.40, abs=0.02)
    torch.manual_seed(0)
    all_tail = agent.sample_taus(200, 100, RiskMeasure.mean_cvar(0.2, 0.0))
    assert float(all_tail.max()) <= 0.2


def test_mean_cvar_round_trips_through_checkpoint(tmp_path: Path) -> None:
    config = IQNConfig(seed=6, hidden_sizes=(16,), cvar_alpha=0.2)
    agent = IQNAgent(4, 3, config, risk_measure=RiskMeasure.mean_cvar(0.2, 0.9))
    path = tmp_path / "agent.pt"
    agent.save(path)
    restored = IQNAgent.from_checkpoint(path)
    assert restored.risk_measure.name == "mean_cvar"
    assert restored.risk_measure.mean_weight == pytest.approx(0.9)
