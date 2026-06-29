"""In-vitro proof: CVaR avoids the high-mean tail arm."""

from __future__ import annotations

import numpy as np
import torch

from optspread.agents.distributional.config import QRDQNConfig
from optspread.agents.distributional.qrdqn_agent import QRDQNAgent
from optspread.agents.distributional.risk import RiskMeasure
from optspread.toys.fat_tail_bandit import bandit_quantiles, greedy_arm


def test_mean_greedy_picks_tail_arm_cvar_avoids_it() -> None:
    assert greedy_arm(RiskMeasure.mean()) == 0
    assert greedy_arm(RiskMeasure.cvar(0.10)) == 1
    assert greedy_arm(RiskMeasure.cvar(0.05)) == 1


def test_qrdqn_agent_action_selection_avoids_tail_under_cvar() -> None:
    values, _taus = bandit_quantiles(n_quantiles=200)
    cfg = QRDQNConfig(n_quantiles=200, hidden_sizes=(8,), norm_obs=False)
    mean_agent = QRDQNAgent(1, 3, cfg, risk_measure=RiskMeasure.mean())
    cvar_agent = QRDQNAgent(1, 3, cfg, risk_measure=RiskMeasure.cvar(0.1))
    for agent in (mean_agent, cvar_agent):
        for param in agent.network.parameters():
            param.data.zero_()
        final = agent.network.net[-1]
        assert isinstance(final, torch.nn.Linear)
        final.bias.data.copy_(torch.as_tensor(values.reshape(-1), dtype=torch.float32))

    obs = np.zeros(1, dtype=np.float32)
    assert mean_agent.act(obs, deterministic=True) == 0
    assert cvar_agent.act(obs, deterministic=True) == 1
