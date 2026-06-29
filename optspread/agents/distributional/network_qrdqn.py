"""QR-DQN network: state -> action x fixed quantile values."""

from __future__ import annotations

import torch
from torch import nn

from optspread.agents.distributional.quantile_loss import quantile_midpoints
from optspread.agents.ppo.network import layer_init


class QRDQNNetwork(nn.Module):
    """Small MLP with an ``n_actions * n_quantiles`` head."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_quantiles: int = 200,
        hidden_sizes: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        layers: list[nn.Module] = []
        last = obs_dim
        for hidden in hidden_sizes:
            layers.append(layer_init(nn.Linear(last, hidden)))
            layers.append(nn.ReLU())
            last = hidden
        layers.append(layer_init(nn.Linear(last, n_actions * n_quantiles), std=0.01))
        self.net = nn.Sequential(*layers)
        self.quantile_fractions: torch.Tensor
        self.register_buffer("quantile_fractions", quantile_midpoints(n_quantiles))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(obs)
        return out.view(obs.shape[0], self.n_actions, self.n_quantiles)
