"""Implicit Quantile Network with cosine tau embedding."""

from __future__ import annotations

import math

import torch
from torch import nn

from optspread.agents.ppo.network import layer_init


class IQNNetwork(nn.Module):
    """IQN: ``(state, tau) -> action quantile values``."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_size: int = 64,
        cosine_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.cosine_embed_dim = cosine_embed_dim
        self.state_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )
        self.tau_net = nn.Sequential(
            layer_init(nn.Linear(cosine_embed_dim, hidden_size)),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, n_actions), std=0.01),
        )

    def cosine_embedding(self, taus: torch.Tensor) -> torch.Tensor:
        """Return cosine basis embeddings, shape ``(B, N, cosine_embed_dim)``."""
        if taus.ndim != 2:
            raise ValueError("taus must have shape (batch, n_quantiles)")
        i = torch.arange(self.cosine_embed_dim, device=taus.device, dtype=taus.dtype)
        return torch.cos(math.pi * taus.unsqueeze(-1) * i.view(1, 1, -1))

    def forward(self, obs: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        """Return values with shape ``(batch, n_quantiles, n_actions)``."""
        state = self.state_net(obs).unsqueeze(1)
        tau = self.tau_net(self.cosine_embedding(taus))
        values: torch.Tensor = self.head(state * tau)
        return values

    def action_quantiles(self, obs: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        """Return values with shape ``(batch, n_actions, n_quantiles)``."""
        return self.forward(obs, taus).permute(0, 2, 1)
