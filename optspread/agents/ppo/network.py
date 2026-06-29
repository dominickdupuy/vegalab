"""ActorCritic MLP for discrete PPO.

Follows the standard CleanRL initialization scheme: orthogonal weights, hidden
layers gained by sqrt(2) with tanh, the policy head gained by 0.01 (so the initial
policy is near-uniform — critical for exploration / avoiding instant collapse) and
the value head by 1.0. Capacity is intentionally tiny (two width-64 layers by
default); Wave 0 must be fittable by a small network.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal weight init with constant bias — the PPO-standard scheme."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers.append(layer_init(nn.Linear(last, h)))
        layers.append(nn.Tanh())
        last = h
    layers.append(layer_init(nn.Linear(last, out_dim), std=out_gain))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Discrete actor-critic with separate (default) or shared trunks."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        *,
        shared_trunk: bool = False,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.shared_trunk = shared_trunk
        # ``trunk`` is the shared body (Identity when trunks are separate, so the
        # forward path is uniform); ``actor``/``critic`` are the heads or, when
        # separate, the full per-branch MLPs.
        self.trunk: nn.Module
        self.actor: nn.Module
        self.critic: nn.Module

        if shared_trunk:
            trunk_layers: list[nn.Module] = []
            last = obs_dim
            for h in hidden_sizes:
                trunk_layers.append(layer_init(nn.Linear(last, h)))
                trunk_layers.append(nn.Tanh())
                last = h
            self.trunk = nn.Sequential(*trunk_layers)
            self.actor = layer_init(nn.Linear(last, n_actions), std=0.01)
            self.critic = layer_init(nn.Linear(last, 1), std=1.0)
        else:
            self.trunk = nn.Identity()
            self.actor = _mlp(obs_dim, hidden_sizes, n_actions, out_gain=0.01)
            self.critic = _mlp(obs_dim, hidden_sizes, 1, out_gain=1.0)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.actor(self.trunk(x))
        return logits

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        value: torch.Tensor = self.critic(self.trunk(x))
        return value

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (action, log_prob, entropy, value). Samples if ``action`` is None."""
        logits = self._logits(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.get_value(x)

    @torch.no_grad()
    def act(self, x: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """Inference-only action selection (argmax if deterministic, else sample)."""
        logits = self._logits(x)
        if deterministic:
            greedy: torch.Tensor = torch.argmax(logits, dim=-1)
            return greedy
        sampled: torch.Tensor = Categorical(logits=logits).sample()
        return sampled
