"""RolloutBuffer + GAE.

Fixed-size on-policy storage for one rollout of ``(num_steps, num_envs)``
transitions, plus the generalized-advantage estimator. ``compute_gae`` is a free
function (pure given its inputs) so it can be checked against a hand-computed
example in ``test_gae`` — the GAE recursion is the easiest place for an
off-by-one to hide.

The recursion is CleanRL's verbatim: at the final stored step it bootstraps from
the post-rollout ``next_value``/``next_done``; elsewhere from the next stored
step. A terminal/truncation flag zeroes the bootstrap through ``nextnonterminal``.
"""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (advantages, returns), each shape ``(num_steps, num_envs)``.

    ``rewards``/``values``/``dones`` are ``(T, N)``; ``next_value``/``next_done``
    are ``(N,)`` — the value/done one step past the end of the rollout, used to
    bootstrap the last step. ``returns = advantages + values``.
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_nonterminal = 1.0 - next_done
            next_values = next_value
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


class RolloutBuffer:
    """Stores one PPO rollout and flattens it for minibatch updates."""

    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, device: torch.device) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.device = device
        shape = (num_steps, num_envs)
        self.obs = torch.zeros((*shape, obs_dim), device=device)
        self.actions = torch.zeros(shape, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(shape, device=device)
        self.rewards = torch.zeros(shape, device=device)
        self.dones = torch.zeros(shape, device=device)
        self.values = torch.zeros(shape, device=device)

    def add(
        self,
        t: int,
        obs: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self.obs[t] = obs
        self.actions[t] = action
        self.logprobs[t] = logprob
        self.rewards[t] = reward
        self.dones[t] = done
        self.values[t] = value.flatten()

    def compute_returns(
        self, next_value: torch.Tensor, next_done: torch.Tensor, *, gamma: float, gae_lambda: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_gae(
            self.rewards,
            self.values,
            self.dones,
            next_value.flatten(),
            next_done,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

    def flatten(self, advantages: torch.Tensor, returns: torch.Tensor) -> dict[str, torch.Tensor]:
        """Collapse ``(T, N, ...)`` into ``(T*N, ...)`` batch tensors for updates."""
        return {
            "obs": self.obs.reshape(-1, self.obs_dim),
            "actions": self.actions.reshape(-1),
            "logprobs": self.logprobs.reshape(-1),
            "values": self.values.reshape(-1),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
        }
