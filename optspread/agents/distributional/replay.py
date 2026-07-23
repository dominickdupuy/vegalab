"""Uniform numpy ring replay buffer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    obs: NDArray[np.float32]
    actions: NDArray[np.int64]
    rewards: NDArray[np.float32]
    next_obs: NDArray[np.float32]
    dones: NDArray[np.float32]


class UniformReplayBuffer:
    """Fixed-size ring buffer with optional profitable-transition over-sampling.

    ``reward_priority_boost`` counteracts CVaR flat-collapse by making positive-
    reward transitions more learnable. This intentionally adds a mild optimistic
    trading bias during optimization, so there is no importance-sampling correction;
    deployment CVaR still tail-filters action selection.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        reward_priority_boost: float = 0.0,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if reward_priority_boost < 0.0:
            raise ValueError("reward_priority_boost must be non-negative")
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.reward_priority_boost = reward_priority_boost
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def add(
        self,
        obs: NDArray[np.float32],
        action: int,
        reward: float,
        next_obs: NDArray[np.float32],
        done: bool,
    ) -> None:
        idx = self._pos
        self.obs[idx] = np.asarray(obs, dtype=np.float32)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.next_obs[idx] = np.asarray(next_obs, dtype=np.float32)
        self.dones[idx] = 1.0 if done else 0.0
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self.capacity, self._size + 1)

    def state_dict(self) -> dict[str, Any]:
        """Serializable contents for training snapshots (only the filled region)."""
        n = self._size
        return {
            "obs": self.obs[:n].copy(),
            "next_obs": self.next_obs[:n].copy(),
            "actions": self.actions[:n].copy(),
            "rewards": self.rewards[:n].copy(),
            "dones": self.dones[:n].copy(),
            "pos": self._pos,
            "size": n,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore contents saved by :meth:`state_dict` into this buffer."""
        n = int(state["size"])
        if n > self.capacity:
            raise ValueError("snapshot larger than buffer capacity")
        self.obs[:n] = np.asarray(state["obs"], dtype=np.float32)
        self.next_obs[:n] = np.asarray(state["next_obs"], dtype=np.float32)
        self.actions[:n] = np.asarray(state["actions"], dtype=np.int64)
        self.rewards[:n] = np.asarray(state["rewards"], dtype=np.float32)
        self.dones[:n] = np.asarray(state["dones"], dtype=np.float32)
        self._pos = int(state["pos"])
        self._size = n

    def sample(self, batch_size: int, rng: np.random.Generator) -> ReplayBatch:
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        if self._size > 0 and self.reward_priority_boost > 0.0:
            w = np.asarray(
                1.0 + self.reward_priority_boost * (self.rewards[: self._size] > 0.0),
                dtype=np.float64,
            )
            p = w / w.sum()
            idx = rng.choice(self._size, size=batch_size, p=p)
        else:
            idx = rng.integers(0, self._size, size=batch_size)
        return ReplayBatch(
            obs=self.obs[idx].copy(),
            actions=self.actions[idx].copy(),
            rewards=self.rewards[idx].copy(),
            next_obs=self.next_obs[idx].copy(),
            dones=self.dones[idx].copy(),
        )
