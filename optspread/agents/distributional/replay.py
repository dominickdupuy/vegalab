"""Uniform numpy ring replay buffer."""

from __future__ import annotations

from dataclasses import dataclass

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
    """Fixed-size ring buffer with uniform sampling."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.obs_dim = obs_dim
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

    def sample(self, batch_size: int, rng: np.random.Generator) -> ReplayBatch:
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        idx = rng.integers(0, self._size, size=batch_size)
        return ReplayBatch(
            obs=self.obs[idx].copy(),
            actions=self.actions[idx].copy(),
            rewards=self.rewards[idx].copy(),
            next_obs=self.next_obs[idx].copy(),
            dones=self.dones[idx].copy(),
        )
