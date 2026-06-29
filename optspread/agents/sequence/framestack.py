"""Frame-stacking observation wrapper for hidden-regime inference."""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

Obs = NDArray[np.float32]


class FrameStackObservation(gym.ObservationWrapper[Obs, np.int64, Obs]):
    """Concatenate the last ``k`` observations; no recurrent off-policy needed."""

    def __init__(self, env: gym.Env[Obs, np.int64], k: int) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        super().__init__(env)
        self.k = k
        self.frames: deque[Obs] = deque(maxlen=k)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("FrameStackObservation requires a Box observation space")
        low = np.tile(env.observation_space.low, k).astype(np.float32)
        high = np.tile(env.observation_space.high, k).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Obs, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(np.asarray(obs, dtype=np.float32))
        return self.observation(obs), info

    def observation(self, observation: Obs) -> Obs:
        if len(self.frames) < self.k:
            for _ in range(self.k - len(self.frames)):
                self.frames.append(np.asarray(observation, dtype=np.float32))
        return np.concatenate(list(self.frames), dtype=np.float32)

    def step(self, action: np.int64) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(np.asarray(obs, dtype=np.float32))
        return self.observation(obs), float(reward), terminated, truncated, info
