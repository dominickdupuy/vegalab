"""Short-horizon MDP where tail damage realizes one step after entry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from optspread.agents.distributional.risk import RiskMeasure


@dataclass(frozen=True, slots=True)
class MDPArmDistribution:
    rewards: NDArray[np.float64]
    probs: NDArray[np.float64]

    @property
    def mean(self) -> float:
        return float((self.rewards * self.probs).sum())

    def cvar(self, alpha: float) -> float:
        order = np.argsort(self.rewards)
        rewards = self.rewards[order]
        probs = self.probs[order]
        remaining = alpha
        total = 0.0
        for reward, prob in zip(rewards, probs, strict=True):
            take = min(remaining, float(prob))
            total += take * float(reward)
            remaining -= take
            if remaining <= 1e-12:
                break
        return total / alpha


class FatTailMDPEnv(gym.Env[NDArray[np.float32], np.int64]):
    """Two-step env: action 0 gives upfront reward then delayed crash risk."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._rng = np.random.default_rng(0)
        self._state = 0
        self._pending_tail = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = 0
        self._pending_tail = False
        return self._obs(), {}

    def step(
        self, action: np.int64 | int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self._state == 0:
            self._state = 1
            if int(action) == 0:
                self._pending_tail = True
                return self._obs(), 1.0, False, False, {}
            return self._obs(), 0.35, False, False, {}
        reward = -12.0 if self._pending_tail and self._rng.random() < 0.1 else 1.0
        if not self._pending_tail:
            reward = 0.35
        return self._obs(), reward, True, False, {}

    def _obs(self) -> NDArray[np.float32]:
        return np.asarray([float(self._state), float(self._pending_tail)], dtype=np.float32)


def mdp_action_distributions(gamma: float = 0.99) -> tuple[MDPArmDistribution, MDPArmDistribution]:
    tail = MDPArmDistribution(
        rewards=np.asarray([1.0 + gamma * -12.0, 1.0 + gamma * 1.0], dtype=np.float64),
        probs=np.asarray([0.1, 0.9], dtype=np.float64),
    )
    safe = MDPArmDistribution(
        rewards=np.asarray([0.35 + gamma * 0.35], dtype=np.float64),
        probs=np.asarray([1.0], dtype=np.float64),
    )
    return tail, safe


def greedy_mdp_action(risk: RiskMeasure, gamma: float = 0.99) -> int:
    tail, safe = mdp_action_distributions(gamma)
    if risk.name == "mean":
        scores = [tail.mean, safe.mean]
    else:
        scores = [tail.cvar(risk.alpha), safe.cvar(risk.alpha)]
    return int(np.argmax(scores))
