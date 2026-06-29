"""One-step fat-tail bandit with known arm distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from optspread.agents.distributional.risk import RiskMeasure


@dataclass(frozen=True, slots=True)
class TwoPointArm:
    good_reward: float
    bad_reward: float
    bad_prob: float

    def sample(self, rng: np.random.Generator) -> float:
        return self.bad_reward if rng.random() < self.bad_prob else self.good_reward

    def quantiles(self, taus: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(taus <= self.bad_prob, self.bad_reward, self.good_reward)

    @property
    def mean(self) -> float:
        return self.bad_prob * self.bad_reward + (1.0 - self.bad_prob) * self.good_reward


DEFAULT_ARMS: tuple[TwoPointArm, ...] = (
    TwoPointArm(good_reward=1.2, bad_reward=-12.0, bad_prob=0.05),
    TwoPointArm(good_reward=0.4, bad_reward=0.4, bad_prob=0.0),
    TwoPointArm(good_reward=0.0, bad_reward=0.0, bad_prob=0.0),
)


class FatTailBanditEnv(gym.Env[NDArray[np.float32], np.int64]):
    """One-state one-step bandit: highest-mean arm has a catastrophic tail."""

    def __init__(self, arms: tuple[TwoPointArm, ...] = DEFAULT_ARMS) -> None:
        super().__init__()
        self.arms = arms
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(arms))
        self._rng = np.random.default_rng(0)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(
        self, action: np.int64 | int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        reward = self.arms[int(action)].sample(self._rng)
        return np.zeros(1, dtype=np.float32), reward, True, False, {}


def bandit_quantiles(
    arms: tuple[TwoPointArm, ...] = DEFAULT_ARMS, n_quantiles: int = 200
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    taus = ((2.0 * np.arange(1, n_quantiles + 1) - 1.0) / (2.0 * n_quantiles)).astype(np.float64)
    values = np.stack([arm.quantiles(taus) for arm in arms], axis=0)
    return values, taus


def greedy_arm(risk: RiskMeasure, arms: tuple[TwoPointArm, ...] = DEFAULT_ARMS) -> int:
    values, taus = bandit_quantiles(arms)
    scores = risk.from_quantiles(values, taus)
    return int(np.argmax(scores))
