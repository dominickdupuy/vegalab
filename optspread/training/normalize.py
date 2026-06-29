"""Causal (online) normalization.

The observation normalizer updates its running mean/variance ONLY from the stream
it has already seen — never from a precomputed pass over a full, future-containing
dataset. That is mandatory for honest evaluation on real data in a later phase and
is a cheap habit to build now on synthetic Wave 0 (CLAUDE.md / brief section 1.5).

Uses Welford's algorithm in its batched (Chan et al.) form so a whole rollout
step's worth of observations updates the stats in one shot, numerically stably.

Evaluation freezes the stats gathered during training (``frozen=True``): the eval
stream is normalized by what the agent knew at train time, which is both causal
and identical across agents being compared.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class CausalObsNormalizer:
    """Per-feature running standardizer with clipping.

    Parameters
    ----------
    dim:
        Observation dimensionality.
    clip:
        Symmetric clip applied to standardized values (PPO-standard 10.0).
    epsilon:
        Variance floor to avoid division by zero before enough data is seen.
    """

    def __init__(self, dim: int, *, clip: float = 10.0, epsilon: float = 1e-8) -> None:
        self.dim = dim
        self.clip = clip
        self.epsilon = epsilon
        self._mean = np.zeros(dim, dtype=np.float64)
        self._m2 = np.zeros(dim, dtype=np.float64)
        self._count = 0.0
        self.frozen = False

    # -- stats ------------------------------------------------------------- #

    @property
    def var(self) -> NDArray[np.float64]:
        return self._m2 / self._count if self._count > 1 else np.ones(self.dim)

    @property
    def std(self) -> NDArray[np.float64]:
        return np.sqrt(self.var + self.epsilon)

    def update(self, batch: NDArray[np.float32]) -> None:
        """Fold a batch ``(n, dim)`` of observations into the running stats."""
        if self.frozen:
            return
        x = np.asarray(batch, dtype=np.float64).reshape(-1, self.dim)
        n = x.shape[0]
        if n == 0:
            return
        batch_mean = x.mean(axis=0)
        batch_m2 = ((x - batch_mean) ** 2).sum(axis=0)
        delta = batch_mean - self._mean
        total = self._count + n
        self._mean += delta * (n / total)
        # Chan's parallel variance combination.
        self._m2 += batch_m2 + delta**2 * (self._count * n / total)
        self._count = total

    def normalize(self, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        """Standardize and clip using the CURRENT stats (no peeking ahead)."""
        x = np.asarray(batch, dtype=np.float32)
        if self._count < 1:
            return np.clip(x, -self.clip, self.clip)
        z = (x - self._mean.astype(np.float32)) / self.std.astype(np.float32)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    # -- persistence ------------------------------------------------------- #

    def state_dict(self) -> dict[str, NDArray[np.float64] | float | bool]:
        return {"mean": self._mean.copy(), "m2": self._m2.copy(), "count": self._count}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self._mean = np.asarray(state["mean"], dtype=np.float64)
        self._m2 = np.asarray(state["m2"], dtype=np.float64)
        self._count = float(state["count"])  # type: ignore[arg-type]

    def save(self, path: str | Path) -> None:
        np.savez(path, mean=self._mean, m2=self._m2, count=self._count)

    def load(self, path: str | Path) -> None:
        data = np.load(path)
        self._mean = data["mean"]
        self._m2 = data["m2"]
        self._count = float(data["count"])


class ReturnNormalizer:
    """Optional reward/return scaler — OFF by default in PPOConfig.

    Reward normalization can mask the deliberate composite-reward weighting (the
    brief warns against it), so it is opt-in. When enabled it scales rewards by a
    running std of the discounted return, the PPO-standard trick for value-loss
    stability. Kept causal like the obs normalizer.
    """

    def __init__(self, *, gamma: float = 0.99, epsilon: float = 1e-8) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self._mean = 0.0
        self._m2 = 0.0
        self._count = 0.0
        self._ret = 0.0  # running discounted return accumulator

    @property
    def std(self) -> float:
        var = self._m2 / self._count if self._count > 1 else 1.0
        return float(np.sqrt(var + self.epsilon))

    def scale(self, reward: float, done: bool) -> float:
        self._ret = self._ret * self.gamma * (0.0 if done else 1.0) + reward
        self._count += 1
        delta = self._ret - self._mean
        self._mean += delta / self._count
        self._m2 += delta * (self._ret - self._mean)
        return reward / self.std
