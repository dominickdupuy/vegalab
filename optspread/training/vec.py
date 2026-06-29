"""Vector-env construction with deterministic per-episode path seeding.

Two facts drive this module:

1. **Decorrelated, fresh paths.** PPO needs each episode to be an independent GBM
   realization. The bare env reuses ``config.seed`` on a no-arg ``reset()``, so on
   autoreset every episode would replay the same path. ``EpisodeSeedWrapper`` fixes
   that by drawing the next episode's seed from its own deterministic sequence.

2. **SAME_STEP autoreset.** Gymnasium 1.x defaults to NEXT_STEP autoreset, which
   inserts a phantom step (agent action ignored, terminal obs echoed) after every
   episode — awkward and slightly biasing for an on-policy buffer. We build the
   vector env in ``SAME_STEP`` mode instead: the sub-env resets within the
   terminal ``step()`` and exposes the true terminal observation under
   ``info["final_obs"]``, exactly the classic CleanRL ``final_observation`` flow.

Determinism: the whole vector env is a pure function of ``base_seed`` — env *i*'s
episode-seed stream is anchored at ``base_seed + i``, so a run reproduces exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from numpy.typing import NDArray

from optspread.training.env_factory import EnvFactory

Obs = NDArray[np.float32]


class EpisodeSeedWrapper(gym.Wrapper[Obs, np.int64, Obs, np.int64]):
    """Injects a fresh, deterministic path seed into every ``reset()``.

    An explicit ``seed`` re-anchors the sequence (so the vector env's first
    seeded reset is honoured); thereafter — including on autoreset — each episode
    draws the next seed from the anchored stream.
    """

    def __init__(self, env: gym.Env[Obs, np.int64], seed: int) -> None:
        super().__init__(env)
        self._seed_seq = np.random.default_rng(seed)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Obs, dict[str, Any]]:
        if seed is not None:
            self._seed_seq = np.random.default_rng(seed)
        episode_seed = int(self._seed_seq.integers(0, 2**31 - 1))
        return self.env.reset(seed=episode_seed, options=options)


def make_env_thunk(factory: EnvFactory, seed: int) -> Callable[[], gym.Env[Obs, np.int64]]:
    """A zero-arg builder for one seeded env (what the vector API expects)."""

    def _thunk() -> gym.Env[Obs, np.int64]:
        return EpisodeSeedWrapper(factory.make(), seed)

    return _thunk


def make_vector_env(factory: EnvFactory, num_envs: int, base_seed: int) -> SyncVectorEnv:
    """Build a deterministic ``SyncVectorEnv`` of ``num_envs`` seeded SpreadEnvs."""
    thunks = [make_env_thunk(factory, base_seed + i) for i in range(num_envs)]
    return SyncVectorEnv(thunks, autoreset_mode=AutoresetMode.SAME_STEP)
