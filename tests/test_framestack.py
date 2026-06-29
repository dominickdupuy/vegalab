"""Frame-stack observation wrapper."""

from __future__ import annotations

import numpy as np

from optspread.agents.sequence.framestack import FrameStackObservation
from optspread.config import EnvConfig
from optspread.envs.builder import EnvBundle, build_default_env


def test_framestack_shapes_and_updates() -> None:
    env = FrameStackObservation(build_default_env(EnvBundle(env=EnvConfig(episode_length=3))), k=3)
    obs, _ = env.reset(seed=7)
    base_dim = env.env.observation_space.shape[0]
    assert obs.shape == (base_dim * 3,)
    first = obs.copy()
    obs2, *_ = env.step(np.int64(0))
    assert obs2.shape == obs.shape
    assert not np.array_equal(first, obs2)
