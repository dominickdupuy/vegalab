"""Uniform replay buffer correctness."""

from __future__ import annotations

import numpy as np

from optspread.agents.distributional.replay import UniformReplayBuffer


def test_replay_ring_overwrites_oldest_and_samples_copies() -> None:
    buf = UniformReplayBuffer(capacity=3, obs_dim=2)
    for i in range(5):
        obs = np.asarray([i, i + 1], dtype=np.float32)
        buf.add(obs, i, float(i), obs + 1, done=i % 2 == 0)
    assert buf.size == 3
    rng = np.random.default_rng(0)
    batch = buf.sample(8, rng)
    assert batch.obs.shape == (8, 2)
    assert batch.actions.min() >= 2
    before = buf.obs.copy()
    batch.obs[0, 0] = -999.0
    assert np.array_equal(buf.obs, before)
