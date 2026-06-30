"""Reward-priority replay sampling behavior."""

from __future__ import annotations

import numpy as np

from optspread.agents.distributional.replay import UniformReplayBuffer


def _filled_buffer(*, reward_priority_boost: float) -> UniformReplayBuffer:
    buf = UniformReplayBuffer(capacity=10, obs_dim=2, reward_priority_boost=reward_priority_boost)
    rewards = (-1.0, 0.0, -0.5, -2.0, 1.0, 0.0, -1.5, 2.0, 0.0, -0.25)
    for i, reward in enumerate(rewards):
        obs = np.asarray([i, i + 100], dtype=np.float32)
        next_obs = obs + 0.5
        buf.add(obs, action=i, reward=reward, next_obs=next_obs, done=i % 3 == 0)
    return buf


def test_reward_priority_oversamples_profitable_transitions() -> None:
    buf = _filled_buffer(reward_priority_boost=9.0)
    batch = buf.sample(20_000, np.random.default_rng(7))

    population_positive_frac = float(np.mean(buf.rewards[: buf.size] > 0.0))
    sampled_positive_frac = float(np.mean(batch.rewards > 0.0))

    assert population_positive_frac == 0.2
    assert sampled_positive_frac > 0.5


def test_zero_reward_priority_samples_uniformly() -> None:
    buf = _filled_buffer(reward_priority_boost=0.0)
    batch = buf.sample(20_000, np.random.default_rng(7))

    population_positive_frac = float(np.mean(buf.rewards[: buf.size] > 0.0))
    sampled_positive_frac = float(np.mean(batch.rewards > 0.0))

    assert population_positive_frac == 0.2
    assert abs(sampled_positive_frac - population_positive_frac) < 0.02


def test_reward_priority_sampling_is_deterministic_for_same_seed() -> None:
    first = _filled_buffer(reward_priority_boost=9.0).sample(128, np.random.default_rng(123))
    second = _filled_buffer(reward_priority_boost=9.0).sample(128, np.random.default_rng(123))

    assert np.array_equal(first.actions, second.actions)
    assert np.array_equal(first.rewards, second.rewards)


def test_reward_priority_replay_batch_shapes_and_dtypes_are_unchanged() -> None:
    buf = _filled_buffer(reward_priority_boost=9.0)
    batch = buf.sample(16, np.random.default_rng(9))

    assert batch.obs.shape == (16, 2)
    assert batch.obs.dtype == np.float32
    assert batch.actions.shape == (16,)
    assert batch.actions.dtype == np.int64
    assert batch.rewards.shape == (16,)
    assert batch.rewards.dtype == np.float32
    assert batch.next_obs.shape == (16, 2)
    assert batch.next_obs.dtype == np.float32
    assert batch.dones.shape == (16,)
    assert batch.dones.dtype == np.float32
