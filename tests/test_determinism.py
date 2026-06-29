"""Determinism contract: same seed => byte-identical trajectory (invariant #3)."""

from __future__ import annotations

import numpy as np

from optspread.envs.builder import build_default_env


def _rollout(seed: int, actions: list[int]) -> tuple[list, list, list]:  # type: ignore[type-arg]
    env = build_default_env()
    obs, _ = env.reset(seed=seed)
    observations = [obs.copy()]
    rewards: list[float] = []
    pnls: list[float] = []
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        observations.append(obs.copy())
        rewards.append(r)
        pnls.append(info["pnl"])
        if term or trunc:
            break
    return observations, rewards, pnls


def test_same_seed_is_byte_identical() -> None:
    actions = [i % 19 for i in range(63)]
    obs_a, rew_a, pnl_a = _rollout(123, actions)
    obs_b, rew_b, pnl_b = _rollout(123, actions)
    assert len(obs_a) == len(obs_b)
    for a, b in zip(obs_a, obs_b, strict=True):
        assert np.array_equal(a, b)  # exact, not approximate
    assert rew_a == rew_b
    assert pnl_a == pnl_b


def test_different_seed_diverges() -> None:
    actions = [14] * 63  # always-on short strangle
    _, rew_a, _ = _rollout(1, actions)
    _, rew_b, _ = _rollout(2, actions)
    assert rew_a != rew_b  # different Brownian path => different P&L


def test_reset_reseeds_path() -> None:
    env = build_default_env()
    env.reset(seed=99)
    first = [env.step(14)[1] for _ in range(10)]
    env.reset(seed=99)
    second = [env.step(14)[1] for _ in range(10)]
    assert first == second
