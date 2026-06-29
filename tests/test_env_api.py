"""Gymnasium API conformance for SpreadEnv."""

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from optspread.actions.library import N_ACTIONS
from optspread.envs.builder import build_default_env


def test_passes_gymnasium_env_checker() -> None:
    env = build_default_env()
    # Wraps reset/step and validates spaces, dtypes, seeding and the 5-tuple.
    check_env(env, skip_render_check=True)


def test_action_and_observation_spaces() -> None:
    env = build_default_env()
    assert env.action_space.n == N_ACTIONS
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert obs.dtype == np.float32


def test_episode_terminates_at_horizon() -> None:
    env = build_default_env()
    env.reset(seed=7)
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(0)
        steps += 1
        assert steps <= 1000  # guard against a runaway loop
    assert terminated
    assert steps == env.config.episode_length


def test_info_contains_diagnostics() -> None:
    env = build_default_env()
    env.reset(seed=3)
    _, _, _, _, info = env.step(5)
    for key in ("day", "action_id", "did_trade", "cost", "pnl", "margin", "reward_breakdown"):
        assert key in info
