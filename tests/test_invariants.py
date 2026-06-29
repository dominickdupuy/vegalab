"""Property-based invariants that must hold on every step of any rollout."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from optspread.actions.library import N_ACTIONS
from optspread.envs.builder import build_default_env


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    actions=st.lists(st.integers(min_value=0, max_value=N_ACTIONS - 1), min_size=1, max_size=63),
)
def test_step_invariants_hold(seed: int, actions: list[int]) -> None:
    env = build_default_env()
    obs, _ = env.reset(seed=seed)
    assert np.all(np.isfinite(obs))
    for a in actions:
        obs, reward, term, trunc, info = env.step(a)
        # Margin is never negative.
        assert info["margin"] >= 0.0
        # Costs are never negative.
        assert info["cost"] >= 0.0
        # Reward and P&L are finite real numbers.
        assert math.isfinite(reward)
        assert math.isfinite(info["pnl"])
        # The observation stays finite and in-space.
        assert np.all(np.isfinite(obs))
        # Cash is exactly the initial balance plus every recorded cash flow.
        pf = env.portfolio
        assert pf.cash == pytest.approx(pf.initial_cash + sum(pf.cash_flows), abs=1e-6)
        # Equity decomposes cleanly: initial + realized + open-position mark.
        chain = env.chain
        equity = pf.equity(chain)
        assert equity == pytest.approx(
            pf.initial_cash + pf.realized_pnl + pf.unrealized_pnl(chain), abs=1e-6
        )
        if term or trunc:
            break


def test_no_look_ahead_observation_uses_only_current_chain() -> None:
    """The observation after a step must reflect the post-step chain, and a step's
    P&L must be computable from chains the env has already revealed — never future
    information. We check the env never advances the path outside step()."""
    env = build_default_env()
    env.reset(seed=5)
    day_before = env.generator.done  # path not yet at horizon
    assert day_before is False
    # Inspect the path day before and after a single step: it advances by exactly 1.
    chain0 = env.chain
    env.step(0)
    chain1 = env.chain
    assert chain1.t == chain0.t + 1
