"""Baseline agents + the Wave-0 economic sanity gate (CLAUDE.md).

These are the acceptance tests for the economic expectation, run on a modest
episode count so they stay fast while still being statistically meaningful.
"""

from __future__ import annotations

import numpy as np

from optspread.agents.baselines import AlwaysOnAgent, FlatAgent, RandomAgent
from optspread.cli.smoke_run import evaluate
from optspread.config import CostConfig
from optspread.envs.builder import EnvBundle


def test_flat_agent_is_always_flat() -> None:
    agent = FlatAgent()
    assert all(agent.act(np.zeros(12, dtype=np.float32)) == 0 for _ in range(5))


def test_random_agent_is_reproducible_after_reset() -> None:
    a = RandomAgent(seed=7)
    first = [a.act(np.zeros(12, dtype=np.float32)) for _ in range(20)]
    a.reset()
    second = [a.act(np.zeros(12, dtype=np.float32)) for _ in range(20)]
    assert first == second


def test_always_on_holds_one_structure() -> None:
    a = AlwaysOnAgent(action_id=14)
    assert all(a.act(np.zeros(12, dtype=np.float32)) == 14 for _ in range(5))


# -- the economic sanity gate ------------------------------------------------ #

EPISODES = 200


def _no_cost() -> EnvBundle:
    return EnvBundle(cost=CostConfig(half_spread_bps=0.0, min_cost_per_leg=0.0))


def test_flat_has_exactly_zero_pnl() -> None:
    stats = evaluate(_no_cost(), lambda _s: FlatAgent(), EPISODES)
    assert stats.mean == 0.0
    assert stats.std == 0.0


def test_wave0_no_cost_always_on_is_zero_expectancy() -> None:
    # Fair IV => zero expectancy before costs. If this MAKES money, it's a bug
    # (pricing inconsistency / cost sign / look-ahead) — not a discovery.
    stats = evaluate(_no_cost(), lambda _s: AlwaysOnAgent(action_id=14), EPISODES)
    assert abs(stats.mean) <= 4.0 * stats.stderr


def test_wave0_costs_make_always_on_bleed() -> None:
    no_cost = evaluate(_no_cost(), lambda _s: AlwaysOnAgent(action_id=14), EPISODES)
    with_cost = evaluate(EnvBundle(), lambda _s: AlwaysOnAgent(action_id=14), EPISODES)
    # Costs strictly worsen expectancy; with costs the mean is negative.
    assert with_cost.mean < no_cost.mean
    assert with_cost.mean < 0.0


def test_wave0_random_bleeds_faster_than_always_on() -> None:
    always_on = evaluate(EnvBundle(), lambda _s: AlwaysOnAgent(action_id=14), EPISODES)
    random = evaluate(EnvBundle(), lambda s: RandomAgent(seed=s), EPISODES)
    assert random.mean < always_on.mean
