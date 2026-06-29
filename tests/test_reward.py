"""Acceptance tests for the reward subsystem."""

from __future__ import annotations

import pytest

from optspread.config import RewardConfig
from optspread.reward.components import (
    CVaRPenalty,
    DifferentialSharpe,
    MarginNormalizer,
    MTMPnL,
    Sortino,
    empirical_cvar,
)
from optspread.reward.composite import CompositeReward, build_default_reward
from optspread.reward.context import StepContext


def _ctx(pnl: float, margin: float = 100.0, equity: float = 0.0) -> StepContext:
    return StepContext(pnl=pnl, margin=margin, equity=equity, day=0, did_trade=False)


def test_mtm_is_scaled_pnl() -> None:
    c = MTMPnL(RewardConfig(pnl_scale=10.0))
    assert c.update(_ctx(50.0)) == pytest.approx(5.0)


def test_margin_normalizer_divides_by_margin_with_floor() -> None:
    c = MarginNormalizer(RewardConfig(margin_floor=1.0))
    assert c.update(_ctx(20.0, margin=200.0)) == pytest.approx(0.1)
    # FLAT (zero margin) falls back to the floor instead of dividing by zero.
    assert c.update(_ctx(20.0, margin=0.0)) == pytest.approx(20.0)


def test_default_reward_is_pure_mtm_under_wave0_weights() -> None:
    # Default config: mtm_weight=1, all others 0 -> reward == pnl.
    reward = build_default_reward(RewardConfig())
    reward.reset()
    total = reward.update(_ctx(37.5))
    assert total == pytest.approx(37.5)
    bd = reward.last_breakdown
    assert bd["mtm"] == pytest.approx(37.5)
    assert bd["margin_normalized"] == 0.0
    assert bd["diff_sharpe"] == 0.0
    assert bd["sortino"] == 0.0
    assert bd["cvar"] == 0.0


def test_composite_weights_sum_components() -> None:
    reward = CompositeReward(
        [
            (MTMPnL(RewardConfig(pnl_scale=1.0)), 2.0),
            (MarginNormalizer(RewardConfig(margin_floor=1.0)), 0.5),
        ]
    )
    reward.reset()
    # 2*pnl + 0.5*(pnl/margin) = 2*10 + 0.5*(10/100)
    total = reward.update(_ctx(10.0, margin=100.0))
    assert total == pytest.approx(2 * 10.0 + 0.5 * (10.0 / 100.0))
    assert set(reward.last_breakdown) == {"mtm", "margin_normalized"}


def test_diff_sharpe_first_step_is_zero_and_rewards_consistency() -> None:
    c = DifferentialSharpe(RewardConfig(eta=0.1, pnl_scale=1.0))
    c.reset()
    assert c.update(_ctx(1.0)) == 0.0  # first step only seeds the EMAs
    # A steady positive stream should yield non-negative differential Sharpe.
    vals = [c.update(_ctx(1.0)) for _ in range(5)]
    assert all(v >= -1e-9 for v in vals)


def test_diff_sharpe_reset_clears_state() -> None:
    cfg = RewardConfig(eta=0.1, pnl_scale=1.0)
    c = DifferentialSharpe(cfg)
    c.reset()
    c.update(_ctx(1.0))
    first_after_seed = c.update(_ctx(2.0))
    c.reset()
    assert c.update(_ctx(1.0)) == 0.0  # back to seeding behaviour
    # Same sequence after reset reproduces the same value -> deterministic.
    assert c.update(_ctx(2.0)) == pytest.approx(first_after_seed)


def test_sortino_ignores_upside_volatility() -> None:
    c = Sortino(RewardConfig(eta=0.1, pnl_scale=1.0))
    c.reset()
    c.update(_ctx(1.0))  # seed
    # With no downside seen, a positive return is not penalised (>= 0).
    assert c.update(_ctx(2.0)) >= 0.0


def test_empirical_cvar_matches_hand_value() -> None:
    # Worst 25% of these 8 returns = the 2 smallest = -5, -4 -> mean -4.5.
    returns = [-5.0, -4.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    assert empirical_cvar(returns, 0.25) == pytest.approx(-4.5)
    assert empirical_cvar([], 0.25) == 0.0


def test_cvar_penalty_is_nonpositive_and_zero_for_benign() -> None:
    c = CVaRPenalty(RewardConfig(cvar_alpha=0.5, cvar_threshold=-0.02, pnl_scale=1.0), window=8)
    c.reset()
    # All-positive returns: tail CVaR > threshold -> penalty clamps to 0.
    for _ in range(8):
        assert c.update(_ctx(1.0)) == 0.0
    c.reset()
    # A heavy loss drags the tail below threshold -> strictly negative penalty.
    pen = c.update(_ctx(-100.0))
    assert pen < 0.0
