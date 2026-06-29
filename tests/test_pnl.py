"""Acceptance tests for portfolio P&L and cash accounting."""

from __future__ import annotations

import pytest

from optspread.actions.templates import (
    BullPutSpreadTemplate,
    LongCallTemplate,
    ShortStrangleTemplate,
)
from optspread.portfolio import pnl
from optspread.portfolio.position import Portfolio
from tests.conftest import build_fair_chain

MULT = 100.0


def test_credit_structure_increases_cash_immediately() -> None:
    chain = build_fair_chain()
    legs = BullPutSpreadTemplate(5).build(chain, 0.16)
    credit_per_share = -pnl.net_premium_per_share(legs)
    assert credit_per_share > 0
    pf = Portfolio(multiplier=MULT)
    pf.reset(initial_cash=100_000.0)
    pf.open(legs, action_id=5, margin=0.0, day=0, cost=0.0)
    # Cash rose by exactly the net credit * multiplier.
    assert pf.cash == pytest.approx(100_000.0 + credit_per_share * MULT)


def test_debit_structure_decreases_cash_immediately() -> None:
    chain = build_fair_chain()
    legs = LongCallTemplate().build(chain, 0.40)
    debit_per_share = pnl.net_premium_per_share(legs)
    assert debit_per_share > 0
    pf = Portfolio(multiplier=MULT)
    pf.reset(initial_cash=100_000.0)
    pf.open(legs, action_id=1, margin=0.0, day=0, cost=0.0)
    assert pf.cash == pytest.approx(100_000.0 - debit_per_share * MULT)


def test_round_trip_zero_change_zero_cost_is_zero_pnl() -> None:
    chain = build_fair_chain()
    legs = ShortStrangleTemplate().build(chain, 0.16)
    pf = Portfolio(multiplier=MULT)
    pf.reset(initial_cash=100_000.0)
    pf.open(legs, action_id=14, margin=0.0, day=0, cost=0.0)
    # Close on the SAME chain (no price change) with no cost.
    rt = pf.close(chain, cost=0.0)
    assert rt == pytest.approx(0.0, abs=1e-9)
    assert pf.realized_pnl == pytest.approx(0.0, abs=1e-9)
    assert pf.cash == pytest.approx(100_000.0, abs=1e-9)


def test_round_trip_with_cost_per_side_is_minus_2c() -> None:
    chain = build_fair_chain()
    legs = ShortStrangleTemplate().build(chain, 0.16)
    pf = Portfolio(multiplier=MULT)
    pf.reset(initial_cash=100_000.0)
    c = 3.5
    pf.open(legs, action_id=14, margin=0.0, day=0, cost=c)
    pf.close(chain, cost=c)
    assert pf.realized_pnl == pytest.approx(-2 * c, abs=1e-9)


def test_unrealized_mtm_equals_value_minus_entry() -> None:
    chain0 = build_fair_chain(spot=5000.0)
    legs = LongCallTemplate().build(chain0, 0.40)
    pf = Portfolio(multiplier=MULT)
    pf.reset(initial_cash=100_000.0)
    pf.open(legs, action_id=1, margin=0.0, day=0, cost=0.0)
    # Move the market up: the long call gains.
    chain1 = build_fair_chain(spot=5200.0)
    expected = pnl.unrealized_pnl(legs, chain1, MULT)
    assert pf.unrealized_pnl(chain1) == pytest.approx(expected)
    assert pf.unrealized_pnl(chain1) > 0  # long call benefits from the rally


def test_cash_conservation_equals_realized() -> None:
    chain0 = build_fair_chain(spot=5000.0)
    legs = BullPutSpreadTemplate(5).build(chain0, 0.16)
    pf = Portfolio(multiplier=MULT)
    pf.reset(initial_cash=100_000.0)
    pf.open(legs, action_id=5, margin=0.0, day=0, cost=1.0)
    chain1 = build_fair_chain(spot=5100.0)
    pf.close(chain1, cost=1.0)
    assert sum(pf.cash_flows) == pytest.approx(pf.realized_pnl, abs=1e-9)
    assert pf.cash == pytest.approx(pf.initial_cash + pf.realized_pnl, abs=1e-9)
