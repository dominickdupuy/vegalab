"""Acceptance tests for margin/reg_t.py."""

from __future__ import annotations

import pytest

from optspread.actions.templates import (
    BullPutSpreadTemplate,
    FlatTemplate,
    LongCallTemplate,
    ShortStrangleTemplate,
)
from optspread.config import MarginConfig
from optspread.margin.reg_t import RegTStyleMargin
from optspread.portfolio import pnl
from optspread.portfolio.position import Position
from tests.conftest import build_fair_chain

MULT = 100.0


def _position(legs, action_id: int = 0) -> Position:  # type: ignore[no-untyped-def]
    return Position(
        legs=tuple(legs),
        action_id=action_id,
        margin=0.0,
        open_day=0,
        entry_cash_flow=pnl.opening_cash_flow(legs, MULT),
    )


def test_flat_has_zero_margin() -> None:
    chain = build_fair_chain()
    mm = RegTStyleMargin(MarginConfig())
    assert mm.margin(_position(FlatTemplate().build(chain, 0.40)), chain) == 0.0


def test_defined_risk_spread_is_width_minus_credit() -> None:
    chain = build_fair_chain()
    legs = BullPutSpreadTemplate(5).build(chain, 0.16)
    short_put = next(leg for leg in legs if leg.is_short)
    long_put = next(leg for leg in legs if leg.is_long)
    width = short_put.strike - long_put.strike
    credit = -pnl.net_premium_per_share(legs)
    mm = RegTStyleMargin(MarginConfig())
    expected = (width - credit) * MULT
    assert mm.margin(_position(legs), chain) == pytest.approx(expected, rel=1e-9)


def test_long_only_debit_margin_is_premium() -> None:
    chain = build_fair_chain()
    legs = LongCallTemplate().build(chain, 0.40)
    premium = pnl.net_premium_per_share(legs)
    mm = RegTStyleMargin(MarginConfig())
    assert mm.margin(_position(legs), chain) == pytest.approx(premium * MULT, rel=1e-9)


def test_undefined_risk_is_positive_and_larger_than_defined() -> None:
    chain = build_fair_chain()
    mm = RegTStyleMargin(MarginConfig())
    naked = ShortStrangleTemplate().build(chain, 0.16)
    defined = BullPutSpreadTemplate(5).build(chain, 0.16)
    m_naked = mm.margin(_position(naked), chain)
    m_defined = mm.margin(_position(defined), chain)
    assert m_naked > 0
    # Naked Reg-T notional is materially larger than the capped defined-risk margin.
    # (Not a fixed multiple: the 5-strike wing widens the defined-risk worst case.)
    assert m_naked > m_defined


def test_margin_is_always_nonnegative() -> None:
    chain = build_fair_chain()
    mm = RegTStyleMargin(MarginConfig())
    for tmpl in (LongCallTemplate(), BullPutSpreadTemplate(5), ShortStrangleTemplate()):
        legs = tmpl.build(chain, 0.16)
        assert mm.margin(_position(legs), chain) >= 0.0
