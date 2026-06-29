"""Acceptance tests for costs/spread_cost.py."""

from __future__ import annotations

import pytest

from optspread.config import CostConfig
from optspread.costs.spread_cost import QuotedSpreadCost
from optspread.instruments.leg import OptionLeg
from tests.conftest import build_fair_chain


def _atm_leg(chain, qty: int = -1) -> OptionLeg:  # type: ignore[no-untyped-def]
    k = chain.nearest_strike(chain.spot)
    return OptionLeg("C", k, 0, qty, chain.price("C", k, 0))


def test_cost_scales_linearly_with_number_of_legs() -> None:
    chain = build_fair_chain()
    cm = QuotedSpreadCost(CostConfig())
    leg = _atm_leg(chain)
    c1 = cm.cost([leg], chain)
    c3 = cm.cost([leg, leg, leg], chain)
    assert c3 == pytest.approx(3 * c1)


def test_cost_scales_with_quoted_half_spread() -> None:
    chain = build_fair_chain()
    leg = _atm_leg(chain)
    cheap = QuotedSpreadCost(CostConfig(half_spread_bps=5.0)).cost([leg], chain)
    pricey = QuotedSpreadCost(CostConfig(half_spread_bps=20.0)).cost([leg], chain)
    assert pricey == pytest.approx(4 * cheap, rel=1e-6)


def test_deeper_otm_legs_cost_more() -> None:
    chain = build_fair_chain()
    cm = QuotedSpreadCost(CostConfig(min_cost_per_leg=0.0))
    atm = OptionLeg("C", chain.nearest_strike(chain.spot), 0, -1, 1.0)
    otm = OptionLeg("C", chain.nearest_strike(chain.spot * 1.15), 0, -1, 1.0)
    assert cm.cost([otm], chain) > cm.cost([atm], chain)


def test_cost_is_nonnegative_and_respects_floor() -> None:
    chain = build_fair_chain()
    cm = QuotedSpreadCost(CostConfig(half_spread_bps=0.0, min_cost_per_leg=0.25))
    leg = _atm_leg(chain)
    # With zero bps the floor dominates: 0.25 per share * 100 multiplier.
    assert cm.cost([leg], chain) == pytest.approx(0.25 * 100.0)


def test_cost_scales_with_quantity() -> None:
    chain = build_fair_chain()
    cm = QuotedSpreadCost(CostConfig())
    one = OptionLeg("C", chain.nearest_strike(chain.spot), 0, -1, 1.0)
    two = OptionLeg("C", chain.nearest_strike(chain.spot), 0, -2, 1.0)
    assert cm.cost([two], chain) == pytest.approx(2 * cm.cost([one], chain))
