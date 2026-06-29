"""Payoff-oracle tests, parametrized over the whole action library.

Two layers of checking:
1. Generic: the template's ``analytic_payoff`` equals an INDEPENDENT
   reconstruction of (sum of per-leg signed intrinsic) minus net premium, across
   a grid of terminal spots. Catches leg-construction and sign bugs.
2. Structural bounds: long-call max loss / breakeven; vertical credit max
   profit & max loss; condor / butterfly max loss. External knowledge, not a
   re-derivation of the same formula.
"""

from __future__ import annotations

import numpy as np
import pytest

from optspread.actions.library import ACTION_LIBRARY
from optspread.actions.margin_class import MarginClass
from optspread.actions.templates import (
    BullPutSpreadTemplate,
    IronCondorTemplate,
    LongCallTemplate,
    LongPutTemplate,
    ShortStrangleTemplate,
)
from tests.conftest import build_fair_chain


def _independent_payoff(legs: list, terminal_spot: float) -> float:
    total = 0.0
    for leg in legs:
        intrinsic = (
            max(terminal_spot - leg.strike, 0.0)
            if leg.right == "C"
            else max(leg.strike - terminal_spot, 0.0)
        )
        total += leg.qty * intrinsic - leg.qty * leg.entry_price
    return total


@pytest.mark.parametrize("spec", ACTION_LIBRARY, ids=[s.name for s in ACTION_LIBRARY])
def test_analytic_payoff_matches_independent(spec) -> None:  # type: ignore[no-untyped-def]
    chain = build_fair_chain()
    legs = spec.template.build(chain, spec.delta_bucket)
    if not legs:  # FLAT
        assert spec.template.analytic_payoff(legs, chain.spot) == 0.0
        return
    grid = np.linspace(chain.spot * 0.7, chain.spot * 1.3, 61)
    for s in grid:
        got = spec.template.analytic_payoff(legs, float(s))
        exp = _independent_payoff(legs, float(s))
        assert got == pytest.approx(exp, abs=1e-9)


@pytest.mark.parametrize("spec", ACTION_LIBRARY, ids=[s.name for s in ACTION_LIBRARY])
def test_every_template_builds_valid_legs(spec) -> None:  # type: ignore[no-untyped-def]
    chain = build_fair_chain()
    legs = spec.template.build(chain, spec.delta_bucket)
    for leg in legs:
        assert leg.strike in set(chain.strikes.tolist())
        assert leg.entry_price >= 0.0
        assert leg.expiry_idx < chain.n_expiries


def test_long_call_max_loss_is_premium_and_breakeven() -> None:
    chain = build_fair_chain()
    legs = LongCallTemplate().build(chain, 0.40)
    (call,) = legs
    prem = call.entry_price
    # Deep OTM at expiry => lose the whole premium.
    assert LongCallTemplate().analytic_payoff(legs, chain.spot * 0.5) == pytest.approx(-prem)
    # Breakeven at K + premium.
    be = call.strike + prem
    assert LongCallTemplate().analytic_payoff(legs, be) == pytest.approx(0.0, abs=1e-6)


def test_long_put_max_loss_is_premium() -> None:
    chain = build_fair_chain()
    legs = LongPutTemplate().build(chain, 0.40)
    (put,) = legs
    assert LongPutTemplate().analytic_payoff(legs, chain.spot * 1.5) == pytest.approx(
        -put.entry_price
    )


def test_vertical_credit_bounds() -> None:
    chain = build_fair_chain()
    tmpl = BullPutSpreadTemplate(wing_strikes=5)
    legs = tmpl.build(chain, 0.16)
    short_put = next(leg for leg in legs if leg.is_short)
    long_put = next(leg for leg in legs if leg.is_long)
    width = short_put.strike - long_put.strike
    credit = -sum(leg.qty * leg.entry_price for leg in legs)  # net premium received
    assert credit > 0  # it's a credit spread
    # Max profit = credit (spot well above the short strike).
    assert tmpl.analytic_payoff(legs, chain.spot * 1.3) == pytest.approx(credit, abs=1e-6)
    # Max loss = -(width - credit) (spot below the long strike).
    assert tmpl.analytic_payoff(legs, chain.spot * 0.5) == pytest.approx(
        -(width - credit), abs=1e-6
    )


def test_iron_condor_max_loss_is_wing_minus_credit() -> None:
    chain = build_fair_chain()
    tmpl = IronCondorTemplate(wing_strikes=5)
    legs = tmpl.build(chain, 0.16)
    credit = -sum(leg.qty * leg.entry_price for leg in legs)
    assert credit > 0
    # Width of one (symmetric) wing.
    put_short = next(leg for leg in legs if leg.right == "P" and leg.is_short)
    put_long = next(leg for leg in legs if leg.right == "P" and leg.is_long)
    width = put_short.strike - put_long.strike
    max_loss = tmpl.analytic_payoff(legs, chain.spot * 0.5)
    assert max_loss == pytest.approx(-(width - credit), abs=1e-6)


def test_short_strangle_is_credit_and_undefined() -> None:
    chain = build_fair_chain()
    tmpl = ShortStrangleTemplate()
    legs = tmpl.build(chain, 0.16)
    credit = -sum(leg.qty * leg.entry_price for leg in legs)
    assert credit > 0
    assert tmpl.margin_class() is MarginClass.UNDEFINED_RISK
    # Loss grows without bound as spot falls (undefined risk).
    loss_near = tmpl.analytic_payoff(legs, chain.spot * 0.8)
    loss_far = tmpl.analytic_payoff(legs, chain.spot * 0.5)
    assert loss_far < loss_near < credit


def test_margin_classes_present() -> None:
    classes = {spec.template.margin_class() for spec in ACTION_LIBRARY}
    assert MarginClass.FLAT in classes
    assert MarginClass.LONG_ONLY in classes
    assert MarginClass.DEFINED_RISK in classes
    assert MarginClass.UNDEFINED_RISK in classes
