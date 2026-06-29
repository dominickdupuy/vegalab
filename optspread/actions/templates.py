"""Spread templates (Strategy pattern).

Each template knows how to *build* its legs from a chain snapshot and a delta
bucket, declares its margin class, and provides an ``analytic_payoff`` ORACLE:
the structure's per-share P&L at a terminal spot assuming every leg settles at
intrinsic. That oracle is what the payoff tests check (and the known structural
bounds are checked independently in the tests).

Strike placement
----------------
- The primary (short or directional) leg's strike is chosen so its BS delta
  matches the requested bucket, then snapped to the chain's strike grid.
- Protective wings are placed a fixed number of grid steps away (``wing_strikes``)
  so the dollar width — and therefore the defined-risk margin — is deterministic.
- ATM-anchored structures (straddle, butterfly body) use the grid strike nearest
  spot and ignore the bucket for the body (the bucket then sizes the wings).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from optspread.actions.margin_class import MarginClass
from optspread.instruments.chain import ChainSnapshot
from optspread.instruments.leg import OptionLeg, Right
from optspread.pricing.strike_solver import strike_from_delta


class SpreadTemplate(ABC):
    """Abstract spread structure."""

    name: str

    @abstractmethod
    def build(self, chain: ChainSnapshot, delta_bucket: float) -> list[OptionLeg]:
        """Construct the legs of this structure on ``chain`` at ``delta_bucket``."""

    @abstractmethod
    def margin_class(self) -> MarginClass:
        """Declare the margin treatment of this structure."""

    def analytic_payoff(self, legs: Sequence[OptionLeg], terminal_spot: float) -> float:
        """Per-share terminal P&L at ``terminal_spot`` (all legs at intrinsic).

        Defined from first principles: each leg contributes
        ``qty * intrinsic(terminal_spot)`` and the up-front net premium
        ``sum(qty * entry_price)`` is subtracted. Subclasses inherit this; the
        tests cross-check it against each structure's known max-profit / max-loss
        bounds, which are independent external knowledge.
        """
        terminal_value = sum(leg.qty * leg.intrinsic(terminal_spot) for leg in legs)
        net_premium = sum(leg.qty * leg.entry_price for leg in legs)
        return float(terminal_value - net_premium)


# --------------------------------------------------------------------------- #
# Helpers shared by concrete templates.
# --------------------------------------------------------------------------- #


def _grid_step(chain: ChainSnapshot) -> float:
    return float(chain.strikes[1] - chain.strikes[0])


def _snapped_delta_strike(
    chain: ChainSnapshot, right: Right, delta_bucket: float, expiry_idx: int
) -> float:
    sigma = chain.iv_at(expiry_idx, chain.spot)
    T = float(chain.expiries[expiry_idx])
    raw = strike_from_delta(right, delta_bucket, chain.spot, chain.r, chain.q, sigma, T)
    return chain.nearest_strike(raw)


def _leg(chain: ChainSnapshot, right: Right, strike: float, expiry_idx: int, qty: int) -> OptionLeg:
    strike = chain.nearest_strike(strike)
    price = chain.price(right, strike, expiry_idx)
    # Stamp the ABSOLUTE expiry day so the leg ages (earns theta) once held.
    return OptionLeg(
        right=right,
        strike=strike,
        expiry_idx=expiry_idx,
        qty=qty,
        entry_price=price,
        expiry_day=chain.expiry_day_for(expiry_idx),
    )


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #


class FlatTemplate(SpreadTemplate):
    """The always-available null action: no legs, no margin."""

    name = "flat"

    def build(self, chain: ChainSnapshot, delta_bucket: float) -> list[OptionLeg]:
        return []

    def margin_class(self) -> MarginClass:
        return MarginClass.FLAT


class LongCallTemplate(SpreadTemplate):
    """Directional debit: buy one call at the bucket delta."""

    name = "long_call"

    def build(self, chain: ChainSnapshot, delta_bucket: float) -> list[OptionLeg]:
        k = _snapped_delta_strike(chain, "C", delta_bucket, 0)
        return [_leg(chain, "C", k, 0, +1)]

    def margin_class(self) -> MarginClass:
        return MarginClass.LONG_ONLY


class LongPutTemplate(SpreadTemplate):
    """Directional debit: buy one put at the bucket delta."""

    name = "long_put"

    def build(self, chain: ChainSnapshot, delta_bucket: float) -> list[OptionLeg]:
        k = _snapped_delta_strike(chain, "P", delta_bucket, 0)
        return [_leg(chain, "P", k, 0, +1)]

    def margin_class(self) -> MarginClass:
        return MarginClass.LONG_ONLY


class _VerticalTemplate(SpreadTemplate):
    """Common machinery for two-strike, single-expiry verticals."""

    def __init__(self, wing_strikes: int = 5) -> None:
        self.wing_strikes = wing_strikes

    def margin_class(self) -> MarginClass:
        return MarginClass.DEFINED_RISK


class BullPutSpreadTemplate(_VerticalTemplate):
    """Credit: short put at bucket delta, long put ``wing_strikes`` below."""

    name = "bull_put_spread"

    def build(self, chain: ChainSnapshot, delta_bucket: float) -> list[OptionLeg]:
        step = _grid_step(chain)
        k_short = _snapped_delta_strike(chain, "P", delta_bucket, 0)
        k_long = k_short - self.wing_strikes * step
        return [_leg(chain, "P", k_short, 0, -1), _leg(chain, "P", k_long, 0, +1)]


class BearCallSpreadTemplate(_VerticalTemplate):
    """Credit: short call at bucket delta, long call ``wing_strikes`` above."""

    name = "bear_call_spread"

    def build(self, chain: ChainSnapshot, delta_bucket: float) -> list[OptionLeg]:
        step = _grid_step(chain)
        k_short = _snapped_delta_strike(chain, "C", delta_bucket, 0)
        k_long = k_short + self.wing_strikes * step
        return [_leg(chain, "C", k_short, 0, -1), _leg(chain, "C", k_long, 0, +1)]


class BullCallSpreadTemplate(_VerticalTemplate):
    """Debit: long call at bucket delta, short call ``wing_strikes`` above."""

    name = "bull_call_spread"

    def build(self, chain: ChainSnapshot, delta_bucket: float) -> list[OptionLeg]:
        step = _grid_step(chain)
        k_long = _snapped_delta_strike(chain, "C", delta_bucket, 0)
        k_short = k_long + self.wing_strikes * step
        return [_leg(chain, "C", k_long, 0, +1), _leg(chain, "C", k_short, 0, -1)]


class BearPutSpreadTemplate(_VerticalTemplate):
    """Debit: long put at bucket delta, short put ``wing_strikes`` below."""

    name = "bear_put_spread"

    def build(self, chain: ChainSnapshot, delta_bucket: float) -> list[OptionLeg]:
        step = _grid_step(chain)
        k_long = _snapped_delta_strike(chain, "P", delta_bucket, 0)
        k_short = k_long - self.wing_strikes * step
        return [_leg(chain, "P", k_long, 0, +1), _leg(chain, "P", k_short, 0, -1)]


class IronCondorTemplate(SpreadTemplate):
    """Defined risk: short strangle at bucket delta + protective wings."""

    name = "iron_condor"

    def __init__(self, wing_strikes: int = 5) -> None:
        self.wing_strikes = wing_strikes

    def build(self, chain: ChainSnapshot, delta_bucket: float) -> list[OptionLeg]:
        step = _grid_step(chain)
        kp_short = _snapped_delta_strike(chain, "P", delta_bucket, 0)
        kc_short = _snapped_delta_strike(chain, "C", delta_bucket, 0)
        kp_long = kp_short - self.wing_strikes * step
        kc_long = kc_short + self.wing_strikes * step
        return [
            _leg(chain, "P", kp_short, 0, -1),
            _leg(chain, "P", kp_long, 0, +1),
            _leg(chain, "C", kc_short, 0, -1),
            _leg(chain, "C", kc_long, 0, +1),
        ]

    def margin_class(self) -> MarginClass:
        return MarginClass.DEFINED_RISK


class IronButterflyTemplate(SpreadTemplate):
    """Defined risk: short ATM straddle + wings ``wing_strikes`` out."""

    name = "iron_butterfly"

    def __init__(self, wing_strikes: int = 5) -> None:
        self.wing_strikes = wing_strikes

    def build(self, chain: ChainSnapshot, delta_bucket: float) -> list[OptionLeg]:
        step = _grid_step(chain)
        k_body = chain.nearest_strike(chain.spot)
        kc_long = k_body + self.wing_strikes * step
        kp_long = k_body - self.wing_strikes * step
        return [
            _leg(chain, "C", k_body, 0, -1),
            _leg(chain, "P", k_body, 0, -1),
            _leg(chain, "C", kc_long, 0, +1),
            _leg(chain, "P", kp_long, 0, +1),
        ]

    def margin_class(self) -> MarginClass:
        return MarginClass.DEFINED_RISK


class ShortStrangleTemplate(SpreadTemplate):
    """Undefined risk: naked short put + short call at the bucket delta."""

    name = "short_strangle"

    def build(self, chain: ChainSnapshot, delta_bucket: float) -> list[OptionLeg]:
        kp = _snapped_delta_strike(chain, "P", delta_bucket, 0)
        kc = _snapped_delta_strike(chain, "C", delta_bucket, 0)
        return [_leg(chain, "P", kp, 0, -1), _leg(chain, "C", kc, 0, -1)]

    def margin_class(self) -> MarginClass:
        return MarginClass.UNDEFINED_RISK


class ShortStraddleTemplate(SpreadTemplate):
    """Undefined risk: naked short ATM call + short ATM put (bucket ignored)."""

    name = "short_straddle"

    def build(self, chain: ChainSnapshot, delta_bucket: float) -> list[OptionLeg]:
        k = chain.nearest_strike(chain.spot)
        return [_leg(chain, "C", k, 0, -1), _leg(chain, "P", k, 0, -1)]

    def margin_class(self) -> MarginClass:
        return MarginClass.UNDEFINED_RISK


class CalendarSpreadTemplate(SpreadTemplate):
    """Defined risk, two expiries: short near call + long far call, same strike.

    NOTE on the oracle: ``analytic_payoff`` (inherited) values BOTH legs at
    intrinsic, i.e. the limiting diagram if the underlying sat at ``terminal_spot``
    through both expiries. The real calendar keeps time value in the far leg at the
    near expiry; that richer marking lives in the MTM path, not in this terminal
    oracle. The tests check the oracle, not live calendar P&L.
    """

    name = "calendar_spread"

    def build(self, chain: ChainSnapshot, delta_bucket: float) -> list[OptionLeg]:
        if chain.n_expiries < 2:
            raise ValueError("calendar spread requires at least two expiries")
        k = _snapped_delta_strike(chain, "C", delta_bucket, 0)
        return [_leg(chain, "C", k, 0, -1), _leg(chain, "C", k, 1, +1)]

    def margin_class(self) -> MarginClass:
        return MarginClass.DEFINED_RISK


class RatioCallSpreadTemplate(SpreadTemplate):
    """Undefined risk (the 'ratio / broken-wing' slot).

    Long one call at the bucket delta, short two calls ``wing_strikes`` above.
    The extra naked short call leaves unbounded upside risk, so this is
    UNDEFINED_RISK (documented per the brief's ratio/broken-wing entry).
    """

    name = "ratio_call_spread"

    def __init__(self, wing_strikes: int = 5) -> None:
        self.wing_strikes = wing_strikes

    def build(self, chain: ChainSnapshot, delta_bucket: float) -> list[OptionLeg]:
        step = _grid_step(chain)
        k_long = _snapped_delta_strike(chain, "C", delta_bucket, 0)
        k_short = k_long + self.wing_strikes * step
        return [_leg(chain, "C", k_long, 0, +1), _leg(chain, "C", k_short, 0, -2)]

    def margin_class(self) -> MarginClass:
        return MarginClass.UNDEFINED_RISK
