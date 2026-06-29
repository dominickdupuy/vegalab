"""Reg-T-style margin: defined- vs undefined-risk sizing.

The requirement is inferred *structurally* from the legs (it does not need to be
told the template), so it stays correct if a template's construction changes:

- FLAT (no legs): zero.
- Every short leg covered by a same-right long (calls capped by a higher-strike
  long, puts capped by a lower-strike long) => DEFINED RISK. Margin is the
  worst-case terminal loss, i.e. ``max(0, -min_spot payoff) * multiplier``. For a
  vertical this reduces exactly to ``(width - credit) * multiplier``; for a pure
  debit structure it is the premium paid.
- Any uncovered short leg => UNDEFINED RISK. Each naked short carries a Reg-T
  notional requirement (a fraction of underlying notional, floored), which is
  materially larger than the defined-risk equivalent.
"""

from __future__ import annotations

from collections.abc import Sequence

from optspread.config import MarginConfig
from optspread.instruments.chain import ChainSnapshot
from optspread.instruments.leg import OptionLeg
from optspread.portfolio.position import Position


def _expand_units(legs: Sequence[OptionLeg], right: str) -> tuple[list[float], list[float]]:
    """Return (long_strikes, short_strikes) as unit-contract lists for one right."""
    longs: list[float] = []
    shorts: list[float] = []
    for leg in legs:
        if leg.right != right:
            continue
        for _ in range(abs(leg.qty)):
            (longs if leg.is_long else shorts).append(leg.strike)
    return longs, shorts


def _count_naked_shorts(legs: Sequence[OptionLeg]) -> int:
    """Number of short unit contracts not covered by a protective long.

    A short call needs a long call at strike >= its strike; a short put needs a
    long put at strike <= its strike. Greedy tightest-cover matching.
    """
    naked = 0
    # Calls: cover each short with the lowest-strike long that is >= short strike.
    long_c, short_c = _expand_units(legs, "C")
    long_c.sort()
    for ks in sorted(short_c):
        cover = next((kl for kl in long_c if kl >= ks), None)
        if cover is None:
            naked += 1
        else:
            long_c.remove(cover)
    # Puts: cover each short with the highest-strike long that is <= short strike.
    long_p, short_p = _expand_units(legs, "P")
    long_p.sort(reverse=True)
    for ks in sorted(short_p, reverse=True):
        cover = next((kl for kl in long_p if kl <= ks), None)
        if cover is None:
            naked += 1
        else:
            long_p.remove(cover)
    return naked


def _terminal_payoff(legs: Sequence[OptionLeg], spot: float) -> float:
    value = sum(leg.qty * leg.intrinsic(spot) for leg in legs)
    premium = sum(leg.qty * leg.entry_price for leg in legs)
    return float(value - premium)


def _worst_case_loss_per_share(legs: Sequence[OptionLeg]) -> float:
    """Max loss per share for a defined-risk structure (>= 0).

    The terminal payoff is piecewise-linear with kinks only at strikes, so its
    minimum over [0, inf) is attained at a strike or at the boundaries.
    """
    strikes = sorted({leg.strike for leg in legs})
    candidates = [0.0, *strikes, 2.0 * max(strikes)]
    worst = min(_terminal_payoff(legs, s) for s in candidates)
    return max(0.0, -worst)


class RegTStyleMargin:
    """Implements ``MarginModel`` from a ``MarginConfig``."""

    def __init__(self, config: MarginConfig) -> None:
        self.config = config

    def _naked_requirement_per_short(self, spot: float) -> float:
        cfg = self.config
        standard = cfg.naked_margin_rate * spot
        floor = cfg.naked_floor_rate * spot
        return max(standard, floor) * cfg.multiplier

    def margin(self, position: Position, chain: ChainSnapshot) -> float:
        legs = position.legs
        if not legs:  # FLAT
            return 0.0
        naked = _count_naked_shorts(legs)
        if naked == 0:
            # Defined risk (or pure long debit): worst-case terminal loss.
            loss = _worst_case_loss_per_share(legs) * self.config.defined_risk_multiplier
            return loss * self.config.multiplier
        # Undefined risk: Reg-T notional per naked short, plus the defined-risk
        # margin of any covered remainder (here dominated by the naked term).
        return naked * self._naked_requirement_per_short(chain.spot)
