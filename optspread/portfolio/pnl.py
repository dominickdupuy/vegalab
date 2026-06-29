"""Pure P&L / cash-flow accounting for option-leg structures.

Sign & accounting conventions (all per the contract in CLAUDE.md)
-----------------------------------------------------------------
- ``qty`` is signed (+long / -short); ``entry_price`` and current prices are
  per-share; dollar figures multiply by ``multiplier`` (contract size).
- Opening cash flow ``= -sum(qty * entry_price) * mult``. A credit structure
  (net short premium) puts cash IN immediately (credit-up-front); a debit
  structure takes cash out.
- Position market value ``= sum(qty * price_now) * mult``. Long legs are assets
  (+), short legs are liabilities (-). Equity = cash + position market value, so
  opening a position changes equity only by the transaction cost.
- Realized P&L on close ``= sum(qty * (close - entry)) * mult`` minus costs.
- Unrealized MTM ``= sum(qty * (price_now - entry)) * mult``.

These functions are side-effect-free; the stateful book lives in Portfolio.
"""

from __future__ import annotations

from collections.abc import Sequence

from optspread.instruments.chain import ChainSnapshot
from optspread.instruments.leg import OptionLeg


def net_premium_per_share(legs: Sequence[OptionLeg]) -> float:
    """Signed net premium per share: positive = net debit paid."""
    return float(sum(leg.qty * leg.entry_price for leg in legs))


def opening_cash_flow(legs: Sequence[OptionLeg], multiplier: float) -> float:
    """Cash received at open (credit > 0, debit < 0)."""
    return -net_premium_per_share(legs) * multiplier


def position_market_value(
    legs: Sequence[OptionLeg], chain: ChainSnapshot, multiplier: float
) -> float:
    """Mark-to-market asset value of the legs (short legs are negative)."""
    return float(sum(leg.qty * chain.price_leg(leg) for leg in legs) * multiplier)


def closing_cash_flow(legs: Sequence[OptionLeg], chain: ChainSnapshot, multiplier: float) -> float:
    """Cash from unwinding the legs at current chain prices.

    Equal to the position's market value: sell longs (receive), buy back shorts
    (pay).
    """
    return position_market_value(legs, chain, multiplier)


def unrealized_pnl(legs: Sequence[OptionLeg], chain: ChainSnapshot, multiplier: float) -> float:
    """Mark-to-market P&L vs entry (current leg values minus entry)."""
    return float(
        sum(leg.qty * (chain.price_leg(leg) - leg.entry_price) for leg in legs) * multiplier
    )


def realized_pnl_on_close(
    legs: Sequence[OptionLeg],
    chain: ChainSnapshot,
    multiplier: float,
    cost_open: float = 0.0,
    cost_close: float = 0.0,
) -> float:
    """Realized P&L of a full round trip, net of opening and closing costs."""
    gross = unrealized_pnl(legs, chain, multiplier)
    return gross - cost_open - cost_close
