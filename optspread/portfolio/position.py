"""Position value object and the stateful Portfolio book.

``Position`` is an immutable record of an open structure. ``Portfolio`` is the one
piece of deliberately-mutable state on the trading side: it holds cash, the
current open position (at most one in Phase 1), reserved margin and the running
realized P&L, and applies the cash-flow accounting from ``pnl.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from optspread.instruments.chain import ChainSnapshot
from optspread.instruments.leg import OptionLeg
from optspread.portfolio import pnl


@dataclass(frozen=True, slots=True)
class Position:
    """An open structure: its legs plus bookkeeping metadata."""

    legs: tuple[OptionLeg, ...]
    action_id: int
    margin: float
    open_day: int
    entry_cash_flow: float

    @property
    def is_empty(self) -> bool:
        return len(self.legs) == 0


@dataclass
class Portfolio:
    """Mutable cash/position book for one episode."""

    multiplier: float
    cash: float = 0.0
    initial_cash: float = 0.0
    position: Position | None = None
    realized_pnl: float = 0.0
    cash_flows: list[float] = field(default_factory=list)

    def reset(self, initial_cash: float) -> None:
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.position = None
        self.realized_pnl = 0.0
        self.cash_flows = []

    # -- queries ----------------------------------------------------------- #

    @property
    def has_position(self) -> bool:
        return self.position is not None and not self.position.is_empty

    @property
    def margin_used(self) -> float:
        return self.position.margin if self.position is not None else 0.0

    def position_value(self, chain: ChainSnapshot) -> float:
        if not self.has_position:
            return 0.0
        assert self.position is not None
        return pnl.position_market_value(self.position.legs, chain, self.multiplier)

    def equity(self, chain: ChainSnapshot) -> float:
        """Cash plus mark-to-market value of any open position."""
        return self.cash + self.position_value(chain)

    def unrealized_pnl(self, chain: ChainSnapshot) -> float:
        if not self.has_position:
            return 0.0
        assert self.position is not None
        return pnl.unrealized_pnl(self.position.legs, chain, self.multiplier)

    # -- mutations --------------------------------------------------------- #

    def open(
        self,
        legs: Sequence[OptionLeg],
        action_id: int,
        margin: float,
        day: int,
        cost: float,
    ) -> None:
        """Open a new position: take in premium (credit) / pay it (debit), pay cost."""
        if self.has_position:
            raise RuntimeError("cannot open: a position is already open")
        legs_t = tuple(legs)
        cf = pnl.opening_cash_flow(legs_t, self.multiplier)
        self.cash += cf - cost
        self.cash_flows.append(cf - cost)
        self.realized_pnl -= cost  # transaction cost is realized immediately
        self.position = Position(
            legs=legs_t,
            action_id=action_id,
            margin=margin,
            open_day=day,
            entry_cash_flow=cf,
        )

    def close(self, chain: ChainSnapshot, cost: float) -> float:
        """Close the open position; return the realized P&L of the round trip."""
        if not self.has_position:
            return 0.0
        assert self.position is not None
        legs = self.position.legs
        cf = pnl.closing_cash_flow(legs, chain, self.multiplier)
        self.cash += cf - cost
        self.cash_flows.append(cf - cost)
        round_trip = pnl.unrealized_pnl(legs, chain, self.multiplier) - cost
        self.realized_pnl += round_trip
        self.position = None
        return round_trip
