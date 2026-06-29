"""OptionLeg value object: one option position within a structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Right = Literal["C", "P"]


@dataclass(frozen=True, slots=True)
class OptionLeg:
    """A single option leg.

    Immutable. ``qty`` sign encodes direction: ``+1`` long, ``-1`` short (any
    integer magnitude allowed for ratio structures). ``expiry_idx`` selects which
    of the chain's available expiries this leg traded at open (used for the IV
    lookup / calendar spreads). ``expiry_day`` is the ABSOLUTE day index on which
    this leg expires; pricing decays its time-to-expiry as the path advances, so
    a held short option earns theta. ``expiry_day == -1`` means "no absolute
    expiry": the leg is priced at the chain's current tenor for ``expiry_idx``
    (legacy behaviour used by pure unit tests). ``entry_price`` is the per-share
    premium at which the leg was opened.
    """

    right: Right
    strike: float
    expiry_idx: int
    qty: int
    entry_price: float
    expiry_day: int = -1

    def __post_init__(self) -> None:
        if self.right not in ("C", "P"):
            raise ValueError(f"right must be 'C' or 'P', got {self.right!r}")
        if self.qty == 0:
            raise ValueError("qty must be non-zero")
        if self.strike <= 0.0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.expiry_idx < 0:
            raise ValueError(f"expiry_idx must be >= 0, got {self.expiry_idx}")

    @property
    def is_long(self) -> bool:
        return self.qty > 0

    @property
    def is_short(self) -> bool:
        return self.qty < 0

    def intrinsic(self, spot: float) -> float:
        """Per-share intrinsic value (>= 0), ignoring sign/quantity."""
        if self.right == "C":
            return max(spot - self.strike, 0.0)
        return max(self.strike - spot, 0.0)
