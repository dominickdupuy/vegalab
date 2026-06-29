"""CostModel protocol: the swappable transaction-cost engine.

Phase 1 ships a quoted-spread model calibrated by config; a later phase swaps in
one fit to real OptionMetrics bid/ask quotes behind this exact interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from optspread.instruments.chain import ChainSnapshot
from optspread.instruments.leg import OptionLeg


@runtime_checkable
class CostModel(Protocol):
    def cost(self, legs: Sequence[OptionLeg], chain: ChainSnapshot) -> float:
        """Non-negative dollar cost to open OR close ``legs`` on ``chain``."""
        ...
