"""MarginModel protocol: the swappable buying-power engine."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from optspread.instruments.chain import ChainSnapshot
from optspread.portfolio.position import Position


@runtime_checkable
class MarginModel(Protocol):
    def margin(self, position: Position, chain: ChainSnapshot) -> float:
        """Non-negative buying-power requirement to hold ``position``."""
        ...
