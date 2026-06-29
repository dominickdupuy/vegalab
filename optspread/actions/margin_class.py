"""Margin classification for spread structures."""

from __future__ import annotations

from enum import Enum


class MarginClass(Enum):
    """How a structure's buying-power requirement is computed.

    - FLAT: no position, zero margin.
    - LONG_ONLY: net debit, all long; risk capped at premium paid.
    - DEFINED_RISK: max loss bounded by construction (verticals, condors, flies,
      calendars); margin ~ (width - credit).
    - UNDEFINED_RISK: a naked short leg with theoretically unbounded loss
      (strangles, straddles, ratios); margin is a Reg-T notional requirement.
    """

    FLAT = "FLAT"
    LONG_ONLY = "LONG_ONLY"
    DEFINED_RISK = "DEFINED_RISK"
    UNDEFINED_RISK = "UNDEFINED_RISK"
