"""Option quote hygiene filters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OptionQuote:
    date: str
    strike: float
    expiry_days: int
    right: str
    best_bid: float
    best_offer: float
    implied_vol: float | None
    volume: int = 0
    open_interest: int = 0


def is_clean_quote(
    quote: OptionQuote,
    *,
    min_volume: int = 0,
    min_open_interest: int = 0,
) -> bool:
    """Return whether a raw option quote is usable for execution/cost calibration."""
    if quote.best_bid <= 0.0:
        return False
    if quote.best_offer <= quote.best_bid:
        return False
    if quote.implied_vol is None or quote.implied_vol <= 0.0:
        return False
    if quote.volume < min_volume:
        return False
    return quote.open_interest >= min_open_interest


def filter_quotes(
    quotes: Iterable[OptionQuote],
    *,
    min_volume: int = 0,
    min_open_interest: int = 0,
) -> list[OptionQuote]:
    """Apply hygiene filters without mutating input rows."""
    return [
        quote
        for quote in quotes
        if is_clean_quote(
            quote,
            min_volume=min_volume,
            min_open_interest=min_open_interest,
        )
    ]
