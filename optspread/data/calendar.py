"""Trading calendar and fold-boundary helpers."""

from __future__ import annotations

from datetime import date, timedelta


def business_days(start: date, end: date) -> list[date]:
    """Inclusive Monday-Friday calendar, sufficient for tests and offline fixtures."""
    if end < start:
        raise ValueError("end must be >= start")
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def third_friday(year: int, month: int) -> date:
    """Return the standard monthly SPX third-Friday expiration date."""
    first = date(year, month, 1)
    offset = (4 - first.weekday()) % 7
    first_friday = first + timedelta(days=offset)
    return first_friday + timedelta(days=14)
