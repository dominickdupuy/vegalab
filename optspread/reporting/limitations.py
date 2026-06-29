"""Codified limitations and negative-result collection."""

from __future__ import annotations

DEFAULT_LIMITATIONS: tuple[str, ...] = (
    "EOD-only daily decisions; no intraday risk management.",
    "Single-underlying SPX scope.",
    "Synthetic-realism cap even with domain randomization.",
    "Few real tail events in one historical path.",
    "Static-CVaR time-inconsistency under dynamic bootstrapping.",
    "Distilled regime map is lossy and must report fidelity.",
)


def limitations_text(extra: list[str] | None = None) -> str:
    items = list(DEFAULT_LIMITATIONS)
    if extra:
        items.extend(extra)
    return "\n".join(f"- {item}" for item in items)
