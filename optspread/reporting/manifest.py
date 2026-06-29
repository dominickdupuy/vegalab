"""Exhibit manifest mapping figures/tables to thesis sections."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Exhibit:
    exhibit_id: str
    section: str
    title: str
    synthetic_or_real: str


def validate_manifest(exhibits: list[Exhibit]) -> bool:
    ids = [ex.exhibit_id for ex in exhibits]
    if len(ids) != len(set(ids)):
        return False
    return all(ex.synthetic_or_real in {"synthetic", "real", "mixed"} for ex in exhibits)
