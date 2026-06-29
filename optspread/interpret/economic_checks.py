"""Codified economic-sensibility checks for distilled policies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EconomicCheckResult:
    passed: bool
    reason: str


def check_nontrivial_interaction(rules: list[str]) -> EconomicCheckResult:
    """Require at least one rule containing an explicit interaction/exception."""
    found = any(("except" in rule.lower()) or ("and" in rule.lower()) for rule in rules)
    return EconomicCheckResult(
        passed=found,
        reason="nontrivial interaction present" if found else "no interaction/exception rule found",
    )
