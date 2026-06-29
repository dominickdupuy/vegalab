"""CBOE index mechanics as action-library mappings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CBOEBenchmarkSpec:
    name: str
    action_id: int
    description: str


CBOE_BENCHMARKS: dict[str, CBOEBenchmarkSpec] = {
    "CNDR": CBOEBenchmarkSpec("CNDR", 11, "short ~20-delta strangle with long wings"),
    "BFLY": CBOEBenchmarkSpec("BFLY", 13, "short ATM straddle with OTM wings"),
    "PUT": CBOEBenchmarkSpec("PUT", 5, "cash-secured put-write proxy via bull put spread"),
    "BXM": CBOEBenchmarkSpec("BXM", 7, "covered-call proxy via bear call spread"),
}
