"""MarketSnapshot: everything the env sees at one step.

A ChainSnapshot plus a dict of regime features (VRP, IV-rank, momentum, ...). In
Wave-0 most regime features are degenerate constants, but the schema is fixed now
so the observation never changes when richer generators arrive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from optspread.instruments.chain import ChainSnapshot
from optspread.market.surface import IVSurface

# Canonical, ordered regime-feature schema. The ObservationBuilder reads these
# keys in this order; generators must emit all of them (constants are fine).
REGIME_FEATURE_KEYS: tuple[str, ...] = (
    "trailing_momentum",  # standardized recent log-return drift
    "realized_vol",  # trailing realized vol estimate
    "vrp",  # variance-risk premium (IV^2 - RV^2); ~0 in Wave-0
    "iv_rank",  # 0..1; constant 0.5 in Wave-0
    "term_slope",  # far_iv - near_iv; ~0 in Wave-0
)


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    chain: ChainSnapshot
    t: int
    regime_features: dict[str, float] = field(default_factory=dict)
    surface: IVSurface | None = None

    def __post_init__(self) -> None:
        missing = [k for k in REGIME_FEATURE_KEYS if k not in self.regime_features]
        if missing:
            raise ValueError(f"regime_features missing keys: {missing}")

    def feature_vector(self) -> list[float]:
        """Regime features in the canonical schema order."""
        return [float(self.regime_features[k]) for k in REGIME_FEATURE_KEYS]
