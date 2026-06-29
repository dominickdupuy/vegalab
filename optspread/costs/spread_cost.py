"""Quoted-spread transaction cost (Wave-0).

The trader crosses half the quoted bid/ask spread on each leg, each time it is
opened or closed. The quoted half-spread is modeled as a base number of basis
points of the underlying, widened for deeper-OTM strikes (where real option
markets are thinner), and floored at a per-leg minimum. Total cost is the sum
over legs, so it is linear in the number of legs by construction.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from optspread.config import CostConfig
from optspread.instruments.chain import ChainSnapshot
from optspread.instruments.leg import OptionLeg


class QuotedSpreadCost:
    """Implements ``CostModel`` from a ``CostConfig``."""

    def __init__(self, config: CostConfig) -> None:
        self.config = config

    def _leg_half_spread_per_share(self, leg: OptionLeg, chain: ChainSnapshot) -> float:
        cfg = self.config
        base = cfg.half_spread_bps / 1e4 * chain.spot
        # Deeper-OTM strikes quote wider: widen by |log-moneyness|.
        log_moneyness = abs(float(np.log(leg.strike / chain.spot)))
        widen = 1.0 + cfg.otm_widening * log_moneyness
        per_share = base * widen
        return max(per_share, cfg.min_cost_per_leg)

    def cost(self, legs: Sequence[OptionLeg], chain: ChainSnapshot) -> float:
        cfg = self.config
        total = 0.0
        for leg in legs:
            hs = self._leg_half_spread_per_share(leg, chain)
            total += hs * cfg.multiplier * abs(leg.qty)
        return float(total)
