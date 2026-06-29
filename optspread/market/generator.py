"""PriceGenerator protocol: the swappable market engine.

Phase 1 ships the fair-IV GBM generator; later phases swap in Heston/Bates/
regime-switching behind this exact interface. The env depends ONLY on this
Protocol, never on a concrete generator.

NO-LOOK-AHEAD CONTRACT: ``step`` is the ONLY method that advances the underlying
path. ``reset`` returns the day-0 snapshot; each ``step`` returns the snapshot at
the next close. Nothing the env can observe at time t depends on a draw from t+1.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from optspread.market.snapshot import MarketSnapshot


@runtime_checkable
class PriceGenerator(Protocol):
    def reset(self, rng: np.random.Generator) -> MarketSnapshot:
        """Reset to day 0 using ``rng`` and return the initial snapshot."""
        ...

    def step(self) -> MarketSnapshot:
        """Advance exactly one trading day; return the new snapshot."""
        ...

    @property
    def done(self) -> bool:
        """True once the configured horizon has been reached."""
        ...
