"""Episode-level mixture generator for earlier-wave rehearsal."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from optspread.market.generator import PriceGenerator
from optspread.market.snapshot import MarketSnapshot


@runtime_checkable
class _HasCurrentParams(Protocol):
    @property
    def current_params(self) -> Mapping[str, float]:
        """Sampled generator parameters, when a concrete generator exposes them."""
        ...


class RehearsalGenerator:
    """Select a primary or earlier-wave sub-generator once per episode."""

    def __init__(
        self,
        *,
        primary: PriceGenerator,
        others: Sequence[PriceGenerator],
        rehearsal_fraction: float,
    ) -> None:
        if not 0.0 <= rehearsal_fraction <= 1.0:
            raise ValueError("rehearsal_fraction must be in [0,1]")
        self._primary = primary
        self._others = tuple(others)
        self._rehearsal_fraction = float(rehearsal_fraction)
        self._active: PriceGenerator | None = None

    def reset(self, rng: np.random.Generator) -> MarketSnapshot:
        """Choose the episode generator, then delegate reset with the same RNG."""
        if self._others and self._rehearsal_fraction > 0.0:
            r = float(rng.random())
            if r < self._rehearsal_fraction:
                idx = int(rng.integers(0, len(self._others)))
                self._active = self._others[idx]
                return self._active.reset(rng)
        self._active = self._primary
        return self._active.reset(rng)

    def step(self) -> MarketSnapshot:
        if self._active is None:
            raise RuntimeError("step() called before reset()")
        return self._active.step()

    @property
    def done(self) -> bool:
        return False if self._active is None else self._active.done

    @property
    def current_params(self) -> dict[str, float]:
        if self._active is None or not isinstance(self._active, _HasCurrentParams):
            return {}
        return dict(self._active.current_params)
