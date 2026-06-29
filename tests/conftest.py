"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from optspread.instruments.chain import ChainSnapshot


def build_fair_chain(
    spot: float = 5000.0,
    sigma: float = 0.20,
    r: float = 0.05,
    q: float = 0.0,
    n_each_side: int = 40,
    spacing_pct: float = 0.01,
    expiries: tuple[float, ...] = (21 / 252, 42 / 252),
    t: int = 0,
) -> ChainSnapshot:
    """A flat-IV (fair) chain centered on spot, matching Wave-0 conventions."""
    step = spot * spacing_pct
    lo = spot - n_each_side * step
    strikes = np.round(lo + np.arange(2 * n_each_side + 1) * step, 6)
    exp = np.array(expiries, dtype=np.float64)
    ivs = np.full((exp.shape[0], strikes.shape[0]), sigma, dtype=np.float64)
    return ChainSnapshot(
        strikes=strikes.astype(np.float64),
        ivs=ivs,
        expiries=exp,
        spot=spot,
        r=r,
        q=q,
        t=t,
    )


@pytest.fixture
def fair_chain() -> ChainSnapshot:
    return build_fair_chain()
