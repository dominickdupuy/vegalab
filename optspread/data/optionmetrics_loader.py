"""Minimal OptionMetrics-style CSV loader.

This is intentionally schema-first and offline-testable. Real WRDS extraction can
write the same columns and reuse this loader without changing the environment.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from optspread.market.surface import DEFAULT_DELTA_GRID, DEFAULT_MATURITY_GRID_DAYS, IVSurface


@dataclass(frozen=True, slots=True)
class SurfaceRow:
    date: str
    spot: float
    surface: IVSurface


def load_surface_csv(path: str | Path, *, r: float = 0.0, q: float = 0.0) -> list[SurfaceRow]:
    """Load rows with columns date, spot, and iv_<maturity_days>_<delta_pct>."""
    rows: list[SurfaceRow] = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for t, raw in enumerate(reader):
            spot = float(raw["spot"])
            ivs = np.asarray(
                [
                    [
                        float(raw[f"iv_{int(maturity)}_{int(delta * 100)}"])
                        for delta in DEFAULT_DELTA_GRID
                    ]
                    for maturity in DEFAULT_MATURITY_GRID_DAYS
                ],
                dtype=np.float64,
            )
            surface = IVSurface(
                deltas=DEFAULT_DELTA_GRID,
                maturity_days=DEFAULT_MATURITY_GRID_DAYS,
                ivs=ivs,
                spot=spot,
                r=r,
                q=q,
                t=t,
            )
            rows.append(SurfaceRow(date=raw["date"], spot=spot, surface=surface))
    if not rows:
        raise ValueError("surface csv contained no rows")
    return rows
