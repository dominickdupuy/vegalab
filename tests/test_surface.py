"""IVSurface grid integrity and chain derivation."""

from __future__ import annotations

import numpy as np

from optspread.market.surface import IVSurface


def test_flat_surface_interpolates_and_builds_chain() -> None:
    surface = IVSurface.flat(sigma=0.22, spot=5000.0, r=0.05, q=0.0, t=3)
    assert surface.iv_at_delta_maturity(0.5, 21.0) == 0.22
    chain = surface.to_chain(
        expiry_days=(21, 42),
        n_strikes_each_side=4,
        strike_spacing_pct=0.01,
    )
    assert chain.ivs.shape == (2, 9)
    assert np.allclose(chain.ivs, 0.22)
    assert chain.t == 3


def test_surface_rejects_bad_shape() -> None:
    try:
        IVSurface(
            deltas=np.asarray([0.1, 0.5]),
            maturity_days=np.asarray([10.0, 20.0]),
            ivs=np.ones((1, 2)),
            spot=100.0,
            r=0.0,
            q=0.0,
            t=0,
        )
    except ValueError as exc:
        assert "ivs shape" in str(exc)
    else:
        raise AssertionError("expected shape validation failure")
