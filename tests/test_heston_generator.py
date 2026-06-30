"""Wave-2 Heston generator and GV_2 checks."""

from __future__ import annotations

import numpy as np

from optspread.config import GBMConfig
from optspread.eval.generator_validation import validate_wave2_heston
from optspread.market.heston import HestonGenerator
from optspread.market.surface import DEFAULT_DELTA_GRID, DEFAULT_MATURITY_GRID_DAYS

_HESTON_PARAMS = {
    "kappa": 3.0,
    "theta": 0.04,
    "sigma_v": 0.75,
    "rho": -0.80,
    "v0": 0.04,
}


def test_heston_surface_shape_and_positivity() -> None:
    gen = HestonGenerator(GBMConfig(n_days=2), params=_HESTON_PARAMS)
    snapshot = gen.reset(np.random.default_rng(10))

    assert snapshot.surface is not None
    assert snapshot.surface.ivs.shape == (
        DEFAULT_MATURITY_GRID_DAYS.shape[0],
        DEFAULT_DELTA_GRID.shape[0],
    )
    assert np.all(snapshot.surface.ivs > 0.0)
    assert snapshot.chain.ivs.shape[0] == len(gen.config.expiry_days)


def test_heston_negative_rho_produces_downward_skew() -> None:
    gen = HestonGenerator(GBMConfig(n_days=2), params=_HESTON_PARAMS)
    snapshot = gen.reset(np.random.default_rng(11))

    assert snapshot.surface is not None
    low_strike_iv = snapshot.surface.ivs[:, -1]
    high_strike_iv = snapshot.surface.ivs[:, 0]
    assert float(np.mean(low_strike_iv - high_strike_iv)) > 0.001


def test_heston_generator_internals_not_in_observation_features() -> None:
    gen = HestonGenerator(GBMConfig(n_days=2), params=_HESTON_PARAMS)
    snapshot = gen.reset(np.random.default_rng(12))

    hidden_keys = {"kappa", "theta", "sigma_v", "rho", "v0", "variance", "v"}
    assert hidden_keys.isdisjoint(snapshot.regime_features)
    assert gen.current_params["rho"] < 0.0


def test_heston_determinism_same_seed_same_path() -> None:
    first = _roll_heston(seed=13)
    second = _roll_heston(seed=13)

    assert len(first) == len(second)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_allclose(left[0], right[0])
        np.testing.assert_allclose(left[1], right[1])
        assert left[2] == right[2]


def test_gv2_heston_passes_default_priors() -> None:
    result = validate_wave2_heston(
        lambda: HestonGenerator.randomized(GBMConfig(n_days=20)),
        episodes=6,
        seed=50_000,
    )
    assert result.passed, result.reason


def _roll_heston(seed: int) -> list[tuple[float, np.ndarray, dict[str, float]]]:
    gen = HestonGenerator(GBMConfig(n_days=3), params=_HESTON_PARAMS)
    snapshot = gen.reset(np.random.default_rng(seed))
    out = [
        (
            snapshot.chain.spot,
            snapshot.surface.ivs.copy() if snapshot.surface is not None else np.empty((0, 0)),
            dict(snapshot.regime_features),
        )
    ]
    while not gen.done:
        snapshot = gen.step()
        out.append(
            (
                snapshot.chain.spot,
                snapshot.surface.ivs.copy() if snapshot.surface is not None else np.empty((0, 0)),
                dict(snapshot.regime_features),
            )
        )
    return out
