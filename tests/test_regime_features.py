"""Causal regime-feature contract tests."""

from __future__ import annotations

import numpy as np

from optspread.config import GBMConfig
from optspread.features.regime_features import build_regime_features
from optspread.market.gbm import GBMGenerator
from optspread.market.snapshot import REGIME_FEATURE_KEYS
from optspread.market.surface import DEFAULT_DELTA_GRID, DEFAULT_MATURITY_GRID_DAYS, IVSurface


def test_regime_features_do_not_use_future_values() -> None:
    rng = np.random.default_rng(12_345)
    log_returns = rng.normal(loc=0.0001, scale=0.01, size=100).tolist()
    iv_history = np.clip(
        0.22 + np.cumsum(rng.normal(loc=0.0, scale=0.002, size=100)),
        0.05,
        1.00,
    ).tolist()
    prefix_len = 48
    surface = _nonflat_surface()

    baseline = build_regime_features(
        surface=surface,
        log_returns=log_returns[:prefix_len],
        iv_history=iv_history[:prefix_len],
    )
    extended_logs = log_returns[:prefix_len] + rng.normal(0.0, 0.03, size=20).tolist()
    extended_ivs = iv_history[:prefix_len] + rng.uniform(0.10, 0.50, size=20).tolist()
    recomputed = build_regime_features(
        surface=surface,
        log_returns=extended_logs[:prefix_len],
        iv_history=extended_ivs[:prefix_len],
    )

    assert tuple(baseline) == REGIME_FEATURE_KEYS
    assert len(REGIME_FEATURE_KEYS) == 24
    assert baseline == recomputed

    mutated_logs = list(extended_logs)
    mutated_ivs = list(extended_ivs)
    mutated_logs[prefix_len + 3] = -9.0
    mutated_ivs[prefix_len + 3] = 9.0
    mutated_recomputed = build_regime_features(
        surface=surface,
        log_returns=mutated_logs[:prefix_len],
        iv_history=mutated_ivs[:prefix_len],
    )

    assert baseline == mutated_recomputed


def test_regime_features_all_keys_present_and_finite_for_rich_and_sparse_history() -> None:
    rng = np.random.default_rng(202)
    surface = _nonflat_surface()
    rich = build_regime_features(
        surface=surface,
        log_returns=rng.normal(0.0, 0.01, size=80).tolist(),
        iv_history=(0.20 + np.cumsum(rng.normal(0.0, 0.001, size=80))).tolist(),
    )
    empty = build_regime_features(surface=surface, log_returns=[], iv_history=[])
    length_one = build_regime_features(surface=surface, log_returns=[0.001], iv_history=[0.20])

    _assert_feature_schema(rich)
    _assert_feature_schema(empty)
    _assert_feature_schema(length_one)


def test_regime_features_hidden_internals_are_disjoint() -> None:
    features = build_regime_features(
        surface=_nonflat_surface(),
        log_returns=[0.001, -0.002, 0.003, -0.001],
        iv_history=[0.20, 0.21, 0.205, 0.215],
    )
    hidden = {"kappa", "theta", "sigma_v", "rho", "v0", "vrp_vol_premium", "sigma", "variance", "v"}

    assert hidden.isdisjoint(features)


def test_regime_features_flat_surface_shape_terms_are_degenerate() -> None:
    sigma = 0.231
    surface = IVSurface.flat(sigma=sigma, spot=5000.0, r=0.04, q=0.0, t=0)
    features = build_regime_features(
        surface=surface,
        log_returns=[0.001, -0.002, 0.003],
        iv_history=[sigma, sigma, sigma],
    )

    assert features["skew"] == 0.0
    assert features["smile_curvature"] == 0.0
    assert features["term_curvature"] == 0.0
    assert features["skew_term"] == 0.0
    assert abs(features["atm_iv"] - sigma) < 1.0e-9


def test_regime_features_known_value_sanity() -> None:
    log_returns = [0.001] * 20 + [-0.002] * 20 + [-0.030, -0.025, -0.020, -0.015]
    features = build_regime_features(
        surface=_nonflat_surface(),
        log_returns=log_returns,
        iv_history=[0.20 + 0.001 * idx for idx in range(len(log_returns))],
    )

    assert features["skew"] > 0.0
    assert features["path_drawdown"] > 0.0
    assert features["realized_skew"] < 0.0


def test_wave0_gbm_generator_emits_all_regime_feature_keys() -> None:
    gen = GBMGenerator(GBMConfig(n_days=4))
    snapshot = gen.reset(np.random.default_rng(99))
    snapshots = [snapshot]
    while not gen.done:
        snapshots.append(gen.step())

    for snap in snapshots:
        assert snap.surface is not None
        _assert_feature_schema(snap.regime_features)
        assert "sigma" not in snap.regime_features


def _nonflat_surface() -> IVSurface:
    deltas = DEFAULT_DELTA_GRID.astype(np.float64)
    maturities = DEFAULT_MATURITY_GRID_DAYS.astype(np.float64)
    rows = []
    for maturity in maturities:
        base = 0.18 + 0.00008 * maturity
        skew = 0.04 * (deltas - 0.50)
        curvature = 0.06 * (deltas - 0.50) ** 2
        rows.append(base + skew + curvature)
    return IVSurface(
        deltas=deltas,
        maturity_days=maturities,
        ivs=np.asarray(rows, dtype=np.float64),
        spot=5000.0,
        r=0.04,
        q=0.0,
        t=0,
    )


def _assert_feature_schema(features: dict[str, float]) -> None:
    assert tuple(features) == REGIME_FEATURE_KEYS
    assert set(features) == set(REGIME_FEATURE_KEYS)
    for value in features.values():
        assert isinstance(value, float)
        assert np.isfinite(value)
