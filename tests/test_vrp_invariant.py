"""Wave-1 GBM+VRP generator invariant."""

from __future__ import annotations

import numpy as np
import pytest

from optspread.config import GBMConfig
from optspread.eval.generator_validation import validate_wave1_vrp
from optspread.market.gbm_vrp import GBMVRPGenerator


def test_wave1_vrp_feature_is_positive_on_average() -> None:
    result = validate_wave1_vrp(
        lambda: GBMVRPGenerator(GBMConfig(n_days=20), sigma=0.16, vrp_vol_premium=0.05),
        episodes=8,
        threshold=0.001,
    )
    assert result.passed, result.reason


def test_gbm_vrp_implied_vol_exceeds_realized_sigma_initially() -> None:
    gen = GBMVRPGenerator(GBMConfig(n_days=5), sigma=0.16, vrp_vol_premium=0.05)
    snap = gen.reset(np.random.default_rng(4))
    assert snap.surface is not None
    assert snap.surface.iv_at_delta_maturity(0.5, 21.0) == pytest.approx(0.21)
    assert snap.regime_features["vrp"] > 0.0
