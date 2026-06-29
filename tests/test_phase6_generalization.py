"""Phase 6 held-out-generator helpers."""

from __future__ import annotations

import numpy as np

from optspread.config import GBMConfig
from optspread.evaluation.generalization import graceful_degradation
from optspread.evaluation.structural_distance import wasserstein_1d
from optspread.market.garch import GARCHGenerator


def test_garch_generator_and_generalization_helpers() -> None:
    gen = GARCHGenerator(GBMConfig(n_days=3))
    snap = gen.reset(np.random.default_rng(0))
    assert snap.surface is not None
    snap2 = gen.step()
    assert snap2.t == 1
    assert wasserstein_1d(np.asarray([0.0, 1.0]), np.asarray([1.0, 2.0])) == 1.0
    assert graceful_degradation(1.0, 0.7, max_relative_drop=0.5).passed
    assert not graceful_degradation(1.0, 0.2, max_relative_drop=0.5).passed
