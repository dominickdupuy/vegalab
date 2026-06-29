"""Risk functional tests against known distributions."""

from __future__ import annotations

import numpy as np

from optspread.agents.distributional.risk import RiskMeasure


def test_mean_and_cvar_from_quantiles() -> None:
    values = np.asarray([[-10.0, 1.0, 1.0, 1.0], [0.25, 0.25, 0.25, 0.25]])
    taus = np.asarray([0.125, 0.375, 0.625, 0.875])
    mean_scores = RiskMeasure.mean().from_quantiles(values, taus)
    cvar_scores = RiskMeasure.cvar(0.25).from_quantiles(values, taus)
    assert np.allclose(mean_scores, [-1.75, 0.25])
    assert np.allclose(cvar_scores, [-10.0, 0.25])


def test_cvar_from_samples_uses_worst_tail() -> None:
    samples = np.asarray([-5.0, -4.0, 1.0, 2.0, 3.0, 4.0])
    assert RiskMeasure.mean().from_samples(samples) == np.mean(samples)
    assert RiskMeasure.cvar(2 / 6).from_samples(samples) == -4.5
