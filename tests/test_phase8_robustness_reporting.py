"""Phase 8 robustness and reporting helpers."""

from __future__ import annotations

import numpy as np

from optspread.attribution.shap_analysis import permutation_importance
from optspread.config import RewardConfig
from optspread.reporting.exhibits import standard_exhibits
from optspread.reporting.limitations import limitations_text
from optspread.reporting.manifest import validate_manifest
from optspread.robustness.ablations import drop_reward_term
from optspread.robustness.alpha_sweep import AlphaPoint, tail_improves_as_alpha_decreases
from optspread.robustness.cost_sensitivity import break_even_cost_multiple
from optspread.robustness.seed_ensemble import summarize


def test_robustness_reporting_helpers() -> None:
    cfg = RewardConfig(cvar_weight=0.5)
    assert drop_reward_term(cfg, "cvar").cvar_weight == 0.0
    assert break_even_cost_multiple([1.0, 2.0, 3.0], [0.2, 0.0, -0.1]) == 2.0
    assert tail_improves_as_alpha_decreases(
        [
            AlphaPoint(alpha=0.1, mean_return=0.1, cvar=-0.5),
            AlphaPoint(alpha=0.5, mean_return=0.2, cvar=-1.0),
        ]
    )
    summary = summarize(np.asarray([1.0, 2.0, 3.0]))
    assert summary.mean == 2.0
    importance = permutation_importance(lambda x: float(x[:, 0].mean()), np.ones((4, 2)))
    assert np.allclose(importance, 0.0)
    exhibits = standard_exhibits()
    assert validate_manifest(exhibits)
    assert "EOD-only" in limitations_text()
