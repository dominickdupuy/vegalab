"""Phase 5 baselines and sim-to-real diagnostics."""

from __future__ import annotations

import numpy as np

from optspread.baselines.buy_and_hold import buy_and_hold_returns
from optspread.baselines.cboe_indices import CBOE_BENCHMARKS
from optspread.baselines.vrp_heuristic import vrp_heuristic_action
from optspread.eval.sim2real import feature_coverage
from optspread.finetune.finetuner import make_finetune_plan


def test_baselines_and_gap_helpers() -> None:
    returns = buy_and_hold_returns(np.asarray([100.0, 101.0, 99.0]))
    assert np.allclose(returns, [0.01, -2 / 101])
    assert vrp_heuristic_action({"vrp": 0.1, "iv_rank": 0.9}) != 0
    assert vrp_heuristic_action({"vrp": -0.1, "iv_rank": 0.9}) == 0
    assert "CNDR" in CBOE_BENCHMARKS
    coverage = feature_coverage(
        np.asarray([[0.5, 1.5], [2.0, 2.0]]),
        np.asarray([[0.0, 1.0], [1.0, 3.0]]),
    )
    assert coverage["feature_0"] == 0.5
    assert coverage["feature_1"] == 1.0
    assert not make_finetune_plan(has_validation_fold=False).allowed
    assert make_finetune_plan(has_validation_fold=True).allowed
