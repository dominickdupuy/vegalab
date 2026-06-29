"""Phase 7 distillation and regime-map helpers."""

from __future__ import annotations

import numpy as np

from optspread.interpret.clustering import kmeans
from optspread.interpret.coverage_sampler import latin_hypercube
from optspread.interpret.economic_checks import check_nontrivial_interaction
from optspread.interpret.fidelity import action_agreement, value_regret
from optspread.interpret.regime_map import build_regime_cells
from optspread.interpret.rollout_logger import RolloutDataset, append_rollouts
from optspread.interpret.viper import fit_weighted_stump


def test_distillation_map_helpers() -> None:
    x = np.asarray([[0.0], [0.1], [1.0], [1.1]], dtype=np.float64)
    y = np.asarray([0, 0, 1, 1], dtype=np.int64)
    stump = fit_weighted_stump(x, y)
    pred = stump.predict(x)
    assert action_agreement(y, pred) == 1.0
    labels, centers = kmeans(np.c_[x[:, 0], x[:, 0]], k=2, seed=1)
    assert labels.shape == (4,)
    assert centers.shape == (2, 2)
    cells = build_regime_cells(labels, y, np.asarray([0.0, 1.0, 2.0, 3.0]))
    assert sum(cell.count for cell in cells) == 4
    regret = value_regret(np.asarray([[1.0, 0.0], [0.0, 2.0]]), np.asarray([0, 0]))
    assert regret == 1.0
    sample = latin_hypercube(5, 3, np.random.default_rng(0))
    assert sample.shape == (5, 3)
    assert check_nontrivial_interaction(["high VRP and low jump -> credit"]).passed
    ds = append_rollouts(
        [
            RolloutDataset(
                np.zeros((1, 2), dtype=np.float32),
                np.asarray([0], dtype=np.int64),
                np.asarray([0.0]),
            )
        ]
    )
    assert ds.observations.shape == (1, 2)
