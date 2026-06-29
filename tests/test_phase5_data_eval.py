"""Phase 5 real-data adapter and evaluation infrastructure."""

from __future__ import annotations

from datetime import date

import numpy as np

from optspread.agents.base import FlatAgent
from optspread.config import GBMConfig
from optspread.data.calendar import business_days, third_friday
from optspread.data.hygiene import OptionQuote, filter_quotes, is_clean_quote
from optspread.data.optionmetrics_loader import SurfaceRow
from optspread.data.real_generator import RealDataReplay
from optspread.envs.builder import EnvBundle
from optspread.eval.evaluator import Evaluator
from optspread.eval.metrics import MetricSuite
from optspread.evaluation.deflated_sharpe import deflated_sharpe_ratio
from optspread.evaluation.significance import bootstrap_mean_difference_ci
from optspread.evaluation.walkforward import WalkForwardSplitter
from optspread.market.surface import IVSurface
from optspread.training.env_factory import EnvFactory


def test_hygiene_filters_bad_quotes() -> None:
    clean = OptionQuote("2020-01-02", 100.0, 30, "C", 1.0, 1.2, 0.2, volume=10)
    crossed = OptionQuote("2020-01-02", 100.0, 30, "C", 1.2, 1.1, 0.2)
    assert is_clean_quote(clean)
    assert filter_quotes([clean, crossed]) == [clean]


def test_calendar_helpers() -> None:
    assert third_friday(2026, 6) == date(2026, 6, 19)
    assert len(business_days(date(2026, 6, 1), date(2026, 6, 7))) == 5


def test_real_generator_is_drop_in_for_shared_evaluator() -> None:
    rows = [
        SurfaceRow(
            str(i),
            5000.0 + i,
            IVSurface.flat(sigma=0.2, spot=5000.0 + i, r=0.0, q=0.0, t=i),
        )
        for i in range(6)
    ]
    cfg = GBMConfig(n_days=5)
    factory = EnvFactory(EnvBundle(gbm=cfg, generator_factory=lambda: RealDataReplay(rows, cfg)))
    report = Evaluator(factory, eval_seeds=(1, 2), metrics=MetricSuite()).run(
        FlatAgent(), deterministic=True
    )
    assert report.flat_frequency == 1.0
    assert report.mean_pnl == 0.0


def test_walkforward_purge_embargo_and_statistics() -> None:
    folds = WalkForwardSplitter(train_size=10, test_size=5, purge=2, embargo=1).split(25)
    assert folds[0].train_end == 10
    assert folds[0].test_start == 12
    assert set(folds[0].train_indices).isdisjoint(set(folds[0].test_indices))
    assert deflated_sharpe_ratio(1.0, n_returns=100, n_trials=1) > 0.99
    ci = bootstrap_mean_difference_ci(
        np.asarray([2.0, 2.0, 2.0]),
        np.asarray([1.0, 1.0, 1.0]),
        n_boot=100,
    )
    assert ci[0] == ci[1] == 1.0
