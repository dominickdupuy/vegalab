"""Phase 5 real-data adapter and evaluation infrastructure."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

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


def test_real_generator_warmup_seeds_history_before_first_tradeable_row() -> None:
    spots = [100.0, 102.0, 101.0, 105.0, 104.0, 107.0]
    rows = [
        SurfaceRow(
            str(i),
            spot,
            IVSurface.flat(sigma=0.2, spot=spot, r=0.0, q=0.0, t=10_000 + i),
        )
        for i, spot in enumerate(spots)
    ]
    replay = RealDataReplay(rows, GBMConfig(n_days=2), warmup_rows=3)

    snapshot = replay.reset(np.random.default_rng(0))

    assert snapshot.chain.spot == spots[3]
    assert snapshot.t == 0
    assert snapshot.chain.t == 0
    assert snapshot.regime_features["realized_vol"] != 0.2
    assert snapshot.regime_features["vrp"] != 0.0
    assert not replay.done

    assert replay.step().chain.spot == spots[4]
    assert not replay.done
    next_snapshot = replay.step()
    assert next_snapshot.chain.spot == spots[5]
    assert next_snapshot.t == 2
    assert replay.done


def test_real_generator_sample_window_uses_rng_and_causal_lead_in() -> None:
    spots = [100.0, 103.0, 101.0, 106.0, 102.0, 109.0, 104.0, 111.0, 107.0, 114.0]
    rows = [
        SurfaceRow(
            str(i),
            spot,
            IVSurface.flat(sigma=0.2 + i * 0.001, spot=spot, r=0.0, q=0.0, t=10_000 + i),
        )
        for i, spot in enumerate(spots)
    ]
    replay = RealDataReplay(rows, GBMConfig(n_days=99), warmup_rows=2, sample_window=4)
    rng = np.random.default_rng(0)
    expected_rng = np.random.default_rng(0)
    first_start = int(expected_rng.integers(2, len(rows) - 4 + 1))
    second_start = int(expected_rng.integers(2, len(rows) - 4 + 1))

    first = replay.reset(rng)

    assert first_start != second_start
    assert first.chain.spot == spots[first_start]
    assert first.t == 0
    assert first.chain.t == 0
    assert first.regime_features["iv_rank"] != 0.5
    assert replay.step().chain.spot == spots[first_start + 1]
    assert replay.step().chain.spot == spots[first_start + 2]
    assert replay.step().chain.spot == spots[first_start + 3]
    assert replay.done

    second = replay.reset(rng)

    assert second.chain.spot == spots[second_start]
    assert second.t == 0
    assert second.chain.t == 0


def test_real_generator_rejects_oversized_sample_window() -> None:
    rows = [
        SurfaceRow(
            str(i),
            5000.0 + i,
            IVSurface.flat(sigma=0.2, spot=5000.0 + i, r=0.0, q=0.0, t=i),
        )
        for i in range(5)
    ]
    with pytest.raises(ValueError, match="sample_window plus warmup_rows"):
        RealDataReplay(rows, warmup_rows=2, sample_window=4)


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
