"""No-edge gate logic on synthetic EvalReports: pass zero-edge, fail planted-edge."""

from __future__ import annotations

import numpy as np

from optspread.eval.metrics import EvalReport
from optspread.eval.no_edge_gate import evaluate_no_edge


def _report(*, flat_freq: float, ci: tuple[float, float], mean: float) -> EvalReport:
    """Hand-built report exercising only the fields the gate reads."""
    return EvalReport(
        per_step_returns=np.zeros(1),
        episode_returns=np.zeros(1),
        action_frequencies={0: flat_freq, 14: 1.0 - flat_freq},
        mean_pnl=mean,
        pnl_ci=ci,
        sharpe=0.0,
        sortino=0.0,
        cvar_95=0.0,
        max_drawdown=0.0,
        turnover=0.0,
    )


def test_zero_edge_flat_dominant_passes() -> None:
    rep = _report(flat_freq=0.9, ci=(-50.0, 40.0), mean=-2.0)
    res = evaluate_no_edge(rep, with_costs=False, flat_threshold=0.8)
    assert res.passed
    assert "no systematic edge" in res.reason


def test_planted_positive_edge_fails() -> None:
    # CI strictly above zero: the agent reliably makes money -> Phase 1 bug.
    rep = _report(flat_freq=0.95, ci=(120.0, 300.0), mean=210.0)
    res = evaluate_no_edge(rep, with_costs=False, flat_threshold=0.8)
    assert not res.passed
    assert "positive edge" in res.reason


def test_low_flat_frequency_fails_risk_adjusted_gate() -> None:
    # Not profitable, but the risk-adjusted gate also demands FLAT dominance.
    rep = _report(flat_freq=0.30, ci=(-80.0, 20.0), mean=-30.0)
    res = evaluate_no_edge(rep, with_costs=True, flat_threshold=0.8)
    assert not res.passed
    assert "FLAT frequency" in res.reason


def test_pure_pnl_ablation_ignores_flatness() -> None:
    # Part (b): no FLAT preference expected; only profitability binds.
    indifferent = _report(flat_freq=0.05, ci=(-30.0, 30.0), mean=0.5)
    assert evaluate_no_edge(indifferent, with_costs=False, flat_threshold=0.0).passed

    profitable = _report(flat_freq=0.05, ci=(10.0, 50.0), mean=30.0)
    assert not evaluate_no_edge(profitable, with_costs=False, flat_threshold=0.0).passed
