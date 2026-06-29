"""MetricSuite + EvalReport: the distributional scorecard.

A deliberate design choice: the report carries the FULL per-step and per-episode
return arrays, not just summary scalars. The no-edge gate and the Phase-3
"distributional beats expected-value" claim are statements about whole
distributions (tails, dispersion), so the distribution is the primary artifact and
the scalars are conveniences computed from it.

All statistics are pure functions of the collected arrays and deterministic (the
bootstrap CI threads its own seeded ``Generator``), so two eval runs of the same
agent over the same seeds produce byte-identical reports.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optspread.actions.library import FLAT_ACTION_ID, N_ACTIONS


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Frozen result of evaluating one agent over a fixed set of eval seeds."""

    per_step_returns: NDArray[np.float64]  # full distribution, NOT just the mean
    episode_returns: NDArray[np.float64]
    action_frequencies: dict[int, float]
    mean_pnl: float
    pnl_ci: tuple[float, float]
    sharpe: float
    sortino: float
    cvar_95: float
    max_drawdown: float
    turnover: float

    @property
    def flat_frequency(self) -> float:
        """Share of steps the agent chose FLAT (action 0)."""
        return self.action_frequencies.get(FLAT_ACTION_ID, 0.0)


@dataclass(frozen=True, slots=True)
class MetricSuite:
    """Computes an ``EvalReport`` from collected rollout arrays.

    Parameters
    ----------
    trading_days_per_year:
        Annualization factor for Sharpe/Sortino (one step == one trading day).
    cvar_alpha:
        Tail fraction for CVaR (0.05 -> worst 5%).
    ci_alpha:
        Two-sided confidence level complement (0.05 -> 95% CI).
    n_boot, ci_seed:
        Bootstrap resample count and seed (determinism).
    """

    trading_days_per_year: int = 252
    cvar_alpha: float = 0.05
    ci_alpha: float = 0.05
    n_boot: int = 10_000
    ci_seed: int = 0

    def compute(
        self,
        *,
        per_step_returns: NDArray[np.float64],
        episode_returns: NDArray[np.float64],
        action_counts: dict[int, int],
        equity_curves: Sequence[NDArray[np.float64]],
        n_trades: int,
    ) -> EvalReport:
        steps = np.asarray(per_step_returns, dtype=np.float64)
        eps = np.asarray(episode_returns, dtype=np.float64)
        total_actions = max(1, sum(action_counts.values()))
        freqs = {a: action_counts.get(a, 0) / total_actions for a in range(N_ACTIONS)}
        n_episodes = max(1, len(eps))
        return EvalReport(
            per_step_returns=steps,
            episode_returns=eps,
            action_frequencies=freqs,
            mean_pnl=float(eps.mean()) if eps.size else 0.0,
            pnl_ci=self._bootstrap_ci(eps),
            sharpe=self._sharpe(steps),
            sortino=self._sortino(steps),
            cvar_95=self._cvar(steps),
            max_drawdown=self._mean_max_drawdown(equity_curves),
            turnover=n_trades / n_episodes,
        )

    # -- individual statistics -------------------------------------------- #

    def _annualize(self) -> float:
        return float(np.sqrt(self.trading_days_per_year))

    def _sharpe(self, r: NDArray[np.float64]) -> float:
        if r.size < 2:
            return 0.0
        std = r.std(ddof=1)
        return float(r.mean() / std * self._annualize()) if std > 1e-12 else 0.0

    def _sortino(self, r: NDArray[np.float64]) -> float:
        if r.size < 2:
            return 0.0
        downside = np.minimum(r, 0.0)
        dd = np.sqrt((downside**2).mean())
        return float(r.mean() / dd * self._annualize()) if dd > 1e-12 else 0.0

    def _cvar(self, r: NDArray[np.float64]) -> float:
        if r.size == 0:
            return 0.0
        ordered = np.sort(r)
        k = max(1, int(r.size * self.cvar_alpha))
        return float(ordered[:k].mean())

    def _bootstrap_ci(self, eps: NDArray[np.float64]) -> tuple[float, float]:
        if eps.size == 0:
            return (0.0, 0.0)
        if eps.size == 1:
            return (float(eps[0]), float(eps[0]))
        rng = np.random.default_rng(self.ci_seed)
        idx = rng.integers(0, eps.size, size=(self.n_boot, eps.size))
        means = eps[idx].mean(axis=1)
        lo = float(np.percentile(means, 100 * self.ci_alpha / 2))
        hi = float(np.percentile(means, 100 * (1 - self.ci_alpha / 2)))
        return (lo, hi)

    @staticmethod
    def _mean_max_drawdown(curves: Sequence[NDArray[np.float64]]) -> float:
        dds = []
        for curve in curves:
            c = np.asarray(curve, dtype=np.float64)
            if c.size == 0:
                continue
            running_max = np.maximum.accumulate(c)
            dd = (running_max - c) / np.where(running_max != 0, running_max, 1.0)
            dds.append(float(dd.max()))
        return float(np.mean(dds)) if dds else 0.0
