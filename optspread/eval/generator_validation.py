"""Generator-validation gates that run before agent training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from optspread.market.generator import PriceGenerator
from optspread.market.snapshot import MarketSnapshot


@dataclass(frozen=True, slots=True)
class GeneratorValidationResult:
    passed: bool
    statistic: float
    threshold: float
    reason: str


def validate_wave1_vrp(
    make_generator: Callable[[], PriceGenerator],
    *,
    episodes: int = 16,
    threshold: float = 0.0005,
    seed: int = 40_000,
) -> GeneratorValidationResult:
    """GV_1: mean IV^2 - realized^2 is positive."""
    vrps: list[float] = []
    for i in range(episodes):
        gen = make_generator()
        snapshot = gen.reset(np.random.default_rng(seed + i))
        while not gen.done:
            vrps.append(float(snapshot.regime_features["vrp"]))
            snapshot = gen.step()
        vrps.append(float(snapshot.regime_features["vrp"]))
    mean_vrp = float(np.mean(vrps)) if vrps else 0.0
    passed = mean_vrp > threshold
    return GeneratorValidationResult(
        passed=passed,
        statistic=mean_vrp,
        threshold=threshold,
        reason=(
            f"mean VRP {mean_vrp:.6f} > {threshold:.6f}"
            if passed
            else f"mean VRP {mean_vrp:.6f} <= {threshold:.6f}"
        ),
    )


def validate_wave2_heston(
    make_generator: Callable[[], PriceGenerator],
    *,
    episodes: int = 16,
    skew_threshold: float = 0.001,
    vrp_threshold: float = 0.0005,
    iv_rank_std_threshold: float = 0.05,
    term_slope_std_threshold: float = 0.001,
    atm_lag1_threshold: float = 0.10,
    atm_change_ratio_threshold: float = 0.50,
    seed: int = 50_000,
) -> GeneratorValidationResult:
    """GV_2: Heston surfaces exhibit negative-rho skew, SV dynamics, and VRP."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    skews: list[float] = []
    vrps: list[float] = []
    iv_ranks: list[float] = []
    term_slopes: list[float] = []
    atm_paths: list[list[float]] = []

    for i in range(episodes):
        gen = make_generator()
        snapshot = gen.reset(np.random.default_rng(seed + i))
        episode_atm: list[float] = []
        while not gen.done:
            _collect_wave2_snapshot(
                snapshot,
                skews=skews,
                vrps=vrps,
                iv_ranks=iv_ranks,
                term_slopes=term_slopes,
                atms=episode_atm,
            )
            snapshot = gen.step()
        _collect_wave2_snapshot(
            snapshot,
            skews=skews,
            vrps=vrps,
            iv_ranks=iv_ranks,
            term_slopes=term_slopes,
            atms=episode_atm,
        )
        atm_paths.append(episode_atm)

    mean_skew = _safe_mean(skews)
    mean_vrp = _safe_mean(vrps)
    iv_rank_std = _safe_std(iv_ranks)
    term_slope_std = _safe_std(term_slopes)
    atm_lag1 = _safe_mean([_lag_autocorr(path, 1) for path in atm_paths])
    atm_lag5 = _safe_mean([_lag_autocorr(path, 5) for path in atm_paths])
    atm_change_ratio = _safe_mean([_change_variance_ratio(path) for path in atm_paths])

    skew_pass = mean_skew > skew_threshold
    vrp_pass = mean_vrp > vrp_threshold
    variation_pass = (
        iv_rank_std > iv_rank_std_threshold and term_slope_std > term_slope_std_threshold
    )
    mean_reversion_pass = (
        atm_lag1 > atm_lag1_threshold and atm_change_ratio < atm_change_ratio_threshold
    )
    passed = skew_pass and vrp_pass and variation_pass and mean_reversion_pass
    reason = (
        f"skew90-10={mean_skew:.6f} (>{skew_threshold:.6f}), "
        f"iv_rank_std={iv_rank_std:.4f} (>{iv_rank_std_threshold:.4f}), "
        f"term_slope_std={term_slope_std:.6f} (>{term_slope_std_threshold:.6f}), "
        f"atm_acf1={atm_lag1:.3f} (>{atm_lag1_threshold:.3f}), "
        f"atm_acf5={atm_lag5:.3f}, "
        f"atm_change_var_ratio={atm_change_ratio:.3f} (<{atm_change_ratio_threshold:.3f}), "
        f"mean_vrp={mean_vrp:.6f} (>{vrp_threshold:.6f})"
    )
    return GeneratorValidationResult(
        passed=passed,
        statistic=mean_skew,
        threshold=skew_threshold,
        reason=reason,
    )


def _collect_wave2_snapshot(
    snapshot: MarketSnapshot,
    *,
    skews: list[float],
    vrps: list[float],
    iv_ranks: list[float],
    term_slopes: list[float],
    atms: list[float],
) -> None:
    surface = snapshot.surface
    if surface is None:
        return
    maturity_skews = [
        surface.iv_at_delta_maturity(0.90, float(days))
        - surface.iv_at_delta_maturity(0.10, float(days))
        for days in surface.maturity_days
    ]
    skews.append(float(np.mean(maturity_skews)))
    atms.append(surface.iv_at_delta_maturity(0.50, float(surface.maturity_days[0])))
    vrps.append(float(snapshot.regime_features["vrp"]))
    iv_ranks.append(float(snapshot.regime_features["iv_rank"]))
    term_slopes.append(float(snapshot.regime_features["term_slope"]))


def _safe_mean(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.mean(finite)) if finite.size else 0.0


def _safe_std(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.std(finite)) if finite.size else 0.0


def _lag_autocorr(values: list[float], lag: int) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if lag <= 0 or arr.size <= lag:
        return 0.0
    x = arr[:-lag]
    y = arr[lag:]
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 1.0e-12 or y_std <= 1.0e-12:
        return 0.0
    return float(np.mean((x - float(np.mean(x))) * (y - float(np.mean(y)))) / (x_std * y_std))


def _change_variance_ratio(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 3:
        return 0.0
    level_var = float(np.var(arr))
    if level_var <= 1.0e-12:
        return 0.0
    return float(np.var(np.diff(arr)) / level_var)
