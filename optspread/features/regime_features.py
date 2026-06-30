"""Regime features computed causally from path and standardized IV surfaces."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from optspread.features.causal import realized_vol, trailing_window
from optspread.market.snapshot import REGIME_FEATURE_KEYS
from optspread.market.surface import IVSurface


def _finite(x: float, default: float) -> float:
    """Return a finite Python float, falling back to ``default`` otherwise."""
    value = float(x)
    if np.isfinite(value):
        return value
    return float(default)


def build_regime_features(
    *,
    surface: IVSurface,
    log_returns: Sequence[float],
    iv_history: Sequence[float],
    realized_window: int = 21,
) -> dict[str, float]:
    """Build the canonical observation feature dict from current observable data."""
    eps = 1.0e-12
    tdays = surface.trading_days_per_year
    md = surface.maturity_days
    near = float(md[0])
    far = float(md[-1])
    mid = float(md[len(md) // 2])

    atm_near = _finite(surface.iv_at_delta_maturity(0.50, near), 0.0)
    atm_far = _finite(surface.iv_at_delta_maturity(0.50, far), atm_near)
    rv = realized_vol(
        log_returns,
        window=realized_window,
        trading_days_per_year=tdays,
    )
    if rv == 0.0:
        rv = atm_near
    iv_vals = trailing_window(iv_history, max(1, len(iv_history)))
    if iv_vals.size > 1 and float(iv_vals.max()) > float(iv_vals.min()):
        iv_rank = (atm_near - float(iv_vals.min())) / (float(iv_vals.max()) - float(iv_vals.min()))
    else:
        iv_rank = 0.5
    recent = trailing_window(log_returns, min(realized_window, max(1, len(log_returns))))
    daily_vol = atm_near / np.sqrt(tdays)
    momentum = float(recent.mean() / daily_vol) if recent.size and daily_vol > 0 else 0.0

    vrp = atm_near * atm_near - rv * rv
    skew = _finite(
        surface.iv_at_delta_maturity(0.90, near) - surface.iv_at_delta_maturity(0.10, near),
        0.0,
    )
    smile_curvature = _finite(
        surface.iv_at_delta_maturity(0.10, near)
        + surface.iv_at_delta_maturity(0.90, near)
        - 2.0 * atm_near,
        0.0,
    )
    term_curvature = _finite(
        2.0 * surface.iv_at_delta_maturity(0.50, mid)
        - atm_near
        - surface.iv_at_delta_maturity(0.50, far),
        0.0,
    )
    skew_term = _finite(
        (surface.iv_at_delta_maturity(0.90, far) - surface.iv_at_delta_maturity(0.10, far)) - skew,
        0.0,
    )

    iv_n = len(iv_history)
    iv_change_1d = iv_history[-1] - iv_history[-2] if iv_n >= 2 else 0.0
    iv_change_5d = iv_history[-1] - iv_history[-6] if iv_n >= 6 else 0.0
    iv_recent = trailing_window(iv_history, min(22, max(1, iv_n)))
    iv_diffs = np.diff(iv_recent)
    vol_of_vol = float(iv_diffs.std(ddof=1) * np.sqrt(tdays)) if iv_diffs.size >= 2 else 0.0
    iv_z_window = trailing_window(iv_history, min(63, max(1, iv_n)))
    iv_z_std = float(iv_z_window.std(ddof=1)) if iv_z_window.size >= 2 else 0.0
    iv_zscore = (
        float((iv_history[-1] - float(iv_z_window.mean())) / iv_z_std)
        if iv_n >= 2 and iv_z_std > eps
        else 0.0
    )

    ret_n = len(log_returns)
    ret_5 = trailing_window(log_returns, min(5, max(1, ret_n)))
    ret_63 = trailing_window(log_returns, min(63, max(1, ret_n)))
    momentum_5d = float(ret_5.mean() / daily_vol) if ret_n >= 1 and daily_vol > 0 else 0.0
    momentum_63d = float(ret_63.mean() / daily_vol) if ret_n >= 1 and daily_vol > 0 else 0.0
    rv_5d = realized_vol(log_returns, window=5, trading_days_per_year=tdays)
    rv_63d = realized_vol(log_returns, window=63, trading_days_per_year=tdays)
    rv_term_ratio = rv_5d / rv_63d - 1.0 if rv_63d > eps else 0.0
    ret_63_std = float(ret_63.std()) if ret_63.size >= 4 else 0.0
    if ret_63.size >= 4 and ret_63_std > eps:
        standardized = (ret_63 - float(ret_63.mean())) / ret_63_std
        realized_skew = float(np.mean(standardized**3))
        realized_kurt = float(np.mean(standardized**4) - 3.0)
    else:
        realized_skew = 0.0
        realized_kurt = 0.0
    ret_21 = trailing_window(log_returns, min(21, max(1, ret_n)))
    downside_semidev = (
        float(np.sqrt(np.mean(np.minimum(ret_21, 0.0) ** 2)) * np.sqrt(tdays))
        if ret_21.size >= 1
        else 0.0
    )
    if ret_63.size >= 1:
        cumulative_returns = np.cumsum(ret_63)
        running_peak = max(0.0, float(np.max(cumulative_returns)))
        path_drawdown = max(0.0, running_peak - float(cumulative_returns[-1]))
    else:
        path_drawdown = 0.0
    vrp_norm = vrp / max(rv * rv, eps)

    features = {
        "trailing_momentum": _finite(momentum, 0.0),
        "realized_vol": _finite(rv, atm_near),
        "vrp": _finite(vrp, 0.0),
        "iv_rank": _finite(float(np.clip(iv_rank, 0.0, 1.0)), 0.5),
        "term_slope": _finite(atm_far - atm_near, 0.0),
        "skew": _finite(skew, 0.0),
        "smile_curvature": _finite(smile_curvature, 0.0),
        "term_curvature": _finite(term_curvature, 0.0),
        "atm_iv": _finite(atm_near, atm_near),
        "skew_term": _finite(skew_term, 0.0),
        "iv_change_1d": _finite(iv_change_1d, 0.0),
        "iv_change_5d": _finite(iv_change_5d, 0.0),
        "vol_of_vol": _finite(vol_of_vol, 0.0),
        "iv_zscore": _finite(iv_zscore, 0.0),
        "momentum_5d": _finite(momentum_5d, 0.0),
        "momentum_63d": _finite(momentum_63d, 0.0),
        "realized_vol_5d": _finite(rv_5d, 0.0),
        "realized_vol_63d": _finite(rv_63d, 0.0),
        "rv_term_ratio": _finite(rv_term_ratio, 0.0),
        "realized_skew": _finite(realized_skew, 0.0),
        "realized_kurt": _finite(realized_kurt, 0.0),
        "downside_semidev": _finite(downside_semidev, 0.0),
        "path_drawdown": _finite(path_drawdown, 0.0),
        "vrp_norm": _finite(vrp_norm, 0.0),
    }
    missing = [key for key in REGIME_FEATURE_KEYS if key not in features]
    if missing:
        raise ValueError(f"feature pipeline missing canonical keys: {missing}")
    return features
