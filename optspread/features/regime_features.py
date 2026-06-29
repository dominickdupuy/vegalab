"""Regime features computed causally from path and standardized IV surfaces."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from optspread.features.causal import realized_vol, trailing_window
from optspread.market.snapshot import REGIME_FEATURE_KEYS
from optspread.market.surface import IVSurface


def build_regime_features(
    *,
    surface: IVSurface,
    log_returns: Sequence[float],
    iv_history: Sequence[float],
    realized_window: int = 21,
) -> dict[str, float]:
    """Build the canonical observation feature dict from current observable data."""
    atm_near = surface.iv_at_delta_maturity(0.50, float(surface.maturity_days[0]))
    atm_far = surface.iv_at_delta_maturity(0.50, float(surface.maturity_days[-1]))
    rv = realized_vol(
        log_returns,
        window=realized_window,
        trading_days_per_year=surface.trading_days_per_year,
    )
    if rv == 0.0:
        rv = atm_near
    iv_vals = trailing_window(iv_history, max(1, len(iv_history)))
    if iv_vals.size > 1 and float(iv_vals.max()) > float(iv_vals.min()):
        iv_rank = (atm_near - float(iv_vals.min())) / (float(iv_vals.max()) - float(iv_vals.min()))
    else:
        iv_rank = 0.5
    recent = trailing_window(log_returns, min(realized_window, max(1, len(log_returns))))
    daily_vol = atm_near / np.sqrt(surface.trading_days_per_year)
    momentum = float(recent.mean() / daily_vol) if recent.size and daily_vol > 0 else 0.0
    features = {
        "trailing_momentum": momentum,
        "realized_vol": rv,
        "vrp": atm_near * atm_near - rv * rv,
        "iv_rank": float(np.clip(iv_rank, 0.0, 1.0)),
        "term_slope": atm_far - atm_near,
    }
    missing = [key for key in REGIME_FEATURE_KEYS if key not in features]
    if missing:
        raise ValueError(f"feature pipeline missing canonical keys: {missing}")
    return features
