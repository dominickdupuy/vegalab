"""ObservationBuilder: the fixed observation schema the agent will see.

The schema is frozen NOW (Phase 1) so a Phase-2 policy network never has to be
re-architected when richer generators or features arrive: new information must be
folded into existing slots, or appended, never inserted. The vector is purely a
function of information available at the close of the step — no look-ahead.

Layout (all float32):
    [0:R]    regime features, canonical order, R=len(REGIME_FEATURE_KEYS)
    [R]      has_position flag (0/1)
    [R+1]    margin held / initial_cash
    [R+2]    unrealized P&L / initial_cash
    [R+3]    days position has been held / episode_length
    [R+4]    held action_id / N_ACTIONS  (0 when flat)
    [R+5]    episode progress: day / episode_length
    [R+6]    equity / initial_cash
    [R+7]    differential-Sharpe EMA A (mean return)        }  risk-reward state, so
    [R+8]    differential-Sharpe EMA B (mean squared return)}  the risk-adjusted
    [R+9]    current drawdown from equity high-water / cash  }  reward is Markovian
    [R+10]   min days-to-expiry of held legs / episode_length}  (see CLAUDE.md)

APPEND-ONLY: the final four non-regime slots were added in Phase 2 to make the
differential-Sharpe and drawdown reward terms observable to a feed-forward
policy. New information must extend this vector, never reorder it — a saved
policy's input layout is fixed forever.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from optspread.actions.library import N_ACTIONS
from optspread.config import EnvConfig
from optspread.market.snapshot import REGIME_FEATURE_KEYS, MarketSnapshot
from optspread.portfolio.position import Portfolio

OBS_NAMES: tuple[str, ...] = (
    *REGIME_FEATURE_KEYS,
    "has_position",
    "margin_frac",
    "unrealized_pnl_frac",
    "days_held_frac",
    "action_frac",
    "time_frac",
    "equity_frac",
    "sharpe_a",
    "sharpe_b",
    "drawdown_frac",
    "dte_frac",
)


class ObservationBuilder:
    """Builds the fixed-length observation vector. Implements the obs contract."""

    def __init__(self, config: EnvConfig) -> None:
        self.config = config

    @property
    def dim(self) -> int:
        return len(OBS_NAMES)

    @property
    def names(self) -> tuple[str, ...]:
        return OBS_NAMES

    def build(
        self,
        snapshot: MarketSnapshot,
        portfolio: Portfolio,
        day: int,
        *,
        drawdown: float = 0.0,
        risk_state: dict[str, float] | None = None,
    ) -> NDArray[np.float32]:
        """Assemble the observation vector.

        ``drawdown`` (dollar peak-to-current equity drop, >= 0) and ``risk_state``
        (the reward's exposed Markov state, e.g. the differential-Sharpe EMAs) are
        supplied by the env, which owns the high-water mark and holds the reward.
        Both default to neutral so the builder can be exercised standalone.
        """
        cfg = self.config
        chain = snapshot.chain
        equity = portfolio.equity(chain)
        rs = risk_state or {}
        if portfolio.has_position:
            assert portfolio.position is not None
            held_action = portfolio.position.action_id
            days_held = day - portfolio.position.open_day
            has_pos = 1.0
            dte = self._min_days_to_expiry(portfolio, day)
        else:
            held_action = 0
            days_held = 0
            has_pos = 0.0
            dte = 0
        values = [
            *snapshot.feature_vector(),
            has_pos,
            portfolio.margin_used / cfg.initial_cash,
            portfolio.unrealized_pnl(chain) / cfg.initial_cash,
            days_held / cfg.episode_length,
            held_action / N_ACTIONS,
            day / cfg.episode_length,
            equity / cfg.initial_cash,
            rs.get("sharpe_a", 0.0),
            rs.get("sharpe_b", 0.0),
            drawdown / cfg.initial_cash,
            dte / cfg.episode_length,
        ]
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _min_days_to_expiry(portfolio: Portfolio, day: int) -> int:
        """Smallest remaining tenor (days) across held legs; 0 if none/unknown.

        Legs with an absolute ``expiry_day`` decay toward it; ``-1`` means no fixed
        expiry (legacy unit-test legs) and contributes nothing.
        """
        assert portfolio.position is not None
        dtes = [leg.expiry_day - day for leg in portfolio.position.legs if leg.expiry_day >= 0]
        return max(0, min(dtes)) if dtes else 0
