"""Reward components: the swappable terms of the reward.

Each component is a small, stateful object that turns a ``StepContext`` into a
scalar contribution. The env sums them (weighted) through ``CompositeReward``.

NO ENTROPY BONUS LIVES HERE. Exploration entropy is part of the PPO objective in
Phase 2 (agent-side); putting it in the reward would double-count it and corrupt
the Wave-0 economic expectation. This is invariant #4 — do not add it.

Stateful components (the differential Sharpe/Sortino EMAs, the CVaR window) must
be reset at the start of every episode via ``reset()`` so trajectories stay
deterministic and independent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

from optspread.config import RewardConfig
from optspread.reward.context import StepContext


class RewardComponent(ABC):
    """A single additive term of the reward."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier used as the key in the per-step breakdown."""

    @abstractmethod
    def reset(self) -> None:
        """Clear any per-episode state. Called once before each episode."""

    @abstractmethod
    def update(self, ctx: StepContext) -> float:
        """Return this term's scalar contribution for the step."""

    def observable_state(self) -> dict[str, float]:
        """Markov state this component exposes to the observation, if any.

        History-dependent components (the Sharpe/Sortino EMAs) override this so a
        feed-forward policy can see the state its risk-adjusted reward depends on
        (see the Markovian-reward-state requirement in CLAUDE.md). Default: none.
        """
        return {}


class MTMPnL(RewardComponent):
    """Raw mark-to-market dollar P&L, scaled. The economic backbone of reward."""

    def __init__(self, config: RewardConfig) -> None:
        self._scale = config.pnl_scale

    @property
    def name(self) -> str:
        return "mtm"

    def reset(self) -> None:  # stateless
        pass

    def update(self, ctx: StepContext) -> float:
        return ctx.pnl / self._scale


class MarginNormalizer(RewardComponent):
    """P&L per dollar of buying power held (a return-on-margin signal).

    Divides the step P&L by the margin held, floored to avoid blow-ups when a
    FLAT/near-zero-margin position would otherwise divide by ~0.
    """

    def __init__(self, config: RewardConfig) -> None:
        self._floor = config.margin_floor

    @property
    def name(self) -> str:
        return "margin_normalized"

    def reset(self) -> None:  # stateless
        pass

    def update(self, ctx: StepContext) -> float:
        return ctx.pnl / max(ctx.margin, self._floor)


class DifferentialSharpe(RewardComponent):
    """Online differential Sharpe ratio (Moody & Saffell, 1998).

    Maintains EMAs ``A`` (mean return) and ``B`` (mean squared return) with rate
    ``eta`` and returns the marginal influence of the latest return on the
    Sharpe ratio. This is a dense, online surrogate for end-of-episode Sharpe.
    """

    def __init__(self, config: RewardConfig) -> None:
        self._eta = config.eta
        self._scale = config.pnl_scale
        self._A = 0.0
        self._B = 0.0
        self._initialised = False

    @property
    def name(self) -> str:
        return "diff_sharpe"

    def reset(self) -> None:
        self._A = 0.0
        self._B = 0.0
        self._initialised = False

    def update(self, ctx: StepContext) -> float:
        R = ctx.pnl / self._scale
        if not self._initialised:
            # First observation only seeds the EMAs; no Sharpe yet.
            self._A = R
            self._B = R * R
            self._initialised = True
            return 0.0
        dA = self._eta * (R - self._A)
        dB = self._eta * (R * R - self._B)
        var = self._B - self._A * self._A
        d_sharpe = 0.0 if var <= 1e-12 else (self._B * dA - 0.5 * self._A * dB) / (var**1.5)
        self._A += dA
        self._B += dB
        return d_sharpe

    def observable_state(self) -> dict[str, float]:
        # Expose the two EMAs so the policy can see the differential-Sharpe state
        # that scales its reward; otherwise it fights hidden state.
        return {"sharpe_a": self._A, "sharpe_b": self._B}


class Sortino(RewardComponent):
    """Online differential downside-deviation ratio (Sortino flavour).

    Like ``DifferentialSharpe`` but the denominator tracks only downside
    (negative-return) variance, so upside volatility is not penalised.
    """

    def __init__(self, config: RewardConfig) -> None:
        self._eta = config.eta
        self._scale = config.pnl_scale
        self._A = 0.0
        self._DD2 = 0.0
        self._initialised = False

    @property
    def name(self) -> str:
        return "sortino"

    def reset(self) -> None:
        self._A = 0.0
        self._DD2 = 0.0
        self._initialised = False

    def update(self, ctx: StepContext) -> float:
        R = ctx.pnl / self._scale
        downside = min(R, 0.0)
        if not self._initialised:
            self._A = R
            self._DD2 = downside * downside
            self._initialised = True
            return 0.0
        dd = self._DD2**0.5
        if dd <= 1e-12:
            # No downside observed yet: any non-negative return is "free".
            d_sortino = 0.0 if R <= 0.0 else R / 1.0
        elif R > 0.0:
            d_sortino = (R - 0.5 * self._A) / dd
        else:
            d_sortino = (self._DD2 * (R - 0.5 * self._A) - 0.5 * self._A * R * R) / (dd**3)
        d_sortino *= self._eta
        self._A += self._eta * (R - self._A)
        self._DD2 += self._eta * (downside * downside - self._DD2)
        return d_sortino


def empirical_cvar(returns: list[float], alpha: float) -> float:
    """Empirical CVaR (expected shortfall) at level ``alpha`` (<= 0 for losses).

    The mean of the worst ``alpha`` fraction of returns. Returns 0.0 for an empty
    sample. At least one observation is always included in the tail.
    """
    if not returns:
        return 0.0
    ordered = sorted(returns)
    k = max(1, int(len(ordered) * alpha))
    tail = ordered[:k]
    return sum(tail) / len(tail)


class CVaRPenalty(RewardComponent):
    """Tail-risk shaping penalty: the rolling empirical CVaR of step returns.

    Maintains a window of recent returns and reports their CVaR_alpha (a
    non-positive shaping signal when the tail is loss-heavy). Penalises only the
    part of the tail below ``cvar_threshold`` so a benign return distribution
    contributes ~0.
    """

    def __init__(self, config: RewardConfig, window: int = 64) -> None:
        self._alpha = config.cvar_alpha
        self._threshold = config.cvar_threshold
        self._scale = config.pnl_scale
        self._window = window
        self._returns: deque[float] = deque(maxlen=window)

    @property
    def name(self) -> str:
        return "cvar"

    def reset(self) -> None:
        self._returns.clear()

    def update(self, ctx: StepContext) -> float:
        self._returns.append(ctx.pnl / self._scale)
        cvar = empirical_cvar(list(self._returns), self._alpha)
        # Only the shortfall below the threshold is penalised; >= threshold -> 0.
        return min(0.0, cvar - self._threshold)
