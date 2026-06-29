"""CompositeReward: weighted sum of reward components with per-term logging.

The env holds exactly one of these (dependency-injected). It owns no economic
logic itself — it just combines the injected components by their configured
weights and exposes the last per-component breakdown for diagnostics.
"""

from __future__ import annotations

from collections.abc import Sequence

from optspread.config import RewardConfig
from optspread.reward.components import (
    CVaRPenalty,
    DifferentialSharpe,
    MarginNormalizer,
    MTMPnL,
    RewardComponent,
    Sortino,
)
from optspread.reward.context import StepContext


class CompositeReward:
    """Weighted sum of ``RewardComponent`` terms.

    Parameters
    ----------
    components:
        The terms to combine, paired with their scalar weights. Order is
        preserved for the breakdown. A weight of 0 disables a term's
        contribution to the total but it is still evaluated and logged (so
        ablations stay observable).
    """

    def __init__(self, components: Sequence[tuple[RewardComponent, float]]) -> None:
        self._components = list(components)
        self._last_breakdown: dict[str, float] = {}

    @property
    def last_breakdown(self) -> dict[str, float]:
        """Per-component weighted contributions from the most recent ``update``."""
        return dict(self._last_breakdown)

    def reset(self) -> None:
        for component, _ in self._components:
            component.reset()
        self._last_breakdown = {}

    def observable_state(self) -> dict[str, float]:
        """Merged Markov state the components expose to the observation.

        Algorithm-agnostic: any component may surface history-dependent state it
        wants the policy to see (the differential-Sharpe EMAs do). The
        ``ObservationBuilder`` reads fixed keys with safe defaults, so this stays
        valid whatever component set is wired.
        """
        state: dict[str, float] = {}
        for component, _ in self._components:
            state.update(component.observable_state())
        return state

    def update(self, ctx: StepContext) -> float:
        total = 0.0
        breakdown: dict[str, float] = {}
        for component, weight in self._components:
            contribution = weight * component.update(ctx)
            breakdown[component.name] = contribution
            total += contribution
        self._last_breakdown = breakdown
        return total


def build_default_reward(config: RewardConfig) -> CompositeReward:
    """Wire the standard component set with weights taken from ``config``.

    Every term is always instantiated so it is logged in the breakdown; the
    config weights decide what actually enters the total. The Wave-0 default
    (``mtm_weight=1``, all others 0) reduces the reward to raw P&L.
    """
    components: list[tuple[RewardComponent, float]] = [
        (MTMPnL(config), config.mtm_weight),
        (MarginNormalizer(config), config.margin_normalized_weight),
        (DifferentialSharpe(config), config.sharpe_weight),
        (Sortino(config), config.sortino_weight),
        (CVaRPenalty(config), config.cvar_weight),
    ]
    return CompositeReward(components)
