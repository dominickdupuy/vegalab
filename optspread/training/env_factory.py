"""EnvFactory: the single sanctioned way to build a ``SpreadEnv`` for Phase 2/3.

Both the trainer and the evaluator construct envs through this factory so training
and evaluation are byte-for-byte the same environment — the credibility of the
eventual "distributional beats expected-value" claim rests on that (brief 1.1).
The factory just holds an ``EnvBundle`` (all the injected-dependency configs) and
delegates to the Phase 1 ``build_default_env`` wiring; it adds no economic logic.

Reward variants for the no-edge gate (full risk-adjusted vs pure-PnL ablation)
are expressed as different bundles, so the *only* thing that changes between them
is the ``RewardConfig`` — never the construction path.
"""

from __future__ import annotations

from dataclasses import replace

from optspread.actions.library import N_ACTIONS
from optspread.config import RewardConfig
from optspread.envs.builder import EnvBundle, build_default_env
from optspread.envs.observation import ObservationBuilder
from optspread.envs.spread_env import SpreadEnv


class EnvFactory:
    """Builds identically-configured ``SpreadEnv`` instances on demand."""

    def __init__(self, bundle: EnvBundle | None = None) -> None:
        self.bundle = bundle or EnvBundle()

    def make(self) -> SpreadEnv:
        """Construct a fresh env with the bundle's injected dependencies."""
        return build_default_env(self.bundle)

    def with_reward(self, reward: RewardConfig) -> EnvFactory:
        """Return a sibling factory identical but for the reward weighting.

        Used to express the gate's pure-PnL ablation without touching any other
        part of the environment (costs, generator, margin all unchanged).
        """
        return EnvFactory(replace(self.bundle, reward=reward))

    def with_costs(self, cost: object) -> EnvFactory:
        """Return a sibling factory with a different ``CostConfig`` (e.g. no costs)."""
        return EnvFactory(replace(self.bundle, cost=cost))  # type: ignore[arg-type]

    # -- static descriptors (no env instance needed) ----------------------- #

    @property
    def obs_dim(self) -> int:
        return ObservationBuilder(self.bundle.env).dim

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    @property
    def episode_length(self) -> int:
        return self.bundle.env.episode_length
