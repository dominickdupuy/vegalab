"""Convenience wiring: assemble a default ``SpreadEnv`` from configs.

The env constructor stays pure dependency-injection; this helper is the one place
that picks the Wave-0 concrete implementations (GBM generator, quoted-spread cost,
Reg-T margin, default composite reward). Tests and the CLI call it so they don't
each re-list the wiring, but the env itself never imports a concrete dependency.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from optspread.config import (
    CostConfig,
    EnvConfig,
    GBMConfig,
    MarginConfig,
    RewardConfig,
)
from optspread.costs.spread_cost import QuotedSpreadCost
from optspread.envs.observation import ObservationBuilder
from optspread.envs.spread_env import SpreadEnv
from optspread.margin.reg_t import RegTStyleMargin
from optspread.market.gbm import GBMGenerator
from optspread.market.generator import PriceGenerator
from optspread.reward.composite import build_default_reward


@dataclass(frozen=True, slots=True)
class EnvBundle:
    """All configs needed to build a Wave-0 env; defaults give the sanity baseline."""

    env: EnvConfig = field(default_factory=EnvConfig)
    gbm: GBMConfig = field(default_factory=GBMConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    margin: MarginConfig = field(default_factory=MarginConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    generator_factory: Callable[[], PriceGenerator] | None = None


def build_default_env(bundle: EnvBundle | None = None) -> SpreadEnv:
    """Build a Wave-0 ``SpreadEnv`` with the standard concrete dependencies."""
    b = bundle or EnvBundle()
    generator = b.generator_factory() if b.generator_factory is not None else GBMGenerator(b.gbm)
    return SpreadEnv(
        config=b.env,
        generator=generator,
        cost_model=QuotedSpreadCost(b.cost),
        margin_model=RegTStyleMargin(b.margin),
        reward=build_default_reward(b.reward),
        observation_builder=ObservationBuilder(b.env),
    )
