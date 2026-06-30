"""Curriculum-aware ``EnvFactory`` construction.

The reward and cost models are held FIXED across curriculum waves (a core project
invariant); only the price generator changes from wave to wave. This single
builder is shared by the training CLIs and the behavioral-validation CLI so the
training distribution and the validation distribution can never silently drift.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from optspread.config import CostConfig, EnvConfig, GBMConfig, RewardConfig
from optspread.curriculum.waves import wave1_spec, wave2_spec
from optspread.envs.builder import EnvBundle
from optspread.market.gbm import GBMGenerator
from optspread.market.generator import PriceGenerator
from optspread.market.rehearsal_generator import RehearsalGenerator
from optspread.training.env_factory import EnvFactory
from optspread.training.phase2 import curriculum_reward, no_cost_config, phase2_risk_reward


def default_reward_for_wave(wave_id: int) -> RewardConfig:
    """Reward each wave uses when one is not supplied explicitly.

    Wave 0 keeps the soft-CVaR no-edge gate reward (its thesis question is
    environment honesty / FLAT-dominance). Waves >= 1 train on mark-to-market
    P&L only, with tail-aversion delivered agent-side.
    """
    return phase2_risk_reward() if wave_id == 0 else curriculum_reward()


def _wave_generator_factory(wave_id: int, cfg: GBMConfig) -> Callable[[], PriceGenerator]:
    if wave_id == 0:
        return lambda: GBMGenerator(cfg)
    if wave_id == 1:
        return wave1_spec(cfg).make_generator
    if wave_id == 2:
        return wave2_spec(cfg).make_generator
    raise ValueError(f"unsupported wave: {wave_id}")


def wave_factory(
    wave_id: int,
    *,
    episode_length: int = 63,
    reward: RewardConfig | None = None,
    with_costs: bool = True,
    rehearsal_fraction: float = 0.0,
    rehearsal_waves: Sequence[int] | None = None,
) -> EnvFactory:
    """Build the ``EnvFactory`` for a curriculum wave.

    Wave 0 is the fair-IV GBM sanity baseline; Wave 1 is GBM + VRP; Wave 2 is
    Heston SV. The cost model is held fixed across waves; the training reward
    defaults per wave (see ``default_reward_for_wave``) unless overridden. When
    ``rehearsal_fraction`` is positive for Wave 1+, training episodes are mixed
    between the target wave and strictly earlier waves by injecting a
    ``RehearsalGenerator``; pure evaluation should keep the default fraction 0.
    """
    if not 0.0 <= rehearsal_fraction <= 1.0:
        raise ValueError("rehearsal_fraction must be in [0,1]")
    cfg = GBMConfig(n_days=episode_length)
    bundle_reward = reward or default_reward_for_wave(wave_id)
    cost = CostConfig() if with_costs else no_cost_config()
    generator_factory: Callable[[], PriceGenerator] | None
    if rehearsal_fraction > 0.0 and wave_id >= 1:
        waves = sorted(rehearsal_waves) if rehearsal_waves is not None else list(range(wave_id))
        invalid_waves = [w for w in waves if w < 0 or w >= wave_id]
        if invalid_waves:
            raise ValueError(
                f"rehearsal waves must be earlier than wave {wave_id}: {invalid_waves}"
            )
        rehearsal_wave_ids = tuple(waves)
        primary_generator_factory = _wave_generator_factory(wave_id, cfg)
        other_generator_factories = tuple(
            _wave_generator_factory(w, cfg) for w in rehearsal_wave_ids
        )

        def make_rehearsal_generator() -> PriceGenerator:
            return RehearsalGenerator(
                primary=primary_generator_factory(),
                others=[make_generator() for make_generator in other_generator_factories],
                rehearsal_fraction=rehearsal_fraction,
            )

        generator_factory = make_rehearsal_generator
    elif wave_id == 0:
        generator_factory = None
    else:
        generator_factory = _wave_generator_factory(wave_id, cfg)
    return EnvFactory(
        EnvBundle(
            env=EnvConfig(episode_length=episode_length),
            gbm=cfg,
            cost=cost,
            reward=bundle_reward,
            generator_factory=generator_factory,
        )
    )
