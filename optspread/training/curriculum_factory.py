"""Curriculum-aware ``EnvFactory`` construction.

The reward and cost models are held FIXED across curriculum waves (a core project
invariant); only the price generator changes from wave to wave. This single
builder is shared by the training CLIs and the behavioral-validation CLI so the
training distribution and the validation distribution can never silently drift.
"""

from __future__ import annotations

from optspread.config import CostConfig, EnvConfig, GBMConfig, RewardConfig
from optspread.curriculum.waves import wave1_spec, wave2_spec
from optspread.envs.builder import EnvBundle
from optspread.training.env_factory import EnvFactory
from optspread.training.phase2 import curriculum_reward, no_cost_config, phase2_risk_reward


def default_reward_for_wave(wave_id: int) -> RewardConfig:
    """Reward each wave uses when one is not supplied explicitly.

    Wave 0 keeps the soft-CVaR no-edge gate reward (its thesis question is
    environment honesty / FLAT-dominance). Waves >= 1 train on mark-to-market
    P&L only, with tail-aversion delivered agent-side.
    """
    return phase2_risk_reward() if wave_id == 0 else curriculum_reward()


def wave_factory(
    wave_id: int,
    *,
    episode_length: int = 63,
    reward: RewardConfig | None = None,
    with_costs: bool = True,
) -> EnvFactory:
    """Build the ``EnvFactory`` for a curriculum wave.

    Wave 0 is the fair-IV GBM sanity baseline; Wave 1 is GBM + VRP; Wave 2 is
    Heston SV. The cost model is held fixed across waves; the training reward
    defaults per wave (see ``default_reward_for_wave``) unless overridden.
    """
    cfg = GBMConfig(n_days=episode_length)
    bundle_reward = reward or default_reward_for_wave(wave_id)
    cost = CostConfig() if with_costs else no_cost_config()
    if wave_id == 0:
        generator_factory = None
    elif wave_id == 1:
        generator_factory = wave1_spec(cfg).make_generator
    elif wave_id == 2:
        generator_factory = wave2_spec(cfg).make_generator
    else:
        raise ValueError(f"unsupported wave: {wave_id}")
    return EnvFactory(
        EnvBundle(
            env=EnvConfig(episode_length=episode_length),
            gbm=cfg,
            cost=cost,
            reward=bundle_reward,
            generator_factory=generator_factory,
        )
    )
