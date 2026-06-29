"""Curriculum-aware ``EnvFactory`` construction.

The reward and cost models are held FIXED across curriculum waves (a core project
invariant); only the price generator changes from wave to wave. This single
builder is shared by the training CLIs and the behavioral-validation CLI so the
training distribution and the validation distribution can never silently drift.
"""

from __future__ import annotations

from optspread.config import CostConfig, EnvConfig, GBMConfig, RewardConfig
from optspread.curriculum.waves import wave1_spec
from optspread.envs.builder import EnvBundle
from optspread.training.env_factory import EnvFactory
from optspread.training.phase2 import no_cost_config, phase2_risk_reward


def wave_factory(
    wave_id: int,
    *,
    episode_length: int = 63,
    reward: RewardConfig | None = None,
    with_costs: bool = True,
) -> EnvFactory:
    """Build the ``EnvFactory`` for a curriculum wave.

    Wave 0 is the fair-IV GBM sanity baseline; waves >= 1 inject their stylized
    -fact generator while reusing the identical reward and cost models.
    """
    cfg = GBMConfig(n_days=episode_length)
    bundle_reward = reward or phase2_risk_reward()
    cost = CostConfig() if with_costs else no_cost_config()
    if wave_id == 0:
        generator_factory = None
    elif wave_id == 1:
        generator_factory = wave1_spec(cfg).make_generator
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
