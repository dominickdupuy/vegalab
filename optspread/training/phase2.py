"""Phase-2 presets: Wave-0 PPO configs, rewards, and env factories.

These helpers keep the reported Phase-2 gate reproducible. They are deliberately
small wrappers around the generic configs: Phase 3 should still reuse
``EnvFactory``/``Evaluator``/``MetricSuite`` directly, not copy a bespoke setup.
"""

from __future__ import annotations

from optspread.agents.ppo.config import PPOConfig
from optspread.config import CostConfig, EnvConfig, RewardConfig
from optspread.envs.builder import EnvBundle
from optspread.training.env_factory import EnvFactory


def phase2_risk_reward(*, pnl_scale: float = 10_000.0) -> RewardConfig:
    """Main Wave-0 gate reward: scaled MTM plus a soft CVaR tail penalty.

    Differential Sharpe and Sortino components are still instantiated and logged
    by the reward system, but their weights stay at zero for this gate because
    Wave 0's thesis question is environment honesty, not reward-shaping tuning.
    The native tail objective arrives in Phase 3.
    """
    return RewardConfig(
        pnl_scale=pnl_scale,
        mtm_weight=1.0,
        margin_normalized_weight=0.0,
        sharpe_weight=0.0,
        sortino_weight=0.0,
        cvar_weight=0.25,
        cvar_threshold=-0.02,
    )


def pure_pnl_reward(*, pnl_scale: float = 10_000.0) -> RewardConfig:
    """Pure-PnL ablation reward for the no-cost indifference cross-check."""
    return RewardConfig(
        pnl_scale=pnl_scale,
        mtm_weight=1.0,
        margin_normalized_weight=0.0,
        sharpe_weight=0.0,
        sortino_weight=0.0,
        cvar_weight=0.0,
    )


def no_cost_config() -> CostConfig:
    """Zero transaction-cost config for no-cost Wave-0 checks."""
    return CostConfig(half_spread_bps=0.0, min_cost_per_leg=0.0)


def phase2_factory(
    *,
    reward: RewardConfig,
    with_costs: bool,
    episode_length: int = 63,
) -> EnvFactory:
    """Build the standard Phase-2 Wave-0 factory."""
    cost = CostConfig() if with_costs else no_cost_config()
    return EnvFactory(
        EnvBundle(env=EnvConfig(episode_length=episode_length), cost=cost, reward=reward)
    )


def phase2_ppo_config(
    *,
    seed: int,
    total_timesteps: int = 131_072,
    num_envs: int = 8,
    num_steps: int = 64,
    learning_rate: float = 1e-3,
    ent_coef: float = 0.005,
    target_kl: float | None = 0.05,
) -> PPOConfig:
    """Compact, deterministic Wave-0 PPO recipe used by the gate command."""
    return PPOConfig(
        seed=seed,
        num_envs=num_envs,
        num_steps=num_steps,
        total_timesteps=total_timesteps,
        num_minibatches=4,
        update_epochs=8,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        anneal_lr=False,
        target_kl=target_kl,
        norm_obs=True,
        norm_reward=False,
    )
