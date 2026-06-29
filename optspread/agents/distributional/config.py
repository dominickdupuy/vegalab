"""Pydantic configs for QR-DQN and IQN."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class DistributionalConfig(BaseModel):
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    learning_rate: float = Field(default=5e-5, gt=0.0)
    batch_size: int = Field(default=64, gt=0)
    replay_size: int = Field(default=500_000, gt=0)
    learning_starts: int = Field(default=30_000, ge=0)
    total_timesteps: int = Field(default=100_000, gt=0)
    train_freq: int = Field(default=1, gt=0)
    gradient_steps: int = Field(default=1, gt=0)
    target_update_interval: int = Field(default=2_000, gt=0)
    epsilon_start: float = Field(default=1.0, ge=0.0, le=1.0)
    epsilon_end: float = Field(default=0.02, ge=0.0, le=1.0)
    epsilon_decay_steps: int = Field(default=50_000, gt=0)
    cvar_alpha: float = Field(default=0.05, gt=0.0, le=1.0)
    behavior_risk: str = "mean"
    # Risk measure used for the BOOTSTRAP target's next-action selection. "mean"
    # learns the return distribution under a risk-neutral greedy policy and applies
    # CVaR only at deployment (agent.risk_measure) -- this avoids the nested
    # CVaR-greedy "blindness to success" that otherwise collapses the agent to
    # FLAT and never learns a +EV trade. "cvar" recovers the nested (time-
    # consistent but conservative) iterated-CVaR bootstrap.
    bootstrap_risk: str = "mean"
    hidden_sizes: tuple[int, ...] = (64, 64)
    huber_kappa: float = Field(default=1.0, gt=0.0)
    norm_obs: bool = True
    seed: int = 1
    device: str = "cpu"

    @model_validator(mode="after")
    def _check_epsilon(self) -> DistributionalConfig:
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end must be <= epsilon_start")
        return self


class QRDQNConfig(DistributionalConfig):
    n_quantiles: int = Field(default=200, gt=0)


class IQNConfig(DistributionalConfig):
    n_quantiles: int = Field(default=32, gt=0)
    n_target_quantiles: int = Field(default=32, gt=0)
    n_action_quantiles: int = Field(default=32, gt=0)
    cosine_embed_dim: int = Field(default=64, gt=0)
