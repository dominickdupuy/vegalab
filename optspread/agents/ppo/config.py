"""PPOConfig: every hyperparameter, defaulted to the brief's starting point.

Pydantic so a run is fully serializable alongside its checkpoint. Defaults are the
brief's section-4 table; the network-capacity defaults are deliberately small (an
overfitting control — Wave 0 must be fittable by a tiny MLP, and if it is not,
that is itself a red flag, not a reason to scale up).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class PPOConfig(BaseModel):
    # -- rollout / scale --------------------------------------------------- #
    num_envs: int = Field(default=16, gt=0)
    num_steps: int = Field(default=256, gt=0)  # per-env rollout length
    total_timesteps: int = Field(default=2_000_000, gt=0)

    # -- optimization ------------------------------------------------------ #
    learning_rate: float = Field(default=2.5e-4, gt=0.0)
    anneal_lr: bool = True
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    update_epochs: int = Field(default=8, gt=0)
    num_minibatches: int = Field(default=8, gt=0)
    clip_coef: float = Field(default=0.2, gt=0.0)
    clip_vloss: bool = True
    ent_coef: float = Field(default=0.01, ge=0.0)  # the anti-collapse lever
    vf_coef: float = Field(default=0.5, ge=0.0)
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    target_kl: float | None = Field(default=0.02)  # None disables early stop

    # -- normalization ----------------------------------------------------- #
    norm_adv: bool = True
    norm_obs: bool = True  # causal
    norm_reward: bool = False  # off by default (can mask reward weighting)

    # -- network capacity (small on purpose) ------------------------------- #
    hidden_sizes: tuple[int, ...] = (64, 64)
    shared_trunk: bool = False

    # -- bookkeeping ------------------------------------------------------- #
    seed: int = 1
    device: str = "cpu"  # CPU is sufficient for Wave 0 and fully deterministic

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_iterations(self) -> int:
        return self.total_timesteps // self.batch_size

    @model_validator(mode="after")
    def _check_divisible(self) -> PPOConfig:
        if self.batch_size % self.num_minibatches != 0:
            raise ValueError(
                f"batch_size {self.batch_size} not divisible by num_minibatches "
                f"{self.num_minibatches}"
            )
        return self
