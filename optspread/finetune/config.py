"""Fine-tuning configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FineTuneConfig(BaseModel):
    learning_rate: float = Field(default=1e-5, gt=0.0)
    max_steps: int = Field(default=10_000, gt=0)
    patience: int = Field(default=5, gt=0)
    freeze_trunk: bool = True
