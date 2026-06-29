"""Conservative fine-tuning utilities."""

from __future__ import annotations

from dataclasses import dataclass

from optspread.finetune.config import FineTuneConfig


@dataclass(frozen=True, slots=True)
class FineTunePlan:
    allowed: bool
    reason: str
    config: FineTuneConfig


def make_finetune_plan(
    *, has_validation_fold: bool, config: FineTuneConfig | None = None
) -> FineTunePlan:
    cfg = config or FineTuneConfig()
    if not has_validation_fold:
        return FineTunePlan(False, "validation fold required for early stopping", cfg)
    return FineTunePlan(True, "low-LR validation-stopped fine-tuning permitted", cfg)
