"""Curriculum runner skeleton: GV -> train -> BV -> FF -> promote."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from optspread.curriculum.promotion import (
    BehavioralResult,
    ForgettingResult,
    PromotionDecision,
    promotion_gate,
)
from optspread.curriculum.waves import WaveSpec
from optspread.eval.generator_validation import GeneratorValidationResult


@dataclass(frozen=True, slots=True)
class CurriculumWaveResult:
    wave_id: int
    generator: GeneratorValidationResult
    promotion: PromotionDecision


class CurriculumRunner:
    """Minimal orchestration shell for the strict Phase-4 gate order."""

    def __init__(
        self,
        *,
        generator_validator: Callable[[WaveSpec], GeneratorValidationResult],
    ) -> None:
        self.generator_validator = generator_validator

    def validate_only(self, wave: WaveSpec) -> CurriculumWaveResult:
        """Run GV and block promotion until BV/FF are supplied by training code."""
        gv = self.generator_validator(wave)
        pending_behavior = _pending_behavior()
        pending_forgetting = ForgettingResult(False, "forgetting check pending")
        return CurriculumWaveResult(
            wave_id=wave.wave_id,
            generator=gv,
            promotion=promotion_gate(gv, pending_behavior, pending_forgetting),
        )


def _pending_behavior() -> BehavioralResult:
    return BehavioralResult(False, float("nan"), "behavioral validation pending")
