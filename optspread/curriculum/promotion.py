"""Promotion-gate decisions for curriculum waves."""

from __future__ import annotations

from dataclasses import dataclass

from optspread.curriculum.predictions import PreRegisteredPrediction
from optspread.eval.generator_validation import GeneratorValidationResult


@dataclass(frozen=True, slots=True)
class BehavioralResult:
    passed: bool
    statistic: float
    reason: str


@dataclass(frozen=True, slots=True)
class ForgettingResult:
    passed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    passed: bool
    reason: str


def evaluate_behavioral_prediction(
    prediction: PreRegisteredPrediction,
    value: float,
) -> BehavioralResult:
    passed = prediction.evaluate(value)
    op = prediction.direction
    reason = (
        f"{prediction.statistic}={value:.4f} {op} {prediction.threshold:.4f}"
        if passed
        else f"{prediction.statistic}={value:.4f} failed {op} {prediction.threshold:.4f}"
    )
    return BehavioralResult(passed=passed, statistic=value, reason=reason)


def promotion_gate(
    generator: GeneratorValidationResult,
    behavioral: BehavioralResult,
    forgetting: ForgettingResult,
) -> PromotionDecision:
    failures = []
    if not generator.passed:
        failures.append(f"GV failed: {generator.reason}")
    if not behavioral.passed:
        failures.append(f"BV failed: {behavioral.reason}")
    if not forgetting.passed:
        failures.append(f"FF failed: {forgetting.reason}")
    if failures:
        return PromotionDecision(False, "; ".join(failures))
    return PromotionDecision(True, "generator, behavioral, and forgetting gates passed")
