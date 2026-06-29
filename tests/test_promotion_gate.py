"""Pre-registered prediction and promotion gate logic."""

from __future__ import annotations

from optspread.curriculum.predictions import WAVE1_CREDIT_VRP
from optspread.curriculum.promotion import (
    ForgettingResult,
    evaluate_behavioral_prediction,
    promotion_gate,
)
from optspread.eval.generator_validation import GeneratorValidationResult


def test_behavioral_prediction_direction() -> None:
    assert evaluate_behavioral_prediction(WAVE1_CREDIT_VRP, 0.2).passed
    assert not evaluate_behavioral_prediction(WAVE1_CREDIT_VRP, 0.0).passed


def test_promotion_gate_requires_all_three_parts() -> None:
    gv = GeneratorValidationResult(True, 0.2, 0.1, "ok")
    bv = evaluate_behavioral_prediction(WAVE1_CREDIT_VRP, 0.2)
    ff = ForgettingResult(True, "ok")
    assert promotion_gate(gv, bv, ff).passed
    assert not promotion_gate(gv, bv, ForgettingResult(False, "forgot wave 0")).passed
