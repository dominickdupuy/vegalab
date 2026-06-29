"""Pre-registered behavioral predictions for curriculum waves."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreRegisteredPrediction:
    wave: int
    hypothesis: str
    statistic: str
    threshold: float
    direction: str

    def evaluate(self, value: float) -> bool:
        if self.direction == ">":
            return value > self.threshold
        if self.direction == ">=":
            return value >= self.threshold
        if self.direction == "<":
            return value < self.threshold
        if self.direction == "<=":
            return value <= self.threshold
        raise ValueError(f"unsupported direction {self.direction!r}")


WAVE1_CREDIT_VRP = PreRegisteredPrediction(
    wave=1,
    hypothesis="credit-structure frequency rises with the VRP feature",
    statistic="corr(credit_indicator, vrp)",
    threshold=0.10,
    direction=">",
)
