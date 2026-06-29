"""Wave registry for the synthetic curriculum."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from optspread.config import GBMConfig
from optspread.curriculum.predictions import WAVE1_CREDIT_VRP, PreRegisteredPrediction
from optspread.market.gbm_vrp import GBMVRPGenerator
from optspread.market.generator import PriceGenerator
from optspread.market.priors import GBMVRPPriors


@dataclass(frozen=True, slots=True)
class WaveSpec:
    wave_id: int
    name: str
    make_generator: Callable[[], PriceGenerator]
    prediction: PreRegisteredPrediction | None


def wave1_spec(config: GBMConfig | None = None, priors: GBMVRPPriors | None = None) -> WaveSpec:
    cfg = config or GBMConfig()
    p = priors or GBMVRPPriors()
    return WaveSpec(
        wave_id=1,
        name="GBM + VRP",
        make_generator=lambda: GBMVRPGenerator.randomized(cfg, p),
        prediction=WAVE1_CREDIT_VRP,
    )


WAVES: dict[int, WaveSpec] = {1: wave1_spec()}
