"""Run Phase-4 generator-validation gates before curriculum training."""

from __future__ import annotations

import argparse

from optspread.config import GBMConfig
from optspread.eval.generator_validation import validate_wave1_vrp, validate_wave2_heston
from optspread.market.gbm_vrp import GBMVRPGenerator
from optspread.market.heston import HestonGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate synthetic curriculum generators")
    parser.add_argument("--wave", type=int, choices=(1, 2), default=1)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.0005)
    args = parser.parse_args()

    if args.wave == 1:
        result = validate_wave1_vrp(
            lambda: GBMVRPGenerator.randomized(GBMConfig()),
            episodes=args.episodes,
            threshold=args.threshold,
        )
        print(f"GV_1 Wave 1 VRP: {'PASS' if result.passed else 'FAIL'} - {result.reason}")
    else:
        result = validate_wave2_heston(
            lambda: HestonGenerator.randomized(GBMConfig()),
            episodes=args.episodes,
        )
        print(f"GV_2 Wave 2 Heston SV: {'PASS' if result.passed else 'FAIL'} - {result.reason}")
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
