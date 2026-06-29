"""Acceptance tests for pricing/strike_solver.py."""

from __future__ import annotations

import pytest

from optspread.pricing.black_scholes import bs_delta
from optspread.pricing.strike_solver import strike_from_delta

BUCKETS = [0.10, 0.16, 0.25, 0.40]


@pytest.mark.parametrize("right", ["C", "P"])
@pytest.mark.parametrize("bucket", BUCKETS)
def test_recovered_delta_matches_target(right: str, bucket: float) -> None:
    S, r, q, sigma, T = 5000.0, 0.05, 0.0, 0.2, 21 / 252
    K = strike_from_delta(right, bucket, S, r, q, sigma, T)
    recovered = abs(float(bs_delta(right, S, K, r, q, sigma, T)))
    assert recovered == pytest.approx(bucket, abs=1e-4)


def test_call_otm_strike_above_spot() -> None:
    S = 5000.0
    # A 0.16-delta call is OTM => strike above spot.
    K = strike_from_delta("C", 0.16, S, 0.05, 0.0, 0.2, 21 / 252)
    assert K > S


def test_put_otm_strike_below_spot() -> None:
    S = 5000.0
    K = strike_from_delta("P", 0.16, S, 0.05, 0.0, 0.2, 21 / 252)
    assert K < S


def test_lower_delta_is_further_otm() -> None:
    S = 5000.0
    k10 = strike_from_delta("C", 0.10, S, 0.05, 0.0, 0.2, 21 / 252)
    k40 = strike_from_delta("C", 0.40, S, 0.05, 0.0, 0.2, 21 / 252)
    assert k10 > k40  # lower-delta call is a higher strike


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        strike_from_delta("C", 0.0, 5000.0, 0.05, 0.0, 0.2, 0.1)
    with pytest.raises(ValueError):
        strike_from_delta("X", 0.16, 5000.0, 0.05, 0.0, 0.2, 0.1)
    with pytest.raises(ValueError):
        strike_from_delta("C", 0.16, 5000.0, 0.05, 0.0, 0.2, 0.0)
