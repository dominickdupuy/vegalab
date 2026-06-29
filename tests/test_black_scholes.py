"""Acceptance tests for pricing/black_scholes.py."""

from __future__ import annotations

import numpy as np
import pytest

from optspread.pricing.black_scholes import (
    bs_delta,
    bs_gamma,
    bs_price,
    bs_theta,
    bs_vega,
)


def test_golden_value_atm_call() -> None:
    # S=100,K=100,r=0,q=0,sigma=0.2,T=1 => call ~ 7.9656
    c = bs_price("C", 100.0, 100.0, 0.0, 0.0, 0.2, 1.0)
    assert c == pytest.approx(7.9656, abs=1e-3)


def test_put_equals_call_at_atm_zero_rate() -> None:
    c = bs_price("C", 100.0, 100.0, 0.0, 0.0, 0.2, 1.0)
    p = bs_price("P", 100.0, 100.0, 0.0, 0.0, 0.2, 1.0)
    assert p == pytest.approx(c, abs=1e-10)


def test_put_call_parity_grid() -> None:
    S = np.array([80.0, 100.0, 120.0])
    K = np.array([90.0, 100.0, 110.0])
    r, q, sigma, T = 0.03, 0.01, 0.25, 0.75
    c = bs_price("C", S, K, r, q, sigma, T)
    p = bs_price("P", S, K, r, q, sigma, T)
    lhs = c - p
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    np.testing.assert_allclose(lhs, rhs, atol=1e-10)


@pytest.mark.parametrize("right", ["C", "P"])
def test_delta_matches_finite_difference(right: str) -> None:
    S, K, r, q, sigma, T = 100.0, 95.0, 0.02, 0.01, 0.3, 0.5
    h = 1e-4
    fd = (bs_price(right, S + h, K, r, q, sigma, T) - bs_price(right, S - h, K, r, q, sigma, T)) / (
        2 * h
    )
    analytic = bs_delta(right, S, K, r, q, sigma, T)
    assert analytic == pytest.approx(float(fd), abs=1e-5)


def test_gamma_matches_finite_difference() -> None:
    S, K, r, q, sigma, T = 100.0, 95.0, 0.02, 0.01, 0.3, 0.5
    h = 1e-2
    fd = (
        bs_price("C", S + h, K, r, q, sigma, T)
        - 2 * bs_price("C", S, K, r, q, sigma, T)
        + bs_price("C", S - h, K, r, q, sigma, T)
    ) / (h * h)
    assert bs_gamma(S, K, r, q, sigma, T) == pytest.approx(float(fd), abs=1e-4)


def test_vega_matches_finite_difference() -> None:
    S, K, r, q, sigma, T = 100.0, 95.0, 0.02, 0.01, 0.3, 0.5
    h = 1e-5
    fd = (bs_price("C", S, K, r, q, sigma + h, T) - bs_price("C", S, K, r, q, sigma - h, T)) / (
        2 * h
    )
    assert bs_vega(S, K, r, q, sigma, T) == pytest.approx(float(fd), abs=1e-3)


@pytest.mark.parametrize("right", ["C", "P"])
def test_theta_matches_finite_difference(right: str) -> None:
    # theta = dPrice/d(calendar time) = -dPrice/dT_to_expiry
    S, K, r, q, sigma, T = 100.0, 95.0, 0.02, 0.01, 0.3, 0.5
    h = 1e-5
    dprice_dTexp = (
        bs_price(right, S, K, r, q, sigma, T + h) - bs_price(right, S, K, r, q, sigma, T - h)
    ) / (2 * h)
    assert bs_theta(right, S, K, r, q, sigma, T) == pytest.approx(-float(dprice_dTexp), abs=1e-2)


def test_delta_bounds_and_monotonic() -> None:
    S = np.linspace(60, 140, 81)
    K, r, q, sigma, T = 100.0, 0.0, 0.0, 0.2, 1.0
    cd = bs_delta("C", S, K, r, q, sigma, T)
    pd = bs_delta("P", S, K, r, q, sigma, T)
    assert np.all(cd >= 0.0) and np.all(cd <= 1.0)
    assert np.all(pd <= 0.0) and np.all(pd >= -1.0)
    # call delta strictly increasing in spot
    assert np.all(np.diff(cd) > 0)
    assert np.all(np.diff(pd) > 0)
    # put-call delta relation at q=0: dC - dP = 1
    np.testing.assert_allclose(cd - pd, 1.0, atol=1e-10)


@pytest.mark.parametrize("right", ["C", "P"])
def test_t_to_zero_returns_intrinsic(right: str) -> None:
    for S in [80.0, 100.0, 120.0]:
        val = bs_price(right, S, 100.0, 0.01, 0.0, 0.2, 0.0)
        intrinsic = max(S - 100.0, 0.0) if right == "C" else max(100.0 - S, 0.0)
        assert np.isfinite(val)
        assert val == pytest.approx(intrinsic, abs=1e-12)


def test_no_nan_for_extreme_inputs() -> None:
    for T in [0.0, 1e-8, 10.0]:
        for sigma in [0.0, 1e-8, 2.0]:
            for S in [1.0, 100.0, 10000.0]:
                for fn in (
                    lambda S=S, sigma=sigma, T=T: bs_price("C", S, 100.0, 0.01, 0.0, sigma, T),
                    lambda S=S, sigma=sigma, T=T: bs_delta("P", S, 100.0, 0.01, 0.0, sigma, T),
                    lambda S=S, sigma=sigma, T=T: bs_gamma(S, 100.0, 0.01, 0.0, sigma, T),
                    lambda S=S, sigma=sigma, T=T: bs_vega(S, 100.0, 0.01, 0.0, sigma, T),
                    lambda S=S, sigma=sigma, T=T: bs_theta("C", S, 100.0, 0.01, 0.0, sigma, T),
                ):
                    assert np.all(np.isfinite(fn()))
