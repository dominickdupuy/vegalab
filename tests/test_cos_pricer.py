"""Phase-4 pricer seam checks."""

from __future__ import annotations

import numpy as np
import pytest

from optspread.market.pricing.char_funcs import black_scholes_cf
from optspread.market.pricing.cos_pricer import COSPricer
from optspread.market.pricing.mc_oracle import black_scholes_mc_price, heston_mc_price
from optspread.pricing.black_scholes import bs_price


def test_cos_pricer_black_scholes_branch_matches_bs() -> None:
    pricer = COSPricer()
    got = pricer.black_scholes_price(
        "C", spot=100.0, strike=105.0, r=0.03, q=0.01, sigma=0.22, T=0.75
    )
    expected = bs_price("C", 100.0, 105.0, 0.03, 0.01, 0.22, 0.75)
    assert got == expected


@pytest.mark.parametrize(
    ("right", "strike", "maturity"),
    [
        ("C", 85.0, 0.25),
        ("P", 95.0, 0.5),
        ("C", 100.0, 1.0),
        ("P", 115.0, 1.75),
        ("C", 130.0, 2.0),
    ],
)
def test_cos_core_with_black_scholes_cf_matches_exact(
    right: str, strike: float, maturity: float
) -> None:
    pricer = COSPricer(n_terms=256, truncation=10.0)
    got = pricer.black_scholes_cos_price(
        right,
        spot=100.0,
        strike=strike,
        r=0.03,
        q=0.01,
        sigma=0.22,
        T=maturity,
    )
    expected = bs_price(right, 100.0, strike, 0.03, 0.01, 0.22, maturity)
    assert abs(got - expected) < 1.0e-4


def test_black_scholes_mc_oracle_close_to_exact_price() -> None:
    exact = bs_price("P", 100.0, 95.0, 0.03, 0.0, 0.20, 0.5)
    mc = black_scholes_mc_price(
        "P",
        spot=100.0,
        strike=95.0,
        r=0.03,
        q=0.0,
        sigma=0.20,
        T=0.5,
        n_paths=200_000,
        seed=123,
    )
    assert abs(mc - exact) < 0.05


def test_black_scholes_characteristic_function_at_zero_is_one() -> None:
    cf0 = black_scholes_cf(
        np.asarray([0.0 + 0.0j]),
        spot=100.0,
        r=0.03,
        q=0.0,
        sigma=0.2,
        T=1.0,
    )
    assert np.allclose(cf0, 1.0 + 0.0j)


def test_heston_cos_near_zero_vol_of_vol_matches_black_scholes() -> None:
    pricer = COSPricer(n_terms=512, truncation=12.0)
    sigma = 0.20
    got = pricer.heston_price(
        "C",
        spot=100.0,
        strike=103.0,
        r=0.02,
        q=0.01,
        T=0.75,
        kappa=1.4,
        theta=sigma * sigma,
        sigma_v=1.0e-6,
        rho=-0.5,
        v0=sigma * sigma,
    )
    expected = bs_price("C", 100.0, 103.0, 0.02, 0.01, sigma, 0.75)
    assert abs(got - expected) < 1.0e-4


def test_heston_cos_matches_fang_oosterlee_published_call_benchmark() -> None:
    pricer = COSPricer(n_terms=256, truncation=10.0)
    got = pricer.heston_price(
        "C",
        spot=100.0,
        strike=100.0,
        r=0.0,
        q=0.0,
        T=1.0,
        kappa=1.5768,
        theta=0.0398,
        sigma_v=0.5751,
        rho=-0.5711,
        v0=0.0175,
    )
    print(f"Fang-Oosterlee Heston COS call benchmark: {got:.12f} vs 5.785000000000")
    assert abs(got - 5.785) < 1.0e-2


@pytest.mark.parametrize(
    ("strike", "maturity", "seed"),
    [
        (90.0, 0.5, 1090),
        (100.0, 1.0, 1100),
        (110.0, 1.5, 1110),
    ],
)
def test_heston_cos_agrees_with_seeded_mc_oracle(strike: float, maturity: float, seed: int) -> None:
    pricer = COSPricer(n_terms=512, truncation=12.0)
    params = {
        "spot": 100.0,
        "strike": strike,
        "r": 0.01,
        "q": 0.0,
        "T": maturity,
        "kappa": 1.8,
        "theta": 0.04,
        "sigma_v": 0.45,
        "rho": -0.65,
        "v0": 0.035,
    }
    cos_price = pricer.heston_price("C", **params)
    mc_price, mc_se = heston_mc_price(
        "C",
        **params,
        n_paths=200_000,
        n_steps=250,
        seed=seed,
    )
    print(
        "Heston COS vs MC "
        f"K={strike:.0f} T={maturity:.2f}: COS={cos_price:.6f}, "
        f"MC={mc_price:.6f}, SE={mc_se:.6f}"
    )
    assert abs(cos_price - mc_price) <= 3.0 * mc_se


def test_heston_cos_put_call_parity() -> None:
    pricer = COSPricer(n_terms=512, truncation=12.0)
    params = {
        "spot": 102.0,
        "strike": 97.0,
        "r": 0.035,
        "q": 0.012,
        "T": 1.25,
        "kappa": 1.6,
        "theta": 0.045,
        "sigma_v": 0.50,
        "rho": -0.55,
        "v0": 0.03,
    }
    call = pricer.heston_price("C", **params)
    put = pricer.heston_price("P", **params)
    parity = params["spot"] * np.exp(-params["q"] * params["T"]) - params["strike"] * np.exp(
        -params["r"] * params["T"]
    )
    assert abs(call - put - parity) < 1.0e-6
