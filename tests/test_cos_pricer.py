"""Phase-4 pricer seam checks."""

from __future__ import annotations

import numpy as np

from optspread.market.pricing.char_funcs import black_scholes_cf
from optspread.market.pricing.cos_pricer import COSPricer
from optspread.market.pricing.mc_oracle import black_scholes_mc_price
from optspread.pricing.black_scholes import bs_price


def test_cos_pricer_black_scholes_branch_matches_bs() -> None:
    pricer = COSPricer()
    got = pricer.black_scholes_price(
        "C", spot=100.0, strike=105.0, r=0.03, q=0.01, sigma=0.22, T=0.75
    )
    expected = bs_price("C", 100.0, 105.0, 0.03, 0.01, 0.22, 0.75)
    assert got == expected


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
