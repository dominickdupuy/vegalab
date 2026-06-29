"""Characteristic functions used by synthetic pricers.

The Black-Scholes characteristic function is used by the COS/MC tests. Heston and
Bates characteristic functions are provided for later waves; they are not used by
Wave 1's hot path.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_C = NDArray[np.complex128]


def black_scholes_cf(
    u: _C,
    *,
    spot: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
) -> _C:
    """Characteristic function of ``log(S_T)`` under Black-Scholes."""
    x0 = np.log(spot)
    mean = x0 + (r - q - 0.5 * sigma * sigma) * T
    return np.asarray(np.exp(1j * u * mean - 0.5 * sigma * sigma * T * u * u), dtype=np.complex128)


def heston_cf(
    u: _C,
    *,
    spot: float,
    r: float,
    q: float,
    T: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    v0: float,
) -> _C:
    """Heston log-price characteristic function under the risk-neutral measure."""
    x0 = np.log(spot)
    iu = 1j * u
    d = np.sqrt((rho * sigma_v * iu - kappa) ** 2 + sigma_v**2 * (iu + u**2))
    g = (kappa - rho * sigma_v * iu - d) / (kappa - rho * sigma_v * iu + d)
    exp_dt = np.exp(-d * T)
    one_minus_g_exp = 1.0 - g * exp_dt
    one_minus_g = 1.0 - g
    C = (r - q) * iu * T + (kappa * theta / sigma_v**2) * (
        (kappa - rho * sigma_v * iu - d) * T - 2.0 * np.log(one_minus_g_exp / one_minus_g)
    )
    D = ((kappa - rho * sigma_v * iu - d) / sigma_v**2) * ((1.0 - exp_dt) / one_minus_g_exp)
    return np.asarray(np.exp(C + D * v0 + iu * x0), dtype=np.complex128)


def bates_cf(
    u: _C,
    *,
    spot: float,
    r: float,
    q: float,
    T: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    v0: float,
    jump_lambda: float,
    jump_mu: float,
    jump_sigma: float,
) -> _C:
    """Bates characteristic function = Heston times compensated lognormal jumps."""
    heston = heston_cf(
        u,
        spot=spot,
        r=r,
        q=q,
        T=T,
        kappa=kappa,
        theta=theta,
        sigma_v=sigma_v,
        rho=rho,
        v0=v0,
    )
    compensator = np.exp(jump_mu + 0.5 * jump_sigma**2) - 1.0
    jump_cf = np.exp(
        jump_lambda
        * T
        * (np.exp(1j * u * jump_mu - 0.5 * jump_sigma**2 * u * u) - 1.0 - 1j * u * compensator)
    )
    return np.asarray(heston * jump_cf, dtype=np.complex128)
