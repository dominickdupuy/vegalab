"""European option pricer entry point for synthetic surfaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optspread.market.pricing.char_funcs import bates_cf, black_scholes_cf, heston_cf
from optspread.pricing.black_scholes import bs_price

_ComplexArray = NDArray[np.complex128]
_FloatArray = NDArray[np.float64]
_CharacteristicFunction = Callable[[_ComplexArray], _ComplexArray]

_MIN_VARIANCE = 1.0e-14
_NEAR_ZERO_VOL_OF_VOL = 1.0e-6


@dataclass(frozen=True, slots=True)
class COSPricer:
    """European option pricer using the Fang-Oosterlee COS expansion."""

    n_terms: int = 256
    truncation: float = 10.0

    def price_from_cf(
        self,
        right: str,
        *,
        strike: float,
        r: float,
        T: float,
        cf: _CharacteristicFunction,
        c1: float,
        c2: float,
    ) -> float:
        """Price a European option from a log-price characteristic function."""
        if self.n_terms < 2:
            raise ValueError("n_terms must be at least 2")
        if strike <= 0.0:
            raise ValueError("strike must be positive")
        if T < 0.0:
            raise ValueError("T must be non-negative")

        normalized_right = right.upper()
        if normalized_right not in {"C", "P"}:
            raise ValueError(f"right must be 'C' or 'P', got {right!r}")

        width = self.truncation * float(np.sqrt(max(c2, _MIN_VARIANCE)))
        a = c1 - width
        b = c1 + width
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            raise ValueError("invalid COS truncation interval")

        frequencies = np.arange(self.n_terms, dtype=np.float64) * np.pi / (b - a)
        u = np.asarray(frequencies, dtype=np.complex128)
        coefficients = _payoff_coefficients(
            normalized_right,
            strike=strike,
            a=a,
            b=b,
            u=frequencies,
        )

        weights = np.ones(self.n_terms, dtype=np.float64)
        weights[0] = 0.5
        cf_values = cf(u)
        terms = weights * np.real(cf_values * np.exp(-1j * u * a)) * coefficients
        price = float(np.exp(-r * T) * np.sum(terms))
        return max(price, 0.0)

    def black_scholes_price(
        self,
        right: str,
        *,
        spot: float,
        strike: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
    ) -> float:
        """Exact European Black-Scholes price kept as the Wave-1 hot-path baseline."""
        return float(bs_price(right, spot, strike, r, q, sigma, T))

    def black_scholes_cos_price(
        self,
        right: str,
        *,
        spot: float,
        strike: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
    ) -> float:
        """European Black-Scholes price through the generic COS core."""
        variance = sigma * sigma * T
        c1 = float(np.log(spot) + (r - q - 0.5 * sigma * sigma) * T)
        c2 = max(float(variance), _MIN_VARIANCE)

        def cf(u: _ComplexArray) -> _ComplexArray:
            return black_scholes_cf(u, spot=spot, r=r, q=q, sigma=sigma, T=T)

        return self.price_from_cf(right, strike=strike, r=r, T=T, cf=cf, c1=c1, c2=c2)

    def heston_price(
        self,
        right: str,
        *,
        spot: float,
        strike: float,
        r: float,
        q: float,
        T: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        v0: float,
    ) -> float:
        """European Heston price through the COS core."""
        normalized_right = right.upper()
        if normalized_right not in {"C", "P"}:
            raise ValueError(f"right must be 'C' or 'P', got {right!r}")
        _validate_positive_spot_strike(spot=spot, strike=strike)
        if T <= 0.0:
            return _intrinsic_value(normalized_right, spot=spot, strike=strike)
        _validate_heston_inputs(
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v,
            rho=rho,
            v0=v0,
        )
        if sigma_v <= _NEAR_ZERO_VOL_OF_VOL:
            return self._deterministic_variance_price(
                normalized_right,
                spot=spot,
                strike=strike,
                r=r,
                q=q,
                T=T,
                kappa=kappa,
                theta=theta,
                v0=v0,
            )

        c1, c2 = _heston_cumulants(
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

        def cf(u: _ComplexArray) -> _ComplexArray:
            return heston_cf(
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

        call_price = self.price_from_cf("C", strike=strike, r=r, T=T, cf=cf, c1=c1, c2=c2)
        if normalized_right == "C":
            return call_price
        return call_price - spot * float(np.exp(-q * T)) + strike * float(np.exp(-r * T))

    def heston_call_prices(
        self,
        *,
        spot: float,
        strikes: _FloatArray,
        r: float,
        q: float,
        T: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        v0: float,
    ) -> _FloatArray:
        """Vectorized Heston call prices for one maturity and many strikes."""
        if self.n_terms < 2:
            raise ValueError("n_terms must be at least 2")
        if spot <= 0.0 or np.any(strikes <= 0.0):
            raise ValueError("spot and strikes must be positive")
        if T <= 0.0:
            return np.maximum(spot - strikes, 0.0).astype(np.float64)
        _validate_heston_inputs(
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v,
            rho=rho,
            v0=v0,
        )
        if sigma_v <= _NEAR_ZERO_VOL_OF_VOL:
            integrated_variance = _deterministic_integrated_variance(
                T=T,
                kappa=kappa,
                theta=theta,
                v0=v0,
            )
            sigma = float(np.sqrt(max(integrated_variance / T, 0.0)))
            return np.asarray(bs_price("C", spot, strikes, r, q, sigma, T), dtype=np.float64)

        c1, c2 = _heston_cumulants(
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
        width = self.truncation * float(np.sqrt(max(c2, _MIN_VARIANCE)))
        a = c1 - width
        b = c1 + width
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            raise ValueError("invalid COS truncation interval")

        frequencies = np.arange(self.n_terms, dtype=np.float64) * np.pi / (b - a)
        u = np.asarray(frequencies, dtype=np.complex128)
        weights = np.ones(self.n_terms, dtype=np.float64)
        weights[0] = 0.5
        cf_values = heston_cf(
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
        basis = weights * np.real(cf_values * np.exp(-1j * u * a))
        coefficients = np.asarray(
            [
                _payoff_coefficients("C", strike=float(strike), a=a, b=b, u=frequencies)
                for strike in strikes
            ],
            dtype=np.float64,
        )
        prices = np.asarray(np.exp(-r * T) * (coefficients @ basis), dtype=np.float64)
        return np.asarray(np.maximum(prices, 0.0), dtype=np.float64)

    def bates_price(
        self,
        right: str,
        *,
        spot: float,
        strike: float,
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
    ) -> float:
        """European Bates price through the COS core."""
        normalized_right = right.upper()
        if normalized_right not in {"C", "P"}:
            raise ValueError(f"right must be 'C' or 'P', got {right!r}")
        _validate_positive_spot_strike(spot=spot, strike=strike)
        if T <= 0.0:
            return _intrinsic_value(normalized_right, spot=spot, strike=strike)

        c1, c2 = _bates_cumulants(
            spot=spot,
            r=r,
            q=q,
            T=T,
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v,
            rho=rho,
            v0=v0,
            jump_lambda=jump_lambda,
            jump_mu=jump_mu,
            jump_sigma=jump_sigma,
        )

        if sigma_v <= _NEAR_ZERO_VOL_OF_VOL:

            def cf(u: _ComplexArray) -> _ComplexArray:
                return _deterministic_variance_bates_cf(
                    u,
                    spot=spot,
                    r=r,
                    q=q,
                    T=T,
                    kappa=kappa,
                    theta=theta,
                    v0=v0,
                    jump_lambda=jump_lambda,
                    jump_mu=jump_mu,
                    jump_sigma=jump_sigma,
                )

        else:

            def cf(u: _ComplexArray) -> _ComplexArray:
                return bates_cf(
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
                    jump_lambda=jump_lambda,
                    jump_mu=jump_mu,
                    jump_sigma=jump_sigma,
                )

        call_price = self.price_from_cf("C", strike=strike, r=r, T=T, cf=cf, c1=c1, c2=c2)
        if normalized_right == "C":
            return call_price
        return call_price - spot * float(np.exp(-q * T)) + strike * float(np.exp(-r * T))

    def _deterministic_variance_price(
        self,
        right: str,
        *,
        spot: float,
        strike: float,
        r: float,
        q: float,
        T: float,
        kappa: float,
        theta: float,
        v0: float,
    ) -> float:
        integrated_variance = _deterministic_integrated_variance(
            T=T, kappa=kappa, theta=theta, v0=v0
        )
        sigma = float(np.sqrt(max(integrated_variance / T, 0.0)))
        return float(bs_price(right, spot, strike, r, q, sigma, T))


def _payoff_coefficients(
    right: str, *, strike: float, a: float, b: float, u: _FloatArray
) -> _FloatArray:
    log_strike = float(np.log(strike))
    if right == "C":
        if log_strike >= b:
            return np.zeros_like(u, dtype=np.float64)
        lower = max(log_strike, a)
        return (2.0 / (b - a)) * (
            _chi(a=a, lower=lower, upper=b, u=u) - strike * _psi(a=a, lower=lower, upper=b, u=u)
        )

    if log_strike <= a:
        return np.zeros_like(u, dtype=np.float64)
    upper = min(log_strike, b)
    return (2.0 / (b - a)) * (
        strike * _psi(a=a, lower=a, upper=upper, u=u) - _chi(a=a, lower=a, upper=upper, u=u)
    )


def _chi(*, a: float, lower: float, upper: float, u: _FloatArray) -> _FloatArray:
    upper_shift = u * (upper - a)
    lower_shift = u * (lower - a)
    numerator = np.exp(upper) * (np.cos(upper_shift) + u * np.sin(upper_shift)) - np.exp(lower) * (
        np.cos(lower_shift) + u * np.sin(lower_shift)
    )
    return np.asarray(numerator / (1.0 + u * u), dtype=np.float64)


def _psi(*, a: float, lower: float, upper: float, u: _FloatArray) -> _FloatArray:
    out = np.empty_like(u, dtype=np.float64)
    out[0] = upper - lower
    out[1:] = (np.sin(u[1:] * (upper - a)) - np.sin(u[1:] * (lower - a))) / u[1:]
    return out


def _heston_cumulants(
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
) -> tuple[float, float]:
    _validate_positive_spot_strike(spot=spot, strike=1.0)
    _validate_heston_inputs(
        kappa=kappa,
        theta=theta,
        sigma_v=sigma_v,
        rho=rho,
        v0=v0,
    )

    exp_kT = float(np.exp(-kappa * T))
    integrated_variance = _deterministic_integrated_variance(T=T, kappa=kappa, theta=theta, v0=v0)
    c1 = float(np.log(spot) + (r - q) * T - 0.5 * integrated_variance)
    if sigma_v <= _NEAR_ZERO_VOL_OF_VOL:
        return c1, max(integrated_variance, _MIN_VARIANCE)

    c2 = (
        sigma_v * T * kappa * exp_kT * (v0 - theta) * (8.0 * kappa * rho - 4.0 * sigma_v)
        + kappa * rho * sigma_v * (1.0 - exp_kT) * (16.0 * theta - 8.0 * v0)
        + 2.0 * theta * kappa * T * (-4.0 * kappa * rho * sigma_v + sigma_v**2 + 4.0 * kappa**2)
        + sigma_v**2 * ((theta - 2.0 * v0) * exp_kT**2 + theta * (6.0 * exp_kT - 7.0) + 2.0 * v0)
        + 8.0 * kappa**2 * (v0 - theta) * (1.0 - exp_kT)
    ) / (8.0 * kappa**3)
    if not np.isfinite(c2) or c2 <= 0.0:
        c2 = integrated_variance
    return c1, max(float(c2), _MIN_VARIANCE)


def _bates_cumulants(
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
) -> tuple[float, float]:
    if jump_lambda < 0.0:
        raise ValueError("jump_lambda must be non-negative")
    if jump_sigma < 0.0:
        raise ValueError("jump_sigma must be non-negative")
    c1, c2 = _heston_cumulants(
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
    jump_compensator = float(np.exp(jump_mu + 0.5 * jump_sigma * jump_sigma) - 1.0)
    jump_c1 = jump_lambda * T * (jump_mu - jump_compensator)
    jump_c2 = jump_lambda * T * (jump_sigma * jump_sigma + jump_mu * jump_mu)
    return c1 + jump_c1, max(c2 + jump_c2, _MIN_VARIANCE)


def _deterministic_integrated_variance(*, T: float, kappa: float, theta: float, v0: float) -> float:
    if T <= 0.0:
        return 0.0
    return float(theta * T + (v0 - theta) * (1.0 - np.exp(-kappa * T)) / kappa)


def _deterministic_variance_bates_cf(
    u: _ComplexArray,
    *,
    spot: float,
    r: float,
    q: float,
    T: float,
    kappa: float,
    theta: float,
    v0: float,
    jump_lambda: float,
    jump_mu: float,
    jump_sigma: float,
) -> _ComplexArray:
    integrated_variance = _deterministic_integrated_variance(
        T=T,
        kappa=kappa,
        theta=theta,
        v0=v0,
    )
    mean = np.log(spot) + (r - q) * T - 0.5 * integrated_variance
    diffusion_cf = np.exp(1j * u * mean - 0.5 * integrated_variance * u * u)
    jump_compensator = np.exp(jump_mu + 0.5 * jump_sigma * jump_sigma) - 1.0
    jump_cf = np.exp(
        jump_lambda
        * T
        * (
            np.exp(1j * u * jump_mu - 0.5 * jump_sigma * jump_sigma * u * u)
            - 1.0
            - 1j * u * jump_compensator
        )
    )
    return np.asarray(diffusion_cf * jump_cf, dtype=np.complex128)


def _validate_heston_inputs(
    *,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    v0: float,
) -> None:
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    if theta < 0.0 or v0 < 0.0:
        raise ValueError("theta and v0 must be non-negative")
    if sigma_v < 0.0:
        raise ValueError("sigma_v must be non-negative")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be in [-1, 1]")


def _validate_positive_spot_strike(*, spot: float, strike: float) -> None:
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive")


def _intrinsic_value(right: str, *, spot: float, strike: float) -> float:
    if right == "C":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)
