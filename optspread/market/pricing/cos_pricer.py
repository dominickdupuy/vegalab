"""European option pricer entry point for synthetic surfaces.

The hot-path interface is intentionally model-neutral. Wave 1 uses the exact
Black-Scholes specialization; later Heston/Bates waves can swap the characteristic
function implementation behind the same surface-generation seam.
"""

from __future__ import annotations

from dataclasses import dataclass

from optspread.pricing.black_scholes import bs_price


@dataclass(frozen=True, slots=True)
class COSPricer:
    """European pricer facade.

    The current Phase-4 Wave-1 implementation uses the exact Black-Scholes branch
    for validation and surface generation. The class name and call shape reserve
    the COS-method seam required by later characteristic-function generators.
    """

    n_terms: int = 256
    truncation: float = 10.0

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
        """Exact European Black-Scholes price used as the Wave-1 COS baseline."""
        return float(bs_price(right, spot, strike, r, q, sigma, T))
