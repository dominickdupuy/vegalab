"""Risk functionals over learned return distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class RiskMeasure:
    """Mean, lower-tail CVaR, or spectral Mean-CVaR action-value functional.

    ``mean_cvar`` is the two-point spectral risk measure
    ``w * E[Z] + (1 - w) * CVaR_alpha(Z)`` — the literature's direct remedy for
    CVaR's "blindness to success" (pure tail weighting assigns zero weight to
    the upside that justifies trading; see MODEL_CANDIDATES_RESEARCH.md, C1).
    """

    name: str
    alpha: float = 1.0
    mean_weight: float = 0.0

    @classmethod
    def mean(cls) -> RiskMeasure:
        return cls("mean", 1.0)

    @classmethod
    def cvar(cls, alpha: float) -> RiskMeasure:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("CVaR alpha must be in (0, 1]")
        return cls("cvar", alpha)

    @classmethod
    def mean_cvar(cls, alpha: float, mean_weight: float) -> RiskMeasure:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("CVaR alpha must be in (0, 1]")
        if not 0.0 <= mean_weight <= 1.0:
            raise ValueError("mean_weight must be in [0, 1]")
        return cls("mean_cvar", alpha, mean_weight)

    @classmethod
    def upper_cvar(cls, beta: float) -> RiskMeasure:
        """Optimistic upper-tail mean over the top ``beta`` of the distribution.

        ``beta=1`` reduces to the mean. Used as an exploration (behavior-policy)
        risk attitude, never for deployment.
        """
        if not 0.0 < beta <= 1.0:
            raise ValueError("upper-CVaR beta must be in (0, 1]")
        return cls("upper_cvar", beta)

    def from_quantiles(
        self,
        values: NDArray[np.float64],
        taus: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Apply the risk measure to the last axis of quantile values."""
        arr = np.asarray(values, dtype=np.float64)
        if self.name == "mean" or (self.name != "mean_cvar" and self.alpha >= 1.0):
            return np.asarray(arr.mean(axis=-1), dtype=np.float64)
        if self.name == "upper_cvar":
            if taus is not None:
                tau = np.asarray(taus, dtype=np.float64)
                mask = tau >= 1.0 - self.alpha
                if not np.any(mask):
                    mask[np.argmax(tau)] = True
                return np.asarray(arr[..., mask].mean(axis=-1), dtype=np.float64)
            ordered = np.sort(arr, axis=-1)
            k = max(1, int(np.floor(ordered.shape[-1] * self.alpha)))
            return np.asarray(ordered[..., -k:].mean(axis=-1), dtype=np.float64)
        cvar = self._cvar_from_quantiles(arr, taus)
        if self.name == "mean_cvar":
            blend = self.mean_weight * arr.mean(axis=-1) + (1.0 - self.mean_weight) * cvar
            return np.asarray(blend, dtype=np.float64)
        return cvar

    def _cvar_from_quantiles(
        self,
        arr: NDArray[np.float64],
        taus: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        if self.alpha >= 1.0:
            return np.asarray(arr.mean(axis=-1), dtype=np.float64)
        if taus is not None:
            tau = np.asarray(taus, dtype=np.float64)
            mask = tau <= self.alpha
            if not np.any(mask):
                mask[np.argmin(tau)] = True
            return np.asarray(arr[..., mask].mean(axis=-1), dtype=np.float64)
        ordered = np.sort(arr, axis=-1)
        k = max(1, int(np.floor(ordered.shape[-1] * self.alpha)))
        return np.asarray(ordered[..., :k].mean(axis=-1), dtype=np.float64)

    def from_samples(self, values: NDArray[np.float64]) -> float:
        """Apply the risk measure to raw samples."""
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return 0.0
        if self.name == "mean" or (self.name != "mean_cvar" and self.alpha >= 1.0):
            return float(arr.mean())
        ordered = np.sort(arr)
        k = max(1, int(np.floor(ordered.size * self.alpha)))
        cvar = float(ordered[:k].mean())
        if self.name == "mean_cvar":
            return self.mean_weight * float(arr.mean()) + (1.0 - self.mean_weight) * cvar
        return cvar

    def torch_from_quantiles(
        self, values: torch.Tensor, taus: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Torch equivalent over the last axis."""
        if self.name == "mean" or (self.name != "mean_cvar" and self.alpha >= 1.0):
            return values.mean(dim=-1)
        if taus is not None:
            mask = taus <= self.alpha
            if not bool(mask.any()):
                mask[torch.argmin(taus)] = True
            cvar = values[..., mask].mean(dim=-1)
        else:
            ordered, _ = torch.sort(values, dim=-1)
            k = max(1, int(np.floor(ordered.shape[-1] * self.alpha)))
            cvar = ordered[..., :k].mean(dim=-1)
        if self.name == "mean_cvar":
            return self.mean_weight * values.mean(dim=-1) + (1.0 - self.mean_weight) * cvar
        return cvar
