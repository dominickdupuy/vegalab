"""Risk functionals over learned return distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class RiskMeasure:
    """Mean or lower-tail CVaR action-value functional."""

    name: str
    alpha: float = 1.0

    @classmethod
    def mean(cls) -> RiskMeasure:
        return cls("mean", 1.0)

    @classmethod
    def cvar(cls, alpha: float) -> RiskMeasure:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("CVaR alpha must be in (0, 1]")
        return cls("cvar", alpha)

    def from_quantiles(
        self,
        values: NDArray[np.float64],
        taus: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Apply the risk measure to the last axis of quantile values."""
        arr = np.asarray(values, dtype=np.float64)
        if self.name == "mean" or self.alpha >= 1.0:
            return arr.mean(axis=-1)
        if taus is not None:
            tau = np.asarray(taus, dtype=np.float64)
            mask = tau <= self.alpha
            if not np.any(mask):
                mask[np.argmin(tau)] = True
            return arr[..., mask].mean(axis=-1)
        ordered = np.sort(arr, axis=-1)
        k = max(1, int(np.floor(ordered.shape[-1] * self.alpha)))
        return ordered[..., :k].mean(axis=-1)

    def from_samples(self, values: NDArray[np.float64]) -> float:
        """Apply the risk measure to raw samples."""
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return 0.0
        if self.name == "mean" or self.alpha >= 1.0:
            return float(arr.mean())
        ordered = np.sort(arr)
        k = max(1, int(np.floor(ordered.size * self.alpha)))
        return float(ordered[:k].mean())

    def torch_from_quantiles(
        self, values: torch.Tensor, taus: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Torch equivalent over the last axis."""
        if self.name == "mean" or self.alpha >= 1.0:
            return values.mean(dim=-1)
        if taus is not None:
            mask = taus <= self.alpha
            if not bool(mask.any()):
                mask[torch.argmin(taus)] = True
            return values[..., mask].mean(dim=-1)
        ordered, _ = torch.sort(values, dim=-1)
        k = max(1, int(np.floor(ordered.shape[-1] * self.alpha)))
        return ordered[..., :k].mean(dim=-1)
