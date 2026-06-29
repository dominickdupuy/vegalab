"""Quantile Huber loss for QR-DQN/IQN.

Implements Dabney et al.'s quantile-regression loss with Huber smoothing. The
core is pure tensor algebra so it can be tested against hand calculations.
"""

from __future__ import annotations

import torch


def quantile_midpoints(n_quantiles: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return QR-DQN fixed quantile fractions ``(2i-1)/(2N)``."""
    i = torch.arange(1, n_quantiles + 1, dtype=torch.float32, device=device)
    return (2.0 * i - 1.0) / (2.0 * n_quantiles)


def huber_loss(u: torch.Tensor, *, kappa: float = 1.0) -> torch.Tensor:
    """Huber loss ``0.5*u^2`` near zero, linear outside ``kappa``."""
    abs_u = u.abs()
    return torch.where(abs_u <= kappa, 0.5 * u.pow(2), kappa * (abs_u - 0.5 * kappa))


def quantile_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    taus: torch.Tensor,
    *,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Scalar quantile-Huber loss.

    Parameters
    ----------
    pred:
        Predicted quantiles, shape ``(batch, n_pred)``.
    target:
        Target quantiles, shape ``(batch, n_target)``.
    taus:
        Quantile fractions for ``pred``, shape ``(n_pred,)`` or
        ``(batch, n_pred)``.
    """
    per_sample = quantile_huber_loss_per_sample(pred, target, taus, kappa=kappa)
    return per_sample.mean()


def quantile_huber_loss_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    taus: torch.Tensor,
    *,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Per-batch quantile-Huber losses, shape ``(batch,)``."""
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError("pred and target must be rank-2 tensors")
    if pred.shape[0] != target.shape[0]:
        raise ValueError("pred and target batch sizes must match")

    u = target.unsqueeze(1) - pred.unsqueeze(2)  # (B, N_pred, N_target)
    if taus.ndim == 1:
        tau = taus.view(1, -1, 1)
    elif taus.ndim == 2:
        tau = taus.unsqueeze(2)
    else:
        raise ValueError("taus must be rank 1 or rank 2")
    weight = (tau - (u.detach() < 0.0).float()).abs()
    loss = weight * huber_loss(u, kappa=kappa) / kappa
    return loss.sum(dim=1).mean(dim=1)
