"""Quantile Huber loss hand checks."""

from __future__ import annotations

import torch

from optspread.agents.distributional.quantile_loss import (
    quantile_huber_loss,
    quantile_midpoints,
)


def test_quantile_midpoints() -> None:
    taus = quantile_midpoints(4)
    assert torch.allclose(taus, torch.tensor([0.125, 0.375, 0.625, 0.875]))


def test_quantile_huber_asymmetry() -> None:
    pred = torch.tensor([[0.0]])
    tau = torch.tensor([0.25])
    positive_error = quantile_huber_loss(pred, torch.tensor([[1.0]]), tau)
    negative_error = quantile_huber_loss(pred, torch.tensor([[-1.0]]), tau)
    # Huber(±1)=0.5; weights are tau=0.25 for positive u and 1-tau=0.75 for negative.
    assert positive_error.item() == torch.tensor(0.125).item()
    assert negative_error.item() == torch.tensor(0.375).item()
