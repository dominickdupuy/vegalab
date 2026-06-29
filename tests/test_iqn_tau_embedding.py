"""IQN tau embedding shape and tau sensitivity."""

from __future__ import annotations

import torch

from optspread.agents.distributional.network_iqn import IQNNetwork


def test_iqn_embedding_shapes_and_tau_sensitivity() -> None:
    torch.manual_seed(0)
    net = IQNNetwork(obs_dim=3, n_actions=2, hidden_size=16, cosine_embed_dim=8)
    obs = torch.ones((4, 3))
    taus_low = torch.full((4, 5), 0.1)
    taus_high = torch.full((4, 5), 0.9)
    emb = net.cosine_embedding(taus_low)
    assert emb.shape == (4, 5, 8)
    low = net.action_quantiles(obs, taus_low)
    high = net.action_quantiles(obs, taus_high)
    assert low.shape == (4, 2, 5)
    assert not torch.allclose(low, high)
