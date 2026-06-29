"""Small supervised distribution-recovery checks for QR-DQN and IQN."""

from __future__ import annotations

import torch

from optspread.agents.distributional.network_iqn import IQNNetwork
from optspread.agents.distributional.network_qrdqn import QRDQNNetwork
from optspread.agents.distributional.quantile_loss import quantile_huber_loss


def test_qrdqn_recovers_static_quantiles() -> None:
    torch.manual_seed(0)
    net = QRDQNNetwork(obs_dim=1, n_actions=1, n_quantiles=8, hidden_sizes=(16,))
    opt = torch.optim.Adam(net.parameters(), lr=0.03)
    obs = torch.zeros((32, 1))
    target = torch.linspace(-1.0, 1.0, 8).repeat(32, 1)
    for _ in range(160):
        pred = net(obs)[:, 0, :]
        loss = quantile_huber_loss(pred, target, net.quantile_fractions)
        opt.zero_grad()
        loss.backward()
        opt.step()
    recovered = net(torch.zeros((1, 1)))[:, 0, :].detach().flatten()
    assert torch.mean(torch.abs(recovered - torch.linspace(-1.0, 1.0, 8))).item() < 0.25


def test_iqn_recovers_linear_quantile_function() -> None:
    torch.manual_seed(0)
    net = IQNNetwork(obs_dim=1, n_actions=1, hidden_size=16, cosine_embed_dim=16)
    opt = torch.optim.Adam(net.parameters(), lr=0.03)
    obs = torch.ones((32, 1))
    for _ in range(220):
        taus = torch.rand((32, 8))
        target = 2.0 * taus - 1.0
        pred = net.action_quantiles(obs, taus)[:, 0, :]
        loss = torch.mean((pred - target) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    grid = torch.linspace(0.1, 0.9, 9).view(1, -1)
    recovered = net.action_quantiles(torch.ones((1, 1)), grid)[:, 0, :].detach()
    assert torch.mean(torch.abs(recovered - (2.0 * grid - 1.0))).item() < 0.20
