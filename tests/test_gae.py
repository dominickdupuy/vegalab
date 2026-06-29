"""GAE against hand-computed examples — the easiest place for an off-by-one bug."""

from __future__ import annotations

import torch

from optspread.agents.ppo.buffer import compute_gae


def test_gae_undiscounted_full_lambda() -> None:
    """gamma=lambda=1, zero values: every advantage is the undiscounted reward-to-go."""
    rewards = torch.tensor([[0.0], [0.0], [1.0]])
    values = torch.zeros((3, 1))
    dones = torch.zeros((3, 1))
    adv, ret = compute_gae(
        rewards,
        values,
        dones,
        next_value=torch.zeros(1),
        next_done=torch.zeros(1),
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert torch.allclose(adv, torch.tensor([[1.0], [1.0], [1.0]]))
    assert torch.allclose(ret, torch.tensor([[1.0], [1.0], [1.0]]))


def test_gae_with_terminal_masks_bootstrap() -> None:
    """A done at t=2 must zero the bootstrap from t=1 into t=2.

    By hand (gamma=0.9, lambda=0.5, values=0, next_value=0):
      adv[2] = 3
      adv[1] = 2                      (nextnonterminal = 1 - dones[2] = 0)
      adv[0] = 1 + 0.9*0.5*1*2 = 1.9  (nextnonterminal = 1 - dones[1] = 1)
    """
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.zeros((3, 1))
    dones = torch.tensor([[0.0], [0.0], [1.0]])
    adv, ret = compute_gae(
        rewards,
        values,
        dones,
        next_value=torch.zeros(1),
        next_done=torch.zeros(1),
        gamma=0.9,
        gae_lambda=0.5,
    )
    assert torch.allclose(adv, torch.tensor([[1.9], [2.0], [3.0]]))
    assert torch.allclose(ret, torch.tensor([[1.9], [2.0], [3.0]]))


def test_gae_uses_value_baseline() -> None:
    """With nonzero values the advantage subtracts the baseline (single step)."""
    rewards = torch.tensor([[1.0]])
    values = torch.tensor([[0.4]])
    dones = torch.zeros((1, 1))
    adv, ret = compute_gae(
        rewards,
        values,
        dones,
        next_value=torch.tensor([2.0]),
        next_done=torch.zeros(1),
        gamma=0.5,
        gae_lambda=1.0,
    )
    # delta = 1 + 0.5*2*1 - 0.4 = 1.6 ; returns = adv + value = 2.0
    assert torch.allclose(adv, torch.tensor([[1.6]]))
    assert torch.allclose(ret, torch.tensor([[2.0]]))
