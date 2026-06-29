"""ActorCritic: shapes, near-uniform init policy, distribution validity."""

from __future__ import annotations

import numpy as np
import torch

from optspread.agents.ppo.network import ActorCritic

OBS_DIM = 16
N_ACTIONS = 19


def test_action_and_value_shapes() -> None:
    net = ActorCritic(OBS_DIM, N_ACTIONS)
    x = torch.zeros((32, OBS_DIM))
    action, logprob, entropy, value = net.get_action_and_value(x)
    assert action.shape == (32,)
    assert logprob.shape == (32,)
    assert entropy.shape == (32,)
    assert value.shape == (32, 1)
    assert torch.all((action >= 0) & (action < N_ACTIONS))
    assert torch.all(torch.isfinite(logprob))


def test_policy_head_starts_near_uniform() -> None:
    """The 0.01 policy-head gain should make the initial policy near-uniform, so
    entropy starts close to log(N) — the anti-collapse precondition."""
    torch.manual_seed(0)
    net = ActorCritic(OBS_DIM, N_ACTIONS)
    x = torch.randn((256, OBS_DIM))
    _, _, entropy, _ = net.get_action_and_value(x)
    max_entropy = np.log(N_ACTIONS)
    assert entropy.mean().item() > 0.95 * max_entropy


def test_value_head_gain_larger_than_policy_head() -> None:
    net = ActorCritic(OBS_DIM, N_ACTIONS, (64, 64))
    actor_out = net.actor[-1].weight.std().item()
    critic_out = net.critic[-1].weight.std().item()
    # Policy head intentionally tiny (gain 0.01) vs value head (gain 1.0).
    assert actor_out < critic_out


def test_evaluate_fixed_actions_matches_logprob() -> None:
    net = ActorCritic(OBS_DIM, N_ACTIONS)
    x = torch.randn((8, OBS_DIM))
    action, logprob, _, _ = net.get_action_and_value(x)
    _, logprob2, _, _ = net.get_action_and_value(x, action)
    assert torch.allclose(logprob, logprob2)


def test_deterministic_act_is_argmax() -> None:
    net = ActorCritic(OBS_DIM, N_ACTIONS)
    x = torch.randn((10, OBS_DIM))
    greedy = net.act(x, deterministic=True)
    expected = torch.argmax(net._logits(x), dim=-1)
    assert torch.equal(greedy, expected)
