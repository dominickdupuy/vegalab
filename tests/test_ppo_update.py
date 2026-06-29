"""PPO update correctness: finite losses, no instant entropy collapse, and the
classic sanity check — overfit a trivial one-state bandit to its optimal action.

If PPO cannot solve a one-state bandit, the clipped-surrogate update, GAE wiring,
or optimizer step is wrong; everything downstream (the no-edge gate) would be
uninterpretable. This is the cheapest possible end-to-end correctness probe.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from optspread.agents.ppo.config import PPOConfig
from optspread.agents.ppo.ppo_agent import PPOAgent
from optspread.agents.ppo.trainer import PPOTrainer

OPTIMAL_ACTION = 2
N_BANDIT_ACTIONS = 3


class BanditEnv(gym.Env[NDArray[np.float32], np.int64]):
    """One-state, one-step bandit: reward 1 for the optimal action, else 0."""

    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_BANDIT_ACTIONS)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(
        self, action: np.int64 | int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        reward = 1.0 if int(action) == OPTIMAL_ACTION else 0.0
        return np.zeros(1, dtype=np.float32), reward, True, False, {}


class BanditFactory:
    """Duck-typed to the slice of EnvFactory the trainer uses (``make``)."""

    def make(self) -> BanditEnv:
        return BanditEnv()


def _config() -> PPOConfig:
    # KL-limited, gentle LR so learning is gradual (entropy must not collapse in
    # the first update) while still converging to the optimal arm over many iters.
    return PPOConfig(
        num_envs=8,
        num_steps=16,
        total_timesteps=8 * 16 * 200,  # 200 iterations
        num_minibatches=4,
        update_epochs=4,
        learning_rate=5e-3,
        ent_coef=0.0,
        gamma=0.99,
        target_kl=0.03,  # early-stop keeps each update from destroying the policy
        norm_obs=True,
        seed=0,
    )


def test_ppo_overfits_one_state_bandit() -> None:
    cfg = _config()
    agent = PPOAgent(obs_dim=1, n_actions=N_BANDIT_ACTIONS, config=cfg)
    trainer = PPOTrainer(agent, BanditFactory(), cfg, env_base_seed=12345)  # type: ignore[arg-type]
    history = trainer.train()

    # Loss components stay finite throughout.
    for s in history:
        assert math.isfinite(s.policy_loss)
        assert math.isfinite(s.value_loss)
        assert math.isfinite(s.entropy)

    # No instant collapse: entropy near log(3) early and still meaningful after one update.
    max_entropy = math.log(N_BANDIT_ACTIONS)
    assert history[0].entropy > 0.6 * max_entropy
    assert history[1].entropy > 0.2 * max_entropy

    # It learns: entropy falls and the greedy action is optimal.
    assert history[-1].entropy < history[0].entropy
    greedy = [agent.act(np.zeros(1, dtype=np.float32), deterministic=True) for _ in range(20)]
    assert all(a == OPTIMAL_ACTION for a in greedy)


def test_save_load_roundtrip(tmp_path: Any) -> None:
    cfg = _config()
    agent = PPOAgent(obs_dim=1, n_actions=N_BANDIT_ACTIONS, config=cfg)
    PPOTrainer(agent, BanditFactory(), cfg, env_base_seed=7).train()  # type: ignore[arg-type]
    path = tmp_path / "agent.pt"
    agent.save(path)

    restored = PPOAgent(obs_dim=1, n_actions=N_BANDIT_ACTIONS, config=cfg)
    restored.load(path)
    obs = np.zeros(1, dtype=np.float32)
    assert restored.act(obs, deterministic=True) == agent.act(obs, deterministic=True)
