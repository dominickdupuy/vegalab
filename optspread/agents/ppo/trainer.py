"""PPOTrainer: rollout collection + clipped-surrogate update loop.

Structure and update math follow CleanRL's ``ppo.py`` (Huang et al., 2022)
line-for-line; the differences are bookkeeping only: the vector env runs in
SAME_STEP autoreset, observations are normalized causally through the agent's
normalizer, and randomness is threaded (torch seed for action sampling, a
dedicated numpy ``Generator`` for minibatch shuffling) so a run is reproducible
without ever touching global ``np.random`` (Phase-1 determinism invariant).

The entropy bonus is part of the objective here — never the reward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from optspread.actions.library import N_ACTIONS
from optspread.agents.ppo.buffer import RolloutBuffer
from optspread.agents.ppo.config import PPOConfig
from optspread.agents.ppo.ppo_agent import PPOAgent
from optspread.training.env_factory import EnvFactory
from optspread.training.logging import MetricLogger
from optspread.training.seeding import seed_everything
from optspread.training.vec import make_vector_env


@dataclass(frozen=True, slots=True)
class UpdateStats:
    """Per-iteration diagnostics — the dials the brief says to watch."""

    global_step: int
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clipfrac: float
    explained_variance: float
    learning_rate: float
    mean_episodic_return: float


class PPOTrainer:
    """Trains a ``PPOAgent`` against a ``SpreadEnv`` vector via on-policy PPO."""

    def __init__(
        self,
        agent: PPOAgent,
        env_factory: EnvFactory,
        config: PPOConfig,
        *,
        logger: MetricLogger | None = None,
        env_base_seed: int | None = None,
    ) -> None:
        self.agent = agent
        self.env_factory = env_factory
        self.config = config
        self.logger = logger
        self.device = agent.device
        # Training path seeds live in their own space, kept away from eval seeds.
        self.env_base_seed = config.seed if env_base_seed is None else env_base_seed
        self._shuffle_rng = np.random.default_rng(config.seed)
        self.optimizer = torch.optim.Adam(
            agent.network.parameters(), lr=config.learning_rate, eps=1e-5
        )

    # -- public API -------------------------------------------------------- #

    def train(self) -> list[UpdateStats]:
        """Run the full schedule; return per-iteration stats."""
        cfg = self.config
        seed_everything(cfg.seed)
        envs = make_vector_env(self.env_factory, cfg.num_envs, base_seed=self.env_base_seed)
        buffer = RolloutBuffer(cfg.num_steps, cfg.num_envs, self.agent.obs_dim, self.device)

        next_obs = self._observe(envs.reset(seed=self.env_base_seed)[0])
        next_done = torch.zeros(cfg.num_envs, device=self.device)
        running_return = np.zeros(cfg.num_envs, dtype=np.float64)
        recent_returns: list[float] = []
        history: list[UpdateStats] = []
        global_step = 0

        for iteration in range(1, cfg.num_iterations + 1):
            if cfg.anneal_lr:
                frac = 1.0 - (iteration - 1) / cfg.num_iterations
                self.optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

            for step in range(cfg.num_steps):
                global_step += cfg.num_envs
                prev_obs, prev_done = next_obs, next_done
                with torch.no_grad():
                    action, logprob, _, value = self.agent.network.get_action_and_value(prev_obs)
                # gymnasium's vector step is loosely typed; annotate the unpack.
                obs_np: NDArray[np.float32]
                reward: NDArray[np.float32]
                term: NDArray[np.bool_]
                trunc: NDArray[np.bool_]
                obs_np, reward, term, trunc, _info = envs.step(action.cpu().numpy())
                done_np = np.logical_or(term, trunc)
                reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
                buffer.add(step, prev_obs, action, logprob, reward_t, prev_done, value)

                running_return += np.asarray(reward, dtype=np.float64)
                for i in range(cfg.num_envs):
                    if done_np[i]:
                        recent_returns.append(float(running_return[i]))
                        running_return[i] = 0.0

                next_obs = self._observe(obs_np)
                next_done = torch.as_tensor(done_np, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                next_value = self.agent.network.get_value(next_obs).reshape(-1)
            advantages, returns = buffer.compute_returns(
                next_value, next_done, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda
            )

            stats = self._update(buffer, advantages, returns, global_step, recent_returns)
            history.append(stats)
            if self.logger is not None:
                self._log(stats, buffer)
            recent_returns.clear()

        envs.close()
        return history

    # -- internals --------------------------------------------------------- #

    def _observe(self, raw: NDArray[np.float32]) -> torch.Tensor:
        """Causally update the obs normalizer with the new batch and return it normalized."""
        norm = self.agent.normalizer
        if norm is not None:
            norm.update(raw)
            raw = norm.normalize(raw)
        return torch.as_tensor(raw, dtype=torch.float32, device=self.device)

    def _update(
        self,
        buffer: RolloutBuffer,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        global_step: int,
        recent_returns: list[float],
    ) -> UpdateStats:
        cfg = self.config
        b = buffer.flatten(advantages, returns)
        b_inds = np.arange(cfg.batch_size)
        clipfracs: list[float] = []
        approx_kl = torch.zeros(1, device=self.device)
        pg_loss = v_loss = entropy_loss = torch.zeros(1, device=self.device)

        for _epoch in range(cfg.update_epochs):
            self._shuffle_rng.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                mb = b_inds[start : start + cfg.minibatch_size]
                _, newlogprob, entropy, newvalue = self.agent.network.get_action_and_value(
                    b["obs"][mb], b["actions"][mb]
                )
                logratio = newlogprob - b["logprobs"][mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                mb_adv = b["advantages"][mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_unclipped = (newvalue - b["returns"][mb]) ** 2
                    v_clipped = b["values"][mb] + torch.clamp(
                        newvalue - b["values"][mb], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b["returns"][mb]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b["returns"][mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.network.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

            if cfg.target_kl is not None and approx_kl.item() > cfg.target_kl:
                break

        explained_var = self._explained_variance(b["values"], b["returns"])
        mean_ret = float(np.mean(recent_returns)) if recent_returns else float("nan")
        return UpdateStats(
            global_step=global_step,
            policy_loss=float(pg_loss.item()),
            value_loss=float(v_loss.item()),
            entropy=float(entropy_loss.item()),
            approx_kl=float(approx_kl.item()),
            clipfrac=float(np.mean(clipfracs)) if clipfracs else 0.0,
            explained_variance=explained_var,
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            mean_episodic_return=mean_ret,
        )

    @staticmethod
    def _explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
        y_pred = values.cpu().numpy()
        y_true = returns.cpu().numpy()
        var_y = float(np.var(y_true))
        return float("nan") if var_y == 0 else 1.0 - float(np.var(y_true - y_pred)) / var_y

    def _log(self, stats: UpdateStats, buffer: RolloutBuffer) -> None:
        assert self.logger is not None
        step = stats.global_step
        self.logger.log_scalars(
            {
                "charts/learning_rate": stats.learning_rate,
                "charts/mean_episodic_return": stats.mean_episodic_return,
                "losses/policy_loss": stats.policy_loss,
                "losses/value_loss": stats.value_loss,
                "losses/entropy": stats.entropy,
                "losses/approx_kl": stats.approx_kl,
                "losses/clipfrac": stats.clipfrac,
                "losses/explained_variance": stats.explained_variance,
            },
            step,
        )
        actions = buffer.actions.cpu().numpy().reshape(-1)
        total = max(1, actions.size)
        for a in range(N_ACTIONS):
            self.logger.log_scalar(
                f"actions/freq_{a:02d}", float(np.sum(actions == a)) / total, step
            )
