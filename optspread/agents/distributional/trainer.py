"""Off-policy distributional trainer for QR-DQN and IQN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from optspread.agents.distributional.config import DistributionalConfig, IQNConfig, QRDQNConfig
from optspread.agents.distributional.iqn_agent import IQNAgent
from optspread.agents.distributional.qrdqn_agent import QRDQNAgent
from optspread.agents.distributional.quantile_loss import quantile_huber_loss
from optspread.agents.distributional.replay import UniformReplayBuffer
from optspread.agents.distributional.risk import RiskMeasure
from optspread.training.env_factory import EnvFactory
from optspread.training.logging import MetricLogger
from optspread.training.seeding import seed_everything


class DistributionalAgent(Protocol):
    obs_dim: int
    n_actions: int
    device: torch.device
    risk_measure: RiskMeasure

    def sync_target(self) -> None: ...

    def prepare_obs(self, obs: NDArray[np.float32], *, update: bool) -> torch.Tensor: ...

    def greedy_actions(
        self,
        obs: torch.Tensor,
        *,
        risk: RiskMeasure,
        use_target: bool,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class DistributionalUpdateStats:
    global_step: int
    loss: float
    epsilon: float
    mean_reward: float
    replay_size: int
    quantile_crossing_frac: float


class DistributionalTrainer:
    """Single-env off-policy loop with target net and epsilon-greedy exploration."""

    def __init__(
        self,
        agent: DistributionalAgent,
        env_factory: EnvFactory,
        config: DistributionalConfig,
        *,
        logger: MetricLogger | None = None,
        env_seed: int | None = None,
    ) -> None:
        self.agent = agent
        self.env_factory = env_factory
        self.config = config
        self.logger = logger
        self.env_seed = config.seed if env_seed is None else env_seed
        self.rng = np.random.default_rng(config.seed)
        self.optimizer = torch.optim.Adam(self._parameters(), lr=config.learning_rate)
        self.replay = UniformReplayBuffer(
            config.replay_size,
            agent.obs_dim,
            reward_priority_boost=config.reward_priority_boost,
        )
        # The bootstrap target's next-action selection risk (see config docstring).
        self._target_risk = (
            RiskMeasure.mean()
            if config.bootstrap_risk == "mean"
            else RiskMeasure.cvar(config.cvar_alpha)
        )

    def train(self) -> list[DistributionalUpdateStats]:
        cfg = self.config
        seed_everything(cfg.seed)
        env = self.env_factory.make()
        obs, _info = env.reset(seed=self.env_seed)
        self.agent.prepare_obs(obs.reshape(1, -1), update=True)
        episode_reward = 0.0
        recent_rewards: list[float] = []
        history: list[DistributionalUpdateStats] = []
        last_loss = 0.0
        last_crossing = 0.0

        for step in range(1, cfg.total_timesteps + 1):
            epsilon = self._epsilon(step)
            action = self._select_behavior_action(obs, epsilon, risk=self.behavior_risk_at(step))
            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            self.agent.prepare_obs(next_obs.reshape(1, -1), update=True)
            self.replay.add(obs, action, float(reward), next_obs, done)
            episode_reward += float(reward)
            obs = next_obs

            if done:
                recent_rewards.append(episode_reward)
                episode_reward = 0.0
                obs, _info = env.reset(seed=int(self.rng.integers(0, 2**31 - 1)))
                self.agent.prepare_obs(obs.reshape(1, -1), update=True)

            if step >= cfg.learning_starts and step % cfg.train_freq == 0:
                for _ in range(cfg.gradient_steps):
                    last_loss, last_crossing = self._gradient_step()
            if step % cfg.target_update_interval == 0:
                self.agent.sync_target()
                stats = DistributionalUpdateStats(
                    global_step=step,
                    loss=last_loss,
                    epsilon=epsilon,
                    mean_reward=float(np.mean(recent_rewards)) if recent_rewards else float("nan"),
                    replay_size=self.replay.size,
                    quantile_crossing_frac=last_crossing,
                )
                history.append(stats)
                self._log(stats)
                recent_rewards.clear()

        env.close()
        return history

    def _parameters(self) -> list[torch.nn.Parameter]:
        if isinstance(self.agent, QRDQNAgent | IQNAgent):
            return list(self.agent.network.parameters())
        raise TypeError("unsupported distributional agent")

    def behavior_risk_at(self, step: int) -> RiskMeasure:
        """Behavior-policy risk attitude for this step.

        With ``behavior_seek_start`` set, exploration starts optimistic (upper-tail
        taus) and the band widens linearly to the full risk-neutral range — the
        Jung et al. risk-scheduling direction, keeping upside transitions in
        replay late into training. Otherwise falls back to ``behavior_risk``.
        """
        seek = self.config.behavior_seek_start
        if seek is not None:
            horizon = self.config.behavior_seek_decay_steps or self.config.total_timesteps
            frac = min(1.0, step / horizon)
            beta = seek + frac * (1.0 - seek)
            return RiskMeasure.upper_cvar(beta)
        if self.config.behavior_risk == "mean":
            return RiskMeasure.mean()
        return self.agent.risk_measure

    def _select_behavior_action(
        self,
        obs: NDArray[np.float32],
        epsilon: float,
        risk: RiskMeasure | None = None,
    ) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.agent.n_actions))
        if risk is None:
            risk = (
                RiskMeasure.mean()
                if self.config.behavior_risk == "mean"
                else self.agent.risk_measure
            )
        x = self.agent.prepare_obs(obs.reshape(1, -1), update=False)
        with torch.no_grad():
            return int(self.agent.greedy_actions(x, risk=risk, use_target=False).item())

    def _gradient_step(self) -> tuple[float, float]:
        batch = self.replay.sample(self.config.batch_size, self.rng)
        obs = self.agent.prepare_obs(batch.obs, update=False)
        next_obs = self.agent.prepare_obs(batch.next_obs, update=False)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.agent.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.agent.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.agent.device)

        if isinstance(self.agent, QRDQNAgent) and isinstance(self.config, QRDQNConfig):
            pred_all = self.agent.quantiles(obs, use_target=False)
            pred = pred_all[torch.arange(actions.shape[0], device=self.agent.device), actions]
            with torch.no_grad():
                next_actions = self.agent.greedy_actions(
                    next_obs, risk=self._target_risk, use_target=True
                )
                next_all = self.agent.quantiles(next_obs, use_target=True)
                next_q = next_all[
                    torch.arange(actions.shape[0], device=self.agent.device), next_actions
                ]
                target = (
                    rewards.unsqueeze(1) + self.config.gamma * (1.0 - dones.unsqueeze(1)) * next_q
                )
            taus = self.agent.network.quantile_fractions
            loss = quantile_huber_loss(pred, target, taus, kappa=self.config.huber_kappa)
            crossing = _crossing_fraction(pred.detach())
        elif isinstance(self.agent, IQNAgent) and isinstance(self.config, IQNConfig):
            taus = torch.rand((actions.shape[0], self.config.n_quantiles), device=self.agent.device)
            pred_all = self.agent.quantiles(obs, taus, use_target=False)
            pred = pred_all[torch.arange(actions.shape[0], device=self.agent.device), actions]
            with torch.no_grad():
                next_actions = self.agent.greedy_actions(
                    next_obs, risk=self._target_risk, use_target=True
                )
                target_taus = torch.rand(
                    (actions.shape[0], self.config.n_target_quantiles), device=self.agent.device
                )
                next_all = self.agent.quantiles(next_obs, target_taus, use_target=True)
                next_q = next_all[
                    torch.arange(actions.shape[0], device=self.agent.device), next_actions
                ]
                target = (
                    rewards.unsqueeze(1) + self.config.gamma * (1.0 - dones.unsqueeze(1)) * next_q
                )
            loss = quantile_huber_loss(pred, target, taus, kappa=self.config.huber_kappa)
            crossing = _crossing_fraction(pred.detach())
        else:
            raise TypeError("agent/config mismatch")

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item()), crossing

    def _epsilon(self, step: int) -> float:
        cfg = self.config
        frac = min(1.0, step / cfg.epsilon_decay_steps)
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    def _log(self, stats: DistributionalUpdateStats) -> None:
        if self.logger is None:
            return
        self.logger.log_scalars(
            {
                "distributional/loss": stats.loss,
                "distributional/epsilon": stats.epsilon,
                "distributional/mean_reward": stats.mean_reward,
                "distributional/replay_size": float(stats.replay_size),
                "distributional/quantile_crossing_frac": stats.quantile_crossing_frac,
            },
            stats.global_step,
        )


def _crossing_fraction(quantiles: torch.Tensor) -> float:
    if quantiles.shape[-1] < 2:
        return 0.0
    crossings = quantiles[..., 1:] < quantiles[..., :-1]
    return float(crossings.float().mean().item())
