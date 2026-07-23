"""IQN agent implementing the shared Phase-2 Agent protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from optspread.agents.distributional.config import IQNConfig
from optspread.agents.distributional.network_iqn import IQNNetwork
from optspread.agents.distributional.risk import RiskMeasure
from optspread.training.normalize import CausalObsNormalizer

Observation = NDArray[np.float32]


class IQNAgent:
    """Checkpointable IQN policy with U(0, alpha) CVaR action sampling."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: IQNConfig,
        *,
        risk_measure: RiskMeasure | None = None,
    ) -> None:
        self.config = config
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = torch.device(config.device)
        self.risk_measure = risk_measure or RiskMeasure.cvar(config.cvar_alpha)
        hidden = config.hidden_sizes[0] if config.hidden_sizes else 64
        self.network = IQNNetwork(obs_dim, n_actions, hidden, config.cosine_embed_dim).to(
            self.device
        )
        self.target_network = IQNNetwork(obs_dim, n_actions, hidden, config.cosine_embed_dim).to(
            self.device
        )
        self.sync_target()
        self.normalizer: CausalObsNormalizer | None = (
            CausalObsNormalizer(obs_dim) if config.norm_obs else None
        )

    def act(self, obs: Observation, deterministic: bool) -> int:
        x = self.prepare_obs(np.asarray(obs, dtype=np.float32).reshape(1, -1), update=False)
        risk = self.risk_measure if deterministic else RiskMeasure.mean()
        return int(self.greedy_actions(x, risk=risk, use_target=False).item())

    def save(self, path: Path) -> None:
        payload: dict[str, Any] = {
            "network": self.network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "config": self.config.model_dump(),
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "risk": {
                "name": self.risk_measure.name,
                "alpha": self.risk_measure.alpha,
                "mean_weight": self.risk_measure.mean_weight,
            },
            "normalizer": self.normalizer.state_dict() if self.normalizer else None,
        }
        torch.save(payload, path)

    def load(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(payload["network"])
        self.target_network.load_state_dict(payload["target_network"])
        risk = payload.get("risk")
        if isinstance(risk, dict):
            self.risk_measure = RiskMeasure(
                str(risk["name"]), float(risk["alpha"]), float(risk.get("mean_weight", 0.0))
            )
        if self.normalizer is not None and payload.get("normalizer") is not None:
            self.normalizer.load_state_dict(payload["normalizer"])

    @classmethod
    def from_checkpoint(cls, path: Path, *, device: str | None = None) -> IQNAgent:
        payload = torch.load(path, map_location=device or "cpu", weights_only=False)
        config = IQNConfig(**payload["config"])
        if device is not None:
            config = config.model_copy(update={"device": device})
        risk_raw = payload.get("risk", {"name": "cvar", "alpha": config.cvar_alpha})
        risk = RiskMeasure(
            str(risk_raw["name"]), float(risk_raw["alpha"]), float(risk_raw.get("mean_weight", 0.0))
        )
        agent = cls(int(payload["obs_dim"]), int(payload["n_actions"]), config, risk_measure=risk)
        agent.load(path)
        return agent

    def sync_target(self) -> None:
        self.target_network.load_state_dict(self.network.state_dict())

    def prepare_obs(self, obs: Observation, *, update: bool) -> torch.Tensor:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1, self.obs_dim)
        if self.normalizer is not None:
            if update:
                self.normalizer.update(arr)
            arr = self.normalizer.normalize(arr)
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    def sample_taus(self, batch_size: int, n_quantiles: int, risk: RiskMeasure) -> torch.Tensor:
        u = torch.rand((batch_size, n_quantiles), device=self.device)
        if risk.name == "cvar":
            return u * risk.alpha
        if risk.name == "mean_cvar":
            # Mixture sampling realizes w*E[Z] + (1-w)*CVaR_alpha in expectation:
            # each tau is full-range with prob. mean_weight, tail-restricted otherwise.
            full_range = (
                torch.rand((batch_size, n_quantiles), device=self.device) < risk.mean_weight
            )
            return torch.where(full_range, u, u * risk.alpha)
        return u

    def quantiles(self, obs: torch.Tensor, taus: torch.Tensor, *, use_target: bool) -> torch.Tensor:
        net = self.target_network if use_target else self.network
        return net.action_quantiles(obs, taus)

    def greedy_actions(
        self,
        obs: torch.Tensor,
        *,
        risk: RiskMeasure,
        use_target: bool,
    ) -> torch.Tensor:
        taus = self.sample_taus(obs.shape[0], self.config.n_action_quantiles, risk)
        values = self.quantiles(obs, taus, use_target=use_target).mean(dim=-1)
        return torch.argmax(values, dim=1)
