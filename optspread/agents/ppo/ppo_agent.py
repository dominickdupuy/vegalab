"""PPOAgent: the inference-side ``Agent`` wrapping the trained network.

Owns the actor-critic and (optionally) the causal observation normalizer, so a
saved checkpoint is fully self-contained: ``act`` takes a RAW observation,
normalizes it with the frozen training-time stats, and returns an action id. The
trainer drives learning through ``self.network`` directly; ``act`` is eval-only
and never mutates the normalizer (eval must use causal, frozen stats).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from optspread.agents.ppo.config import PPOConfig
from optspread.agents.ppo.network import ActorCritic
from optspread.training.normalize import CausalObsNormalizer

Observation = NDArray[np.float32]


class PPOAgent:
    """Implements the shared ``Agent`` protocol over an ``ActorCritic``."""

    def __init__(self, obs_dim: int, n_actions: int, config: PPOConfig) -> None:
        self.config = config
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = torch.device(config.device)
        self.network = ActorCritic(
            obs_dim, n_actions, config.hidden_sizes, shared_trunk=config.shared_trunk
        ).to(self.device)
        self.normalizer: CausalObsNormalizer | None = (
            CausalObsNormalizer(obs_dim) if config.norm_obs else None
        )

    # -- Agent protocol ---------------------------------------------------- #

    def act(self, obs: Observation, deterministic: bool) -> int:
        x = self._prepare(obs)
        action = self.network.act(x, deterministic)
        return int(action.item())

    def save(self, path: Path) -> None:
        payload: dict[str, Any] = {
            "network": self.network.state_dict(),
            "config": self.config.model_dump(),
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "normalizer": self.normalizer.state_dict() if self.normalizer else None,
        }
        torch.save(payload, path)

    def load(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(payload["network"])
        if self.normalizer is not None and payload.get("normalizer") is not None:
            self.normalizer.load_state_dict(payload["normalizer"])

    @classmethod
    def from_checkpoint(cls, path: Path, *, device: str | None = None) -> PPOAgent:
        """Construct and restore a PPO agent from a saved checkpoint."""
        payload = torch.load(path, map_location=device or "cpu", weights_only=False)
        config = PPOConfig(**payload["config"])
        if device is not None:
            config = config.model_copy(update={"device": device})
        agent = cls(int(payload["obs_dim"]), int(payload["n_actions"]), config)
        agent.load(path)
        return agent

    # -- helpers ----------------------------------------------------------- #

    def _prepare(self, obs: Observation) -> torch.Tensor:
        arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if self.normalizer is not None:
            arr = self.normalizer.normalize(arr)
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)
