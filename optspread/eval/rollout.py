"""Policy rollout traces for curriculum behavioral-validation gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optspread.agents.base import Agent
from optspread.eval.behavioral import credit_structure_indicator
from optspread.market.snapshot import REGIME_FEATURE_KEYS
from optspread.training.env_factory import EnvFactory


@dataclass(frozen=True, slots=True)
class RolloutTrace:
    """Decision-time actions and market-observable features from policy rollout."""

    actions: NDArray[np.int64]
    features: NDArray[np.float64]
    feature_names: tuple[str, ...] = REGIME_FEATURE_KEYS

    def feature(self, name: str) -> NDArray[np.float64]:
        """Return one feature column by its canonical regime-feature name."""
        try:
            idx = self.feature_names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return self.features[:, idx]


def collect_rollout_trace(
    agent: Agent,
    env_factory: EnvFactory,
    eval_seeds: list[int] | tuple[int, ...],
    *,
    deterministic: bool = True,
) -> RolloutTrace:
    """Collect decision-time feature/action pairs using the shared env factory.

    Features are copied from the observation before the action is applied, so the
    behavioral statistic tests exactly what the policy could observe at decision
    time. Generator internals, latent parameters, and regime labels are never
    read.
    """
    n_features = len(REGIME_FEATURE_KEYS)
    actions: list[int] = []
    features: list[NDArray[np.float64]] = []
    env = env_factory.make()
    try:
        for seed in eval_seeds:
            obs, _info = env.reset(seed=seed)
            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.act(obs, deterministic)
                actions.append(action)
                features.append(np.asarray(obs[:n_features], dtype=np.float64))
                obs, _reward, terminated, truncated, _info = env.step(action)
    finally:
        env.close()
    return RolloutTrace(
        actions=np.asarray(actions, dtype=np.int64),
        features=np.asarray(features, dtype=np.float64).reshape(-1, n_features),
    )


def wave1_credit_vrp_statistic(trace: RolloutTrace) -> float:
    """Correlation used by BV_1: credit-structure choice vs observed VRP."""
    credit = credit_structure_indicator(trace.actions)
    vrp = trace.feature("vrp")
    if credit.size < 2 or float(np.std(credit)) == 0.0 or float(np.std(vrp)) == 0.0:
        return float("nan")
    return float(np.corrcoef(credit, vrp)[0, 1])


def wave2_ivrank_statistic(trace: RolloutTrace) -> float:
    """Correlation used by BV_2: credit-structure choice vs observed IV rank."""
    credit = credit_structure_indicator(trace.actions)
    iv_rank = trace.feature("iv_rank")
    if credit.size < 2 or float(np.std(credit)) == 0.0 or float(np.std(iv_rank)) == 0.0:
        return float("nan")
    return float(np.corrcoef(credit, iv_rank)[0, 1])
