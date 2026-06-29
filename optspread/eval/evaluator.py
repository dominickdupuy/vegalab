"""Evaluator: run a policy over fixed eval seeds and produce an ``EvalReport``.

One eval seed == one full episode on a freshly built env reset to that seed, so
evaluation is deterministic and reproducible. Eval seeds MUST be disjoint from the
seeds used to generate training paths (otherwise the no-edge result is
contaminated by memorized paths); the constructor enforces this when given the
training seeds.

The agent receives the RAW observation and is responsible for any (causal)
normalization internally, so the same ``Evaluator`` scores a PPO agent and a
scripted baseline without special-casing either.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from optspread.agents.base import Agent
from optspread.eval.metrics import EvalReport, MetricSuite
from optspread.training.env_factory import EnvFactory


def assert_disjoint(train_seeds: Sequence[int], eval_seeds: Sequence[int]) -> None:
    """Raise if any eval seed was also used for training (look-ahead via memorization)."""
    overlap = set(train_seeds) & set(eval_seeds)
    if overlap:
        raise ValueError(f"eval seeds overlap training seeds: {sorted(overlap)}")


class Evaluator:
    """Runs an ``Agent`` over a fixed seed set and scores it with a ``MetricSuite``."""

    def __init__(
        self,
        env_factory: EnvFactory,
        eval_seeds: Sequence[int],
        metrics: MetricSuite,
        *,
        train_seeds: Sequence[int] | None = None,
    ) -> None:
        if train_seeds is not None:
            assert_disjoint(train_seeds, eval_seeds)
        self.env_factory = env_factory
        self.eval_seeds = list(eval_seeds)
        self.metrics = metrics

    def run(self, agent: Agent, deterministic: bool) -> EvalReport:
        per_step: list[float] = []
        episode_returns: list[float] = []
        action_counts: dict[int, int] = defaultdict(int)
        equity_curves: list[NDArray[np.float64]] = []
        n_trades = 0

        env = self.env_factory.make()
        for seed in self.eval_seeds:
            obs, info = env.reset(seed=seed)
            curve = [float(info["equity"])]
            ep_return = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.act(obs, deterministic)
                obs, _reward, terminated, truncated, info = env.step(action)
                pnl = float(info["pnl"])
                per_step.append(pnl)
                ep_return += pnl
                action_counts[action] += 1
                if info["did_trade"]:
                    n_trades += 1
                curve.append(float(info["equity"]))
            episode_returns.append(ep_return)
            equity_curves.append(np.asarray(curve, dtype=np.float64))

        return self.metrics.compute(
            per_step_returns=np.asarray(per_step, dtype=np.float64),
            episode_returns=np.asarray(episode_returns, dtype=np.float64),
            action_counts=dict(action_counts),
            equity_curves=equity_curves,
            n_trades=n_trades,
        )
