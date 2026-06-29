"""Final-agent Wave-0 no-edge gate wrapper."""

from __future__ import annotations

from optspread.agents.base import Agent
from optspread.eval.evaluator import Evaluator
from optspread.eval.metrics import MetricSuite
from optspread.eval.no_edge_gate import NoEdgeResult, evaluate_no_edge
from optspread.training.phase2 import phase2_factory, phase2_risk_reward


def final_no_edge(agent: Agent, *, eval_episodes: int = 50) -> NoEdgeResult:
    factory = phase2_factory(reward=phase2_risk_reward(), with_costs=True)
    evaluator = Evaluator(factory, tuple(range(120_000, 120_000 + eval_episodes)), MetricSuite())
    report = evaluator.run(agent, deterministic=True)
    return evaluate_no_edge(report, with_costs=True, flat_threshold=0.8)
