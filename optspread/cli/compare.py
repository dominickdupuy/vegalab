"""Compare saved agents through the shared Evaluator/MetricSuite."""

from __future__ import annotations

import argparse
from pathlib import Path

from optspread.agents.base import Agent, FlatAgent
from optspread.agents.distributional.iqn_agent import IQNAgent
from optspread.agents.distributional.qrdqn_agent import QRDQNAgent
from optspread.agents.ppo.ppo_agent import PPOAgent
from optspread.eval.evaluator import Evaluator
from optspread.eval.metrics import EvalReport, MetricSuite
from optspread.training.phase2 import phase2_factory, phase2_risk_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare checkpoints on shared Wave-0 eval")
    parser.add_argument("--ppo", type=Path)
    parser.add_argument("--qrdqn", type=Path)
    parser.add_argument("--iqn", type=Path)
    parser.add_argument("--include-flat", action="store_true")
    parser.add_argument("--eval-seed-start", type=int, default=70_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    args = parser.parse_args()

    agents: list[tuple[str, Agent]] = []
    if args.include_flat:
        agents.append(("flat", FlatAgent()))
    if args.ppo is not None:
        agents.append(("ppo", PPOAgent.from_checkpoint(args.ppo)))
    if args.qrdqn is not None:
        agents.append(("qrdqn", QRDQNAgent.from_checkpoint(args.qrdqn)))
    if args.iqn is not None:
        agents.append(("iqn", IQNAgent.from_checkpoint(args.iqn)))
    if not agents:
        raise SystemExit("provide at least one checkpoint or --include-flat")

    factory = phase2_factory(reward=phase2_risk_reward(), with_costs=True)
    eval_seeds = tuple(range(args.eval_seed_start, args.eval_seed_start + args.eval_episodes))
    evaluator = Evaluator(factory, eval_seeds, MetricSuite(n_boot=1_000))
    reports = [(name, evaluator.run(agent, deterministic=True)) for name, agent in agents]
    print(_table(reports))


def _table(reports: list[tuple[str, EvalReport]]) -> str:
    lines = [
        "| Agent | Mean P&L | Sharpe | Sortino | CVaR95 | MaxDD | Turnover | FLAT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, report in reports:
        lines.append(
            "| "
            f"{name} | "
            f"{report.mean_pnl:+.2f} | "
            f"{report.sharpe:+.3f} | "
            f"{report.sortino:+.3f} | "
            f"{report.cvar_95:+.2f} | "
            f"{report.max_drawdown:.3f} | "
            f"{report.turnover:.2f} | "
            f"{report.flat_frequency:.3f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
