"""Evaluate a saved PPO checkpoint through the shared Phase-2 evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

from optspread.agents.ppo.ppo_agent import PPOAgent
from optspread.eval.evaluator import Evaluator
from optspread.eval.metrics import MetricSuite
from optspread.eval.no_edge_gate import evaluate_no_edge
from optspread.training.harness import save_eval_artifacts
from optspread.training.phase2 import phase2_factory, phase2_risk_reward, pure_pnl_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PPO checkpoint on Wave 0")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/eval"))
    parser.add_argument("--eval-seed-start", type=int, default=70_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument(
        "--reward-preset",
        choices=("phase2-risk", "pure-pnl"),
        default="phase2-risk",
    )
    parser.add_argument("--no-costs", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--flat-threshold", type=float, default=0.8)
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()

    reward = phase2_risk_reward() if args.reward_preset == "phase2-risk" else pure_pnl_reward()
    factory = phase2_factory(reward=reward, with_costs=not args.no_costs)
    eval_seeds = tuple(range(args.eval_seed_start, args.eval_seed_start + args.eval_episodes))
    evaluator = Evaluator(factory, eval_seeds, MetricSuite(n_boot=2_000))
    agent = PPOAgent.from_checkpoint(args.checkpoint)
    report = evaluator.run(agent, deterministic=not args.stochastic)
    gate = None
    if not args.no_gate:
        gate = evaluate_no_edge(
            report,
            with_costs=not args.no_costs,
            flat_threshold=args.flat_threshold,
        )
    report_path, returns_path = save_eval_artifacts(report, args.out_dir, prefix="eval", gate=gate)

    print(f"eval report: {report_path}")
    print(f"returns: {returns_path}")
    print(
        "eval: "
        f"flat={report.flat_frequency:.3f}, "
        f"mean_pnl={report.mean_pnl:+.2f}, "
        f"ci=({report.pnl_ci[0]:+.2f}, {report.pnl_ci[1]:+.2f})"
    )
    if gate is not None:
        print(f"gate: {'PASS' if gate.passed else 'FAIL'} — {gate.reason}")


if __name__ == "__main__":
    main()
