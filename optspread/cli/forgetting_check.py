"""FF_n: catastrophic-forgetting check for curriculum checkpoints.

After training on a later wave, re-evaluate the agent on Wave 0 (fair-IV GBM) and
require the no-edge property to still hold: the agent must stay FLAT-dominant and
must not manufacture positive PnL on an edgeless market. A checkpoint that starts
"making money" on Wave 0 has either forgotten the Wave-0 lesson or is exploiting a
leak — both are failures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from optspread.agents.base import Agent
from optspread.agents.distributional.iqn_agent import IQNAgent
from optspread.agents.distributional.qrdqn_agent import QRDQNAgent
from optspread.agents.ppo.ppo_agent import PPOAgent
from optspread.eval.evaluator import Evaluator
from optspread.eval.metrics import MetricSuite
from optspread.eval.no_edge_gate import evaluate_no_edge
from optspread.training.curriculum_factory import wave_factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave-0 forgetting check for a checkpoint")
    parser.add_argument("--wave", type=int, default=1, help="Wave just trained (for the label).")
    parser.add_argument("--agent-kind", choices=("ppo", "qrdqn", "iqn"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-seed-start", type=int, default=70_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--episode-length", type=int, default=63)
    parser.add_argument("--flat-threshold", type=float, default=0.8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    agent = _load_agent(args.agent_kind, args.checkpoint, args.device)
    factory = wave_factory(0, episode_length=args.episode_length)
    seeds = tuple(range(args.eval_seed_start, args.eval_seed_start + args.eval_episodes))
    report = Evaluator(factory, seeds, MetricSuite(n_boot=2_000)).run(agent, deterministic=True)
    gate = evaluate_no_edge(report, with_costs=True, flat_threshold=args.flat_threshold)

    payload = {
        "wave_trained": args.wave,
        "agent_kind": args.agent_kind,
        "checkpoint": str(args.checkpoint),
        "flat_frequency": report.flat_frequency,
        "mean_pnl": report.mean_pnl,
        "pnl_ci": [report.pnl_ci[0], report.pnl_ci[1]],
        "passed": gate.passed,
        "reason": gate.reason,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    verdict = "PASS" if gate.passed else "FAIL"
    print(
        f"FF_{args.wave}: {verdict} — Wave-0 flat={report.flat_frequency:.3f} "
        f"mean_pnl={report.mean_pnl:+.2f} ci=({report.pnl_ci[0]:+.2f},{report.pnl_ci[1]:+.2f}) "
        f"— {gate.reason}"
    )


def _load_agent(kind: str, checkpoint: Path, device: str) -> Agent:
    if kind == "ppo":
        return PPOAgent.from_checkpoint(checkpoint, device=device)
    if kind == "qrdqn":
        return QRDQNAgent.from_checkpoint(checkpoint, device=device)
    if kind == "iqn":
        return IQNAgent.from_checkpoint(checkpoint, device=device)
    raise ValueError(f"unsupported agent kind: {kind}")


if __name__ == "__main__":
    main()
