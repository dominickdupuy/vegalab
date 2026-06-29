"""Train the Phase-2 PPO baseline on Wave 0."""

from __future__ import annotations

import argparse
from pathlib import Path

from optspread.agents.ppo.ppo_agent import PPOAgent
from optspread.agents.ppo.trainer import PPOTrainer
from optspread.eval.evaluator import Evaluator
from optspread.eval.metrics import MetricSuite
from optspread.training.curriculum_factory import wave_factory
from optspread.training.harness import TrainHarness
from optspread.training.logging import MetricLogger
from optspread.training.phase2 import (
    phase2_ppo_config,
    phase2_risk_reward,
    pure_pnl_reward,
)
from optspread.training.seeding import run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on Wave-0 fair-IV GBM")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default="phase2_ppo_wave0")
    parser.add_argument("--wave", type=int, default=0, help="Curriculum wave (0 fair-IV, 1 VRP).")
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help="Checkpoint to warm-start from (previous wave).",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--env-seed-start", type=int, default=10_000)
    parser.add_argument("--eval-seed-start", type=int, default=70_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--total-timesteps", type=int, default=131_072)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--target-kl", type=float, default=0.05)
    parser.add_argument(
        "--reward-preset",
        choices=("phase2-risk", "pure-pnl"),
        default="phase2-risk",
    )
    parser.add_argument("--no-costs", action="store_true")
    parser.add_argument("--flat-threshold", type=float, default=0.8)
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    reward = phase2_risk_reward() if args.reward_preset == "phase2-risk" else pure_pnl_reward()
    factory = wave_factory(args.wave, reward=reward, with_costs=not args.no_costs)
    cfg = phase2_ppo_config(
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        target_kl=args.target_kl,
    )
    agent = PPOAgent(factory.obs_dim, factory.n_actions, cfg)
    if args.warm_start is not None:
        agent.load(args.warm_start)
    trainer = PPOTrainer(agent, factory, cfg, env_base_seed=args.env_seed_start)
    eval_seeds = tuple(range(args.eval_seed_start, args.eval_seed_start + args.eval_episodes))
    train_seed_anchors = tuple(range(args.env_seed_start, args.env_seed_start + cfg.num_envs))
    evaluator = Evaluator(
        factory,
        eval_seeds,
        MetricSuite(n_boot=2_000),
        train_seeds=train_seed_anchors,
    )
    out_dir = run_dir(args.run_root, args.run_name, args.seed)
    logger = MetricLogger(out_dir / "tb", enabled=not args.no_tensorboard)
    # The no-edge (FLAT-dominant) gate only applies to Wave 0; later waves trade.
    skip_gate = args.no_gate or args.wave != 0
    gate_with_costs = None if skip_gate else not args.no_costs
    flat_threshold = None if skip_gate else args.flat_threshold
    result = TrainHarness(
        agent=agent,
        trainer=trainer,
        evaluator=evaluator,
        run_dir=out_dir,
        logger=logger,
    ).run(gate_with_costs=gate_with_costs, flat_threshold=flat_threshold)

    print(f"checkpoint: {result.checkpoint_path}")
    print(f"eval report: {result.report_path}")
    print(f"returns: {result.returns_path}")
    print(
        "eval: "
        f"flat={result.report.flat_frequency:.3f}, "
        f"mean_pnl={result.report.mean_pnl:+.2f}, "
        f"ci=({result.report.pnl_ci[0]:+.2f}, {result.report.pnl_ci[1]:+.2f})"
    )
    if result.gate is not None:
        print(f"gate: {'PASS' if result.gate.passed else 'FAIL'} — {result.gate.reason}")


if __name__ == "__main__":
    main()
