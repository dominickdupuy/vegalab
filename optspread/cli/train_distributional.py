"""Train QR-DQN or IQN on Wave 0 through the shared Phase-2 harness."""

from __future__ import annotations

import argparse
from pathlib import Path

from optspread.agents.distributional.config import IQNConfig, QRDQNConfig
from optspread.agents.distributional.iqn_agent import IQNAgent
from optspread.agents.distributional.qrdqn_agent import QRDQNAgent
from optspread.agents.distributional.risk import RiskMeasure
from optspread.agents.distributional.trainer import DistributionalTrainer
from optspread.eval.evaluator import Evaluator
from optspread.eval.metrics import MetricSuite
from optspread.training.curriculum_factory import wave_factory
from optspread.training.harness import TrainHarness
from optspread.training.logging import MetricLogger
from optspread.training.seeding import run_dir


def _parse_rehearsal_waves(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    parts = raw.split(",")
    if any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError("expected comma-separated wave ids")
    try:
        return tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integer wave ids") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Phase-3 distributional agent")
    parser.add_argument("--algo", choices=("qrdqn", "iqn"), default="qrdqn")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default="phase3_distributional_wave0")
    parser.add_argument(
        "--wave",
        type=int,
        default=0,
        help="Curriculum wave (0 fair-IV, 1 VRP, 2 Heston SV).",
    )
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help="Checkpoint to warm-start from (previous wave).",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env-seed", type=int, default=30_000)
    parser.add_argument("--eval-seed-start", type=int, default=90_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-quantiles", type=int, default=None)
    parser.add_argument("--cvar-alpha", type=float, default=0.05)
    parser.add_argument("--risk", choices=("mean", "cvar"), default="cvar")
    parser.add_argument("--rehearsal-fraction", type=float, default=0.0)
    parser.add_argument("--rehearsal-waves", type=_parse_rehearsal_waves, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    # Reward defaults per wave (Wave 0: no-edge gate reward; Wave 1+: MTM-only,
    # tail-aversion is agent-side via CVaR action selection).
    # Training mixes earlier-wave rehearsal; evaluation stays on the pure target wave.
    train_factory = wave_factory(
        args.wave,
        with_costs=True,
        rehearsal_fraction=args.rehearsal_fraction,
        rehearsal_waves=args.rehearsal_waves,
    )
    eval_factory = wave_factory(args.wave, with_costs=True)
    risk = RiskMeasure.mean() if args.risk == "mean" else RiskMeasure.cvar(args.cvar_alpha)
    cfg: QRDQNConfig | IQNConfig
    agent: QRDQNAgent | IQNAgent
    if args.algo == "qrdqn":
        cfg = QRDQNConfig(
            seed=args.seed,
            total_timesteps=args.total_timesteps,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            n_quantiles=args.n_quantiles or 200,
            cvar_alpha=args.cvar_alpha,
        )
        agent = QRDQNAgent(train_factory.obs_dim, train_factory.n_actions, cfg, risk_measure=risk)
    else:
        cfg = IQNConfig(
            seed=args.seed,
            total_timesteps=args.total_timesteps,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            n_quantiles=args.n_quantiles or 32,
            cvar_alpha=args.cvar_alpha,
        )
        agent = IQNAgent(train_factory.obs_dim, train_factory.n_actions, cfg, risk_measure=risk)

    if args.warm_start is not None:
        agent.load(args.warm_start)
        # ``load`` restores the checkpoint's risk measure; re-apply the risk
        # selected for THIS wave so warm-starting transfers weights only.
        agent.risk_measure = risk

    trainer = DistributionalTrainer(agent, train_factory, cfg, env_seed=args.env_seed)
    eval_seeds = tuple(range(args.eval_seed_start, args.eval_seed_start + args.eval_episodes))
    evaluator = Evaluator(
        eval_factory, eval_seeds, MetricSuite(n_boot=1_000), train_seeds=(args.env_seed,)
    )
    out_dir = run_dir(args.run_root, f"{args.run_name}_{args.algo}_{args.risk}", args.seed)
    logger = MetricLogger(out_dir / "tb", enabled=not args.no_tensorboard)
    # The no-edge (FLAT-dominant) gate only applies to Wave 0; later waves are
    # expected to trade, so gating on FLAT-dominance there would be wrong.
    gate_with_costs = True if args.wave == 0 else None
    flat_threshold = 0.8 if args.wave == 0 else None
    result = TrainHarness(
        agent=agent,
        trainer=trainer,
        evaluator=evaluator,
        run_dir=out_dir,
        logger=logger,
    ).run(
        gate_with_costs=gate_with_costs,
        flat_threshold=flat_threshold,
        artifact_prefix=args.algo,
    )
    print(f"checkpoint: {result.checkpoint_path}")
    print(f"eval report: {result.report_path}")
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
