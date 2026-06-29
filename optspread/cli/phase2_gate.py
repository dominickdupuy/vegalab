"""Run the full Phase-2 Wave-0 PPO no-edge gate and write a report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from optspread.agents.ppo.ppo_agent import PPOAgent
from optspread.agents.ppo.trainer import PPOTrainer, UpdateStats
from optspread.eval.evaluator import Evaluator
from optspread.eval.metrics import MetricSuite
from optspread.training.harness import TrainHarness
from optspread.training.logging import MetricLogger
from optspread.training.phase2 import (
    phase2_factory,
    phase2_ppo_config,
    phase2_risk_reward,
    pure_pnl_reward,
)


@dataclass(frozen=True, slots=True)
class GateRecord:
    suite: str
    seed: int
    passed: bool
    flat_frequency: float
    mean_pnl: float
    pnl_ci: tuple[float, float]
    first_entropy: float
    last_entropy: float
    last_approx_kl: float
    last_explained_variance: float
    top_actions: str
    reason: str
    checkpoint_path: Path
    report_path: Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete Phase-2 no-edge gate")
    parser.add_argument("--run-root", type=Path, default=Path("runs/phase2_gate"))
    parser.add_argument("--report-path", type=Path, default=Path("phases/PHASE2_GATE_REPORT.md"))
    parser.add_argument("--seeds", default="13,32,43")
    parser.add_argument("--pure-seeds", default="")
    parser.add_argument("--risk-timesteps", type=int, default=131_072)
    parser.add_argument("--pure-timesteps", type=int, default=32_768)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--n-boot", type=int, default=2_000)
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    risk_seeds = _parse_seeds(args.seeds)
    pure_seeds = (
        _parse_seeds(args.pure_seeds)
        if args.pure_seeds
        else tuple(seed + 100 for seed in risk_seeds)
    )

    risk_records = [
        _run_gate_seed(
            suite="risk_adjusted",
            seed=seed,
            run_root=args.run_root,
            total_timesteps=args.risk_timesteps,
            eval_episodes=args.eval_episodes,
            n_boot=args.n_boot,
            with_costs=True,
            flat_threshold=0.8,
            env_seed_start=10_000 + idx * 1_000,
            eval_seed_start=70_000,
            reward_kind="phase2-risk",
            tensorboard=not args.no_tensorboard,
        )
        for idx, seed in enumerate(risk_seeds)
    ]
    pure_records = [
        _run_gate_seed(
            suite="pure_pnl_no_cost",
            seed=seed,
            run_root=args.run_root,
            total_timesteps=args.pure_timesteps,
            eval_episodes=args.eval_episodes,
            n_boot=args.n_boot,
            with_costs=False,
            flat_threshold=0.0,
            env_seed_start=20_000 + idx * 1_000,
            eval_seed_start=80_000,
            reward_kind="pure-pnl",
            tensorboard=not args.no_tensorboard,
        )
        for idx, seed in enumerate(pure_seeds)
    ]

    _write_report(args.report_path, risk_records, pure_records)
    all_passed = all(record.passed for record in (*risk_records, *pure_records))
    print(f"report: {args.report_path}")
    print(f"phase2 gate: {'PASS' if all_passed else 'FAIL'}")
    if not all_passed:
        raise SystemExit(1)


def _run_gate_seed(
    *,
    suite: str,
    seed: int,
    run_root: Path,
    total_timesteps: int,
    eval_episodes: int,
    n_boot: int,
    with_costs: bool,
    flat_threshold: float,
    env_seed_start: int,
    eval_seed_start: int,
    reward_kind: str,
    tensorboard: bool,
) -> GateRecord:
    reward = phase2_risk_reward() if reward_kind == "phase2-risk" else pure_pnl_reward()
    factory = phase2_factory(reward=reward, with_costs=with_costs)
    cfg = phase2_ppo_config(
        seed=seed,
        total_timesteps=total_timesteps,
        ent_coef=0.005 if reward_kind == "phase2-risk" else 0.02,
    )
    agent = PPOAgent(factory.obs_dim, factory.n_actions, cfg)
    trainer = PPOTrainer(agent, factory, cfg, env_base_seed=env_seed_start)
    eval_seeds = tuple(range(eval_seed_start, eval_seed_start + eval_episodes))
    train_seed_anchors = tuple(range(env_seed_start, env_seed_start + cfg.num_envs))
    evaluator = Evaluator(
        factory,
        eval_seeds,
        MetricSuite(n_boot=n_boot),
        train_seeds=train_seed_anchors,
    )
    run_dir = run_root / suite / f"seed_{seed}"
    logger = MetricLogger(run_dir / "tb", enabled=tensorboard)
    result = TrainHarness(
        agent=agent,
        trainer=trainer,
        evaluator=evaluator,
        run_dir=run_dir,
        logger=logger,
    ).run(
        checkpoint_name="ppo_agent.pt",
        gate_with_costs=with_costs,
        flat_threshold=flat_threshold,
        artifact_prefix=suite,
    )
    assert result.gate is not None
    first, last = _ppo_stats(result.history)
    return GateRecord(
        suite=suite,
        seed=seed,
        passed=result.gate.passed,
        flat_frequency=result.report.flat_frequency,
        mean_pnl=result.report.mean_pnl,
        pnl_ci=result.report.pnl_ci,
        first_entropy=first.entropy,
        last_entropy=last.entropy,
        last_approx_kl=last.approx_kl,
        last_explained_variance=last.explained_variance,
        top_actions=_top_actions(result.report.action_frequencies),
        reason=result.gate.reason,
        checkpoint_path=result.checkpoint_path,
        report_path=result.report_path,
    )


def _ppo_stats(history: Sequence[object]) -> tuple[UpdateStats, UpdateStats]:
    if not history:
        raise ValueError("PPO history is empty")
    first = history[0]
    last = history[-1]
    if not isinstance(first, UpdateStats) or not isinstance(last, UpdateStats):
        raise TypeError("phase2_gate expects PPO UpdateStats history")
    return first, last


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _top_actions(freqs: dict[int, float], *, n: int = 5) -> str:
    ordered = sorted(freqs.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(f"{action}:{freq:.3f}" for action, freq in ordered[:n])


def _write_report(
    path: Path,
    risk_records: list[GateRecord],
    pure_records: list[GateRecord],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_records = [*risk_records, *pure_records]
    all_passed = all(record.passed for record in all_records)
    text = [
        "# Phase 2 Gate Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Overall result: **{'PASS' if all_passed else 'FAIL'}**",
        "",
        "## Risk-Adjusted PPO Gate",
        "",
        "Reward: scaled MTM P&L + soft CVaR penalty. Costs: default quoted spread. "
        "Pass condition: FLAT frequency >= 0.80 and mean-PnL CI lower bound <= 0.",
        "",
        _records_table(risk_records),
        "",
        "## Pure-PnL No-Cost Ablation",
        "",
        "Reward: scaled pure MTM P&L. Costs: zero. "
        "Pass condition: no statistically reliable positive mean P&L.",
        "",
        _records_table(pure_records),
        "",
        "## Conclusion",
        "",
        (
            "Phase 2 is complete: PPO does not find a systematic Wave-0 edge, "
            "and the project may proceed to Phase 3."
            if all_passed
            else "Phase 2 is not complete: at least one seed failed the no-edge gate."
        ),
        "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def _records_table(records: list[GateRecord]) -> str:
    lines = [
        "| Seed | Gate | FLAT | Mean P&L | 95% CI | "
        "Entropy First→Last | KL | Expl.Var | Top Actions |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            "| "
            f"{record.seed} | "
            f"{'PASS' if record.passed else 'FAIL'} | "
            f"{record.flat_frequency:.3f} | "
            f"{record.mean_pnl:+.2f} | "
            f"[{record.pnl_ci[0]:+.2f}, {record.pnl_ci[1]:+.2f}] | "
            f"{record.first_entropy:.3f}→{record.last_entropy:.3f} | "
            f"{record.last_approx_kl:.4f} | "
            f"{record.last_explained_variance:.3f} | "
            f"{record.top_actions} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
