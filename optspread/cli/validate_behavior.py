"""Run pre-registered curriculum behavioral validations on trained checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from optspread.agents.base import Agent
from optspread.agents.distributional.iqn_agent import IQNAgent
from optspread.agents.distributional.qrdqn_agent import QRDQNAgent
from optspread.agents.distributional.risk import RiskMeasure
from optspread.agents.ppo.ppo_agent import PPOAgent
from optspread.eval.rollout import (
    collect_rollout_trace,
    wave1_credit_vrp_statistic,
    wave2_ivrank_statistic,
)
from optspread.training.curriculum_factory import wave_factory
from optspread.training.env_factory import EnvFactory


def deploy_risk_override(args: argparse.Namespace) -> RiskMeasure | None:
    """Deployment-scoring override (agent-side action selection; harness untouched)."""
    if args.deploy_risk == "checkpoint":
        return None
    if args.deploy_risk == "mean":
        return RiskMeasure.mean()
    if args.deploy_risk == "cvar":
        return RiskMeasure.cvar(args.deploy_alpha)
    return RiskMeasure.mean_cvar(args.deploy_alpha, args.mean_weight)


def add_deploy_risk_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--deploy-risk",
        choices=("checkpoint", "mean", "cvar", "mean_cvar"),
        default="checkpoint",
        help="Override the checkpoint's action-selection risk (distributional agents only).",
    )
    parser.add_argument("--deploy-alpha", type=float, default=0.2)
    parser.add_argument("--mean-weight", type=float, default=0.9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate trained-agent curriculum behavior")
    parser.add_argument("--wave", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--agent-kind",
        choices=("ppo", "qrdqn", "iqn"),
        required=True,
        help="Checkpoint format to load.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-seed-start", type=int, default=90_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--episode-length", type=int, default=63)
    parser.add_argument("--min-corr", type=float, default=0.0)
    parser.add_argument("--out", type=Path)
    add_deploy_risk_args(parser)
    args = parser.parse_args()

    agent = _load_agent(args.agent_kind, args.checkpoint, args.device)
    override = deploy_risk_override(args)
    if override is not None:
        if not isinstance(agent, IQNAgent | QRDQNAgent):
            raise SystemExit("--deploy-risk override requires a distributional agent")
        agent.risk_measure = override
    factory = _factory_for_wave(args.wave, args.episode_length)
    seeds = tuple(range(args.eval_seed_start, args.eval_seed_start + args.eval_episodes))
    trace = collect_rollout_trace(agent, factory, seeds, deterministic=True)

    if args.wave == 1:
        statistic = wave1_credit_vrp_statistic(trace)
        statistic_name = "corr(credit_indicator, vrp)"
    else:
        statistic = wave2_ivrank_statistic(trace)
        statistic_name = "corr(credit_indicator, iv_rank)"
    passed = math.isfinite(statistic) and statistic > args.min_corr
    payload = {
        "wave": args.wave,
        "agent_kind": args.agent_kind,
        "checkpoint": str(args.checkpoint),
        "statistic": statistic_name,
        "value": statistic if math.isfinite(statistic) else None,
        "threshold": args.min_corr,
        "passed": passed,
        "n_decisions": int(trace.actions.size),
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    verdict = "PASS" if passed else "FAIL"
    value = "nan" if payload["value"] is None else f"{statistic:+.6f}"
    print(f"BV_{args.wave}: {verdict} — {statistic_name}={value} > {args.min_corr:+.6f}")


def _load_agent(kind: str, checkpoint: Path, device: str) -> Agent:
    if kind == "ppo":
        return PPOAgent.from_checkpoint(checkpoint, device=device)
    if kind == "qrdqn":
        return QRDQNAgent.from_checkpoint(checkpoint, device=device)
    if kind == "iqn":
        return IQNAgent.from_checkpoint(checkpoint, device=device)
    raise ValueError(f"unsupported agent kind: {kind}")


def _factory_for_wave(wave: int, episode_length: int) -> EnvFactory:
    return wave_factory(wave, episode_length=episode_length)


if __name__ == "__main__":
    main()
