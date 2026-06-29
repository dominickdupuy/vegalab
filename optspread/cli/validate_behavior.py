"""Run pre-registered curriculum behavioral validations on trained checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from optspread.agents.base import Agent
from optspread.agents.distributional.iqn_agent import IQNAgent
from optspread.agents.distributional.qrdqn_agent import QRDQNAgent
from optspread.agents.ppo.ppo_agent import PPOAgent
from optspread.config import EnvConfig, GBMConfig
from optspread.curriculum.waves import wave1_spec
from optspread.envs.builder import EnvBundle
from optspread.eval.rollout import collect_rollout_trace, wave1_credit_vrp_statistic
from optspread.training.env_factory import EnvFactory
from optspread.training.phase2 import phase2_risk_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate trained-agent curriculum behavior")
    parser.add_argument("--wave", type=int, choices=(1,), default=1)
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
    args = parser.parse_args()

    agent = _load_agent(args.agent_kind, args.checkpoint, args.device)
    factory = _factory_for_wave(args.wave, args.episode_length)
    seeds = tuple(range(args.eval_seed_start, args.eval_seed_start + args.eval_episodes))
    trace = collect_rollout_trace(agent, factory, seeds, deterministic=True)

    if args.wave == 1:
        statistic = wave1_credit_vrp_statistic(trace)
        statistic_name = "corr(credit_indicator, vrp)"
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
    cfg = GBMConfig(n_days=episode_length)
    if wave == 1:
        spec = wave1_spec(cfg)
    else:
        raise ValueError(f"unsupported wave: {wave}")
    return EnvFactory(
        EnvBundle(
            env=EnvConfig(episode_length=episode_length),
            gbm=cfg,
            reward=phase2_risk_reward(),
            generator_factory=spec.make_generator,
        )
    )


if __name__ == "__main__":
    main()
