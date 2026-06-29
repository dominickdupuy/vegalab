"""Run the Phase-4 Wave-1 BV_1 + FF_1 gate across seed ensembles."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

AgentName = Literal["iqn", "ppo"]


@dataclass(frozen=True, slots=True)
class CliArgs:
    agents: tuple[AgentName, ...]
    seeds_iqn: tuple[int, ...]
    seeds_ppo: tuple[int, ...]
    total_timesteps: int
    out_dir: Path
    skip_train: bool
    min_corr: float
    cvar_alpha: float


@dataclass(frozen=True, slots=True)
class SeedResult:
    agent: AgentName
    seed: int
    warm_start: Path
    checkpoint: Path
    bv_json: Path
    ff_json: Path
    bv_corr: float | None
    bv_passed: bool
    ff_passed: bool

    def to_json(self) -> dict[str, object]:
        return {
            "agent": self.agent,
            "seed": self.seed,
            "warm_start": str(self.warm_start),
            "checkpoint": str(self.checkpoint),
            "bv_json": str(self.bv_json),
            "ff_json": str(self.ff_json),
            "bv_corr": self.bv_corr,
            "bv_passed": self.bv_passed,
            "ff_passed": self.ff_passed,
            "passed": self.bv_passed and self.ff_passed,
        }


@dataclass(frozen=True, slots=True)
class AgentAggregate:
    agent: AgentName
    n_seeds: int
    n_corr: int
    corr_mean: float | None
    corr_std: float | None
    bv_pass_count: int
    ff_pass_count: int
    passed: bool

    def to_json(self) -> dict[str, object]:
        return {
            "agent": self.agent,
            "n_seeds": self.n_seeds,
            "n_corr": self.n_corr,
            "corr_mean": self.corr_mean,
            "corr_std": self.corr_std,
            "bv_pass_count": self.bv_pass_count,
            "ff_pass_count": self.ff_pass_count,
            "passed": self.passed,
        }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seed_results: list[SeedResult] = []
    for agent in args.agents:
        for seed in _seeds_for_agent(args, agent):
            seed_results.append(_run_seed(agent, seed, args))

    aggregates = [_aggregate_agent(agent, seed_results) for agent in args.agents]
    results_path = args.out_dir / "PHASE4_BV1_RESULTS.json"
    _write_results(results_path, args, seed_results, aggregates)
    _print_summary(seed_results, aggregates, results_path)

    if not all(aggregate.passed for aggregate in aggregates):
        raise SystemExit(1)


def _parse_args(argv: Sequence[str] | None) -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Run the Phase-4 Wave-1 BV_1 + FF_1 gate across seeds"
    )
    parser.add_argument("--agents", default="iqn,ppo")
    parser.add_argument("--seeds-iqn", default="601,602,603")
    parser.add_argument("--seeds-ppo", default="13,32,43")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase4_bv1"))
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse existing Wave-1 checkpoints under --out-dir instead of retraining.",
    )
    parser.add_argument("--min-corr", type=float, default=0.10)
    parser.add_argument("--cvar-alpha", type=float, default=0.1)
    namespace = parser.parse_args(argv)

    total_timesteps = int(namespace.total_timesteps)
    if total_timesteps <= 0:
        raise SystemExit("--total-timesteps must be positive")

    return CliArgs(
        agents=_parse_agents(str(namespace.agents)),
        seeds_iqn=_parse_seeds(str(namespace.seeds_iqn), "--seeds-iqn"),
        seeds_ppo=_parse_seeds(str(namespace.seeds_ppo), "--seeds-ppo"),
        total_timesteps=total_timesteps,
        out_dir=Path(namespace.out_dir),
        skip_train=bool(namespace.skip_train),
        min_corr=float(namespace.min_corr),
        cvar_alpha=float(namespace.cvar_alpha),
    )


def _parse_agents(raw: str) -> tuple[AgentName, ...]:
    agents: list[AgentName] = []
    seen: set[AgentName] = set()
    for part in raw.split(","):
        value = part.strip().lower()
        if not value:
            continue
        if value == "iqn":
            agent: AgentName = "iqn"
        elif value == "ppo":
            agent = "ppo"
        else:
            raise SystemExit(f"--agents only supports iqn and ppo; got {value!r}")
        if agent not in seen:
            agents.append(agent)
            seen.add(agent)
    if not agents:
        raise SystemExit("--agents must include at least one agent")
    return tuple(agents)


def _parse_seeds(raw: str, flag: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        try:
            seeds.append(int(value))
        except ValueError as exc:
            raise SystemExit(f"{flag} contains a non-integer seed: {value!r}") from exc
    if not seeds:
        raise SystemExit(f"{flag} must include at least one seed")
    return tuple(seeds)


def _seeds_for_agent(args: CliArgs, agent: AgentName) -> tuple[int, ...]:
    if agent == "iqn":
        return args.seeds_iqn
    return args.seeds_ppo


def _run_seed(agent: AgentName, seed: int, args: CliArgs) -> SeedResult:
    warm_start = _wave0_checkpoint(agent, seed)
    checkpoint = _wave1_checkpoint(agent, seed, args.out_dir)

    if not warm_start.exists():
        raise FileNotFoundError(f"missing Wave-0 warm-start checkpoint: {warm_start}")

    if args.skip_train:
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing Wave-1 checkpoint for --skip-train: {checkpoint}")
    else:
        _run_cli(_train_command(agent, seed, warm_start, args))
        if not checkpoint.exists():
            raise FileNotFoundError(f"trainer completed but did not write checkpoint: {checkpoint}")

    gate_dir = args.out_dir / "gate_json" / agent / f"seed_{seed}"
    bv_json = gate_dir / "BV_1.json"
    ff_json = gate_dir / "FF_1.json"
    _run_cli(_bv_command(agent, checkpoint, bv_json, args.min_corr))
    _run_cli(_ff_command(agent, checkpoint, ff_json))

    bv_payload = _load_json_object(bv_json)
    ff_payload = _load_json_object(ff_json)
    return SeedResult(
        agent=agent,
        seed=seed,
        warm_start=warm_start,
        checkpoint=checkpoint,
        bv_json=bv_json,
        ff_json=ff_json,
        bv_corr=_float_or_none_field(bv_payload, "value"),
        bv_passed=_bool_field(bv_payload, "passed"),
        ff_passed=_bool_field(ff_payload, "passed"),
    )


def _wave0_checkpoint(agent: AgentName, seed: int) -> Path:
    if agent == "iqn":
        return Path("runs") / "phase3_distributional_wave0_iqn_cvar" / f"seed_{seed}" / "agent.pt"
    return Path("runs") / "phase2_gate" / "risk_adjusted" / f"seed_{seed}" / "ppo_agent.pt"


def _wave1_checkpoint(agent: AgentName, seed: int, out_dir: Path) -> Path:
    if agent == "iqn":
        return out_dir / "wave1_iqn_cvar" / f"seed_{seed}" / "agent.pt"
    return out_dir / "wave1_ppo" / f"seed_{seed}" / "agent.pt"


def _train_command(agent: AgentName, seed: int, warm_start: Path, args: CliArgs) -> list[str]:
    if agent == "iqn":
        return [
            sys.executable,
            "-m",
            "optspread.cli.train_distributional",
            "--algo",
            "iqn",
            "--run-root",
            str(args.out_dir),
            "--run-name",
            "wave1",
            "--wave",
            "1",
            "--warm-start",
            str(warm_start),
            "--risk",
            "cvar",
            "--cvar-alpha",
            str(args.cvar_alpha),
            "--seed",
            str(seed),
            "--env-seed",
            str(40_000 + seed),
            "--total-timesteps",
            str(args.total_timesteps),
            "--no-tensorboard",
        ]
    return [
        sys.executable,
        "-m",
        "optspread.cli.train",
        "--run-root",
        str(args.out_dir),
        "--run-name",
        "wave1_ppo",
        "--wave",
        "1",
        "--warm-start",
        str(warm_start),
        "--seed",
        str(seed),
        "--env-seed-start",
        str(50_000 + seed),
        "--total-timesteps",
        str(args.total_timesteps),
        "--no-gate",
        "--no-tensorboard",
    ]


def _bv_command(agent: AgentName, checkpoint: Path, out_path: Path, min_corr: float) -> list[str]:
    return [
        sys.executable,
        "-m",
        "optspread.cli.validate_behavior",
        "--wave",
        "1",
        "--agent-kind",
        agent,
        "--checkpoint",
        str(checkpoint),
        "--min-corr",
        str(min_corr),
        "--out",
        str(out_path),
    ]


def _ff_command(agent: AgentName, checkpoint: Path, out_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "optspread.cli.forgetting_check",
        "--wave",
        "1",
        "--agent-kind",
        agent,
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(out_path),
    ]


def _run_cli(command: Sequence[str]) -> None:
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(_format_subprocess_failure(command, completed))


def _format_subprocess_failure(
    command: Sequence[str], completed: subprocess.CompletedProcess[str]
) -> str:
    lines = [
        f"command failed with exit code {completed.returncode}:",
        " ".join(command),
    ]
    if completed.stdout:
        lines.extend(("", "stdout:", completed.stdout.rstrip()))
    if completed.stderr:
        lines.extend(("", "stderr:", completed.stderr.rstrip()))
    return "\n".join(lines)


def _load_json_object(path: Path) -> dict[str, object]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast(dict[str, object], parsed)


def _bool_field(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"JSON field {key!r} must be a bool")
    return value


def _float_or_none_field(payload: Mapping[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"JSON field {key!r} must be numeric or null")
    number = float(value)
    return number if math.isfinite(number) else None


def _aggregate_agent(agent: AgentName, seed_results: Sequence[SeedResult]) -> AgentAggregate:
    agent_results = [result for result in seed_results if result.agent == agent]
    corr_values = [
        result.bv_corr
        for result in agent_results
        if result.bv_corr is not None and math.isfinite(result.bv_corr)
    ]
    corr_mean, corr_std = _mean_and_pstdev(corr_values)
    return AgentAggregate(
        agent=agent,
        n_seeds=len(agent_results),
        n_corr=len(corr_values),
        corr_mean=corr_mean,
        corr_std=corr_std,
        bv_pass_count=sum(1 for result in agent_results if result.bv_passed),
        ff_pass_count=sum(1 for result in agent_results if result.ff_passed),
        passed=bool(agent_results)
        and all(result.bv_passed and result.ff_passed for result in agent_results),
    )


def _mean_and_pstdev(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def _write_results(
    path: Path,
    args: CliArgs,
    seed_results: Sequence[SeedResult],
    aggregates: Sequence[AgentAggregate],
) -> None:
    payload = {
        "phase": 4,
        "gate": "BV_1+FF_1",
        "out_dir": str(args.out_dir),
        "skip_train": args.skip_train,
        "min_corr": args.min_corr,
        "cvar_alpha": args.cvar_alpha,
        "total_timesteps": args.total_timesteps,
        "per_seed": [result.to_json() for result in seed_results],
        "aggregate": {aggregate.agent: aggregate.to_json() for aggregate in aggregates},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_summary(
    seed_results: Sequence[SeedResult],
    aggregates: Sequence[AgentAggregate],
    results_path: Path,
) -> None:
    print(f"results: {results_path}")
    print("agent seed BV_1 FF_1 corr checkpoint")
    for result in seed_results:
        print(
            f"{result.agent} {result.seed} "
            f"{_verdict(result.bv_passed)} {_verdict(result.ff_passed)} "
            f"{_format_float(result.bv_corr)} {result.checkpoint}"
        )
    print("agent gate bv_pass ff_pass corr_mean corr_std n")
    for aggregate in aggregates:
        print(
            f"{aggregate.agent} {_verdict(aggregate.passed)} "
            f"{aggregate.bv_pass_count}/{aggregate.n_seeds} "
            f"{aggregate.ff_pass_count}/{aggregate.n_seeds} "
            f"{_format_float(aggregate.corr_mean)} "
            f"{_format_float(aggregate.corr_std)} "
            f"{aggregate.n_seeds}"
        )
    for aggregate in aggregates:
        print(
            f"BV_1[{aggregate.agent}]: {_verdict(aggregate.passed)} "
            f"(corr mean={_format_float(aggregate.corr_mean)} "
            f"std={_format_float(aggregate.corr_std)} n={aggregate.n_seeds})"
        )


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _format_float(value: float | None) -> str:
    if value is None:
        return "nan"
    return f"{value:+.4f}"


if __name__ == "__main__":
    main()
