"""Zero-shot walk-forward evaluation on REAL OptionMetrics SPX surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from optspread.agents.base import Agent, FlatAgent, Observation
from optspread.agents.distributional.iqn_agent import IQNAgent
from optspread.agents.distributional.qrdqn_agent import QRDQNAgent
from optspread.agents.ppo.ppo_agent import PPOAgent
from optspread.baselines.vrp_heuristic import vrp_heuristic_action
from optspread.config import CostConfig, EnvConfig, GBMConfig
from optspread.data.optionmetrics_loader import SurfaceRow, load_surface_csv
from optspread.data.real_generator import RealDataReplay
from optspread.envs.builder import EnvBundle
from optspread.eval.evaluator import Evaluator
from optspread.eval.metrics import EvalReport, MetricSuite
from optspread.evaluation.deflated_sharpe import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from optspread.evaluation.significance import bootstrap_mean_difference_ci
from optspread.evaluation.walkforward import Fold, WalkForwardSplitter
from optspread.market.snapshot import REGIME_FEATURE_KEYS
from optspread.training.env_factory import EnvFactory
from optspread.training.phase2 import curriculum_reward

POLICY_CHOICES: tuple[str, ...] = ("ppo", "qrdqn", "iqn", "flat", "vrp_heuristic")
RL_POLICIES = frozenset({"ppo", "qrdqn", "iqn"})
BASELINE_POLICIES = ("flat", "vrp_heuristic")
TRADING_DAYS_PER_YEAR = 252

FloatArray = NDArray[np.float64]


class VRPHeuristicAgent:
    """Protocol-conforming wrapper around the naive VRP/IV-rank baseline."""

    def act(self, obs: Observation, deterministic: bool) -> int:
        features = {key: float(obs[idx]) for idx, key in enumerate(REGIME_FEATURE_KEYS)}
        return vrp_heuristic_action(features)

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="REAL data zero-shot walk-forward evaluation for frozen policies."
    )
    parser.add_argument(
        "--surface-csv", type=Path, default=Path("data/optionmetrics_spx_surface.csv")
    )
    parser.add_argument(
        "--agent",
        dest="agents",
        action="append",
        choices=POLICY_CHOICES,
        help="Policy to evaluate; repeat for multiple policies.",
    )
    parser.add_argument(
        "--agent-kind",
        dest="agent_kinds",
        action="append",
        choices=POLICY_CHOICES,
        help="Alias for --agent, kept for checkpoint-oriented runs.",
    )
    parser.add_argument("--checkpoint", type=Path, help="Frozen RL checkpoint path.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episode-length", type=int, default=63)
    parser.add_argument("--train-size", type=int, default=1260)
    parser.add_argument("--test-size", type=int, default=252)
    parser.add_argument("--purge", type=int, default=21)
    parser.add_argument("--embargo", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=21)
    parser.add_argument("--max-folds", type=int, help="Optional smoke-test fold cap.")
    parser.add_argument(
        "--cost-mult",
        dest="cost_mults",
        action="append",
        type=float,
        help="Transaction-cost multiplier; repeat for stress runs.",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/phase5_real_eval.json"))
    args = parser.parse_args()

    _validate_args(args)
    cost_mults = _cost_multipliers(args)
    rows = load_surface_csv(args.surface_csv)
    folds = WalkForwardSplitter(
        train_size=args.train_size,
        test_size=args.test_size,
        purge=args.purge,
        embargo=args.embargo,
    ).split(len(rows))
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    if not folds:
        raise SystemExit("walk-forward splitter produced no folds")

    policy_names = _policy_names(args)
    policies = [(name, _load_policy(name, args.checkpoint, args.device)) for name in policy_names]

    n_policies = len(policies)
    cost_stress: dict[str, object] = {}
    edge_by_policy: dict[str, list[tuple[float, float]]] = {}
    for cost_mult in cost_mults:
        per_policy: dict[str, object] = {}
        pooled_returns: dict[str, FloatArray] = {}
        for name, policy in policies:
            reports: list[EvalReport] = []
            fold_payloads: list[dict[str, object]] = []
            for fold in folds:
                report = _run_fold(policy, rows, fold, args.warmup, cost_mult)
                reports.append(report)
                fold_payloads.append(_fold_payload(report, rows, fold))
            pooled = _pool_returns(reports)
            pooled_returns[name] = pooled
            per_policy[name] = {
                "label": "REAL zero-shot frozen-agent evaluation",
                "per_fold": fold_payloads,
                "aggregate": _aggregate_payload(reports, pooled, n_policies),
            }

        comparisons = _comparison_payloads(pooled_returns, policy_names)
        cost_stress[_cost_label(cost_mult)] = {
            "per_policy": per_policy,
            "comparisons": comparisons,
        }
        _record_rl_edges(edge_by_policy, policy_names, pooled_returns, cost_mult)

    break_even = _break_even_payload(edge_by_policy)
    payload: dict[str, object] = {
        "config": {
            "label": "REAL data, zero-shot, frozen agents, no training",
            "surface_csv": str(args.surface_csv),
            "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
            "policies": policy_names,
            "cost_multipliers": cost_mults,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "purge": args.purge,
            "embargo": args.embargo,
            "warmup": args.warmup,
            "episode_length_arg": args.episode_length,
            "fold_episode_steps": args.test_size - 1,
            "max_folds": args.max_folds,
            "n_folds": len(folds),
            "frozen_agent": True,
            "zero_gradient_steps": True,
        },
        "cost_stress": cost_stress,
        "break_even": break_even,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"REAL zero-shot frozen-agent walk-forward; wrote {args.out}")
    for cost_mult in cost_mults:
        cost_payload = cost_stress[_cost_label(cost_mult)]
        if not isinstance(cost_payload, dict):
            raise TypeError("cost stress payload must be a dict")
        per_policy_payload = cost_payload["per_policy"]
        if not isinstance(per_policy_payload, dict):
            raise TypeError("per-policy payload must be a dict")
        comparisons_payload = cost_payload["comparisons"]
        if not isinstance(comparisons_payload, dict):
            raise TypeError("comparisons payload must be a dict")
        print(f"=== cost x{_cost_label(cost_mult)} ===")
        print(_table(policy_names, per_policy_payload, comparisons_payload))
    print(_break_even_note(break_even))


def _validate_args(args: argparse.Namespace) -> None:
    if args.train_size <= 0:
        raise SystemExit("--train-size must be positive")
    if args.test_size < 2:
        raise SystemExit("--test-size must be at least 2")
    if args.episode_length <= 0:
        raise SystemExit("--episode-length must be positive")
    if args.purge < 0 or args.embargo < 0 or args.warmup < 0:
        raise SystemExit("--purge, --embargo, and --warmup must be non-negative")
    if args.max_folds is not None and args.max_folds <= 0:
        raise SystemExit("--max-folds must be positive when provided")
    if args.cost_mults is not None and any(cost_mult < 0.0 for cost_mult in args.cost_mults):
        raise SystemExit("--cost-mult must be non-negative")


def _cost_multipliers(args: argparse.Namespace) -> list[float]:
    if args.cost_mults is None:
        return [1.0]
    return _dedupe_floats(args.cost_mults)


def _policy_names(args: argparse.Namespace) -> list[str]:
    requested: list[str] = []
    if args.agents is not None:
        requested.extend(args.agents)
    if args.agent_kinds is not None:
        requested.extend(args.agent_kinds)

    if not requested:
        if args.checkpoint is not None:
            raise SystemExit("--checkpoint requires --agent or --agent-kind in {ppo,qrdqn,iqn}")
        requested.extend(BASELINE_POLICIES)

    has_rl = any(name in RL_POLICIES for name in requested)
    if args.checkpoint is not None and not has_rl:
        raise SystemExit("--checkpoint was provided, but no RL policy was requested")
    if args.checkpoint is None and has_rl:
        raise SystemExit("RL policy evaluation requires --checkpoint")
    if has_rl:
        requested.extend(BASELINE_POLICIES)

    return _dedupe(requested)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _dedupe_floats(items: list[float]) -> list[float]:
    seen: set[float] = set()
    out: list[float] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _load_policy(kind: str, checkpoint: Path | None, device: str) -> Agent:
    if kind == "flat":
        return FlatAgent()
    if kind == "vrp_heuristic":
        return VRPHeuristicAgent()
    if checkpoint is None:
        raise ValueError(f"{kind} requires a checkpoint")
    if kind == "ppo":
        return PPOAgent.from_checkpoint(checkpoint, device=device)
    if kind == "qrdqn":
        return QRDQNAgent.from_checkpoint(checkpoint, device=device)
    if kind == "iqn":
        return IQNAgent.from_checkpoint(checkpoint, device=device)
    raise ValueError(f"unsupported policy: {kind}")


def _run_fold(
    agent: Agent,
    rows: list[SurfaceRow],
    fold: Fold,
    warmup: int,
    cost_mult: float,
) -> EvalReport:
    test_rows = rows[fold.test_start : fold.test_end]
    lead_in = rows[max(0, fold.test_start - warmup) : fold.test_start]
    fold_rows = lead_in + test_rows
    warmup_rows = len(lead_in)
    n_steps = len(test_rows) - 1
    gbm = GBMConfig(n_days=n_steps)

    def generator_factory() -> RealDataReplay:
        return RealDataReplay(fold_rows, gbm, warmup_rows=warmup_rows)

    factory = EnvFactory(
        EnvBundle(
            env=EnvConfig(episode_length=n_steps),
            gbm=gbm,
            cost=_scaled_cost_config(cost_mult),
            reward=curriculum_reward(),
            generator_factory=generator_factory,
        )
    )
    return Evaluator(factory, eval_seeds=(0,), metrics=MetricSuite()).run(
        agent,
        deterministic=True,
    )


def _scaled_cost_config(cost_mult: float) -> CostConfig:
    defaults = CostConfig()
    return CostConfig(
        half_spread_bps=defaults.half_spread_bps * cost_mult,
        min_cost_per_leg=defaults.min_cost_per_leg * cost_mult,
        otm_widening=defaults.otm_widening,
        multiplier=defaults.multiplier,
    )


def _fold_payload(report: EvalReport, rows: list[SurfaceRow], fold: Fold) -> dict[str, object]:
    return {
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "start_date": rows[fold.test_start].date,
        "end_date": rows[fold.test_end - 1].date,
        "per_step_returns": _float_list(report.per_step_returns),
        "sharpe": report.sharpe,
        "sortino": report.sortino,
        "cvar_95": report.cvar_95,
        "max_drawdown": report.max_drawdown,
        "mean_pnl": report.mean_pnl,
        "flat_frequency": report.flat_frequency,
        "action_frequencies": _top_action_frequencies(report.action_frequencies),
    }


def _aggregate_payload(
    reports: list[EvalReport],
    pooled: FloatArray,
    n_policies: int,
) -> dict[str, object]:
    fold_sharpes = np.asarray([report.sharpe for report in reports], dtype=np.float64)
    sharpe = _sharpe(pooled)
    n_returns = int(pooled.size)
    return {
        "n_steps": n_returns,
        "n_folds": len(reports),
        "sharpe": sharpe,
        "sortino": _sortino(pooled),
        "cvar_95": _cvar_95(pooled),
        "max_drawdown": _max_drawdown_from_pooled_pnl(pooled),
        "mean_pnl": float(pooled.mean()) if pooled.size else 0.0,
        "mean_daily_pnl": float(pooled.mean()) if pooled.size else 0.0,
        "probabilistic_sharpe_ratio": probabilistic_sharpe_ratio(sharpe, 0.0, n_returns),
        "deflated_sharpe_ratio": deflated_sharpe_ratio(
            sharpe,
            n_returns=n_returns,
            n_trials=n_policies,
        ),
        "fold_sharpe_mean": float(fold_sharpes.mean()) if fold_sharpes.size else 0.0,
        "fold_sharpe_std": float(fold_sharpes.std(ddof=1)) if fold_sharpes.size > 1 else 0.0,
        "flat_frequency_mean": float(
            np.asarray([report.flat_frequency for report in reports], dtype=np.float64).mean()
        )
        if reports
        else 0.0,
    }


def _comparison_payloads(
    pooled_returns: dict[str, FloatArray],
    policy_names: list[str],
) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    flat = pooled_returns.get("flat")
    heuristic = pooled_returns.get("vrp_heuristic")
    for name in policy_names:
        returns = pooled_returns[name]
        policy_comparisons: dict[str, object] = {}
        if flat is not None:
            policy_comparisons["vs_flat_mean_pnl_ci"] = _paired_ci(returns, flat)
        if heuristic is not None:
            policy_comparisons["vs_vrp_heuristic_mean_pnl_ci"] = _paired_ci(returns, heuristic)
        comparisons[name] = policy_comparisons
    return comparisons


def _record_rl_edges(
    edge_by_policy: dict[str, list[tuple[float, float]]],
    policy_names: list[str],
    pooled_returns: dict[str, FloatArray],
    cost_mult: float,
) -> None:
    flat = pooled_returns.get("flat")
    if flat is None:
        return
    flat_mean = float(flat.mean()) if flat.size else 0.0
    for name in policy_names:
        if name not in RL_POLICIES:
            continue
        returns = pooled_returns[name]
        mean_edge = (float(returns.mean()) if returns.size else 0.0) - flat_mean
        edge_by_policy.setdefault(name, []).append((cost_mult, mean_edge))


def _break_even_payload(edge_by_policy: dict[str, list[tuple[float, float]]]) -> object:
    if not edge_by_policy:
        return None
    per_policy: dict[str, object] = {}
    for name, edges in edge_by_policy.items():
        ordered = sorted(edges, key=lambda item: item[0])
        break_even = next((cost_mult for cost_mult, edge in ordered if edge <= 0.0), None)
        per_policy[name] = {
            "cost_mult": break_even,
            "edge_mean_pnl_vs_flat": {_cost_label(cost_mult): edge for cost_mult, edge in ordered},
        }
    return {
        "definition": "Smallest tested cost multiplier where pooled mean PnL edge vs FLAT <= 0.",
        "per_policy": per_policy,
    }


def _break_even_note(break_even: object) -> str:
    if break_even is None:
        return "Break-even cost: n/a (no RL agent evaluated)."
    if not isinstance(break_even, dict):
        raise TypeError("break-even payload must be a dict")
    per_policy = break_even["per_policy"]
    if not isinstance(per_policy, dict):
        raise TypeError("break-even per-policy payload must be a dict")
    notes: list[str] = []
    for name, payload in per_policy.items():
        if not isinstance(name, str):
            raise TypeError("break-even policy key must be a string")
        if not isinstance(payload, dict):
            raise TypeError("break-even policy payload must be a dict")
        cost_mult = payload["cost_mult"]
        edges = payload["edge_mean_pnl_vs_flat"]
        if not isinstance(edges, dict):
            raise TypeError("break-even edge payload must be a dict")
        max_cost_mult = max(float(label) for label in edges)
        if cost_mult is None:
            notes.append(f"{name} still beats FLAT through x{_cost_label(max_cost_mult)}")
        else:
            notes.append(f"{name} stops beating FLAT at x{_cost_label(_payload_float(cost_mult))}")
    return "Break-even cost: " + "; ".join(notes) + "."


def _cost_label(cost_mult: float) -> str:
    return str(float(cost_mult))


def _paired_ci(a: FloatArray, b: FloatArray) -> list[float]:
    lo, hi = bootstrap_mean_difference_ci(a, b)
    return [lo, hi]


def _pool_returns(reports: list[EvalReport]) -> FloatArray:
    if not reports:
        return np.asarray([], dtype=np.float64)
    return np.concatenate([report.per_step_returns for report in reports]).astype(np.float64)


def _float_list(values: FloatArray) -> list[float]:
    return [float(value) for value in values]


def _top_action_frequencies(frequencies: dict[int, float], limit: int = 5) -> dict[str, float]:
    items = [(action, freq) for action, freq in frequencies.items() if freq > 0.0]
    items.sort(key=lambda item: (-item[1], item[0]))
    return {str(action): float(freq) for action, freq in items[:limit]}


def _sharpe(returns: FloatArray) -> float:
    if returns.size < 2:
        return 0.0
    std = returns.std(ddof=1)
    return float(returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR)) if std > 1e-12 else 0.0


def _sortino(returns: FloatArray) -> float:
    if returns.size < 2:
        return 0.0
    downside = np.minimum(returns, 0.0)
    downside_dev = np.sqrt((downside**2).mean())
    if downside_dev <= 1e-12:
        return 0.0
    return float(returns.mean() / downside_dev * np.sqrt(TRADING_DAYS_PER_YEAR))


def _cvar_95(returns: FloatArray) -> float:
    if returns.size == 0:
        return 0.0
    ordered = np.sort(returns)
    k = max(1, int(returns.size * 0.05))
    return float(ordered[:k].mean())


def _max_drawdown_from_pooled_pnl(returns: FloatArray) -> float:
    if returns.size == 0:
        return 0.0
    equity = EnvConfig().initial_cash + np.concatenate(
        [np.asarray([0.0], dtype=np.float64), np.cumsum(returns)]
    )
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / np.where(running_max != 0.0, running_max, 1.0)
    return float(drawdown.max())


def _table(
    policy_names: list[str],
    per_policy: dict[str, object],
    comparisons: dict[str, object],
) -> str:
    lines = [
        "| Policy | Sharpe | Sortino | CVaR95 | MaxDD | Mean PnL | PnL CI vs FLAT |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in policy_names:
        policy_payload = per_policy[name]
        if not isinstance(policy_payload, dict):
            raise TypeError("policy payload must be a dict")
        aggregate = policy_payload["aggregate"]
        if not isinstance(aggregate, dict):
            raise TypeError("aggregate payload must be a dict")
        ci = _table_ci(name, comparisons)
        lines.append(
            "| "
            f"{name} | "
            f"{_payload_float(aggregate['sharpe']):+.3f} | "
            f"{_payload_float(aggregate['sortino']):+.3f} | "
            f"{_payload_float(aggregate['cvar_95']):+.2f} | "
            f"{_payload_float(aggregate['max_drawdown']):.3f} | "
            f"{_payload_float(aggregate['mean_pnl']):+.2f} | "
            f"{ci} |"
        )
    return "\n".join(lines)


def _table_ci(name: str, comparisons: dict[str, object]) -> str:
    comparison = comparisons[name]
    if not isinstance(comparison, dict):
        raise TypeError("comparison payload must be a dict")
    ci = comparison.get("vs_flat_mean_pnl_ci")
    if not isinstance(ci, list) or len(ci) != 2:
        return "n/a"
    return f"[{float(ci[0]):+.2f}, {float(ci[1]):+.2f}]"


def _payload_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    raise TypeError(f"expected numeric payload value, got {type(value).__name__}")


if __name__ == "__main__":
    main()
