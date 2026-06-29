"""TrainHarness: algorithm-agnostic training/evaluation/checkpoint orchestration."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from optspread.agents.base import Agent
from optspread.eval.evaluator import Evaluator
from optspread.eval.metrics import EvalReport
from optspread.eval.no_edge_gate import NoEdgeResult, evaluate_no_edge
from optspread.training.logging import MetricLogger


class TrainerProtocol(Protocol):
    """Minimal trainer contract: own the algorithm loop and return diagnostics."""

    def train(self) -> Sequence[object]:
        """Run training and return per-update diagnostics."""
        ...


@dataclass(frozen=True, slots=True)
class HarnessResult:
    """Artifacts produced by one training/evaluation run."""

    history: Sequence[object]
    report: EvalReport
    checkpoint_path: Path
    report_path: Path
    returns_path: Path
    gate: NoEdgeResult | None


class TrainHarness:
    """Wire a trainer, evaluator, logger, and checkpoint directory together.

    The harness deliberately does not know PPO internals. Phase 3 can provide a
    different trainer with the same ``train()`` shape and still reuse the same
    evaluator, metric suite, logging, and artifact format.
    """

    def __init__(
        self,
        *,
        agent: Agent,
        trainer: TrainerProtocol,
        evaluator: Evaluator,
        run_dir: str | Path,
        logger: MetricLogger | None = None,
    ) -> None:
        self.agent = agent
        self.trainer = trainer
        self.evaluator = evaluator
        self.run_dir = Path(run_dir)
        self.logger = logger

    def run(
        self,
        *,
        checkpoint_name: str = "agent.pt",
        deterministic_eval: bool = True,
        gate_with_costs: bool | None = None,
        flat_threshold: float | None = None,
        artifact_prefix: str = "eval",
    ) -> HarnessResult:
        """Train, checkpoint, evaluate, optionally gate, and save artifacts."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        history = list(self.trainer.train())

        checkpoint_path = self.run_dir / checkpoint_name
        self.agent.save(checkpoint_path)

        report = self.evaluator.run(self.agent, deterministic=deterministic_eval)
        step = _history_global_step(history)
        if self.logger is not None:
            log_eval_report(self.logger, report, step=step, prefix=artifact_prefix)

        gate = None
        if gate_with_costs is not None and flat_threshold is not None:
            gate = evaluate_no_edge(
                report,
                with_costs=gate_with_costs,
                flat_threshold=flat_threshold,
            )

        report_path, returns_path = save_eval_artifacts(
            report,
            self.run_dir,
            prefix=artifact_prefix,
            gate=gate,
            history=history,
        )
        if self.logger is not None:
            self.logger.close()

        return HarnessResult(
            history=history,
            report=report,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            returns_path=returns_path,
            gate=gate,
        )


def log_eval_report(logger: MetricLogger, report: EvalReport, *, step: int, prefix: str) -> None:
    """Log full return distributions and scalar eval diagnostics."""
    logger.log_distribution(f"{prefix}/episode_returns", report.episode_returns, step)
    logger.log_distribution(f"{prefix}/per_step_returns", report.per_step_returns, step)
    logger.log_scalars(
        {
            f"{prefix}/mean_pnl": report.mean_pnl,
            f"{prefix}/pnl_ci_lo": report.pnl_ci[0],
            f"{prefix}/pnl_ci_hi": report.pnl_ci[1],
            f"{prefix}/sharpe": report.sharpe,
            f"{prefix}/sortino": report.sortino,
            f"{prefix}/cvar_95": report.cvar_95,
            f"{prefix}/max_drawdown": report.max_drawdown,
            f"{prefix}/turnover": report.turnover,
            f"{prefix}/flat_frequency": report.flat_frequency,
        },
        step,
    )
    for action_id, freq in report.action_frequencies.items():
        logger.log_scalar(f"{prefix}/action_freq_{action_id:02d}", freq, step)


def save_eval_artifacts(
    report: EvalReport,
    out_dir: str | Path,
    *,
    prefix: str,
    gate: NoEdgeResult | None = None,
    history: Sequence[object] = (),
) -> tuple[Path, Path]:
    """Persist eval summaries as JSON and full return arrays as NPZ."""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    report_path = path / f"{prefix}_report.json"
    returns_path = path / f"{prefix}_returns.npz"
    payload = {
        "report": report_to_dict(report),
        "gate": no_edge_to_dict(gate) if gate is not None else None,
        "history": [_history_item_to_dict(item) for item in history],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez(
        returns_path,
        per_step_returns=report.per_step_returns,
        episode_returns=report.episode_returns,
    )
    return report_path, returns_path


def report_to_dict(report: EvalReport) -> dict[str, object]:
    """JSON-safe summary of an ``EvalReport``."""
    top_actions = sorted(report.action_frequencies.items(), key=lambda item: item[1], reverse=True)
    return {
        "mean_pnl": _json_number(report.mean_pnl),
        "pnl_ci": [_json_number(report.pnl_ci[0]), _json_number(report.pnl_ci[1])],
        "sharpe": _json_number(report.sharpe),
        "sortino": _json_number(report.sortino),
        "cvar_95": _json_number(report.cvar_95),
        "max_drawdown": _json_number(report.max_drawdown),
        "turnover": _json_number(report.turnover),
        "flat_frequency": _json_number(report.flat_frequency),
        "n_episode_returns": int(report.episode_returns.size),
        "n_per_step_returns": int(report.per_step_returns.size),
        "action_frequencies": {
            str(action_id): _json_number(freq)
            for action_id, freq in report.action_frequencies.items()
        },
        "top_actions": [
            {"action_id": int(action_id), "frequency": _json_number(freq)}
            for action_id, freq in top_actions[:8]
        ],
    }


def no_edge_to_dict(result: NoEdgeResult) -> dict[str, object]:
    """JSON-safe summary of a no-edge decision."""
    return {
        "passed": result.passed,
        "flat_frequency": _json_number(result.flat_frequency),
        "mean_pnl_ci": [
            _json_number(result.mean_pnl_ci[0]),
            _json_number(result.mean_pnl_ci[1]),
        ],
        "reason": result.reason,
    }


def _history_global_step(history: Sequence[object]) -> int:
    if not history:
        return 0
    value = getattr(history[-1], "global_step", 0)
    return int(value) if isinstance(value, int | np.integer) else 0


def _history_item_to_dict(item: object) -> dict[str, object]:
    fields = getattr(item, "__dataclass_fields__", None)
    if isinstance(fields, dict):
        return {str(name): _json_value(getattr(item, str(name))) for name in fields}
    return {"repr": repr(item)}


def _json_value(value: object) -> object:
    if isinstance(value, bool | str) or value is None:
        return value
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return _json_number(float(value))
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _json_number(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None
