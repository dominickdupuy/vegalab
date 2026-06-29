"""MetricLogger: scalars, histograms, and full return distributions.

Writes to TensorBoard (the brief's minimum) and also retains an in-memory history
so tests and the CLI can assert on metrics without parsing event files. Logging
the *full* eval return distribution and per-action frequencies — not just the
mean — is a hard requirement: the no-edge gate and the Phase 3 comparison are
statements about distributions, so the distribution must be first-class.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class MetricLogger:
    """Thin TensorBoard wrapper with an in-memory scalar history."""

    def __init__(self, log_dir: str | Path | None = None, *, enabled: bool = True) -> None:
        self.history: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self._writer: Any = None
        if enabled and log_dir is not None:
            from torch.utils.tensorboard.writer import SummaryWriter

            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.history[tag].append((step, float(value)))
        if self._writer is not None:
            self._writer.add_scalar(tag, float(value), step)

    def log_scalars(self, values: dict[str, float], step: int) -> None:
        for tag, value in values.items():
            self.log_scalar(tag, value, step)

    def log_histogram(self, tag: str, values: NDArray[np.float64], step: int) -> None:
        if self._writer is not None:
            self._writer.add_histogram(tag, np.asarray(values), step)

    def log_distribution(self, tag: str, values: NDArray[np.float64], step: int) -> None:
        """Log a distribution as a histogram plus mean/std/p5/p95 summary scalars."""
        arr = np.asarray(values, dtype=np.float64)
        self.log_histogram(f"{tag}/hist", arr, step)
        if arr.size:
            self.log_scalar(f"{tag}/mean", float(arr.mean()), step)
            self.log_scalar(f"{tag}/std", float(arr.std()), step)
            self.log_scalar(f"{tag}/p05", float(np.percentile(arr, 5)), step)
            self.log_scalar(f"{tag}/p95", float(np.percentile(arr, 95)), step)

    def latest(self, tag: str) -> float | None:
        """Most recent value logged under ``tag`` (for programmatic checks)."""
        series = self.history.get(tag)
        return series[-1][1] if series else None

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
