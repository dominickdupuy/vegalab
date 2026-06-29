"""Global seeding protocol and per-run directories.

Phase 1 threads a single ``numpy.random.Generator`` explicitly and never touches
the global ``np.random.*`` state (determinism invariant in CLAUDE.md). Phase 2
adds PyTorch and Python's ``random``; we seed those here. We deliberately do NOT
call ``np.random.seed`` — numpy randomness stays threaded through ``make_rng`` and
the per-env seed sequences, so the env's determinism contract is untouched.

On CPU the small actor-critic MLP is deterministic given a fixed torch seed, so a
full run reproduces byte-for-byte. ``deterministic_torch`` additionally turns on
``use_deterministic_algorithms`` for defence in depth; on CUDA there is residual
nondeterminism we do not chase in this phase.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python ``random`` and PyTorch. Numpy is seeded per-component, not here."""
    random.seed(seed)
    torch.manual_seed(seed)
    if deterministic_torch:
        # CPU MLP path is already deterministic; this guards against silent
        # nondeterministic kernels if the network ever grows.
        torch.use_deterministic_algorithms(True, warn_only=True)


def run_dir(base: str | Path, run_name: str, seed: int) -> Path:
    """Create and return ``base/run_name/seed_<seed>`` for a single run's artifacts."""
    path = Path(base) / run_name / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path
