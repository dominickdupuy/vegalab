"""The algorithm-agnostic Agent contract shared by Phase 2 (PPO) and Phase 3.

The ``Evaluator``, ``MetricSuite`` and ``TrainHarness`` only ever see an ``Agent``;
this is the seam that lets the eventual on-policy-vs-distributional comparison run
through identical eval code. Any new agent — PPO here, QR-DQN/IQN later — conforms
to this and is immediately evaluable and checkpointable.

``act`` consumes a RAW environment observation: an agent that normalizes inputs
owns its (causal) normalizer internally and applies it in ``act``, so the saved
checkpoint is self-contained and eval needs no external normalization state.

Distinct from ``agents/baselines.py``: those are the Phase-1 open-loop scripted
policies (signature ``act(obs)``) used by the smoke-run sanity gate. The agents
here carry the richer ``act(obs, deterministic)`` + ``save``/``load`` contract the
learning harness needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from optspread.actions.library import FLAT_ACTION_ID, N_ACTIONS

Observation = NDArray[np.float32]


@runtime_checkable
class Agent(Protocol):
    """Minimal contract every evaluable/checkpointable policy implements."""

    def act(self, obs: Observation, deterministic: bool) -> int:
        """Map a raw observation to an action id."""
        ...

    def save(self, path: Path) -> None:
        """Persist all state needed to reproduce ``act`` after ``load``."""
        ...

    def load(self, path: Path) -> None:
        """Restore state saved by ``save``."""
        ...


class FlatAgent:
    """Always FLAT — the protocol-conforming do-nothing control for the harness."""

    def act(self, obs: Observation, deterministic: bool) -> int:
        return FLAT_ACTION_ID

    def save(self, path: Path) -> None:  # nothing to persist
        pass

    def load(self, path: Path) -> None:
        pass


class RandomAgent:
    """Uniform-random action each step, threading its own seeded RNG.

    Used by ``test_harness_smoke`` to drive the shared harness end-to-end without
    a network, and as a null comparator for the metrics.
    """

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def act(self, obs: Observation, deterministic: bool) -> int:
        return int(self._rng.integers(0, N_ACTIONS))

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass
