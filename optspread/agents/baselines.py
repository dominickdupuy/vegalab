"""Scripted, non-learning baseline agents.

These exist to exercise the env and to anchor the Wave-0 economic sanity gate
(see CLAUDE.md). They are NOT RL agents — no learning, no state beyond an RNG:

- ``FlatAgent``      always picks FLAT (action 0); the do-nothing control.
- ``AlwaysOnAgent``  always holds one fixed credit structure; with fair IV this
                     must read ~0 mean P&L with no costs and < 0 with costs.
- ``RandomAgent``    picks a uniformly random action each step; it churns the
                     book and therefore bleeds transaction costs the fastest.

Each agent maps an observation to an action id. The observation is ignored by the
scripted policies (they are open-loop), but the signature matches what a Phase-2
policy will use.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from optspread.actions.library import FLAT_ACTION_ID, N_ACTIONS

Observation = NDArray[np.float32]


class Agent(Protocol):
    def act(self, obs: Observation) -> int:
        """Return an action id for the given observation."""
        ...

    def reset(self) -> None:
        """Clear any per-episode state."""
        ...


class FlatAgent:
    """Always FLAT: the do-nothing control. Pays no costs, holds no risk."""

    def reset(self) -> None:
        pass

    def act(self, obs: Observation) -> int:
        return FLAT_ACTION_ID


class AlwaysOnAgent:
    """Always holds one fixed structure (default: a short strangle, action 14).

    Because the env only trades when the target differs from what is held, this
    pays the open cost once and then carries the position — the cleanest probe of
    the fair-IV zero-expectancy property.
    """

    def __init__(self, action_id: int = 14) -> None:
        if not 0 <= action_id < N_ACTIONS:
            raise ValueError(f"action_id {action_id} out of range")
        self.action_id = action_id

    def reset(self) -> None:
        pass

    def act(self, obs: Observation) -> int:
        return self.action_id


class RandomAgent:
    """Uniformly random action each step. Threads its own seeded RNG.

    It constantly re-establishes different structures, so it crosses the spread
    repeatedly and bleeds transaction costs faster than the always-on agent.
    """

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        # Re-seed so repeated episodes from the same agent are reproducible.
        self._rng = np.random.default_rng(self._seed)

    def act(self, obs: Observation) -> int:
        return int(self._rng.integers(0, N_ACTIONS))
