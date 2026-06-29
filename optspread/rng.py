"""RNG helpers.

Every source of randomness in the project flows through a single
``numpy.random.Generator`` that is created here and threaded explicitly into
the environment and generator. We NEVER touch the global ``np.random.*`` state;
doing so would break determinism (see the determinism invariant in CLAUDE.md).
"""

from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Construct a fresh, independent ``Generator`` seeded deterministically.

    Same seed in => identical stream out. This is the only sanctioned way to
    obtain randomness in optspread.
    """
    return np.random.default_rng(seed)
