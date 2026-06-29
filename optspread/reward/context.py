"""StepContext: the immutable per-step information a reward sees.

The env assembles one of these each ``step()`` and hands it to the reward. It
carries only what is realised at the close of the step — never any look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StepContext:
    """Everything a reward component is allowed to look at for one step.

    Attributes
    ----------
    pnl:
        Dollar mark-to-market change in equity over the step, already net of any
        transaction costs charged this step. This is the raw reward signal.
    margin:
        Buying-power requirement held *during* the step (>= 0). Used to
        risk-normalise the P&L.
    equity:
        Account equity at the close of the step.
    day:
        Episode day index at the close of the step (0-based).
    did_trade:
        Whether a position was opened or closed this step.
    """

    pnl: float
    margin: float
    equity: float
    day: int
    did_trade: bool
