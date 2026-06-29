"""Naive VRP/IV-rank heuristic baseline."""

from __future__ import annotations

from optspread.actions.library import FLAT_ACTION_ID


def vrp_heuristic_action(
    features: dict[str, float],
    *,
    credit_action_id: int = 11,
    vrp_threshold: float = 0.0,
    iv_rank_threshold: float = 0.5,
) -> int:
    """Sell a defined-risk credit structure when VRP and IV-rank are high."""
    if (
        features.get("vrp", 0.0) > vrp_threshold
        and features.get("iv_rank", 0.0) >= iv_rank_threshold
    ):
        return credit_action_id
    return FLAT_ACTION_ID
