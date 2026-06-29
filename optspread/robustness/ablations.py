"""Reward ablation config helpers."""

from __future__ import annotations

from optspread.config import RewardConfig


def drop_reward_term(config: RewardConfig, term: str) -> RewardConfig:
    """Return a reward config with one component weight set to zero."""
    mapping = {
        "mtm": "mtm_weight",
        "margin": "margin_normalized_weight",
        "sharpe": "sharpe_weight",
        "sortino": "sortino_weight",
        "cvar": "cvar_weight",
    }
    if term not in mapping:
        raise ValueError(f"unknown reward term {term!r}")
    return config.model_copy(update={mapping[term]: 0.0})
