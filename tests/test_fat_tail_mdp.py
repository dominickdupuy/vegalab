"""Fat-tail MDP analytic policy check."""

from __future__ import annotations

from optspread.agents.distributional.risk import RiskMeasure
from optspread.toys.fat_tail_mdp import greedy_mdp_action, mdp_action_distributions


def test_mean_greedy_takes_delayed_tail_cvar_avoids() -> None:
    tail, safe = mdp_action_distributions()
    assert tail.mean > safe.mean
    assert tail.cvar(0.1) < safe.cvar(0.1)
    assert greedy_mdp_action(RiskMeasure.mean()) == 0
    assert greedy_mdp_action(RiskMeasure.cvar(0.1)) == 1
