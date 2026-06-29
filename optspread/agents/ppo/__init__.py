"""PPO baseline agent (Phase 2).

Adapted from CleanRL's single-file ``ppo.py`` (Huang et al., 2022,
https://github.com/vwxyzjn/cleanrl) — used as the line-by-line correctness
reference — restructured into the repo's module layout: ``network`` (actor-critic),
``buffer`` (rollout storage + GAE), ``config`` (hyperparameters), ``ppo_agent``
(the inference-side ``Agent``), and ``trainer`` (rollout + clipped update loop).

The entropy bonus lives HERE, in the PPO objective — never in the reward (Phase-1
invariant #4). PPO consumes the full Phase-1 composite reward unchanged.
"""
