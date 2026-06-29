"""Shared evaluation harness reused unchanged by Phase 3.

``MetricSuite`` (distributional metrics), ``Evaluator`` (fixed-seed policy rollout
to an ``EvalReport``) and ``no_edge_gate`` (the Phase-2 definition of done). These
are algorithm-agnostic: PPO and the Phase-3 distributional agent are scored by the
same instances over the same eval seeds, so the comparison is apples-to-apples by
construction (CLAUDE.md shared-harness contract).
"""
