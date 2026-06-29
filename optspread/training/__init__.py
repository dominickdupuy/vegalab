"""Shared training spine reused unchanged by Phase 3.

This package owns the algorithm-agnostic plumbing — seeding, the env factory,
vector-env construction, causal observation normalization, and metric logging.
PPO (here) and the Phase-3 distributional agent both build on these so the
headline comparison runs on a byte-identical env/eval/metrics substrate (see the
shared-harness contract in CLAUDE.md). Nothing PPO-specific lives here.
"""
