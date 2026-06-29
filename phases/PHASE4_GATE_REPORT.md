# Phase 4 Gate Report

Generated: 2026-06-28T22:10:00

Overall implementation status: **FOUNDATION + WAVE 1 GV PASS + BV TOOLING PASS**

## Implemented

- `IVSurface` standardized delta/maturity grid and chain derivation.
- Backward-compatible `MarketSnapshot.surface` field.
- `EnvBundle.generator_factory` hook for non-GBM synthetic generators.
- Black-Scholes pricer seam, characteristic-function module, and MC oracle for
  pricer cross-checks.
- Domain-randomization priors and `ParamSampler`.
- Causal regime-feature helpers.
- Wave 1 `GBMVRPGenerator`, expressing VRP as physical path volatility below
  risk-neutral implied volatility.
- Generator-validation scaffolding, behavioral-stat helpers, pre-registered
  Wave 1 prediction, promotion-gate logic, rehearsal helper, Wave registry, and
  frame-stack wrapper.
- Wave 1 behavioral-validation rollout trace and `validate_behavior` CLI for
  checking `corr(credit_indicator, vrp)` on trained PPO/QR-DQN/IQN checkpoints.

## GV_1 — Wave 1 Generator Validation

Status: **PASS**

Run:

```bash
python -m optspread.cli.validate_generators --wave 1
```

The automated test `tests/test_vrp_invariant.py` validates that
`mean(IV^2 - realized^2)` is positive under the configured premium.

## Pending Phase-4 Gates

- BV_1: train PPO and distributional agents on Wave 1 and validate that
  credit-structure frequency rises with the VRP feature via
  `python -m optspread.cli.validate_behavior --wave 1 ...`.
- FF_1: re-evaluate Wave 0 after Wave 1 training to rule out catastrophic
  forgetting.
- Waves 2–6: not started. Per the Phase 4 brief, these should be added one at a
  time only after GV/BV/FF pass for the current wave.

## Validation

Repository quality gates are green:

```bash
python -m ruff check optspread tests
python -m ruff format --check optspread tests
python -m mypy --strict optspread
python -m pytest tests/ --cov=optspread --cov-report=term-missing
```

Current full-suite result after Phase 5–8 scaffolding: **166 passed**, strict
mypy green, ruff green.
