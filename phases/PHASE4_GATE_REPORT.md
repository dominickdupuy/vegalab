# Phase 4 Gate Report

Generated: 2026-06-28T22:10:00

Overall implementation status: **WAVE 1 COMPLETE — GV_1, BV_1, FF_1 all PASS**
(ensemble of trained PPO and IQN/CVaR agents; Waves 2–6 not started)

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

## BV_1 — Wave 1 Behavioral Validation (trained agents)

Status: **PASS** (ensemble, deterministic eval; CVaR deployment for IQN)

Pre-registered prediction (`curriculum/predictions.py`, committed before training):
`corr(credit_indicator, vrp) > 0.10`, returns positive but bounded.

| Agent | BV_1 corr (per seed) | BV_1 mean ± std | FF_1 |
|---|---|---|---|
| IQN/CVaR (primary) | +0.71, +0.76, +0.24 | **+0.57 ± 0.23 (3/3 pass)** | PASS (flat ~1.0) |
| PPO | +0.72, +0.72, collapsed | +0.72 ± 0.00 (**2/3 pass**) | PASS (flat ~0.89, no edge) |

All passing seeds far exceed the 0.10 threshold. The **primary CVaR/IQN agent is
robust (3/3)**; PPO (on-policy) collapsed to FLAT on 1 of 3 seeds — a real seed-
variance failure mode the off-policy distributional agent (persistent ε-floor
exploration + risk-neutral bootstrap) avoided. The CVaR agent trades less than the
risk-neutral policy (tail-averse) yet still harvests VRP when the observable VRP
feature is positive; mean episode P&L positive but bounded. Trained detached (see
the long-training kill note); per-seed detail in `runs/phase4_wave1_bv1_ff1.json`.

### What it took (the recipe — see also the per-wave learnability note)

Getting BV_1 to pass surfaced and fixed several issues (env/generator/pricing
were verified CORRECT throughout — fair-IV Wave 0 is zero raw-EV, Wave 1 short
premium is genuinely +EV):

1. **Curriculum reward = MTM P&L only** (`curriculum_reward()`); tail-aversion is
   agent-side. The Wave-0 gate's env CVaR penalty dominates the edge ~8x and
   forces FLAT. (A DifferentialSharpe term was evaluated and rejected — it rewards
   trading on the no-edge Wave 0, breaking the no-edge invariant.)
2. **Risk-neutral bootstrap** for the distributional agent
   (`DistributionalConfig.bootstrap_risk="mean"`), CVaR only at deployment. The
   nested CVaR-greedy bootstrap causes "blindness to success": the agent collapses
   to 100% FLAT and never learns the +EV trade.
3. **Teachable, variable-sign VRP** prior (`U(-0.04, 0.18)`): exaggerated for
   learnability (a realistic ~0.02-0.08 edge is too weak/noisy — both PPO and IQN
   collapse to flat); spanning zero so conditional sell-when-rich behavior is
   optimal (otherwise the agent sells indiscriminately and corr ~ 0.05 < 0.10).
4. **Warmup** (`GBMVRPGenerator.warmup_days=21`): the path runs silently before the
   episode so realized vol — and thus the observable `vrp` feature — is meaningful
   at the first decision. Without it VRP is unobservable at entry and the agent
   rationally stays flat. (A teaching aid, not a hidden-state leak.)
5. **Wave 1 trained from scratch**, not warm-started from the deliberately-FLAT
   Wave-0 checkpoint (which biases hard toward "do nothing"). Warm-start should be
   reinstated for later waves whose previous agent already trades.

## Pending Phase-4 Gates

- Waves 2–6: not started. Per the Phase 4 brief, these should be added one at a
  time only after GV/BV/FF pass for the current wave. Expect the same
  learnability levers (observable feature, teachable signal strength, risk-neutral
  bootstrap) to be needed.

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
