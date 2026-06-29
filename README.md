# optspread — Phase 2 complete

A deterministic SPX options spread-selection RL research scaffold. Phase 1 built
the leak-resistant Gymnasium environment, cost model, margin model, reward
components, and fair-IV Wave-0 sanity checks. Phase 2 adds the shared training /
evaluation harness and a PPO baseline, then validates the **Wave-0 no-edge gate**
across multiple seeds.

Everything is dependency-injected and built to be swapped later
(GBM → Heston/Bates, synthetic costs → OptionMetrics quotes, reward ablations).
See `CLAUDE.md` for the non-negotiable invariants.

## Layout

| Package | Responsibility |
|---|---|
| `pricing/` | Black-Scholes price/greeks + delta→strike solver (pure, vectorized) |
| `instruments/` | `OptionLeg`, `ChainSnapshot`, `MarketSnapshot` value objects |
| `actions/` | Spread templates (Strategy), the locked `ACTION_LIBRARY`, margin classes |
| `market/` | `PriceGenerator` protocol + `GBMGenerator` (fair-IV, zero-drift) |
| `portfolio/` | Pure P&L functions + the mutable `Portfolio` cash/position book |
| `costs/` | `CostModel` protocol + `QuotedSpreadCost` |
| `margin/` | `MarginModel` protocol + `RegTStyleMargin` (structural defined/undefined risk) |
| `reward/` | `RewardComponent`s + weighted `CompositeReward` (no entropy — that's agent-side) |
| `envs/` | `ObservationBuilder` (fixed schema) + `SpreadEnv` (pure dependency injection) |
| `agents/` | Scripted baselines plus PPO actor-critic, rollout buffer, trainer, checkpointable agent |
| `training/` | Shared `EnvFactory`, vector envs, causal normalization, logging, `TrainHarness` |
| `eval/` | Shared `Evaluator`, `MetricSuite`, full return distributions, no-edge gate logic |
| `cli/` | `smoke_run`, PPO train/evaluate commands, and the complete Phase-2 gate runner |

## Quick start

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
make check                                   # ruff + mypy --strict + pytest --cov
python -m optspread.cli.smoke_run --episodes 400
python -m optspread.cli.phase2_gate --no-tensorboard
```

Build an env in code (all dependencies injected; the builder picks the Wave-0
concrete implementations):

```python
from optspread.envs.builder import build_default_env

env = build_default_env()
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(14)  # short-strangle action
```

## Time decay & rolling (the Phase-1 model)

Held legs carry an **absolute expiry day** and their time-to-expiry decays as the
path advances, so a short option genuinely earns theta. Newly-opened legs always
get fresh 21/42-day tenors ("measured from each step's today"). When a held
position reaches expiry the env settles it at intrinsic (no spread cost) and the
agent may re-establish — so an always-on credit agent rolls roughly every 21 days.

Freezing the tenor instead (the original scaffold) removed theta and made every
short structure bleed ~$3.8k/contract of unhedged gamma — a violation of the
Wave-0 expectation that the test suite now guards against.

## Wave-0 economic sanity gate

The generator prices the chain at the **same sigma that drives the path** (fair
IV, zero drift), so the expectations are pinned and falsifiable:

- **No costs:** every structure is ~zero-expectancy. An always-on credit agent
  reads ~0 mean P&L within Monte-Carlo noise.
- **With costs:** every structure has negative expectancy, FLAT dominates, and a
  churning random agent bleeds the spread fastest.

If an always-on agent *makes money* with fair IV and no costs, that is a **bug**
(pricing inconsistency, cost sign error, or look-ahead) — not a discovery.

Representative run (`--episodes 400`, mean P&L per 63-day episode ± standard error):

| Agent | No costs | With costs |
|---|---:|---:|
| FLAT | `+0.00 ± 0.00` | `+0.00 ± 0.00` |
| ALWAYS-ON (short strangle) | `+608 ± 595` | `-1113 ± 595` |
| RANDOM | `+153 ± 495` | `-67583 ± 580` |

```
Sanity gate:
  [PASS] no-cost always-on ~ 0      (|mean| <= 3*se)
  [PASS] with-cost always-on < 0
  [PASS] random bleeds faster
  Wave-0 expectation HOLDS.
```

(The exact numbers vary with seed/episode count; the *inequalities* are what the
gate — and `tests/test_baselines.py` — enforce.)

## Phase-2 PPO no-edge gate

The PPO baseline is trained only on Wave 0. The definition of done is not
profitability: it is evidence that PPO **does not** find systematic edge in a
fair-IV, zero-drift simulator.

Run:

```bash
python -m optspread.cli.phase2_gate --no-tensorboard
```

The current official report is `phases/PHASE2_GATE_REPORT.md`: three
risk-adjusted PPO seeds pass FLAT-dominance with default costs, and three pure-PnL
no-cost ablation seeds show no statistically reliable positive mean P&L. This
completes Phase 2 and permits Phase 3 distributional-agent work.

## Phase-3 distributional agents

Phase 3 scaffolding is implemented under `optspread/agents/distributional/`:

- QR-DQN fixed-quantile network and CVaR/mean action selection.
- IQN cosine tau embedding network and `U(0, alpha)` CVaR acting.
- Uniform replay buffer, quantile Huber loss, off-policy trainer, and toy
  fat-tail bandit/MDP validations.
- `train_distributional.py` and `compare.py` CLIs using the same Phase-2
  evaluator and metrics.

See `phases/PHASE3_GATE_REPORT.md` for current gate status. G1-G6 pass, including
compact multi-seed QR-DQN/IQN Wave-0 no-edge checks, so the project may proceed
to Phase 4.

## Phase-4 curriculum foundation

Phase 4 has started with the surface-first architecture and Wave 1 generator:

- `IVSurface` standardizes synthetic surfaces on a delta/maturity grid and
  derives tradeable chains for the existing env.
- `GBMVRPGenerator` adds the first curriculum feature: options priced richer than
  physical realized volatility.
- Domain-randomization priors, causal feature helpers, generator-validation,
  behavioral-validation tracing, promotion-gate scaffolding, rehearsal helper,
  and frame stacking are in place.

Run generator validation:

```bash
python -m optspread.cli.validate_generators --wave 1
```

Run Wave 1 behavioral validation on a trained checkpoint:

```bash
python -m optspread.cli.validate_behavior --wave 1 --agent-kind ppo --checkpoint runs/.../agent.pt
```

See `phases/PHASE4_GATE_REPORT.md`. Current status: foundation, GV_1, and BV_1
tooling pass; trained-agent Wave 1 behavioral validation and forgetting checks
are next.

## Phase-5–8 infrastructure

The non-data-dependent infrastructure for later phases is scaffolded and tested:

- Phase 5: `RealDataReplay`, walk-forward/purge/embargo, hygiene, baseline, and
  sim-to-real diagnostic helpers, plus a local WRDS discovery/export CLI.
- Phase 6: held-out GARCH-style generator, structural-distance, and graceful
  degradation helpers.
- Phase 7: rollout logging, coverage sampling, clustering, VIPER-style stump,
  fidelity, regime map, and economic-check helpers.
- Phase 8: final no-edge wrapper, ablation/sweep/cost/ensemble helpers,
  attribution fallback, exhibit manifest, and limitations.

See `phases/PHASE5_GATE_REPORT.md` through `phases/PHASE8_GATE_REPORT.md`.
Real-data gates, final-agent held-out evaluation, distillation, and the full
robustness battery remain pending because they require proprietary OptionMetrics
data and final trained checkpoint ensembles.

WRDS extraction is run from a normal local shell with WRDS network access:

```powershell
.venv\Scripts\python -m optspread.cli.wrds_optionmetrics discover --libraries optionm optionm_all optionmsamp_us --out data\wrds_optionmetrics_schema.json
```

After confirming table and column names in the discovery JSON, export a
loader-compatible SPX surface CSV:

```powershell
.venv\Scripts\python -m optspread.cli.wrds_optionmetrics export-surface --library optionm --surface-table-pattern "vsurfd{year}" --secid 108105 --start 1996-01-01 --end 2024-12-31 --out data\optionmetrics_spx_surface.csv --date-col date --maturity-col days --delta-col delta --iv-col impl_volatility --spot-library optionm --spot-table-pattern "secprd{year}" --spot-col close
```

The table/column arguments are intentionally configurable because WRDS
OptionMetrics layouts vary by entitlement.

## Quality gates

`make check` runs ruff (lint + format), `mypy --strict`, and `pytest --cov`.
Phase 2 is green with strict typing and the full test suite. No `# type: ignore` without a
one-line justification.
