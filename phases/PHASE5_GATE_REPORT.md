# Phase 5 Gate Report

Generated: 2026-06-28T23:05:00

Overall implementation status: **INFRASTRUCTURE PASS / REAL DATA PENDING**

## Implemented

- OptionMetrics-style CSV surface loader with standardized `IVSurface` output.
- WRDS connection helpers and schema-flexible OptionMetrics discovery/export CLI
  that writes the same CSV format consumed by `load_surface_csv`.
- `RealDataReplay`, a drop-in generator that replays historical surface/path rows
  through the existing environment and shared evaluator.
- Quote hygiene filters, business-day and monthly-expiry helpers.
- Walk-forward splitter with purge and embargo.
- Probabilistic/deflated Sharpe approximations and paired bootstrap CI helper.
- Baseline scaffolds: buy-and-hold, CBOE benchmark action mappings, naive
  VRP/IV-rank heuristic.
- Conservative fine-tuning plan guard requiring a validation fold.
- Sim-to-real feature-coverage diagnostic.

## Real data extracted (2026-06-29)

SPX OptionMetrics surface extracted locally from WRDS to
`data/optionmetrics_spx_surface.csv`:

- **7,446 trading days, 1996-01-04 → 2025-08-29**, spot 598.5 → 6501.9
  (matches SPX history), standardized to the project delta/maturity grid.
- Source tables: `optionm.vsurfd{year}` (surface) and `optionm.secprd{year}`
  (spot), SPX `secid=108105`. Calls only (`cp_flag='C'`) — calls and puts at the
  same |delta| are opposite strike wings and must not be averaged.
- Skew validated as economically sane: delta-0.90 (low-strike) IV ≫ delta-0.10
  (high-strike) IV across the curve.
- 17 days dropped where OptionMetrics has NULL `impl_volatility` at source
  (a late-July/Aug-2020 vendor gap); IVs are never fabricated. March-2020 tail
  event is fully present.

Two extraction bugs were fixed to make this work:

1. `wrds.Connection.raw_sql` is unusable on the installed sqlalchemy/pandas
   stack; queries now run via `db.engine` + `sqlalchemy.text()` (`_read_sql`).
2. The standardizer previously averaged call and put rows at the same |delta|,
   blending opposite strike wings; `SurfaceQueryConfig` now filters to one
   `cp_flag`.

## Gates

- Data integrity: **PASS** — 7,446 clean SPX surface days extracted; skew,
  spot, and trading-day counts validated.
- WRDS access: **LOCAL EXTRACTION COMPLETE** — extracted from a local shell into
  `data/` (the Codex sandbox still cannot open TCP 9737).
- Drop-in generator path: **PASS on REAL data** — a frozen Phase-3 IQN/CVaR
  Wave-0 checkpoint runs zero-shot through `EnvFactory`/`Evaluator` on the real
  surface with matching `obs_dim=16` (the surface keystone holds). The Wave-0
  agent correctly stays 100% FLAT (no learned edge yet ⇒ no hallucinated edge on
  real data, no leak).
- Walk-forward correctness: **PASS** — purge and embargo behavior is tested.
- Baselines: **SCAFFOLDED** — mechanics are represented; published-index
  reconciliation requires Cboe/OptionMetrics sample data.
- Zero-shot/fine-tuned real OOS: **PIPELINE PROVEN / HEADLINE PENDING** — the
  real-data evaluation path runs end-to-end; the headline tail-adjusted OOS
  comparison still requires the Phase-4 curriculum checkpoint (Waves 1–6), which
  has not been trained.
- Sim-to-real gap and cost stress: **PENDING** — requires the curriculum
  checkpoint and real evaluation output.

## Local WRDS export entry point

Run discovery from a normal PowerShell terminal with WRDS network access:

```bash
python -m optspread.cli.wrds_optionmetrics discover --libraries optionm optionm_all optionmsamp_us --out data/wrds_optionmetrics_schema.json
```

Then export SPX standardized surfaces after confirming table/column names:

```bash
python -m optspread.cli.wrds_optionmetrics export-surface --library optionm --surface-table-pattern "vsurfd{year}" --secid 108105 --start 1996-01-01 --end 2024-12-31 --out data/optionmetrics_spx_surface.csv --date-col date --maturity-col days --delta-col delta --iv-col impl_volatility --spot-library optionm --spot-table-pattern "secprd{year}" --spot-col close
```

The table and column names above are configurable because WRDS OptionMetrics
layouts vary by entitlement; use discovery output as the source of truth.

## Validation

Automated tests:

- `tests/test_phase5_data_eval.py`
- `tests/test_phase5_baselines_sim2real.py`

Current result: infrastructure tests pass under full repo quality gates.
Latest full-suite result: **166 passed**, strict mypy green, ruff green.
