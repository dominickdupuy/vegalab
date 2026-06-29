# Phase 5 Implementation Brief for Claude Code
## SPX Options Spread-Selection RL — Sim-to-Real Fine-Tuning & Walk-Forward on OptionMetrics

> **Scope of this phase.** Take the curriculum-trained agent from Phase 4 to **real OptionMetrics IvyDB US SPX data**: build the real-data adapter (a drop-in generator), evaluate **zero-shot** transfer, apply **light** fine-tuning, and run a rigorous **walk-forward** out-of-sample evaluation against a full baseline battery (buy-and-hold, always-on single structures, the CBOE benchmark indices, and a naive IV-rank/VRP heuristic). The deliverable is an honest out-of-sample verdict on **tail-adjusted** metrics plus an explicit **sim-to-real gap diagnosis**. The win condition is risk-adjusted, not raw mean return. If real performance collapses despite clean synthetic results, that is a finding to report, not to hide.
>
> **Depends on:** Phase 4 complete. The curriculum agent passed every wave's GV/BV/FF gates, the Wave-3 headline holds in-sim, and the env is surface-driven (so real data is a drop-in by construction).

---

## 0. Stack decisions (fixed)

- **Data:** OptionMetrics IvyDB US via WRDS. **Smoothed volatility-surface file** (standardized deltas 10–90, constant maturities 10d–2y) for the surface and features; **raw option-price file** (best_bid/best_offer, volume, OI, IV, greeks) for execution and cost calibration. **Primary instrument: SPX** (European, cash-settled; standard monthly third-Friday roll to match the CBOE indices). EOD only, daily-decision.
- **Real data is a drop-in generator.** Because Phase 4 generators emit a standardized IV surface, the real-data adapter implements the **same** `SurfaceGenerator` interface and the env/features/costs/agents/eval are **unchanged**. This is the entire sim-to-real bridge; do not fork the env.
- **Evaluation protocol:** walk-forward with **purge + embargo**; combinatorial purged cross-validation (CPCV) as the gold-standard option. **Deflated/probabilistic Sharpe** for significance under multiple testing.
- **Hyperparameters are frozen from the synthetic phase.** Real data is for light fine-tuning and evaluation only, **never** hyperparameter search. Any tuning against test folds is leakage.
- Everything else (PyTorch, the shared harness, mypy strict, ruff, causal features) carries over unchanged.

---

## 1. Design principles specific to this phase

1. **The real adapter is a generator, not a new env.** Build `RealDataReplay(SurfaceGenerator)` that reads (spot, IVSurface) per trading day from disk. The agent acts; the market evolves along the historical path regardless (valid market-replay assumption for a small participant whose orders do not move SPX). Verify by test that the env, agents, and evaluator are byte-for-byte the same code paths as synthetic.
2. **Zero-shot first, fine-tune second, report both.** The **zero-shot** result (Phase 4 agent, no fine-tuning, evaluated on real walk-forward) is the strongest evidence the curriculum taught economics rather than simulator artifacts. Fine-tuning then adapts to real microstructure. The pair (zero-shot vs fine-tuned) quantifies the sim-to-real gap and how much adaptation costs or helps.
3. **Light fine-tuning, or you erase the robustness you paid for.** Domain randomization across Phase 4 bought generalization. Aggressive fine-tuning on one historical path overfits it and throws that away. Use low learning rate, few steps, **early stopping on a validation fold**, optional lower-layer freezing. Conservative by default.
4. **Leakage is the enemy.** No hyperparameter tuning on test folds; purge and embargo around option-lifetime and episode boundaries; many folds, not one split; deflated Sharpe for selection bias; obs-normalization stats fit on **training** folds only. Treat every convenience as a potential leak until proven causal.
5. **Pre-commit the primary agent.** The Phase 4 **CVaR/IQN agent is the registered primary** before you look at real results. PPO and the risk-neutral distributional agent are secondary comparisons. Trying many and reporting the winner is selection bias.
6. **Risk-adjusted is the win condition.** The agent should beat the naive VRP heuristic and the always-on structures on **Sharpe, Sortino, CVaR/ES, max drawdown, tail ratio**, not necessarily on raw mean. Say this up front so a smaller mean with a much better tail reads as success, which it is.

---

## 2. Repository additions

```
optspread/
  data/
    optionmetrics_loader.py   # load IvyDB US (SPX secid): surface file + raw option file; align dates
    real_generator.py         # RealDataReplay(SurfaceGenerator): emits (spot, IVSurface) per day from disk
    hygiene.py                # filters: bid>0, no crossed/locked quotes, stale-price drop, IV present, liquidity
    calendar.py               # trading calendar; monthly/weekly expiry grid; roll alignment
  evaluation/
    walkforward.py            # WalkForwardSplitter: anchored/rolling folds + purge + embargo
    cpcv.py                   # combinatorial purged cross-validation (optional, gold standard)
    deflated_sharpe.py        # deflated & probabilistic Sharpe ratio (Lopez de Prado)
    significance.py           # block-bootstrap CIs on metric differences
  baselines/
    cboe_indices.py           # PUT, WPUT, BXM, BXMD, BXY, CNDR, BFLY replicated IN-ENV under one cost model
    vrp_heuristic.py          # naive: sell premium (credit structure) when VRP/IV-rank high, else flat
    buy_and_hold.py           # SPX total return
  finetune/
    finetuner.py              # conservative fine-tuning: low LR, early stop on val fold, optional freeze
    config.py
  eval/
    sim2real.py               # gap diagnostics: feature-distribution synthetic-vs-real, per-period, per-regime
  cli/
    finetune.py
    walkforward_eval.py
    sim2real_report.py
tests/
  test_optionmetrics_loader.py    # schema, SPX secid filter, surface-vs-raw IV reconciliation, date alignment
  test_real_generator_dropin.py   # RealDataReplay satisfies SurfaceGenerator; env/agents/eval code paths unchanged
  test_hygiene.py
  test_walkforward.py             # no train/test overlap; purge + embargo correct; episodes don't cross folds
  test_no_lookahead_real.py       # feature/normalizer at t uses only data <= t
  test_cboe_indices.py            # replicated index rules match published mechanics on sample dates
  test_deflated_sharpe.py
```

---

## 3. The real-data adapter and data hygiene

- **Loader:** pull SPX index options by secid. Use the **smoothed surface** for the standardized IV grid (features + the surface the env trades off of) and the **raw file** for actual quoted spreads (cost calibration) and tradeable contract prices. Reconcile surface-file IVs against raw-file IVs on sample dates (a Phase 0 check; re-confirm).
- **Instrument convention:** standard monthly SPX (third-Friday, AM-settled) as primary to align with the CBOE indices; document the roll (e.g. enter ~30–45 DTE, manage/close on a fixed rule). SPXW weeklys are an extension, not the primary.
- **Features:** VRP = SPX 30-day ATM IV (VIX-like, from the surface) minus trailing 21-day realized vol; IV rank/percentile over a trailing 1-year window; term-structure slope, 25-delta RR and fly, realized skew/kurt, multi-horizon momentum, jump proxy. **All trailing-window / point-in-time.**
- **Hygiene filters:** require `best_bid > 0`, drop crossed/locked quotes, drop stale prices, require IV present, apply a liquidity floor (volume or OI) for tradeable contracts. OptionMetrics includes delisted contracts, so survivorship is largely handled, but verify.
- **SPX simplification persists:** European/cash-settled means **no early-assignment or pin modeling**; daily EOD means **no intraday** management. State both as scope limits.

---

## 4. The baseline battery (replicate in-env under one cost model)

Replicate each baseline **inside the same env and cost model** so the comparison is apples-to-apples; optionally cross-check the CBOE ones against the published index series (noting their cost models differ).

- **Buy-and-hold SPX** (total return).
- **Always-on single structures:** each template held continuously (the env already supports this from Phase 1).
- **CBOE benchmark indices** (current methodology):
  - **PUT** — fully-collateralized ATM monthly SPX put-write; **WPUT** weekly variant.
  - **BXM** — covered-call, ATM monthly; **BXMD** ~30-delta OTM call; **BXY** 2% OTM call.
  - **CNDR** — short ~20-delta strangle + long ~5-delta wings, monthly, T-bill collateralized (cite the current methodology PDF; older 2015 docs used ~15-delta shorts).
  - **BFLY** — short ATM straddle + 5% OTM wings, T-bill collateralized.
- **Naive IV-rank/VRP heuristic** — the rules-based strawman: deploy a credit structure when VRP/IV-rank is high, else flat. The agent must beat this to justify its complexity.

---

## 5. Walk-forward, fine-tuning, and leakage controls

- **WalkForwardSplitter:** anchored (expanding) or rolling folds; train/fine-tune on past, test on the next out-of-sample window, roll forward. Aggregate OOS performance across **many folds**, never a single split.
- **Purge + embargo:** because options and positions have multi-day lifetimes, **align episode boundaries to fold boundaries** (positions do not span folds) and add an **embargo gap** after each test window so adjacent folds do not leak through overlapping market state.
- **CPCV (optional, gold standard):** generates many backtest paths with purging for a distribution of OOS outcomes rather than one number; use if time permits.
- **Fine-tuning protocol:** start from the Phase 4 checkpoint; low LR, few steps; for the off-policy distributional agent, seed the replay buffer with real-data rollouts; **early-stop on a validation fold disjoint from the test folds**; consider freezing the trunk and adapting only the head. Optional **block-bootstrap** of the real surface/path to create more fine-tuning episodes (note the assumption; keep zero-shot as the clean measure).
- **Normalization:** obs-normalizer running stats fit on **training folds only**, applied causally to test. Never use test-period statistics.
- **No hyperparameter search on test.** If fine-tuning hyperparameters must be chosen, choose them on validation folds via nested CV.

---

## 6. Validation gates (the definition of done)

1. **Data integrity.** Loader passes schema and hygiene; surface-vs-raw IV reconciliation within tolerance; `test_no_lookahead_real` green (features and normalizer at t use only data <= t).
2. **Drop-in confirmed.** `test_real_generator_dropin` shows the env, agents, and evaluator run on real data through the **same code paths** as synthetic, no env changes.
3. **Walk-forward correctness.** No train/test overlap; purge + embargo enforced; episodes do not cross folds; many folds.
4. **Baselines validated.** CBOE replications match published mechanics on sample dates; VRP heuristic and buy-and-hold implemented.
5. **Zero-shot reported.** The Phase 4 agent is evaluated on real walk-forward with no fine-tuning, as the primary sim-to-real generalization result.
6. **Fine-tuning is conservative and leak-free.** Early-stopped on validation folds; no test-fold tuning; zero-shot vs fine-tuned both reported.
7. **The headline gate.** Out-of-sample, the **CVaR agent beats the naive VRP heuristic and the always-on structures on tail-adjusted metrics** (Sharpe, Sortino, CVaR/ES, max drawdown, tail ratio), with **block-bootstrap CIs** on the differences and a **deflated Sharpe** that survives the multiple-testing correction, across folds and seeds with reported dispersion. Beating on mean is not required; beating on tail-adjusted is.
8. **Sim-to-real gap diagnosed.** Explicit comparison of synthetic-eval vs real-eval for the same agent, with attribution (distribution shift, cost realism, unseen regime, sample limitation).
9. **Cost robustness.** The verdict survives a cost stress test (e.g. 1.5–2x modeled quoted spread).

---

## 7. Sim-to-real gap diagnosis (a first-class deliverable, not an afterthought)

Build `eval/sim2real.py` to answer *why* real differs from synthetic:
- **Feature-distribution comparison:** are real feature values inside the Phase 4 domain-randomization priors, or out-of-distribution? Out-of-prior real regimes explain zero-shot failures and tell you to widen priors.
- **Per-period decomposition:** does failure cluster in specific historical windows (2008, 2018, 2020)? Few real tail events is a known limitation; do not over-claim tail skill from 2–3 crises.
- **Per-regime decomposition:** map real days to the nearest synthetic regime and compare performance.
- **Cost attribution:** re-run with synthetic-calibrated vs real-quoted spreads to isolate how much of the gap is microstructure.

**The honest pivot:** if real performance collapses despite clean synthetic results, reframe the thesis as a **rigorous sim-to-real gap study** (what transfers, what doesn't, and why). That is still a strong, publishable contribution. State this option plainly rather than torturing the agent into a real-data win.

---

## 8. Pitfalls to engineer against

- **Backtest overfitting / leakage** (the dominant risk): no test-fold hyperparameter search; purge + embargo; many folds; deflated Sharpe. (Bailey & Lopez de Prado.)
- **Over-fine-tuning** erases domain-randomization robustness: conservative LR, early stop, report zero-shot.
- **Rosy costs:** calibrate slippage to real quoted spreads, widen in stress, stress-test 1.5–2x.
- **Look-ahead** via features or normalization stats: causal, point-in-time, training-fold-only stats.
- **Multiple testing / selection bias:** pre-commit the CVaR agent as primary; others secondary.
- **Few real tail events:** the reason synthetic training existed; acknowledge, do not over-claim.
- **Single-path overfitting in fine-tuning:** there is one SPX history; treat zero-shot as the clean generalization measure and keep fine-tuning light.
- **Survivorship/data quality:** filter properly; verify delisted contracts are present.

---

## 9. Build order and process

**Build order (do not reorder):**
1. `data/optionmetrics_loader.py`, `hygiene.py`, `calendar.py` + `test_optionmetrics_loader.py`, `test_hygiene.py`.
2. `data/real_generator.py` (RealDataReplay) + `test_real_generator_dropin.py` (proves env/agents/eval unchanged) and `test_no_lookahead_real.py`.
3. `evaluation/walkforward.py` (+ optional `cpcv.py`), `deflated_sharpe.py`, `significance.py` + `test_walkforward.py`, `test_deflated_sharpe.py`.
4. `baselines/cboe_indices.py`, `vrp_heuristic.py`, `buy_and_hold.py` + `test_cboe_indices.py`.
5. **Zero-shot** walk-forward evaluation of the Phase 4 CVaR agent (and PPO, risk-neutral) vs the full battery.
6. `finetune/finetuner.py` + conservative fine-tuning on training folds, early stop on validation.
7. `cli/walkforward_eval.py`: fine-tuned walk-forward vs battery; bootstrap CIs, deflated Sharpe, dispersion.
8. `eval/sim2real.py` + `cli/sim2real_report.py`: gap diagnosis; cost stress test.

**Process instructions for Claude Code:**
- **Freeze synthetic-phase hyperparameters.** Real data is fine-tune + eval only. Never search hyperparameters on test folds.
- **Pre-commit the CVaR/IQN agent as primary** before inspecting real results; log PPO and risk-neutral as secondary.
- Keep all features and normalization causal and training-fold-only; assert no look-ahead.
- Reuse the shared env/eval/metric harness; `RealDataReplay` is the only new "generator."
- **Report zero-shot and fine-tuned both.** Lead the sim-to-real story with zero-shot.
- `make check` green (mypy strict, ruff) after each module; commit at module granularity referencing the gate that passed.
- **Do not start Phase 6 (held-out rough-vol generator) or Phase 7 (distillation).** If a task drifts there, stop and confirm.
- End with a report: data-integrity and drop-in confirmation, the zero-shot and fine-tuned walk-forward tables vs the full battery on tail-adjusted metrics with CIs and deflated Sharpe and per-seed/per-fold dispersion, the sim-to-real gap diagnosis with attribution, the cost stress result, and an explicit statement of whether the OOS gate passed (and if not, whether to pivot to the gap-study framing) and whether the project may proceed to Phase 6.

---

### Ready-to-paste kickoff prompt

> Implement Phase 5 of the project as described in `PHASE5_BRIEF.md`, building on the completed Phase 4 repo. Build `RealDataReplay` as a drop-in `SurfaceGenerator` that reads OptionMetrics IvyDB US SPX data (smoothed surface for features/surface, raw file for costs), and prove by test that the env, agents, and evaluator run on real data through the same code paths as synthetic with no env changes. Build the walk-forward splitter with purge and embargo (episodes aligned to fold boundaries), deflated Sharpe, and block-bootstrap CIs. Replicate the baseline battery in-env under one cost model: buy-and-hold, always-on single structures, the CBOE indices (PUT/WPUT/BXM/BXMD/BXY/CNDR/BFLY), and the naive IV-rank/VRP heuristic. First evaluate the Phase 4 CVaR agent zero-shot (no fine-tuning) on real walk-forward; then fine-tune conservatively (low LR, early stop on a validation fold disjoint from test, no test-fold hyperparameter search) and re-evaluate. Freeze all synthetic-phase hyperparameters; pre-commit the CVaR/IQN agent as the primary. The gate is that the CVaR agent beats the naive VRP heuristic and the always-on structures on tail-adjusted metrics (Sharpe, Sortino, CVaR/ES, max drawdown, tail ratio) out-of-sample, with bootstrap CIs and a deflated Sharpe surviving multiple-testing correction, across folds and seeds. Keep all features causal and normalization training-fold-only. Produce an explicit sim-to-real gap diagnosis and a cost stress test, report zero-shot and fine-tuned both, and if real collapses despite clean synthetic results, surface the gap-study pivot honestly rather than overfitting. Keep mypy strict and ruff green, and end by stating whether the OOS gate passed and whether the project may proceed to Phase 6.
