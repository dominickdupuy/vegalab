# optspread — Empirical Audit: Bugs & Design-vs-Reality Drift

This audit is executed evidence: every claim below was produced by running code against the repo (`optspread`), not by reading it. Method was adversarial — a first pass proposed findings, a second pass actively tried to refute each one (alternate implementations, control cases, repeated seeds); findings that didn't survive are recorded in Section E rather than discarded. Environment caveat: Python 3.14, numpy 2.2.6, torch 2.9.0 were used, which drifts from the repo's pinned `torch>=2.6,<2.7` / py3.11–3.12 target — but the reward, statistics, interpret, GARCH-surface, and CBOE findings are pure-numpy/pydantic and not version-sensitive, and the GV_2 flakiness (BUG-04) was separately confirmed to be pure-NumPy/PCG64 behavior, not version drift.

## Global reproduction notes

- Run all commands from repo root `C:\Users\hackathon\Code\vegalab` with `PYTHONPATH` set to that root (so `import optspread` works).
- Environment used: Python 3.14, numpy 2.2.6, torch 2.9.0. The repo pins `torch>=2.6,<2.7` and targets py3.11/3.12 — torch/python are drifted from pins here. This does not affect the pure-numpy/pydantic findings (reward, statistics, interpret, GARCH surface, CBOE baseline); the GV_2 flakiness (BUG-04) was independently confirmed to be a pure-NumPy/PCG64 effect, not version drift.
- Repo tests must be run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` prefixed — a broken global `seleniumbase` pytest plugin otherwise crashes collection with a `--browser` conflict.
- Baseline: 203/204 of the repo's own tests pass on this box; the single failure is BUG-04.
- Every item is labeled synthetic vs real where relevant. This is an SPX (European, cash-settled, EOD) research repo — synthetic curriculum generators (Heston, GARCH) vs real OptionMetrics/CBOE-derived components are distinct evidentiary categories.

## Summary table

| ID | Title | Severity | Verdict | Key number |
|---|---|---|---|---|
| BUG-01 | DifferentialSharpe returns eta×D_t, not D_t | Tier 0 — defect | CONFIRMED | ratio = 0.010001 ≈ eta (std 1.3e-4), reproduced 2x |
| BUG-02 | Phase-6 GARCH generator emits a flat IV surface | Tier 0 — defect | CONFIRMED | cross-sectional std = 0.000e+00 vs Heston 0.024/0.040 |
| BUG-03 | "Deflated Sharpe" ignores skew/kurtosis and trial dispersion | Tier 0 — defect | CONFIRMED | PSR diff 0.00e+00 across skew ±3.6/kurt ±11; DSR identical 1.0000 across variance 0.0002 vs 7.01 |
| BUG-04 | Wave-2 "GV_2 PASS" typically fails (flaky) | Tier 0 — defect | CONFIRMED | 4/14 seeds pass (28.6%); atm_change_var_ratio 0.549±0.124 |
| BUG-05 | shap_analysis.py has no SHAP, only permutation importance | Tier 1 — stub | CONFIRMED | `import shap` → ModuleNotFoundError; gate report self-admits "FULL SHAP PENDING" |
| BUG-06 | VIPER distillation is one decision stump, not a tree | Tier 1 — stub | CONFIRMED | conjunctive-rule train acc 0.77 vs majority baseline 0.7525 |
| BUG-07 | CBOE PUT baseline maps to wrong payoff, never run | Tier 1 — stub | CONFIRMED | mapped loss caps at -227.85 vs real short-put -2395.19 at spot 2500 |
| BUG-08 | economic_checks is a lexical "and"/"except" string search | Tier 1 — stub | CONFIRMED | reckless rule passes (has "and"); sound rule fails (lacks "and") |
| BUG-09 | Ablations/sweeps are arithmetic on hand-supplied numbers; capacity_sweep missing | Tier 1 — stub | CONFIRMED | drop_reward_term does zero training; `capacity_sweep` → ImportError |
| BUG-10 | "block bootstrap" is plain i.i.d. resampling | Tier 1 — stub | CONFIRMED | shuffle-invariance ratio 0.9834 vs true block-bootstrap 0.2789 |
| BUG-11 | rough Bergomi / SABR / Bates generators don't exist | Tier 2 — missing | CONFIRMED (Phase 6 disclosed pending) | 3x ModuleNotFoundError/AttributeError |
| BUG-12 | Per-regime return distributions from critic don't exist | Tier 2 — missing | CONFIRMED | RegimeCell has scalar mean_return only, no quantiles/CVaR |
| BUG-13 | Walk-forward is expanding-only; CPCV missing | Tier 2 — missing (CPCV named optional in brief) | CONFIRMED | all 7 folds have train_start=0; `cpcv` → ImportError |
| BUG-14 | Curriculum Waves 3-6 have no generators, hard-capped at Wave 2 | Tier 2 — missing (disclosed in gate report) | CONFIRMED | factory raises ValueError outside {0,1,2}; CLI choices=(1,2) |
| BUG-15 | CurriculumRunner.validate_only hardcodes False | Tier 3 — orphaned | CONFIRMED | PromotionDecision always False; reasons are hardcoded strings |
| BUG-16 | FrameStackObservation wired into nothing | Tier 3 — orphaned | CONFIRMED | only definition + own test; no EnvFactory/CLI/config use |
| BUG-17 | In-vitro CVaR gyms never trained; tests use closed-form helpers only | Tier 3 — orphaned | CONFIRMED | tests import greedy_arm/greedy_mdp_action only; envs have no external callers |
| BUG-18 | 3 of 5 reward components dead in every real training run | Tier 3 — orphaned | CONFIRMED | margin_normalized/sharpe/sortino weight = 0.0 at every construction site |
| BUG-19 | "tail ratio" metric named nowhere computed | Tier 3 — phantom | CONFIRMED | `grep tail_ratio optspread/` → 0 matches |
| BUG-20 | reporting/exhibits.py is inert placeholder metadata | Tier 3 — orphaned | CONFIRMED | Exhibit has no artifact/path field; no runs/reports/exhibits dirs exist |
| NOTE-A | Nested-CVaR bootstrap is not dead code | Correction | REFUTED (of "dead code" claim) | trainer runs 4 gradient cycles with bootstrap_risk="cvar", no errors |
| NOTE-B | IQN CVaR tau sampling is genuinely correct | Verified-correct | CONFIRMED | 32,000 taus ≤ alpha, KS p=0.95 vs Uniform(0,alpha) |
| NOTE-C | Core numeric spine (pricing, margin, COS pricer, PPO, harness) is sound | Verified-correct | CONFIRMED | 203/204 tests pass; COS pricer matches published Fang-Oosterlee benchmark 5.785 |

---

## A. Tier 0 — Defects (wrong while looking right)

### BUG-01: DifferentialSharpe returns eta×D_t instead of D_t (silent ~100x under-scaling)
- **Severity / type**: Tier 0 — defect
- **Location**: `optspread/reward/components.py`, `DifferentialSharpe.update()` (~lines 112-126)
- **Designed to**: Emit the Moody-Saffell differential Sharpe ratio `D_t` each step so that, over a stationary return stream, it converges to (tracks) the batch Sharpe ratio, per CLAUDE.md's Markovian risk-reward requirement.
- **Actually does**: Bakes the EMA learning rate `eta` into the numerator terms used for `D_t`, so the emitted value is `eta * D_t`, not `D_t`.
- **Executed evidence**: Fed a stationary i.i.d. N(0.05, 1) stream (4000–5000 draws) through the real component vs a from-scratch Moody-Saffell reference implementation. Steady-state ratio (code / reference) = **0.010001**, matching `eta` exactly, with std **1.3e-4**. Independently reproduced twice: **0.010000 ± 1.5e-17** and **0.010001**. Source: `dA = self._eta * (R - self._A)` and `dB = self._eta * (R*R - self._B)` bake eta into `dA`/`dB`; those eta-scaled values then feed the `d_sharpe` numerator `(B*dA - 0.5*A*dB) / var**1.5` (should use the *unscaled* `R-A`, `R^2-B` there), while separately being correctly used to advance the EMA state.
- **Reproduce**: Instantiate `DifferentialSharpe()` from `optspread/reward/components.py`, feed a seeded `numpy.random.Generator` stream of ~4000-5000 draws from N(0.05, 1) through `.update(R)`, compare steady-state output against a hand-written reference implementing `D_t = (B*(R-A) - 0.5*A*(R^2-B)) / (B - A^2)**1.5` with the *same* unscaled increments feeding the EMA. Take the ratio of the two steady-state series.
- **Root cause**: The eta-scaled `dA`/`dB` deltas (correctly used to update the EMA state `A`, `B`) are reused directly in the `d_sharpe` numerator, where the derivation calls for the unscaled increments `(R - A)` and `(R^2 - B)`.
- **Suggested fix**: Compute `dA_raw = R - self._A` and `dB_raw = R*R - self._B` separately; use `dA_raw`/`dB_raw` in the `d_sharpe` numerator, and apply `eta * dA_raw` / `eta * dB_raw` only when advancing `self._A`/`self._B`.
- **Catcher test**: Over a long stationary i.i.d. stream, the cumulative/steady-state `DifferentialSharpe` output should track the batch Sharpe ratio of that stream within tolerance; currently it is low by a factor of `eta` (~100x for eta=0.01). Impact: whenever `sharpe_weight != 0`, the differential-Sharpe reward term is ~2 orders of magnitude weaker than intended. This is exactly the acceptance test the Phase-1 brief calls for ("diff-Sharpe EMA converges to batch Sharpe of a stationary i.i.d. stream within tolerance") but it was never written, which is why this shipped.

### BUG-02: Phase-6 GARCH "held-out generator" emits a flat IV surface (cannot be out-of-distribution) — SYNTHETIC
- **Severity / type**: Tier 0 — defect
- **Location**: `optspread/market/garch.py:63` (`GARCHGenerator._snapshot` calls `IVSurface.flat`); `optspread/market/surface.py:64` (`flat = np.full((n_maturities, n_deltas), sigma)`)
- **Designed to**: Serve as a structurally out-of-distribution held-out generator (Phase 6) with GJR leverage asymmetry and Duan risk-neutralization producing a real skew/term-structure surface, per CLAUDE.md's surface-keystone invariant ("every generator emits a standardized IV surface").
- **Actually does**: Emits a perfectly flat surface — the same scalar `ann_vol` value broadcast into every (delta, maturity) cell.
- **Executed evidence**: Instantiated a real `GARCHGenerator`, reset and stepped it, and measured cross-sectional dispersion of the surface. Per-day std **across deltas = 0.000e+00** and **across maturities = 0.000e+00** (machine precision — every cell identical). Heston control on the same harness: **0.024 (deltas) / 0.040 (maturities)**. Independently reproduced (GARCH 0.000e+00 vs Heston 2.4e-2/4.0e-2 both times). The only nonzero GARCH variation observed is day-to-day drift of the scalar `ann_vol`, not any cross-sectional shape.
- **Reproduce**: Instantiate `GARCHGenerator` (as configured in `optspread/market/garch.py`), call `.reset()` then `.step()` for several simulated days, read the resulting `IVSurface` grid each day, and compute `np.std` across the delta axis and across the maturity axis. Compare against the same measurement on a `HestonGenerator` instance.
- **Root cause**: `_snapshot` calls `IVSurface.flat(sigma)` — a helper that broadcasts a single scalar volatility into the full grid — instead of pricing the surface through a model that produces skew and term structure.
- **Suggested fix**: Price the GARCH surface through a real, skew-producing pricer implementing the GJR leverage asymmetry and Duan risk-neutralization the brief specifies, rather than flattening to a scalar.
- **Catcher test**: `GARCHGenerator` surface must have nonzero cross-sectional std across deltas AND across maturities on every simulated day. Impact: a flat surface has no skew, term structure, or smile, so it cannot be structurally OOD versus any generator that prices a real surface — it fails the CLAUDE.md hard-stop condition ("a held-out family that is not actually out-of-distribution is not a valid test — fix or drop it"), which should have halted Phase 6 on this generator.

### BUG-03: "Deflated Sharpe" is not the Bailey/Lopez de Prado deflated Sharpe
- **Severity / type**: Tier 0 — defect
- **Location**: `optspread/evaluation/deflated_sharpe.py:20` (`probabilistic_sharpe_ratio`), `:34` (`deflated_sharpe_ratio`)
- **Designed to**: Compute the Probabilistic Sharpe Ratio (denominator adjusted for return skew/kurtosis) and the Deflated Sharpe Ratio (benchmark adjusted for the expected maximum Sharpe under the null across the actual number/dispersion of trials), per PHASE5_BRIEF's requirement for deflated/probabilistic Sharpe significance testing.
- **Actually does**: PSR ignores skew and kurtosis entirely (bit-identical output regardless of higher moments). DSR ignores the actual dispersion of trial Sharpes — its signature doesn't even accept a trial-Sharpe array — and uses a hardcoded constant in place of a derived null-benchmark term.
- **Executed evidence**: (a) PSR: two 500-obs series with identical Sharpe (0.0905) but skew −3.62/kurtosis 11.14 vs skew +0.09/kurtosis −0.05 → repo PSR bit-identical, **diff 0.00e+00**; a correct PSR with the skew/kurtosis denominator differs by **1.98e-2** (denominator moves 0.9957 → 1.1620) between the two series. (b) DSR: signature is `(observed_sharpe, *, n_returns, n_trials, benchmark_sharpe=0.0)` — no trial-Sharpe array parameter; fed 20 trials of variance 0.0002 vs variance 7.01 → repo DSR identical at **1.0000** in both cases; a correct expected-max-under-null benchmark differs **201x** (0.025 vs 5.034) between the two variance regimes. `trial_penalty = sqrt(2) * 0.1 * sqrt(n_trials - 1)` — the `0.1` is a hardcoded constant with no derivation shown.
- **Reproduce**: Call `probabilistic_sharpe_ratio` on two 500-observation synthetic return series with matched Sharpe ratio but different skew/kurtosis (e.g. one heavily left-skewed/fat-tailed, one near-Gaussian) and diff the outputs. Call `deflated_sharpe_ratio(observed_sharpe, n_returns=..., n_trials=20, ...)` twice, once backing it by 20 trial-Sharpes with variance 0.0002 and once with variance 7.01, and diff the outputs — note the function signature has no parameter to even pass the trial-Sharpe array in.
- **Root cause**: PSR formula omits the skew/kurtosis correction term in the standard-error denominator; DSR substitutes a fixed constant (`0.1`) for the variance-of-trial-Sharpes term instead of computing it from actual trial outcomes.
- **Suggested fix**: Implement the real PSR (skew/kurtosis-adjusted denominator per Bailey & Lopez de Prado) and real DSR (benchmark Sharpe = expected max under the null, computed from the variance across the actual trial Sharpes, not a hardcoded constant).
- **Catcher test**: DSR output must change when trial-Sharpe dispersion changes at fixed observed Sharpe and n_trials; PSR output must change when return skew/kurtosis changes at fixed Sharpe. Both currently do not change. Impact: any "significant after multiple-testing correction" or "deflated Sharpe survives" claim in this repo rests on a statistic blind to higher moments and to trial dispersion — the exact quantities these corrections exist to capture. Applies to both SYNTHETIC and REAL evaluation (this stats layer is used in Phase 5).

### BUG-04: Wave-2 generator-validation "GV_2 PASS" typically FAILS (flaky, under-calibrated) — SYNTHETIC
- **Severity / type**: Tier 0 — defect
- **Location**: `tests/test_heston_generator.py::test_gv2_heston_passes_default_priors`; `optspread/eval/generator_validation.py:52-129` (`validate_wave2_heston`)
- **Designed to**: Confirm (per CLAUDE.md's "generator validation before agent training" invariant) that the Heston Wave-2 generator, under its default calibration priors, robustly produces the intended mean-reversion stylized fact before any agent is trained on it — this is the checked-in "GV_2 PASS" referenced in `PHASE4_GATE_REPORT.md`.
- **Actually does**: Passes only on a minority of seeds; the `atm_change_var_ratio` mean-reversion sub-check is marginal and frequently exceeds the gate threshold.
- **Executed evidence**: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_heston_generator.py -q` **FAILS** on this box with `atm_change_var_ratio=0.510` vs the `<0.500` gate (the sole failing sub-check; skew/VRP/IV-rank/term-slope/ACF1 all pass). A 14-seed sweep (same construction: `HestonGenerator.randomized(GBMConfig(n_days=20))`, episodes=6): **PASS rate 28.6% (4/14)**, `atm_change_var_ratio` mean=**0.549**, std=**0.124**, range **0.327–0.773**. Confirmed this is a pure NumPy/PCG64 code path — not numpy/torch version drift — so it reflects a genuinely marginal/under-calibrated mean-reversion check, not an environment artifact.
- **Reproduce**: `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_heston_generator.py -q` (single seed as checked in). For the sweep: loop 14+ seeds through `HestonGenerator.randomized(GBMConfig(n_days=20))`, run `validate_wave2_heston` with `episodes=6` each time, and tabulate the pass rate and `atm_change_var_ratio` distribution.
- **Root cause**: The Heston mean-reversion prior (kappa/theta) does not robustly produce `atm_change_var_ratio < 0.5`; the checked-in gate report reflects a single favorable seed rather than a robust property of the default priors.
- **Suggested fix**: Recalibrate the Heston mean-reversion prior so `atm_change_var_ratio` is robustly below the gate across seeds, or revisit/justify the 0.500 threshold with a documented rationale.
- **Catcher test**: `validate_wave2_heston` must pass across a seed sweep (e.g. majority of N ≥ 14 seeds), not a single hand-picked seed. Impact: the checked-in "GV_2 PASS" per `PHASE4_GATE_REPORT.md` is more accurately described as "typically FAILS" (28.6% pass rate observed).

---

## B. Tier 1 — Stubs mislabeled as real components

### BUG-05: shap_analysis.py contains no SHAP (only permutation importance)
- **Severity / type**: Tier 1 — stub
- **Location**: `optspread/attribution/shap_analysis.py`
- **Designed to**: Provide SHAP (Shapley-value) feature attribution for Phase 8's robustness/attribution battery.
- **Actually does**: Provides one function, `permutation_importance`, which shuffles a column and measures score degradation — no coalition/Shapley computation, no additivity property.
- **Executed evidence**: `import shap` → `ModuleNotFoundError` (not a dependency; not listed in `pyproject.toml`). Ran `permutation_importance` on a known linear scoring function → scores **[0.317, 0.0, −0.397]**, correctly zeroing the irrelevant column — textbook shuffle-column importance, not Shapley values. `PHASE8_GATE_REPORT.md` self-admits: "PERMUTATION HELPER PASS / FULL SHAP PENDING."
- **Reproduce**: `python -c "import shap"` from the repo's environment (fails). Then call `permutation_importance` from `optspread/attribution/shap_analysis.py` on a hand-built linear function `f(x) = a*x0 + b*x1 + 0*x2` and confirm column 2's importance is ~0 while the shuffle-based mechanism (not Shapley) drives the result.
- **Root cause**: SHAP was never implemented or added as a dependency; permutation importance was substituted under the same module name.
- **Suggested fix**: Either implement real SHAP (add the `shap` dependency, wire up a KernelSHAP/TreeSHAP explainer) or rename the module and drop the "SHAP" claim from reporting.
- **Catcher test**: If SHAP is claimed anywhere in reporting, a Shapley-additivity property test (sum of attributions = f(x) − E[f]) must hold; currently there is no such property to test since no Shapley computation exists.

### BUG-06: VIPER distillation is a single decision stump, not a depth≤6 tree; cannot express a regime→structure map
- **Severity / type**: Tier 1 — stub
- **Location**: `optspread/interpret/viper.py` (`fit_weighted_stump` / `DecisionStump`)
- **Designed to**: Distill the trained policy into a depth-bounded (≤6) decision tree via VIPER-style DAGGER resampling, producing the investor-facing regime→structure map (Phase 7).
- **Actually does**: Fits a single-split decision stump (one feature, one threshold, two leaf actions) — no depth parameter, no recursion/children, no DAGGER/resampling loop.
- **Executed evidence**: Fit on a conjunctive XOR-like dataset (label=1 iff `x0>0.5 AND x1>0.5`, n=400) → train accuracy **0.77**, barely above the majority-class baseline **0.7525**. A single-feature-separable control dataset → accuracy **1.0** (stump handles that trivially, confirming the harness itself is correct). `DecisionStump` dataclass has only `feature`/`threshold`/`left_action`/`right_action` fields — no depth param, no recursion/children, no DAGGER/resampling loop in the module.
- **Reproduce**: Build a synthetic dataset where the target rule requires two features conjunctively (e.g. `label = (x0 > 0.5) & (x1 > 0.5)`), fit `fit_weighted_stump` on it, and measure training accuracy against the majority-class baseline. Compare against a single-feature-separable dataset as a control.
- **Root cause**: The module implements only the single-split primitive; the recursive tree-growing and DAGGER trajectory-resampling loop the brief specifies were never built on top of it.
- **Suggested fix**: Implement a depth-bounded (≤6) tree with recursive splitting plus a DAGGER-style resampling loop, as the Phase 7 brief specifies.
- **Catcher test**: The distilled model must reach high accuracy on a conjunctive-rule fixture (a single stump structurally cannot, regardless of tuning). Impact: the phase's investor-facing deliverable — a conditional regime→structure map — cannot be represented by a one-split stump whenever the true decision boundary is conjunctive/multi-feature, which is the expected case for a regime map.

### BUG-07: CBOE PUT baseline maps to a structurally wrong payoff and is never run — REAL baseline
- **Severity / type**: Tier 1 — stub
- **Location**: `optspread/baselines/cboe_indices.py` (`CBOE_BENCHMARKS`: `PUT` → `action_id 5` `bull_put_spread@0.16`; `BXM` → `bear_call_spread`; `CNDR` → `iron_condor`; `BFLY` → `iron_butterfly`)
- **Designed to**: Provide the CBOE PUT-Write Index as a real reference strategy the trained agent must beat, as the brief's headline "beats the CBOE indices" comparison.
- **Actually does**: Maps `PUT` to a defined-risk bull put spread (a vertical), not the actual CBOE PUT index mechanic (a cash-secured short ATM put), and the mapping is never invoked anywhere in the real evaluation pipeline.
- **Executed evidence**: Built the mapped `bull_put_spread` template vs a real cash-secured short-ATM put on the same chain (spot 5000, 20% flat IV); payoffs swept across spot 2500→6000: mapped bull_put_spread loss **caps at −227.85** for all spots below 4500, while the real short ATM put loss grows linearly to **−2395.19** at spot 2500 — over **10x worse** and unbounded vs capped. `evaluate_real.py` `POLICY_CHOICES = ('ppo', 'qrdqn', 'iqn', 'flat', 'vrp_heuristic')` — no CBOE reference option exists; `CBOE_BENCHMARKS` is imported nowhere in the eval battery. Also missing index mappings for WPUT/BXMD/BXY.
- **Reproduce**: Construct the `bull_put_spread` template that `CBOE_BENCHMARKS['PUT']` points to (action_id 5, strike offset 0.16) on a chain with spot=5000 and flat 20% IV. Separately construct a cash-secured short ATM put on the same chain. Sweep terminal spot from 2500 to 6000 and compare P&L curves. Then `grep -rn CBOE_BENCHMARKS optspread/eval optspread/baselines` and inspect `evaluate_real.py`'s `POLICY_CHOICES` to confirm it is never wired in.
- **Root cause**: The CBOE index name was mapped to a similarly-directional but structurally different (defined-risk, capped-loss) template instead of replicating the index's actual cash-secured/covered mechanics; the mapping was also never connected to the evaluation CLI.
- **Suggested fix**: Replicate the actual index mechanics (cash-secured short put for PUT, covered call for BXM, etc.) and wire `CBOE_BENCHMARKS` into the real evaluation battery (`evaluate_real.py`'s `POLICY_CHOICES`).
- **Catcher test**: The PUT-baseline payoff must match a cash-secured short-put shape (unbounded linear downside below the strike), not a capped vertical-spread shape. Impact: the brief's headline "beats the CBOE indices" comparison currently uses a defined-risk vertical that is not the index it claims to be, and isn't wired into the real evaluation pipeline regardless.

### BUG-08: economic_checks.check_nontrivial_interaction is a lexical "and"/"except" string search, not an economic check
- **Severity / type**: Tier 1 — stub
- **Location**: `optspread/interpret/economic_checks.py:14-20`
- **Designed to**: Verify distilled regime→structure rules exhibit genuine economic interaction logic (e.g., conditioning correctly on VRP/jump-risk direction), part of Phase 7's "economic-sensibility" gate.
- **Actually does**: Checks only whether the rule's text contains the literal substring "and" or "except" — a grammar check, not an economic-directionality check.
- **Executed evidence**: A reckless rule, "jump risk high and skew flat → sell naked premium," → `passed=True` (only because it contains "and"). A sound rule, "elevated IV rank → sell iron condor" (no "and"/"except"), → `passed=False`. Source: `any(("except" in r.lower()) or ("and" in r.lower()) for r in rules)`.
- **Reproduce**: Call `check_nontrivial_interaction` (or equivalent function in `optspread/interpret/economic_checks.py:14-20`) with a list containing the reckless rule string above and observe `passed=True`; call it with the sound rule string above and observe `passed=False`.
- **Root cause**: The check was implemented as a substring search on the rule's natural-language text rather than as an evaluation of the rule's actual economic directionality (e.g., does it sell premium when VRP is high, does it de-risk when jump risk is high).
- **Suggested fix**: Replace the string search with a check of actual economic directionality — e.g. assert VRP-high implies sell-premium action, jump-risk-high implies defensive/reduced-size action — evaluated against the rule's structured condition/action fields, not its prose.
- **Catcher test**: An economically-backwards rule (e.g. one that increases naked exposure precisely when jump risk is high) must FAIL the check regardless of whether its text happens to contain the word "and." Impact: this Phase 7 economic-sensibility gate currently cannot catch reward-hacked or economically nonsensical distilled rules, which is exactly the CLAUDE.md hard-stop condition it exists to guard against.

### BUG-09: robustness/ablations & sweeps are arithmetic on hand-supplied numbers (no retrain/agent/env); capacity_sweep missing
- **Severity / type**: Tier 1 — stub
- **Location**: `optspread/robustness/ablations.py` (`drop_reward_term`), `alpha_sweep.py`, `cost_sensitivity.py`, `seed_ensemble.py`
- **Designed to**: Run the Phase 8 "headline" ablation battery — retrain (or re-evaluate) the agent under each reward-term-dropped / alpha / cost / seed variant and report the resulting tail metrics.
- **Actually does**: `drop_reward_term` returns a `RewardConfig` copy with the target weight (e.g. `cvar_weight`) set to 0.0 and performs no training or evaluation at all. `alpha_sweep`/`cost_sensitivity`/`seed_ensemble` run as pure arithmetic over hand-supplied lists, with no agent, env, or rollout involved. `capacity_sweep` does not exist as a module.
- **Executed evidence**: `drop_reward_term(cvar)` returns a `RewardConfig` copy with `cvar_weight` **0.4 → 0.0** and does **no training**. `alpha_sweep`/`cost_sensitivity`/`seed_ensemble` ran as pure math over lists, e.g. `break_even_cost_multiple([1, 1.5, 2, 3], [0.02, 0.01, -0.001, -0.02]) = 2.0`; `summarize([...])` → mean **0.508** / std **0.0277**. `from optspread.robustness import capacity_sweep` → **ImportError** (file absent).
- **Reproduce**: Call `drop_reward_term(cvar_weight=0.4)` (or equivalent) from `optspread/robustness/ablations.py` and confirm it returns only a config object with no side effects (no env rollout, no agent forward pass). Call `break_even_cost_multiple` and `summarize` from `cost_sensitivity.py`/`seed_ensemble.py` and confirm they operate on plain Python lists you supply, not on outputs of an actual run. Try `from optspread.robustness import capacity_sweep`.
- **Root cause**: The retrain-and-compare orchestration layer (spin up env + agent per variant, roll out, compute tail metrics) that the brief calls "the headline" was never built; only the config-construction and pure-arithmetic summary helpers exist.
- **Suggested fix**: Build the retrain+evaluate+compare orchestration that actually instantiates each variant's env/agent, runs rollouts, and reports the resulting tail metrics; add the missing `capacity_sweep` module.
- **Catcher test**: The ablation battery must actually produce per-variant tail metrics derived from real rollouts (currently it returns config objects or pure arithmetic on caller-supplied numbers, with no environment or agent invoked).

### BUG-10: significance.py "block bootstrap" is plain i.i.d. resampling (destroys serial correlation)
- **Severity / type**: Tier 1 — stub
- **Location**: `optspread/evaluation/significance.py:21-23` (`idx = rng.integers(0, a.size, size=(n_boot, a.size)); diffs = (a[idx] - b[idx]).mean(axis=1)`)
- **Designed to**: Provide block-bootstrap confidence intervals that preserve serial correlation structure in return series, as PHASE5_BRIEF explicitly requests ("block-bootstrap CIs").
- **Actually does**: Draws each resampled index i.i.d. uniformly at random (no blocks), which is standard (non-block) bootstrap and is invariant to the original series' time-ordering.
- **Executed evidence**: On an AR(1) φ=0.9 series, the repo's implementation gives CI width **original 0.30400 vs shuffled 0.29897** → ratio **0.9834** (essentially order-invariant, as expected for i.i.d. resampling). A true circular block bootstrap (block=20) on the same series gives **original 1.05088 vs shuffled 0.29312** → ratio **0.2789** (correctly collapses once autocorrelation is destroyed by shuffling, since only the true block bootstrap can detect and exploit the serial correlation in the unshuffled series).
- **Reproduce**: Generate an AR(1) series with φ=0.9. Compute the repo's bootstrap CI width on it and on a randomly shuffled copy of it; take the ratio. Separately implement a circular/stationary block bootstrap with block length 20, compute CI widths the same way, and take that ratio. Compare the two ratios.
- **Root cause**: `significance.py` draws bootstrap indices independently per-position (`rng.integers(0, a.size, size=(n_boot, a.size))`) rather than resampling contiguous blocks, so it cannot preserve or reflect serial correlation.
- **Suggested fix**: Implement a circular or stationary block bootstrap (resample contiguous blocks of a chosen length, wrapping around) in place of the current i.i.d. index draw.
- **Catcher test**: Bootstrap CI width on an autocorrelated series must shrink materially after the series is shuffled (destroying the autocorrelation); currently it stays ~unchanged (ratio ≈0.98 instead of the expected large drop).

---

## C. Tier 2 — Missing (designed, absent)

### BUG-11: rough Bergomi, SABR, and a Bates generator do not exist — SYNTHETIC
- **Severity / type**: Tier 2 — missing. **Gate-report disclosure**: Phase 6 (rough Bergomi/SABR) is disclosed as pending in the gate reports; Bates/Wave-3 is the brief's stated headline wave and is *not* separately flagged as absent in the same way.
- **Location**: `optspread/market/rough_bergomi.py`, `optspread/market/sabr.py`, `optspread/market/bates.py` (all absent); `optspread/market/pricing/char_funcs.py` (`bates_cf`, present but orphaned); `optspread/market/pricing/cos_pricer.py` (`COSPricer.bates_price`, present but orphaned/untested)
- **Designed to**: Provide (a) the Phase-6 primary held-out OOD family — rough Bergomi, described in the brief as "the decisive tail test" — plus SABR, and (b) the Wave-3 Bates jump-diffusion generator, described as where "the headline result" (CVaR vs PPO tail defense under jumps) is forged.
- **Actually does**: None of the three generators exist as importable classes.
- **Executed evidence**: `import optspread.market.rough_bergomi` / `.sabr` / `.bates` all → `ModuleNotFoundError`. `RoughBergomiGenerator` / `SABRGenerator` / `BatesGenerator` → `AttributeError`. `bates_cf` and `COSPricer.bates_price` import **successfully** but have **zero generator callers** anywhere in the codebase (orphaned pricing primitives), and `bates_price` is untested.
- **Reproduce**: `python -c "import optspread.market.rough_bergomi"`, `python -c "import optspread.market.sabr"`, `python -c "import optspread.market.bates"` — all fail. `grep -rn "bates_cf\|bates_price" optspread/ tests/` to confirm the char-function/pricer primitives exist but have no generator wired to them.
- **Root cause**: The Wave-3 (Bates) and Phase-6 (rough Bergomi, SABR) generator implementations were never built; only a Bates characteristic function and COS-pricer entry point exist as disconnected primitives.
- **Suggested fix**: Implement `RoughBergomiGenerator`, `SABRGenerator`, and `BatesGenerator` producing standardized IV surfaces per the surface-keystone invariant, and wire `bates_cf`/`bates_price` into the new Bates generator.
- **Catcher test**: `from optspread.market.bates import BatesGenerator` (and rough_bergomi/sabr equivalents) must succeed and produce a valid `IVSurface` via the standard generator interface; currently all three raise `ModuleNotFoundError`. Impact: Wave 3 (the brief's headline CVaR-vs-PPO tail-defense-under-jumps result) and the Phase-6 primary OOD family both have no supporting code.

### BUG-12: Per-regime return distributions from the distributional critic don't exist
- **Severity / type**: Tier 2 — missing (not flagged as pending in any gate report reviewed)
- **Location**: `optspread/interpret/` — `per_regime_dist.py` and `attribution.py` absent; `optspread/interpret/regime_map.py` (`RegimeCell`) present but scalar-only
- **Designed to**: Deliver the brief's stated unique investor payoff — per-regime return *distributions* (quantiles/CVaR) sourced from the trained distributional critic, attached to each cell of the distilled regime map.
- **Actually does**: `RegimeCell` stores only a scalar `mean_return`; nothing in `interpret/` reads from the distributional critic at all.
- **Executed evidence**: `from optspread.interpret import per_regime_dist` and `from optspread.interpret import attribution` both → `ImportError`. `RegimeCell` fields = `[cell_id, modal_action, mean_return, count]` — scalar mean only, no quantiles/CVaR/distribution field. No file under `optspread/interpret/` imports `optspread.agents.distributional` (the module housing the critic). Example instantiated cell: `RegimeCell(cell_id=0, modal_action=3, mean_return=-0.0748, count=29)`.
- **Reproduce**: `python -c "from optspread.interpret import per_regime_dist"` and `python -c "from optspread.interpret import attribution"` (both fail). Inspect `RegimeCell`'s dataclass fields in `optspread/interpret/regime_map.py`. `grep -rn "agents.distributional" optspread/interpret/` (zero matches).
- **Root cause**: The distillation pipeline (`regime_map.py`) was built against the agent's scalar action/return outputs and was never connected to the distributional critic's quantile/CVaR outputs.
- **Suggested fix**: Extend `RegimeCell` (or add a companion structure) to carry per-cell return quantiles/CVaR sourced from the trained distributional critic, and add the module(s) that compute and attach them.
- **Catcher test**: A distilled regime cell must expose a return distribution (e.g. quantiles or CVaR at a stated alpha) sourced from the critic, not just a scalar mean; currently no such field or computation exists anywhere in `interpret/`.

### BUG-13: Walk-forward is expanding-only; CPCV missing
- **Severity / type**: Tier 2 — missing. CPCV is named in PHASE5_BRIEF as an *optional* gold-standard deliverable, so its absence is lower-severity than the rest of this tier; the expanding-only limitation of the shipped walk-forward splitter is not separately disclosed.
- **Location**: `optspread/evaluation/walkforward.py`; `optspread/evaluation/cpcv.py` (absent)
- **Designed to**: Provide walk-forward splitting (with purge + embargo) supporting rolling-window folds, plus optionally Combinatorial Purged Cross-Validation (CPCV) as a gold-standard alternative, per PHASE5_BRIEF.
- **Actually does**: `WalkForwardSplitter` only ever expands the training window from a fixed start; there is no rolling-window mode. CPCV does not exist.
- **Executed evidence**: `WalkForwardSplitter(train_size=200, test_size=100, purge=5, embargo=10).split(1000)` → all **7 folds** have `train_start=0` (train length grows **200 → 860**, no rolling-window mode observed). `from optspread.evaluation import cpcv` → `ImportError`.
- **Reproduce**: Instantiate `WalkForwardSplitter(train_size=200, test_size=100, purge=5, embargo=10)` and call `.split(1000)`; inspect each fold's `train_start` and train length. Try `from optspread.evaluation import cpcv`.
- **Root cause**: The splitter was implemented with a single expanding-window strategy; a rolling-window option and the CPCV module were never added.
- **Suggested fix**: Add a rolling-window mode (fixed train-window length that slides forward) to `WalkForwardSplitter`, and implement `cpcv.py` per the brief's optional gold-standard spec.
- **Catcher test**: `WalkForwardSplitter` configured for rolling-window mode must produce folds with a constant (non-growing) train length and advancing `train_start`; currently `train_start` is always 0 regardless of configuration.

### BUG-14: Curriculum Waves 3-6 have no generators and the wiring hard-caps at Wave 2 — SYNTHETIC
- **Severity / type**: Tier 2 — missing. **Gate-report disclosure**: honestly disclosed — CLAUDE.md's own Phase-4 status line and the gate report state "Waves 2-6 not started" beyond what's complete.
- **Location**: `optspread/curriculum/dynamic_skew.py`, `regime_switching.py`, `microstructure.py` (all absent); `optspread/training/curriculum_factory.py` (`_wave_generator_factory`); `optspread/cli/validate_generators.py`; `optspread/curriculum/waves.py` (`WAVES` registry)
- **Designed to**: Progressively add one new generative feature per wave (per CLAUDE.md's curriculum discipline invariant) through Wave 6 (dynamic skew, regime switching, microstructure, etc.).
- **Actually does**: Only Waves 1 and 2 exist; the factory and CLI both hard-cap at Wave 2.
- **Executed evidence**: `dynamic_skew.py` / `regime_switching.py` / `microstructure.py` are absent from the repo. `optspread/training/curriculum_factory.py`'s `_wave_generator_factory` raises `ValueError` for any `wave_id` not in `{0, 1, 2}`. `cli/validate_generators.py --wave` has `choices=(1, 2)`. `curriculum/waves.py`'s `WAVES` registry contains only keys `{1, 2}`.
- **Reproduce**: `grep -rn "class.*Generator" optspread/curriculum/` to confirm only Wave 1/2 generators exist. Call `_wave_generator_factory(wave_id=3)` in `optspread/training/curriculum_factory.py` and observe the `ValueError`. Run `python -m optspread.cli.validate_generators --wave 3` and observe the CLI reject it (choices are 1,2 only).
- **Root cause**: Waves 3-6 (Bates/jump-diffusion, dynamic skew, regime switching, microstructure) were never implemented; this matches CLAUDE.md's own stated Phase-4 status ("Wave 1 COMPLETE... Wave 2 may now begin") — the work genuinely has not progressed past Wave 2 yet.
- **Suggested fix**: Implement Waves 3-6 generators one at a time per the curriculum discipline invariant (one new generative feature per wave, with pre-registered behavioral predictions and generator validation before agent training).
- **Catcher test**: N/A as a pure regression catcher (this is honestly-disclosed incomplete work, not a defect) — track via the existing gate-report status table rather than a pytest assertion.

---

## D. Tier 3 — Orphaned / dead / phantom

### BUG-15: CurriculumRunner.validate_only hardcodes False (can never promote a wave); zero call sites
- **Severity / type**: Tier 3 — orphaned
- **Location**: `optspread/curriculum/runner.py:38-39`
- **Designed to**: Provide the automated promotion decision (GV + BV + FF checks) that gates advancing from one curriculum wave to the next, per CLAUDE.md's curriculum discipline invariant.
- **Actually does**: Always returns a failing `PromotionDecision` regardless of input, because its behavioral-validation and forgetting-check sub-results are hardcoded to `False`/`nan` with placeholder reason strings; it also has no callers outside its own module.
- **Executed evidence**: Ran the real method with generator-validation forced to PASS → `PromotionDecision(passed=False, reason='BV failed: behavioral validation pending; FF failed: forgetting check pending')`; `BehavioralResult(False, nan, "behavioral validation pending")` and `ForgettingResult(False, "forgetting check pending")` are hardcoded inline in the source. Grep across the repo: zero call sites outside its own module and a prose mention in a brief.
- **Reproduce**: Instantiate `CurriculumRunner`, force/mock its generator-validation (GV) sub-check to pass, then call `validate_only` (or the method containing lines 38-39 of `runner.py`) and inspect the returned `PromotionDecision`, `BehavioralResult`, and `ForgettingResult`.
- **Root cause**: The behavioral-validation and forgetting-check logic were stubbed with hardcoded failing placeholders and never completed; the real curriculum promotion in practice runs via ad-hoc CLI scripts instead of this class.
- **Suggested fix**: Either implement real behavioral-validation and forgetting-check logic in `CurriculumRunner` and wire it into the actual training entrypoint, or remove the class if the ad-hoc CLI scripts are the intended long-term path.
- **Catcher test**: `CurriculumRunner.validate_only` must be able to return `passed=True` under some real input (currently structurally impossible since two of its three sub-checks are hardcoded to fail).

### BUG-16: FrameStackObservation (CLAUDE.md's prescribed distributional-agent memory mechanism) is wired into nothing
- **Severity / type**: Tier 3 — orphaned
- **Location**: `optspread/agents/sequence/framestack.py`
- **Designed to**: Serve as the distributional agent's memory mechanism (CLAUDE.md explicitly prescribes frame-stacking, not recurrence, for the IQN/QR-DQN agent).
- **Actually does**: Exists as a tested, working class with no production caller.
- **Executed evidence**: Its unit test passes (`1 passed`), but grepping across `optspread/` and `tests/` shows only the class definition plus its own dedicated test file — no `EnvFactory`, CLI, or config path instantiates it.
- **Reproduce**: `grep -rn "FrameStackObservation" optspread/ tests/` and confirm matches are limited to `optspread/agents/sequence/framestack.py` and its test file.
- **Root cause**: The class was implemented and unit-tested in isolation but never connected into the `EnvFactory`/training-config pipeline that would actually give the distributional agent frame-stacked memory.
- **Suggested fix**: Wire `FrameStackObservation` into `EnvFactory`/the distributional-agent training config so the shipped IQN/QR-DQN agent actually receives stacked frames, per CLAUDE.md's invariant.
- **Catcher test**: A distributional-agent training run configured for frame-stacking must produce observations with the stacked shape; currently no code path constructs `FrameStackObservation` outside its own test.

### BUG-17: In-vitro CVaR gates (G3/G4) test closed-form helpers, not the learning loop; purpose-built gyms never trained
- **Severity / type**: Tier 3 — orphaned
- **Location**: `optspread/toys/fat_tail_bandit.py`, `fat_tail_mdp.py`; `tests/test_fat_tail_bandit.py`, `test_fat_tail_mdp.py`
- **Designed to**: Provide in-vitro toy environments (fat-tail bandit and MDP) that a trained agent runs in to empirically demonstrate CVaR-vs-expected-value action selection (Phase 3's G3/G4 gates), per CLAUDE.md's "in-vitro fat-tail proof" gate.
- **Actually does**: The tests exercise only closed-form analytic helper functions (`greedy_arm`, `greedy_mdp_action`) operating on hardcoded distributions — no agent is trained or run in the toy envs.
- **Executed evidence**: Tests import only `greedy_arm`/`greedy_mdp_action` (pure numpy analytic calculations on hardcoded distributions). `FatTailMDPEnv`/`FatTailBanditEnv` instantiate without error but have zero constructors anywhere outside their own module — never driven by a training loop. The one test resembling an "agent" test zeroes the network and hand-injects the answer directly into the bias term, rather than training it.
- **Reproduce**: `grep -rn "FatTailBanditEnv\|FatTailMDPEnv" optspread/ tests/` to confirm no training-loop caller exists. Read `tests/test_fat_tail_bandit.py`/`test_fat_tail_mdp.py` to confirm they call only the standalone `greedy_arm`/`greedy_mdp_action` functions, and inspect the "agent" test to see the hand-injected bias.
- **Root cause**: The toy envs were built and the closed-form optimal-action helpers were validated against them, but the step of actually training the real agent implementation inside these envs (to empirically validate the training loop's CVaR behavior, not just the closed-form math) was never done.
- **Suggested fix**: Add a test that trains the actual `IQNAgent`/distributional trainer inside `FatTailBanditEnv`/`FatTailMDPEnv` and confirms its learned action selection matches the closed-form CVaR-optimal answer.
- **Catcher test**: A real (non-hand-injected) trained agent run inside `FatTailBanditEnv` must converge to the CVaR-optimal arm identified by `greedy_arm`; currently no test exercises the trained agent this way.

### BUG-18: Three of five reward components are dead in every real training run
- **Severity / type**: Tier 3 — orphaned
- **Location**: `optspread/reward/*`, `optspread/training/phase2.py`, `curriculum_factory.py`, `config.py`
- **Designed to**: Provide a composable, weighted-sum reward where every term (MTM P&L, margin normalizer, differential Sharpe, Sortino, CVaR penalty) is independently enable-able via its weight, per CLAUDE.md's reward-and-comparison-discipline invariant.
- **Actually does**: `margin_normalized_weight` and `sortino_weight` are 0.0 at every single `RewardConfig` construction site in the repo (default and every preset) — never enabled anywhere, with no documented rationale. `cvar_weight` is nonzero (0.25) only in the Wave-0 gate reward and 0.0 for Waves 1-6.
- **Executed evidence**: `margin_normalized_weight`, `sharpe_weight`, `sortino_weight` = **0.0** at every `RewardConfig` construction site repo-wide (default + all presets). `cvar_weight = 0.25` only in the Wave-0 gate reward, `0.0` for waves 1-6. A live rollout breakdown of `curriculum_reward()` shows: only `'mtm'` is nonzero each step; `margin_normalized`/`diff_sharpe`/`sortino`/`cvar` are each exactly `0.0` every step.
- **Reproduce**: `grep -rn "margin_normalized_weight\|sharpe_weight\|sortino_weight\|cvar_weight" optspread/training/ optspread/curriculum/` to enumerate every `RewardConfig(...)` construction call and its weight values. Separately, instantiate the curriculum reward function and roll out a few steps, logging the per-component breakdown dict each step.
- **Root cause**: `margin_normalized` and `sortino` were apparently never turned on anywhere (no rationale documented); `sharpe_weight` and `cvar_weight` are disabled specifically for curriculum waves due to a documented Wave-0-distortion concern (this part has a stated rationale, unlike the other two).
- **Suggested fix**: Either enable `margin_normalized_weight` and `sortino_weight` somewhere with a stated purpose (e.g. an ablation config, a later-wave preset), or explicitly document why they're permanently disabled — per CLAUDE.md's own hard-stop ("a reward-term ablation is inert → investigate; do not silently keep the term").
- **Catcher test**: At least one `RewardConfig` construction site in the training pipeline must set `margin_normalized_weight != 0` and `sortino_weight != 0` (or the config/reward module must document why not), and a rollout under that config must show nonzero contributions from those components in the reward breakdown.

### BUG-19: "tail ratio" is a phantom metric — named as required, computed nowhere
- **Severity / type**: Tier 3 — phantom
- **Location**: `optspread/eval/metrics.py`
- **Designed to**: Provide the tail-ratio metric named as a required tail-adjusted metric in both CLAUDE.md ("the win condition is tail-adjusted... not raw mean") and PHASE3_BRIEF's gate G6.
- **Actually does**: Does not exist anywhere in the codebase under this or any equivalent name.
- **Executed evidence**: `grep -rn tail_ratio optspread/` → **0 matches**. `EvalReport` fields = `per_step_returns, episode_returns, action_frequencies, mean_pnl, pnl_ci, sharpe, sortino, cvar_95, max_drawdown, turnover` — no `tail_ratio` field.
- **Reproduce**: `grep -rn tail_ratio optspread/` from repo root (zero matches). Inspect `EvalReport`'s dataclass fields in `optspread/eval/metrics.py`.
- **Root cause**: The metric was named in the brief/CLAUDE.md as a required exhibit but was never implemented in the metrics module.
- **Suggested fix**: Implement `tail_ratio` (commonly, e.g., the ratio of average gain in the top return decile to average loss in the bottom return decile, or a similar upside/downside tail ratio) and add it as an `EvalReport` field, computed alongside `sharpe`/`sortino`/`cvar_95`.
- **Catcher test**: `EvalReport` must expose a `tail_ratio` field populated with a real computed value from `per_step_returns`/`episode_returns`; currently the field does not exist.

### BUG-20: reporting/exhibits.py is inert placeholder metadata; no artifacts exist
- **Severity / type**: Tier 3 — orphaned/phantom
- **Location**: `optspread/reporting/exhibits.py`, `manifest.py`
- **Designed to**: Generate and track the write-up's standard exhibits (charts/tables) per Phase 8's reporting deliverable.
- **Actually does**: `standard_exhibits()` returns hardcoded metadata tuples describing exhibits that don't exist as generated files; the `Exhibit` dataclass has no field to even point at an artifact.
- **Executed evidence**: `standard_exhibits()` returns **5 hardcoded** `Exhibit(exhibit_id, section, title, synthetic_or_real)` tuples; the `Exhibit` dataclass has **no artifact/file-path field**. No `runs/`, `reports/`, `exhibits/`, or `artifacts/` directories exist anywhere in the repo; none of the 5 exhibit IDs correspond to a generated file on disk.
- **Reproduce**: Call `standard_exhibits()` from `optspread/reporting/exhibits.py` and inspect the 5 returned tuples plus the `Exhibit` dataclass definition (no path/artifact field). `ls`/`Get-ChildItem` the repo root and confirm no `runs/`, `reports/`, `exhibits/`, or `artifacts/` directories exist.
- **Root cause**: The exhibits module was built as descriptive metadata (what exhibits *should* exist) without ever being connected to code that actually generates and saves the corresponding chart/table files.
- **Suggested fix**: Add an artifact/file-path field to `Exhibit`, and implement the generation code that actually produces each of the 5 standard exhibits (charts/tables) and writes them to a tracked output directory.
- **Catcher test**: For each `Exhibit` returned by `standard_exhibits()`, a file must exist on disk at its associated path; currently the dataclass has no such path field and no files exist at all.

---

## E. Corrections & Verified-Correct

### NOTE-A: The nested-CVaR bootstrap is NOT dead code (REFUTED — a preliminary static read overstated this)
- **Type**: Correction (adversarial pass refuted an initial claim)
- **Location**: `optspread/agents/distributional/trainer.py:~76-81`; `optspread/finetune/finetune_real.py:~221-223`
- **What was initially claimed**: The `bootstrap_risk="cvar"` (nested-CVaR bootstrap) code path is unreachable/dead.
- **What execution showed**: A real `DistributionalTrainer` + `IQNAgent` ran **4 gradient-step cycles** with `bootstrap_risk="cvar"` against the toy env, with no errors — the branch is live and reachable. It defaults to `"mean"` and is forbidden only at the fine-tune CLI boundary (`finetune_real.py:~221-223` raises `SystemExit` if `bootstrap_risk != "mean"`), by documented design (the CLAUDE.md-noted "blindness to success" mitigation — explore/bootstrap risk-neutral, apply CVaR only at deployment action-selection).
- **Conclusion**: The shipped "CVaR agent" uses mean-bootstrap during training plus CVaR at action-selection time, by deliberate choice — not because the nested-CVaR path is broken or dead. The nested-CVaR path exists and works but is not the shipped recipe. This is a **design note / caveat** (consistent with CLAUDE.md's explicit statement that static/nested CVaR choices should be documented as caveats), not a bug.
- **Reproduce**: Construct `DistributionalTrainer`/`IQNAgent` with `bootstrap_risk="cvar"` against one of the toy envs (`optspread/toys/`) and run several gradient-step cycles, confirming no exceptions. Separately, inspect `finetune_real.py:~221-223`'s `SystemExit` guard requiring `bootstrap_risk == "mean"` at the real-data fine-tune CLI boundary.

### NOTE-B: IQN CVaR tau sampling is genuinely real (verified positive)
- **Type**: Verified-correct
- **Location**: `optspread/agents/distributional/` (IQN network, cosine embedding, quantile-Huber loss, CVaR-from-quantiles action selection); `trainer.py` (`_gradient_step`)
- **Evidence**: 32,000 CVaR-acting taus were all `<= alpha=0.1`, with a KS test **p=0.95** against Uniform(0, alpha) — consistent with correctly-implemented restricted-tau sampling for CVaR action selection. Training-time taus are full `U(0,1)` via `torch.rand` in `trainer._gradient_step`, as expected for standard IQN quantile regression (as opposed to the restricted acting-time taus used for CVaR selection).
- **Conclusion**: The core distributional machinery (cosine embedding, quantile-Huber loss, CVaR-from-quantiles action selection) is sound and matches the IQN/CVaR design CLAUDE.md specifies (`tau ~ U(0, alpha)` for CVaR action selection).
- **Reproduce**: Sample a large number (e.g. 32,000) of acting-time taus from the IQN agent's CVaR action-selection path with `alpha=0.1`, confirm all values are `<= 0.1`, and run a KS test against `Uniform(0, 0.1)`. Separately, instrument `trainer._gradient_step` to confirm training-time taus are drawn from full `U(0,1)`.

### NOTE-C: Solid foundation — core numeric spine is genuinely built and tested
- **Type**: Verified-correct
- **Evidence**: 203/204 repo tests pass (the single failure is BUG-04). Verified as genuinely real and tested: Black-Scholes and greeks, the strike solver, the 13 spread templates plus the 19-action library, P&L accounting, Reg-T margin, the cost model, the COS pricer (validated against a published Fang-Oosterlee benchmark value of **5.785** and against a Monte Carlo oracle), the PPO implementation (CleanRL-faithful), the byte-identical shared `EnvFactory`/`Evaluator`/`MetricSuite`/`TrainHarness` spine used across agents, the Heston Wave-1/Wave-2 generators with real P-vs-Q VRP, `RealDataReplay`, and the WRDS extractor.
- **Reproduce**: `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q` (203 passed, 1 failed — the BUG-04 failure). Separately, run the COS pricer against the published Fang-Oosterlee test case and confirm the **5.785** benchmark value, and against a Monte Carlo oracle for a cross-check.
