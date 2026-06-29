# Phase 4 Implementation Brief for Claude Code
## SPX Options Spread-Selection RL — The Synthetic Curriculum (Waves 1–6)

> **Scope of this phase.** Build the synthetic market generators that add one generative feature at a time (VRP, stochastic vol, jumps, dynamic skew, hidden regimes, microstructure), the domain-randomization infrastructure, and the curriculum runner that walks both agents (PPO and the distributional/CVaR agent from Phase 3) through the waves with **pre-registered promotion gates**. This is the methodological spine of the thesis. **Wave 3 is where the headline result is forged** (the CVaR agent defends the tail that the expected-value agent chases). **Wave 5 is what justifies the entire "regime-conditional" framing** (the agent must infer a hidden regime from observables alone). The deliverable is a trained agent that has cleanly mastered each feature in isolation, with a per-wave evidence trail, ready for real-data fine-tuning in Phase 5.
>
> **Depends on:** Phase 3 complete and green. QR-DQN and IQN pass the in-vitro fat-tail gates and the Wave-0 no-edge gate; the shared harness and `compare.py` work.

---

## 0. Stack decisions (fixed)

- **Option pricing for synthetic surfaces:** characteristic-function pricing via the **COS method (Fang–Oosterlee)** or Carr–Madan FFT for European options (SPX is European, so this is exact and fast). Monte Carlo only as a cross-check oracle, never in the hot path.
- **Generator output is a standardized IV surface** (see section 1, the single most important design decision in this phase), on the **same delta/maturity grid as the OptionMetrics smoothed surface** the real data will provide.
- **Domain randomization** parameters via Pydantic priors + a `ParamSampler`; resampled per episode.
- **Hidden-regime inference:** frame-stacking (history of k observations) as the primary mechanism; recurrent (LSTM) PPO as an optional extension. Recurrent off-policy distributional (R2D2-style) is **out of scope** — too heavy for the timeline; frame-stacking gives the distributional agent the memory it needs.
- Everything else (PyTorch, the shared `training/`/`eval/` harness, mypy strict, ruff, explicit RNG, causal features) carries over unchanged.

---

## 1. The architectural keystone: generators emit a standardized IV surface

**Every generator produces, for each trading day, two things: the underlying spot, and a standardized implied-volatility surface on the exact grid the OptionMetrics smoothed-surface file uses (deltas 10–90, constant maturities 10d–2y).** The environment builds tradeable option prices by interpolating that surface to the specific strikes and expiries of the chosen structure, and the feature pipeline computes regime features from the same surface.

Why this matters: it makes the environment **generator-agnostic and identical for synthetic and real data.** In Phase 5, real data is just "the surface and the path come from OptionMetrics instead of a generator," with **zero changes to the env, the features, the costs, the agents, or the eval.** That is the clean sim-to-real bridge. It also means each generator's pricer runs **once per simulated day to produce the surface**, not once per option per step, which keeps the COS pricing affordable.

Refactor the Phase 1 `MarketSnapshot`/`ChainSnapshot` so the chain is **derived from a surface**. Wave 0's flat constant-IV "surface" is the degenerate case; this is backward compatible.

The variance risk premium is then expressed cleanly as a **measure difference**: simulate the underlying under the physical measure P (real `kappa, theta`), but price the surface under the risk-neutral measure Q with a **VRP-adjusted long-run variance** (`theta_Q > theta_P`), so options are systematically richer than realized vol. This unifies Wave 1's VRP into the Wave 2+ structure rather than bolting it on.

---

## 2. The waves (one generative feature each; do not combine)

| Wave | Generator | New feature turned on | Pre-registered prediction |
|---|---|---|---|
| 0 | GBM, fair IV | none (baseline) | FLAT-dominant, no edge (already gated in Phase 2/3) |
| 1 | GBM + VRP | options priced richer than realized (P vs Q gap) | credit-structure frequency rises with the VRP feature; returns positive but bounded |
| 2 | Heston SV | mean-reverting stochastic vol + vol-of-vol | action choice conditions on IV rank / term-structure slope |
| 3 | Bates (Heston + jumps) | jumps / fat tails | defined-risk share rises with the jump proxy; **CVaR agent's ES materially beats PPO's** |
| 4 | Dynamic skew + leverage | moving skew (sticky-strike vs sticky-delta), regime drift | directional/RR/broken-wing use tracks drift; skew drives wing placement |
| 5 | Markov regime-switching, **regime HIDDEN** | unobservable calm/stressed/trending switching | policy clusters into regime-conditioned modes inferred from observables alone |
| 6 | Microstructure | stress-widening bid-ask, discrete strikes/expiries, liquidity by moneyness, margin realism | avoids illiquid wings in stress; turnover/cost drag sensible |

**Wave-specific notes:**
- **Wave 2 (Heston):** sample `kappa in [1,10], theta in [0.02,0.09], sigma_v in [0.2,1.0], rho in [-0.9,-0.3], v0 ~ theta` per episode. `rho` must be **negative** (leverage effect) — a sign error here is the classic calibration bug. IV rank and term slope only become informative because vol now mean-reverts; verify that before training (section 5).
- **Wave 3 (Bates):** add log-normal Poisson jumps `lambda in [0.1,2]/yr, mu_J in [-0.10,-0.02], sigma_J in [0.02,0.10]`. The agent sees a **market-observable jump proxy** (risk-neutral skew/kurtosis from the surface, or smile curvature), **never lambda directly.** This is the wave where the central result lives — budget the most time here.
- **Wave 4 (dynamic skew):** make skew time-varying — e.g. a stochastic skew factor or regime-dependent `rho`/jump params, plus a regime-switching drift so directional structures earn their place. Implement sticky-strike vs sticky-delta as an explicit, configurable rule for how the surface moves when spot moves.
- **Wave 5 (hidden regime):** a Markov chain over 2–3 regimes, each parameterizing the Wave-4 generator, with persistent transitions (diagonal ~0.95–0.99). The **regime label and true params are never in the observation** (assert it). The agent gets memory via frame-stacking so it can infer the regime from the observable feature history.
- **Wave 6 (microstructure):** bid-ask that **widens as a function of vol/regime**, **discrete listed strikes and weekly/monthly expiries** (round to grid), and **liquidity by moneyness** (deep-OTM wings cost more). This is where the Phase 1 `CostModel` gets calibrated to realistic, state-dependent spreads. **Note:** SPX is European and cash-settled, so **assignment/pin risk is N/A** here — flag that it would matter only for the single-name extension.

---

## 3. Repository additions

```
optspread/
  market/
    surface.py             # IVSurface: standardized (delta x maturity) grid; interpolate_to(strike, expiry)
    base_generator.py      # SurfaceGenerator(PriceGenerator): emits (spot, IVSurface) per day; ParamSampler hook
    gbm_vrp.py             # Wave 1
    heston.py              # Wave 2  (P-measure path; Q-measure surface via VRP-adjusted theta)
    bates.py               # Wave 3
    dynamic_skew.py        # Wave 4
    regime_switching.py    # Wave 5 (wraps a base generator; regime HIDDEN)
    microstructure.py      # Wave 6 (state-dependent bid-ask, discretization, liquidity)
    priors.py              # DR priors per generator (Pydantic); ParamSampler
    pricing/
      cos_pricer.py        # COS-method European pricer from a characteristic function
      char_funcs.py        # Heston, Bates characteristic functions
      mc_oracle.py         # Monte-Carlo pricer used ONLY to test the COS pricer
  features/
    regime_features.py     # VRP, IV rank/pct, term slope, 25d RR & fly, realized skew/kurt, momentum, jump proxy
    causal.py              # point-in-time guarantees (trailing windows only)
  curriculum/
    waves.py               # WAVES registry: id -> (generator factory, DR priors, cost config, prediction)
    predictions.py         # PreRegisteredPrediction: hypothesis, statistic, threshold, direction
    promotion.py           # promotion gate: prediction met AND no catastrophic forgetting
    rehearsal.py           # mix a fraction of earlier-wave episodes into training
    runner.py              # CurriculumRunner: warm-start, train, validate, gate, re-eval, advance
  agents/
    sequence/
      framestack.py        # k-observation history wrapper (works for PPO and distributional)
      recurrent_ppo.py     # OPTIONAL: LSTM PPO
  eval/
    generator_validation.py # validate the SIM produces the intended stylized fact (no agent)
    behavioral.py           # per-wave behavioral tests on the trained policy
tests/
  test_cos_pricer.py        # COS vs MC oracle and vs published Heston/Bates benchmark values; parity
  test_char_funcs.py
  test_surface.py           # grid integrity, interpolation, no-arb sanity (monotonic, butterfly>=0)
  test_vrp_invariant.py     # IV systematically exceeds realized by the configured premium
  test_domain_randomization.py # params resampled per episode; obs excludes internals
  test_regime_hidden.py     # regime label & true params absent from observation
  test_framestack.py
  test_generator_validation.py
  test_promotion_gate.py
  test_behavioral_wave1.py ... test_behavioral_wave6.py
```

---

## 4. Domain randomization and the curriculum runner

- **`ParamSampler`** draws generator params from wide-but-plausible priors **every episode**. This is the primary overfitting control. Priors should bracket SPX-plausible values (the Wave 2/3 ranges above are starting points). Consider **Automatic Domain Randomization** (widen a prior once the agent masters its current range) as an optional enhancement; start with fixed wide priors.
- **Hidden internals everywhere:** the observation is built only from the surface and the path. Sampled params, the latent variance, and the regime label are never exposed. Enforce with a test.
- **`CurriculumRunner`** for each wave: **warm-start from the previous wave's checkpoint** (curriculum = transfer learning), train under DR, then run the validation gates (section 5). Promote only on pass.
- **Anti-forgetting:** after each new wave, **re-evaluate waves 0..i-1** and require performance within tolerance; **rehearse** by mixing a small fraction of earlier-wave episodes into training. Catastrophic forgetting across waves is a real and common failure; budget for it.

---

## 5. Validation gates per wave (the definition of done)

Each wave i in 1..6 passes **three** gates, in order. This two-stage structure (validate the sim, then validate the agent) is what catches calibration bugs before they masquerade as agent failures.

**GV_i — Generator validation (no agent).** Before training anything, prove the generator produces the intended stylized fact on simulated data:
- Wave 1: mean(IV − realized) > 0 and approximately the configured premium; VRP feature varies.
- Wave 2: IV rank spans its range across episodes; term-structure slope varies and mean-reverts; `rho < 0` produces a downward skew.
- Wave 3: simulated returns are fat-tailed (excess kurtosis > 0); the jump proxy moves with realized jump activity.
- Wave 4: the skew measure (25d RR) is **time-varying**, not static; drift regimes are distinguishable.
- Wave 5: regimes are statistically distinguishable in the observable features (e.g. a classifier on features recovers the hidden regime well above chance — confirming the info is *there to be inferred*).
- Wave 6: bid-ask widens with vol; strikes/expiries are on the discrete grid; deep-OTM spreads exceed ATM.
**If GV_i fails, fix the generator. Do not train the agent on a sim that lacks the feature.** This is the antidote to the "wrong sign on rho / vol-of-vol too small / jump intensity too high" class of bugs.

**BV_i — Behavioral validation (trained agent), pre-registered.** The prediction and its statistical test are committed to `curriculum/predictions.py` **before** the training run. Each is a concrete, falsifiable test:
- Wave 1: corr(credit-structure indicator, VRP feature) > 0, significant; episode Sharpe positive but bounded.
- Wave 2: mutual information / correlation between IV-rank (and term slope) and chosen structure exceeds threshold.
- Wave 3 (**headline**): (a) corr(defined-risk indicator, jump proxy) > 0; (b) **ES_95 / CVaR_95 of the CVaR agent is materially better (less negative) than PPO's** on the identical Wave-3 eval via the shared suite; (c) the **isolation ablation** from Phase 3 (scalar CVaR penalty weight = 0) shows the CVaR agent still controls the tail while PPO does not.
- Wave 4: directional/RR/broken-wing usage tracks the drift regime; skew features predict wing placement.
- Wave 5: the agent's behavior **clusters into regime-conditioned modes** — e.g. adjusted Rand index between (clusters of the obs→action map) and the true hidden regime exceeds threshold, **despite the regime never being observed.** This is the experiment that earns the "regime-conditional" claim.
- Wave 6: turnover and cost drag are economically sensible; the agent avoids illiquid deep-OTM wings in high-stress states.

**FF_i — Forgetting check.** Re-evaluate waves 0..i-1; require performance within tolerance (no catastrophic forgetting).

**Promote to wave i+1 only if GV_i, BV_i, and FF_i all pass, across at least 3–5 seeds with reported dispersion.**

---

## 6. The Wave-3 headline (the central Results table)

At Wave 3, run PPO, the risk-neutral distributional agent, and the CVaR distributional agent through the **identical** Wave-3 generator and the **shared** `Evaluator`/`MetricSuite`, and emit the head-to-head: mean return, Sharpe, Sortino, **CVaR/ES, max drawdown, tail ratio**. The expected prediction: PPO and risk-neutral chase the higher-mean tail-selling behavior and carry a worse left tail; the CVaR agent gives up a little mean for a materially better ES/CVaR/max-drawdown. Then the **isolation ablation** (scalar penalty off) shows the native distributional objective alone carries the tail control. This table plus the Wave-3 per-regime return distributions is the empirical heart of the thesis.

---

## 7. Pitfalls to engineer against

- **Two features at once.** Strictly one new generative feature per wave, or you cannot attribute behavior. Resist the urge to "just also add jumps."
- **Training before generator validation.** GV_i precedes agent training, always. A non-informative or sign-flipped feature wastes a week of training and produces a confusing null.
- **Look-ahead in features.** IV rank, realized vol, momentum, realized skew/kurt use **trailing windows only**. Enforce via `features/causal.py` and a test.
- **Catastrophic forgetting.** Rehearsal + per-wave re-eval (FF_i). Without it, mastering Wave 5 can erase Wave 1 behavior.
- **DR priors mis-sized.** Too narrow -> the agent overfits a sliver of parameter space; too wide -> it cannot learn. Tune; consider ADR.
- **Sim-artifact exploitation.** If the agent posts implausible edge, suspect it is exploiting a predictable-vol or pricing artifact. The real defense is the **held-out generator test in Phase 6** (rough-vol/SABR); do not declare victory on in-family results alone.
- **Pricing in the hot path.** Price the **surface once per simulated day**, cache it; never price per-option-per-step. Validate the COS pricer against the MC oracle and published benchmarks first.
- **Recurrent off-policy temptation.** Do not build R2D2-style recurrent QR-DQN/IQN. Use frame-stacking for the distributional agent; reserve LSTM for PPO only if needed.
- **Leaking the regime.** Assert the hidden regime label and true params never enter the observation (test), or Wave 5 proves nothing.

---

## 8. Build order (do not reorder)

1. `market/surface.py` + `pricing/cos_pricer.py` + `char_funcs.py` + `mc_oracle.py`; `test_cos_pricer.py`, `test_surface.py` (pricer correct and fast before any generator uses it).
2. Refactor `MarketSnapshot` to be surface-derived (backward compatible with Wave 0).
3. `market/priors.py` + `ParamSampler` + `test_domain_randomization.py`; `features/regime_features.py` + `causal.py`.
4. `curriculum/predictions.py`, `promotion.py`, `rehearsal.py`, `runner.py` + `test_promotion_gate.py`; `eval/generator_validation.py` + `eval/behavioral.py`.
5. **Wave 1:** `gbm_vrp.py` + `test_vrp_invariant.py` + GV_1, train (warm-start from Wave 0), BV_1, FF_1.
6. **Wave 2:** `heston.py` (P-path, Q-surface) + GV_2, train, BV_2, FF_2.
7. **Wave 3:** `bates.py` + GV_3, train, BV_3 (**headline table + isolation ablation**), FF_3.
8. **Wave 4:** `dynamic_skew.py` + GV_4, train, BV_4, FF_4.
9. `agents/sequence/framestack.py` + `test_framestack.py` (and optional `recurrent_ppo.py`).
10. **Wave 5:** `regime_switching.py` + `test_regime_hidden.py` + GV_5, train (frame-stacked), BV_5 (**regime-conditioning demonstration**), FF_5.
11. **Wave 6:** `microstructure.py` + cost-model calibration + GV_6, train, BV_6, FF_6.
12. Full cross-wave re-eval; multi-seed dispersion; per-wave diagnostics write-up.

**Update `CLAUDE.md`** with: the surface-as-generator-output keystone (synthetic and real share the env), the one-feature-per-wave rule, the GV-before-training rule, the pre-registered-prediction requirement, the hidden-internals invariant, the frame-stack-not-recurrent decision for the distributional agent, and the anti-forgetting protocol.

---

## 9. Process instructions for Claude Code

- **Commit each wave's pre-registered prediction (and its test) BEFORE running that wave's training.** Timestamp it. This is the methodological backbone; a prediction written after seeing results is not a test.
- Work **one wave at a time**; do not scaffold multiple generators in parallel. Each wave: GV -> train -> BV -> FF -> promote.
- Validate the COS pricer against the MC oracle and published Heston/Bates benchmark values before any generator depends on it.
- Reuse the shared harness and `compare.py` unchanged; the only new agent code is the frame-stack/recurrent wrapper.
- Keep all features causal (trailing windows); assert hidden internals are absent from observations.
- `make check` green (mypy strict, ruff) after each module; commit at module/wave granularity referencing the gate that passed.
- Log per wave: generator-validation statistics, the pre-registered prediction outcome, per-action frequencies vs the relevant feature, full eval return distributions, the Wave-3 head-to-head, the Wave-5 regime-clustering figure, and forgetting-check results across earlier waves, all with per-seed dispersion.
- **Do not start Phase 5 (real-data fine-tuning) or Phase 6 (held-out generator).** But because the env is surface-driven, confirm in a comment that swapping a generator for the OptionMetrics surface would require no env changes — that is the Phase 5 entry point.
- End with a report: per-wave GV/BV/FF outcomes with dispersion, the Wave-3 headline table and isolation ablation, the Wave-5 regime-conditioning evidence, the cross-wave forgetting check, and an explicit statement that the curriculum is complete and the project may proceed to Phase 5.

---

### Ready-to-paste kickoff prompt

> Implement Phase 4 of the project as described in `PHASE4_BRIEF.md`, building on the completed Phase 3 repo. Start with the surface keystone (section 1): every generator emits a standardized IV surface on the OptionMetrics grid, and the env derives tradeable prices and features from it, so synthetic and real data share one env. Build and benchmark the COS pricer (vs an MC oracle and published Heston/Bates values) before any generator uses it. Then walk Waves 1–6 strictly one feature at a time using the CurriculumRunner: for each wave, commit the pre-registered prediction first, pass generator-validation (the sim produces the feature) before training, warm-start from the prior wave, then pass the pre-registered behavioral gate and the forgetting check across at least 3 seeds before promoting. Express VRP as a P-vs-Q measure difference (Q-surface uses a VRP-adjusted long-run variance). Keep all generator internals and the hidden regime out of the observation (assert it). Use frame-stacking, not recurrent off-policy, to give the distributional agent memory for Wave 5. At Wave 3 produce the headline head-to-head (PPO vs risk-neutral vs CVaR on ES/CVaR/maxDD/tail-ratio) plus the isolation ablation. At Wave 5 demonstrate regime-conditioned behavior clustering against the hidden regime. Reuse the shared harness and compare.py unchanged. Do not start real-data fine-tuning. Keep mypy strict and ruff green, and end by stating whether the curriculum is complete and the project may proceed to Phase 5.
