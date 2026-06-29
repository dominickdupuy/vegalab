# Phase 6 Implementation Brief for Claude Code
## SPX Options Spread-Selection RL — The Held-Out-Generator Generalization Test

> **Scope of this phase.** Take the **frozen** agent trained on the Heston+Bates family (Phase 4, and the lightly fine-tuned Phase 5 variant) and evaluate it **zero-shot, with no retraining**, on **structurally different** market generators it has never seen: **rough Bergomi** (the primary held-out family), **SABR**, and **GARCH/GJR-GARCH** paths. Success is **graceful degradation, not collapse**: the agent should still beat its baselines on tail-adjusted metrics on a model family built on a different mathematical mechanism, and the CVaR agent's tail advantage should survive a **different tail-generating mechanism** than the Bates jumps it trained against. This is the strongest single piece of generalization evidence in the thesis, and the strongest test of the central tail claim. This phase may overlap Phase 5.
>
> **Depends on:** Phase 4 complete (the curriculum agent and its seed ensemble exist); Phase 5's `RealDataReplay` and walk-forward infra exist (reused for context, not required to start). The env is surface-driven, so a held-out generator is a drop-in.

---

## 0. Stack decisions (fixed)

- **Held-out families, in priority order:**
  1. **Rough Bergomi** (rough volatility, fractional/Volterra driver, Hurst `H` ~ 0.05–0.15). The primary test: genuinely **non-Markovian** vol with a **power-law short-maturity skew** that Heston structurally cannot produce. Strongest evidence.
  2. **SABR** (Hagan): different smile parameterization (CEV backbone, lognormal vol-of-vol).
  3. **GARCH / GJR-GARCH**: discrete-time conditional heteroskedasticity with leverage; a different vol-clustering and tail mechanism, priced under a risk-neutral GARCH measure (Duan LRNVR).
  - Optional, if time permits: a **Quant-GAN** data-driven generator (maximally different, no economic priors). Heavy; do not block the phase on it.
- **Held-out generators are drop-in `SurfaceGenerator`s** (Phase 4 keystone): each emits (spot, IVSurface) on the standardized OptionMetrics grid, so the env, agents, and evaluator are unchanged.
- **Frozen agent, zero-shot only.** No gradient steps on any held-out family. Assert it.
- **Rough Bergomi pricing is Monte-Carlo** (no closed form): MC with variance reduction, pricing the **surface once per simulated day** (keystone), never per option per step.
- Everything else (PyTorch, shared harness, mypy strict, ruff) unchanged.

---

## 1. Design principles specific to this phase

1. **"Structurally different" must be proven, not asserted.** A held-out family that is just Heston with other parameters is not a test. Build a **structural-distance diagnostic** that shows each held-out family produces surfaces/feature distributions **out-of-distribution** relative to the Heston+Bates training priors (different ATM-skew term structure, different smile convexity dynamics). Rough Bergomi's power-law short-maturity skew is the clean example: Heston cannot match it, so it is genuinely held out.
2. **Frozen agent, no retraining.** The entire value of this experiment is that the agent never saw the family. Any fine-tuning on the held-out data destroys the test. Enforce by asserting zero optimizer steps during held-out evaluation.
3. **This is the decisive test of the tail claim.** Bates generates tails via Poisson jumps; rough vol and GJR-GARCH generate tails via different mechanisms (rough vol bursts, asymmetric clustering). If the **CVaR agent still defends the tail** under a mechanism it never trained on, the tail-awareness is real, not Bates-specific. Center the analysis on whether the CVaR > risk-neutral > PPO tail ranking **survives** out-of-family.
4. **Report the full seed ensemble; never cherry-pick.** Evaluate every Phase 4 seed on every held-out family and report the **distribution** (mean and dispersion). A single transferring seed is not a result. Optionally build an ensemble policy (vote/average) as the more robust deployable artifact.
5. **Graceful degradation is defined quantitatively, in advance.** Pre-declare the bound: the agent still beats the always-on structures and the VRP heuristic on tail-adjusted metrics out-of-family, the agent ranking is preserved, and the metric drop stays within a stated tolerance. Decide this before looking at results.

---

## 2. Repository additions

```
optspread/
  market/
    rough_bergomi.py        # rough Bergomi: Volterra/hybrid scheme path + variance; surface via MC pricer
    sabr.py                 # SABR: Hagan implied-vol -> surface (per-maturity / dynamic SABR for term structure)
    garch.py                # GJR-GARCH simulation + risk-neutral (Duan LRNVR) pricing
    pricing/
      rbergomi_pricer.py    # MC pricing (variance reduction); builds the standardized surface per day
      sabr_iv.py            # Hagan expansion -> IV at the standardized (delta x maturity) grid
      garch_pricer.py       # Duan-style GARCH option pricing / risk-neutral MC
  evaluation/
    structural_distance.py  # quantify train-family vs held-out-family difference in surface/feature space
    generalization.py       # frozen-agent zero-shot eval: in-family vs out-of-family via the shared suite
    ensemble.py             # evaluate the full seed ensemble; report dispersion; optional vote/average policy
  cli/
    heldout_eval.py
    generalization_report.py
tests/
  test_rough_bergomi.py     # power-law short-maturity ATM skew; H controls roughness; parity on the surface
  test_sabr.py              # Hagan smile shape; coherent multi-maturity surface; parity
  test_garch.py             # volatility clustering; GJR leverage asymmetry; risk-neutralization sane
  test_structural_distance.py # held-out families are OOD vs the Heston+Bates training priors (the "really different" proof)
  test_generalization.py    # frozen-agent pipeline; assert zero optimizer steps on held-out data
  test_ensemble.py
```

---

## 3. The held-out generators (build each as a structurally distinct family)

- **Rough Bergomi (primary).** Simulate the variance process driven by a fractional/Volterra kernel (hybrid scheme of Bennedsen-Lunde-Pakkanen for efficiency) with `H` ~ 0.1, correlation `rho` ~ -0.9, vol-of-vol `eta` ~ 1.9. Price the surface by **Monte Carlo** (no closed form) with variance reduction (antithetic/control variates), once per simulated day. The structural signature to validate: **ATM skew explodes like a power law ~ T^(H-1/2) at short maturity**, which Heston cannot reproduce. Layer VRP as the same P-vs-Q measure adjustment used in Phase 4.
- **SABR (secondary).** Use Hagan's implied-vol expansion to populate the standardized surface; `beta` (backbone), `alpha` (vol level), `rho` (skew), `nu` (vol-of-vol). SABR is natively per-expiry, so build a **coherent multi-maturity surface** (per-maturity SABR fits or a dynamic-SABR term structure); document the choice. Structural signature: the characteristic SABR smile and its distinct backbone dynamics.
- **GARCH / GJR-GARCH (secondary).** Simulate returns under GJR-GARCH (captures the leverage asymmetry), price options under the risk-neutral GARCH measure (**Duan's locally risk-neutral valuation**). Structural signature: volatility clustering and asymmetric response to negative returns, a discrete-time tail mechanism unlike continuous-time jumps.

Each must pass **generator validation** (its structural signature is present) before any agent evaluation, exactly as in Phase 4.

---

## 4. The structural-distance diagnostic (makes "held-out" rigorous)

Build `structural_distance.py` to quantify, in surface/feature space, how different each held-out family is from the Heston+Bates training distribution:
- Compare distributions of **ATM-skew term structure** (slope of skew vs maturity), **smile convexity**, **vol-of-vol realized**, and the **VRP and jump-proxy features** between training-family samples and held-out samples.
- Report a divergence (e.g. distributional distance, or the fraction of held-out feature vectors falling outside the training-prior support). 
- The deliverable: a figure/table showing the held-out families are **out-of-distribution**, with rough Bergomi clearly the most distant. This pre-empts the "too similar, so not a real test" objection and is itself a robustness-section exhibit.

---

## 5. Validation gates (the definition of done)

1. **GV (generator validation).** Each held-out generator produces a sensible standardized surface **and** its intended structural signature (rough power-law skew; SABR smile; GARCH clustering + GJR asymmetry). Pricers validated (rough-Bergomi MC against a reference/benchmark where available; SABR Hagan against known values; GARCH risk-neutralization sane). Parity/no-arb sanity on each surface.
2. **Structural distance confirmed.** The diagnostic shows each held-out family is OOD vs the training priors, rough Bergomi most strongly. (If a family is not meaningfully different, it is not a valid held-out test, fix it or drop it.)
3. **Frozen, zero-shot.** `test_generalization` asserts **no optimizer steps** occur during held-out evaluation. The Phase 4 (and Phase 5) agents are evaluated as-is through the shared `Evaluator`/`MetricSuite`.
4. **The generalization gate (pre-declared).** On each held-out family, out-of-family:
   - the agent still **beats the always-on structures and the VRP heuristic** on tail-adjusted metrics (Sharpe, Sortino, CVaR/ES, max drawdown, tail ratio);
   - the **agent ranking is preserved** (CVaR best on tail, then risk-neutral, then PPO);
   - the metric **drop vs in-family stays within the pre-declared tolerance** (graceful, not collapse).
   - **Critically:** the **CVaR agent's tail advantage survives** the different tail mechanism, especially on rough Bergomi and GJR-GARCH.
5. **Ensemble dispersion.** Every Phase 4 seed is evaluated on every held-out family; the **full distribution** is reported (mean and spread). No seed is cherry-picked. Optional ensemble policy reported.

A large, uniform collapse means the agent learned generator-specific quirks; report it honestly (it bounds the contribution and motivates wider priors or the gap-study framing from Phase 5).

---

## 6. Pitfalls to engineer against

- **Held-out family too similar.** Without the structural-distance proof, a "held-out" Heston variant tests nothing. Lead with rough Bergomi precisely because it is non-Markovian and Heston-inexpressible.
- **Seed cherry-picking.** Report the full ensemble distribution, not the best seed. Pre-commit to reporting all.
- **Accidental retraining / leakage.** Frozen agent only; assert zero gradient steps; do not let the obs-normalizer adapt to held-out statistics in a way that constitutes fitting (freeze normalization stats from training).
- **Rough-Bergomi pricing cost.** MC-only and expensive; use the hybrid scheme + variance reduction and price the **surface once per day** (cache), consistent with the keystone. Do not price per option per step.
- **SABR term structure.** SABR is per-expiry; build a coherent multi-maturity surface and document it, or the surface will be inconsistent across maturities.
- **GARCH risk-neutralization.** Get the measure change (Duan LRNVR) right; a wrong change of measure corrupts the VRP and the whole surface.
- **Over-claiming.** Generalizing to one family is good; to several structurally different families is strong. State the scope honestly; the GAN family is optional, not required.

---

## 7. Build order and process

**Build order (do not reorder):**
1. `evaluation/structural_distance.py` + `test_structural_distance.py` (define "different" first, so you can validate each family as you build it).
2. `market/rough_bergomi.py` + `pricing/rbergomi_pricer.py` + `test_rough_bergomi.py` (primary held-out; GV + structural-distance check).
3. `market/sabr.py` + `pricing/sabr_iv.py` + `test_sabr.py` (secondary; GV + structural-distance).
4. `market/garch.py` + `pricing/garch_pricer.py` + `test_garch.py` (secondary; GV + structural-distance).
5. `evaluation/generalization.py` (frozen-agent zero-shot eval) + `evaluation/ensemble.py` + `test_generalization.py`, `test_ensemble.py`.
6. `cli/heldout_eval.py`, `cli/generalization_report.py`: run the full seed ensemble on every held-out family; emit the in-family vs out-of-family tables and the structural-distance exhibit.

**Process instructions for Claude Code:**
- **Frozen agent, zero gradient steps on held-out data.** Assert it in code and test.
- Reuse the shared env/eval/metric harness and the Phase 5 baselines unchanged; held-out generators are drop-in `SurfaceGenerator`s.
- Validate each held-out generator (GV) and confirm it is OOD via the structural-distance diagnostic **before** evaluating the agent on it.
- Evaluate the **full seed ensemble**; report the distribution and dispersion; never report a single seed.
- Pre-declare the graceful-degradation tolerance before inspecting results.
- `make check` green (mypy strict, ruff) after each module; commit at module granularity referencing the gate that passed.
- This phase overlaps Phase 5; it does not depend on Phase 5 fine-tuning, only on the Phase 4 ensemble (evaluate both the Phase 4 and the Phase 5 agents if available).
- **Do not start Phase 7 (VIPER distillation) or Phase 8 (robustness battery/ablations).** If a task drifts there, stop and confirm.
- End with a report: per-family generator-validation and structural-distance results (proving each is genuinely held out), the in-family vs out-of-family tail-adjusted metric tables across the seed ensemble with dispersion, an explicit verdict on whether the CVaR tail advantage survived each different tail mechanism, and a statement of whether the generalization gate passed and the project may proceed to Phase 7.

---

### Ready-to-paste kickoff prompt

> Implement Phase 6 of the project as described in `PHASE6_BRIEF.md`, building on the completed Phase 4 (and Phase 5) repo. Build the structural-distance diagnostic first, then three held-out `SurfaceGenerator`s the agent never trained on, in priority order: rough Bergomi (primary; non-Markovian, MC-priced surface with a power-law short-maturity skew), SABR (Hagan, coherent multi-maturity surface), and GJR-GARCH (Duan risk-neutral pricing). Validate each generator's structural signature and confirm via the structural-distance diagnostic that it is out-of-distribution versus the Heston+Bates training priors, rough Bergomi most strongly. Then evaluate the FROZEN Phase 4 (and Phase 5) seed ensemble zero-shot on each held-out family through the shared evaluator, asserting zero gradient steps occur. Pre-declare the graceful-degradation tolerance. The gate is that out-of-family the agent still beats the always-on structures and the VRP heuristic on tail-adjusted metrics, the CVaR > risk-neutral > PPO tail ranking is preserved, the drop stays within tolerance, and crucially the CVaR agent's tail advantage survives the different tail-generating mechanisms (rough vol, GJR asymmetry) it never trained on. Report the full seed ensemble distribution, never a single seed; price each surface once per day; reuse the harness and baselines unchanged. Keep mypy strict and ruff green, and end by stating whether the generalization gate passed and the project may proceed to Phase 7.
