# Phase 8 Implementation Brief for Claude Code
## SPX Options Spread-Selection RL — Robustness Battery, Ablations & Write-Up

> **Scope of this phase.** Close every "is this real?" loop the earlier phases opened: re-confirm the **final** agent finds no edge on Wave 0, **ablate** each reward term and the algorithm itself to prove each earns its place, sweep **CVaR alpha** and **cost** to map the risk/return and cost-robustness frontiers, run **full SHAP** attribution to name the feature drivers, report **everything as ensemble mean +/- dispersion**, state **limitations and negative results** plainly, and assemble the **thesis exhibits** mapped to their sections. This is the phase that separates an A thesis from a demo. The composable, weighted reward from Phase 1 makes most ablations a **config change, not a code change**, which is why it was built that way.
>
> **Depends on:** Phases 1–7 complete. The primary CVaR agent is trained, real-data evaluated (5), generalization-tested (6), and distilled into a regime map (7).

---

## 0. Design principles specific to this phase

1. **Every claim is an ensemble claim.** No headline number is a single seed. Report mean and dispersion across seeds (and folds for real data). Optionally ship the ensemble policy as the deployable artifact.
2. **An ablation that doesn't move is a finding, not a non-event.** If dropping the CVaR term, the differential Sharpe, or margin normalization changes nothing, that term is inert (redundant or mis-weighted), and you investigate rather than quietly keep it. Each term must earn its place with an **interpretable degradation** when removed.
3. **Label synthetic vs real on every exhibit.** Never present synthetic results as real-market evidence. Lead real claims with real data; use synthetic for mechanism and stress.
4. **The cost frontier is the honest summary.** Report the **break-even cost multiple** at which the edge disappears. A result surviving 2x quoted spread is credible; one dying at 1.1x is fragile, and say so.
5. **Pre-register the primary claims; deflate for multiple testing.** This phase runs many sweeps. Pre-commit which comparisons are the headline (CVaR vs PPO on tail-adjusted OOS metrics) and apply deflated/probabilistic significance so the breadth of testing does not manufacture a false positive.
6. **Negative results and limitations are part of the contribution.** Regimes where the agent lost, structures it never used, the EOD/single-underlying/synthetic-realism limits, the static-CVaR time-inconsistency, the lossy map. State them plainly.

---

## 1. Repository additions

```
optspread/
  robustness/
    final_no_edge.py      # re-run the Wave-0 no-edge gate with the FINAL trained agent (leak check)
    ablations.py          # reward-component + algorithm ablations (retrain variants via reward weight configs)
    alpha_sweep.py        # CVaR alpha frontier: tail metric vs mean across alpha in {0.01..1.0}
    capacity_sweep.py     # network-capacity sensitivity (modest capacity should suffice OOS)
    cost_sensitivity.py   # slippage-multiplier sweep; report break-even cost
    seed_ensemble.py      # aggregate across seeds/folds; dispersion; optional ensemble policy
  attribution/
    shap_analysis.py      # global + local SHAP on policy/critic; feature-driver ranking (+ permutation cross-check)
  reporting/
    exhibits.py           # assemble all thesis figures/tables
    manifest.py           # map each exhibit -> thesis section
    limitations.py        # codified limitations + negative-result collection
  cli/
    run_robustness.py
    build_exhibits.py
tests/
  test_ablations.py        # weight=0 disables a term; ablation variants differ as expected on a toy
  test_alpha_sweep.py      # monotonicity: more risk-averse alpha -> better tail metric
  test_cost_sensitivity.py # break-even logic correct
  test_seed_ensemble.py    # dispersion aggregation correct; no single-seed leakage into headline
  test_shap_analysis.py    # SHAP attributions sane on a constructed policy; multicollinearity flagged
```

---

## 2. The robustness battery (build and run each)

**(1) Final-agent Wave-0 no-edge.** Re-run the Phase 2 no-edge gate with the **fully trained final agent**. It must still be FLAT-dominant on fair-IV Wave 0. This catches any leak introduced anywhere across the long pipeline. A failure here invalidates downstream results, so run it first.

**(2) Reward-component ablations.** Retrain variants with each term zeroed (config-only, thanks to the composable reward), evaluate via the shared suite:
- drop **CVaR penalty** -> tails should worsen (ES/CVaR degrade);
- drop **differential Sharpe** -> risk-adjusted performance degrades, returns more volatile;
- drop **Sortino** (if used) -> downside control degrades;
- drop **margin normalization** -> the agent optimizes raw dollars not return-on-capital, distorting structure choice (e.g. over-sizing undefined-risk).
Each must produce an **interpretable degradation**. Document any term whose removal is inert.

**(3) Algorithm ablation (the headline, formalized).** Aggregate **PPO (expected value) vs risk-neutral distributional vs CVaR distributional**, plus the **isolation ablation** (scalar CVaR penalty off, only native distributional tail control), across all waves and the real walk-forward. This is the central Robustness/Results exhibit: the CVaR agent wins on tail-adjusted metrics, and the native distributional objective carries the tail control on its own.

**(4) CVaR alpha frontier.** Sweep alpha in {0.01, 0.05, 0.1, 0.25, 0.5, 1.0}. Trace tail metric (CVaR/ES, max drawdown) vs mean return: more risk-averse alpha should buy a better tail at the cost of some mean, smoothly. This figure proves the risk lever works exactly as designed and lets an investor pick their point on the frontier.

**(5) Capacity sweep.** Vary network width/depth. Modest capacity should suffice OOS; larger capacity should not improve (and may overfit), confirming the capacity-as-overfitting-control choice.

**(6) Cost-sensitivity.** Sweep the slippage multiplier (0.5x, 1x, 1.5x, 2x, 3x quoted spread). Report the **break-even cost** at which the edge vanishes. The headline result should survive a stated multiple (target >= 1.5–2x).

**(7) Full SHAP attribution.** Global SHAP (feature-importance ranking of the regime features driving structure choice and the CVaR value) and local SHAP (per-regime explanations). Cross-check against permutation importance because **SHAP can be unstable under correlated regime features** (flag multicollinearity). The drivers must match the Phase 7 economic story (VRP/IV-rank -> credit selling; jump proxy -> tail defense).

**(8) Seed ensemble + dispersion everywhere.** Aggregate every headline metric across seeds and folds; report dispersion; never a hero run.

---

## 3. Validation gates (the definition of done)

1. Final-agent **Wave-0 no-edge holds** (no leak across the pipeline).
2. Each **reward ablation** produces an interpretable degradation; every term earns its place or is documented; inert terms investigated.
3. The **algorithm ablation** is aggregated across waves + real data; CVaR wins on tail-adjusted metrics; the isolation ablation holds.
4. The **CVaR alpha frontier** is sensible and (near-)monotone (more averse -> better tail, lower mean).
5. **Capacity** sweep confirms modest capacity suffices OOS.
6. **Cost break-even** reported; headline survives the stated multiple.
7. **SHAP** drivers named and consistent with the Phase 7 map; multicollinearity caveated.
8. **Ensemble dispersion** reported on every headline number; no single-seed claims.
9. **Limitations and negative results** stated plainly (EOD-only, single underlying SPX, synthetic-realism cap, few real tail events, static-CVaR time-inconsistency, lossy map).
10. **Exhibit manifest** maps every figure/table to a thesis section (Results, Robustness & Diagnostics, Discussion, Conclusion).

---

## 4. Pitfalls to engineer against

- **Best-seed reporting.** Ensemble + dispersion always; the test asserts no single seed leaks into a headline.
- **Synthetic-as-real conflation.** Label every exhibit; lead real claims with real data.
- **Omitting negative results.** Collect and report them; a clean-only narrative is a red flag to a committee.
- **Inert ablations ignored.** If removing a term does nothing, investigate (redundant or mis-weighted), do not keep it silently.
- **Rosy costs.** The break-even cost is the honest summary; do not report only the 1x result.
- **SHAP on correlated features.** Regime features are correlated; SHAP attributions can be unstable. Cross-check with permutation importance and caveat.
- **Multiple-testing inflation.** Many sweeps -> deflate the headline significance and pre-register the primary claims.
- **Treating a lossy distilled map as the policy.** Carry the Phase 7 fidelity number into the write-up; the map summarizes, it does not equal, the policy.

---

## 5. Build order and process

**Build order (do not reorder):**
1. `robustness/final_no_edge.py` (reuse the Phase 2 gate) -> confirm no leak with the final agent.
2. `robustness/ablations.py` + `test_ablations.py` (reward-component and algorithm ablations; retrain variants via weight configs).
3. `robustness/alpha_sweep.py`, `capacity_sweep.py`, `cost_sensitivity.py` + their tests.
4. `robustness/seed_ensemble.py` + `test_seed_ensemble.py` (aggregation used by everything above).
5. `attribution/shap_analysis.py` + `test_shap_analysis.py`.
6. `reporting/exhibits.py`, `manifest.py`, `limitations.py`.
7. `cli/run_robustness.py`, `cli/build_exhibits.py`: run the full battery, assemble exhibits, emit the manifest.

**Process instructions for Claude Code:**
- Use the **composable reward weight configs** for ablations (config change, not code change); confirm `weight=0` disables a term (test).
- **Pre-register the primary claims** (CVaR vs PPO tail-adjusted OOS) and **deflate** significance for the breadth of sweeps.
- **Label synthetic vs real** on every exhibit; lead real claims with real data.
- **Ensemble + dispersion on every headline**; the seed-ensemble test guards against single-seed leakage.
- Cross-check SHAP with permutation importance; flag multicollinearity.
- Collect **negative results** and codify **limitations** as first-class outputs.
- `make check` green (mypy strict, ruff) after each module; commit at module granularity referencing the gate that passed.
- This is the final build phase: end by emitting the **complete exhibit set mapped to thesis sections**, the ablation/sweep/cost/SHAP results with dispersion, the stated limitations and negative results, and a statement that the robustness battery is complete and the thesis is ready to write.

---

### Ready-to-paste kickoff prompt

> Implement Phase 8 of the project as described in `PHASE8_BRIEF.md`, building on the completed Phases 1–7 repo. First re-run the Wave-0 no-edge gate with the final trained agent to confirm no leak entered the pipeline. Then run the robustness battery: reward-component ablations (drop CVaR, differential Sharpe, margin normalization, each via reward weight configs, each must degrade interpretably), the formalized algorithm ablation (PPO vs risk-neutral vs CVaR plus the scalar-penalty-off isolation, aggregated across all waves and the real walk-forward), the CVaR alpha frontier (more risk-averse should buy a better tail for less mean, near-monotone), a capacity sweep (modest capacity should suffice OOS), and a cost-sensitivity sweep reporting the break-even cost multiple. Run full SHAP attribution (global + local) cross-checked with permutation importance and flag multicollinearity; the drivers must match the Phase 7 economic story. Report every headline number as ensemble mean and dispersion across seeds and folds, never a single seed. Pre-register the primary claims and deflate significance for the breadth of testing. Label every exhibit synthetic vs real, lead real claims with real data, and codify limitations (EOD-only, single-underlying SPX, synthetic-realism cap, few real tail events, static-CVaR time-inconsistency, lossy map) and negative results as first-class outputs. Assemble all figures/tables and map each to its thesis section via an exhibit manifest. Keep mypy strict and ruff green, and end by stating that the robustness battery is complete and the thesis is ready to write.
