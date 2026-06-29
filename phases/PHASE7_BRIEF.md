# Phase 7 Implementation Brief for Claude Code
## SPX Options Spread-Selection RL — Interpretability: the Investor-Facing Regime→Structure Map

> **Scope of this phase.** Extract the human-readable, investor-usable artifact the whole thesis was designed to produce: a **regime→structure map** ("when the market looks like X, take spread Y") together with, for each regime, the **full return distribution** of the recommended structure straight from the distributional critic. This is done **post-hoc** on the frozen primary agent (the Phase 4/5 CVaR/IQN agent), by rolling it out across the regime space, **clustering** regime features into archetypes, **distilling** the policy into a shallow decision tree via **VIPER**, and reading **per-regime return distributions** off the critic. The gate is **economic sensibility**: the distilled rules must reproduce the policy faithfully **and** make economic sense. Nonsense rules with good PnL are a reward-hacking alarm, not a result to ship.
>
> **Depends on:** Phases 4–6. The primary agent is trained, real-data evaluated (Phase 5), and shown to generalize out-of-family (Phase 6). The distributional critic is available for per-regime distributions.

---

## 0. The closed-form tension, resolved (read this first)

The original design goal was a policy that is **not** a closed-form rule, yet an interpretable takeaway investors can use. These only conflict if you confuse the **policy** with the **takeaway**:

- The **policy** is a rich nonlinear function over the full regime state. It is *allowed and expected* to be complex (it captures interactions across VRP, IV rank, skew, jump risk, term structure that no single formula expresses).
- The **takeaway** is a **post-hoc, deliberately lossy summary** of that policy: a shallow tree and a low-dimensional map. Its **fidelity** is measured and reported, not assumed to be perfect.

So a shallow tree that does **not** perfectly reproduce the policy is not a failure, it is the evidence that the agent learned something richer than a formula, while the map still gives investors an actionable approximation. This framing is the philosophical contribution; state it explicitly in the thesis. Report fidelity honestly: high fidelity means the structure is genuinely simple; moderate fidelity means the map is a useful summary of a policy that does more.

---

## 1. Design principles specific to this phase

1. **Distill the registered primary, frozen.** Distill the Phase 4/5 **CVaR/IQN** agent (the pre-committed primary). No retraining. PPO and the risk-neutral agent may be distilled for comparison, but the headline artifact is the CVaR agent's.
2. **Cover the regime space, not just the trajectory.** On-policy rollout only visits states the policy leads to, which undersamples the map. Use **VIPER's Q-DAGGER resampling** (the student's own distribution) for distillation **fidelity**, and additionally **broadly sample the regime-feature space** (Latin hypercube / grid / DR draws) for **map coverage**, so the map describes regimes an investor could actually face.
3. **Weight distillation by what the agent values.** VIPER resamples states by how much the action choice matters in the oracle's value. For the distributional agent, "value" is the **CVaR functional**, so weight by the **CVaR-value gap** between the best and second-best action. States where choosing wrong costs the most tail get the most attention, which is exactly where the map must be right.
4. **Keep the tree human-readable.** Depth <= 4–6 (simulatable). A depth-20 tree is "explainable" but not interpretable; resist the urge to grow the tree to chase fidelity. If shallow fidelity is low, report it as a finding, do not bury it in a deep tree.
5. **Per-regime distributions are the unique payoff.** The distributional critic gives, per regime, the **full return distribution** of each structure (median, spread, CVaR-5), not just an expected value. "Sell the condor here, and here is its return distribution including a CVaR-5 of −Y%" is the takeaway single-feature heuristics cannot produce. This is what the distributional architecture was *for*.
6. **Economic sensibility is the gate.** Codify the checks. Sensible rules + the surfacing of a nontrivial **interaction** the naive heuristics miss is the win. Nonsense rules trigger a reward-hacking investigation (cross-checked against Phase 6 held-out and Phase 8 ablations), not a paper-over.

---

## 2. Repository additions

```
optspread/
  interpret/
    rollout_logger.py     # roll frozen policy across regime draws; log (features, action, return_dist, cvar_gap)
    coverage_sampler.py   # broad regime-space sampling (LHS/grid/DR draws) for map coverage
    attribution.py        # lightweight feature-importance to choose the map's dominant axes (full SHAP = Phase 8)
    clustering.py         # cluster regime features into archetypes; k via silhouette/BIC; centroid descriptions
    viper.py              # Q-DAGGER distillation -> depth-bounded decision tree; CVaR-gap state weighting
    fidelity.py           # action-agreement + CVaR value-regret of tree vs neural policy
    regime_map.py         # readable 2D/3D map over dominant axes; modal structure + conditional return per cell
    per_regime_dist.py    # per-regime/per-cell return distributions from the distributional critic
    economic_checks.py    # codified economic-sensibility checks on the distilled rules
  cli/
    distill.py
    build_regime_map.py
    interpret_report.py
tests/
  test_viper.py           # distills a KNOWN synthetic policy to high fidelity; depth bound respected
  test_fidelity.py        # agreement + value-regret metrics correct on constructed cases
  test_clustering.py      # stable assignments; silhouette/BIC selection sane
  test_regime_map.py      # cells populated; modal action + conditional return correct
  test_economic_checks.py # checks fire correctly on sensible vs nonsense rule sets
```

---

## 3. The pipeline (build in this order)

**(1) Rollout + coverage dataset.** Roll the frozen CVaR agent across: all synthetic waves under domain randomization, and the real OOS walk-forward states (Phase 5). For each state log the regime-feature vector, the chosen structure, the distributional critic's **return distribution** for that action, and the **CVaR-gap** to the runner-up action. Add **broad coverage samples** of the feature space so sparsely-visited regimes still appear on the map.

**(2) Attribution -> map axes.** A lightweight feature-importance pass (permutation importance on the action choice, or a quick surrogate) identifies the **dominant regime axes** driving structure choice (expected: VRP / IV rank, jump proxy, term-structure slope, skew). These become the axes of the readable map. (The full SHAP treatment is Phase 8; here you only need to pick axes.)

**(3) Clustering -> archetypes.** Cluster the standardized regime features (k-means or GMM; choose k by silhouette/BIC) into market archetypes (e.g. calm-high-VRP, stressed-high-jump, trending). Describe each by its centroid in interpretable feature terms and its **modal recommended structure**. Treat clusters as **descriptive archetypes, not causal claims**.

**(4) VIPER distillation -> shallow tree.** Q-DAGGER: iteratively roll the current tree, query the neural policy for correct actions, resample states weighted by the **CVaR-gap**, refit a depth-<=6 decision tree on regime features. The tree's splits are regime features, so leaves read as economic rules ("VRP high AND jump-proxy low -> iron condor"). Validate on a **known synthetic policy** first (it must distill to high fidelity) before trusting it on the real agent.

**(5) Fidelity.** Report **action-agreement** (tree vs neural policy) on the student distribution and on the broad-coverage sample, and **CVaR value-regret** (how much tail the tree gives up vs the neural policy). Frame the map's fidelity honestly per section 0.

**(6) Regime map + per-regime distributions.** Build the readable 2D/3D map over the dominant axes; each cell holds the **modal recommended structure** and its **conditional risk-adjusted return**. For each archetype/cell, extract the **per-regime return distribution** from the distributional critic for the recommended structure and key alternatives. This map + distributions is the deliverable.

**(7) Economic-sensibility checks.** Codify and run them (section 4).

---

## 4. The economic-sensibility gate (the definition of done)

The distilled rules must satisfy codified checks and surface a nontrivial interaction:

- **Directional sanity:** high VRP / high IV rank -> sell premium (credit structures); elevated jump proxy -> buy wings / defined risk / flat; benign trend regime -> directional structures appear where appropriate.
- **The interesting interaction:** the agent should reveal at least one **conditional rule the naive single-feature heuristics miss**, e.g. "high IV rank says sell vol, **except** when term structure is backwardated **and** the jump proxy is elevated, where defined-risk or flat dominates." This interaction is the thesis's economic contribution, the thing the Phase 5 naive-VRP baseline cannot capture.
- **Consistency with the held-out and ablation evidence:** the rules should be consistent with the Phase 6 generalization behavior and not depend on a single sim quirk.

**If the rules are economic nonsense despite good PnL: STOP and investigate.** Either the agent reward-hacked a simulator/cost artifact (cross-check Phase 6 held-out collapse and Phase 8 ablations) or the distillation is too lossy (fidelity too low to trust the tree). Do not present nonsense rules as findings.

**Pass conditions:** high-enough action-agreement to call the tree a faithful-summary (report the number), economically sensible rules, at least one surfaced interaction, and per-regime distributions extracted, across the seed ensemble with dispersion.

---

## 5. Pitfalls to engineer against

- **Deep-but-uninterpretable trees.** Keep depth <= 4–6. Simulatability is the point; a deep tree defeats it.
- **Treating a low-fidelity tree as the policy.** Always report fidelity; frame the map as a lossy approximation (section 0). Low shallow-fidelity is itself a (publishable) finding about policy complexity.
- **Over-claiming causality from clusters.** Clusters are descriptive archetypes. Use careful language; do not assert the regime *causes* the optimal structure.
- **On-policy-only coverage.** Add broad coverage sampling, or the map has holes in regimes the policy avoids visiting but investors will face.
- **Seed cherry-picking.** Distill across the ensemble (or a representative seed) and report fidelity/rule dispersion; do not pick the prettiest tree.
- **Nonsense rules with good PnL.** A reward-hacking alarm. Investigate against Phase 6/8 evidence; never paper over.
- **Distilling action while ignoring value.** For the map you need the action; for fidelity you also need CVaR value-regret, since two policies can agree on most actions but differ exactly where the tail matters.

---

## 6. Build order and process

**Build order (do not reorder):**
1. `interpret/rollout_logger.py` + `coverage_sampler.py`.
2. `interpret/attribution.py` (pick dominant map axes) + `interpret/clustering.py` + `test_clustering.py`.
3. `interpret/viper.py` + `interpret/fidelity.py` + `test_viper.py` (validate on a known synthetic policy first), `test_fidelity.py`.
4. `interpret/regime_map.py` + `interpret/per_regime_dist.py` + `test_regime_map.py`.
5. `interpret/economic_checks.py` + `test_economic_checks.py`.
6. `cli/distill.py`, `cli/build_regime_map.py`, `cli/interpret_report.py`: distill the primary agent, build the map, extract per-regime distributions, run the economic checks, assemble the deliverable.

**Process instructions for Claude Code:**
- Distill the **frozen registered primary** (CVaR/IQN) agent; no retraining. Distill PPO/risk-neutral only as comparisons.
- Validate the VIPER implementation on a **known synthetic policy** (must reach high fidelity within the depth bound) before trusting it on the real agent.
- Use **CVaR-gap weighting** in Q-DAGGER and **report CVaR value-regret**, not just action-agreement.
- Keep tree depth <= 6; **report fidelity honestly** and frame the map as a lossy summary of a richer policy.
- Add **broad coverage sampling** so the map spans the regime space; treat clusters as descriptive, not causal.
- Run the **economic-sensibility checks as a gate**; nonsense rules -> investigate reward hacking against Phase 6/8, do not ship.
- Reuse the distributional critic for **per-regime return distributions** (the unique investor payoff).
- `make check` green (mypy strict, ruff) after each module; commit at module granularity referencing the gate that passed.
- **Do not start Phase 8 (full robustness battery / ablations / SHAP).** The lightweight attribution here is only to choose map axes. If a task drifts into the full battery, stop and confirm.
- End with the **deliverable** (the regime→structure map, the shallow decision tree, the archetype table, and per-regime return distributions, with a plain-language statement of the economic logic and at least one surfaced interaction) plus a report of distillation fidelity and economic-check outcomes across the ensemble, and a statement of whether the economic-sensibility gate passed and the project may proceed to Phase 8.

---

### Ready-to-paste kickoff prompt

> Implement Phase 7 of the project as described in `PHASE7_BRIEF.md`, building on the completed Phases 4–6 repo. Work on the frozen primary CVaR/IQN agent, no retraining. Roll it out across synthetic (all waves, domain-randomized) and real OOS states, and add broad coverage sampling of the regime-feature space for map completeness. Pick the dominant map axes via lightweight feature importance, cluster regime features into archetypes (k by silhouette/BIC, treated as descriptive not causal), and distill the policy into a depth<=6 decision tree via VIPER Q-DAGGER weighted by the CVaR-gap. Validate the VIPER implementation on a known synthetic policy first. Report fidelity as both action-agreement and CVaR value-regret, and frame the map explicitly as a lossy human-readable summary of a deliberately richer (non-closed-form) policy. Build the readable regime→structure map over the dominant axes with the modal recommended structure and conditional return per cell, and extract per-regime return distributions from the distributional critic. The gate is economic sensibility: codified checks (sell-vol-when-VRP-high, defend-when-jump-high) must pass and the agent must surface at least one nontrivial interaction the naive single-feature heuristics miss; if rules are nonsense despite good PnL, stop and investigate reward hacking against the Phase 6 held-out and Phase 8 ablations rather than shipping. Report across the seed ensemble with dispersion. Keep mypy strict and ruff green, and end with the deliverable map plus distributions and a statement of whether the economic-sensibility gate passed and the project may proceed to Phase 8.
