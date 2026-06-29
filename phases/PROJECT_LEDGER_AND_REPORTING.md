# PROJECT LEDGER & RESULTS-REPORTING PROTOCOL
## SPX Options Spread-Selection RL — Thesis Build Tracker and Reporting Spec

This is the front-door document for the eight-phase build (`PHASE1_BRIEF.md` ... `PHASE8_BRIEF.md`, guardrails in `CLAUDE.md`). It does two things:
1. **Tracks** the whole project in one place (the snapshot + per-phase ledger).
2. **Specifies the exact format to report results back to me** so I can assess each gate and recommend the next best step. Fill the relevant blocks as phases complete and paste them back. The schemas are structured on purpose: paste them as-is.

> **Send incrementally.** You do not wait until the end. After each phase (or a hard-stop trigger mid-phase), fill that phase's block, update the Global Snapshot, add a Decision Request, and paste all three. That is one "report."

---

## 1. The reporting protocol (what to send me, every time)

Each report you paste back contains **exactly three things, in this order**:

1. **The Global Snapshot** (Section 2) — the one-screen status of all eight phases. Always include it so I have context.
2. **The completed phase block(s)** since your last report (Section 4) — the structured results for whatever finished.
3. **A Decision Request** (Section 5) — what you want me to weigh in on.

**Formatting rules (so I can parse and trust the numbers):**
- Keep the field labels exactly as written in the schemas. Fill values after the colon.
- **Always label synthetic vs real** on every metric.
- **Always include seed/fold dispersion** (mean and a spread: std, IQR, or min–max). A bare number with no dispersion, I will treat as provisional.
- **Use the metric dictionary in Section 6** so a number like "Sharpe 1.2" is unambiguous.
- **Flag any hard-stop (Section 7) immediately**, even mid-phase, even if nothing else is ready.
- Attach raw artifacts only if you want deep analysis: a metrics CSV, a learning-curve plot, the distilled tree, the regime-map figure. The structured block is enough for a go/no-go; artifacts are for when something looks off.

**What I do with each report:** confirm the gate genuinely passed (not just "numbers exist"), sanity-check the magnitudes against what the economics should produce, catch silent failure modes (leakage, reward-hacking, single-seed shimmer, sim-as-real conflation), then recommend **proceed / iterate / pivot** and pre-empt the next phase's specific risks.

---

## 2. Global Snapshot schema (top of every report)

```
=== GLOBAL SNAPSHOT (as of YYYY-MM-DD) ===
Primary agent: CVaR/IQN   |   Repo make check: PASS/FAIL   |   Seeds run: N

Phase 1 Env/pricing/reward      : NOT_STARTED | IN_PROGRESS | PASSED | PASSED_W_CAVEATS | FAILED | BLOCKED
Phase 2 PPO + no-edge gate      : ...
Phase 3 Distributional + CVaR   : ...
Phase 4 Curriculum W1..W6       : ... (note furthest wave passed, e.g. "W3 PASSED, W4 IN_PROGRESS")
Phase 5 Real fine-tune+walkfwd  : ...
Phase 6 Held-out generalization : ...
Phase 7 Distillation/regime map : ...
Phase 8 Robustness/ablations    : ...

Hard-stops triggered since last report: NONE | <list>
Single biggest open question right now : <one line>
```

---

## 3. Per-phase ledger (your running notes; not all of this is pasted to me)

Keep a short freeform log per phase here for your own tracking (dates, decisions, dead-ends). When you report, you distill it into the structured block in Section 4. This section is yours; Section 4 is mine.

---

## 4. Per-phase report blocks (fill and paste these)

### Phase 1 — Environment, pricing, reward
```
[PHASE 1]
Status: PASSED | PASSED_W_CAVEATS | FAILED | BLOCKED
Acceptance tests passing: __/__ (list any failing)
Closed-form oracle checks (BS golden, payoff per template, parity): PASS/FAIL
P&L checks (credit-up-front, round-trip=-2c, MTM): PASS/FAIL
Margin checks (defined=width-credit, undefined>>defined, flat=0): PASS/FAIL
Cost scaling (legs, width, moneyness): PASS/FAIL
Reward checks (diff-Sharpe converges, denom guard, weight=0 ablates): PASS/FAIL
Env: check_env PASS/FAIL | determinism (same seed=same traj) PASS/FAIL | no-lookahead PASS/FAIL
Action library final count: __ (templates x buckets + flat)
Smoke (Wave-0 scripted): always-on no-cost mean PnL = __ [CI __], with-cost mean PnL = __
Deviations/surprises: <...>
Blockers/questions: <...>
```

### Phase 2 — PPO baseline + Wave-0 no-edge gate
```
[PHASE 2]
Status: ...
PPO stability: entropy collapse? Y/N | final entropy __ | explained variance __ | approx KL __
NO-EDGE GATE (a) risk-adjusted reward: FLAT frequency = __ (target >=0.80) | credit-structure bias? Y/N
  mean eval PnL no-cost CI = [__,__] (should include ~0) | with-cost mean = __ (should be <0)
NO-EDGE GATE (b) pure-PnL ablation: action distribution ~uniform? Y/N | any structure systematically +PnL? Y/N
Seeds: __ | dispersion of FLAT freq: __ (mean +/- spread)
Shared harness reuse-ready (Evaluator/MetricSuite/EnvFactory): Y/N
Verdict: gate PASSED / FAILED
Deviations/surprises: <...>
Blockers/questions: <...>
```

### Phase 3 — Distributional agent (QR-DQN -> IQN) + CVaR
```
[PHASE 3]
Status: ...
G1 quantile loss + risk-measure unit tests: PASS/FAIL
G2 distribution recovery (QR-DQN & IQN, quantiles & CVaR within tol): PASS/FAIL
G3 fat-tail BANDIT: mean-greedy picks tail arm? Y/N | CVaR_alpha avoids it? Y/N  [the in-vitro proof]
G4 fat-tail MDP: CVaR bootstrap avoids tail? Y/N | risk-neutral does not? Y/N
G5 Wave-0 no-edge (distributional): FLAT freq = __ (expect >= PPO's)
G6 compare.py null head-to-head on Wave-0 via shared suite: PASS/FAIL
Tail resolution: QR-DQN N = __ (floor(alpha*N) at alpha=0.05 = __) | IQN K = __
CVaR under-exploration mitigation used: behavior-decoupled | alpha-anneal | blended | (+ eps floor __)
Seeds: __ | dispersion: <...>
Deviations/surprises: <...>
Blockers/questions: <...>
```

### Phase 4 — Synthetic curriculum (report per wave as it passes)
```
[PHASE 4 / WAVE _]
Status: ...
Pre-registered prediction (committed BEFORE training): <paste the registered prediction>
GENERATOR VALIDATION (sim produces the stylized fact, no agent): PASS/FAIL | key stat: <...>
  (W2: rho<0 skew? IV rank varies?  W3: excess kurtosis>0? jump proxy moves?  W5: regimes separable in features?)
BEHAVIORAL VALIDATION (pre-registered test on trained agent): PASS/FAIL
  key statistic + threshold: <e.g. corr(credit, VRP)=__ (>0 reqd); or corr(defined-risk, jump-proxy)=__>
FORGETTING CHECK (re-eval waves 0..i-1 within tol): PASS/FAIL
Tail-adjusted metrics this wave (SYNTHETIC), mean +/- dispersion across __ seeds:
  | agent        | mean ret | Sharpe | Sortino | CVaR95(ES) | maxDD | tail ratio |
  | PPO          |          |        |         |            |       |            |
  | RN-distrib   |          |        |         |            |       |            |
  | CVaR-distrib |          |        |         |            |       |            |
[WAVE 3 ONLY] Headline: CVaR ES better than PPO by __ (material? Y/N) | isolation ablation (scalar penalty off) holds? Y/N
[WAVE 5 ONLY] Regime-conditioning: ARI(behavior clusters vs hidden regime) = __ (regime never observed) | framestack k=__
Deviations/surprises: <...>
Blockers/questions: <...>
```

### Phase 5 — Real-data fine-tune + walk-forward
```
[PHASE 5]
Status: ...
Data integrity: surface-vs-raw IV reconciled? Y/N | no-lookahead test PASS/FAIL | RealDataReplay drop-in (env unchanged) PASS/FAIL
Walk-forward: folds = __ | purge+embargo applied? Y/N | episodes cross folds? (must be N)
ZERO-SHOT (Phase 4 agent, no fine-tune) on REAL OOS, vs battery, tail-adjusted, mean +/- dispersion:
  | strategy            | mean | Sharpe | Sortino | CVaR95 | maxDD | tail | DSR |
  | CVaR agent (0-shot) |      |        |         |        |       |      |     |
  | naive VRP heuristic |      |        |         |        |       |      |     |
  | always-on CNDR      |      |        |         |        |       |      |     |
  | PUT / BXM           |      |        |         |        |       |      |     |
  | buy-hold SPX        |      |        |         |        |       |      |     |
FINE-TUNED (light) on REAL OOS, same table: <paste or "similar to zero-shot, deltas: ...">
GATE: CVaR agent beats naive-VRP and always-on on tail-adjusted OOS? Y/N | DSR survives multiple-testing? Y/N
Cost stress: edge survives __x quoted spread (break-even at __x)
SIM-TO-REAL GAP: real features in-prior or OOD? __ | failures cluster in (2008/2018/2020)? __ | gap mostly (shift/cost/regime)? __
Verdict: PASSED on tail-adjusted | FAILED -> consider gap-study pivot
Deviations/surprises: <...>
Blockers/questions: <...>
```

### Phase 6 — Held-out generalization
```
[PHASE 6]
Status: ...
Held-out families built + generator-validated: rough Bergomi __ | SABR __ | GARCH __
Structural-distance (OOD vs Heston+Bates training priors): rBergomi __ | SABR __ | GARCH __  (rBergomi most distant? Y/N)
Frozen agent, zero gradient steps asserted: Y/N
ZERO-SHOT out-of-family, tail-adjusted, full seed ensemble (mean +/- dispersion):
  | family      | CVaR agent CVaR95 | vs in-family drop | still beats baselines? | ranking preserved? |
  | rBergomi    |                   |                   |                        |                    |
  | SABR        |                   |                   |                        |                    |
  | GARCH       |                   |                   |                        |                    |
CVaR tail advantage survives the DIFFERENT tail mechanism (rough vol / GJR)? Y/N  [the decisive check]
Graceful degradation within pre-declared tolerance? Y/N
Deviations/surprises: <...>
Blockers/questions: <...>
```

### Phase 7 — Distillation / regime map
```
[PHASE 7]
Status: ...
VIPER validated on known synthetic policy (high fidelity within depth bound)? Y/N
Tree depth: __ (<=6) | action-agreement fidelity: __% | CVaR value-regret: __
Clusters / map axes chosen: <e.g. IV-rank x jump-proxy> | k = __ (silhouette/BIC)
ECONOMIC-SENSIBILITY GATE: directional sanity PASS/FAIL | surfaced interaction the naive heuristics miss? Y/N
  the interaction (one line): <e.g. "high IV rank -> sell vol, EXCEPT backwardated term + high jump -> defined-risk/flat">
Per-regime return distributions extracted from critic? Y/N
Rules economically sensible (not nonsense-with-good-PnL)? Y/N  [if N -> reward-hacking investigation]
Fidelity framing stated (map is lossy summary of richer policy)? Y/N
Deviations/surprises: <...>
Blockers/questions: <...>
```

### Phase 8 — Robustness / ablations / write-up
```
[PHASE 8]
Status: ...
Final-agent Wave-0 no-edge re-confirmed (no leak in pipeline)? Y/N
Reward ablations (each must degrade interpretably), drop-> effect:
  drop CVaR -> tails: __ | drop diff-Sharpe -> __ | drop margin-norm -> __ | any INERT term? <which>
Algorithm ablation aggregated (waves+real): CVaR wins tail-adjusted? Y/N | isolation holds? Y/N
CVaR alpha frontier monotone (more averse -> better tail, less mean)? Y/N | points: <alpha: (mean, CVaR95) ...>
Capacity sweep: modest capacity sufficient OOS? Y/N
Cost break-even multiple: __x (target >=1.5-2x)
SHAP top drivers: <ranked features> | consistent with Phase 7 map? Y/N | multicollinearity flagged? Y/N
Ensemble dispersion reported on all headlines? Y/N
Limitations + negative results stated: <one line each>
Exhibit manifest (figure/table -> thesis section) complete? Y/N
Deviations/surprises: <...>
Blockers/questions: <...>
```

---

## 5. Decision-Request schema (end of every report)

```
=== DECISION REQUEST ===
Context (1-2 lines): <where you are, what just finished>
The decision I want from you: <proceed? iterate? pivot? which option?>
Options I'm weighing:
  A) <...>
  B) <...>
My current lean: <A/B and why, or "unsure">
Constraints: <time left, compute, committee deadline, anything binding>
Anything that looked off: <numbers that surprised you, even if the gate passed>
```

If you only want a sanity check and not a decision, say so; I will just confirm the gate and flag risks.

---

## 6. Metric dictionary (so numbers are unambiguous)

When you report a metric, it means this (note any deviation):
- **Mean return** — annualized mean of per-period returns, **net of costs**. State the period (daily/monthly) used to annualize.
- **Sharpe** — annualized, mean excess return / std; state rf (0 or T-bill) and the return frequency.
- **Sortino** — annualized; downside-deviation denominator, MAR = 0.
- **CVaR95 / ES** — expected return in the worst **5%** tail (alpha = 0.05) of the per-period return distribution; **report as a negative number** (an expected loss). State alpha if not 0.05.
- **Max drawdown (maxDD)** — peak-to-trough equity decline, as a fraction.
- **Tail ratio** — |95th-percentile gain| / |5th-percentile loss| (define if you use another convention).
- **Return-on-margin** — period PnL / average margin (buying power) used.
- **Turnover** — structure switches (or notional traded / capital) per period; state which.
- **DSR (Deflated Sharpe Ratio)** — probability the true Sharpe > 0 after correcting for the number of trials and non-normal returns (Lopez de Prado). Report the DSR or probabilistic-Sharpe value.
- **FLAT frequency** — fraction of decisions where the agent holds no position.
- **Fidelity** — tree-vs-neural action-agreement (%) and CVaR value-regret.
- **Break-even cost multiple** — the slippage multiplier at which the tail-adjusted advantage over the relevant baseline reaches zero.
- **Dispersion** — across seeds (synthetic) and across folds (real); report mean and a spread (std / IQR / min-max), and say which.

---

## 7. Hard-stop checklist (report immediately if ANY trips, even mid-phase)

```
[ ] Any agent makes money on fair-IV Wave 0  -> env leak/bug (debug Phase 1)
[ ] A generator fails its validation (no stylized fact) -> fix sim before training
[ ] A held-out family is NOT out-of-distribution -> not a valid test
[ ] Distilled rules are economic nonsense despite good PnL -> reward-hacking investigation
[ ] Real performance collapses despite clean synthetic -> consider gap-study pivot (don't overfit)
[ ] A reward-term ablation is inert -> investigate (redundant/mis-weighted)
[ ] make check red / a determinism or no-lookahead test fails -> stop, fix before reporting numbers
```

For any trip: paste the Global Snapshot, the relevant phase block (even partial), and a Decision Request labeled `HARD-STOP`. Do not work around it.

---

## 8. First report I expect from you

When Phase 1 (or your first milestone) is done, paste:
1. the **Global Snapshot**,
2. the **[PHASE 1]** block,
3. a **Decision Request**.

I will confirm the environment is genuinely leak-free and the reward is wired correctly (the foundation everything else rests on), flag anything that will bite later, and green-light Phase 2. From there we go phase by phase, and at each gate I will tell you whether to proceed, iterate, or pivot, and what to watch for next.
