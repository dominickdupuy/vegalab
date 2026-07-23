# Phase 4 Gate Report

Generated: 2026-06-28T22:10:00

Overall implementation status: **WAVE 1 COMPLETE (rich 35-dim obs) — GV_1, BV_1,
FF_1 all PASS** for the primary IQN/CVaR ensemble. **Wave 2 (Heston SV): GV_2 PASS,
BV_2 NOT yet passing for the primary agent — DECISION PENDING** (see the Wave-2
section). See the 2026-06-30 rich-observation section below; original 16-dim
results follow it.

## 2026-07-23 — C0 checkpoint diagnostics: distribution healthy; blindness-to-success
## reproduced; mean-gap signal ordering robust but its level drifts negative

**Motivation.** The model-candidate research (`MODEL_CANDIDATES_RESEARCH.md`, summarized in
`MODEL_CANDIDATES_BRIEF.md`) pre-registered two free checks on a trained Wave-2 IQN
checkpoint before any new training: (a) has the learned return distribution collapsed toward
its mean (a published Huber-quantile-loss risk that would make CVaR action selection
degenerate)? (b) does Q(best trade) − Q(FLAT) respond to the iv_rank entry signal under mean
vs CVaR scoring? New tools: `optspread/eval/checkpoint_diagnostics.py`,
`optspread/cli/diagnose_checkpoint.py`, `optspread/cli/train_resumable.py`.

**Setup.** One seed (901), gate recipe (IQN width 256, alpha 0.2 CVaR deploy, eps-floor
0.04, Wave 2, 150k steps) trained via the resumable loop; diagnostics at the 105k snapshot
and the final 150k checkpoint; 10 eval episodes (630 states) per environment per point;
zero gradient steps at evaluation. **Single-seed, preliminary — not an ensemble claim.**

| Statistic (Wave 2 edge sim) | @105k | @150k |
|---|---|---|
| (a) spread(q90−q10)/value-scale | 1.82 (alive) | **1.56 (alive)** |
| (b) mean-gap, high-signal quartile | **+0.0083** | **−0.0330** |
| (b) mean-gap, low-signal quartile | −0.0013 | −0.0413 |
| (b) signal ordering (hi − lo) | +0.0095 | **+0.0083** |
| (b) CVaR-gap, high-signal | −0.0485 | −0.0812 |
| trade-preferred states, mean / CVaR scoring | 30.5% / 10.2% | 11.3% / 3.5% |
| Wave-0 control: mean-gap (CVaR-gap) | −0.022 (−0.064) | −0.038 (−0.110) |

**Findings.**
1. **No quantile collapse** at either point — the distributional machinery is live; the
   published Huber-collapse pathology does not apply to this agent. Scoring-side candidates
   have a real target.
2. **Blindness-to-success reproduced in our own numbers at 105k:** mean values learned the
   conditional edge (positive gap exactly in the high-signal quartile, flat on the no-edge
   control), while CVaR-of-quantiles scoring suppressed it — ranking high-signal trades
   *worse* than low-signal (their tails are scarier). At 105k a mean-weighted spectral
   blend (weight ≳ 0.85 on the mean) would have traded conditionally AND stayed flat on
   Wave 0 from a scoring change alone.
3. **The mean-gap LEVEL drifts negative with continued training** (105k → 150k) while the
   signal ordering stays intact (+0.0095 → +0.0083). At 150k both components are negative,
   so no convex mean/CVaR blend trades — deployment-scoring alone is no longer sufficient
   at the full recipe length. This extends the earlier "more steps did not help"
   observation: on the mean-gap level, more steps actively hurt.
4. **Implication for the candidate ranking:** spectral/Mean-CVaR scoring (#1) remains
   necessary at deployment but is not sufficient alone; the pre-registered next experiment
   is #1 combined with alpha-annealed training (#2), with the falsifiable prediction that
   the mean-gap level stays positive in high-signal states when the behavior policy is not
   tail-averse during learning. Wave-0 flatness holds under every scoring at both points.

**Provenance.** Full numbers: `phases/phase4_c0_diagnostics_150k.json` (emitted by the
diagnostics CLI). Training log: `phases/phase4_wave2_seed901_training.log`. The checkpoint
binary is not committed; regenerate with
`python -m optspread.cli.train_resumable --run-dir <dir> --wave 2 --seed 901` (~2.5 h) —
the run is resumable and progress-visible. Caveat: torch 2.9/py3.14 environment (drifted
from the repo pin); the diagnostics are pure-numpy over the checkpoint and a resumed run
is scientifically equivalent but not byte-identical to an uninterrupted one.

---

## 2026-06-30 — Wave 2 (Heston SV): GV_2 PASS, BV_2 blocked on signal/risk fork

**GV_2: PASS** (mean_vrp +0.0012, skew, iv_rank_std 0.327, term-slope, mean-reversion
acf, change_var_ratio all hold) after teachability calibration: v0_theta_mult prior
widened 0.80–1.20 → 0.70–1.85, Heston warmup 21 → 8, vrp_theta_mult 1.3 → 2.0
(commit `0912efd`). A no-agent signal probe shows entry IV-rank predicts forward
VRP with a real but **modest** gradient: high-IV-rank forward-VRP ≈ 4× low
(+0.0039 vs +0.0010), +EV overall, raw corr +0.03.

**BV_2 (corr(credit_indicator, iv_rank) > 0.10), trained 3 seeds each, 100k steps:**

| Agent | BV_2 | FF_2 | Note |
|---|---|---|---|
| IQN/CVaR (primary), α=0.1 | **0/3** (nan/−0.04) | 3/3 | stays ~99% FLAT — won't trade the thin edge |
| PPO baseline | 2/3 (corr +0.10, +0.32) | 1/3 | LEARNS the conditioning but over-trades Wave-0 |

**Deploy-risk diagnostic (re-deploy the SAME trained IQN checkpoints at relaxed
risk, no retraining):** trade-freq and corr rise as conservatism relaxes —
cvar0.1 ≈ 0% flat → cvar0.2 ≈ 0.5% → **mean ≈ 5% trade, corr +0.12 (seed 813),
+0.095 (812)**. So the agent learned only a WEAK edge AND α=0.1 deployment fully
suppresses it.

**Diagnosis:** the Heston mean-reversion VRP is a far subtler edge than Wave-1's
exaggerated premium. PPO (EV) learns it; the tail-averse CVaR/IQN primary stays
flat. Passing BV_2 for the PRIMARY agent needs BOTH a stronger learnable edge and
a less-conservative curriculum deployment. **Open fork (awaiting user decision):**
- (A) strengthen the Heston signal further (higher vrp_theta_mult and/or kappa
  floor so forward VRP is more predictable from IV-rank) — risk: realism / GV_2
  mean_vrp balance;
- (B) deploy/train the curriculum CVaR agent at a higher α (0.2–0.3) — still
  tail-aware; the headline "CVaR beats EV on tail" holds at α=0.2;
- (C) combine (A)+(B) (most likely to pass).
Held here per the pre-agreed checkpoint rather than tuning the generator further
unsupervised. Per-seed detail: `runs/phase4_wave2_bv2_ff2.json`.

### Update — combined fix (stronger signal + α=0.2) + off-policy diagnosis

User chose the combined fix. Applied: kappa prior [1,10]→[4,12] (faster
mean-reversion ⇒ forward VRP more predictable from IV-rank; probe mean forward-VRP
doubled to +0.0044, GV_2 still PASS, commit `db2bbe5`) and deploy/train the
curriculum CVaR agent at **α=0.2**. Result (3 seeds, 100k steps):

| Agent | BV_2 | FF_2 |
|---|---|---|
| IQN/CVaR α=0.2 | **0/3** (still ~99% FLAT) | 3/3 |
| PPO | **2/3** (corr +0.37, +0.50 — stronger!) | 2/3 |

The stronger signal markedly helped PPO but NOT the CVaR/IQN primary. A
deploy-risk probe (re-deploy the SAME trained checkpoints at relaxed risk) shows
even at **mean (risk-neutral)** deployment IQN trades only ~1–2% → the off-policy
agent **did not learn to value the trade**; this is NOT α-conservatism.

**Refined diagnosis:** on-policy PPO (policy-gradient + entropy) captures the
subtle Heston mean-reversion edge; off-policy IQN (ε-greedy Q-learning) under-
learns it — the mean Q-advantage of trading over FLAT stays near zero. Likely
**undertraining**: Wave-2 IQN ran only 100k steps (cut for Heston's ~3× cost),
vs the **150k** the Wave-1 width-256 ensemble needed to pass — on a harder signal.
**Next (in progress):** IQN-only Wave-2 retrain at 150k steps, α=0.2, same boosted
generator, to test the undertraining hypothesis. PPO ensemble (2/3 BV) kept.

### Update — "blindness to success" fix: success-weighted replay is the lever

Literature (Greenberg et al., CeSoR, NeurIPS 2022) names our failure "blindness to
success": CVaR's tail-only weighting discards the successful trajectories that would
teach the agent to trade. Our training is already risk-neutral (soft-risk endpoint),
so the missing half is CeSoR's cross-entropy over-sampling of successful transitions,
implemented as **success-weighted replay** (`reward_priority_boost`, commit b6b4768).

A/B on Wave 2 (IQN width-256, α=0.2, 150k):

| Config | IQN trade-freq | BV_2 | FF_2 |
|---|---|---|---|
| uniform replay (100k & 150k) | ~0–2% (flat) | 0/3 | 3/3 |
| reward_priority_boost=6.0 | 13–17% (trades) | 1/3 (seed +0.14) | 0/3 |

**Findings:** (1) more steps alone (100k→150k) did NOT help — falsifies undertraining;
the flat-collapse is a credit-assignment/SNR problem. (2) Success-weighted replay
**works as a knob on trade frequency** — it broke the flat-collapse and got the CVaR
agent trading (corr up to +0.17). (3) boost 6.0 OVER-corrects: the agent trades
indiscriminately incl. ~100% on no-edge Wave 0 (FF fails) and conditioning dilutes.
**Refined run (in progress):** boost 2.5 + light Wave-0 rehearsal 0.12 — the two
compose (Wave-0 rehearsal episodes have no profitable trades, so the boost won't
over-sample them), targeting conditional trading (BV pass) + no-edge flatness (FF
pass). If it lands both gates, this is the reusable Wave-3–6 lever. See
[[reference-cvar-blindness-cesor]] in memory.

---


## 2026-06-30 — Wave 1 re-validated on the expanded 35-dim observation

Per the user directive to "give the agent as much information as possible relevant
to learning patterns" for real-data readiness, the canonical regime-feature block
was expanded **5 → 24 causal, no-look-ahead features** (obs_dim 16 → 35; commit
`e996cb7`): surface shape (skew, smile/term curvature, ATM level, skew-term), IV
dynamics (1d/5d change, vol-of-vol, IV z-score), and return dynamics (5d/63d
momentum & realized vol, RV term ratio, realized skew/kurtosis, downside
semideviation, path drawdown, normalized VRP). The whole curriculum is retrained
on this obs (old checkpoints are obs_dim=16 and incompatible).

**BV_1 / FF_1 (rich obs), primary IQN/CVaR ensemble, deterministic, CVaR deploy:**

| Agent (recipe) | BV_1 corr(credit,vrp) | BV_1 pass | FF_1 (Wave-0 no-edge) |
|---|---|---|---|
| **IQN/CVaR, width 256, no rehearsal** | +0.64..+0.76 | **3/3 PASS** | **3/3 PASS** (flat 0.93–0.95) |
| IQN/CVaR, width 128, no rehearsal | +0.66..+0.80 | 3/5 PASS | 5/5 PASS |
| PPO baseline, rehearsal 0.25 | +0.58..+0.77 | 5/5 PASS | 3/5 PASS |

All passing corrs ≫ the 0.10 threshold. Per-seed detail in
`runs/phase4_wave1_v4.json`.

### What the rich-obs revalidation took (additions to the original recipe)

1. **Network capacity must scale with the obs.** At the original width 64 the
   value-based CVaR/IQN agent **collapses to FLAT** on most seeds with the 35-dim
   obs (it cannot value trading through the wider, noisier input) — BV ~1–2/5.
   **Width 256 restores a robust 3/3** (width 128 gives majority 3/5). This is a
   genuine curse-of-dimensionality + CVaR-conservatism tension, documented, not a
   bug. New flag: `--hidden-sizes` (commit `962182d`).
2. **CVaR alone keeps IQN FF-robust — IQN needs NO rehearsal.** The tail-averse
   CVaR deployment makes the distributional agent stay flat on the no-edge Wave 0
   without any rehearsal (FF 3/3–5/5). Adding Wave-0 rehearsal to IQN is
   counter-productive: it over-reinforces flatness and worsens the FLAT collapse.
3. **PPO (EV-maximizer) DOES need rehearsal.** With rich features and no rehearsal
   PPO over-trades on Wave-0 noise and loses (FF 0/3). Wave-0 rehearsal
   (`--rehearsal-fraction`, injected `RehearsalGenerator`, commit `c0b6865`)
   recovers FF to 3/5. Rehearsal is a per-algorithm training hyperparameter (like
   the differing learning rates); **evaluation is identical pure-wave BV/FF for
   both agents**, so the comparison stays fair. This is itself a thesis-supporting
   finding: the EV agent chases noise features; the CVaR agent does not.
4. Exploration floor (`--epsilon-end`/`--epsilon-decay-steps`, commit `0d83342`)
   is exposed for the distributional agent; width was the dominant lever.

**Adopted Wave-2+ recipe:** IQN/CVaR width 256, no rehearsal, ε-floor 0.04,
CVaR α=0.1 at deployment, MTM-only reward, risk-neutral bootstrap, from scratch;
PPO baseline with Wave-0/earlier-wave rehearsal for FF.

---

## Original 16-dim results (superseded by the rich-obs section above)


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
