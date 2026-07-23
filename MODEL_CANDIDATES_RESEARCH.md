# Model Candidates for Passing BV_2/FF_2 — Open-Discovery Research Report

**Task:** produce a list of models/algorithm changes to fine-tune so the primary risk-sensitive
agent passes both Wave-2 gates: **BV** (trade conditionally on the thin Heston VRP signal,
corr > 0.10) and **FF** (stay flat on the no-edge fair-IV control). Requested by Dominick,
researched 2026-07-06/07.

**Method and epistemic status.** Open-discovery deep-research run (5 parallel search angles →
37 sources fetched → 181 falsifiable claims extracted → 3-vote adversarial verification per
claim). **No candidate model names were placed in the research prompts** — agents were given
the failure-mode data and constraints only, and asked what the field actually uses
(2020–2026). Verification was interrupted by a usage limit: **~14 claims survived full 3-0/2-1
adversarial verification, 4 claims were refuted and killed, the rest are source-quoted but not
yet adversarially verified.** Labels below: **[VERIFIED]** (survived adversarial votes),
**[UNVERIFIED]** (extracted with quote, votes incomplete), **[KILLED]** (refuted by ≥2/3
adversarial reviewers — listed at the end so we don't rely on them).

> **2026-07-21 verification pass:** every remaining [UNVERIFIED] claim was checked against its
> primary source by 12 dedicated adversarial verifier agents. Labels below are updated in
> place; unsupported details were DELETED (see the Verification Log at the end of this file).
> No [UNVERIFIED] labels remain.

**The diagnosed failure this list targets** (from `phases/PHASE4_GATE_REPORT.md`): on-policy
PPO learns the conditional edge (corr +0.37/+0.50); off-policy IQN with CVaR deployment
collapses to ~99% FLAT and never learns the trade's value (Q-advantage of trading ≈ 0 even
redeployed risk-neutral). Undertraining falsified. Success-weighted replay (uncorrected)
breaks flat-collapse at boost 6.0 but then fails the no-edge gate.

---

## 1. What the field actually says about this exact problem

- **The pathology is named and actively studied.** "Blindness to success" — risk-averse RL
  ignoring high-return strategies and converging to passive policies — is documented across
  multiple 2022–2025 papers. [VERIFIED] CVaR policy gradients "cannot distinguish returns due
  to good actions from returns due to lucky stochasticity" and converge to overly conservative
  passive policies on the Betting Game benchmark (Return Capping, arXiv:2504.20887).
  [VERIFIED] The state-of-the-art risk-averse baseline WCSAC gets stuck in passive suboptimal
  policies on rare-hazard GuardedMaze (60%/30% convergence) while optimistic exploration
  (ORAC) recovers 100%/80% (arXiv:2507.08793).
- **The project's own design choice (risk-neutral training + CVaR at deployment) is
  corroborated — with thin evidence.** [VERIFIED 2026-07-21, tempered] A real-market
  natural-gas futures study (arXiv:2501.04421) uses exactly this recipe (confirmed from
  Algorithm 1: risk-neutral bootstrap target, CVaR only in the action-selection argmax). The
  ">32%" win is real but **C51-specific** (QR-DQN/IQN beat classical RL by far smaller
  margins), and the claimed alpha-monotonicity **fails for QR-DQN by the paper's own data**
  ("QR-DQN₁ presents higher risk aversion than QR-DQN₀.₇") — it holds only as a general trend
  for C51/IQN. Evidence quality: 4 backtest windows, one instrument (TTF front-month), zero
  transaction costs, no seeds/error bars, and the authors needed post-hoc Sharpe reward
  shaping to obtain risk aversion at all. Treat as suggestive, not validation.
- **But the field also says CVaR-of-quantiles action selection is mathematically shaky.**
  [VERIFIED] Applying a fixed risk measure per decision step (iterated/nested CVaR — already
  rejected in this repo for collapsing to FLAT) is overly conservative and optimizes an
  ill-defined objective (arXiv:2307.00547). [VERIFIED — sweep-2 3-0 + 2026-07-21 Thm-3.2
  corroboration] Lim & Malik (NeurIPS 2022) prove the standard "greedy CVaR-of-quantiles +
  distributional Bellman" recipe can converge to a policy optimal under *neither* the static
  nor the dynamic CVaR criterion; arXiv:2307.00547's Theorem 3.2 independently proves the
  recursive risk-sensitive operator is biased whenever the risk functional isn't the mean.
  (Their advertised operator *fix* remains REFUTED per sweep 2 — the impossibility half
  stands, the repair is unproven.)
- **The current value-based frontier builds ON IQN rather than replacing it.**
  [VERIFIED 2026-07-21] BTR (ICML 2025 poster, arXiv:2411.03820) = IQN + Munchausen + Impala
  ResNet + spectral norm + vectorization + maxpooling (all six confirmed from the paper);
  its ablations found architecture the dominant lever ("Impala had the largest effect on
  performance (+142% IQM)" — Fig. 5 caption) — independently corroborating the project's
  width-256 finding. IQM 7.4 on Atari-60 confirmed from the abstract. [VERIFIED] BBF
  (arXiv:2305.19452) shows value-based agents reaching super-human Atari performance within
  ~100k steps — the project's step budget is feasible for value-based RL *with the right
  supporting components*.

---

## 2. Ranked candidates

Ranked by mechanism-fit to the diagnosed failure × evidence strength × implementation surface.
Each entry: mechanism vs failure mode | evidence | predicted BV/FF interaction | effort |
pre-registerable prediction.

### C1. Spectral risk measures (Mean-CVaR) in the value-based agent — "QR-SRM"
- **Mechanism:** pure CVaR weights only the left tail → the agent literally cannot see the
  upside that justifies trading (the blindness). Spectral/Mean-CVaR risk weights BOTH the tail
  and the body/upside. Attacks the root cause while staying risk-averse and value-based.
- **Evidence:** [VERIFIED] QR-SRM (arXiv:2501.02087) names blindness-to-success explicitly and
  proposes spectral measures as the fix; it is a QR-DQN-style bi-level extension (outer
  closed-form risk-dual update + inner QR-DQN), i.e., a small delta on this repo's existing
  QR-DQN/IQN code. Related: [VERIFIED 2026-07-21] AC-SRM (arXiv:2507.03900) optimizes *static*
  spectral risk by augmenting the state with accumulated discounted reward and discount factor
  ("S_{t+1}=S_t+C_tR_t, C_{t+1}=γC_t", citing Bäuerle & Glauner 2021) — Markovian via state
  augmentation, so it does NOT violate the no-recurrence invariant. DELETED on verification:
  AC-SRM's "consistently outperforms CVaR-AC/Tamar/Dabney-style baselines" — those baselines
  were never actually benchmarked in the paper (only its own SAC/TD3/DSAC/ORAAC/CODA variants,
  with wins real on tail metrics but mixed on means).
- **Gates:** BV ↑ (upside now visible → trade gets valued); FF neutral (still risk-averse; a
  no-edge trade has negative expected spectral value after costs).
- **Effort:** low-medium (risk-weighting change + dual variable on existing nets).
- **Prediction:** Mean-CVaR action selection (e.g., 0.5·mean + 0.5·CVaR_0.2) lifts IQN
  trade-frequency off ~0 without any replay boost; BV_2 corr > 0.10 on ≥2/3 seeds; FF_2 flat
  fraction stays ≥ threshold.

### C2. Risk-level scheduling: train permissive, anneal to the deployment alpha
- **Mechanism:** static risk-aversion from step 0 prevents ever *discovering* the edge;
  annealing (risk-neutral → target alpha) lets the agent learn values first and become
  conservative second.
- **Evidence:** [VERIFIED 2026-07-21, corrected] Risk-scheduling in distributional RL
  (arXiv:2206.14170): SMAC MMM2 Table-1 numbers confirmed — statically risk-averse 0.002
  win-rate vs 0.607 for the same endpoint reached by scheduling. Two corrections: the paper's
  schedule anneals **risk-seeking → risk-averse** (starts sampling quantile fractions from
  U[0.75,1], not from risk-neutral), and the setting is multi-agent DMIX/QMIX on StarCraft
  (ICML **workshop** paper, 5 seeds) — quote the numbers as directional support for
  "anneal, don't fix, the risk level," not as transferable magnitude. [VERIFIED 2026-07-21]
  A 2025 trading application (Tail-Safe Hedging, arXiv:2510.04555) anneals alpha from a
  permissive 0.10 toward 0.025 (schedule form unspecified in the paper) with a PID
  tail-coverage controller, explicitly to avoid instability from premature tightening.
- **Gates:** BV ↑; FF risk — mid-training the permissive agent trades the no-edge world, so FF
  must be evaluated at the annealed endpoint; if FF fails, combine with C1 (anneal *into* a
  spectral measure, not pure CVaR).
- **Effort:** trivial (schedule on the existing `sample_taus` alpha).
- **Prediction:** alpha annealed 1.0 → 0.2 over the first half of training beats fixed
  alpha=0.2 on BV_2 corr on ≥2/3 seeds with FF_2 unchanged at endpoint.

### C3. Optimistic exploration for risk-averse agents (ORAC-style), replacing ε-greedy
- **Mechanism:** the diagnosed failure is "never learns the trade has value" — an exploration/
  credit problem. ORAC explores by maximizing an *upper confidence bound* of value (optimism)
  while keeping the risk-averse objective for exploitation — risk-sensitivity stays agent-side,
  reward untouched.
- **Evidence:** [VERIFIED] ORAC (arXiv:2507.08793): exploration-side not reward-side;
  rare-hazard GuardedMaze WCSAC 60%/30% → ORAC 100%/80%. Directly analogous: the profitable
  credit-spread state is the "rarely-discovered reward" here.
- **Gates:** BV ↑; FF neutral (optimism decays with epistemic uncertainty; on no-edge worlds
  the learned value of trading is genuinely negative after costs).
- **Effort:** medium — needs an uncertainty estimate: bootstrapped quantile heads or a small
  ensemble over the existing IQN.
- **Prediction:** UCB-style behavior policy raises trade-freq during training ≥5x vs ε=0.04
  floor and yields BV_2 corr > 0.10 on ≥2/3 seeds without replay boost.

### C4. Action-gap operators: clipped Advantage Learning / consistent Bellman operator
- **Mechanism:** the measured failure is Q-advantage(trade vs FLAT) ≈ 0 — below approximation
  noise. Gap-increasing operators provably widen the best-vs-second-best value gap so a thin
  true edge survives function-approximation error. One-line TD-target change.
- **Evidence:** [VERIFIED] Consistent Bellman operator increases the action gap and mitigates
  greedy-selection corruption by approximation error (Bellemare et al., arXiv:1512.04860);
  optimality-preservation conditions proven. [VERIFIED] Plain Advantage Learning is *harmful*
  when the approximate greedy action is wrong (common under under-exploration — i.e., during
  the current flat-collapse); use **clipped AL** (AAAI 2022, arXiv:2203.11677), which only
  widens gaps below a threshold and is proven optimality-preserving.
- **Gates:** BV ↑ (amplifies the thin real gap); FF ↑ **in principle** (on no-edge, the true
  gap favors FLAT and gets amplified too — this candidate is unusual in plausibly helping BOTH
  gates; pre-register that).
- **Effort:** low (TD-target modification in the existing trainer).
- **Prediction:** clipped-AL on IQN raises mean Q-advantage of trading in high-signal states
  above 0 while making FLAT's advantage on Wave-0 MORE negative; passes both gates on ≥2/3 seeds.
- **Caveat:** pair with C2 or C3 — gap amplification with a wrong greedy action entrenches it.

### C5. Distributional-critic + PPO hybrid (PG-Rainbow / "IQN-CVaR-PPO")
- **Mechanism:** exploit the observed asymmetry — PPO already learns the edge. Give the
  on-policy actor a distributional IQN critic and a CVaR-weighted advantage, so the *headline
  claim* (distributional risk-sensitive beats EV on tails) survives with the learner that works.
- **Evidence:** [VERIFIED 2026-07-21, two independent 2024/2025 sources] PG-Rainbow
  (arXiv:2407.13146): IQN critic feeding PPO via `V(s) = f(V_θ(s) ⊙ q_φ(s))` (Hadamard +
  distillation net, confirmed from the paper's equation); crucial implementation warning
  confirmed directionally — disjoint/naive IQN-as-value-net substitution degrades performance
  while the distillation-mediated fusion works ("fails completely" is one fetch's phrasing;
  treat strength of wording as moderate). The DemonAttack masking demonstration (scalar V(s)
  hides action-conditioned distribution info) is confirmed (Figs. 2 vs 6). Note: PG-Rainbow's
  policy remains mean-based — no CVaR anywhere in that paper; the risk machinery comes from
  the second source. Tail-Safe Hedging (arXiv:2510.04555): IQN-CVaR-PPO with CVaR-weighted
  GAE + KL/entropy-regularized clipped actor confirmed from its Eqs. 8-10.
- **Gates:** BV ↑↑ (inherits PPO's demonstrated pass); FF — needs the rehearsal recipe PPO
  already uses (2/3 with rehearsal 0.12–0.25).
- **Effort:** medium-high (new actor-update path; reuse existing IQN + PPO modules).
- **Flag for Dominick:** this moves the primary agent from off-policy value-based to on-policy
  actor-critic — a thesis-framing decision, not just a hyperparameter. The comparison "EV PPO
  vs CVaR-critic PPO, same reward" arguably matches the study's stated claim even better.
- **Prediction:** IQN-critic PPO with CVaR-weighted GAE at alpha 0.2 passes BV_2 on ≥2/3 seeds
  and beats EV-PPO on eval CVaR/Sortino (the actual headline metric) on Wave 2.

### C6. Information-gain prioritized replay (UPER) — replace both TD-PER and the raw success-boost
- **Mechanism:** options P&L is aleatoric-noise-dominated; TD-error PER provably over-samples
  irreducible noise ("noisy-TV of replay"), and the current `reward_priority_boost` oversamples
  lucky outcomes (why boost 6.0 broke FF). UPER prioritizes by epistemic/aleatoric ratio —
  sampling what the agent can *learn from*, not what was merely surprising or profitable.
- **Evidence:** [VERIFIED 2026-07-21] UPER (RLC 2025): all three sub-claims confirmed from the
  primary PDF — PER's aleatoric-noise bias ("resembles the noisy TV problem", their words),
  the information-gain priority p=½log(1+Ê/Â) from an N=10 QR-DQN ensemble (Eq. 11), and the
  Atari-57 win over PER/QR-DQN/QR-PER/QR-ENS-PER with prioritization causally isolated.
  Caveat: "significantly" is the authors' characterization (visually non-overlapping ±2SD
  bands), not a formal statistical test.
- **Gates:** BV ↑ (signal-bearing transitions get replayed); FF ↑ vs the boost (no bias toward
  lucky no-edge wins — the exact failure of boost 6.0).
- **Effort:** medium (needs a small QR ensemble for the uncertainty split).
- **Prediction:** UPER at boost-equivalent strength reaches trade-freq ≥ 10% on Wave 2 while
  keeping Wave-0 flat-fraction ≥ 0.9 — the combination boost 6.0 could not achieve.

### C7. BBF-package sample-efficiency levers: annealed n-step, SPR auxiliary loss, reset-based width scaling
- **Mechanism:** three composable credit/representation levers from the strongest
  sample-efficiency line: (a) n-step annealed 10 → 3 propagates the sparse profitable-trade
  signal fast, then sharpens; (b) a self-predictive (SPR) auxiliary loss supplies learning
  signal the near-zero-SNR reward cannot; (c) width scaling *with* shrink-and-perturb resets +
  weight decay — directly qualifies this repo's width-256 finding if pushing capacity further.
- **Evidence:** [VERIFIED] BBF reaches super-human Atari at ~100k steps. [VERIFIED 2026-07-21]
  its ablations, confirmed with direct quotes from the full-text PDFs: n-step annealed 10→3
  over ~10k grad steps post-reset "yields a much stronger agent than using a fixed value of
  n=3 ... or n=10" (Fig. 5); removing SPR causes "a substantial performance degradation";
  and "the performance of SR-SPR collapses as network size increases" without
  shrink-and-perturb resets + weight decay (Fig. 3) — the width-collapse is SR-SPR-specific
  (BBF's base agent), exactly as stated (arXiv:2305.19452, arXiv:2007.05929).
- **Gates:** BV ↑; FF neutral.
- **Effort:** low each; composable with everything above.
- **Prediction:** n-step 10→3 annealing alone lifts BV_2 corr by ≥0.05 mean over fixed n=3 at
  equal steps.

### C8. Munchausen target bonus on IQN
- **Mechanism:** adds a scaled log-policy term to the bootstrap target — implicit KL/entropy
  regularization that keeps action probabilities from collapsing onto FLAT early (agent-side,
  reward unchanged).
- **Evidence:** [VERIFIED 2026-07-21] Munchausen-RL confirmed as a BTR component (ICML 2025):
  adds the scaled log-policy term "α τ ln π(a|s)" to the bootstrap target, and per the paper
  "As Munchausen does not use argmax over the next state, Double DQN is obsolete" — an
  agent-side value-target modification, reward untouched.
- **Gates:** BV ↑ (anti-premature-collapse); FF slight risk (entropy pressure → some no-edge
  trading; small coefficient).
- **Effort:** low.
- **Prediction:** M-IQN raises trade-freq during training and passes BV_2 on ≥1 more seed than
  baseline IQN at equal config.

### C9. Modified distributional Bellman operator for risk-sensitive control (Lim & Malik)
- **Mechanism:** fixes the proven incoherence of greedy CVaR-of-quantiles + distributional
  Bellman (converges to neither static nor dynamic CVaR optimum). Small extension of the
  QR-DQN/IQN family.
- **Evidence:** premise [VERIFIED — sweep-2 3-0]; the operator *fix* itself [REFUTED — sweep-2
  0-3, see S2.4]. The incoherence theorem stands; the advertised repair is unproven. This
  candidate survives only as "de-risk the correctness story," not as a fix.
- **Gates:** BV ↑ possible; FF unknown. Primarily de-risks the *correctness* of whatever CVaR
  policy is finally reported.
- **Effort:** medium.
- **Prediction:** under the fixed operator, the learned policy's realized tail metrics match
  the intended static-CVaR preference better than the current recipe (measurable on the toy
  fat-tail MDP first).

### C10. [FLAGGED — invariant change] History-dependent static-risk methods (TQL) / recurrent off-policy
- [VERIFIED 2026-07-21, with the qualification that matters] arXiv:2307.00547's claim is NOT
  absolute: plain (non-augmented) state-Markovian DP cannot optimize general
  trajectory-level distortion risk — but the paper itself concedes "very few distortion risk
  measures (e.g. CVaR) can rely on sufficient statistics to achieve Markovian policy
  optimization (Bäuerle and Ott, 2011)". So for CVaR specifically, state augmentation (see
  S2.3) escapes the impossibility without recurrence. TQL's own guarantee is monotone policy
  improvement (Thm 4.2); its operator is only non-expansive, not contractive (Thm 4.3), so
  global value-iteration convergence is NOT guaranteed. Recurrent TQL stays flagged; the
  augmented-state route folded into C1/S2.3 is the invariant-legal path.

### C0 (do first, free): two diagnostics before any new training
- **Quantile-collapse check** [VERIFIED 2026-07-21 as the paper's claim; evidence THIN]: the
  asymmetric Huber loss (kappa=1) collapsing the return distribution toward its mean is a
  real published finding (arXiv:2305.16877, UAI 2025: end-of-training spread 0.144 with Huber
  vs 1.25 with pure L1 on IQN/Atari-5, 5 seeds) — but it rests on one paper, one
  architecture, five games, a single time-snapshot, no formal proof, and **no independent
  replication** (the original QR-DQN/IQN papers use kappa=1 as default and report no
  pathology). Treat as a plausible risk flag, not settled law — which is exactly why the
  in-vitro check matters: measure quantile spread on the existing Wave-2 checkpoints; if
  collapsed, CVaR action selection is degenerate (CVaR ≈ mean) and much of the observed
  behavior is explained; then switch the quantile loss (expectile/dual or smaller kappa).
- **Advantage-gap audit:** log Q(trade)−Q(FLAT) in high-signal vs no-signal states across
  training — the direct observable C4 targets, and the cleanest pre-registration statistic.

---

## 3. Findings that REFRAME the problem (report to Dominick, decide deliberately)

- **Stay-flat as architecture, not learning** [VERIFIED 2026-07-21]: confirmed from Tail-Safe
  Hedging's Sec. 3.3 (Eqs. 15-17) — an ellipsoidal no-trade band and a sign-consistency gate
  enforced as hard constraints in a convex CBF-QP layer sitting on top of the learned policy;
  the no-trade behavior is NOT learned. Translation here: FF-style no-edge discipline could
  be a *verifiable structural gate* (trade only when the critic's spectral advantage clears
  costs by a margin) rather than an emergent behavior. This changes what the FF gate tests
  (the system, not the policy), so it's a thesis-design decision.
- **Static-vs-iterated CVaR incoherence is load-bearing, not a footnote** [VERIFIED for the
  conservatism claim]: the repo's CLAUDE.md caveat ("static CVaR is time-inconsistent") is
  where the field says the flat-collapse partly originates. Whatever ships, state *which* CVaR
  the final agent approximates.
- **Episodic-fitness methods sidestep per-step credit assignment entirely**
  [VERIFIED 2026-07-21, scope-limited]: confirmed from the ERL survey (arXiv:2303.04150) —
  "EC is indifferent to the reward sparsity by using an episodic fitness metric", and the
  ERL hybrid (population → replay buffer → periodic gradient injection) "has outperformed
  PPO, DDPG, and GA on MuJoCo continuous control benchmarks". **Scope caveat:** every cited
  result is continuous-control (MuJoCo, DDPG/TD3/SAC-family); no discrete-action ERL result
  exists in the surveyed evidence — applying it to this repo's 19-action setting is an
  extrapolation, not a demonstrated finding. A heavier lift; attacks the diagnosed mechanism
  from outside the TD paradigm.

## 4. Killed claims (refuted 2/3+ by adversarial review — do NOT build on these)

1. "Blindness-to-success is *provably* a local-optimum barrier under stated conditions" — the
   mechanism is real (see C1/C2 evidence) but this strong provability framing did not survive.
2. "Soft risk (annealing alone) is *sufficient* to bypass the barrier" — annealing helps (C2);
   sufficiency was refuted (consistent with this repo's own data: alpha 0.2 alone didn't fix it).
3. "Return Capping empirically beats EV-PPO, CVaR-PG, CVaR-PPO and MIX across four benchmarks
   incl. all-seeds convergence" — the equivalence *theory* survived 3-0; the empirical-dominance
   claim was refuted. Return capping stays interesting but unproven.
4. One generic "this paper documents the exact vegalab pathology" claim — refuted as
   overreach; the specific, quoted pathology claims above survived on their own evidence.

The pattern: **mechanisms survived; sufficiency/dominance overclaims died.** Calibrate accordingly.

## 5. Suggested experiment order (cheap→expensive, composable)

1. C0 diagnostics (hours) → may reprioritize everything.
2. C2 alpha-annealing + C7a n-step annealing (config-level, days).
3. C1 spectral/Mean-CVaR action selection (small code, days) — the highest verified-evidence fit.
4. C4 clipped-AL target (small code) — pre-register the both-gates prediction.
5. C6 UPER or C3 ORAC (ensemble infrastructure, ~week).
6. C5 hybrid (framing decision with Dominick first).

Every run: 3 seeds, pre-registered prediction committed before training (repo discipline),
BV_2 AND FF_2 reported together — any candidate that passes one by failing the other is a fail.

## 6. Sources (primary)

- CeSoR — Efficient Risk-Averse RL, NeurIPS 2022: arxiv.org/abs/2205.05138
- Return Capping CVaR-PG (2025): arxiv.org/abs/2504.20887
- QR-SRM — Beyond CVaR, spectral risk (2025): arxiv.org/abs/2501.02087
- Risk-perspective exploration/scheduling (2022): arxiv.org/abs/2206.14170
- ORAC — Optimistic risk-averse exploration (2025): arxiv.org/abs/2507.08793
- AC-SRM — static spectral risk actor-critic (2025): arxiv.org/abs/2507.03900
- Lim & Malik — distributional RL for risk-sensitive policies, NeurIPS 2022 (proceedings)
- "Is Risk-Sensitive RL Properly Resolved?" / TQL: arxiv.org/abs/2307.00547
- Action gap — Bellemare et al. 2016: arxiv.org/abs/1512.04860; clipped AL (AAAI 2022): arxiv.org/abs/2203.11677
- BBF (2023): arxiv.org/abs/2305.19452; SPR (2021): arxiv.org/abs/2007.05929
- BTR (ICML 2025): arxiv.org/abs/2411.03820 (code: github.com/VIPTankz/BTR)
- UPER (RLC 2025): rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_45.pdf
- PG-Rainbow (2024): arxiv.org/abs/2407.13146
- Tail-Safe Hedging (2025): arxiv.org/abs/2510.04555
- Natural-gas futures distributional RL (2025): arxiv.org/abs/2501.04421
- IEQN — expectile-quantile distributional RL (2023): arxiv.org/abs/2305.16877
- Evolutionary RL survey (2023): arxiv.org/abs/2303.04150
- DSAC-T (2023/24): arxiv.org/abs/2310.05858
- Munchausen RL context via BTR; FQF: arxiv.org/abs/1911.02140

---
---

# SWEEP 2 — Pure Open Exploration (2026-07-07): Independent Replication + New Findings

**Method change:** second, fully independent deep-research run. The brief contained ONLY raw
observations and the study's design commitments — no discovery angles, no diagnosis language,
no reference to Sweep 1's findings, no deliverable taxonomy. 103/103 agents completed; 101
claims extracted, 25 adversarially verified: **16 confirmed, 9 refuted, 0 unverified.**
All findings below carry full 3-0 or 2-1 adversarial votes.

## S2.1 The convergent diagnosis (sharper than Sweep 1)

**The FLAT collapse is the predicted OPTIMUM of the objectives actually being optimized — not
a training failure.** Three verified findings compose the argument:

1. **[3-0, high] Iterated/nested CVaR → worst-path RL → FLAT is exactly optimal.** Applying
   CVaR inside the Bellman backup optimizes a strictly more conservative objective than static
   total-return CVaR; as alpha→0 it converges to "maximize the minimum cumulative reward"
   (Du, Wang & Huang, ICLR 2023, arXiv:2206.02678; corroborated arXiv:2211.07288,
   arXiv:2111.06803). In an options-selling world every trade has a strictly negative worst
   path and FLAT's worst path is 0 — so the nested-CVaR experiment this repo ran and rejected
   didn't fail; it *succeeded at the wrong objective*.
2. **[3-0×3, high] The current deployment recipe is formally unsound.** Risk-neutral mean
   bootstrap + CVaR-of-quantiles greedy action selection can converge to a policy optimal for
   NEITHER dynamic nor static CVaR (Lim & Malik, NeurIPS 2022; arXiv:2501.02087;
   arXiv:2301.05981). Static CVaR itself is time-inconsistent. → The thesis must state *which*
   CVaR the shipped agent approximates; currently the answer is "neither."
3. **[3-0×2, high] Blindness-to-success is a proven mechanism** with an exponential
   escape-probability barrier (CeSoR paper) — with the honest caveat that the theorem covers
   CVaR policy gradient; the mapping to value-based IQN is analogical (three claims asserting
   *direct* applicability were refuted 0-3).

**[3-0×2, high] The distributional critic is NOT data-starved:** distributional TD has the
same sample complexity as mean TD up to log factors (arXiv:2502.14172, arXiv:2402.07598;
policy-evaluation, linear/categorical scope). The failure is objective/exploration-side, not
statistical.

## S2.2 The success-weighted replay is an INVERTED CeSoR [3-0×3, high]

CeSoR = (1) soft-risk annealing (alpha from 1 → target so early gradients see successes) +
(2) cross-entropy oversampling of **worst-case environment CONDITIONS at the episode level,
with importance-sampling correction**. The repo's `reward_priority_boost` oversamples
**positive-reward TRANSITIONS, uncorrected** — the opposite tail, no IS weights. That
unconditional optimistic bias is a sufficient explanation for why boost 6.0 trades
indiscriminately on the no-edge control. Fix: episode/regime-level adverse-condition sampling
with IS correction, not winner-transition boosting.

## S2.3 New candidates surfaced ONLY by the pure sweep

- **ECRM augmented-action reformulation [3-0, medium]** (arXiv:2301.05981): an
  expectation+CVaR convex-combination risk measure admits an EXACT risk-neutral reformulation
  with an augmented action space (the Rockafellar-Uryasev auxiliary variable becomes extra
  actions). No recurrence, no operator surgery — invariant-compatible. Caveat: validated on
  toy MDPs; guarantees don't cover function approximation.
- **Superiority distribution / DSUP [2-1 + 3-0, medium]** (arXiv:2410.11022): when
  per-decision action gaps are thin, the MEAN is the worst-placed statistic (collapses at
  O(h) while distributions differ at Θ(h^1/2)); the "superiority" distribution — a
  probabilistic generalization of the advantage — is action-gap preserving and provably yields
  the SAME greedy action under any distortion risk measure incl. CVaR. Directly targets the
  observed near-zero mean advantage. Open question: transfer outside the h→0 asymptotic regime.
- **Bäuerle-Ott state augmentation for static CVaR** [VERIFIED 2026-07-21 — theory confirmed,
  practice hard]: Bäuerle & Ott (2011, Math. Methods of OR) prove that augmenting the state
  with a running risk-budget variable makes the augmented state a sufficient statistic — an
  optimal *Markovian, deterministic* policy exists on (s, b) for static total-return CVaR,
  reducing it to an ordinary risk-neutral MDP plus an outer 1-D optimization over b₀.
  Operationalized in tabular/low-dim settings (Chow & Ghavamzadeh 2014; Chow-Tamar-Mannor-
  Pavone 2015; ICML-2023 near-minimax AugMDP). **Practical caveats, all verified:** the
  continuous b-dimension needs function approximation and full traversals ("intractable for
  continuous state spaces" without it); the outer b₀ bisection is load-bearing; Feinberg &
  Ding (arXiv:2211.07288) show the augmented operator has a NON-INFORMATIVE fixed point on
  the practical space of bounded functions (a direct warning for naive deep-RL use);
  Godbout & Durand (arXiv:2507.14005, 2025) prove no single policy is uniformly optimal
  across all alpha; and NO published clean deep discrete-action off-policy implementation
  exists — modern work (e.g., Worst-Cases Policy Gradients) explicitly engineers around the
  augmentation. Also corrected: ORAAC does NOT use this augmentation (it is
  distributional-critic + risk-sensitive action selection). Status: the only principled
  static-CVaR path inside current constraints, but a research contribution, not a drop-in.

## S2.4 Refuted in Sweep 2 (0-3 votes — do not build on)

1. "Soft-risk annealing alone is sufficient" — killed AGAIN (both sweeps agree).
2. "A modified distributional Bellman operator recovers principled CVaR training" — killed;
   **downgrades Sweep-1's C9** (the impossibility half stands; the advertised fix is unproven).
3. "Blindness-to-success directly matches the IQN case" — killed as overreach (analogy only).
4. "QR-iCVaR empirically underperformed static-SRM on the benchmark" — the QR-SRM *empirical
   dominance* claim killed in BOTH sweeps (mechanism confirmed; one author-run benchmark,
   5 seeds, overlapping std devs). Calibrate C1 accordingly.
5-6. CVaR sample-complexity scaling claims (1/tau² data-cost and matching lower bound) — killed.
7. Knife-edge risk-averse-DQN toy-MDP claim — killed.
8. "Action-gap collapse applies to distributional RL as stated" — killed as stated (h→0
   asymptotic; doesn't directly cover daily decisions). Tempers Sweep-1's C4 rationale but
   does not contradict clipped-AL's own optimality proofs.
9. "Pure CVaR gradients provably vanish → do-nothing policy" (secondary-source phrasing) — killed.

## S2.5 Reconciled recommendation (both sweeps + refutations)

**Tier A (convergent across independent sweeps, verified):**
1. **Static spectral risk / Mean-CVaR objective** (QR-SRM structure confirmed in both;
   empirical dominance unproven — treat as hypothesis to pre-register, not fact) — possibly
   via the **ECRM augmented-action** trick for a principled implementation.
2. **CeSoR done right:** alpha annealing 1→target PLUS IS-corrected adverse-CONDITION
   episode sampling — replacing the inverted, biased success-boost.
3. **State the objective:** whatever ships, adopt a formulation with a defined optimum
   (static CVaR via state augmentation, or ECRM) — the current recipe provably has none.

**Tier B (single-sweep, promising):** DSUP superiority distribution (S2), ORAC optimistic
exploration (S1), clipped-AL gap operator (S1, rationale tempered), UPER replay (S1),
BBF levers (S1), IQN-critic+PPO hybrid (S1, thesis-framing decision).

**Dropped/downgraded by adversarial evidence:** Lim-Malik operator *fix* (C9), plain
soft-risk-only annealing as a standalone fix, any claim that QR-SRM is empirically proven.

**Sweep-2 verdict on the gates themselves:** the two tests are NOT mis-posed — but no
well-defined objective underlies the current recipe, so passing both cannot currently be
*expected*; the fix belongs in the risk objective, not in more steps or replay tuning.

## S2.6 Sweep-2 primary sources

- Iterated CVaR RL: arxiv.org/abs/2206.02678; arxiv.org/abs/2211.07288; arxiv.org/abs/2111.06803
- Lim & Malik (NeurIPS 2022): openreview.net/forum?id=wSVEd3Ta42m
- QR-SRM spectral risk: arxiv.org/abs/2501.02087 · ECRM: arxiv.org/abs/2301.05981
- CeSoR: arxiv.org/abs/2205.05138
- Superiority distribution (DSUP): arxiv.org/abs/2410.11022
- Distributional sample-complexity parity: arxiv.org/abs/2502.14172; arxiv.org/abs/2402.07598
- Practitioner corroboration: arxiv.org/abs/2501.04421; arxiv.org/abs/2205.05614; arxiv.org/abs/2010.12245

---
---

# VERIFICATION LOG — 2026-07-21 pass (12 source-level adversarial verifiers)

Every claim previously labeled [UNVERIFIED] was checked against its primary source (papers
fetched and read directly; adversarial stance: attempt to refute first). Outcomes:

**Confirmed (labels updated in place):** BTR (5/5 claims), BBF/SPR (4/4), UPER (3/3),
PG-Rainbow (4/4, one wording-strength caveat), Tail-Safe Hedging (3/4), TQL (3/3 with scope
qualifications), ERL survey (2/2, continuous-control scope flag), risk-scheduling SMAC
numbers (with direction correction), AC-SRM structure + state augmentation (2/3),
natural-gas design-match + C51 >32% (tempered), IEQN Huber-collapse (confirmed as published,
evidence graded thin), Bäuerle-Ott augmentation (theory confirmed; heavy practical caveats).

**Deleted as unsupported/fabricated:**
1. "cosine schedule" for Tail-Safe's alpha annealing — the word appears nowhere in the paper;
   schedule form is unspecified. (Was in C2 and the sweep-2 summary.)
2. AC-SRM "consistently outperforms CVaR-AC/Tamar/Dabney-IQN baselines" — those baselines
   were never run in the paper's experiments.
3. "risk aversion monotone in alpha" (natural-gas study) — refuted for QR-DQN by the paper's
   own data; retained only as a general trend for C51/IQN.

**Corrections recorded:** risk-scheduling anneals risk-seeking→risk-averse (not neutral→averse);
ORAAC does not use Bäuerle-Ott augmentation; TQL's guarantee is monotone policy improvement,
not global convergence; natural-gas evidence = 4 backtest windows, one instrument, zero costs,
no seeds, post-hoc Sharpe shaping.

No [UNVERIFIED] labels remain in this document.
