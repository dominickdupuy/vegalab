# Phase 3 Implementation Brief for Claude Code
## SPX Options Spread-Selection RL — The Distributional, Risk-Sensitive Agent (the Contribution)

> **Scope of this phase.** Build the distributional value-based agent that is the thesis's main contribution: **QR-DQN first** (simpler, fixed quantiles), **then IQN** (implicit quantile network, continuous quantile sampling), both with **CVaR-based action selection**. Reuse the Phase 2 shared harness (`EnvFactory`, `Evaluator`, `MetricSuite`, `Agent`) unchanged so the eventual "distributional beats expected-value on tail metrics" comparison is apples-to-apples by construction. The deliverable is **not** the headline options result yet (that lands in Phase 4, Wave 3). The deliverable is: the distributional machinery, verified to recover known distributions; an **in-vitro proof** that the CVaR agent avoids a tail the expected-value agent chases; and a confirmation that the distributional agent also passes the Wave-0 no-edge gate.
>
> **Depends on:** Phase 2 complete and green. The PPO baseline trains stably, the Wave-0 no-edge gate passed (parts a and b), and the shared `training/`, `eval/`, and `agents/base.py` are reuse-ready.

---

## 0. Stack decisions (fixed)

- **Framework:** PyTorch 2.x (carried from Phase 2).
- **Algorithm family:** off-policy, value-based, distributional. **QR-DQN** (Dabney et al. 2018, AAAI) then **IQN** (Dabney et al. 2018, ICML). No third-party distributional-RL library; implement against the papers and cite them. Optionally cross-check against a reference implementation, but the reported agent is your own so it shares the harness.
- **Replay:** uniform replay buffer (simple numpy ring buffer). Prioritized replay is **off by default** — PER interacts awkwardly with the quantile loss (the TD-error proxy is ambiguous when the target is a distribution); add only as a documented experiment if needed.
- **Exploration:** epsilon-greedy with a floor (value-based, so there is no policy-entropy bonus). See section 4 on the CVaR under-exploration mitigation.
- **Everything else** (Pydantic config, mypy strict, ruff, pytest, explicit RNG, causal obs normalization) carries over unchanged.
- **CPU is fine** for the toys and Wave 0; GPU optional.

---

## 1. Design principles specific to this phase

1. **Reuse the shared harness verbatim.** Import `EnvFactory`, `Evaluator`, `MetricSuite`, and the `Agent` protocol from Phase 2 with zero changes. The distributional agent owns only a new **off-policy trainer** and a new **network/loss**. If you find yourself editing the evaluator or env to fit this agent, stop — that breaks the comparison the whole thesis rests on.
2. **Prove the contribution in vitro before scaling.** Build a tiny fat-tail toy with **known ground-truth return distributions** and show the CVaR agent avoids the catastrophic-tail action the expected-value agent selects. This de-risks everything: if the machinery is broken, you find out in a two-line toy, not buried inside Wave 3 options results.
3. **Hold the reward fixed; vary only the algorithm and the acting risk measure.** Both PPO (Phase 2) and this agent consume the identical Phase 1 composite reward. The distributional agent's only added lever is that it acts on a **CVaR functional of the learned return distribution** rather than the mean. Keep that the single moving part.
4. **Resolution in the tail is a first-class design constraint.** CVaR at small alpha averages only the lowest quantiles. With QR-DQN's fixed N quantiles you must choose N so that `floor(alpha * N)` is large enough (target ~10) for a stable tail estimate; this is precisely why IQN, which samples quantiles continuously, is the better CVaR vehicle and comes second.
5. **Static CVaR is time-inconsistent; say so.** The CVaR-greedy bootstrap optimizes a *nested/iterated* risk measure, not the static CVaR of total episode return. This is a known property, not a bug. Treat it as a discussion-section caveat with a citation, do not pretend to have solved it.
6. **Modest capacity, multi-seed, reported dispersion** — same overfitting discipline as Phase 2.

---

## 2. Repository additions (extend the repo)

```
optspread/
  agents/
    distributional/
      network_qrdqn.py      # QRDQNNetwork: state -> (num_actions x N) fixed-quantile values
      network_iqn.py        # IQNNetwork: (state, tau) -> (num_actions) quantile values; cosine tau embedding
      quantile_loss.py      # quantile_huber_loss(pred, target, taus, kappa)  (pure, tested)
      risk.py               # RiskMeasure: mean (alpha=1), cvar(alpha); from_quantiles(...) and from_samples(...)
      replay.py             # UniformReplayBuffer (numpy ring buffer)
      qrdqn_agent.py        # QRDQNAgent implementing Agent protocol (act via RiskMeasure over quantiles)
      iqn_agent.py          # IQNAgent implementing Agent protocol (act via U(0,alpha) sampling)
      trainer.py            # DistributionalTrainer: off-policy loop, target net, eps-greedy, works for both
      config.py             # QRDQNConfig, IQNConfig (Pydantic)
  toys/
    fat_tail_bandit.py      # K-armed env: one high-mean catastrophic-tail arm; known distributions
    fat_tail_mdp.py         # short-horizon MDP version (tests the CVaR bootstrap over multiple steps)
  cli/
    train_distributional.py # CLI: train QR-DQN or IQN on a wave/toy
    compare.py              # load PPO + distributional checkpoints -> head-to-head table via shared MetricSuite
tests/
  test_quantile_loss.py     # vs hand-computed small example; asymmetry around tau correct
  test_risk_measure.py      # cvar/mean from quantiles & samples vs analytic on known distributions
  test_distribution_recovery.py  # QR-DQN & IQN regress a known mixture; recovered quantiles & CVaR within tol
  test_iqn_tau_embedding.py # cosine embedding shape; outputs vary with tau; approx monotonic in tau
  test_replay.py            # ring buffer correctness, sampling, no aliasing
  test_fat_tail_bandit.py   # mean-greedy picks tail arm; CVaR_alpha-greedy avoids it  (the in-vitro proof)
  test_fat_tail_mdp.py      # CVaR bootstrap policy avoids the tail; risk-neutral does not
  test_compare_harness.py   # distributional agent runs through the SAME Evaluator/MetricSuite as PPO
```

`toys/` and the in-vitro tests are not throwaway: the fat-tail bandit result is a figure in the thesis (the controlled demonstration of the mechanism).

---

## 3. The math (implement exactly; these are the tested cores)

**Quantile Huber loss.** Huber: `L_kappa(u) = 0.5*u^2 if |u|<=kappa else kappa*(|u|-0.5*kappa)`. Quantile Huber:
```
rho_tau^kappa(u) = | tau - 1{u < 0} | * L_kappa(u) / kappa
```
For QR-DQN with predicted quantiles theta_i at fractions tau_i (i=1..N) and target quantiles Ttheta_j (j=1..N'):
```
loss = (1/N') * sum_j sum_i  rho_{tau_i}^kappa( Ttheta_j - theta_i )      # u = target - prediction
```
Use `tau_i = (2i - 1) / (2N)` (the fixed quantile midpoints) for QR-DQN.

**Distributional Bellman target.** For transition `(s, a, r, s', done)` and bootstrap action `a*`:
```
Ttheta_j = r + gamma * (1 - done) * theta_j(s', a*; target_net)
a* = argmax_{a'}  RiskMeasure( Z(s', a'; target_net) )
```
- Risk-neutral agent: `RiskMeasure = mean` over quantiles (this is the "expected-value distributional" agent, the distributional analogue of PPO).
- Risk-sensitive agent: `RiskMeasure = CVaR_alpha`. This makes the learned distribution the return distribution under the CVaR-greedy policy (dynamic, nested CVaR — see the time-inconsistency caveat).

**CVaR from quantiles (QR-DQN).**
```
CVaR_alpha(Z(s,a)) ~= mean{ theta_i(s,a) : tau_i <= alpha }
```
Choose N so `floor(alpha*N) >= ~10`. For alpha = 0.05 use **N = 200**; for alpha = 0.1 use **N >= 100**. (This is the "too few quantiles -> garbage tail" pitfall made concrete.)

**IQN.** State embedding `psi(s)`; quantile embedding via cosine basis
```
phi(tau)_k = ReLU( sum_{j=0}^{n-1} cos(pi * j * tau) * w_{kj} + b_k ),   n ~ 64 cosines
```
combine multiplicatively `psi(s) ⊙ phi(tau)`, then a head -> quantile value per action. Sample `tau ~ U(0,1)` for prediction (N samples) and target (N' samples); quantile-Huber over the sampled fractions.
**CVaR acting in IQN:** sample `tau ~ U(0, alpha)` (K samples, e.g. K=32), average per action, argmax. Expected-value acting is `alpha = 1`. This continuous lower-tail sampling is why IQN gives smooth CVaR at small alpha where QR-DQN is coarse.

---

## 4. Agent / trainer specification

- **Network capacity:** modest MLP trunk (match Phase 2 scale; overfitting control). QR-DQN head outputs `num_actions * N`; IQN head outputs `num_actions` for a given tau. FLAT stays action index 0, always legal.
- **Target network:** hard update every `target_update_interval` steps, or Polyak `tau_polyak ~ 0.005`.
- **Replay warmup:** `learning_starts` (e.g. 20k–50k steps) before updates.
- **Exploration + the CVaR under-exploration trap.** Pure CVaR-greedy can be "blind to success" — it over-avoids actions whose early sampled tail looks bad, under-exploring genuinely good ones (the value-based analogue of the Greenberg et al. 2022 CVaR-policy-gradient pathology). Mitigations, in order of preference:
  1. **Explore with a risk-neutral (mean-greedy) behavior policy**, evaluate/deploy with CVaR-greedy (decoupled behavior/target). Clean and effective.
  2. **Anneal alpha** from 1.0 down to the target over training (risk-neutral early for coverage, risk-averse late).
  3. **Blended acting** `(1-beta)*mean + beta*CVaR_alpha`.
  Always keep an **epsilon-greedy floor** (e.g. 0.02). Document which mitigation you use.
- **Shared trainer:** one `DistributionalTrainer` parameterized by network + loss + risk measure serves both QR-DQN and IQN. It pulls env/seeding/logging/eval from the shared harness.

**Starting hyperparameters (tune from here):**

| Param | QR-DQN | IQN | Notes |
|---|---|---|---|
| quantiles N (pred) | 200 | 32 sampled | N=200 so alpha=0.05 tail has ~10 atoms |
| N' (target) | 200 | 32 sampled | |
| K (acting samples) | n/a | 32 | IQN CVaR sampling |
| cosine embed n | n/a | 64 | |
| huber kappa | 1.0 | 1.0 | |
| replay size | 5e5 | 5e5 | |
| batch size | 64 | 64 | |
| gamma | 0.99 | 0.99 | same as PPO, consistent episode length |
| learning_rate | 5e-5 | 5e-5 | distributional methods like lower LR |
| target_update | 2000 steps | 2000 steps | or Polyak 0.005 |
| learning_starts | 30000 | 30000 | |
| train_freq | 1 / step | 1 / step | |
| epsilon | 1.0 -> 0.02 | 1.0 -> 0.02 | annealed, with floor |
| CVaR alpha | {0.1, 0.05} | {0.1, 0.05} | plus alpha=1.0 (risk-neutral) |

---

## 5. Validation gates (the definition of done)

In order. Do not advance past a failing gate.

**G1 — Quantile machinery (unit).** `quantile_huber_loss` matches a hand-computed small example, including the correct asymmetry around `tau`. `RiskMeasure.cvar`/`.mean` from quantiles and from samples match the analytic mean and CVaR of known distributions (normal, lognormal, a two-point mixture) within tolerance.

**G2 — Distribution recovery.** Train QR-DQN and IQN to regress a known fixed return distribution (e.g. a Gaussian mixture as a one-step target). Recovered quantiles and the recovered CVaR_alpha match the analytic values within tolerance. (If the network cannot even recover a static distribution, the Bellman version will not work.)

**G3 — Fat-tail bandit (the in-vitro proof).** K arms with known distributions; one arm has the **highest mean but a catastrophic left tail**, another has slightly lower mean with bounded downside. Assert:
- mean-greedy (alpha=1) selects the high-mean tail arm,
- CVaR_alpha-greedy (alpha in {0.1, 0.05}) selects the bounded-downside arm.
This is the controlled demonstration of the central claim and a thesis figure.

**G4 — Fat-tail MDP.** Short-horizon version where the tail action's damage realizes a few steps later. The CVaR-bootstrap policy avoids it; the risk-neutral policy does not. Confirms the **distributional Bellman bootstrap under CVaR-greedy** works over multiple steps, not just one.

**G5 — Wave-0 no-edge (method-agnostic, reuse the Phase 2 gate).** On fair-IV, zero-drift Wave 0 the distributional agent must be **FLAT-dominant**. Prediction: it should be *more* strongly FLAT than PPO, because CVaR(FLAT)=0 while CVaR(any active structure)<0 on fair-IV. Multi-seed, reported dispersion. A distributional agent that finds edge on Wave 0 is the same Phase 1 bug signal — stop and debug.

**G6 — Fair-comparison harness.** The distributional agent runs through the **same** `Evaluator`/`MetricSuite`/`EnvFactory` as PPO. `compare.py` emits a head-to-head table (mean return, Sharpe, Sortino, CVaR/ES, max drawdown, tail ratio) for PPO vs risk-neutral-distributional vs CVaR-distributional on a given wave. On Wave 0 this is intentionally a **null result** (all flat) — its purpose here is to prove the comparison pipeline is wired correctly. The non-null headline arrives at Wave 3 in Phase 4.

**Reproducibility:** every gate result holds across at least 3–5 seeds with dispersion reported.

---

## 6. Pitfalls to engineer against

- **Too-coarse tail.** `floor(alpha*N)` too small in QR-DQN gives a noisy CVaR; size N to the smallest alpha you report, or use IQN.
- **Quantile crossing.** Neither QR-DQN nor IQN enforces monotone quantiles; usually benign. Monitor the fraction of crossings; if material, sort quantiles before computing CVaR or note it. Do not silently ignore.
- **CVaR under-exploration / blindness to success.** Use the decoupled risk-neutral behavior policy or alpha-annealing plus an epsilon floor (section 4). Symptom: the agent collapses early onto an over-cautious action and never revisits.
- **Time-inconsistency of static CVaR.** The nested CVaR-greedy target is not the static CVaR of total return. Caveat + citation, not a silent assumption.
- **Target-network staleness vs replay non-stationarity.** Tune `target_update` and `train_freq` together; watch the quantile loss and the mean-quantile (Q) explained variance on a learnable toy.
- **Reward double-counting confusion.** The Phase 1 composite reward already includes a *soft* CVaR penalty. The distributional agent adds a *native* CVaR action objective on top of that same reward. Keep both agents on the identical reward for the headline (apples-to-apples), and run the isolation ablation in section 7 to show the native objective carries the tail control on its own.
- **Bypassing the harness.** Any "quick" custom eval for this agent silently invalidates the comparison. Always go through the shared `Evaluator`.

---

## 7. A clean isolation ablation to build now (strengthens the contribution)

Run two reward configurations and report both:
- **(A) Full composite reward** (includes the soft scalar CVaR penalty) for *both* PPO and the distributional agent. This is the headline, reward held identical.
- **(B) Scalar CVaR penalty weight = 0**, so the *only* tail control is the distributional agent's native CVaR objective. If the CVaR-distributional agent still controls the tail under (B) while PPO (now with no tail term at all) does not, you have shown the native distributional objective alone suffices. That is a strong, clean result and exactly the kind of ablation a committee rewards.

(On Wave 0 both configs are null; wire the ablation now, harvest it at Wave 3.)

---

## 8. Build order (do not reorder)

1. `agents/distributional/quantile_loss.py` + `risk.py` + tests `test_quantile_loss.py`, `test_risk_measure.py` (G1).
2. `agents/distributional/replay.py` + `test_replay.py`.
3. `agents/distributional/network_qrdqn.py`; `test_distribution_recovery.py` for QR-DQN (G2).
4. `agents/distributional/trainer.py` (off-policy loop, target net, eps-greedy, decoupled behavior option) + `qrdqn_agent.py` + `config.py`.
5. `toys/fat_tail_bandit.py`, `toys/fat_tail_mdp.py` + `test_fat_tail_bandit.py`, `test_fat_tail_mdp.py` (G3, G4).
6. Wave-0 no-edge re-run via the reused gate (G5).
7. `agents/distributional/network_iqn.py` + `iqn_agent.py` + `test_iqn_tau_embedding.py`; re-run G2/G3/G4/G5 for IQN.
8. `cli/train_distributional.py`, `cli/compare.py` + `test_compare_harness.py` (G6).
9. Multi-seed runs, dispersion, isolation-ablation wiring (section 7), write-up.

**Update `CLAUDE.md`** with: the harness-reuse rule (no edits to shared eval/env), the QR-DQN-then-IQN ordering and the N-vs-alpha tail-resolution rule, the CVaR under-exploration mitigation in use, the static-CVaR time-inconsistency caveat, and the in-vitro fat-tail gates as prerequisites to any options-wave training.

---

## 9. Process instructions for Claude Code

- Implement against the QR-DQN and IQN papers; cite them in comments. Restructure into the module layout; do not drop in a monolithic reference file.
- Keep mypy strict and ruff green via `make check` after each module.
- Commit at module granularity, referencing the gate (G1–G6) that now passes.
- **Do not edit** `training/`, `eval/`, or `agents/base.py` from Phase 2 except additively and backward-compatibly; the PPO path must stay green.
- **Do not start any options-wave curriculum beyond Wave 0.** Waves 1–6, domain randomization, and the held-out generator are Phase 4. If a task drifts there, stop and confirm.
- Log to TensorBoard/W&B: quantile loss, mean-quantile (Q) explained variance, epsilon, fraction of quantile crossings, per-action frequency, full eval return distribution, and the CVaR/ES/maxDD metrics from the shared suite.
- End with a short report: G1–G6 outcomes with per-seed dispersion, the fat-tail bandit/MDP figures (the in-vitro proof), the Wave-0 no-edge confirmation for both QR-DQN and IQN, a confirmation that `compare.py` produces a valid (null-on-Wave-0) head-to-head via the shared suite, and an explicit statement that the project may proceed to Phase 4 / Wave 1.

---

### Ready-to-paste kickoff prompt

> Implement Phase 3 of the project as described in `PHASE3_BRIEF.md`, building on the completed Phase 2 repo. Reuse the shared `training/`, `eval/`, and `agents/base.py` from Phase 2 unchanged (additive, backward-compatible edits only). Implement QR-DQN first, then IQN, both with CVaR action selection, against the Dabney et al. papers. Proceed through gates G1–G6 in order (section 5): quantile-loss and risk-measure unit tests, distribution recovery, the fat-tail bandit and MDP in-vitro proofs that CVaR avoids the tail the mean chases, the Wave-0 no-edge re-run via the reused gate, and the fair-comparison harness producing a valid null head-to-head against PPO on Wave 0. Size N so the alpha=0.05 tail has ~10 quantiles for QR-DQN; use U(0,alpha) sampling for IQN. Mitigate CVaR under-exploration with a decoupled risk-neutral behavior policy or alpha-annealing plus an epsilon floor, and document the choice. Wire the section-7 isolation ablation but expect nulls on Wave 0. Do not start any curriculum wave beyond Wave 0. Keep mypy strict and ruff green, hold results across at least 3 seeds with reported dispersion, and end by stating whether the project may proceed to Phase 4.
