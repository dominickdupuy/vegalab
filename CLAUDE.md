# CLAUDE.md
## SPX Options Spread-Selection RL — Repo Guardrails

This file is the standing contract for every Claude Code session in this repo. It consolidates the invariants from the eight phase briefs (`PHASE1_BRIEF.md` ... `PHASE8_BRIEF.md`). **Read it before doing anything. Do not violate an invariant to make a task easier; if an invariant is in the way, stop and ask.**

---

## What this project is

A reinforcement-learning study that learns **when to deploy which options spread structure as a function of market regime** on **SPX** (European, cash-settled, daily/EOD decisions). The agent's policy is intentionally **rich and non-closed-form**; the investor takeaway is a **post-hoc, lossy, human-readable regime→structure map** plus **per-regime return distributions** from a distributional critic. The headline contribution is that a **distributional, CVaR-risk-sensitive agent beats an expected-value agent on tail-adjusted metrics**, trained on a **synthetic curriculum** and validated on real OptionMetrics data and on **held-out market models**.

Primary agent (pre-committed): the **CVaR / IQN** distributional agent. PPO and the risk-neutral distributional agent are baselines/comparisons.

---

## Phase map and status

| Phase | What it builds | Hard gate to pass |
|---|---|---|
| 1 | Env, pricing, action library, costs, margin, composable reward | closed-form P&L/payoff tests; Wave-0 plumbing sane |
| 2 | PPO baseline + shared harness | **Wave-0 no-edge gate** (FLAT-dominant; pure-PnL indifference) |
| 3 | Distributional agent (QR-DQN -> IQN) + CVaR selection | in-vitro fat-tail proof; Wave-0 no-edge re-confirmed |
| 4 | Synthetic curriculum Waves 1–6, domain randomization | per-wave GV + pre-registered BV + forgetting check |
| 5 | Real OptionMetrics fine-tune + walk-forward | beats baselines on **tail-adjusted** OOS metrics (leak-free) |
| 6 | Held-out generators (rough Bergomi/SABR/GARCH) | **graceful** zero-shot degradation; CVaR tail edge survives |
| 7 | VIPER distillation -> regime map + per-regime distributions | **economic-sensibility** gate; fidelity reported |
| 8 | Robustness battery, ablations, SHAP, write-up | every reward term earns its place; ensemble dispersion |

> Update this table's status as phases complete. Work **one phase at a time**; pass the gate before promoting.

---

## Cross-cutting invariants (never violate)

### Determinism and randomness
- One explicit, seeded `numpy.random.Generator` (and seeded torch). **Never** call global `np.random.*`. Same seed => identical trajectory (tested).

### No look-ahead, ever
- Features and normalization are **point-in-time / causal**: anything at time `t` uses only data at or before `t`. Trailing windows only. On real data, obs-normalizer stats are fit on **training folds only**.

### Architecture: dependency injection
- The env **constructs none** of its collaborators. The `PriceGenerator`, `CostModel`, `MarginModel`, `RewardFunction`, and `ObservationBuilder` are **injected**. Every one of them is swapped in a later phase; never hard-code them.

### The surface keystone (the sim-to-real bridge)
- **Every generator (synthetic and real) emits a standardized IV surface** on the OptionMetrics grid (deltas 10–90, maturities 10d–2y), plus the spot. The env derives tradeable prices and features from the surface. **Synthetic and real data therefore share one env, unchanged.** Price the **surface once per simulated day** (cache), never per option per step.

### Hidden internals
- The observation is built **only** from the surface and the path. The agent **never** sees true generator parameters, the latent variance, or the regime label. Assert their absence (tested in Waves 5+).

### Reward and comparison discipline
- The reward is a **composable, weighted sum** of components (`MTMPnL`, `MarginNormalizer`, `DifferentialSharpe`, `Sortino`, `CVaRPenalty`). **Setting a weight to 0 disables a term** => ablations are config changes, not code changes.
- **Entropy is agent-side (PPO objective), never in the env reward.**
- **Hold the reward fixed, vary only the algorithm.** PPO and the distributional agent consume the **identical** composite reward; the only difference is expected-value vs CVaR action selection.
- The **shared harness** (`EnvFactory`, `Evaluator`, `MetricSuite`, `Agent`) is reused **byte-for-byte** across PPO and the distributional agents. **Never edit the shared eval/env to fit one agent.** Algorithm-specific code lives only in each trainer/network.

### Markovianity of the risk reward
- The differential-Sharpe and CVaR-penalty terms are history-dependent. The observation **must expose** the risk-reward state (Sharpe EMAs, drawdown, margin) so the risk-adjusted reward is Markovian, or use frame-stacking. Do not fight hidden state.

### Distributional-agent specifics
- **Tail resolution scales with alpha:** size QR-DQN quantiles so `floor(alpha*N) >= ~10` (e.g. N=200 for alpha=0.05); use **IQN** (`tau ~ U(0,alpha)`) for small-alpha CVaR.
- **CVaR under-exploration ("blindness to success"):** explore with a risk-neutral behavior policy (or anneal alpha) plus an epsilon floor.
- **Static CVaR is time-inconsistent.** The nested CVaR-greedy bootstrap optimizes an iterated risk measure, not static total-return CVaR. State it as a caveat; do not pretend otherwise.
- **No recurrent off-policy** (no R2D2-style recurrent QR-DQN/IQN). Use **frame-stacking** for the distributional agent's memory; LSTM only for PPO if needed.

### Curriculum discipline (Phase 4)
- **One new generative feature per wave.** Never combine.
- **Generator validation before agent training:** prove the sim produces the intended stylized fact (no agent) before training on it. This catches calibration sign errors (e.g. `rho` must be negative).
- **Pre-register the behavioral prediction and its test BEFORE the training run** (timestamped commit). A prediction written after results is not a test.
- **Warm-start each wave from the previous checkpoint; rehearse earlier waves; re-evaluate for catastrophic forgetting.**
- **VRP is a P-vs-Q measure difference** (simulate under P, price the surface under a VRP-adjusted Q), not a bolt-on.
- **Current Phase-4 status:** `phases/PHASE4_GATE_REPORT.md` records the surface
  foundation, Wave 1 generator validation (`GV_1`), and BV_1 validation tooling
  as present. Do not start Wave 2 until Wave 1 trained-agent behavioral
  validation (`BV_1`) and forgetting check (`FF_1`) are passed.

### Frozen-agent invariants
- **Held-out generalization (Phase 6) and distillation (Phase 7) take ZERO gradient steps** on the held-out / distillation data. The agent is frozen. Assert it.

### Real-data and evaluation discipline (Phase 5)
- **Hyperparameters are frozen from the synthetic phase.** Real data is light fine-tuning + evaluation only. **No hyperparameter search on test folds.**
- **Walk-forward with purge + embargo**, episodes aligned to fold boundaries; many folds, not one split. **Deflated/probabilistic Sharpe** for significance.
- **Report zero-shot AND fine-tuned.** Lead the sim-to-real story with zero-shot.
- **Pre-commit the primary (CVaR) agent** before inspecting real results (no selection bias).
- **Current Phase-5–8 status:** infrastructure and tests are present for real-data
  replay, walk-forward, held-out generalization, interpretation, and robustness.
  Full empirical gates remain pending until OptionMetrics data and final trained
  checkpoint ensembles are available. User-local WRDS auth is confirmed, but the
  Codex sandbox cannot open TCP 9737 to WRDS, so WRDS extraction must be run from
  a normal local shell into `data/`. See `PHASE5_GATE_REPORT.md` through
  `PHASE8_GATE_REPORT.md`.

### Reporting discipline (everywhere, enforced in Phase 8)
- **Every headline number is an ensemble claim:** mean +/- dispersion across seeds (and folds). **Never a single hero run.**
- **The win condition is tail-adjusted (Sharpe, Sortino, CVaR/ES, max drawdown, tail ratio), not raw mean.** A smaller mean with a much better tail is success.
- **Label every exhibit synthetic vs real.** Never present synthetic as real-market evidence.
- **Report break-even cost; survive a stated cost multiple (target >= 1.5–2x).**
- **The distilled map is a lossy summary; always report its fidelity.** It approximates, it does not equal, the policy.
- **State limitations and negative results plainly** (EOD-only, single-underlying SPX, synthetic-realism cap, few real tail events, static-CVaR time-inconsistency).

### Scope of the instrument
- **SPX: European, cash-settled => no early-assignment or pin modeling. EOD daily => no intraday management.** These are scope limits, stated, not bugs to fix.

---

## Hard-stop conditions (stop and report; do not work around)

- **Any agent "makes money" on fair-IV Wave 0.** This is a leak/bug in the env (pricing inconsistency, cost sign, look-ahead, premium accounting), not a discovery. Stop and debug Phase 1. The same leak fabricates fake skill in every later wave.
- **A generator fails its validation** (does not produce the intended stylized fact). Fix the generator before training any agent on it.
- **A held-out family is not actually out-of-distribution** (structural-distance diagnostic fails). It is not a valid test; fix or drop it.
- **Distilled rules are economic nonsense despite good PnL.** Reward-hacking alarm. Investigate against the Phase 6 held-out behavior and Phase 8 ablations; do not ship.
- **Real performance collapses despite clean synthetic results.** Report honestly and consider the **sim-to-real gap-study** reframing (still a strong contribution). Do not overfit to force a real-data win.
- **A reward-term ablation is inert.** Investigate (redundant or mis-weighted); do not silently keep the term.

---

## Quality gates (every session)

- `make check` must be green: **mypy --strict**, **ruff** (lint + format), **pytest** (with coverage).
- **TDD on correctness-critical numerics**: Black-Scholes, strike solver, template payoffs (vs analytic oracles), P&L, margin, quantile loss, CVaR computation, COS pricer (vs MC oracle / published benchmarks).
- Commit at **module granularity**, referencing the acceptance test / gate that now passes.
- No `# type: ignore` without a one-line justification.
- **Reuse the shared harness**; new algorithm code lives only in its own trainer/network.
- **Do not skip ahead** to a later phase's scope. Each brief states what is out of scope; if a task drifts there, stop and confirm.
- **Update this file** when an invariant or the phase status changes.

---

## Repository shape (grows by phase)

```
optspread/
  pricing/ instruments/ actions/ market/ portfolio/ costs/ margin/ reward/ envs/   # Phase 1
  agents/ (base, ppo/) training/ eval/                                              # Phase 2
  agents/distributional/ toys/                                                      # Phase 3
  market/(heston,bates,...) features/ curriculum/ agents/sequence/                  # Phase 4
  data/ evaluation/ baselines/ finetune/                                            # Phase 5
  market/(rough_bergomi,sabr,garch) evaluation/(generalization,structural_distance) # Phase 6
  interpret/                                                                         # Phase 7
  robustness/ attribution/ reporting/                                               # Phase 8
tests/   cli/   pyproject.toml   Makefile   CLAUDE.md   README.md
```

The load-bearing reusable spine: the **injected dependencies**, the **surface-driven env**, and the **shared `training/`+`eval/` harness**. Almost everything else is a swappable implementation behind an interface.
