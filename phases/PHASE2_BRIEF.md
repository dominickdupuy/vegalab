# Phase 2 Implementation Brief for Claude Code
## SPX Options Spread-Selection RL — PPO Baseline & the Wave-0 No-Edge Gate

> **Scope of this phase.** Build the PPO baseline agent on top of the Phase 1 environment, plus the shared training/evaluation harness that the Phase 3 distributional agent will reuse. Train on **Wave 0 only**. The deliverable is not "a profitable agent" — it is a **proof that the agent finds no systematic edge on Wave 0**, which validates that the Phase 1 environment is honest. This is the single most important checkpoint in the entire research program. If PPO makes money on fair-IV, zero-drift Wave 0, you have a bug in Phase 1, not a discovery, and you stop here until it is fixed.
>
> **Depends on:** Phase 1 complete and green (env passes `check_env`, deterministic, P&L/margin/cost/reward tests pass, `smoke_run.py` reproduces the Wave-0 no-edge numbers for scripted agents).

---

## 0. Stack decisions (fixed)

- **Deep learning:** PyTorch 2.x. This is where it enters the project; pin the version.
- **PPO basis:** Adapt **CleanRL's single-file `ppo.py`** as the correctness reference (readable, citable, correct), restructured into the repo's module layout. Do **not** use Stable-Baselines3 as the baseline: it is opaque to instrument and its internals will not be reused by the custom QR-DQN/IQN agent in Phase 3, which would make the headline comparison less clean. (SB3 is acceptable only as a throwaway sanity cross-check, not as the reported baseline.)
- **Vectorized envs:** Gymnasium vector API (`SyncVectorEnv` for determinism; `AsyncVectorEnv` only if throughput demands it).
- **Experiment tracking:** TensorBoard (minimum) or Weights & Biases. Log scalars *and* the full eval return distribution and per-action histograms.
- **Everything else** (config via Pydantic, mypy strict, ruff, pytest, uv/Poetry, explicit RNG) carries over from Phase 1 unchanged.
- **CPU is sufficient** for Wave 0 (small MLP, cheap env). Prefer CPU for full determinism; treat GPU as optional.

---

## 1. Design principles specific to this phase

1. **Share the eval, env, and metrics; not the training loop.** PPO is on-policy (rollout-based) and the Phase 3 agents are off-policy (replay-based), so a single `Agent.train()` abstraction is the wrong cut. Instead, build a shared `EnvFactory`, `Evaluator`, and `MetricSuite` that **both** phases use identically, and let each algorithm own its trainer. The credibility of the eventual "distributional beats expected-value" claim rests entirely on the env, the reward, the eval episodes, the seeds, and the metrics being byte-for-byte identical across the two agents. Build that shared spine now.
2. **Hold the reward fixed across agents; vary only the algorithm.** PPO uses the **full Phase 1 composite reward** (dense MTM P&L, differential Sharpe, optional Sortino, the soft CVaR penalty, margin normalization). Phase 3's distributional agent will use the *same* composite reward for its dense signal and additionally select actions by a CVaR functional of the learned return distribution. That isolates the contribution to "soft tail penalty in a scalar reward" vs "native tail objective over the distribution," rather than confounding reward design with algorithm.
3. **Make the risk-adjusted reward Markovian w.r.t. the observation.** The differential-Sharpe and CVaR-penalty terms are history-dependent (they depend on recent returns and current drawdown). A feed-forward policy cannot optimize them unless the relevant state is observable. **Confirm the `ObservationBuilder` exposes the reward-relevant state** (the differential-Sharpe EMAs A and B or the current Sharpe estimate, current drawdown, margin used, position encoding, days-to-expiry). If Phase 1 omitted these, extend the observation schema here. Otherwise the agent is fighting hidden state and the no-edge result will be noisy and uninterpretable.
4. **Modest network capacity is a deliberate overfitting control,** not laziness. Small MLP, documented. Capacity grows only if Wave 0 cannot be fit, which would itself be a red flag.
5. **Causal normalization only.** Observation running mean/std must update online from the stream, never precomputed over a full (future-containing) dataset. Build the habit now even though Wave 0 is synthetic; Phase 5 real-data eval depends on it.
6. **Determinism for eval, stochasticity for rollouts.** Training samples from the policy; evaluation can use either, but fix eval seeds and report both the mean and the full distribution. Eval path seeds must be disjoint from training path seeds.

---

## 2. Repository additions (extend the Phase 1 repo)

```
optspread/
  agents/
    base.py                 # Agent Protocol: act(obs, deterministic) -> action; load/save
    ppo/
      network.py            # ActorCritic MLP: shared/separate trunk, Categorical head, orthogonal init
      buffer.py             # RolloutBuffer: obs/actions/logprobs/rewards/dones/values; GAE computation
      ppo_agent.py          # PPOAgent implementing Agent protocol (wraps network for inference)
      trainer.py            # PPOTrainer: rollout collection + clipped update loop
      config.py             # PPOConfig (Pydantic): all hyperparameters below
  training/
    env_factory.py          # EnvFactory: builds (vectorized) SpreadEnv with injected deps + seeds
    vec.py                  # vector-env construction + per-env seeding
    normalize.py            # CausalObsNormalizer (Welford), optional ReturnNormalizer (off by default)
    harness.py              # TrainHarness: wires trainer + evaluator + logging + checkpointing
    logging.py              # MetricLogger: scalars, histograms, full return arrays
    seeding.py              # global seeding protocol (python/numpy/torch) + per-seed run dirs
  eval/
    evaluator.py            # Evaluator: runs policy over fixed eval seeds -> EvalReport
    metrics.py              # MetricSuite: mean PnL + CI, Sharpe, Sortino, CVaR/ES, maxDD, turnover,
                            #              per-action frequency, full return distribution
    no_edge_gate.py         # the Phase-2 gate: statistical test of FLAT-dominance / zero-edge
  cli/
    train.py                # CLI: train PPO on a given wave (Wave 0 here)
    evaluate.py             # CLI: load checkpoint, run Evaluator, emit EvalReport + plots
tests/
  test_ppo_network.py       # shapes, init scales, action distribution validity
  test_gae.py               # GAE against a hand-computed small example
  test_ppo_update.py        # loss components finite; entropy decreases but not to 0 instantly;
                            #   overfits a trivial 1-state bandit to the optimal action
  test_evaluator.py         # deterministic eval, disjoint train/eval seeds enforced
  test_no_edge_gate.py      # gate logic on synthetic return streams (passes zero-edge, fails planted-edge)
  test_harness_smoke.py     # a RandomAgent runs end-to-end through the shared harness
```

The shared `training/`, `eval/`, and `agents/base.py` are the parts Phase 3 imports unchanged. Keep them algorithm-agnostic.

---

## 3. Key interface contracts

```python
# agents/base.py
class Agent(Protocol):
    def act(self, obs: np.ndarray, deterministic: bool) -> int: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...

# eval/evaluator.py
@dataclass(frozen=True)
class EvalReport:
    per_step_returns: np.ndarray      # full distribution, NOT just the mean
    episode_returns: np.ndarray
    action_frequencies: dict[int, float]
    mean_pnl: float
    pnl_ci: tuple[float, float]       # bootstrap or t-CI
    sharpe: float
    sortino: float
    cvar_95: float
    max_drawdown: float
    turnover: float

class Evaluator:
    def __init__(self, env_factory: EnvFactory, eval_seeds: Sequence[int], metrics: MetricSuite): ...
    def run(self, agent: Agent, deterministic: bool) -> EvalReport: ...

# eval/no_edge_gate.py
@dataclass(frozen=True)
class NoEdgeResult:
    passed: bool
    flat_frequency: float
    mean_pnl_ci: tuple[float, float]
    reason: str
def evaluate_no_edge(report: EvalReport, *, with_costs: bool, flat_threshold: float) -> NoEdgeResult: ...
```

The same `Evaluator` and `MetricSuite` instances are reused for Phase 3, so the comparison is apples-to-apples by construction.

---

## 4. PPO specification

- **Objective:** clipped surrogate `L_clip = E[min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t)]`, plus value loss `c_vf * MSE(V, returns)` (value clipping optional), minus entropy bonus `c_ent * H[pi]`.
- **Advantages:** GAE(lambda); normalize advantages per minibatch.
- **Policy:** `Categorical` over the discrete action library; the FLAT action is always index 0 and always legal.
- **Network:** small MLP actor-critic. Suggested: shared or separate trunks of 2 hidden layers, width 64–256, `tanh` activations, **orthogonal init** with policy-head gain ~0.01 and value-head gain ~1.0 (the standard PPO init). Keep it small; document the parameter count.
- **Optimizer:** Adam, learning rate ~2.5e-4 to 3e-4 with **linear annealing** to 0; `max_grad_norm = 0.5`.
- **Update:** multiple epochs over shuffled minibatches; optional **target-KL early stop** (~0.015–0.03) to prevent destructive updates.
- **Normalization:** observation normalization **on** (causal). Reward normalization **off by default** (it can mask the deliberate composite-reward weighting; enable only if value loss is unstable, and log that you did).

**Starting hyperparameters (PPOConfig defaults, tune from here):**

| Param | Start | Notes |
|---|---|---|
| `num_envs` | 16 | decorrelate rollouts |
| `num_steps` | 256 | per-env rollout length |
| `total_timesteps` | 2_000_000 | Wave 0 is easy; raise if entropy hasn't settled |
| `gamma` | 0.99 | set consistently with episode length |
| `gae_lambda` | 0.95 | |
| `update_epochs` | 8 | |
| `num_minibatches` | 8 | |
| `clip_coef` | 0.2 | |
| `ent_coef` | 0.01 | **the anti-collapse lever**; tune first if policy goes deterministic |
| `vf_coef` | 0.5 | |
| `max_grad_norm` | 0.5 | |
| `learning_rate` | 2.5e-4 | linear anneal |
| `target_kl` | 0.02 | optional early stop |
| `norm_adv` | true | |
| `norm_obs` | true | causal |
| `norm_reward` | false | flag, default off |

**Episode definition for Wave 0:** a fixed window of daily decisions (e.g. one expiry cycle ~21 trading days, or a quarter ~63). Pick one, keep `gamma` consistent with it, document it. Randomize the **path seed** every episode (fresh GBM realization) while keeping **IV fair** (option IV equals the generating sigma) so there is genuinely no edge. You may draw sigma from a narrow prior per episode to begin building the domain-randomization habit, but the fairness invariant (IV == realized sigma) must hold every episode.

---

## 5. The Wave-0 no-edge gate (the definition of done)

This is the deliverable. Under the **full risk-adjusted composite reward**, the sharp, testable prediction is stronger than "zero mean PnL":

> With E[return] approximately 0 and Var > 0, any active structure adds risk for zero expected reward, so a risk-adjusted objective should make **FLAT strictly preferred**. The CVaR penalty makes this strict even before costs.

So the gate has two parts:

**(a) Risk-adjusted reward, the main gate.** After training on Wave 0:
- FLAT is the **dominant action** in evaluation (FLAT frequency above a high threshold, e.g. 0.80–0.90).
- Per-action frequencies show **no systematic preference for credit structures** (a preference for short-vol structures on fair-IV Wave 0 is the signature of phantom edge = a Phase 1 bug).
- Mean eval PnL confidence interval includes ~0 with no costs, and is negative with costs.

**(b) Pure-PnL ablation, the cross-check.** Re-train with a pure-PnL reward (all risk terms weighted 0), no costs:
- The agent should be approximately **indifferent** (roughly uniform action use, no structure earning systematically positive PnL). This confirms the *environment* has no phantom edge independent of how the reward is shaped. If some structure prints positive PnL here, the bug is in Phase 1 (pricing inconsistency, cost sign, look-ahead, or premium accounting), not in the agent.

**Stability sub-checks (must also hold):** entropy does not collapse to ~0 in the first few updates; value loss decreases and explained variance becomes positive; approx KL stays bounded; results reproduce across at least 3–5 seeds with **reported dispersion** (never a single hero run).

**If the gate fails because the agent finds edge: STOP.** Do not add features, do not proceed to Wave 1. Return to Phase 1 and debug, because the same leak would manufacture fake "skill" in every later wave.

---

## 6. Pitfalls to engineer against

- **Premature policy / entropy collapse.** Monitor entropy every update; if it crashes, raise `ent_coef` or lower LR. A policy that goes deterministic in the first few updates has not learned, it has collapsed.
- **Value function not learning.** Watch explained variance and value loss. Fix via advantage normalization, value-loss clipping, and consistent return scaling.
- **Reward-scale domination.** Log **each composite-reward component's magnitude** separately. If raw PnL is 100x the differential-Sharpe term, the risk shaping is inert. Re-weight in the env (Phase 1 config), do not paper over it with reward normalization.
- **Hidden-state non-Markovianity.** If the risk-reward state (Sharpe EMAs, drawdown, margin) is not in the observation, the policy cannot optimize the risk-adjusted reward and the no-edge result will be muddy. Fix the observation, not the reward.
- **Eval contamination / look-ahead in normalization.** Eval seeds disjoint from training; obs-normalization stats causal.
- **Seed fragility.** Report dispersion across seeds; a result that only holds for one seed is not a result. (This doubles as an overfitting control for the thesis.)
- **Tuning on the gate.** Do not tune hyperparameters against the exact eval seeds used to report the gate. Use separate tuning seeds.

---

## 7. Build order (do not reorder)

1. `training/seeding.py`, `training/env_factory.py`, `training/vec.py`, `training/normalize.py` (causal obs normalizer), `training/logging.py`.
2. `eval/metrics.py` (MetricSuite) and `eval/evaluator.py` + `test_evaluator.py`; then `eval/no_edge_gate.py` + `test_no_edge_gate.py` (test the gate logic on synthetic streams: it must pass a zero-edge stream and fail a planted-edge stream).
3. `agents/base.py` (Agent protocol) and `test_harness_smoke.py` (a RandomAgent runs end-to-end through the shared harness).
4. `agents/ppo/network.py` + `test_ppo_network.py`.
5. `agents/ppo/buffer.py` (GAE) + `test_gae.py` (hand-computed example).
6. `agents/ppo/trainer.py` + `agents/ppo/ppo_agent.py` + `agents/ppo/config.py`; `test_ppo_update.py` (overfit a trivial 1-state bandit to confirm the update is correct).
7. `training/harness.py` and `cli/train.py`, `cli/evaluate.py`.
8. Train PPO on Wave 0; tune to stability (entropy, KL, explained variance).
9. Run the **no-edge gate** (parts a and b), multi-seed, with dispersion.
10. Write the diagnostics summary and confirm the shared harness is reuse-ready for Phase 3.

**Update `CLAUDE.md`** with: the shared-harness contract (Phase 3 reuses `EnvFactory`/`Evaluator`/`MetricSuite` unchanged), the hold-reward-fixed-vary-algorithm rule, the Markovian-reward-state requirement, and the no-edge gate as a hard stop.

---

## 8. Process instructions for Claude Code

- Pin PyTorch; seed python, numpy, and torch through `training/seeding.py`. On CPU, runs should be fully reproducible; if using CUDA, set deterministic flags and note residual nondeterminism.
- Use CleanRL's `ppo.py` as the line-by-line correctness reference and cite it in a comment; restructure into the module layout rather than dropping in a monolith.
- Log to TensorBoard/W&B from the first training run: entropy, approx KL, clip fraction, policy/value loss, explained variance, learning rate, **per-action frequency**, and the **full eval return distribution** (histogram), plus equity curves.
- Run `make check` after each module; keep mypy strict and ruff green.
- Commit at module granularity, referencing the acceptance test that now passes.
- **Do not implement QR-DQN, IQN, distributional critics, or any replay-buffer agent.** That is Phase 3. If a task drifts that way, stop and confirm.
- Do not touch Phase 1 numerics except to (a) extend the observation schema for reward-state observability if needed, or (b) fix a bug the no-edge gate exposes. Any Phase 1 change must keep all Phase 1 tests green.
- End with a short report: PPO stability diagnostics, the no-edge gate result (parts a and b) with per-seed dispersion, and an explicit statement of whether the gate passed and the project may proceed to Phase 3 / Wave 1.

---

### Ready-to-paste kickoff prompt

> Implement Phase 2 of the project as described in `PHASE2_BRIEF.md`, building on the completed Phase 1 repo. Start with the shared `training/` and `eval/` harness (section 2 and the build order in section 7), because Phase 3 will reuse it unchanged. Base PPO on CleanRL's `ppo.py`, restructured into the module layout, using the full Phase 1 composite reward unchanged. First confirm the observation exposes the risk-reward state (Sharpe EMAs, drawdown, margin) so the risk-adjusted reward is Markovian; extend the observation if not. Train on Wave 0 only. The phase is done when the no-edge gate in section 5 passes: under the risk-adjusted reward the agent is FLAT-dominant with no systematic credit-structure preference, the pure-PnL ablation shows indifference, and results hold across at least 3 seeds with reported dispersion. If the agent finds edge on Wave 0, STOP and report it as a Phase 1 bug. Do not implement any distributional or replay-buffer agent. Keep mypy strict and ruff green via `make check`.
