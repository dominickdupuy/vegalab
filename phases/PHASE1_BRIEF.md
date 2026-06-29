# Phase 1 Implementation Brief for Claude Code
## SPX Options Spread-Selection RL — Environment & Cost Model

> **Scope of this phase.** Build the options-spread trading environment and its supporting numerics as a *standalone, fully tested artifact*. No RL agent, no PyTorch, no training. The deliverable is a deterministic, Gymnasium-conformant environment whose P&L, margin, cost, and reward machinery is verified against closed-form oracles, drivable by a minimal GBM generator, and exercised by two scripted baseline agents. Everything here must be swappable later (the generator becomes Heston/Bates/regime-switching; the reward gets ablated; costs get recalibrated to real OptionMetrics quotes), so design for substitution from line one.

---

## 0. Stack decisions (fixed)

- **Language:** Python 3.11+. The entire downstream stack (Gymnasium, PPO, QR-DQN/IQN, PyTorch) is Python-native and EOD daily steps are not latency-critical.
- **Numerics:** NumPy + SciPy. Vectorize per-strike/per-leg math. Do **not** add Numba/Cython unless a profiler shows a real hotspot.
- **Env API:** Gymnasium (not legacy Gym). Must pass `gymnasium.utils.env_checker.check_env`.
- **Config:** Pydantic v2 models for all configs (env, generator, costs, margin, reward). No Hydra at this stage.
- **Testing:** pytest + Hypothesis (property-based) + pytest-cov.
- **Quality gates:** mypy `--strict`, ruff (lint + format). CI-style `make check` target.
- **Packaging:** `uv` (preferred) or Poetry, pinned `pyproject.toml`.
- **No PyTorch, no SB3/CleanRL in this phase.** Keep dependencies minimal.

---

## 1. Design principles (apply throughout — this is the "coding design skills" answer)

1. **Dependency injection / inversion.** `SpreadEnv` receives a `PriceGenerator`, `CostModel`, `MarginModel`, `RewardFunction`, and `ObservationBuilder` through its constructor. It constructs none of them internally. This is the single most important rule: every one of these gets replaced in a later phase.
2. **Protocol/ABC interfaces.** Define structural interfaces (`typing.Protocol` or `abc.ABC`) for each injected dependency. Concrete classes implement them. Tests target the interface.
3. **Strategy + Composite patterns for reward.** Each reward term is a `RewardComponent` strategy with a weight; `CompositeReward` is their weighted sum. Setting a weight to 0 disables a term (this is how Phase 8 ablations run, so build it now).
4. **Strategy pattern for spread templates.** Each of the 12–20 structures is a `SpreadTemplate` that knows how to build its legs from a chain snapshot and a delta bucket, and declares its margin class.
5. **Value objects as frozen dataclasses.** `OptionLeg`, `MarketSnapshot`, `Position`, `StepResult`. Immutable where possible; no hidden mutable state outside the env and the stateful reward EMAs.
6. **Pure functions for all numerics.** Black-Scholes pricing/greeks, payoff functions, and strike solving are side-effect-free and deterministic. This is what makes them oracle-testable.
7. **Explicit RNG threading.** One `numpy.random.Generator`, seeded from config, passed explicitly. **Never** call global `np.random.*`. Same seed produces an identical trajectory, enforced by a test.
8. **No look-ahead by construction.** The generator advances state *only* inside `env.step()`. The observation at time *t* is computable from information available at *t*'s close and nothing later. Structurally guarantee this, then assert it.
9. **Strict typing + linting.** mypy strict and ruff clean are merge gates, not afterthoughts.
10. **Observability.** Log the **full per-step return array, action histogram, equity curve, cost drag, and margin usage**, not just scalar reward. You will need distributions, not means, in every later phase.
11. **Numerical robustness.** Handle `T -> 0`, deep ITM/OTM, zero-variance reward streams, and division-by-zero in Sharpe denominators explicitly and test the edges.
12. **Entropy is agent-side, not env-side.** Do **not** put an entropy bonus in the reward. It belongs in the PPO objective in Phase 2. Note this in code comments so it isn't "helpfully" added later.

---

## 2. Repository layout (create exactly this)

```
optspread/
  __init__.py
  config.py                 # Pydantic configs: EnvConfig, GBMConfig, CostConfig, MarginConfig, RewardConfig
  rng.py                    # make_rng(seed) -> np.random.Generator; helpers
  pricing/
    black_scholes.py        # bs_price, bs_delta, bs_gamma, bs_vega, bs_theta (vectorized, pure)
    strike_solver.py        # strike_from_delta via Brent/Newton (scipy.optimize.brentq)
  instruments/
    leg.py                  # OptionLeg (frozen): right, strike, expiry_idx, qty (+/-), entry_price
    chain.py                # ChainSnapshot: strikes, ivs, expiries, spot, r, q at one date
  market/
    generator.py            # PriceGenerator Protocol: reset()->MarketSnapshot, step()->MarketSnapshot
    gbm.py                  # GBMGenerator: constant-vol GBM, FAIR IV (option IV == realized sigma)
    snapshot.py             # MarketSnapshot (frozen): chain + regime feature dict + t
  actions/
    templates.py            # SpreadTemplate ABC + concrete templates (see section 4)
    library.py              # ACTION_LIBRARY: ordered list of (template, delta_bucket); DELTA_BUCKETS={0.10,0.16,0.25,0.40}
    margin_class.py         # MarginClass enum: LONG_ONLY, DEFINED_RISK, UNDEFINED_RISK, FLAT
  portfolio/
    position.py             # Position/Portfolio: holdings, cash, MTM value
    pnl.py                  # mark-to-market, realized/unrealized P&L, credit-up-front accounting
  costs/
    cost_model.py           # CostModel Protocol
    spread_cost.py          # QuotedSpreadCost: half-spread per leg, scales with width/moneyness/#legs
  margin/
    margin_model.py         # MarginModel Protocol
    reg_t.py                # RegTStyleMargin: defined vs undefined risk sizing
  reward/
    components.py           # RewardComponent ABC + MTMPnL, DifferentialSharpe, Sortino, CVaRPenalty, MarginNormalizer
    composite.py            # CompositeReward: weighted sum, per-component logging
  envs/
    spread_env.py           # SpreadEnv(gymnasium.Env): ties everything via injected deps
    observation.py          # ObservationBuilder: (snapshot, portfolio) -> np.ndarray, fixed schema
  agents/
    baselines.py            # RandomAgent, AlwaysOnAgent(action_id), FlatAgent
  cli/
    smoke_run.py            # run baselines N episodes on GBM, dump diagnostics + plots
tests/
  test_black_scholes.py
  test_strike_solver.py
  test_templates_payoff.py  # parametrized over the whole library
  test_pnl.py
  test_margin.py
  test_costs.py
  test_reward.py
  test_env_api.py
  test_determinism.py
  test_invariants.py        # Hypothesis property tests
pyproject.toml
Makefile                    # check = ruff + mypy + pytest --cov
CLAUDE.md                   # architecture + invariants (see section 8)
README.md
```

---

## 3. Key interface contracts (define these first, before implementations)

```python
# market/generator.py
class PriceGenerator(Protocol):
    def reset(self, rng: np.random.Generator) -> MarketSnapshot: ...
    def step(self) -> MarketSnapshot: ...        # advances exactly one trading day
    @property
    def done(self) -> bool: ...

# costs/cost_model.py
class CostModel(Protocol):
    def cost(self, legs: Sequence[OptionLeg], chain: ChainSnapshot) -> float: ...
    # returns a non-negative dollar cost for opening OR closing the given legs

# margin/margin_model.py
class MarginModel(Protocol):
    def margin(self, position: Position, chain: ChainSnapshot) -> float: ...
    # non-negative buying-power requirement for holding `position`

# reward/components.py
class RewardComponent(ABC):
    weight: float
    @abstractmethod
    def update(self, ctx: StepContext) -> float: ...   # may hold internal EMA state
    def reset(self) -> None: ...

# actions/templates.py
class SpreadTemplate(ABC):
    name: str
    @abstractmethod
    def build(self, chain: ChainSnapshot, delta_bucket: float) -> list[OptionLeg]: ...
    @abstractmethod
    def margin_class(self) -> MarginClass: ...
    @abstractmethod
    def analytic_payoff(self, legs: Sequence[OptionLeg], terminal_spot: float) -> float: ...
    # analytic_payoff is the ORACLE used by tests; implement it from first principles
```

`StepContext` carries everything a reward term needs: `step_pnl`, `margin_used`, `recent_returns` (a deque/array), `is_terminal`. Keep it a frozen dataclass.

---

## 4. Action library (the locked 12–20 templates)

Each entry is `(SpreadTemplate, delta_bucket)`; `DELTA_BUCKETS = {0.10, 0.16, 0.25, 0.40}`. Include at minimum, spanning {directional vs neutral} x {credit vs debit} x {defined vs undefined risk} x {single vs multi-expiry}:

| Template | Class | Notes |
|---|---|---|
| Flat / no position | FLAT | the always-available null action; zero legs, zero margin |
| Long call | LONG_ONLY | directional debit, defined risk = premium |
| Long put | LONG_ONLY | directional debit |
| Bull put spread (vertical credit) | DEFINED_RISK | short put at bucket delta, long put one wing below |
| Bear call spread (vertical credit) | DEFINED_RISK | mirror |
| Bull call spread (vertical debit) | net debit, defined | |
| Bear put spread (vertical debit) | net debit, defined | |
| Iron condor | DEFINED_RISK | short strangle at bucket delta + long wings |
| Iron butterfly | DEFINED_RISK | short ATM straddle + wings |
| Short strangle | UNDEFINED_RISK | naked, must carry real margin |
| Short straddle | UNDEFINED_RISK | naked |
| Calendar spread | DEFINED_RISK | **two expiries** — short near, long far, same strike |
| Ratio / broken-wing spread | UNDEFINED_RISK or DEFINED depending on construction | document which |

Short-leg strike is selected by the **delta bucket** via `strike_from_delta`. Wing width is a config parameter (fixed strikes-away or fixed delta wing). Multi-expiry templates (calendar) require the chain to expose at least two expiries; the generator must supply them.

Total actions = (number of delta-bucketed templates x applicable buckets) + the flat action; keep the final count in the 12–20 range. Encode the library as an **ordered list** so `action_id` maps to `(template, bucket)` stably and reproducibly.

---

## 5. The minimal Wave-0 generator (just enough to drive and test the env)

`GBMGenerator`: geometric Brownian motion, **zero drift**, **constant vol sigma**, and crucially **fair IV** — every option in the chain is priced via Black-Scholes at the *same* sigma that generates the underlying path. With zero drift, fair vol, and no costs, every structure has approximately zero expectancy; with costs, every structure has negative expectancy and FLAT dominates. This is the economic sanity baseline the env must reproduce. Regime features in this generator are mostly degenerate (VRP near 0, IV rank undefined/constant); compute what you can (e.g. trailing momentum) and emit the rest as constants. The observation **schema** must already match what real data will provide later, so the env never changes when richer generators arrive.

---

## 6. Reward specification (composable, weighted, ablatable)

`CompositeReward = sum_i weight_i * component_i.update(ctx)`. Components:

- **`MTMPnL`** — dense per-step mark-to-market net P&L (net of costs). The base signal.
- **`MarginNormalizer`** — transforms P&L into return-on-capital: `step_pnl / max(margin_used, floor)`. Implement as a wrapper/transform so it composes with the P&L term.
- **`DifferentialSharpe`** (Moody & Saffell). Maintain EMAs with rate eta:
  - `dA = R_t - A_{t-1}`,  `A_t = A_{t-1} + eta*dA`
  - `dB = R_t^2 - B_{t-1}`,  `B_t = B_{t-1} + eta*dB`
  - `D_t = (B_{t-1}*dA - 0.5*A_{t-1}*dB) / (B_{t-1} - A_{t-1}^2)^{3/2}`
  - Guard the denominator: if `B_{t-1} - A_{t-1}^2 <= epsilon`, return 0.0. The per-step reward contribution is `D_t`.
- **`Sortino`** — like Sharpe but with downside-deviation denominator (EMA of `min(R_t,0)^2`). Rolling/EMA form; optional.
- **`CVaRPenalty`** — rolling empirical CVaR at level alpha of recent step returns; penalize (<= 0) only when the tail breaches a threshold. **Default weight low**, with a comment that the *native* tail handling arrives via the distributional critic in Phase 3; this term exists for the PPO baseline and for ablation.

Each component's contribution is logged separately every step. **No entropy term here.**

---

## 7. Acceptance-test matrix (this IS the definition of done)

Implement TDD: for the correctness-critical modules (pricing, strike solver, template payoffs, P&L, margin) write the failing test first, then the implementation. Phase 1 is complete only when **all** of the following pass under `make check`.

**Black-Scholes (`test_black_scholes.py`)**
- Golden value: `S=100, K=100, r=0, q=0, sigma=0.2, T=1` gives call approximately **7.9656** (and put = call at r=0 ATM).
- Put-call parity `C - P = S*e^{-qT} - K*e^{-rT}` over a grid of inputs.
- Greeks match central finite differences within tolerance.
- Call delta in [0,1], put delta in [-1,0], delta monotonic in moneyness.
- `T -> 0` returns intrinsic value without NaN/inf.

**Strike solver (`test_strike_solver.py`)**
- For each bucket {0.10,0.16,0.25,0.40}, `strike_from_delta` returns a strike whose recomputed delta matches the target within tol, for calls and puts.

**Template payoffs (`test_templates_payoff.py`, parametrized over the whole library)**
- For each template: evaluate terminal payoff over a grid of terminal spots and assert it equals `analytic_payoff` (sum of per-leg intrinsic minus net premium).
- Spot-check known bounds: long call max loss = premium and breakeven = K+premium; vertical credit max profit = net credit, max loss = width - credit; iron condor/butterfly max loss = max wing width - credit.

**P&L (`test_pnl.py`)**
- Opening a credit structure increases cash by the net credit **immediately** (premium up front).
- Round-trip with zero price change and zero cost gives zero realized P&L.
- Round-trip with cost c per side gives realized P&L approximately -2c.
- Unrealized MTM equals current leg values minus entry.

**Margin (`test_margin.py`)**
- Defined-risk spread margin = (width - credit) * multiplier.
- Long-only debit margin = premium paid.
- Undefined-risk (naked) margin is strictly positive and materially larger than the defined-risk equivalent.
- FLAT gives zero margin.

**Costs (`test_costs.py`)**
- Cost scales linearly with number of legs.
- Cost scales with quoted spread / width.
- Deeper-OTM (wider quoted spread) legs cost more.

**Reward (`test_reward.py`)**
- Differential-Sharpe EMA estimate converges to the batch Sharpe of a long stationary i.i.d. return stream within tolerance.
- Zero-variance stream produces no NaN (denominator guard works).
- `CVaRPenalty <= 0` and is nonzero only below threshold.
- Margin-normalized P&L equals raw P&L / margin.
- Setting any component weight to 0 removes its contribution exactly (ablation invariant).

**Env (`test_env_api.py`, `test_determinism.py`, `test_invariants.py`)**
- `gymnasium.utils.env_checker.check_env(SpreadEnv(...))` passes.
- Same seed gives byte-identical trajectory (observations, rewards, actions-applied).
- No-look-ahead: generator state advances only inside `step()`; assert the observation at *t* is independent of the *t+1* draw.
- Hypothesis invariants: margin >= 0 always; cash accounting conserves (sum of cash flows = realized P&L); position MTM finite for all reachable states.

**Smoke validation (`cli/smoke_run.py`) — the economic sanity gate**
- On Wave-0 GBM (fair IV, zero drift):
  - **no costs:** an always-on credit-structure agent shows approximately zero mean P&L within Monte-Carlo noise (report the CI).
  - **with costs:** always-on mean P&L < 0; the random agent bleeds costs faster.
- Dump per-episode: equity curve, action histogram, cost drag, margin-usage series, and the **full per-step return array**.

> If the always-on agent "makes money" on Wave-0 with fair IV, that is a **bug** (pricing inconsistency, cost sign error, or look-ahead), not a discovery. Do not proceed until it reads approximately 0 (no costs) / < 0 (with costs).

---

## 8. Build order (do not reorder — each step depends on the prior)

1. `pyproject.toml`, `Makefile`, `rng.py`, `config.py`, empty package skeleton, `CLAUDE.md`.
2. `pricing/black_scholes.py` + tests (golden value, parity, greeks).
3. `pricing/strike_solver.py` + tests.
4. `instruments/leg.py`, `instruments/chain.py`, `market/snapshot.py`.
5. `actions/margin_class.py`, `actions/templates.py` (with `analytic_payoff` oracles) + payoff tests, then `actions/library.py`.
6. `market/generator.py` + `market/gbm.py` (fair-IV GBM).
7. `portfolio/position.py`, `portfolio/pnl.py` + P&L tests.
8. `costs/cost_model.py`, `costs/spread_cost.py` + cost tests.
9. `margin/margin_model.py`, `margin/reg_t.py` + margin tests.
10. `reward/components.py`, `reward/composite.py` + reward tests.
11. `envs/observation.py`, `envs/spread_env.py` + env API/determinism/invariant tests.
12. `agents/baselines.py`, `cli/smoke_run.py` + the smoke validation.

**`CLAUDE.md` must record:** the injected-dependency architecture, the no-look-ahead and determinism invariants, the "entropy is agent-side" note, the action-library ordering contract, and the Wave-0 no-edge expectation, so future Claude Code sessions don't violate them.

---

## 9. Process instructions for Claude Code

- Work module-by-module in the build order above; run `make check` after each module and keep the tree green.
- **TDD on pricing, strike solver, template payoffs, P&L, and margin** (failing test first). Other modules may be test-after but must still hit the acceptance matrix.
- Enforce mypy strict and ruff on every commit; no `# type: ignore` without a one-line justification.
- Commit at module granularity with messages referencing the acceptance test that now passes.
- Keep all randomness behind the injected `Generator`; flag any global RNG use as a bug.
- Do **not** add training code, neural nets, or PyTorch. If a task seems to want an agent, stop and confirm — that is Phase 2.
- At the end, produce a short `README.md` with setup, `make check`, and how to run `smoke_run.py`, plus a one-paragraph statement of which acceptance tests pass and the Wave-0 smoke numbers.

---

### Ready-to-paste kickoff prompt

> Implement Phase 1 of the project described in `PHASE1_BRIEF.md` (this document). Start by scaffolding the repo exactly as specified in section 2, then proceed strictly in the build order of section 8. Use TDD for pricing, strike solving, template payoffs, P&L, and margin. After each module, run `make check` and keep the tree green under mypy strict and ruff. Do not write any RL agent, neural network, or PyTorch code. Stop and ask me before deviating from the interface contracts in section 3 or the action library in section 4. Phase 1 is done when the full acceptance-test matrix in section 7 passes and `smoke_run.py` reproduces the Wave-0 no-edge result.
