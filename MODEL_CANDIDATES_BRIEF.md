# Model candidates for Wave 2 (BV_2 + FF_2)

Where we stand: our primary agent (IQN, trained risk-neutral, deployed by maximizing
CVaR of its predicted return quantiles) collapses to ~99% FLAT on Wave 2 — its learned value
gap between trading and not-trading sits at ~zero, so the CVaR filter has nothing to work
with. PPO learns the signal (corr up to +0.50) but over-trades the no-edge control. The
research (two independent literature sweeps, claims adversarially verified — full citations
in `MODEL_CANDIDATES_RESEARCH.md`) says this isn't a tuning problem: pure-CVaR action
selection provably can't see upside, and our success-replay hack is a biased version of a
published method. These are the candidates, ordered by evidence and cost.

| # | Candidate | Type of change | Effort |
|---|-----------|----------------|--------|
| 1 | QR-SRM (spectral risk) | action-selection + risk objective | small |
| 2 | Risk-annealed IQN | training schedule | trivial |
| 3 | CeSoR-corrected sampling | replay buffer | small |
| 4 | Clipped Advantage Learning | TD target | small |
| 5 | ORAC-style optimistic exploration | behavior policy | medium |
| 6 | UPER | replay prioritization | medium |
| 7 | BBF levers (n-step + SPR) | training recipe | small |
| 8 | IQN-CVaR-PPO hybrid | different agent class | large |

**First, two free diagnostics on existing checkpoints (no training):** (a) measure the
quantile spread of the trained IQN — there's published evidence the Huber quantile loss we
use can collapse the learned distribution to its mean, and if ours has collapsed, CVaR
selection is doing nothing and #1–#6 are moot until the loss is changed; (b) log
Q(trade)−Q(FLAT) in high-signal vs no-signal states — it's the quantity every candidate
below is trying to move, and it becomes our pre-registration statistic.

**1. QR-SRM — spectral risk measures on our existing QR-DQN/IQN.**
What it is: instead of ranking actions by CVaR alone (worst 10–20% of outcomes), rank them
by a weighted combination of CVaRs at several levels including the mean (e.g. 0.5·mean +
0.5·CVaR_0.2). Why it's different: pure CVaR gives literally zero weight to everything above
the tail, so a trade with good typical P&L and a manageable tail scores no better than one
with no upside at all — this is the exact published pathology ("blindness to success")
behind our flat-collapse, and spectral measures are the literature's direct fix for it.
Why it fits us: it stays value-based and tail-averse (thesis intact), and it's a ~small
change to the action-selection and target code we already have. Honest caveat: the QR-SRM
paper's empirical wins are its own single benchmark; the mechanism is what's verified.

**2. Risk-annealed IQN — schedule alpha instead of fixing it.**
What it is: train with risk-neutral (or high-alpha) action scoring and anneal toward the
deployment alpha over training. Why it's different: today the agent is asked to be
tail-averse before it has learned that trading has any value, so it never escapes FLAT;
annealing lets it learn values first and become conservative second. Evidence: in the one
published head-to-head, a statically risk-averse distributional agent won 0.2% of episodes
where the annealed-to-identical-endpoint agent won 61% (StarCraft, multi-agent — treat as
directional, not a magnitude promise). Why it fits us: it's a schedule on `--cvar-alpha`,
essentially free, and composes with #1.

**3. CeSoR-corrected sampling — fix our replay boost instead of tuning it.**
What it is: the published method our `reward_priority_boost` imitates (CeSoR, NeurIPS 2022)
oversamples hard market *episodes* — not profitable *transitions* — and applies an
importance-sampling correction so value estimates stay unbiased. Why it's different: our
uncorrected winner-boosting feeds the agent an optimistic world model, which is exactly why
boost 6.0 made it trade indiscriminately on the no-edge control. Why it fits us: we already
proved the lever moves trade frequency; this is the version of the same lever that
shouldn't break FF_2. Small change: episode/regime-level sampling weights + IS correction
in the existing buffer.

**4. Clipped Advantage Learning — amplify the decision gap in the TD target.**
What it is: a one-line modification to the Bellman target (Bellemare et al.'s gap-increasing
operator, in its "clipped" form, AAAI 2022) that provably widens the value difference
between the best and second-best action while preserving which action is optimal. Why it's
different: it attacks our measured symptom directly — the trade-vs-FLAT gap being smaller
than network noise — rather than the risk machinery. Why it fits us: the true gap gets
amplified in *both* environments: toward trading where edge exists, toward FLAT where it
doesn't. It's the only candidate that should help both gates simultaneously, which is a
sharp, pre-registerable prediction. Use the clipped variant: the plain operator is known to
misfire when the current greedy action is wrong.

**5. ORAC-style optimistic exploration — replace epsilon-greedy.**
What it is: explore by acting on an upper-confidence estimate of value (from a small
ensemble or bootstrapped heads) while keeping the risk-averse rule for exploitation. Why
it's different: epsilon-greedy random actions almost never produce convincing evidence for
a +0.03-correlation edge; directed optimism seeks out exactly the states where the agent
might be wrong about "trading is worthless." Evidence: on a rare-hazard benchmark the
standard risk-averse baseline got stuck in passive policies 40–70% of the time; the
optimistic-explorer version, almost never. Why it fits us: it targets "never learned the
trade has value" — our confirmed diagnosis — without touching the reward or the deployment
risk rule. Cost: needs the ensemble, so medium.

**6. UPER — prioritize replay by information gain, not TD error.**
What it is: replay transitions ranked by epistemic uncertainty (what the agent could learn)
divided by aleatoric noise (what's just randomness), computed from a QR-DQN ensemble. Why
it's different: options P&L is noise-dominated, and both TD-error PER and our success-boost
provably chase that noise. Evidence: on full Atari-57, UPER beat PER and — the clean part —
beat an *identical-architecture* ensemble that differed only in prioritization. Why it fits
us: it's the principled replacement for both replay hacks at once, and it shares the
ensemble built for #5.

**7. BBF levers — two cheap recipe upgrades.**
(a) n-step annealing: start TD targets at n=10 and shorten to n=3 during training, so credit
reaches the entry decision quickly at first, precision arrives later; ablations show it
beats any fixed n. (b) SPR auxiliary loss: make the network predict its own future latent
states, giving it a learning signal about market structure even while the P&L signal is
too faint to teach anything. Both verified from the strongest sample-efficiency line
(BBF reaches superhuman Atari at our step budget) and stack with everything above.

**8. Fallback: IQN-CVaR-PPO — put the distributional machinery on the agent that works.**
What it is: keep PPO as the learner (it already passes BV_2), replace its critic with our
IQN and weight its advantages by CVaR; reward unchanged. Two 2024/25 papers do this, one on
a trading task with alpha-annealing built in. Implementation warning from that literature:
naively swapping in the IQN critic fails — it must be fused through a small distillation
layer. The real cost isn't code: it changes the headline agent from off-policy value-based
to actor-critic, so it needs a thesis-framing decision before anyone builds it.

**Recommended sequence:** diagnostics → #1+#2 (one combined run) → #3 → #4 → #5+#6 (shared
ensemble) → #8 only if the value-based line is exhausted. Three seeds per config,
prediction committed before training, and BV_2/FF_2 always reported together — passing one
by breaking the other is a fail.
