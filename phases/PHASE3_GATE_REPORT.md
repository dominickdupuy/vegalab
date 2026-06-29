# Phase 3 Gate Report

Generated: 2026-06-28T21:32:00

Overall implementation status: **PASS**

## G1 — Quantile Machinery

Status: **PASS**

- `quantile_huber_loss` is tested against hand-computed asymmetric cases.
- `RiskMeasure.mean` and `RiskMeasure.cvar` are tested from quantiles and raw samples.

## G2 — Distribution Recovery

Status: **PASS**

- QR-DQN recovers a fixed static quantile grid in a compact supervised recovery test.
- IQN recovers a linear quantile function using cosine tau embeddings.

## G3 — Fat-Tail Bandit

Status: **PASS**

- Mean-greedy selection chooses the high-mean catastrophic-tail arm.
- CVaR-greedy selection chooses the bounded-downside arm.
- The test exercises both analytic risk values and the QR-DQN agent action-selection path.

## G4 — Fat-Tail MDP

Status: **PASS**

- Mean-greedy selection chooses the delayed-tail action.
- CVaR-greedy selection avoids it.

## G5 — Wave-0 No-Edge

Status: **PASS**

Compact CVaR Wave-0 multi-seed runs through the reused Phase-2 evaluator:

| Agent | Seed | Timesteps | Alpha | Eval Episodes | Gate | FLAT | Mean P&L |
|---|---:|---:|---:|---:|---|---:|---:|
| QR-DQN | 502 | 6,000 | 0.10 | 30 | PASS | 1.000 | +0.00 |
| QR-DQN | 503 | 6,000 | 0.10 | 30 | PASS | 1.000 | +0.00 |
| QR-DQN | 504 | 6,000 | 0.10 | 30 | PASS | 1.000 | +0.00 |
| IQN | 601 | 10,000 | 0.10 | 30 | PASS | 1.000 | +0.00 |
| IQN | 602 | 6,000 | 0.10 | 30 | PASS | 1.000 | +0.00 |
| IQN | 603 | 6,000 | 0.10 | 30 | PASS | 1.000 | +0.00 |

Commands:

```bash
python -m optspread.cli.train_distributional --algo qrdqn --risk cvar --cvar-alpha 0.1 --n-quantiles 100 --total-timesteps 6000 --learning-starts 500 --eval-episodes 30 --seed 502 --no-tensorboard
python -m optspread.cli.train_distributional --algo iqn --risk cvar --cvar-alpha 0.1 --total-timesteps 10000 --learning-starts 1000 --eval-episodes 30 --seed 601 --no-tensorboard
```

## G6 — Fair Comparison Harness

Status: **PASS**

`compare.py` evaluates Flat, QR-DQN, and IQN through the same
`EnvFactory`/`Evaluator`/`MetricSuite` stack:

| Agent | Mean P&L | Sharpe | Sortino | CVaR95 | MaxDD | Turnover | FLAT |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat | +0.00 | +0.000 | +0.000 | +0.00 | 0.000 | 0.00 | 1.000 |
| qrdqn | +0.00 | +0.000 | +0.000 | +0.00 | 0.000 | 0.00 | 1.000 |
| iqn | +0.00 | +0.000 | +0.000 | +0.00 | 0.000 | 0.00 | 1.000 |

Command:

```bash
python -m optspread.cli.compare --include-flat --qrdqn runs\phase3_distributional_wave0_qrdqn_cvar\seed_502\agent.pt --iqn runs\phase3_distributional_wave0_iqn_cvar\seed_602\agent.pt --eval-episodes 30
```

## Validation

Repository quality gates are green:

```bash
python -m ruff check optspread tests
python -m ruff format --check optspread tests
python -m mypy --strict optspread
python -m pytest tests/ --cov=optspread --cov-report=term-missing
```

Result: **140 passed**, strict mypy green, ruff green.

## Conclusion

Phase 3 is complete at the Wave-0 / in-vitro level specified by the brief:
QR-DQN and IQN machinery are implemented, fat-tail mechanism tests pass, both
agents pass Wave-0 no-edge checks, and `compare.py` produces the expected null
head-to-head through the shared evaluator. The project may proceed to Phase 4.
