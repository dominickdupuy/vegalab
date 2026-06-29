# Phase 2 Gate Report

Generated: 2026-06-28T21:05:58

Overall result: **PASS**

## Risk-Adjusted PPO Gate

Reward: scaled MTM P&L + soft CVaR penalty. Costs: default quoted spread. Pass condition: FLAT frequency >= 0.80 and mean-PnL CI lower bound <= 0.

| Seed | Gate | FLAT | Mean P&L | 95% CI | Entropy First→Last | KL | Expl.Var | Top Actions |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 13 | PASS | 0.995 | -263.08 | [-653.89, +38.40] | 2.933→0.075 | 0.0206 | 0.622 | 0:0.995, 4:0.003, 3:0.001, 1:0.001, 14:0.000 |
| 32 | PASS | 0.996 | -56.50 | [-212.44, +101.14] | 2.931→0.033 | 0.0001 | -0.673 | 0:0.996, 3:0.004, 1:0.000, 2:0.000, 4:0.000 |
| 43 | PASS | 0.912 | -1351.94 | [-2010.97, -768.30] | 2.930→0.065 | 0.0044 | -0.016 | 0:0.912, 7:0.070, 4:0.014, 1:0.003, 2:0.002 |

## Pure-PnL No-Cost Ablation

Reward: scaled pure MTM P&L. Costs: zero. Pass condition: no statistically reliable positive mean P&L.

| Seed | Gate | FLAT | Mean P&L | 95% CI | Entropy First→Last | KL | Expl.Var | Top Actions |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 113 | PASS | 0.005 | -1413.38 | [-3844.15, +1165.47] | 2.926→2.276 | 0.0224 | -0.102 | 7:0.213, 2:0.190, 13:0.123, 4:0.072, 16:0.070 |
| 132 | PASS | 0.001 | +1257.03 | [-1360.20, +3876.92] | 2.929→2.359 | 0.0265 | 0.131 | 16:0.281, 8:0.204, 12:0.151, 10:0.073, 17:0.072 |
| 143 | PASS | 0.030 | +1246.82 | [-784.34, +3366.78] | 2.925→2.485 | 0.0256 | 0.077 | 16:0.193, 8:0.150, 10:0.125, 15:0.091, 14:0.065 |

## Conclusion

Phase 2 is complete: PPO does not find a systematic Wave-0 edge, and the project may proceed to Phase 3.
