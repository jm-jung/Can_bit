# Phase D6 multi-period stability summary

- candidate: min_max_proba=0.575, max_entropy=1.30, end_date=2026-03-03
- PASS rules: cost_on >= -0.08 all, max_drawdown <= 0.15, trades >= 100 all

| period | trades | cost_on | cost_off | max_drawdown | win_rate | mean_hold |
|--------|--------|---------|----------|--------------|----------|-----------|
| 180d | 347 | 0.0125 | 0.0857 | 0.0186 | 0.4380 | 43.3 |
| 365d | 699 | -0.0570 | 0.0862 | 0.1007 | 0.3977 | 43.5 |
| 720d | 1491 | -0.1120 | 0.2005 | 0.1523 | 0.4239 | 44.2 |

## Stability
- cost_on >= -0.08 (all): **FAIL**
- max_drawdown <= 0.15: **FAIL**
- trades >= 100 (all): **OK**
- cost_on sign consistency: **mixed**
- trade density stability: **OK**

## Verdict: **FAIL**
## Recommendation: **further tuning**
