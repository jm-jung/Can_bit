# Phase D9: Cost Sensitivity Summary

## Strategy parameters (fixed)
- symbol=BTCUSDT, timeframe=5m, end_date=2026-03-03
- min_max_proba=0.575, max_entropy=1.3
- min_hold=36, cooldown=12
- time_stop=72, early_exit_bad_k=8, early_exit=on
- regime_filter=off, position_scaling=off
- days_list=[180, 365, 720]

## Cost sensitivity table

| cost_level | 180d cost_on | 365d cost_on | 720d cost_on | 720d MDD | 720d trades | win_rate |
|------------|--------------|--------------|--------------|----------|-------------|----------|
| 0.0000 | 0.0857 | 0.0771 | 0.2005 | 0.0309 | 1491 | 50.50% |
| 0.0002 | 0.0781 | 0.0620 | 0.1649 | 0.0344 | 1491 | 49.64% |
| 0.0004 | 0.0706 | 0.0472 | 0.1303 | 0.0402 | 1491 | 48.49% |
| 0.0006 | 0.0632 | 0.0325 | 0.0967 | 0.0460 | 1491 | 47.92% |
| 0.0008 | 0.0558 | 0.0181 | 0.0642 | 0.0539 | 1491 | 47.06% |
| 0.0010 | 0.0485 | 0.0038 | 0.0326 | 0.0618 | 1491 | 45.91% |

## Additional analysis
- **cost_break_even:** 0.001 (max cost where 720d cost_on >= 0)
- **cost_slope:** -167.9187303537093 (performance change per 0.001 cost increase)

## Verdict
- **EDGE_PRESENT_COST_SENSITIVE**

Export:
- data/backtests/phase_d9_cost0_trades.csv
- data/backtests/phase_d9_cost1_trades.csv
