# Phase D10: Confidence Spread Summary

## A) Strategy parameters
- symbol=BTCUSDT, timeframe=5m, end_date=2026-03-03
- min_max_proba=0.575, max_entropy=1.3
- min_hold=36, cooldown=12
- time_stop=72, early_exit_bad_k=8, early_exit=on
- regime_filter=off, position_scaling=off

## B) Updated baseline reference
- 180d cost_on ≈ 0.012621292690840447
- 365d cost_on ≈ -0.0733818966155878
- 720d cost_on ≈ -0.111968647467793
- 720d trades = 1491

## C) Full result table

| run_id | min_proba_gap | 180d cost_on | 365d cost_on | 720d cost_on | 720d MDD | 720d trades | filtered_trade_ratio |
|--------|---------------|--------------|--------------|--------------|----------|-------------|----------------------|
| phase_d10_baseline | 0.00 | 0.0126 | -0.0734 | -0.1120 | 0.1523 | 1491 | 0.00% |
| phase_d10_gap003 | 0.03 | 0.0126 | -0.0734 | -0.1120 | 0.1523 | 1491 | 15.39% |
| phase_d10_gap005 | 0.05 | 0.0126 | -0.0734 | -0.1120 | 0.1523 | 1491 | 24.47% |
| phase_d10_gap007 | 0.07 | 0.0126 | -0.0734 | -0.1120 | 0.1523 | 1491 | 33.23% |
| phase_d10_gap010 | 0.10 | 0.0126 | -0.0734 | -0.1120 | 0.1523 | 1491 | 44.90% |

## D) Top3 by 720d cost_on

1. **phase_d10_baseline** min_proba_gap=0.00 720d cost_on=-0.1120
2. **phase_d10_gap003** min_proba_gap=0.03 720d cost_on=-0.1120
3. **phase_d10_gap005** min_proba_gap=0.05 720d cost_on=-0.1120

## E) Verdict summary
- BEST run_id: **phase_d10_baseline**
- BEST min_proba_gap: **0.0**
- Spot stability: **STABLE**
- Final verdict: **NO_IMPROVE**
