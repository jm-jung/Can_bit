# Phase E1: Event Feature Impact Summary

**Note:** 365d cost_on deviation 13.28% > 5% (expected ≈-0.057). Possible data drift.

## A) Strategy parameters (fixed)
- symbol=BTCUSDT, timeframe=5m, end_date=2026-03-03
- min_max_proba=0.575, max_entropy=1.3
- min_hold=36, cooldown=12
- time_stop=72, early_exit_bad_k=8, early_exit=on
- regime_filter=off, position_scaling=off
- days_list=[180, 365, 720]

## B) Result table

| feature_preset | feature_count | 180d cost_on | 365d cost_on | 720d cost_on | 720d MDD | 720d trades | win_rate |
|----------------|---------------|--------------|--------------|--------------|----------|-------------|----------|
| base | 11 | -0.0313 | -0.0336 | -0.0313 | 0.0520 | 130 | 32.26% |
| extended_safe | 29 | 0.0125 | -0.0646 | -0.1120 | 0.1523 | 1491 | 39.60% |

## Verdict
- **EVENT_FEATURE_HARMFUL**

Export:
- data/backtests/phase_e1_base_trades_365d.csv
- data/backtests/phase_e1_extended_trades_365d.csv
