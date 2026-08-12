# Phase D12: Directional edge decision rule summary

## Baseline (argmax)
- 720d cost_on: -0.111968647467793
- 720d MDD: 0.15231130635904097
- 720d trades: 1491
- entries_attempted: 81905, entries_executed: 745

## All runs
| run_id | days | decision_mode | min_directional_edge | min_flat_margin | min_confidence | cost_on | MDD | trades | entries_attempted | entries_executed | signal_long | signal_short | signal_flat | directional_rule_reject | avg_long_edge_entry | avg_short_edge_entry | avg_pflat_rejected |
|--------|------|---------------|---------------------|-----------------|----------------|---------|-----|--------|-------------------|------------------|-------------|---------------|--------------|--------------------------|---------------------|----------------------|--------------------|
| phase_d12_baseline | 180 | argmax | None | None | None | -0.0027 | 0.0012 | 3 | 48 | 1 | 8 | 93 | 0 | None | None | None | None |
| phase_d12_baseline | 365 | argmax | None | None | None | -0.0027 | 0.0012 | 3 | 48 | 1 | 8 | 93 | 0 | None | None | None | None |
| phase_d12_baseline | 720 | argmax | None | None | None | -0.1120 | 0.1523 | 1491 | 81905 | 745 | 32473 | 78967 | 0 | None | None | None | None |

## Interpretation
- D12 changes signal generation (direction) not just entry gating; entries_attempted can differ from baseline.
- Compare entries_attempted, entries_executed, trades vs baseline; 720d cost_on and MDD.