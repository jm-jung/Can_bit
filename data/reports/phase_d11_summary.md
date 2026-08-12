# Phase D11: Directional gap + Flat suppression summary

## Baseline (no D11)
- 720d cost_on: -0.111968647467793
- 720d MDD: 0.15231130635904097
- 720d trades: 1491
- entries_attempted: 81905, entries_executed: 745

## All runs (by run_id)
| run_id | days | min_directional_gap | max_flat_entry_proba | cost_on | MDD | trades | entries_attempted | entries_executed | directional_gap_fail | flat_suppression_fail |
|--------|------|---------------------|----------------------|---------|-----|--------|-------------------|------------------|----------------------|------------------------|
| phase_d11_baseline | 180 | None | None | 0.0126 | 0.0186 | 345 | 18981 | 172 | 0 | 0 |
| phase_d11_baseline | 365 | None | None | -0.0734 | 0.0983 | 695 | 39810 | 347 | 0 | 0 |
| phase_d11_baseline | 720 | None | None | -0.1120 | 0.1523 | 1491 | 81905 | 745 | 0 | 0 |
| d11_a_gap0.04 | 180 | 0.04 | None | 0.0126 | 0.0186 | 345 | 18981 | 172 | 2224 | 0 |
| d11_a_gap0.04 | 365 | 0.04 | None | -0.0734 | 0.0983 | 695 | 39810 | 347 | 4761 | 0 |
| d11_a_gap0.04 | 720 | 0.04 | None | -0.1120 | 0.1523 | 1491 | 81905 | 745 | 9916 | 0 |
| d11_b_flat0.35 | 180 | None | 0.35 | 0.0126 | 0.0186 | 345 | 18981 | 172 | 0 | 1933 |
| d11_b_flat0.35 | 365 | None | 0.35 | -0.0734 | 0.0983 | 695 | 39810 | 347 | 0 | 3957 |
| d11_b_flat0.35 | 720 | None | 0.35 | -0.1120 | 0.1523 | 1491 | 81905 | 745 | 0 | 7656 |
| d11_c_gap0.04_flat0.35 | 180 | 0.04 | 0.35 | 0.0126 | 0.0186 | 345 | 18981 | 172 | 2224 | 1933 |
| d11_c_gap0.04_flat0.35 | 365 | 0.04 | 0.35 | -0.0734 | 0.0983 | 695 | 39810 | 347 | 4761 | 3957 |
| d11_c_gap0.04_flat0.35 | 720 | 0.04 | 0.35 | -0.1120 | 0.1523 | 1491 | 81905 | 745 | 9916 | 7656 |

## Interpretation
- If directional_gap_fail_count or flat_suppression_fail_count is large but entries_executed barely changed vs baseline, filter has low practical effect (D10.1-like).
- Compare 720d cost_on and trades vs baseline for improvement.