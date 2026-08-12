# Phase D2.1 stability summary

- baseline: cost_on=-0.1026, trades=883
- TRADES_MIN=530, cost_on_range threshold=0.01

## All runs (run_id | cost_on | MDD | trades)

| run_id | 30d cost_on | 365d cost_on | 365d MDD | trades |
|--------|------------|-------------|----------|-------|
| phase_d21_p056_e125 | 0.0057 | -0.1015 | 0.1252 | 604 |
| phase_d21_p056_e125_spot1 | 0.0057 | -0.1015 | 0.1252 | 604 |
| phase_d21_p056_e125_spot2 | 0.0057 | -0.1015 | 0.1252 | 604 |
| phase_d21_p056_e130 | 0.0176 | -0.0870 | 0.1211 | 729 |
| phase_d21_p056_e130_spot1 | 0.0176 | -0.0870 | 0.1211 | 729 |
| phase_d21_p056_e130_spot2 | 0.0176 | -0.0870 | 0.1211 | 729 |
| phase_d21_p057_e130 | 0.0179 | -0.0651 | 0.1054 | 707 |
| phase_d21_p057_e130_spot1 | 0.0179 | -0.0651 | 0.1054 | 707 |
| phase_d21_p057_e130_spot2 | 0.0179 | -0.0651 | 0.1054 | 707 |

## Per-config stability (3 runs each)

### phase_d21_p056_e125 (p=0.56, e=1.25)
- cost_on_range: 0.0000
- MDD_range: 0.0000
- trades_range: 0
- mean 365d cost_on: -0.1015
- verdict: ADOPT_CANDIDATE

### phase_d21_p056_e130 (p=0.56, e=1.3)
- cost_on_range: 0.0000
- MDD_range: 0.0000
- trades_range: 0
- mean 365d cost_on: -0.0870
- verdict: ADOPT_CANDIDATE

### phase_d21_p057_e130 (p=0.57, e=1.3)
- cost_on_range: 0.0000
- MDD_range: 0.0000
- trades_range: 0
- mean 365d cost_on: -0.0651
- verdict: ADOPT_CANDIDATE

## BEST
- BEST run_id: **phase_d21_p057_e130**
- final verdict: **ADOPT_CANDIDATE**
