# Phase D7 entry tightening summary

## A) Baseline (phase_d7_baseline)
- min_max_proba=0.575, max_entropy=1.30
- 180d cost_on≈0.0125, 365d≈-0.0570, 720d≈-0.1120
- 720d MDD≈0.1523, 720d trades≈1491

## B) Full result table

| run_id | min_max_proba | max_entropy | 180d cost_on | 365d cost_on | 720d cost_on | 720d MDD | 720d trades |
|--------|---------------|-------------|--------------|--------------|--------------|----------|-------------|
| phase_d7_baseline | 0.575 | 1.30 | 0.0125 | -0.0570 | -0.1120 | 0.1523 | 1491 |
| phase_d7_p0580_e130 | 0.580 | 1.30 | 0.0050 | -0.0671 | -0.1260 | 0.1598 | 1467 |
| phase_d7_p0580_e135 | 0.580 | 1.35 | 0.0121 | -0.0904 | -0.1641 | 0.2150 | 1689 |
| phase_d7_p0585_e130 | 0.585 | 1.30 | -0.0054 | -0.0768 | -0.1271 | 0.1538 | 1453 |
| phase_d7_p0585_e135 | 0.585 | 1.35 | 0.0004 | -0.0945 | -0.1602 | 0.2030 | 1659 |
| phase_d7_p0590_e130 | 0.590 | 1.30 | -0.0278 | -0.0965 | -0.1302 | 0.1587 | 1425 |
| phase_d7_p0590_e135 | 0.590 | 1.35 | -0.0234 | -0.1132 | -0.1681 | 0.2094 | 1611 |

## C) Top3 by 720d cost_on

1. **phase_d7_baseline** (p=0.575, e=1.30) 720d cost_on=-0.1120
2. **phase_d7_p0580_e130** (p=0.580, e=1.30) 720d cost_on=-0.1260
3. **phase_d7_p0585_e130** (p=0.585, e=1.30) 720d cost_on=-0.1271

## BEST & verdict
- BEST run_id: **phase_d7_baseline**
- spotcheck stability: **STABLE**
- final verdict: **NO_IMPROVE**
