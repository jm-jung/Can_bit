# Phase D5 entry micro-tuning summary

- anchor (0.57, 1.30): cost_on≈-0.0651, MDD≈0.1054, trades≈707
- ADOPT: cost_on >= baseline+0.003, trades >= baseline*0.8, MDD <= baseline+0.01

| run_id | min_max_proba | max_entropy | 30d cost_on | 365d cost_on | 365d MDD | trades |
|--------|---------------|-------------|------------|-------------|----------|-------|
| phase_d5_p0565_e120 | 0.565 | 1.20 | 0.0044 | -0.1219 | 0.1439 | 502 |
| phase_d5_p0565_e125 | 0.565 | 1.25 | 0.0061 | -0.0971 | 0.1213 | 600 |
| phase_d5_p0565_e130 | 0.565 | 1.30 | 0.0172 | -0.0739 | 0.1140 | 719 |
| phase_d5_p0570_e120 | 0.570 | 1.20 | 0.0077 | -0.1192 | 0.1417 | 500 |
| phase_d5_p0570_e125 | 0.570 | 1.25 | 0.0101 | -0.0844 | 0.1099 | 592 |
| phase_d5_p0570_e130 | 0.570 | 1.30 | 0.0179 | -0.0651 | 0.1054 | 707 |
| phase_d5_p0575_e120 | 0.575 | 1.20 | 0.0087 | -0.1171 | 0.1404 | 498 |
| phase_d5_p0575_e125 | 0.575 | 1.25 | 0.0110 | -0.0828 | 0.1091 | 588 |
| phase_d5_p0575_e130 | 0.575 | 1.30 | 0.0192 | -0.0570 | 0.1007 | 699 |
| phase_d5_p0580_e120 | 0.580 | 1.20 | 0.0087 | -0.1183 | 0.1416 | 496 |
| phase_d5_p0580_e125 | 0.580 | 1.25 | 0.0077 | -0.0880 | 0.1113 | 582 |
| phase_d5_p0580_e130 | 0.580 | 1.30 | 0.0115 | -0.0671 | 0.1039 | 689 |

## BEST spotcheck
- cost_on_range: 0.0000
- MDD_range: 0.0000
- trades_range: 0
- spot stability: **STABLE**

## BEST & verdict
- BEST run_id: **phase_d5_p0575_e130**
- BEST min_max_proba: **0.575**, max_entropy: **1.3**
- final verdict: **ADOPT_CANDIDATE**
