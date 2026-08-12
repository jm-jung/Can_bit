# Phase D3 regime filter summary

- anchor: p057_e130, baseline (off) cost_on≈-0.0651, MDD≈0.1054, trades≈707
- adopt: cost_on >= -0.0751, MDD <= 0.0892 (≥15% MDD reduction)

| run_id | 30d cost_on | 365d cost_on | 365d MDD | trades |
|--------|------------|-------------|----------|-------|
| phase_d3_baseline | 0.0179 | -0.0651 | 0.1054 | 707 |
| phase_d3_ema200 | 0.0097 | -0.0763 | 0.1046 | 675 |
| phase_d3_ema200_slope | 0.0097 | -0.0749 | 0.1048 | 673 |
| phase_d3_vol_filter_0008 | 0.0179 | -0.0682 | 0.1083 | 701 |
| phase_d3_vol_filter_0010 | 0.0173 | -0.0648 | 0.1055 | 693 |

## BEST spotcheck (3 runs)
- cost_on_range: 0.0000
- MDD_range: 0.0000
- trades_range: 0

## BEST & verdict
- BEST run_id: **phase_d3_vol_filter_0010**
- final verdict: **REJECT**
