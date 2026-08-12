# Phase D4 position scaling summary

- anchor: p057_e130, regime off. baseline cost_on≈-0.0651, MDD≈0.1054, trades≈707
- adopt: cost_on >= baseline+0.003, MDD <= baseline+0.005, trades >= baseline*0.95

| run_id | 30d cost_on | 365d cost_on | 365d MDD | trades | mean_position_size | scaled_trade_ratio |
|--------|------------|-------------|----------|-------|-------------------|-------------------|
| phase_d4_baseline | 0.0179 | -0.0651 | 0.1054 | 707 | N/A | N/A |
| phase_d4_A | 0.0122 | -0.0720 | 0.0998 | 707 | 0.5427 | 0.0820 |
| phase_d4_B | 0.0122 | -0.0740 | 0.1010 | 707 | 0.5632 | 0.0820 |
| phase_d4_C | 0.0092 | -0.0788 | 0.1009 | 695 | 0.5696 | 0.0734 |
| phase_d4_D | 0.0094 | -0.0806 | 0.1019 | 695 | 0.6210 | 0.0734 |

## Scaling stats (detail)
- median_position_size: N/A (engine does not expose scale_median in validation result)
- size_bin_counts (365d, scaling runs only):
  - phase_d4_A: {'[0.0]': 0, '[0.25-0.4]': 20, '[0.4-0.6]': 18, '[0.6-0.8]': 11, '[0.8-1.0]': 9}
  - phase_d4_B: {'[0.0]': 0, '[0.25-0.4]': 24, '[0.4-0.6]': 10, '[0.6-0.8]': 10, '[0.8-1.0]': 14}
  - phase_d4_C: {'[0.0]': 29, '[0.25-0.4]': 18, '[0.4-0.6]': 12, '[0.6-0.8]': 8, '[0.8-1.0]': 13}
  - phase_d4_D: {'[0.0]': 29, '[0.25-0.4]': 17, '[0.4-0.6]': 10, '[0.6-0.8]': 5, '[0.8-1.0]': 19}

## BEST spotcheck
- cost_on_range: 0.0000
- MDD_range: 0.0000
- trades_range: 0
- spot stability: **STABLE**

## BEST & verdict
- BEST run_id: **phase_d4_baseline**
- final verdict: **NO_IMPROVE**
