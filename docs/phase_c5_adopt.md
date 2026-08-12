# Phase C5 Time-stop micro sweep → 운영 bars 확정

## 흐름
- **C4**: time_stop 채택, ts72(72 bars) 선택.
- **C5**: time_stop_bars만 미세 조정(60, 72, 84, 96) → 가장 안정적인 값 확정.
- **변경**: 이번 Phase에서 변경은 time_stop_bars 1개뿐(early_exit 등 고정).

## C5 Run 구성
| run_id | time_stop_bars |
|--------|----------------|
| phase_c5_base_ts72 | 72 (baseline) |
| phase_c5_ts60 | 60 |
| phase_c5_ts84 | 84 |
| phase_c5_ts96 | 96 |

## 채택 기준 (baseline=ts72 대비)
1. 365d cost_on_return 최대(덜 음수)
2. 365d max_drawdown baseline 대비 +0.005 이상 악화 시 탈락
3. trades 감소율 -5% 초과 시 탈락
4. 30d cost_on이 baseline 대비 20% 이상 악화 시 탈락
5. 위 통과 후보 중 BEST 1개 → 최종 운영 bars

## 결과 요약
- **표·time_stop 통계**: `data/diagnostics/phase_c5_time_stop_sweep_summary.md` (compare 실행 후 자동 생성)
- **BEST run_id / 채택 bars**: 아래 "최종 운영 권장값"에 compare 완료 후 기입

## 최종 운영 권장값
- **time_stop**: on
- **time_stop_bars**: C5 compare 완료 후 `data/diagnostics/phase_c5_time_stop_sweep_summary.md`의 **Best run → 채택 time_stop_bars** 참고. (baseline 유지 시 72)

## 실행
```bash
./scripts/run_phase_checks.sh C5
```
또는 비교만:
```bash
python -m scripts.compare_baseline_vs_time_stop_sweep \
  --baseline-run-id phase_c5_base_ts72 \
  --other-run-ids phase_c5_ts60,phase_c5_ts84,phase_c5_ts96 \
  --summary-md phase_c5_time_stop_sweep_summary.md \
  --max-30d-cost-degradation 0.2
```
