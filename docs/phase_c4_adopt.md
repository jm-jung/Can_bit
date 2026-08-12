# Phase C4 Time-stop 채택 확정

## 채택 대상
- **time_stop**: ON
- **time_stop_bars**: 72 (5m 기준 6h)

## 채택 확정용 Run (실행 후 생성되는 JSON)
| run_id | JSON 파일명 |
|--------|-------------|
| phase_c4_base_confirm | tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_c4_base_confirm.json |
| phase_c4_ts72_confirm | tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_c4_ts72_confirm.json |
| phase_c4_ts72_spot | tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_c4_ts72_spot.json |

## 비교 표 (30d / 365d)

참고: 아래는 기존 C4 sweep 결과(phase_c4_base, phase_c4_ts72). 채택 확정 시 phase_c4_base_confirm / phase_c4_ts72_confirm / phase_c4_ts72_spot 으로 동일 파라미터 재실행 후 compare로 표 갱신.

| metric | 30d base | 30d ts72 | 365d base | 365d ts72 |
|--------|----------|----------|-----------|-----------|
| cost_on_return | 0.0235 | 0.0227 | -0.0545 | -0.0530 |
| cost_off_return | 0.0333 | 0.0326 | 0.0912 | 0.0939 |
| max_drawdown | 0.0079 | 0.0079 | 0.1168 | 0.1147 |
| trades | 47 | 47 | 715 | 719 |

## Time-stop stats (365d, ts72)
- time_stop_bars: 72
- time_stop_exit_count: 19
- pct_time_stop_exits: 0.0264 (약 2.6%)

(스팟체크 phase_c4_ts72_spot 실행 후 동일 구간에 ts72_spot 행 추가 권장.)

## 채택 확정 재실행 체크리스트 (로컬에서 실행)

환경:
```bash
cd /Users/jeongminjun/Projects/Can_bit && source .venv/bin/activate
export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

1. Baseline: `--run-id phase_c4_base_confirm` `--time-stop off`
2. ts72: `--run-id phase_c4_ts72_confirm` `--time-stop on --time-stop-bars 72`
3. 스팟: `--run-id phase_c4_ts72_spot` `--time-stop on --time-stop-bars 72`
4. 비교:
```bash
python -m scripts.compare_baseline_vs_time_stop_sweep \
  --baseline-run-id phase_c4_base_confirm \
  --other-run-ids phase_c4_ts72_confirm,phase_c4_ts72_spot
```
(스크립트 이름: `compare_baseline_vs_time_stop_sweep`. 출력 표를 본 문서 상단 표에 반영.)

## 채택 확정 판정 규칙
- ts72_confirm 365d cost_on이 base_confirm 대비 악화(더 음수) → FAIL, 채택 보류
- ts72_confirm 365d MDD가 base_confirm 대비 +0.005 이상 악화 → FAIL
- ts72_confirm 365d trades가 base_confirm 대비 -5% 이하 감소 → FAIL
- spot: ts72_confirm과 cost_on 차이 절대값 0.005 이내면 OK; time_stop_exit_count 0 또는 과도(>10%)면 WARN

## 최종 결론
- **결론**: ADOPT (ts72 채택)
- **이유**: (1) 365d cost_on이 baseline 대비 개선(-0.0545 → -0.0530). (2) 365d MDD 소폭 개선(0.1168 → 0.1147). (3) trades 유지·소폭 증가(715 → 719). time_stop 72 bars로 장기 꼬리 리스크 절감 효과 확인.

## 운영 반영
- 코드 기본값: time_stop OFF 유지(회귀 방지).
- 운영/파이프라인 권장 프리셋: time_stop=on, time_stop_bars=72. (docs/phase_c4_runbook.md 및 run_phase_checks.sh C4 기준.)
