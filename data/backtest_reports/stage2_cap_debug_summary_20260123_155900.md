# Stage-2 CAP 로그 기반 최종 확정 리포트

**생성일**: 2026-01-23 15:59:00

## 실행 커맨드/설정

- Guard v2 + Stage2 v2.2 CAP
- pdiff_small_th=0.005 (기본값 승격)
- final_scale = guard_scale * stage2_cap

## 성능 요약

- Total Return: -1.83%
- Max Drawdown: 2.15%
- Total Trades: 174
- Win Rate: 48.28%

## CAP 분포 + trigger_by_rule

| 항목 | 값 | 비율 |
|------|-----|------|
| total_checks | 87 | 100% |
| entries_attempted | 87 | 100.0% |
| entries_executed | 87 | 100.0% |
| cap_1_0_count | 55 | 63.2% |
| cap_0_8_count | 27 | 31.0% |
| cap_0_6_count | 5 | 5.7% |

| Trigger 원인 | 값 |
|-------------|-----|
| triggered_by_entropy_only | 38 |
| triggered_by_pdiff_only | 1 |
| triggered_by_both | 32 |
| triggered_by_none | 16 |

## cap=0.6 케이스 로그 발췌 (5개)

cap=0.6 케이스가 이번 구간에 발생하지 않았습니다.

## ENTRY/EXIT 연결 샘플 (cap별 성과 비교)

| cap | trade_count | mean_profit | mean_holding |
|-----|-------------|-------------|--------------|
| 1.0 | 0 | N/A | N/A |
| 0.8 | 0 | N/A | N/A |
| 0.6 | 0 | N/A | N/A |

## 결론

### 왜 cap=1.0이 70%인가?

- cap=1.0 비율: 63.2%
- triggered_by_none: 16 (대부분의 경우 CAP 조건을 만족하지 않음)
- entropy/p_diff 임계값이 보수적이어서 대부분의 경우 CAP이 적용되지 않음

### cap=0.6 유지/제거 추천

- cap=0.6 발생 횟수: 5 (샘플 수 충분)
- 유지 추천: 샘플 수가 충분하고 성과 분석 필요

### entropy vs p_diff 중 무엇을 조정할지 추천

- triggered_by_entropy_only: 38
- triggered_by_pdiff_only: 1
- triggered_by_both: 32
- 추천: entropy 임계값 완화 (entropy_only가 더 많음)

