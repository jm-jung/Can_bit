# Phase D2 GPT용 한 장 요약

## 목표
- D1_v2에서 거래 수가 -34% 줄어든 것을 일부 회복 (목표 620±50)하면서 cost_on 개선 유지.
- 정책: trades 하한을 -30% → -40% 완화 (baseline×0.60 이상이면 OK).

## 실험 설계
- **Anchor**: D1_v2 BEST (min_max_proba=0.58, max_entropy=1.25).
- **스윕**: anchor 주변만 탐색. min_max_proba ∈ {0.56, 0.57, 0.58}, max_entropy ∈ {1.20, 1.25, 1.30} → 3×3 = 9 runs.
- **고정**: symbol=BTCUSDT, 5m, end_date=2026-03-03, min_hold=36, cooldown=12, regime off, position_scaling off, early_exit on, time_stop 72.

---

## Baseline (검증용)
- phase_d2_baseline: min_max_proba=0.55, max_entropy=1.35
- 365d: cost_on=-0.1026, trades=883 (D1_v2와 동일 구간·동일 수준으로 확인됨)

---

## 스윕 결과 (365d 기준 요약)
| run_id | 365d cost_on | 365d MDD | trades |
|--------|-------------|----------|--------|
| phase_d2_p056_e130 | -0.0651 | 0.1054 | 707 |
| phase_d2_p057_e130 | -0.0671 | 0.1039 | 689 |
| phase_d2_p056_e125 | -0.0844 | 0.1099 | 592 |
| phase_d2_p057_e125 | -0.0880 | 0.1113 | 582 |
| … (나머지 5개는 cost_on 더 음수 또는 trades 더 적음) |

- **Top1 (365d cost_on 최대)**: phase_d2_p056_e130 — cost_on=-0.0651, MDD=0.1054, **trades=707** (목표 620±50 구간 진입).

---

## BEST 후보 및 스팟체크
- **BEST run_id**: phase_d2_p056_e130 (min_max_proba=0.56, max_entropy=1.30)
- 스팟체크 3회 (동일 파라미터 재실행):

| run | 365d cost_on | 365d MDD | trades |
|-----|-------------|----------|--------|
| phase_d2_p056_e130 (원본) | -0.0651 | 0.1054 | 707 |
| phase_d2_best_spot1 | -0.0870 | 0.1211 | 729 |
| phase_d2_best_spot2 | -0.0870 | 0.1211 | 729 |

- **cost_on_range**: 0.0219 (3회 중 최대−최소)
- **MDD_range**: 0.0157, **trades_range**: 22

---

## Verdict 기준 (스펙)
- **ADOPT**: (1) cost_on이 baseline 대비 개선 **and** (2) trades ≥ baseline×0.60 (530 이상) **and** (3) 스팟체크 cost_on_range ≤ 0.002
- 여기서 (3) 미충족: cost_on_range=0.0219 > 0.002 → **REJECT** (스팟 안정성 미달).

---

## 부가 통계 (BEST 365d 트레이드 로그)
- mean_hold: 43.7 bars, median_hold: 36
- exit_reason: signal_exit_th 209, early_exit 58, time_stop 20

---

## GPT에게 요청할 판단 (참고)
- D2 BEST( p056_e130 )는 **거래 수 회복(707)**과 **cost_on 개선(-0.0651 vs baseline -0.1026)**은 달성했으나, **스팟체크 시 cost_on 편차(0.0219)**가 커서 규칙상 ADOPT는 아님.
- 이 결과를 보고 **운영 반영 여부**(그대로 채택 / 조건부 채택 / 재실험 권고 등)와 **다음 액션**(예: p057_e130 검토, 또는 다른 파라미터 추가 스윕)에 대한 판단을 요청할 때 사용.
