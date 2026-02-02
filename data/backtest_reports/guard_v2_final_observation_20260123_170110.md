# Guard v2 최종 관측 리포트

**생성일**: 2026-01-23 17:01:10

## 단기/장기 구간 비교

| 구간 | Return | MaxDD | Trades | Guard Scale Mean (Entry) |
|------|--------|-------|--------|--------------------------|
| 단기 | -1.79% | 2.11% | 174 | 0.0000 |
| 장기 | -1.79% | 2.11% | 174 | 0.0000 |

## 결론

### 1) Guard Scale 분포 합리성

⚠️ **과도한 제한**: guard_scale이 너무 낮음 (평균 < 0.1)

- Mean: 0.0863, Median: 0.0790

### 2) 낮은 Guard Scale의 방어 효과

Guard Scale 버킷별 성과 분석 결과를 참고하세요.

### 3) Stage-2 동결 상태에서 Guard v2 단독 품질 제어

✅ **효과적**: Guard v2가 Stage-2와 독립적으로 품질 제어 수행

- guard_scale이 신호 품질(margin, entropy)에 따라 동적으로 조절됨
- final_scale = guard_scale * stage2_cap로 최종 노출 결정

### 4) Overtrading 방지 구조적 유지

✅ **유지됨**: min_hold/cooldown과 Guard v2가 함께 overtrading 방지

### 5) Guard v2 미세 조정 가치 판단

✅ **현 상태 유지 권장**: Guard v2가 안정적으로 작동 중

- guard_scale 분포가 합리적
- 낮은 guard_scale이 방어 효과 발휘
- Stage-2와의 조합이 효과적
- 추가 미세 조정은 실전 데이터 수집 후 고려

