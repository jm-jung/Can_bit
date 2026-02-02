# Stage-2 v2.2 CAP 성능 평가 보고서

## 개요
Stage-2 v2.2 CAP이 실제로 성능에 도움이 되는지 판정하기 위한 종합 평가 결과입니다.

---

## STEP 1: CAP ON/OFF 비교

### 실행 조건
- **기간**: 2025-01-01 ~ 2025-03-01
- **전략**: ml_tcn
- **Symbol**: BTCUSDT
- **Timeframe**: 5m
- **Direction**: long
- **공통 파라미터**:
  - `min_hold_bars=12`
  - `cooldown_bars=12`
  - `strategy-guard-v2-mode=soft`
  - `strategy-guard-v2-scale-floor=0.02`

### 비교 결과

| 항목 | Guard v2 ON<br>Stage-2 OFF | Guard v2 ON<br>Stage-2 ON (CAP) | 차이 |
|------|------------------|----------------------|------|
| **Total Return** | -1.99% | -1.86% | **+0.13%** |
| **Max Drawdown** | 2.32% | 2.18% | **-0.14%** |
| **Total Trades** | 174 | 174 | 0 |
| **Win Rate** | 48.28% | 48.28% | 0% |
| **Sharpe Ratio** | -0.1352 | -0.1436 | -0.0084 |

### 결론
- CAP ON이 약간의 개선을 보임 (Return +0.13%, MaxDD -0.14%)
- 하지만 개선 폭이 매우 작고, Sharpe Ratio는 오히려 약간 악화됨
- Total Trades와 Win Rate는 동일하므로, CAP이 거래 빈도에는 영향을 주지 않음

---

## STEP 2: CAP 분포 분석

### 분석 방법
최근 실행 로그(`/tmp/step1_cap_on.log`)에서 `[STAGE2][CAP]` 로그를 파싱하여 CAP 값별 분포를 집계했습니다.

### CAP 분포 (로그 샘플 20개 기준)

| CAP 값 | 발생 횟수 | 전체 대비 비율 |
|--------|----------|---------------|
| **1.0** | 14 | **70.0%** |
| **0.55** | 4 | 20.0% |
| **0.35** | 2 | 10.0% |

### 전체 ENTRY 대비 CAP 적용 비율
- **Total Entries**: 87개
- **CAP 로그 샘플**: 20개 (전체의 약 23%)
- **CAP 적용 (cap<1.0)**: 6개 (로그 샘플 기준, 전체 대비 약 6.8%)
- **CAP 미적용 (cap=1.0)**: 14개 (로그 샘플 기준, 전체 대비 약 16.0%)

### 결론
- **CAP이 대부분의 경우 적용되지 않음** (70%가 cap=1.0)
- CAP이 보수적이거나, 실제로 필요한 경우가 적을 수 있음
- 현재 룰(high_entropy_th=0.66, mid_entropy_th=0.64)이 너무 엄격할 가능성

---

## STEP 3: CAP 룰 조정 미니 스윕

### 후보 룰
- **현재**: {0.35, 0.55, 1.0}
- **후보1**: {0.5, 0.7, 1.0} (더 완화된 CAP)
- **후보2**: {0.6, 0.8, 1.0} (더욱 완화된 CAP)

### 실행 계획
각 후보별로 단기 구간(2025-01-01~2025-03-01) 1회씩 실행하여 Return / MaxDD / Trades를 비교합니다.

### 참고
- CAP 룰 조정은 `calculate_stage2_cap()` 함수의 `cap_high`, `cap_mid` 파라미터를 변경하여 수행 가능
- 현재 함수 시그니처에 `cap_high`, `cap_mid`, `cap_default` 파라미터가 추가되어 있음

---

## STEP 4: 연속 CAP 함수 가능성 판단

### 현재 문제점 (계단형 CAP)
1. **과도한 이산화**: {0.35, 0.55, 1.0} 3단계만 존재하여, 중간 값들이 표현되지 않음
2. **임계값 근처 불연속성**: entropy나 p_diff가 임계값 근처에서 약간만 변해도 cap이 급격히 변함
3. **Guard v2 scale과의 곱셈**: `final_scale = guard_scale * cap`에서, cap이 이산적이면 final_scale도 이산적으로 변함

### 연속 CAP 함수 설계

#### 기본 아이디어
```python
# 연속 CAP 함수 예시
def calculate_continuous_cap(entropy, p_diff, 
                              entropy_th_low=0.60, entropy_th_high=0.70,
                              pdiff_th_low=0.001, pdiff_th_high=0.005):
    """
    entropy와 p_diff를 기반으로 연속적인 cap 값을 계산
    
    Returns:
        cap: 0.4 ~ 1.0 사이의 연속 값
    """
    # entropy 기반 감소 (높을수록 cap 감소)
    entropy_factor = 1.0 - clamp((entropy - entropy_th_low) / (entropy_th_high - entropy_th_low), 0, 1)
    
    # p_diff 기반 감소 (작을수록 cap 감소)
    pdiff_factor = clamp((p_diff - pdiff_th_low) / (pdiff_th_high - pdiff_th_low), 0, 1)
    
    # 두 요소의 곱으로 최종 cap 결정
    cap = 0.4 + 0.6 * entropy_factor * pdiff_factor
    
    return clamp(cap, 0.4, 1.0)
```

#### 기대 효과
1. **부드러운 전환**: entropy/p_diff 변화에 따라 cap이 연속적으로 변함
2. **더 정밀한 제어**: Guard v2 scale과 곱해질 때 final_scale도 연속적으로 변함
3. **과도한 페널티 회피**: 계단형보다 덜 공격적인 페널티 적용 가능

#### 구현 최소 변경 포인트
1. **`calculate_stage2_cap()` 함수 수정**:
   - 현재: 조건부 분기로 {0.35, 0.55, 1.0} 반환
   - 변경: 연속 함수로 0.4~1.0 사이 값 반환
2. **호출부 변경 불필요**: 반환 타입이 `float`이므로 기존 코드와 호환
3. **로그 형식 유지**: `cap=0.623` 같은 형식으로 출력 가능

### 결론
- 연속 CAP 함수는 **구현 가능**하며, **기대 효과가 있음**
- 현재 계단형 CAP이 보수적이고 적용 빈도가 낮으므로, 연속 함수로 전환하면 더 정밀한 제어가 가능할 것으로 예상

---

## 종합 결론 및 권장사항

### 현재 상태
1. **CAP ON이 약간의 개선을 보임** (Return +0.13%, MaxDD -0.14%)
2. **하지만 개선 폭이 매우 작음** (통계적으로 유의미하지 않을 수 있음)
3. **CAP 적용 빈도가 낮음** (70%가 cap=1.0, 즉 CAP 미적용)

### 권장사항

#### 옵션 1: CAP 유지 + 룰 조정
- **목적**: CAP 적용 빈도를 높여 효과를 극대화
- **방법**: 
  - entropy/p_diff 임계값 완화 (예: high_entropy_th=0.64, mid_entropy_th=0.62)
  - 또는 cap 값 완화 (예: {0.5, 0.7, 1.0})
- **예상 효과**: CAP 적용 빈도 증가 → 더 큰 성능 개선 가능

#### 옵션 2: 연속 CAP 함수로 전환
- **목적**: 더 정밀한 제어와 부드러운 전환
- **방법**: `calculate_stage2_cap()` 함수를 연속 함수로 변경
- **예상 효과**: Guard v2 scale과의 곱셈에서 더 정밀한 final_scale 제어

#### 옵션 3: CAP 제거
- **목적**: 복잡도 감소, Guard v2만으로 충분할 수 있음
- **방법**: Stage-2 CAP 기능 제거 또는 기본값으로 cap=1.0 고정
- **예상 효과**: 코드 단순화, 유지보수 용이

### 최종 권장사항
**옵션 1 (CAP 유지 + 룰 조정)**을 먼저 시도하고, 효과가 미미하면 **옵션 2 (연속 CAP 함수)**로 전환하는 것을 권장합니다.

---

## 부록: 실행 커맨드

### STEP 1-A: Guard v2 ON + Stage-2 OFF
```bash
python -m src.backtest.run_ml_xgb_backtest \
  --strategy ml_tcn --symbol BTCUSDT --timeframe 5m --direction long \
  --start-date 2025-01-01 --end-date 2025-03-01 \
  --use-optimized-threshold --signal-confirmation-bars 1 \
  --min-hold-bars 12 --cooldown-bars 12 \
  --use-strategy-guard --strategy-guard-v2 --strategy-guard-v2-mode soft \
  --strategy-guard-v2-scale-floor 0.02 \
  --no-save
```

### STEP 1-B: Guard v2 ON + Stage-2 ON (v2.2 CAP)
```bash
python -m src.backtest.run_ml_xgb_backtest \
  --strategy ml_tcn --symbol BTCUSDT --timeframe 5m --direction long \
  --start-date 2025-01-01 --end-date 2025-03-01 \
  --use-optimized-threshold --signal-confirmation-bars 1 \
  --min-hold-bars 12 --cooldown-bars 12 \
  --use-strategy-guard --strategy-guard-v2 --strategy-guard-v2-mode soft \
  --strategy-guard-v2-scale-floor 0.02 \
  --use-stage2 \
  --no-save
```

---

---

## 부록: 현재 기본값 (Default)

**Stage-2 CAP 임계값 기본값** (pdiff_small_005 승격):
- `high_entropy_th = 0.66`
- `mid_entropy_th = 0.64`
- `tiny_pdiff_th = 0.002`
- `small_pdiff_th = 0.005` (기본값, pdiff_small_005 승격, 기존 0.004)

**CLI로 override 예시**:
```bash
# baseline 재현 (small=0.004)
python -m src.backtest.run_ml_xgb_backtest \
  ... \
  --stage2-cap-pdiff-small-th 0.004

# 기본값 사용 (small=0.005, 옵션 생략 가능)
python -m src.backtest.run_ml_xgb_backtest \
  ... \
  --use-stage2
```

---

**작성일**: 2025-01-21  
**평가 기간**: 2025-01-01 ~ 2025-03-01 (단기 구간)  
**최종 업데이트**: 2026-01-22 (기본값 pdiff_small_005 승격)
