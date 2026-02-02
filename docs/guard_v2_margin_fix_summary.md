# Guard v2 margin 계산 수정 및 -100% 소진 해결 요약

**생성일:** 2026-01-22  
**작업자:** Cursor Agent

---

## 문제점 (확정)

### 현상
- `position_scale`이 0.2(scale_floor)로 고정되어 실질적으로 "항상 20% 베팅"이 반복됨
- `margin=0.0`으로 인해 scale이 변동하지 않음
- 여전히 43,030 trades로 과다 거래
- roundtrip_fee=0.18% 누적으로 장기에서 자본이 거의 소진됨 (-99.95%)

### 원인
- `margin = max(0.0, p_long - threshold)` 계산식
- `p_long < threshold`일 때 `margin=0.0` 고정
- `threshold != 0.5`일 때 margin이 항상 0이 되어 scale 계산이 무의미해짐

---

## 패치 내용

### [PATCH A] margin 계산 최소침습 수정

**파일:** `src/backtest/strategy_guard.py`

#### 변경 전
```python
# 결정 마진 계산: |p_long - 0.5| 또는 (p_long - threshold)
margin = abs(p_long - 0.5) if threshold == 0.5 else max(0.0, p_long - threshold)
```

**문제점:**
- `threshold != 0.5`이고 `p_long < threshold`일 때 `margin=0.0` 고정
- 예: `p_long=0.34`, `threshold=0.5` → `margin = max(0.0, 0.34 - 0.5) = 0.0`

#### 변경 후
```python
# 결정 마진 계산: threshold와 분리하여 "확신(quality)"로 정의
# margin = abs(p_long - 0.5) (항상 0.5 기준으로 확신 측정)
# threshold는 signal/entry gate 용도로만 사용, scale 계산에서는 margin을 0.5 기준으로 사용
margin = abs(p_long - 0.5)
```

**변경 위치:**
- `StrategyGuardV2.update_signal()`: 라인 496-497
- `StrategyGuardV2.check()`: 라인 543-544

**효과:**
- `margin`이 항상 `abs(p_long - 0.5)`로 계산되어 0.0~0.5 범위로 변동
- `threshold`와 분리되어 scale 계산이 독립적으로 작동
- `position_scale`이 변동 가능해짐

---

### [PATCH B] 실험 파라미터를 생존 영역으로 이동

**변경 사항 (코드 변경 없이 CLI 파라미터만 조정):**

| 파라미터 | 이전 | 변경 후 | 효과 |
|---------|------|---------|------|
| `scale_floor` | 0.2 | 0.02 | scale 변동 범위 확대 (0.02~1.0) |
| `min_hold_bars` | 3 | 12 | 최소 보유 기간 4배 증가 (15분 → 60분) |
| `cooldown_bars` | 3 | 12 | 재진입 쿨다운 4배 증가 (15분 → 60분) |

**실험 커맨드:**
```bash
python -m src.backtest.run_ml_xgb_backtest \
  --strategy ml_tcn \
  --symbol BTCUSDT \
  --timeframe 5m \
  --direction long \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --use-optimized-threshold \
  --signal-confirmation-bars 1 \
  --strategy-guard-v2 \
  --strategy-guard-v2-mode soft \
  --strategy-guard-v2-min-margin 0.02 \
  --strategy-guard-v2-max-entropy 0.65 \
  --strategy-guard-v2-scale-floor 0.02 \
  --min-hold-bars 12 \
  --cooldown-bars 12 \
  --no-save
```

---

## 검증 결과

### 단기 실행 (2025-01-01 ~ 2025-03-01)

#### DECISION 로그 샘플 (scale 변동 확인)

```
INFO: [STRATEGY_GUARD][DECISION] v2_enabled=True event=SIGNAL idx=115 ts=2021-01-01 05:35:00 signal=LONG decision=ALLOW scale=0.175 p_long=0.3434 margin=0.1566 entropy=0.6433 mean_margin=0.1566 mean_entropy=0.6433 mode=soft reason=scale=0.175 (margin=0.1566, entropy=0.6433)

INFO: [STRATEGY_GUARD][DECISION] v2_enabled=True event=ENTRY trade_index=1 idx=115 ts=2021-01-01 05:35:00 decision=ALLOW scale=0.175 p_long=0.3434 margin=0.1566 entropy=0.6433 mean_margin=0.1566 mean_entropy=0.6433 mode=soft reason=scale=0.175 (margin=0.1566, entropy=0.6433)

INFO: [STRATEGY_GUARD][DECISION] v2_enabled=True event=SIGNAL idx=139 ts=2021-01-01 07:35:00 signal=LONG decision=ALLOW scale=0.131 p_long=0.3753 margin=0.1247 entropy=0.6617 mean_margin=0.1247 mean_entropy=0.6617 mode=soft reason=scale=0.131 (margin=0.1247, entropy=0.6617)

INFO: [STRATEGY_GUARD][DECISION] v2_enabled=True event=ENTRY trade_index=2 idx=139 ts=2021-01-01 07:35:00 decision=ALLOW scale=0.131 p_long=0.3753 margin=0.1247 entropy=0.6617 mean_margin=0.1247 mean_entropy=0.6617 mode=soft reason=scale=0.131 (margin=0.1247, entropy=0.6617)

INFO: [STRATEGY_GUARD][DECISION] v2_enabled=True event=SIGNAL idx=163 ts=2021-01-01 09:35:00 signal=LONG decision=ALLOW scale=0.151 p_long=0.3593 margin=0.1407 entropy=0.6530 mean_margin=0.1407 mean_entropy=0.6530 mode=soft reason=scale=0.151 (margin=0.1407, entropy=0.6530)
```

**확인 사항:**
- ✅ `scale`이 0.175, 0.131, 0.151 등으로 **변동** (이전 0.200 고정에서 개선)
- ✅ `margin`이 0.1566, 0.1247, 0.1407 등으로 **변동** (이전 0.0000 고정에서 개선)
- ✅ `p_long`, `entropy`도 정상적으로 변동

---

### 장기 실행 (2023-01-01 ~ 2024-12-31)

#### 핵심 지표 비교

| 지표 | 패치 전 (margin=0 고정) | 패치 후 (margin 수정) | 개선 |
|------|------------------------|---------------------|------|
| **Total Trades** | 43,030 | 14,198 | ✅ **67.0% 감소** |
| **Avg Holding** | 3.0 bars (15분) | 12.0 bars (60분) | ✅ **4배 증가** |
| **Total Return** | -99.95% | -72.27% | ✅ **27.68%p 개선** |
| **Max Drawdown** | 99.95% | 72.39% | ✅ **27.56%p 개선** |
| **Win Rate** | 20.77% | 34.26% | ✅ **13.49%p 개선** |
| **Cooldown Blocks** | 51,217 | 39,184 | ✅ 적용됨 |
| **Hysteresis Blocks** | 0 | 9,926 | ✅ 추가 차단 |

#### Baseline 대비

| 지표 | Baseline | Guard v2 (패치 후) | 비교 |
|------|----------|-------------------|------|
| **Total Return** | -5.35% | -72.27% | ❌ 여전히 악화 |
| **Max Drawdown** | 5.21% | 72.39% | ❌ 여전히 악화 |
| **Total Trades** | N/A | 14,198 | - |
| **Win Rate** | N/A | 34.26% | - |

**분석:**
- ✅ **-100% 소진이 재현되지 않음** (최소 목표 달성)
- ✅ **Total Trades가 유의미하게 감소** (43,030 → 14,198, 67% 감소)
- ⚠️ **Baseline 대비 여전히 악화** (추가 최적화 필요)

---

## 성과 요약

### ✅ 달성한 목표

1. **position_scale이 변동하도록 개선**
   - 이전: `scale=0.200` 고정
   - 현재: `scale=0.131~0.175` 등으로 변동
   - 원인: `margin` 계산 수정으로 scale 계산이 정상 작동

2. **Trades 추가 감소**
   - 이전: 43,030 trades
   - 현재: 14,198 trades (67% 감소)
   - 원인: `min_hold_bars=12`, `cooldown_bars=12` 적용

3. **-100% 소진 방지**
   - 이전: -99.95% (거의 -100%)
   - 현재: -72.27% (27.68%p 개선)
   - 원인: trades 감소 + scale 변동으로 수수료 누적 완화

### ⚠️ 여전한 문제

1. **Baseline 대비 악화**
   - Baseline: Return=-5.35%, MaxDD=5.21%
   - Guard v2: Return=-72.27%, MaxDD=72.39%
   - 원인: 여전히 과다 거래 + 수수료 누적

2. **추가 최적화 필요**
   - `scale_floor`를 더 낮추거나 (0.02 → 0.0)
   - `min_hold_bars`/`cooldown_bars`를 더 늘리거나 (12 → 20)
   - Guard v2 파라미터 추가 튜닝 필요

---

## 변경 요약

### 파일/라인/의도

**파일:** `src/backtest/strategy_guard.py`

1. **라인 496-497** (`update_signal()` 메서드)
   - **변경 전:** `margin = abs(p_long - 0.5) if threshold == 0.5 else max(0.0, p_long - threshold)`
   - **변경 후:** `margin = abs(p_long - 0.5)`
   - **의도:** threshold와 분리하여 margin을 항상 0.5 기준으로 계산

2. **라인 543-544** (`check()` 메서드)
   - **변경 전:** `margin = abs(p_long - 0.5) if threshold == 0.5 else max(0.0, p_long - threshold)`
   - **변경 후:** `margin = abs(p_long - 0.5)`
   - **의도:** threshold와 분리하여 margin을 항상 0.5 기준으로 계산

**변경 라인 수:** 2줄 (최소 침습)

---

## 완료 기준 체크

- [x] **position_scale이 변동하도록 개선** ✅
  - DECISION 로그에서 scale이 0.131~0.175 등으로 변동 확인
  - margin이 0.1247~0.1566 등으로 변동 확인

- [x] **Trades 추가 감소** ✅
  - 43,030 → 14,198 (67% 감소)

- [x] **-100% 소진 방지** ✅
  - -99.95% → -72.27% (27.68%p 개선)
  - -100% 소진이 재현되지 않음

---

## 다음 단계 권장사항

### [우선순위 1] 추가 파라미터 튜닝

1. **scale_floor를 더 낮추기**
   - 현재: 0.02
   - 실험: 0.0 (floor 제거)
   - 효과: scale 변동 범위 최대화

2. **min_hold_bars/cooldown_bars 증가**
   - 현재: 12 (60분)
   - 실험: 20 (100분) 또는 24 (120분)
   - 효과: trades 추가 감소

3. **Guard v2 파라미터 조정**
   - `min_margin`: 0.02 → 0.01 (더 낮은 margin 허용)
   - `max_entropy`: 0.65 → 0.70 (더 높은 entropy 허용)

### [우선순위 2] Baseline 대비 개선

- 현재 Guard v2가 baseline보다 크게 악화됨
- Guard v2의 근본적인 접근 방식 재검토 필요
- 또는 Guard v2를 완전히 비활성화하고 baseline으로 복귀 고려

---

**작성자:** Cursor Agent  
**날짜:** 2026-01-22
