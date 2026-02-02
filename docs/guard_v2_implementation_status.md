# StrategyGuard v2 Implementation Status

**Generated:** 2026-01-22

## 목적

Guard v1의 한계(win_rate/avg_return 기반, low trade frequency 환경에서 0/0 회피 문제)를 해결하기 위해 Guard v2를 설계/구현.

Guard v2는 **신호 품질/불확실성** 기반으로 판단하며, **position scaling (soft mode)**를 지원.

---

## 구현 완료 사항

### [1] Guard v2 클래스 (`src/backtest/strategy_guard.py`)

- ✅ `StrategyGuardV2Config`: enable_v2, mode(soft/hard), window_signal_stats, min_margin, max_entropy, scale_floor, block_if_scale_below
- ✅ `StrategyGuardV2`: update_signal(), check(), _calculate_entropy(), debug_snapshot()
- ✅ **입력 feature**: p_long, margin, entropy, recent_mean_margin, recent_mean_entropy
- ✅ **출력**: decision (ALLOW/BLOCK), position_scale (0.0~1.0)

### [2] CLI 옵션 (`src/backtest/run_ml_xgb_backtest.py`)

- ✅ `--strategy-guard-v2`
- ✅ `--strategy-guard-v2-mode {soft,hard}`
- ✅ `--strategy-guard-v2-window-signal-stats <int>`
- ✅ `--strategy-guard-v2-min-margin <float>`
- ✅ `--strategy-guard-v2-max-entropy <float>`
- ✅ `--strategy-guard-v2-scale-floor <float>`
- ✅ `--strategy-guard-v2-block-if-scale-below <float>`

### [3] 엔진 연동 (`src/backtest/ml_backtest_engines.py`)

- ✅ Guard v2 초기화 및 update_signal() 호출
- ✅ Guard v2 check() 호출 및 decision/position_scale 획득
- ✅ DECISION 로그 출력 (신호 발생 시점 + ENTRY 이벤트 시점)
- ⚠️ **position_scale 적용**: `entry_cost = base_entry_cost * position_scale`로 구현됨

### [4] 검증 결과

#### A) 단기 실행 (2025-01-01 ~ 2025-03-01)

- ✅ DECISION 로그: 221,596개 발견
- ✅ 필수 필드 포함: scale, p_long, margin, entropy, mean_margin, mean_entropy, mode, reason
- ⚠️ **Scale: 0.200 (고정)** - scale_floor가 항상 적용됨
- ⚠️ **Margin: 0.0000** - p_long이 threshold보다 낮아 margin이 0으로 계산됨

#### B) 장기 실행 (2023-01-01 ~ 2024-12-31)

```
Total Return: -100.00%
Max Drawdown: 100.00%
Total Trades: 221,596
Win Rate: 11.17%
Average Holding Period: 1.0 bars (5분)
```

**Baseline 대비:**
- Baseline: Return=-5.35%, MaxDD=5.21%
- Guard v2: Return=-100.00%, MaxDD=100.00%
- **결과: 크게 악화됨 ❌**

---

## 문제점 분석

### [A] Position Scale 미적용

**현재 구현:**
```python
entry_cost = base_entry_cost * position_scale
```

이것은 **수수료/슬리피지 비용만** 스케일링한 것이고, **실제 포지션 사이즈는 스케일링하지 않음**.

**올바른 구현 (필요):**
```python
position_size = base_position_size * position_scale
# 또는
balance_to_use = balance * position_scale
```

### [B] Margin 계산 문제

모든 샘플에서 `margin=0.0000`으로 나타남. 이는:
1. `p_long < threshold` 상황이 지속되거나
2. Margin 계산 로직에 문제가 있거나
3. Threshold 값이 p_long보다 항상 높게 설정됨

### [C] Scale 고정 (0.200)

`scale_floor=0.2`가 항상 적용되어, scale이 0.200으로 고정됨. 이는:
1. Margin과 entropy 조합이 항상 scale_floor 이하를 만들거나
2. Scale 계산 로직에 문제가 있음

### [D] Overtrading 지속

- Total Trades: 221,596 (매우 높음)
- Average Holding Period: 1.0 bars (5분)
- 이는 Guard v2가 **트레이드를 차단하지 못하고 있음**을 의미

---

## 다음 단계 권장사항

### [우선순위 1] Position Scale 올바른 적용

현재 `entry_cost`에만 적용된 것을 **실제 포지션 사이즈**에 적용하도록 수정 필요.

**Option A:** Balance 기반 스케일링
```python
effective_balance = balance * position_scale
# 이후 entry/exit 계산에 effective_balance 사용
```

**Option B:** Position 기반 스케일링
```python
position_to_enter = base_position * position_scale
```

### [우선순위 2] Margin/Entropy 계산 검증

- Threshold 값 확인 (너무 높게 설정되었는지)
- p_long 분포 확인 (대부분 < 0.5인지)
- Margin 계산 로직 검증 (`abs(p_long - threshold)` vs `p_long - threshold`)

### [우선순위 3] Scale 파라미터 조정

- `min_margin`을 낮추거나 (현재 default: 0.02)
- `max_entropy`를 높이거나 (현재 default: 0.65)
- `scale_floor`를 낮춰서 (현재 default: 0.2) scale 변동폭 확보

---

## CLI 사용 예시

### 단기 검증 (로그 확인용)

```bash
python -m src.backtest.run_ml_xgb_backtest \
  --strategy ml_tcn \
  --symbol BTCUSDT \
  --timeframe 5m \
  --direction long \
  --start-date 2025-01-01 \
  --end-date 2025-03-01 \
  --use-optimized-threshold \
  --signal-confirmation-bars 1 \
  --strategy-guard-v2 \
  --strategy-guard-v2-mode soft \
  --no-save
```

### 장기 실행 (성능 평가용)

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
  --strategy-guard-v2-min-margin 0.01 \
  --strategy-guard-v2-max-entropy 0.70 \
  --strategy-guard-v2-scale-floor 0.1 \
  --no-save
```

---

## 완료 기준 체크

- [x] 스윕 프로세스 중단 및 보존
- [x] Guard v2 코드/CLI 추가
- [x] 단기/장기 실행에서 v2 로그 정상 출력
- [x] v2 DECISION 로그에 필수 필드 포함 (scale, margin, entropy, etc.)
- [ ] **v2 soft mode에서 scale이 실제로 변동** ❌ (현재 0.200 고정)
- [ ] **거래가 0이 아닌 상태로 결과 산출** ✅ (221,596 trades, 하지만 너무 많음)
- [ ] **Baseline 대비 개선** ❌ (Return: -100% vs -5.35%, 크게 악화)

---

## 결론

**현재 상태:**
- Guard v2의 **로깅 인프라와 CLI는 정상 작동**
- 하지만 **position_scale이 실제 포지션 사이즈에 적용되지 않음**
- 결과적으로 **overtrading이 지속되고 성능이 크게 악화됨**

**필요 조치:**
1. Position scale 적용 방식 수정 (entry_cost → position_size or balance)
2. Margin/entropy 계산 로직 검증 및 파라미터 조정
3. 수정 후 재검증 루프 실행

---

**작성자:** Cursor Agent  
**날짜:** 2026-01-22
