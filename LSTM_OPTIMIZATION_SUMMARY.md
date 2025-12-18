# LSTM Threshold Optimizer 개선 요약

## 1. 현재 신호 생성 로직 요약

**파일**: `src/backtest/ml_backtest_engine_impl.py`  
**함수**: `LstmAttnBacktestEngine.generate_signals()` (line 325-456)

### 현재 로직:
```python
# Line 374-379 (수정 전)
if p_long >= long_threshold and p_long >= p_short:
    desired_direction = "LONG"
elif short_threshold is not None and p_short >= short_threshold and p_short > p_long:
    desired_direction = "SHORT"
else:
    desired_direction = "HOLD"  # FLAT
```

### 특징:
- **Threshold 체크**: `p_long >= long_threshold` AND `p_long >= p_short` (LONG)
- **충돌 처리**: `p_long >= p_short` 조건으로 LONG 우선
- **FLAT 미활용**: `proba_flat` 계산하지만 threshold로 사용 안 함
- **노이즈 취약**: 차이가 0.001만 있어도 진입

### 문제점:
1. proba_flat 미활용 (3-class 모델의 핵심)
2. 신호 품질(confidence margin) 미고려
3. 평균 proba 근처(0.39/0.38)인데 threshold가 0.45~0.60으로 높아 신호 부족

---

## 2. 조정 축 후보 리스트 (우선순위별)

### 우선순위 A: 신호 필터/게이팅 (코드 변경 대비 효율 높음) ✅ 구현 완료

#### A1. **confidence_margin** (신호 품질 마진)
- **위치**: `generate_signals()` (line 386-410)
- **필요 이유**: mean_long≈0.39, mean_short≈0.38로 가까워 노이즈에 취약
- **효과**: 낮은 확신도 신호 필터링 → 거래 품질 향상
- **범위**: 0.00 ~ 0.10, step 0.01
- **로직**: `(p_long - p_short) >= confidence_margin` (LONG), `(p_short - p_long) >= confidence_margin` (SHORT)

#### A2. **flat_threshold** (FLAT 클래스 threshold)
- **위치**: `generate_signals()` (line 387)
- **필요 이유**: proba_flat 계산하지만 사용 안 함, 불확실성 신호 활용
- **효과**: 불확실한 시장에서 진입 방지
- **범위**: 0.20 ~ 0.40, step 0.05 (None = 비활성화)
- **로직**: `if p_flat >= flat_threshold: desired_direction = "HOLD"` (LONG/SHORT 체크 전)

#### A3. **min_proba_dominance** (최소 확률 우위)
- **위치**: `generate_signals()` (line 400, 407)
- **필요 이유**: `p_long >= p_short`만 체크 → 0.001 차이로도 진입
- **효과**: 명확한 방향성만 진입
- **범위**: 0.02 ~ 0.10, step 0.01
- **로직**: `(p_long - p_short) >= min_proba_dominance` (LONG), `(p_short - p_long) >= min_proba_dominance` (SHORT)

### 우선순위 B: 포지션 관리

#### B4. **entry_cooldown_bars** (진입 쿨다운)
- **위치**: `execute_trades()`
- **필요 이유**: 연속 진입으로 거래 비용 증가
- **효과**: 거래 빈도 감소, 비용 절감
- **범위**: 0 ~ 20 bars, step 5

#### B5. **min_hold_bars** (최소 보유 기간)
- **위치**: `execute_trades()`
- **필요 이유**: 너무 짧은 보유로 수수료만 발생
- **효과**: 단기 노이즈 거래 방지
- **범위**: 1 ~ 10 bars, step 1

### 우선순위 C: 비용/슬리피지 모델

#### C6. **spread_filter_pct** (스프레드 필터)
- **위치**: 신호 생성 또는 포지션 관리
- **필요 이유**: 변동성이 높을 때 스프레드가 커져 수익성 저하
- **효과**: 불리한 시장 조건에서 진입 방지
- **범위**: 0.0 ~ 0.5%, step 0.1%

### 우선순위 D: 모델/라벨링 (변경 범위 큼)

#### D7. **signal_smoothing_window** (신호 스무딩)
- **위치**: 신호 생성 후처리
- **필요 이유**: 단기 변동성 필터링
- **효과**: 노이즈 감소
- **범위**: 1 ~ 10 bars, step 1

#### D8. **volatility_filter** (변동성 필터)
- **위치**: 신호 생성
- **필요 이유**: 고변동성 시장에서 모델 성능 저하
- **효과**: 불안정한 시장에서 진입 방지
- **범위**: 0.0 ~ 2.0 (ATR/close 비율), step 0.2

---

## 3. 최적화 루프 확장 구현 계획

### ✅ 완료된 작업

1. **신호 생성 로직 확장** (`src/backtest/ml_backtest_engine_impl.py`)
   - `LstmAttnBacktestEngine.generate_signals()`에 3개 필터 추가
   - 추상 메서드 시그니처 업데이트
   - XGBoost 엔진 호환성 유지

2. **CLI 옵션 추가** (`src/optimization/optimize_ml_threshold.py`)
   - `--flat-th-min/max/step`, `--conf-margin-min/max/step`, `--min-dominance-min/max/step` 추가
   - 그리드 생성 로직 확장 (flat_threshold_candidates, confidence_margin_candidates, min_proba_dominance_candidates)

### 🔄 필요한 작업 (구조적 변경)

**현재 구조의 한계**:
- `_optimize_with_overfit_awareness()`는 `(long_thr, short_thr)` 조합만 탐색
- 새 파라미터 축을 포함하려면 nested loop 또는 `itertools.product` 필요

**수정 파일**:
1. `src/optimization/threshold_optimizer.py`
   - `_optimize_with_overfit_awareness()`: 새 파라미터 그리드 받기
   - 조합 생성 로직 확장: `itertools.product(long_candidates, short_candidates, flat_th_candidates, conf_margin_candidates, min_dom_candidates)`
   - `_evaluate_single_threshold_combination()`: 새 파라미터 전달
   - `_worker_evaluate_threshold()`: worker args 튜플 확장

2. `src/optimization/optimize_ml_threshold.py`
   - `run_threshold_optimization_for_ml_strategy()`: 새 파라미터 그리드를 `optimize_threshold_for_strategy()`로 전달
   - `optimize_threshold_for_strategy()`: 새 파라미터 받아서 `_optimize_with_overfit_awareness()`로 전달

**구현 예시**:
```python
# threshold_optimizer.py
from itertools import product

# 조합 생성
all_combinations = list(product(
    long_threshold_candidates,
    short_threshold_candidates,
    flat_threshold_candidates,      # 새로 추가
    confidence_margin_candidates,    # 새로 추가
    min_proba_dominance_candidates  # 새로 추가
))

# 평가 함수 시그니처 확장
def _evaluate_single_threshold_combination(
    long_thr: float,
    short_thr: float | None,
    flat_th: float | None,          # 새로 추가
    conf_margin: float,             # 새로 추가
    min_dom: float,                 # 새로 추가
    ...
) -> Dict[str, Any]:
    # engine.run_backtest() 호출 시 새 파라미터 전달
    result_in = engine.run_backtest(
        ...
        flat_threshold=flat_th,
        confidence_margin=conf_margin,
        min_proba_dominance=min_dom,
    )
```

---

## 4. 평가 지표 강화

### 현재 지표:
- sharpe_in/out, total_return_in/out, win_rate_in/out, trades_in/out, max_drawdown

### 추가 지표 제안:

#### profit_factor (수익 팩터)
- **계산**: `sum(winning_trades) / abs(sum(losing_trades))`
- **의미**: 이익/손실 비율, 1.0 이상이면 수익성
- **위치**: `_compute_trade_stats()` 또는 `_evaluate_single_threshold_combination()`

#### exposure (노출도)
- **계산**: `sum(holding_bars) / total_bars`
- **의미**: 시장 노출 비율
- **위치**: `execute_trades()`에서 추적

#### turnover (회전율)
- **계산**: `total_trades / (total_bars / avg_holding_bars)`
- **의미**: 거래 빈도
- **위치**: `execute_trades()`에서 계산

### 제약 조건 강화:

```python
# 현재
min_trades_out = 50  # 너무 낮음

# 개선안
min_trades_out = 200  # 통계적 유의성 확보
min_profit_factor = 1.2  # 최소 수익 팩터
max_exposure = 0.8  # 최대 노출도
max_turnover = 10.0  # 최대 회전율
```

---

## 5. 즉시 실행 가능한 실험 커맨드

### 실험 1: Threshold 범위를 mean_proba 근처로 조정 ✅
```powershell
python -m src.optimization.threshold_optimizer `
    --strategy ml_lstm_attn `
    --symbol BTCUSDT `
    --timeframe 5m `
    --no-parallel `
    --long-min 0.35 `
    --long-max 0.45 `
    --long-step 0.02 `
    --short-min 0.30 `
    --short-max 0.45 `
    --short-step 0.02 `
    --min-trades-out 200 `
    --min-sharpe-out 0.1
```
**목적**: 평균 proba 근처에서 더 많은 신호 발생, 거래 수 증가

### 실험 2: confidence_margin 필터 포함 ⚠️ (구조 변경 필요)
```powershell
# 현재는 최적화 루프가 새 파라미터를 탐색하지 않음
# 임시 해결책: 코드에서 직접 수정하여 테스트
# src/backtest/ml_backtest_engine_impl.py의 run_backtest에서
# generate_signals 호출 시 confidence_margin=0.03 등으로 설정
```

### 실험 3: min_trades_out 상향 "견고성" 실험 ✅
```powershell
python -m src.optimization.threshold_optimizer `
    --strategy ml_lstm_attn `
    --symbol BTCUSDT `
    --timeframe 5m `
    --no-parallel `
    --long-min 0.35 `
    --long-max 0.50 `
    --long-step 0.03 `
    --short-min 0.30 `
    --short-max 0.45 `
    --short-step 0.03 `
    --min-trades-out 300 `
    --min-sharpe-out 0.2 `
    --min-trades-in 1000
```
**목적**: 통계적 유의성 확보, "덜 하는게 덜 지는" 조합 제거

---

## 6. 리스크/주의사항

### 과최적화 (Overfitting)
- **위험**: 파라미터 축이 많아질수록 과최적화 위험 증가
- **완화**: 
  - out-of-sample 비율 증가 (현재 30% → 40%)
  - gap penalty 강화 (현재 alpha=0.5 → 0.7)
  - 교차 검증 고려

### Lookahead Bias
- **위험**: 미래 정보 누수 (전체 데이터로 proba 계산 후 split)
- **현재 상태**: `get_or_build_predictions()`가 전체 데이터로 proba 계산 → split
- **권장**: Walk-forward 또는 time-series cross-validation 고려

### 비용 모델
- **현재**: 고정 commission (0.04%) + slippage (0.05%)
- **위험**: 실제 거래에서는 변동성에 따라 slippage 증가
- **개선**: 변동성 기반 동적 slippage 모델 고려

### 계산 비용
- **현재**: 49 combinations (7 long × 7 short)
- **추가 후**: 예상 수백~수천 combinations
- **완화**: 
  - 단계적 탐색 (먼저 threshold만, 그 다음 필터 추가)
  - 병렬 처리 활용 (LSTM은 serial 권장이지만 작은 그리드는 parallel 가능)
  - 조기 종료 (min_trades_out 미달 시 즉시 skip)

---

## 구현 상태 요약

### ✅ 완료
1. 신호 생성 로직 확장 (3개 필터 추가)
2. CLI 옵션 추가
3. 그리드 생성 로직 확장

### 🔄 필요
1. 최적화 루프 구조 변경 (nested loop 또는 itertools.product)
2. Worker args 확장 (parallel execution 지원)
3. 평가 지표 강화 (profit_factor, exposure, turnover)

### 📝 다음 단계
1. **즉시 실행**: 실험 1, 3 실행하여 baseline 확보
2. **구조 변경**: 최적화 루프에 새 파라미터 축 추가
3. **평가 지표**: profit_factor, exposure, turnover 계산 및 필터링
4. **검증**: 각 단계별 실험 실행 및 결과 비교

