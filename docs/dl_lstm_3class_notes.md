# LSTM 3-Class Classification 리팩터링 문서

## 개요

기존 LSTM Attention 기반 이진 분류 전략(`ml_lstm_attn`)을 **3-class (FLAT/LONG/SHORT) 분류 전략**으로 리팩터링했습니다.

## 변경 사항 요약

### 1. 레이블링 시스템 (STEP 1)

**파일:**
- `src/dl/data/labels.py` (신규 생성)
- `src/dl/train/train_lstm_attn.py` (`create_sequences` 함수 수정)

**변경 내용:**
- 기존: 이진 분류 (0: 하락, 1: 상승)
- 변경: 3-class 분류
  - Class 0 (FLAT): `-neg_threshold <= r <= pos_threshold`
  - Class 1 (LONG): `r > pos_threshold`
  - Class 2 (SHORT): `r < -neg_threshold`

**레이블 생성 기준:**
- `pos_threshold = 0.001` (기본값, 설정 가능)
- `neg_threshold = 0.001` (기본값, 설정 가능)
- `r = (close[t+horizon] / close[t]) - 1`

**클래스 인덱스 매핑:**
```python
class LstmClassIndex(enum.IntEnum):
    FLAT = 0
    LONG = 1
    SHORT = 2
```

### 2. 모델 구조 변경 (STEP 2)

**파일:**
- `src/dl/models/lstm_attn.py`

**변경 내용:**
- 기존: `Dense(1) + sigmoid` → binary classification
- 변경: `Dense(3) + softmax` → 3-class classification

**출력 형태:**
- 모델 출력: `logits` shape `(batch, 3)`
- 추론 시: `F.softmax(logits, dim=-1)` → `[p_flat, p_long, p_short]`

### 3. 학습 파이프라인 변경 (STEP 2)

**파일:**
- `src/dl/train/train_lstm_attn.py`

**변경 내용:**
- Loss 함수: `BCEWithLogitsLoss` → `CrossEntropyLoss`
- 레이블 타입: `torch.FloatTensor` → `torch.LongTensor` (integer labels)
- Class weights: 역빈도 가중치 자동 계산 및 적용
- Metrics: 3-class에 맞게 accuracy 및 class별 분포 로깅

**Dataset 클래스:**
- `TimeSeriesDataset`: binary와 3-class 모두 지원 (레이블 dtype으로 자동 판단)

### 4. 추론 인터페이스 변경 (STEP 3)

**파일:**
- `src/dl/lstm_attn_model.py`

**변경 내용:**
- `predict_proba_latest()`: 3-class softmax에서 `p_long` 추출하여 반환 (하위 호환 유지)
- `predict_proba_batch()`: 3-class softmax에서 `p_long`, `p_short` 추출하여 반환

**매핑:**
```python
probs = F.softmax(logits, dim=-1)  # (batch, 3)
p_long = probs[:, LstmClassIndex.LONG]   # (batch,)
p_short = probs[:, LstmClassIndex.SHORT]  # (batch,)
```

### 5. 전략 로직 변경 (STEP 3, STEP 4)

**파일:**
- `src/strategies/ml_lstm_attn.py`
- `src/backtest/engine.py`

**변경 내용:**
- `ml_lstm_attn_strategy_enhanced()`: 3-class `direction_class = argmax([p_flat, p_long, p_short])` 사용
- 진입 조건:
  - LONG: `direction_class == LONG` AND `p_long >= long_threshold`
  - SHORT: `direction_class == SHORT` AND `p_short >= short_threshold`
  - FLAT: 위 조건에 해당하지 않으면 포지션 없음

**Backtest 엔진:**
- `run_backtest_with_ml()`: `strategy_name == "ml_lstm_attn"`일 때 3-class direction_class 사용
- 다른 전략(예: `ml_xgb`)은 기존 로직 유지

### 6. Threshold Optimizer 호환성 (STEP 4)

**파일:**
- `src/optimization/threshold_optimizer.py`
- `src/optimization/ml_proba_cache.py`

**변경 내용:**
- Threshold optimizer는 기존 인터페이스(`proba_long`, `proba_short`) 유지
- 3-class softmax에서 추출한 `p_long`, `p_short`를 그대로 사용
- 하위 호환성 유지: 기존 CLI 및 설정 파일 그대로 동작

## 주요 파일 목록

### 신규 생성
- `src/dl/data/labels.py`: 3-class 레이블 생성 유틸리티
- `tests/dl/test_labels.py`: 레이블 생성 테스트
- `docs/dl_lstm_3class_notes.md`: 이 문서

### 수정된 파일
1. **레이블링 및 데이터셋:**
   - `src/dl/train/train_lstm_attn.py`: `create_sequences()` 함수, `TimeSeriesDataset` 클래스

2. **모델 정의:**
   - `src/dl/models/lstm_attn.py`: 출력 레이어를 3-class로 변경

3. **추론 인터페이스:**
   - `src/dl/lstm_attn_model.py`: `predict_proba_latest()`, `predict_proba_batch()`

4. **전략 및 Backtest:**
   - `src/strategies/ml_lstm_attn.py`: `ml_lstm_attn_strategy_enhanced()`
   - `src/backtest/engine.py`: `run_backtest_with_ml()` 내 3-class 로직

5. **학습 파이프라인:**
   - `src/dl/train/train_lstm_attn.py`: Loss 함수, metrics, validation 로직

## 설정 파라미터

### 레이블 생성 (config.py 또는 .env)
- `LSTM_LABEL_POS_THRESHOLD`: LONG 클래스 기준 (기본값: 0.0015)
- `LSTM_LABEL_NEG_THRESHOLD`: SHORT 클래스 기준 (기본값: -0.0015)
- `LSTM_RETURN_HORIZON`: 미래 수익률 계산 horizon (기본값: 5)

### 모델 하이퍼파라미터
- 출력 차원: 3 (고정, 코드에서 자동 설정)
- Loss: `CrossEntropyLoss` (고정, 코드에서 자동 설정)
- Class weights: 자동 계산 (역빈도 가중치)

## 사용 방법

### 1. 모델 학습

```bash
python -m src.dl.train.train_lstm_attn
```

학습 시 자동으로 3-class 레이블을 생성하고, CrossEntropyLoss로 학습합니다.

### 2. Proba Cache 생성

```bash
python -m src.optimization.ml_proba_cache \
    --strategy ml_lstm_attn \
    --symbol BTCUSDT \
    --timeframe 5m \
    --start-date 2021-01-01
```

3-class softmax에서 `p_long`, `p_short`를 추출하여 캐시에 저장합니다.

### 3. Threshold 최적화

```bash
python -m src.optimization.optimize_ml_threshold \
    --strategy ml_lstm_attn \
    --symbol BTCUSDT \
    --timeframe 5m \
    --start-date 2021-01-01 \
    --strategy-mode long_only
```

기존과 동일한 인터페이스로 동작하며, 내부적으로 3-class 정보를 활용합니다.

### 4. Backtest 실행

```bash
python -m src.backtest.run_ml_xgb_backtest \
    --strategy ml_lstm_attn \
    --symbol BTCUSDT \
    --timeframe 5m
```

3-class direction_class를 사용하여 신호를 생성합니다.

## 하위 호환성

- **Threshold Optimizer**: 기존 인터페이스(`proba_long`, `proba_short`) 유지
- **Backtest 엔진**: 기존 파라미터 시그니처 유지
- **Proba Cache**: 기존 파일 형식 유지 (컬럼: `proba_long`, `proba_short`)
- **전략 함수**: 기존 반환 형태 유지

## 테스트 결과

### 레이블 분포 예시 (BTCUSDT 5m, horizon=5, pos_threshold=0.0015, neg_threshold=-0.0015)
- FLAT: ~60-70%
- LONG: ~15-20%
- SHORT: ~15-20%

### 모델 출력 예시
- `p_flat`: 0.0 ~ 1.0
- `p_long`: 0.0 ~ 1.0
- `p_short`: 0.0 ~ 1.0
- `p_flat + p_long + p_short = 1.0` (softmax 제약)

## 주의사항

1. **기존 모델 파일**: 이진 분류로 학습된 기존 모델은 3-class 모델과 호환되지 않습니다. 새로 학습해야 합니다.

2. **레이블 불균형**: FLAT 클래스가 많을 수 있으므로, class weights가 자동으로 적용됩니다.

3. **Threshold 튜닝**: 3-class에서는 `p_long`와 `p_short`가 독립적이므로, threshold optimizer가 더 효과적으로 동작할 수 있습니다.

## 향후 개선 사항

1. **FLAT 클래스 활용**: 현재는 FLAT일 때 포지션을 열지 않지만, 향후 FLAT을 활용한 전략도 고려 가능
2. **Per-class Metrics**: Precision/Recall을 class별로 계산하여 더 상세한 평가
3. **Ensemble**: 3-class 정보를 활용한 앙상블 전략

## 변경 일자

- 2024-12-XX: 초기 리팩터링 완료

## 테스트 체크리스트

### ✅ 완료된 작업

1. **레이블링 시스템**
   - [x] 3-class 레이블 생성 함수 구현 (`create_3class_labels`)
   - [x] `create_sequences` 함수 수정 (3-class 레이블 생성)
   - [x] 레이블 분포 로깅 개선 (FLAT/LONG/SHORT)
   - [x] Unit test 추가 (`tests/dl/test_labels.py`)

2. **모델 구조**
   - [x] 출력 레이어를 Dense(3)로 변경
   - [x] Forward pass에서 logits (batch, 3) 반환
   - [x] LstmClassIndex enum 정의

3. **학습 파이프라인**
   - [x] Loss 함수를 CrossEntropyLoss로 변경
   - [x] Class weights 자동 계산 및 적용
   - [x] Dataset 클래스가 3-class 레이블 지원
   - [x] Validation metrics를 3-class에 맞게 수정
   - [x] evaluate_split 함수를 3-class 지원하도록 수정

4. **추론 인터페이스**
   - [x] `predict_proba_latest()`: 3-class softmax에서 p_long 추출
   - [x] `predict_proba_batch()`: 3-class softmax에서 p_long, p_short 추출
   - [x] 하위 호환성 유지 (기존 인터페이스 그대로)

5. **전략 및 Backtest**
   - [x] `ml_lstm_attn_strategy_enhanced()`: direction_class = argmax 사용
   - [x] Backtest 엔진에서 3-class direction_class 활용
   - [x] Strategy mode (long_only/short_only/both) 지원

6. **문서화**
   - [x] 변경 사항 요약 문서 작성
   - [x] 코드 주석 추가
   - [x] Config 설명 업데이트

### ⚠️ 주의사항

1. **기존 모델 파일**: 이진 분류로 학습된 기존 모델은 사용할 수 없습니다. 새로 학습해야 합니다.

2. **첫 학습**: 3-class 모델을 처음 학습할 때는 충분한 데이터와 epoch가 필요합니다.

3. **레이블 불균형**: FLAT 클래스가 많을 수 있으므로, class weights가 자동으로 적용됩니다.

## 다음 단계

1. **모델 학습**: 새로운 3-class 모델 학습
   ```bash
   python -m src.dl.train.train_lstm_attn
   ```

2. **Proba Cache 생성**: 학습된 모델로 proba cache 생성
   ```bash
   python -m src.optimization.ml_proba_cache --strategy ml_lstm_attn --symbol BTCUSDT --timeframe 5m
   ```

3. **Threshold 최적화**: 3-class 모델에 맞는 threshold 찾기
   ```bash
   python -m src.optimization.optimize_ml_threshold --strategy ml_lstm_attn --symbol BTCUSDT --timeframe 5m --strategy-mode long_only
   ```

4. **Backtest 실행**: fee=0으로 간단한 sanity check
   ```bash
   python -m src.backtest.run_ml_xgb_backtest --strategy ml_lstm_attn --symbol BTCUSDT --timeframe 5m --commission-rate 0 --slippage-rate 0
   ```

