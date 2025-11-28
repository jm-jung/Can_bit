# 이벤트 피처 적용 여부 분석 결과

## [1] 이벤트 피처 정의 파일

### 경로:
- **`src/events/aggregator.py`**: 이벤트 집계 및 피처 생성 로직
- **`src/events/dataset.py`**: 이벤트 데이터셋 관리 및 피처 DataFrame 생성
- **`src/ml/features.py`**: 피처 프레임 빌드 (기술적 피처 + 이벤트 피처 통합)

### 이벤트 피처 목록:

**기본 집계 피처:**
- `event_count_total`: 총 이벤트 개수
- `event_sentiment_mean`: 평균 감정 점수 (-1.0 ~ 1.0)
- `event_sentiment_positive_mean`: 양수 감정 평균 (0.0 ~ 1.0)
- `event_sentiment_negative_mean`: 음수 감정 평균 (-1.0 ~ 0.0)
- `event_max_intensity`: 최대 강도 (0.0 ~ 1.0)
- `event_time_since_last_min`: 마지막 이벤트로부터 경과 시간 (분)

**카테고리별 카운트 피처 (6개):**
- `event_count_influencer`
- `event_count_institution`
- `event_count_macro_policy`
- `event_count_regulation`
- `event_count_geopolitical`
- `event_count_market_structure`

**카테고리별 비율 피처 (6개):**
- `event_share_influencer`
- `event_share_institution`
- `event_share_macro_policy`
- `event_share_regulation`
- `event_share_geopolitical`
- `event_share_market_structure`

**총 이벤트 피처 개수: 14개** (기본 6개 + 카테고리 카운트 6개 + 카테고리 비율 6개, 단 `event_count_total`은 기본에 포함)

### FEATURE_COLS 정의 위치:
- **`src/events/aggregator.py`**의 `_compute_window_features()` 함수에서 동적으로 생성
- **`src/ml/features.py`**의 `build_feature_frame()` 함수에서 기술적 피처와 병합
- 최종 피처 컬럼은 `build_feature_frame()` 반환값의 `.columns` 속성

---

## [2] XGB 학습에서 이벤트 피처 사용 여부 확인

### 학습 스크립트:
- **`src/ml/train_xgb.py`**

### 피처 준비 함수:
- **`build_ml_dataset()`** (from `src/ml/features.py`)
  - 내부에서 **`build_feature_frame()`** 호출
  - `use_events` 파라미터로 이벤트 피처 활성화 여부 제어
  - `settings.EVENTS_ENABLED` 기본값 사용

### 이벤트 피처 포함 여부:
✅ **포함됨** - `build_feature_frame()`에서 `use_events=True`일 때 `merge_price_and_event_features()` 호출

### 학습 시 FEATURE_COLS:
- **위치**: `src/ml/train_xgb.py`의 `train_xgb_model()` 함수
- **변수**: `X.columns` (line 160)
- **현재 로그**: 
  - Line 161: `logger.info(f"Feature count: {len(X.columns)}")`
  - Line 163-164: 이벤트 피처 개수만 샘플 출력

### 로그 패치 제안:

**파일**: `src/ml/train_xgb.py`
**위치**: Line 160-164 이후

```python
logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
logger.info(f"Feature count: {len(X.columns)}")
if use_events:
    event_cols = [c for c in X.columns if c.startswith("event_")]
    logger.info(f"Event features: {len(event_cols)} (sample: {event_cols[:3] if event_cols else []})")

# 🔥 추가할 로그 패치
logger.info("=" * 60)
logger.info("[XGB Train] Feature Columns")
logger.info("=" * 60)
logger.info(f"[XGB Train] Using {len(X.columns)} feature columns:")
logger.info(f"[XGB Train] FEATURE_COLS = {list(X.columns)}")
if use_events:
    event_cols = [c for c in X.columns if c.startswith("event_")]
    logger.info(f"[XGB Train] Event features ({len(event_cols)}): {event_cols}")
    basic_cols = [c for c in X.columns if not c.startswith("event_")]
    logger.info(f"[XGB Train] Basic features ({len(basic_cols)}): {basic_cols}")
logger.info("=" * 60)
```

---

## [3] LSTM-Attn 학습/예측에서 이벤트 피처 사용 여부 확인

### 관련 파일:
- **`src/dl/train/train_lstm_attn.py`**: 학습 스크립트
- **`src/dl/lstm_attn_model.py`**: 모델 래퍼 및 예측 로직

### 피처 준비 함수:
- **`create_sequences()`** (in `train_lstm_attn.py`, line 287)
  - 내부에서 **`build_ml_dataset()`** 호출 (line 367)
  - `use_events=settings.EVENTS_ENABLED` 사용
- **`_extract_features()`** (in `lstm_attn_model.py`, line 153)
  - 내부에서 **`build_feature_frame()`** 호출 (line 165)
  - `use_events=settings.EVENTS_ENABLED` 사용

### 이벤트 피처 포함 여부:
✅ **포함됨** - `build_ml_dataset()` 및 `build_feature_frame()` 모두 `settings.EVENTS_ENABLED` 사용

### 학습 시 FEATURE_COLS:
- **위치**: `src/dl/train/train_lstm_attn.py`의 `create_sequences()` 함수
- **변수**: `feature_cols = X_features.columns.tolist()` (line 395)
- **현재 로그**: 
  - Line 398-402: 기본 피처와 이벤트 피처 개수만 샘플 출력

### 예측 시 FEATURE_COLS:
- **위치**: `src/dl/lstm_attn_model.py`의 `_extract_features()` 함수
- **변수**: `self.feature_cols` (캐시됨)
- **현재 로그**: 
  - Line 130-131: 피처 컬럼 개수와 첫 10개만 출력

### 로그 패치 제안:

**파일 1**: `src/dl/train/train_lstm_attn.py`
**위치**: Line 395-402 이후

```python
feature_cols = X_features.columns.tolist()
feature_dim = len(feature_cols)

logger.info(f"Feature columns ({feature_dim} total):")
logger.info(f"  - Basic features: {[c for c in feature_cols if not c.startswith('event_')]}")
if settings.EVENTS_ENABLED:
    event_cols = [c for c in feature_cols if c.startswith("event_")]
    logger.info(f"  - Event features: {len(event_cols)} (sample: {event_cols[:3]})")

# 🔥 추가할 로그 패치
logger.info("=" * 60)
logger.info("[LSTM Train] Feature Columns")
logger.info("=" * 60)
logger.info(f"[LSTM] Using {len(feature_cols)} feature columns:")
logger.info(f"[LSTM] FEATURE_COLS = {feature_cols}")
if settings.EVENTS_ENABLED:
    event_cols = [c for c in feature_cols if c.startswith("event_")]
    logger.info(f"[LSTM] Event features ({len(event_cols)}): {event_cols}")
    basic_cols = [c for c in feature_cols if not c.startswith("event_")]
    logger.info(f"[LSTM] Basic features ({len(basic_cols)}): {basic_cols}")
logger.info("=" * 60)
```

**파일 2**: `src/dl/lstm_attn_model.py`
**위치**: Line 130-131 이후 (모델 로드 후)

```python
logger.info(f"Feature columns count: {len(self.feature_cols)}")
logger.info(f"Feature columns (first 10): {self.feature_cols[:10]}")

# 🔥 추가할 로그 패치
logger.info("=" * 60)
logger.info("[LSTM Inference] Feature Columns")
logger.info("=" * 60)
logger.info(f"[LSTM] Using {len(self.feature_cols)} feature columns:")
logger.info(f"[LSTM] FEATURE_COLS = {self.feature_cols}")
if settings.EVENTS_ENABLED:
    event_cols = [c for c in self.feature_cols if c.startswith("event_")]
    logger.info(f"[LSTM] Event features ({len(event_cols)}): {event_cols}")
    basic_cols = [c for c in self.feature_cols if not c.startswith("event_")]
    logger.info(f"[LSTM] Basic features ({len(basic_cols)}): {basic_cols}")
logger.info("=" * 60)
```

---

## [4] 백테스트 / threshold 최적화에서 동일 피처 사용 여부 확인

### 관련 파일:
- **`src/backtest/engine.py`**: ML 백테스트 엔진
- **`src/ml/xgb_model.py`**: XGBoost 모델 래퍼 (예측 시 피처 추출)
- **`src/dl/lstm_attn_model.py`**: LSTM 모델 래퍼 (예측 시 피처 추출)
- **`src/optimization/ml_proba_cache.py`**: 최적화용 예측 캐싱

### 백테스트에서 피처 추출:
- **XGBoost**: `src/ml/xgb_model.py`의 `_extract_features()` → `build_feature_frame(use_events=settings.EVENTS_ENABLED)`
- **LSTM**: `src/dl/lstm_attn_model.py`의 `_extract_features()` → `build_feature_frame(use_events=settings.EVENTS_ENABLED)`

### 피처 일관성 확인:

#### ✅ XGBoost:
- **학습**: `build_ml_dataset()` → `build_feature_frame(use_events=settings.EVENTS_ENABLED)`
- **백테스트**: `XGBSignalModel._extract_features()` → `build_feature_frame(use_events=settings.EVENTS_ENABLED)`
- **결론**: **일치함** - 동일한 함수와 파라미터 사용

#### ✅ LSTM-Attn:
- **학습**: `create_sequences()` → `build_ml_dataset(use_events=settings.EVENTS_ENABLED)`
- **백테스트**: `LSTMAttnSignalModel._extract_features()` → `build_feature_frame(use_events=settings.EVENTS_ENABLED)`
- **결론**: **일치함** - 동일한 함수와 파라미터 사용

#### ⚠️ 주의사항:
- XGBoost 모델은 학습 시점의 `feature_names`를 저장하고, 예측 시 `reindex()`로 정렬
- LSTM 모델은 `self.feature_cols`를 캐시하고, 예측 시 `reindex()`로 정렬
- **피처 순서 불일치 가능성**: 학습 시와 예측 시 컬럼 순서가 다를 수 있음 (하지만 `reindex()`로 보정)

### 로그 패치 제안:

**파일**: `src/ml/xgb_model.py`
**위치**: `predict_proba_latest()` 함수 내, line 99 이후 (모델 예측 전)

```python
# Reorder columns to match model's expected order
if model_feature_names:
    last_features = last_features.reindex(columns=model_feature_names, fill_value=0.0)

# 🔥 추가할 로그 패치
logger.debug(f"[XGB Inference] Model expects {len(model_feature_names)} features")
logger.debug(f"[XGB Inference] Provided features: {list(last_features.columns)}")
if settings.EVENTS_ENABLED:
    event_cols = [c for c in last_features.columns if c.startswith("event_")]
    logger.debug(f"[XGB Inference] Event features in input: {event_cols}")
```

---

## [5] 최종 요약 및 결론

### 1) 이벤트 피처 정의 파일:
- **경로**: 
  - `src/events/aggregator.py` (집계 로직)
  - `src/events/dataset.py` (데이터셋 관리)
  - `src/ml/features.py` (피처 통합)
- **이벤트 피처 목록**: 
  - 총 14개: `event_count_total`, `event_sentiment_mean`, `event_sentiment_positive_mean`, `event_sentiment_negative_mean`, `event_max_intensity`, `event_time_since_last_min`, `event_count_{category}` (6개), `event_share_{category}` (6개)

### 2) XGB:
- **학습 FEATURE_COLS**: `build_ml_dataset()` → `build_feature_frame(use_events=settings.EVENTS_ENABLED)` → `X.columns`
- **이벤트 피처 포함 여부**: ✅ **포함됨** (`settings.EVENTS_ENABLED=True`일 때)
- **백테스트 FEATURE_COLS**: `XGBSignalModel._extract_features()` → `build_feature_frame(use_events=settings.EVENTS_ENABLED)`
- **둘이 일치하는지 여부**: ✅ **일치함** - 동일한 함수와 파라미터 사용

### 3) LSTM-Attn:
- **학습 FEATURE_COLS**: `create_sequences()` → `build_ml_dataset(use_events=settings.EVENTS_ENABLED)` → `X_features.columns.tolist()`
- **이벤트 피처 포함 여부**: ✅ **포함됨** (`settings.EVENTS_ENABLED=True`일 때)
- **백테스트 FEATURE_COLS**: `LSTMAttnSignalModel._extract_features()` → `build_feature_frame(use_events=settings.EVENTS_ENABLED)` → `self.feature_cols`
- **둘이 일치하는지 여부**: ✅ **일치함** - 동일한 함수와 파라미터 사용

### 4) 종합 결론:
- **"이벤트 피처가 전체 파이프라인(XGB/LSTM/백테스트/최적화)에 일관되게 적용되는가?"**: ✅ **YES**
- **문제점**: 없음. 모든 경로에서 `settings.EVENTS_ENABLED`를 통해 일관되게 이벤트 피처를 사용/비사용할 수 있음.
- **개선 제안**: 
  1. FEATURE_COLS 로그 추가 (위의 패치 제안 참조)
  2. 피처 순서 일관성 검증 로그 추가 (학습 시와 예측 시 컬럼 순서 비교)

---

## 로그 패치 적용 방법

위에서 제안한 로그 패치를 적용하려면:

1. **XGB 학습**: `src/ml/train_xgb.py`의 line 160-164 이후에 로그 추가
2. **LSTM 학습**: `src/dl/train/train_lstm_attn.py`의 line 395-402 이후에 로그 추가
3. **LSTM 예측**: `src/dl/lstm_attn_model.py`의 line 130-131 이후에 로그 추가
4. **XGB 예측**: `src/ml/xgb_model.py`의 `predict_proba_latest()` 함수에 디버그 로그 추가 (선택사항)

이 로그들을 통해 학습/예측 시점에 실제 사용되는 FEATURE_COLS를 확인할 수 있습니다.

