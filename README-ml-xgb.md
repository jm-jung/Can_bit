# XGBoost ML Strategy - 사용 가이드

## 개요

이 문서는 XGBoost 기반 ML 전략의 학습, 튜닝, 백테스트, 최적화 방법을 설명합니다.

## 라벨 정의

**위치**: `src/ml/features.py`의 `build_ml_dataset()` 함수

**정의**:
- 미래 수익률 계산: `r = (price_{t+horizon} / price_t - 1)`
- 라벨: `y = 1` if `r > 0` (상승), `y = 0` otherwise (하락 또는 동일)
- Horizon: 기본값 5 (1분봉 기준 5분 후 예측)
- TP/SL: 라벨 생성 단계에서는 없음. 백테스트에서 threshold 기반 진입/청산

**특징**:
- 1분봉 특성상 짧은 horizon(5분)은 노이즈에 민감할 수 있음
- 긴 horizon은 중간 변동을 놓칠 수 있음
- 현재 설정(horizon=5)은 실용적인 균형점

## 1. XGB 학습

### 기본 학습 (튜닝 없이)

```bash
python -m src.ml.train_xgb
```

또는 옵션 지정:

```bash
python -m src.ml.train_xgb --horizon 5 --use-events
```

### 하이퍼파라미터 튜닝 + 학습

```bash
python -m src.ml.train_xgb --tune-xgb --cv-folds 3 --max-trials 30 --metric roc_auc
```

**옵션 설명**:
- `--tune-xgb`: 하이퍼파라미터 튜닝 활성화
- `--cv-folds 3`: Time-series CV 폴드 수 (기본: 3)
- `--max-trials 30`: 최대 하이퍼파라미터 조합 수 (기본: 30)
- `--metric roc_auc`: 최적화할 메트릭 (`roc_auc`, `logloss`, `accuracy`)
- `--horizon 5`: 예측 horizon (기본: 5)
- `--use-events`: 이벤트 피처 사용 (기본: settings.EVENTS_ENABLED)
- `--no-events`: 이벤트 피처 비활성화

**튜닝 과정**:
1. 전체 데이터를 시간순으로 정렬
2. Roll-forward 방식으로 CV 폴드 생성 (예: Fold1: train=0~200일, valid=201~230일)
3. 각 하이퍼파라미터 조합에 대해 모든 폴드에서 평가
4. 평균 메트릭이 높고 표준편차가 낮은 조합 선택
5. 최종 모델을 전체 데이터로 학습

**현재 기본 하이퍼파라미터**:
- `max_depth`: 4
- `n_estimators`: 300
- `learning_rate`: 0.05
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `min_child_weight`: 1
- `gamma`: 0.0

## 2. Threshold 최적화

```bash
python -m src.optimization.optimize_ml_threshold --strategy ml_xgb --symbol BTCUSDT --timeframe 1m --metric sharpe
```

**옵션 설명**:
- `--strategy ml_xgb`: 전략 이름
- `--symbol BTCUSDT`: 심볼
- `--timeframe 1m`: 타임프레임
- `--metric sharpe`: 최적화 메트릭 (`sharpe` 또는 `total_return`)

**결과**:
- `data/thresholds/ml_xgb_BTCUSDT_1m.json`에 저장
- 오버피팅 인식 최적화: in-sample/out-of-sample 분할 및 패널티 적용

## 3. 백테스트

### 디버그 백테스트 (짧은 구간, 8일 정도)

```bash
python test_ml_backtest_debug.py
```

### 긴 구간 백테스트

```bash
python -m src.backtest.run_ml_xgb_backtest \
    --strategy ml_xgb \
    --symbol BTCUSDT \
    --timeframe 1m \
    --start-date 2024-01-01 \
    --end-date 2024-06-30 \
    --use-optimized-threshold
```

**옵션 설명**:
- `--strategy ml_xgb`: 전략 이름
- `--symbol BTCUSDT`: 심볼
- `--timeframe 1m`: 타임프레임
- `--start-date YYYY-MM-DD`: 시작 날짜 (선택)
- `--end-date YYYY-MM-DD`: 종료 날짜 (선택)
- `--use-optimized-threshold`: 최적화된 threshold 사용
- `--long-threshold 0.70`: Long threshold 직접 지정 (선택)
- `--short-threshold 0.55`: Short threshold 직접 지정 (선택)
- `--no-save`: 리포트 파일 저장 안 함

**결과**:
- 콘솔에 요약 출력 (Sharpe, WinRate, Return, MDD, trades 등)
- `data/backtest_reports/ml_xgb_BTCUSDT_1m_YYYYMMDD_HHMMSS.json`에 상세 리포트 저장

## 4. 수수료 및 슬리피지

**설정 위치**: `src/core/config.py`

**기본값**:
- `COMMISSION_RATE`: 0.0004 (0.04% = 4 bps per side)
- `SLIPPAGE_RATE`: 0.0005 (0.05% = 5 bps per side)

**적용 방식**:
- 진입: `effective_entry = entry_price * (1 + commission_rate + slippage_rate)` (LONG)
- 청산: `effective_exit = exit_price * (1 - commission_rate - slippage_rate)` (LONG)
- SHORT의 경우 반대 방향으로 적용

**백테스트 리포트에 포함**:
- 모든 리포트에 `commission_rate`와 `slippage_rate`가 명시됨
- 결과 비교 시 어떤 가정 하에 나온 결과인지 명확히 알 수 있음

## 5. 피처 일관성

**학습 시 FEATURE_COLS**:
- `src/ml/train_xgb.py`에서 `build_ml_dataset()` 호출
- `build_feature_frame(use_events=settings.EVENTS_ENABLED)` 사용
- 로그에 전체 FEATURE_COLS 목록 출력

**예측 시 FEATURE_COLS**:
- `src/ml/xgb_model.py`의 `_extract_features()`에서 동일한 함수 사용
- 모델의 `feature_names`와 정렬하여 일관성 보장

**이벤트 피처**:
- `settings.EVENTS_ENABLED=True`일 때 자동 포함
- 접두사 `event_`로 식별 가능
- 총 14개 이벤트 피처 (기본 집계 6개 + 카테고리별 카운트/비율 12개)

## 6. 성능 개선 체크리스트

### 과적합 방지
- [ ] Train/Valid/Test 성능 차이 확인 (큰 차이는 과적합 신호)
- [ ] Time-series CV로 하이퍼파라미터 튜닝
- [ ] Feature importance 확인 (일부 피처에 과도하게 의존하지 않는지)

### 안정성 확인
- [ ] 여러 기간에 걸쳐 백테스트 실행
- [ ] In-sample/Out-of-sample 성능 차이 확인
- [ ] Threshold 최적화 시 오버피팅 패널티 적용

### 실전 준비
- [ ] 수수료/슬리피지 반영 확인
- [ ] 롱/숏 각각의 성능 분석
- [ ] 최대 연속 손실 확인 (리스크 관리)

## 7. 파일 구조

```
src/
├── ml/
│   ├── train_xgb.py          # 학습 스크립트 (튜닝 옵션 포함)
│   ├── xgb_tuning.py          # Time-series CV 튜닝 모듈
│   ├── xgb_model.py           # 모델 래퍼 (예측)
│   ├── features.py            # 피처 생성 (라벨 정의 포함)
│   └── evaluation.py          # 평가 함수
├── backtest/
│   ├── engine.py              # 백테스트 엔진 (수수료/슬리피지 적용)
│   ├── backtest_report.py     # 리포트 생성/저장
│   └── run_ml_xgb_backtest.py # 긴 구간 백테스트 엔트리 포인트
└── optimization/
    ├── threshold_optimizer.py # Threshold 최적화 (오버피팅 인식)
    └── optimize_ml_threshold.py # CLI 엔트리 포인트
```

## 8. 예시 워크플로우

### 전체 파이프라인 실행

```bash
# 1. 하이퍼파라미터 튜닝 + 학습
python -m src.ml.train_xgb --tune-xgb --cv-folds 3 --max-trials 30

# 2. Threshold 최적화
python -m src.optimization.optimize_ml_threshold --strategy ml_xgb --symbol BTCUSDT --timeframe 1m --metric sharpe

# 3. 긴 구간 백테스트
python -m src.backtest.run_ml_xgb_backtest --strategy ml_xgb --symbol BTCUSDT --timeframe 1m --start-date 2024-01-01 --end-date 2024-06-30 --use-optimized-threshold
```

### 빠른 테스트

```bash
# 학습만
python -m src.ml.train_xgb

# 디버그 백테스트
python test_ml_backtest_debug.py
```

## 9. 주의사항

1. **데이터 일관성**: 학습과 예측 시 동일한 피처 세트 사용 확인
2. **시간 순서**: Time-series 데이터이므로 시간 순서 유지 필수
3. **수수료/슬리피지**: 실전과 유사한 수준으로 설정 권장
4. **과적합**: Train 성능이 Valid/Test보다 크게 높으면 재검토 필요
5. **Horizon**: 1분봉 특성상 너무 짧거나 긴 horizon은 성능 저하 가능

## 10. 문제 해결

### 모델 로드 실패
- `models/xgb_model.pkl` 파일 존재 확인
- 피처 개수 불일치 시 재학습 필요

### Threshold 최적화 실패
- 충분한 데이터 확인 (최소 200개 샘플 권장)
- `data/thresholds/` 디렉토리 권한 확인

### 백테스트 결과 이상
- 수수료/슬리피지 설정 확인
- Threshold 값 확인 (최적화된 값 사용 권장)
- 데이터 기간 확인

