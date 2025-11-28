# ML 백테스트 거래 생성 실패 문제 진단

## 문제 요약
Threshold optimizer 실행 시 모든 조합에서 `trades=0, return=0, metric=0`이 발생합니다.

## 진단 결과

### 1. 가능한 원인들

#### A. 모든 예측이 실패하고 있을 가능성
- `run_backtest_with_ml` 함수의 line 370-391에서 예외가 발생하면 `continue`로 넘어가며 해당 인덱스의 신호는 "HOLD"로 남습니다
- 모든 예측이 실패하면 모든 신호가 "HOLD"가 되고 거래가 생성되지 않습니다

#### B. 예측은 성공하지만 모든 proba_up이 threshold 범위 밖
- 예측은 성공하지만 모든 `proba_up` 값이 `long_threshold`보다 작고 `short_threshold`가 None이면 모든 신호가 "HOLD"가 됩니다
- 이 경우 신호는 생성되지만 거래가 생성되지 않습니다

#### C. 모델이 제대로 로드되지 않음
- `get_xgb_model()`이 None을 반환하거나 `model.is_loaded()`가 False를 반환하면 빈 결과를 반환합니다

#### D. 피처 추출/정렬 문제
- `build_feature_frame`이 반환하는 피처와 모델이 예상하는 피처 이름/순서가 맞지 않을 수 있습니다
- `predict_proba_latest`에서 피처 정렬이 실패할 수 있습니다

### 2. 추가된 디버깅 로그

다음 정보를 로깅하도록 수정했습니다:

1. **신호 생성 단계** (`src/backtest/engine.py` line 365-449):
   - 예측 성공/실패 횟수
   - 신호 분포 (LONG/SHORT/HOLD 개수)
   - proba_up 통계 (mean, min, max, std)
   - threshold 대비 proba 분포

2. **거래 실행 단계** (line 451-490):
   - 진입 시도 횟수
   - 청산 실행 횟수
   - 최종 거래 개수
   - 최종 포지션 상태

### 3. 다음 단계

1. **디버깅 스크립트 실행**:
   ```bash
   python test_ml_backtest_debug.py
   ```
   이 스크립트는 다음을 확인합니다:
   - 데이터 로딩 상태
   - 모델 로딩 상태
   - 예측 테스트
   - 백테스트 실행 및 결과

2. **로그 확인**:
   - `[ML Backtest]` 태그가 붙은 로그를 확인하여 어느 단계에서 문제가 발생하는지 확인
   - 특히 다음을 확인:
     - `successful_predictions` vs `prediction_errors`
     - `signal_counts` (LONG/SHORT/HOLD 개수)
     - `proba statistics` (값들이 의미있는 범위에 있는지)
     - `entries_attempted` vs `exits_executed`

3. **가능한 수정 사항**:
   - 예측 실패 시 더 자세한 에러 로깅
   - 피처 정렬 문제 해결
   - 모델 로딩 실패 시 명확한 에러 메시지

## 예상되는 근본 원인

가장 가능성 높은 원인:
1. **모든 예측이 실패**: 피처 추출 또는 모델 호출 시 예외 발생
2. **모든 proba_up이 threshold 범위 밖**: 모델이 항상 비슷한 확률을 예측하거나 threshold가 너무 높음/낮음

## 수정 사항

### 파일: `src/backtest/engine.py`

1. **신호 생성 단계에 상세 로깅 추가** (line 365-449):
   - 예측 성공/실패 통계
   - proba_up 통계 및 분포
   - 신호 생성 통계

2. **거래 실행 단계에 상세 로깅 추가** (line 451-490):
   - 진입/청산 시도 횟수 추적
   - 최종 상태 로깅

이제 실제 실행 시 로그를 통해 정확한 문제 지점을 파악할 수 있습니다.

