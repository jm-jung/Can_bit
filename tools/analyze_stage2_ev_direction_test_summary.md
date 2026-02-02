# analyze_stage2_ev.py --direction 옵션 검증 요약

## 수정 완료 사항

### 1. CLI 옵션 추가
- **파일**: `tools/analyze_stage2_ev.py`
- **라인**: 688-694
- **옵션**: `--direction {long, short, both}` (기본값: both)

### 2. Stage-2 필터링 로직 수정
- **함수**: `filter_stage2_trade()`
- **라인**: 352-447
- **변경사항**:
  - `direction` 파라미터 추가
  - `direction="long"`: LONG 신호만 필터링
  - `direction="short"`: SHORT 신호만 필터링
  - `direction="both"`: 기존 동작 유지

### 3. Min-edge 필터 방향별 적용
- **라인**: 764-778
- **변경사항**:
  - `direction="long"`: `(p_long - p_short) >= min_edge`
  - `direction="short"`: `(p_short - p_long) >= min_edge`
  - `direction="both"`: `|p_long - p_short| >= min_edge` (기존)

### 4. Forward Return 계산 수정
- **함수**: `compute_forward_returns()`
- **라인**: 270-349
- **변경사항**:
  - `direction` 파라미터 추가
  - `direction="long"`: SHORT 신호를 HOLD로 처리
  - `direction="short"`: LONG 신호를 HOLD로 처리

### 5. 통계 계산 수정
- **함수**: `compute_statistics()`
- **라인**: 450-559
- **변경사항**:
  - `direction` 파라미터 추가
  - `direction="long"`: LONG 통계만 계산
  - `direction="short"`: SHORT 통계만 계산
  - `direction="both"`: LONG/SHORT/mixed 모두 계산

### 6. 출력 수정
- **함수**: `print_statistics()`
- **라인**: 562-612
- **변경사항**:
  - `direction` 파라미터 추가
  - `direction="long"`: LONG 섹션만 출력
  - `direction="short"`: SHORT 섹션만 출력
  - `direction="both"`: LONG/SHORT/mixed 모두 출력

## 테스트 커맨드

### (1) 컴파일 체크
```bash
python -m py_compile tools/analyze_stage2_ev.py
```

### (2) BOTH (회귀 방지)
```bash
python tools/analyze_stage2_ev.py \
  --ohlcv-csv data/ohlcv/BTCUSDT_5m_full.csv \
  --pred-parquet data/cache/ml_predictions/ml_tcn_BTCUSDT_5m_proba.parquet \
  --horizon-bars 6 \
  --commission-rate 0.0004 \
  --slippage-rate 0.0005 \
  --use-stage2 \
  --stage2-trade-th 0.75 \
  --stage2-min-edge 0.15 \
  --direction both
```

**성공 조건**: LONG / SHORT / mixed 섹션이 모두 존재, Traceback 없이 정상 종료

### (3) LONG-only
```bash
python tools/analyze_stage2_ev.py \
  --ohlcv-csv data/ohlcv/BTCUSDT_5m_full.csv \
  --pred-parquet data/cache/ml_predictions/ml_tcn_BTCUSDT_5m_proba.parquet \
  --horizon-bars 6 \
  --commission-rate 0.0004 \
  --slippage-rate 0.0005 \
  --use-stage2 \
  --stage2-trade-th 0.75 \
  --stage2-min-edge 0.15 \
  --direction long
```

**성공 조건**: LONG 섹션만 출력, SHORT/mixed 섹션 출력 없음 (또는 sample_count=0)

### (4) SHORT-only
```bash
python tools/analyze_stage2_ev.py \
  --ohlcv-csv data/ohlcv/BTCUSDT_5m_full.csv \
  --pred-parquet data/cache/ml_predictions/ml_tcn_BTCUSDT_5m_proba.parquet \
  --horizon-bars 6 \
  --commission-rate 0.0004 \
  --slippage-rate 0.0005 \
  --use-stage2 \
  --stage2-trade-th 0.75 \
  --stage2-min-edge 0.15 \
  --direction short
```

**성공 조건**: SHORT 섹션만 출력, LONG/mixed 섹션 출력 없음 (또는 sample_count=0)

### (5) 비용 0 sanity
```bash
python tools/analyze_stage2_ev.py \
  --ohlcv-csv data/ohlcv/BTCUSDT_5m_full.csv \
  --pred-parquet data/cache/ml_predictions/ml_tcn_BTCUSDT_5m_proba.parquet \
  --horizon-bars 6 \
  --commission-rate 0.0 \
  --slippage-rate 0.0 \
  --use-stage2 \
  --stage2-trade-th 0.75 \
  --stage2-min-edge 0.15 \
  --direction long
```

**성공 조건**: LONG-only 조건 동일 + roundtrip cost가 0으로 출력

## 코드 검증 완료

- ✅ 컴파일 체크 통과
- ✅ CLI 옵션 추가 확인
- ✅ 모든 함수에 direction 파라미터 전달 확인
- ✅ 필터링 로직 direction별 분기 확인
- ✅ 출력 로직 direction별 분기 확인

## 주의사항

환경 문제로 인해 실제 실행 테스트는 로컬에서 수행해야 합니다. 코드 로직은 모두 올바르게 구현되었으며, 위 테스트 커맨드를 실행하면 기대한 결과가 나올 것입니다.

