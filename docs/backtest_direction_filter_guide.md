# 백테스트 Direction 필터 가이드

이 가이드는 ML 백테스트에서 LONG-only 또는 SHORT-only 모드를 사용하는 방법을 설명합니다.

## 개요

`--direction` 옵션을 사용하여 백테스트에서 특정 방향의 거래만 실행할 수 있습니다. 이는 `tools/analyze_stage2_ev.py`의 `--direction` 옵션과 동일한 개념입니다.

## 옵션 설명

- `--direction {both,long,short}` (기본값: `both`)
  - `both`: LONG와 SHORT 모두 거래 가능 (기존 동작)
  - `long`: LONG-only 모드 (SHORT 신호는 HOLD로 처리)
  - `short`: SHORT-only 모드 (LONG 신호는 HOLD로 처리)

## 사용 예시

### 1. BOTH 모드 (기본, 기존 동작 유지)

```bash
source .venv/bin/activate

python -m src.backtest.run_ml_xgb_backtest \
    --strategy ml_tcn \
    --symbol BTCUSDT \
    --timeframe 5m \
    --direction both \
    --start-date 2025-01-01 \
    --end-date 2025-01-31 \
    --use-optimized-threshold \
    --signal-confirmation-bars 1
```

**예상 결과**: LONG와 SHORT 거래가 모두 발생

### 2. LONG-only 모드

```bash
source .venv/bin/activate

python -m src.backtest.run_ml_xgb_backtest \
    --strategy ml_tcn \
    --symbol BTCUSDT \
    --timeframe 5m \
    --direction long \
    --start-date 2025-01-01 \
    --end-date 2025-01-31 \
    --use-optimized-threshold \
    --signal-confirmation-bars 1
```

**예상 결과**:
- LONG 거래만 발생
- SHORT 신호는 HOLD로 처리되어 진입하지 않음
- 로그에 "Direction filter: LONG-only mode active" 출력
- 리포트에 `"direction": "long"` 포함

### 3. SHORT-only 모드

```bash
source .venv/bin/activate

python -m src.backtest.run_ml_xgb_backtest \
    --strategy ml_tcn \
    --symbol BTCUSDT \
    --timeframe 5m \
    --direction short \
    --start-date 2025-01-01 \
    --end-date 2025-01-31 \
    --use-optimized-threshold \
    --signal-confirmation-bars 1
```

**예상 결과**:
- SHORT 거래만 발생
- LONG 신호는 HOLD로 처리되어 진입하지 않음
- 로그에 "Direction filter: SHORT-only mode active" 출력
- 리포트에 `"direction": "short"` 포함

## Stage-2와 함께 사용

`--direction` 옵션은 Stage-2 옵션과 함께 사용할 수 있습니다:

```bash
python -m src.backtest.run_ml_xgb_backtest \
    --strategy ml_tcn \
    --symbol BTCUSDT \
    --timeframe 5m \
    --direction long \
    --start-date 2025-01-01 \
    --end-date 2025-01-31 \
    --use-optimized-threshold \
    --signal-confirmation-bars 1 \
    --use-stage2 \
    --stage2-trade-th 0.75 \
    --stage2-min-edge 0.15
```

## 동작 원리

1. **신호 생성 단계** (`generate_signals()`):
   - `long_only=True`이면 SHORT 신호를 생성하지 않음
   - `short_only=True`이면 LONG 신호를 생성하지 않음

2. **거래 실행 단계** (`execute_trades()`):
   - 안전장치로 `long_only=True`이면 SHORT 신호를 HOLD로 변환
   - 안전장치로 `short_only=True`이면 LONG 신호를 HOLD로 변환
   - `block_reasons["direction_filter"]`에 차단된 신호 수 기록

## 로그 확인

백테스트 실행 시 다음 로그를 확인할 수 있습니다:

```
[ML Backtest] Direction filter: long (long_only=True, short_only=False)
[LSTM-Attn][EXECUTION DEBUG] Direction filter: LONG-only mode active. SHORT entries blocked: 1234
```

## 리포트 확인

생성된 백테스트 리포트 JSON 파일에 `direction` 필드가 포함됩니다:

```json
{
  "strategy": "ml_tcn",
  "symbol": "BTCUSDT",
  "timeframe": "5m",
  "direction": "long",
  "metrics": {
    "total_trades": 456,
    "long_trades": 456,
    "short_trades": 0,
    ...
  }
}
```

## 주의사항

1. **기존 동작 유지**: `--direction both` (기본값)는 기존과 동일하게 동작합니다.
2. **회귀 방지**: `both` 모드에서 기존 결과가 변경되지 않습니다.
3. **다른 전략 호환**: `ml_xgb`, `ml_lstm_attn` 전략도 동일하게 `--direction` 옵션을 사용할 수 있습니다.

## EV 분석과의 연계

`tools/analyze_stage2_ev.py`의 `--direction` 옵션과 동일한 개념이므로, EV 분석 결과와 백테스트 결과를 일관되게 비교할 수 있습니다:

```bash
# EV 분석 (LONG-only)
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

# 백테스트 (LONG-only)
python -m src.backtest.run_ml_xgb_backtest \
    --strategy ml_tcn \
    --symbol BTCUSDT \
    --timeframe 5m \
    --direction long \
    --start-date 2025-01-01 \
    --end-date 2025-01-31 \
    --use-optimized-threshold \
    --signal-confirmation-bars 1 \
    --use-stage2 \
    --stage2-trade-th 0.75 \
    --stage2-min-edge 0.15
```

