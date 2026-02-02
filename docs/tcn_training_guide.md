# TCN 모델 학습 및 사용 가이드

## 개요

TCN (Temporal Convolutional Network) 모델은 LSTM-Attn과 동일한 데이터셋/피처/라벨 파이프라인을 사용하여 3-class 분류 (FLAT/LONG/SHORT)를 수행합니다.

## 학습 CLI

### 기본 사용법

```bash
python -m src.dl.train.train_tcn \
  --symbol BTCUSDT \
  --timeframe 5m \
  --epochs 50 \
  --out-model models/tcn_v1.pt
```

### 주요 옵션

- `--symbol`: 거래 심볼 (기본값: settings에서 가져옴)
- `--timeframe`: 타임프레임 (기본값: settings에서 가져옴)
- `--feature-preset`: 피처 프리셋 (기본값: extended_safe)
- `--seq-len`: 시퀀스 길이 / window size (기본값: 60)
- `--horizon-bars`: 예측 horizon (기본값: settings에서 가져옴)
- `--epochs`: 학습 에포크 수 (기본값: 50)
- `--batch-size`: 배치 크기 (기본값: 64)
- `--lr`: 학습률 (기본값: 5e-4)
- `--device`: 디바이스 (auto/cpu/cuda, 기본값: auto)
- `--out-model`: 출력 모델 경로 (기본값: models/tcn_v1.pt)
- `--seed`: 랜덤 시드 (기본값: 42)

### 예시: 빠른 테스트 (1 epoch)

```bash
python -m src.dl.train.train_tcn \
  --symbol BTCUSDT \
  --timeframe 5m \
  --epochs 1 \
  --out-model models/tcn_smoke.pt
```

## 캐시 생성

학습된 모델로 예측 캐시를 생성합니다:

```bash
python -m src.optimization.ml_proba_cache \
  --strategy ml_tcn \
  --symbol BTCUSDT \
  --timeframe 5m \
  --force-rebuild
```

**참고**: `ml_proba_cache`는 기본적으로 `models/tcn_v1.pt` 경로를 참조합니다. 다른 경로의 모델을 사용하려면 `TCNSignalModel`의 기본 경로를 수정하거나, 추후 `--model-path` 옵션을 추가할 수 있습니다.

## 백테스트 실행

생성된 캐시를 사용하여 백테스트를 실행합니다:

```bash
python -m src.backtest.run_ml_xgb_backtest \
  --strategy ml_tcn \
  --symbol BTCUSDT \
  --timeframe 5m \
  --use-optimized-threshold \
  --signal-confirmation-bars 1
```

### Stage-2 옵션과 함께 실행

```bash
python -m src.backtest.run_ml_xgb_backtest \
  --strategy ml_tcn \
  --symbol BTCUSDT \
  --timeframe 5m \
  --use-optimized-threshold \
  --signal-confirmation-bars 1 \
  --use-stage2 \
  --stage2-trade-th 0.58 \
  --stage2-min-edge 0.05
```

## EV 분석

LSTM-Attn과 TCN의 EV를 비교합니다:

### LSTM-Attn

```bash
python tools/analyze_stage2_ev.py \
  --ohlcv-csv data/ohlcv/BTCUSDT_5m_full.csv \
  --pred-parquet data/cache/ml_predictions/ml_lstm_attn_BTCUSDT_5m_proba.parquet \
  --horizon-bars 6 \
  --commission-rate 0.0004 \
  --slippage-rate 0.0005 \
  --use-stage2 \
  --stage2-trade-th 0.58 \
  --stage2-min-edge 0.05
```

### TCN

```bash
python tools/analyze_stage2_ev.py \
  --ohlcv-csv data/ohlcv/BTCUSDT_5m_full.csv \
  --pred-parquet data/cache/ml_predictions/ml_tcn_BTCUSDT_5m_proba.parquet \
  --horizon-bars 6 \
  --commission-rate 0.0004 \
  --slippage-rate 0.0005 \
  --use-stage2 \
  --stage2-trade-th 0.58 \
  --stage2-min-edge 0.05
```

## 전체 파이프라인 (한 사이클)

```bash
# 1. 학습
python -m src.dl.train.train_tcn \
  --symbol BTCUSDT \
  --timeframe 5m \
  --epochs 50 \
  --out-model models/tcn_v1.pt

# 2. 캐시 생성
python -m src.optimization.ml_proba_cache \
  --strategy ml_tcn \
  --symbol BTCUSDT \
  --timeframe 5m \
  --force-rebuild

# 3. 백테스트
python -m src.backtest.run_ml_xgb_backtest \
  --strategy ml_tcn \
  --symbol BTCUSDT \
  --timeframe 5m \
  --use-optimized-threshold \
  --signal-confirmation-bars 1

# 4. EV 분석
python tools/analyze_stage2_ev.py \
  --ohlcv-csv data/ohlcv/BTCUSDT_5m_full.csv \
  --pred-parquet data/cache/ml_predictions/ml_tcn_BTCUSDT_5m_proba.parquet \
  --horizon-bars 6 \
  --commission-rate 0.0004 \
  --slippage-rate 0.0005 \
  --use-stage2 \
  --stage2-trade-th 0.58 \
  --stage2-min-edge 0.05
```

## 모델 아키텍처

- **입력**: 시계열 피처 시퀀스 (seq_len, feature_dim)
- **아키텍처**: 1D dilated causal convolution
  - 4개의 Temporal Block 레이어
  - 각 레이어의 채널 수: [64, 64, 64, 64]
  - 커널 크기: 3
  - Dilation: 1, 2, 4, 8 (지수 증가)
- **출력**: 3-class softmax 확률 (FLAT/LONG/SHORT)

## 데이터셋/피처 파이프라인

TCN은 LSTM-Attn과 **완전히 동일한** 데이터셋/피처/라벨 파이프라인을 사용합니다:

- 동일한 `create_sequences()` 함수 사용
- 동일한 피처 추출 (`build_feature_frame`)
- 동일한 라벨 생성 (3-class: FLAT/LONG/SHORT)
- 동일한 정규화 방식 (rolling window z-score)

이를 통해 LSTM-Attn과 TCN을 공정하게 비교할 수 있습니다.

