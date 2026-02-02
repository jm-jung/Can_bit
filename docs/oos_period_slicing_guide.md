# OOS 기간 분리 가이드

이 가이드는 OHLCV CSV를 기간별로 분리하여 OOS(Out-of-Sample) 검증을 수행하는 방법을 설명합니다.

## 도구 개요

`tools/slice_ohlcv_csv.py`는 전체 OHLCV CSV를 지정된 기간으로 잘라서 새로운 CSV 파일로 저장하는 도구입니다.

## 사용 방법

### 기본 사용법

```bash
python tools/slice_ohlcv_csv.py \
    --in-csv data/ohlcv/BTCUSDT_5m_full.csv \
    --out-csv data/ohlcv/BTCUSDT_5m_2024.csv \
    --start 2024-01-01 \
    --end 2024-12-31
```

### 옵션 설명

- `--in-csv`: 입력 OHLCV CSV 파일 경로 (필수)
- `--out-csv`: 출력 CSV 파일 경로 (필수)
- `--start`: 시작 날짜 (YYYY-MM-DD 형식, 포함, 선택)
- `--end`: 종료 날짜 (YYYY-MM-DD 형식, 포함, 선택)

## 실행 예시

### 1. 2023년 전체 데이터 추출

```bash
source .venv/bin/activate

python tools/slice_ohlcv_csv.py \
    --in-csv data/ohlcv/BTCUSDT_5m_full.csv \
    --out-csv data/ohlcv/BTCUSDT_5m_2023.csv \
    --start 2023-01-01 \
    --end 2023-12-31
```

### 2. 2024년 전체 데이터 추출

```bash
source .venv/bin/activate

python tools/slice_ohlcv_csv.py \
    --in-csv data/ohlcv/BTCUSDT_5m_full.csv \
    --out-csv data/ohlcv/BTCUSDT_5m_2024.csv \
    --start 2024-01-01 \
    --end 2024-12-31
```

### 3. 2025년 데이터 추출 (현재 OOS)

```bash
source .venv/bin/activate

python tools/slice_ohlcv_csv.py \
    --in-csv data/ohlcv/BTCUSDT_5m_full.csv \
    --out-csv data/ohlcv/BTCUSDT_5m_2025.csv \
    --start 2025-01-01 \
    --end 2025-12-31
```

## OOS EV 분석 워크플로우

### 1단계: 기간별 CSV 생성

```bash
# 2023년
python tools/slice_ohlcv_csv.py \
    --in-csv data/ohlcv/BTCUSDT_5m_full.csv \
    --out-csv data/ohlcv/BTCUSDT_5m_2023.csv \
    --start 2023-01-01 \
    --end 2023-12-31

# 2024년
python tools/slice_ohlcv_csv.py \
    --in-csv data/ohlcv/BTCUSDT_5m_full.csv \
    --out-csv data/ohlcv/BTCUSDT_5m_2024.csv \
    --start 2024-01-01 \
    --end 2024-12-31

# 2025년 (OOS)
python tools/slice_ohlcv_csv.py \
    --in-csv data/ohlcv/BTCUSDT_5m_full.csv \
    --out-csv data/ohlcv/BTCUSDT_5m_2025.csv \
    --start 2025-01-01 \
    --end 2025-12-31
```

### 2단계: 각 기간별 EV 분석

```bash
# 2023년 EV 분석
python tools/analyze_stage2_ev.py \
    --ohlcv-csv data/ohlcv/BTCUSDT_5m_2023.csv \
    --pred-parquet data/cache/ml_predictions/ml_tcn_BTCUSDT_5m_proba.parquet \
    --horizon-bars 6 \
    --commission-rate 0.0004 \
    --slippage-rate 0.0005 \
    --use-stage2 \
    --stage2-trade-th 0.75 \
    --stage2-min-edge 0.15 \
    --direction long

# 2024년 EV 분석
python tools/analyze_stage2_ev.py \
    --ohlcv-csv data/ohlcv/BTCUSDT_5m_2024.csv \
    --pred-parquet data/cache/ml_predictions/ml_tcn_BTCUSDT_5m_proba.parquet \
    --horizon-bars 6 \
    --commission-rate 0.0004 \
    --slippage-rate 0.0005 \
    --use-stage2 \
    --stage2-trade-th 0.75 \
    --stage2-min-edge 0.15 \
    --direction long

# 2025년 EV 분석 (OOS)
python tools/analyze_stage2_ev.py \
    --ohlcv-csv data/ohlcv/BTCUSDT_5m_2025.csv \
    --pred-parquet data/cache/ml_predictions/ml_tcn_BTCUSDT_5m_proba.parquet \
    --horizon-bars 6 \
    --commission-rate 0.0004 \
    --slippage-rate 0.0005 \
    --use-stage2 \
    --stage2-trade-th 0.75 \
    --stage2-min-edge 0.15 \
    --direction long
```

## 주의사항

1. **예측 캐시 호환성**: `analyze_stage2_ev.py`에서 사용하는 예측 캐시(`--pred-parquet`)는 전체 기간에 대한 예측을 포함해야 합니다. 기간별로 필터링된 CSV를 사용하더라도 예측 캐시는 전체 기간 데이터를 포함해야 합니다.

2. **타임스탬프 정렬**: 생성된 CSV는 timestamp 기준으로 정렬되어 있습니다.

3. **날짜 범위**: `--start`와 `--end`는 모두 포함(inclusive)입니다. 즉, `--end 2024-12-31`은 2024년 12월 31일 23:59:59까지 포함합니다.

## 검증

생성된 CSV 파일을 확인하려면:

```bash
# 파일 존재 확인
ls -lh data/ohlcv/BTCUSDT_5m_2024.csv

# 행 수 확인
python -c "import pandas as pd; df = pd.read_csv('data/ohlcv/BTCUSDT_5m_2024.csv'); print(f'Rows: {len(df):,}'); print(f'Date range: {df[\"timestamp\"].min()} ~ {df[\"timestamp\"].max()}')"
```

