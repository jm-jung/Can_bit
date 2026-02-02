# OOS 기간 분리 실행 명령어 모음

## 빠른 참조: 복사-붙여넣기용 명령어

### 1. 기간별 CSV 생성

```bash
# 가상환경 활성화
source .venv/bin/activate

# 2023년 전체 데이터 추출
python tools/slice_ohlcv_csv.py \
    --in-csv data/ohlcv/BTCUSDT_5m_full.csv \
    --out-csv data/ohlcv/BTCUSDT_5m_2023.csv \
    --start 2023-01-01 \
    --end 2023-12-31

# 2024년 전체 데이터 추출
python tools/slice_ohlcv_csv.py \
    --in-csv data/ohlcv/BTCUSDT_5m_full.csv \
    --out-csv data/ohlcv/BTCUSDT_5m_2024.csv \
    --start 2024-01-01 \
    --end 2024-12-31

# 2025년 데이터 추출 (현재 OOS)
python tools/slice_ohlcv_csv.py \
    --in-csv data/ohlcv/BTCUSDT_5m_full.csv \
    --out-csv data/ohlcv/BTCUSDT_5m_2025.csv \
    --start 2025-01-01 \
    --end 2025-12-31
```

### 2. 생성된 CSV 검증

```bash
# 파일 존재 및 크기 확인
ls -lh data/ohlcv/BTCUSDT_5m_2023.csv
ls -lh data/ohlcv/BTCUSDT_5m_2024.csv
ls -lh data/ohlcv/BTCUSDT_5m_2025.csv

# 행 수 및 날짜 범위 확인
python -c "import pandas as pd; df = pd.read_csv('data/ohlcv/BTCUSDT_5m_2023.csv'); print(f'2023: Rows={len(df):,}, Range={df[\"timestamp\"].min()} ~ {df[\"timestamp\"].max()}')"
python -c "import pandas as pd; df = pd.read_csv('data/ohlcv/BTCUSDT_5m_2024.csv'); print(f'2024: Rows={len(df):,}, Range={df[\"timestamp\"].min()} ~ {df[\"timestamp\"].max()}')"
python -c "import pandas as pd; df = pd.read_csv('data/ohlcv/BTCUSDT_5m_2025.csv'); print(f'2025: Rows={len(df):,}, Range={df[\"timestamp\"].min()} ~ {df[\"timestamp\"].max()}')"
```

### 3. 각 기간별 EV 분석 (LONG-only)

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

### 4. SHORT-only 분석 (선택)

```bash
# 2023년 SHORT-only EV 분석
python tools/analyze_stage2_ev.py \
    --ohlcv-csv data/ohlcv/BTCUSDT_5m_2023.csv \
    --pred-parquet data/cache/ml_predictions/ml_tcn_BTCUSDT_5m_proba.parquet \
    --horizon-bars 6 \
    --commission-rate 0.0004 \
    --slippage-rate 0.0005 \
    --use-stage2 \
    --stage2-trade-th 0.75 \
    --stage2-min-edge 0.15 \
    --direction short

# 2024년 SHORT-only EV 분석
python tools/analyze_stage2_ev.py \
    --ohlcv-csv data/ohlcv/BTCUSDT_5m_2024.csv \
    --pred-parquet data/cache/ml_predictions/ml_tcn_BTCUSDT_5m_proba.parquet \
    --horizon-bars 6 \
    --commission-rate 0.0004 \
    --slippage-rate 0.0005 \
    --use-stage2 \
    --stage2-trade-th 0.75 \
    --stage2-min-edge 0.15 \
    --direction short

# 2025년 SHORT-only EV 분석 (OOS)
python tools/analyze_stage2_ev.py \
    --ohlcv-csv data/ohlcv/BTCUSDT_5m_2025.csv \
    --pred-parquet data/cache/ml_predictions/ml_tcn_BTCUSDT_5m_proba.parquet \
    --horizon-bars 6 \
    --commission-rate 0.0004 \
    --slippage-rate 0.0005 \
    --use-stage2 \
    --stage2-trade-th 0.75 \
    --stage2-min-edge 0.15 \
    --direction short
```

## 전체 워크플로우 (한 번에 실행)

```bash
#!/bin/bash
# OOS 기간 분리 및 EV 분석 전체 워크플로우

source .venv/bin/activate

echo "=== Step 1: 기간별 CSV 생성 ==="
python tools/slice_ohlcv_csv.py \
    --in-csv data/ohlcv/BTCUSDT_5m_full.csv \
    --out-csv data/ohlcv/BTCUSDT_5m_2023.csv \
    --start 2023-01-01 \
    --end 2023-12-31

python tools/slice_ohlcv_csv.py \
    --in-csv data/ohlcv/BTCUSDT_5m_full.csv \
    --out-csv data/ohlcv/BTCUSDT_5m_2024.csv \
    --start 2024-01-01 \
    --end 2024-12-31

python tools/slice_ohlcv_csv.py \
    --in-csv data/ohlcv/BTCUSDT_5m_full.csv \
    --out-csv data/ohlcv/BTCUSDT_5m_2025.csv \
    --start 2025-01-01 \
    --end 2025-12-31

echo "=== Step 2: 생성된 CSV 검증 ==="
ls -lh data/ohlcv/BTCUSDT_5m_202*.csv

echo "=== Step 3: 2023년 EV 분석 (LONG-only) ==="
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

echo "=== Step 4: 2024년 EV 분석 (LONG-only) ==="
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

echo "=== Step 5: 2025년 EV 분석 (LONG-only, OOS) ==="
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

echo "=== 완료 ==="
```

## 참고사항

- 모든 명령어는 프로젝트 루트 디렉토리(`/Users/jeongminjun/Projects/Can_bit`)에서 실행해야 합니다.
- 가상환경 활성화(`source .venv/bin/activate`)는 각 세션마다 필요합니다.
- 예측 캐시 파일(`ml_tcn_BTCUSDT_5m_proba.parquet`)은 전체 기간에 대한 예측을 포함해야 합니다.

