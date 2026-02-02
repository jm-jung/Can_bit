# 백테스트 Direction 필터 실행 명령어 모음

## 빠른 참조: 복사-붙여넣기용 명령어

### 1. 가상환경 활성화 & 도움말 확인

```bash
source .venv/bin/activate
python -m src.backtest.run_ml_xgb_backtest --help | grep -E "(direction|--direction)" -A 2
```

### 2. BOTH 모드 (기본, 회귀 방지)

```bash
source .venv/bin/activate

python -m src.backtest.run_ml_xgb_backtest \
    --strategy ml_tcn \
    --symbol BTCUSDT \
    --timeframe 5m \
    --direction both \
    --start-date 2025-01-01 \
    --end-date 2025-01-10 \
    --use-optimized-threshold \
    --signal-confirmation-bars 1 \
    --no-save
```

**성공 조건**: LONG / SHORT 거래가 모두 발생, Traceback 없이 정상 종료

### 3. LONG-only 모드

```bash
source .venv/bin/activate

python -m src.backtest.run_ml_xgb_backtest \
    --strategy ml_tcn \
    --symbol BTCUSDT \
    --timeframe 5m \
    --direction long \
    --start-date 2025-01-01 \
    --end-date 2025-01-10 \
    --use-optimized-threshold \
    --signal-confirmation-bars 1 \
    --no-save
```

**성공 조건**:
- LONG 거래만 발생 (SHORT 거래 = 0)
- 로그에 "Direction filter: LONG-only mode active" 출력
- 리포트에 `"direction": "long"` 포함
- Traceback 없이 정상 종료

### 4. SHORT-only 모드

```bash
source .venv/bin/activate

python -m src.backtest.run_ml_xgb_backtest \
    --strategy ml_tcn \
    --symbol BTCUSDT \
    --timeframe 5m \
    --direction short \
    --start-date 2025-01-01 \
    --end-date 2025-01-10 \
    --use-optimized-threshold \
    --signal-confirmation-bars 1 \
    --no-save
```

**성공 조건**:
- SHORT 거래만 발생 (LONG 거래 = 0)
- 로그에 "Direction filter: SHORT-only mode active" 출력
- 리포트에 `"direction": "short"` 포함
- Traceback 없이 정상 종료

### 5. Stage-2와 함께 사용 (LONG-only)

```bash
source .venv/bin/activate

python -m src.backtest.run_ml_xgb_backtest \
    --strategy ml_tcn \
    --symbol BTCUSDT \
    --timeframe 5m \
    --direction long \
    --start-date 2025-01-01 \
    --end-date 2025-01-10 \
    --use-optimized-threshold \
    --signal-confirmation-bars 1 \
    --use-stage2 \
    --stage2-trade-th 0.75 \
    --stage2-min-edge 0.15 \
    --no-save
```

### 6. 회귀 테스트 (ml_xgb)

```bash
source .venv/bin/activate

python -m src.backtest.run_ml_xgb_backtest \
    --strategy ml_xgb \
    --symbol BTCUSDT \
    --timeframe 5m \
    --direction both \
    --start-date 2025-01-01 \
    --end-date 2025-01-10 \
    --no-save
```

### 7. 회귀 테스트 (ml_lstm_attn)

```bash
source .venv/bin/activate

python -m src.backtest.run_ml_xgb_backtest \
    --strategy ml_lstm_attn \
    --symbol BTCUSDT \
    --timeframe 5m \
    --direction both \
    --start-date 2025-01-01 \
    --end-date 2025-01-10 \
    --use-optimized-threshold \
    --signal-confirmation-bars 1 \
    --no-save
```

## 전체 워크플로우 (한 번에 실행)

```bash
#!/bin/bash
# 백테스트 Direction 필터 검증 전체 워크플로우

source .venv/bin/activate

echo "=== Step 1: BOTH 모드 (회귀 방지) ==="
python -m src.backtest.run_ml_xgb_backtest \
    --strategy ml_tcn \
    --symbol BTCUSDT \
    --timeframe 5m \
    --direction both \
    --start-date 2025-01-01 \
    --end-date 2025-01-10 \
    --use-optimized-threshold \
    --signal-confirmation-bars 1 \
    --no-save

echo "=== Step 2: LONG-only 모드 ==="
python -m src.backtest.run_ml_xgb_backtest \
    --strategy ml_tcn \
    --symbol BTCUSDT \
    --timeframe 5m \
    --direction long \
    --start-date 2025-01-01 \
    --end-date 2025-01-10 \
    --use-optimized-threshold \
    --signal-confirmation-bars 1 \
    --no-save

echo "=== Step 3: SHORT-only 모드 ==="
python -m src.backtest.run_ml_xgb_backtest \
    --strategy ml_tcn \
    --symbol BTCUSDT \
    --timeframe 5m \
    --direction short \
    --start-date 2025-01-01 \
    --end-date 2025-01-10 \
    --use-optimized-threshold \
    --signal-confirmation-bars 1 \
    --no-save

echo "=== 완료 ==="
```

## 참고사항

- 모든 명령어는 프로젝트 루트 디렉토리(`/Users/jeongminjun/Projects/Can_bit`)에서 실행해야 합니다.
- 가상환경 활성화(`source .venv/bin/activate`)는 각 세션마다 필요합니다.
- `--no-save` 옵션을 사용하면 리포트 파일을 생성하지 않습니다 (스모크 테스트용).
- 리포트를 저장하려면 `--no-save` 옵션을 제거하세요.

