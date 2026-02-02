# StrategyGuard 실행 명령어 모음

## 빠른 참조: 복사-붙여넣기용 명령어

### 1. 가상환경 활성화 & 도움말 확인

```bash
source .venv/bin/activate
python -m src.backtest.run_ml_xgb_backtest --help | grep -i "strategy-guard" -A 3
```

### 2. Guard OFF (기존 동작 유지, 회귀 방지)

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
- 기존과 동일하게 동작
- 로그에 "[StrategyGuard]" 관련 메시지 없음
- Traceback 없이 정상 종료

### 3. Guard ON

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
    --use-strategy-guard \
    --no-save
```

**성공 조건**:
- 로그에 "[StrategyGuard] Enabled" 출력
- 로그에 "[STRATEGY GUARD] StrategyGuard Statistics" 섹션 출력
- ALLOW/BLOCK 상태가 로그에 출력
- `block_reasons`에 `strategy_guard` 카운트 포함 (BLOCK 발생 시)
- Traceback 없이 정상 종료

### 4. Guard ON + Stage-2 조합

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
    --use-strategy-guard \
    --no-save
```

### 5. 전체 워크플로우 (한 번에 실행)

```bash
#!/bin/bash
# StrategyGuard 검증 전체 워크플로우

source .venv/bin/activate

echo "=== Step 1: Guard OFF (회귀 방지) ==="
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

echo "=== Step 2: Guard ON ==="
python -m src.backtest.run_ml_xgb_backtest \
    --strategy ml_tcn \
    --symbol BTCUSDT \
    --timeframe 5m \
    --direction long \
    --start-date 2025-01-01 \
    --end-date 2025-01-10 \
    --use-optimized-threshold \
    --signal-confirmation-bars 1 \
    --use-strategy-guard \
    --no-save

echo "=== 완료 ==="
```

## 예상 로그 출력 (Guard ON)

```
[ML Backtest][LSTM-Attn][StrategyGuard] Enabled. Config: recent_trades_window=20, min_win_rate=0.4, min_avg_return=-0.01
...
[StrategyGuard] BLOCK decision: win_rate=0.350<0.400; avg_return=-0.0123<-0.0100 (recent_trades=20, win_rate=0.350, avg_return=-0.0123)
...
============================================================
[LSTM-Attn][STRATEGY GUARD] StrategyGuard Statistics
============================================================
[LSTM-Attn][STRATEGY GUARD] Total checks: 12345, ALLOW=10000, BLOCK=2345
[LSTM-Attn][STRATEGY GUARD] Current decision: BLOCK
[LSTM-Attn][STRATEGY GUARD] Last block reason: win_rate=0.350<0.400; avg_return=-0.0123<-0.0100
[LSTM-Attn][STRATEGY GUARD] Recent trades tracked: 20, recent signals tracked: 0
============================================================
```

## 참고사항

- 모든 명령어는 프로젝트 루트 디렉토리(`/Users/jeongminjun/Projects/Can_bit`)에서 실행해야 합니다.
- 가상환경 활성화(`source .venv/bin/activate`)는 각 세션마다 필요합니다.
- `--no-save` 옵션을 사용하면 리포트 파일을 생성하지 않습니다 (스모크 테스트용).
- 리포트를 저장하려면 `--no-save` 옵션을 제거하세요.

