# StrategyGuard Phase-2 실행 명령어 모음

## 빠른 참조: 복사-붙여넣기용 명령어

### 1. 가상환경 활성화 & 도움말 확인

```bash
source .venv/bin/activate
python -m src.backtest.run_ml_xgb_backtest --help | grep -i "strategy-guard" -A 10
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

### 3. Guard ON (기본값, MVP 동작)

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
- 로그에 "[StrategyGuard] Enabled (Phase-2: UNBLOCK + 히스테리시스)" 출력
- 로그에 "[STRATEGY GUARD] StrategyGuard Statistics (Phase-2)" 섹션 출력
- UNBLOCK 정보 포함 (unblock_count, unblock_reason)
- Traceback 없이 정상 종료

### 4. Guard ON + 강제 UNBLOCK 조건 (쉬운 UNBLOCK)

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
    --strategy-guard-unblock-win-rate 0.30 \
    --strategy-guard-unblock-avg-return -0.02 \
    --strategy-guard-min-block-trades 3 \
    --no-save
```

**성공 조건**:
- UNBLOCK 조건이 더 쉬워서 UNBLOCK 발생 가능성 증가
- 로그에 "UNBLOCK decision" 출력 (조건 만족 시)
- `state_history`에 BLOCK → ALLOW 전이 기록

### 5. Guard ON + 엄격한 UNBLOCK 조건 (어려운 UNBLOCK)

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
    --strategy-guard-unblock-win-rate 0.55 \
    --strategy-guard-unblock-avg-return 0.01 \
    --strategy-guard-min-block-trades 10 \
    --no-save
```

**성공 조건**:
- UNBLOCK 조건이 더 엄격해져서 UNBLOCK 발생 가능성 감소
- BLOCK 상태가 더 오래 유지됨

## 예상 로그 출력 (Phase-2)

### BLOCK 발생
```
[StrategyGuard] BLOCK decision: win_rate=0.350<0.400; avg_return=-0.0123<-0.0100 (recent_trades=20, win_rate=0.350, avg_return=-0.0123)
```

### UNBLOCK 발생
```
[StrategyGuard] UNBLOCK decision: win_rate=0.460>=0.450; avg_return=0.0012>=0.0000 (win_rate=0.460, avg_return=0.0012, trades_since_block=5, min_block_trades=5)
```

### 최종 통계
```
============================================================
[LSTM-Attn][STRATEGY GUARD] StrategyGuard Statistics (Phase-2)
============================================================
[LSTM-Attn][STRATEGY GUARD] Total checks: 12345, ALLOW=10000, BLOCK=5, UNBLOCK=3
[LSTM-Attn][STRATEGY GUARD] Current decision: ALLOW
[LSTM-Attn][STRATEGY GUARD] Last unblock reason: win_rate=0.460>=0.450; avg_return=0.0012>=0.0000
[LSTM-Attn][STRATEGY GUARD] Recent trades tracked: 20, recent signals tracked: 0, trades_since_block: 0
[LSTM-Attn][STRATEGY GUARD] State transitions: 8 (showing last 5)
[LSTM-Attn][STRATEGY GUARD]   → BLOCK (reason: win_rate=0.350<0.400, trade_index: 100)
[LSTM-Attn][STRATEGY GUARD]   → ALLOW (reason: UNBLOCK: win_rate=0.460>=0.450; avg_return=0.0012>=0.0000, trade_index: 105)
============================================================
```

## 리포트 JSON 예시

```json
{
  "strategy_guard_stats": {
    "total_checks": 12345,
    "block_count": 5,
    "unblock_count": 3,
    "allow_count": 10000,
    "current_decision": "ALLOW",
    "block_reason": "",
    "unblock_reason": "win_rate=0.460>=0.450; avg_return=0.0012>=0.0000",
    "trades_since_block": 0,
    "state_history": [
      {
        "decision": "BLOCK",
        "reason": "win_rate=0.350<0.400",
        "trade_index": 100,
        "timestamp": "2025-01-01 10:00:00"
      },
      {
        "decision": "ALLOW",
        "reason": "UNBLOCK: win_rate=0.460>=0.450; avg_return=0.0012>=0.0000",
        "trade_index": 105,
        "timestamp": "2025-01-01 10:25:00"
      }
    ]
  }
}
```

## Guard 상태 전이 다이어그램

```
[ALLOW] ──(BLOCK 조건 만족)──> [BLOCK]
   ↑                              │
   │                              │ (min_block_trades 미달)
   │                              │
   └──(UNBLOCK 조건 만족 +        │
       min_block_trades 충족)─────┘
```

### 전이 조건

1. **ALLOW → BLOCK**
   - `win_rate < 0.4` 또는 `avg_return < -0.01`
   - 즉시 전이 (히스테리시스 없음)

2. **BLOCK → ALLOW (UNBLOCK)**
   - `win_rate >= 0.45` **AND** `avg_return >= 0.0`
   - **AND** `trades_since_block >= 5`
   - 모든 조건 만족 시 전이

## 참고사항

- 모든 명령어는 프로젝트 루트 디렉토리에서 실행해야 합니다.
- 가상환경 활성화(`source .venv/bin/activate`)는 각 세션마다 필요합니다.
- `--no-save` 옵션을 사용하면 리포트 파일을 생성하지 않습니다 (스모크 테스트용).
- 리포트를 저장하려면 `--no-save` 옵션을 제거하세요.
- Phase-2 옵션을 지정하지 않으면 기본값(MVP 동작)이 사용됩니다.

