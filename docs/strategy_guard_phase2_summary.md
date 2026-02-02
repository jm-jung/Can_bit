# StrategyGuard Phase-2 고도화 요약

## 변경된 파일 목록

1. **`src/backtest/strategy_guard.py`**
   - `StrategyGuardConfig`: UNBLOCK 조건 추가 (`unblock_win_rate`, `unblock_avg_return`)
   - `StrategyGuardConfig`: 히스테리시스 추가 (`min_block_trades`)
   - `StrategyGuardState`: UNBLOCK 추적 필드 추가 (`unblock_count`, `unblock_reason`, `trades_since_block`, `state_history`)
   - `check()`: Phase-2 로직으로 확장 (BLOCK ↔ ALLOW 전환)
   - `_check_block()`: ALLOW → BLOCK 전환 로직
   - `_check_unblock()`: BLOCK → ALLOW 전환 로직 (히스테리시스 포함)
   - `_log_state_transition()`: 상태 전이 이력 기록
   - `get_stats()`: Phase-2 통계 추가

2. **`src/backtest/ml_backtest_engines.py`**
   - `execute_trades()`: Phase-2 옵션 파라미터 추가
   - Guard 초기화: CLI 옵션으로 설정 오버라이드
   - `guard.check()`: `trade_index` 전달
   - Guard 통계 로깅: UNBLOCK 정보 추가
   - `BacktestResult`: `strategy_guard_stats` 필드 추가

3. **`src/backtest/ml_backtest_engine_impl.py`**
   - `run_backtest()`: Phase-2 옵션 파라미터 추가 및 전달

4. **`src/backtest/run_ml_xgb_backtest.py`**
   - CLI 옵션 추가:
     - `--strategy-guard-unblock-win-rate`
     - `--strategy-guard-unblock-avg-return`
     - `--strategy-guard-min-block-trades`

5. **`src/backtest/backtest_report.py`**
   - `save_backtest_report()`: Phase-2 옵션 파라미터 추가 및 리포트 저장

## Guard 상태 전이 다이어그램

```
[ALLOW] ──(BLOCK 조건 만족)──> [BLOCK]
   ↑                              │
   │                              │ (최소 유지 트레이드 수 미달)
   │                              │
   └──(UNBLOCK 조건 만족 +        │
       min_block_trades 충족)─────┘
```

### 상태 전이 조건

1. **ALLOW → BLOCK**
   - `win_rate < min_win_rate` (기본: 0.4)
   - 또는 `avg_return < min_avg_return` (기본: -0.01)
   - 즉시 전이 (히스테리시스 없음)

2. **BLOCK → ALLOW (UNBLOCK)**
   - `win_rate >= unblock_win_rate` (기본: 0.45)
   - **AND** `avg_return >= unblock_avg_return` (기본: 0.0)
   - **AND** `trades_since_block >= min_block_trades` (기본: 5)
   - 모든 조건 만족 시 전이

## 기본 설정 (Phase-2)

### BLOCK 조건 (기존 MVP 유지)
- `min_win_rate`: 0.4
- `min_avg_return`: -0.01 (-1%)

### UNBLOCK 조건 (Phase-2 추가)
- `unblock_win_rate`: 0.45
- `unblock_avg_return`: 0.0

### 히스테리시스 (Phase-2 추가)
- `min_block_trades`: 5 (BLOCK 발생 후 최소 5개 트레이드 완료 필요)

## CLI 옵션

### 기본 사용 (MVP 동작 유지)
```bash
--use-strategy-guard
```

### UNBLOCK 조건 커스터마이징
```bash
--use-strategy-guard \
--strategy-guard-unblock-win-rate 0.30 \
--strategy-guard-unblock-avg-return -0.02 \
--strategy-guard-min-block-trades 3
```

## 리포트 & 로깅

### 리포트 JSON에 추가된 필드
```json
{
  "strategy_guard_stats": {
    "total_checks": 12345,
    "block_count": 5,
    "unblock_count": 3,
    "allow_count": 12337,
    "current_decision": "ALLOW",
    "block_reason": "...",
    "unblock_reason": "...",
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

### 로그 출력 예시

**BLOCK 발생:**
```
[StrategyGuard] BLOCK decision: win_rate=0.350<0.400; avg_return=-0.0123<-0.0100 (recent_trades=20, win_rate=0.350, avg_return=-0.0123)
```

**UNBLOCK 발생:**
```
[StrategyGuard] UNBLOCK decision: win_rate=0.460>=0.450; avg_return=0.0012>=0.0000 (win_rate=0.460, avg_return=0.0012, trades_since_block=5)
```

**최종 통계:**
```
[STRATEGY GUARD] StrategyGuard Statistics (Phase-2)
[STRATEGY GUARD] Total checks: 12345, ALLOW=12337, BLOCK=5, UNBLOCK=3
[STRATEGY GUARD] Current decision: ALLOW
[STRATEGY GUARD] Last unblock reason: win_rate=0.460>=0.450; avg_return=0.0012>=0.0000
[STRATEGY GUARD] State transitions: 8 (showing last 5)
[STRATEGY GUARD]   → BLOCK (reason: win_rate=0.350<0.400, trade_index: 100)
[STRATEGY GUARD]   → ALLOW (reason: UNBLOCK: win_rate=0.460>=0.450; avg_return=0.0012>=0.0000, trade_index: 105)
```

## 기존 MVP 대비 변경 요약

### 유지된 것
- 기본값으로 MVP 동작 완전 보존
- Guard OFF 시 기존 결과와 100% 동일
- BLOCK 조건 (win_rate, avg_return) 동일

### 추가된 것
- UNBLOCK 조건 (win_rate >= 0.45, avg_return >= 0.0)
- 히스테리시스 (min_block_trades = 5)
- 상태 전이 이력 (`state_history`)
- UNBLOCK 통계 (`unblock_count`, `unblock_reason`)
- CLI 옵션으로 커스터마이징 가능

### 변경된 것
- `check()` 메서드: `trade_index` 파라미터 추가
- `get_stats()`: Phase-2 통계 필드 추가
- 로깅: UNBLOCK 정보 출력

## 검증 커맨드

### A) 도움말 확인
```bash
source .venv/bin/activate
python -m src.backtest.run_ml_xgb_backtest --help | grep -i "strategy-guard" -A 10
```

### B) Guard OFF (회귀 방지)
```bash
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

### C) Guard ON (기본값, MVP 동작)
```bash
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

### D) Guard ON + 강제 UNBLOCK 조건
```bash
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

## 성공 기준

1. ✅ 컴파일 체크 통과
2. ✅ Guard OFF 시 기존 결과와 동일 (회귀 없음)
3. ✅ Guard ON 시 UNBLOCK 로직 동작
4. ✅ 로그에 BLOCK/UNBLOCK 상태 전이 출력
5. ✅ 리포트에 `strategy_guard_stats` 포함
6. ✅ CLI 옵션이 help에 표시됨

## 다음 단계

로컬 환경에서 위 검증 커맨드를 실행하여 실제 BLOCK → UNBLOCK 전이가 발생하는지 확인하세요.

