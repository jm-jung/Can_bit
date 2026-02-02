# StrategyGuard 구현 요약

## 변경된 파일 목록

1. **`src/backtest/strategy_guard.py`** (신규)
   - `StrategyGuard` 클래스: 전략 실행 허용/차단 관리
   - `StrategyGuardConfig`: Guard 설정 (MVP: 상수)
   - `StrategyGuardState`: Guard 상태 추적

2. **`src/backtest/ml_backtest_engines.py`**
   - `execute_trades()`에 `use_strategy_guard` 파라미터 추가
   - Guard 초기화 및 개입 로직 추가 (라인 175-177, 285-305)
   - 트레이드 완료 시 `guard.update_trade()` 호출 (라인 380, 546, 832)
   - Guard 통계 로깅 추가 (라인 1013-1030)

3. **`src/backtest/ml_backtest_engine_impl.py`**
   - `LstmAttnBacktestEngine.run_backtest()`에 `use_strategy_guard` 파라미터 추가
   - `execute_trades()` 호출 시 `use_strategy_guard` 전달

4. **`src/backtest/run_ml_xgb_backtest.py`**
   - CLI 옵션 추가: `--use-strategy-guard` (라인 249-257)
   - `engine.run_backtest()` 호출 시 `use_strategy_guard` 전달

5. **`src/backtest/backtest_report.py`**
   - `save_backtest_report()`에 `use_strategy_guard` 파라미터 추가
   - 리포트 JSON에 `use_strategy_guard` 필드 저장

## StrategyGuard 흐름

```
CLI: --use-strategy-guard
  → run_ml_xgb_backtest.py: args.use_strategy_guard = True
  → engine.run_backtest(use_strategy_guard=True, ...)
    → execute_trades(use_strategy_guard=True, ...)
      → Guard 초기화 (StrategyGuard())
      → for each row:
          → guard.update_signal(stage2_trade, timestamp)  [신호 업데이트]
          → guard.check() → ALLOW/BLOCK 판정
          → if BLOCK: signal = "HOLD", continue
          → ... (기존 로직) ...
          → if trade completed: guard.update_trade(profit, timestamp)
      → Guard 통계 로깅
```

## 개입 지점

1. **신호 처리 직전** (`execute_trades()` 라인 285-305)
   - `guard.update_signal()`: Stage-2 통과 여부 기록
   - `guard.check()`: ALLOW/BLOCK 판정
   - BLOCK이면 `signal = "HOLD"`로 변환 후 `continue`

2. **트레이드 완료 시** (라인 380, 546, 832)
   - `guard.update_trade()`: 트레이드 수익률 기록

3. **백테스트 종료 시** (라인 1013-1030)
   - Guard 통계 로깅

## Guard 판정 로직 (MVP)

```python
# 최근 N개 트레이드 성능 계산
recent = recent_trades[-recent_trades_window:]
win_rate = wins / total
avg_return = mean(profits)

# BLOCK 조건
if win_rate < min_win_rate: → BLOCK
if avg_return < min_avg_return: → BLOCK
# (선택) if stage2_pass_rate < min_stage2_pass_rate: → BLOCK

# 그 외 → ALLOW
```

## 기본 설정 (MVP)

- `recent_trades_window`: 20 (최근 20개 트레이드 기준)
- `min_win_rate`: 0.4 (win_rate < 0.4면 BLOCK)
- `min_avg_return`: -0.01 (avg_return < -1%면 BLOCK)
- `min_stage2_pass_rate`: 0.3 (선택, 기본 비활성)

## 검증 커맨드

### A) 도움말 확인
```bash
source .venv/bin/activate
python -m src.backtest.run_ml_xgb_backtest --help | grep strategy-guard
```

### B) Guard OFF (기존 동작 유지)
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

**성공 조건**: 기존과 동일하게 동작, Traceback 없이 정상 종료

### C) Guard ON
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
- 로그에 "[STRATEGY GUARD] StrategyGuard Statistics" 출력
- ALLOW/BLOCK 상태가 로그에 출력
- Traceback 없이 정상 종료

## 기존 전략 보존 확인

- **Guard OFF (기본값)**: `use_strategy_guard=False`이면 Guard가 초기화되지 않음
- **Guard ON**: Guard가 BLOCK해도 기존 direction/Stage-2 로직은 그대로 유지
- **회귀 방지**: Guard OFF 시 기존 결과와 100% 동일해야 함

## 다음 단계

로컬 환경에서 위 검증 커맨드를 실행하여 실제 동작을 확인하세요.

