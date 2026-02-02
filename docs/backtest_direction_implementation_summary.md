# 백테스트 Direction 필터 구현 요약

## 변경된 파일 목록

1. **`src/backtest/run_ml_xgb_backtest.py`**
   - CLI 옵션 추가: `--direction {both,long,short}` (기본값: `both`)
   - `direction`을 `long_only`/`short_only`로 변환하여 엔진에 전달
   - 리포트 저장 시 `direction` 파라미터 전달

2. **`src/backtest/ml_backtest_engines.py`**
   - `execute_trades()`에 `long_only`/`short_only` 파라미터 추가
   - Direction 필터링 로직 추가 (signal 처리 직전)
   - `block_reasons`에 `"direction_filter"` 추가
   - 로깅 개선 (LONG/SHORT entry count, direction filter 상태)

3. **`src/backtest/ml_backtest_engine_impl.py`**
   - `LstmAttnBacktestEngine.run_backtest()`에서 `long_only`/`short_only`를 `execute_trades()`에 전달
   - `generate_signals()`에 `long_only`/`short_only` 전달 (이미 구현됨)

4. **`src/backtest/backtest_report.py`**
   - `save_backtest_report()`에 `direction` 파라미터 추가
   - 리포트 JSON에 `direction` 필드 저장

5. **`docs/backtest_direction_filter_guide.md`** (신규)
   - 사용 가이드 및 예시

## Direction 전달 흐름

```
CLI (run_ml_xgb_backtest.py)
  └─> parse_args() → args.direction
      └─> main() → long_only = (args.direction == "long")
                   short_only = (args.direction == "short")
          └─> engine.run_backtest(long_only=..., short_only=...)
              └─> LstmAttnBacktestEngine.run_backtest()
                  ├─> generate_signals(long_only=..., short_only=...)  [신호 생성 단계 필터링]
                  └─> execute_trades(long_only=..., short_only=...)    [실행 단계 안전장치]
                      └─> signal 처리 시 direction 필터 적용
                          └─> save_backtest_report(direction=args.direction)
```

## 구현 상세

### 1. CLI 옵션 (run_ml_xgb_backtest.py: 237-246)

```python
parser.add_argument(
    "--direction",
    type=str,
    choices=["both", "long", "short"],
    default="both",
    help="거래 방향 필터: 'both' (LONG+SHORT 모두), 'long' (LONG-only), 'short' (SHORT-only). "
         "기본값: both (기존 동작 유지)",
)
```

### 2. Direction 변환 (run_ml_xgb_backtest.py: 342-344)

```python
long_only = (args.direction == "long")
short_only = (args.direction == "short")
logger.info(f"[ML Backtest] Direction filter: {args.direction} (long_only={long_only}, short_only={short_only})")
```

### 3. 신호 생성 단계 필터링 (ml_backtest_engine_impl.py: 395-449)

`generate_signals()`에서 `long_only`/`short_only`를 `ActionDecisionConfig`에 전달하여 신호 생성 단계에서 필터링

### 4. 실행 단계 안전장치 (ml_backtest_engines.py: 272-283)

```python
# [DIRECTION FILTER] long_only/short_only 필터링 (안전장치)
if long_only and signal == "SHORT":
    signal = "HOLD"  # SHORT 신호를 HOLD로 변환
    block_reasons["direction_filter"] = block_reasons.get("direction_filter", 0) + 1
elif short_only and signal == "LONG":
    signal = "HOLD"  # LONG 신호를 HOLD로 변환
    block_reasons["direction_filter"] = block_reasons.get("direction_filter", 0) + 1
```

### 5. 로깅 개선 (ml_backtest_engines.py: 857-870)

```python
logger.info(
    f"{self.log_prefix}[EXECUTION DEBUG] Total trades: {stats['total_trades']}, "
    f"entries={entries_attempted} (LONG={entry_count_long}, SHORT={entry_count_short}), "
    f"exits={exits_executed}, tp={tp_exits}, sl={sl_exits}, flips={flip_count}"
)
if long_only:
    logger.info(
        f"{self.log_prefix}[EXECUTION DEBUG] Direction filter: LONG-only mode active. "
        f"SHORT entries blocked: {block_reasons.get('direction_filter', 0)}"
    )
elif short_only:
    logger.info(
        f"{self.log_prefix}[EXECUTION DEBUG] Direction filter: SHORT-only mode active. "
        f"LONG entries blocked: {block_reasons.get('direction_filter', 0)}"
    )
```

## 검증 커맨드 (복사-붙여넣기용)

### A) 가상환경 활성화 & 기본 도움말 확인

```bash
source .venv/bin/activate
python -m src.backtest.run_ml_xgb_backtest --help | grep -E "(direction|--direction)" -A 2
```

### B) BOTH 모드 스모크 백테스트 (회귀 방지)

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

### C) LONG-only 백테스트

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
- Traceback 없이 정상 종료

### D) SHORT-only 백테스트

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
- Traceback 없이 정상 종료

### E) Stage-2와 함께 사용

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

### F) 회귀 테스트 (ml_xgb, ml_lstm_attn)

```bash
# ml_xgb (기존 동작 유지 확인)
python -m src.backtest.run_ml_xgb_backtest \
    --strategy ml_xgb \
    --symbol BTCUSDT \
    --timeframe 5m \
    --direction both \
    --start-date 2025-01-01 \
    --end-date 2025-01-10 \
    --no-save

# ml_lstm_attn (기존 동작 유지 확인)
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

## 성공 기준

1. ✅ 컴파일 체크 통과
2. ✅ CLI 옵션 추가 확인
3. ✅ Direction이 엔진까지 전달 확인
4. ✅ 신호 생성 단계 필터링 확인
5. ✅ 실행 단계 안전장치 확인
6. ✅ 리포트에 direction 저장 확인
7. ✅ 로깅 개선 확인

## 주의사항

- **기존 동작 유지**: `--direction both` (기본값)는 기존과 동일하게 동작합니다.
- **회귀 방지**: `both` 모드에서 기존 결과가 변경되지 않습니다.
- **다른 전략 호환**: `ml_xgb`, `ml_lstm_attn` 전략도 동일하게 `--direction` 옵션을 사용할 수 있습니다.

## 다음 단계

로컬 환경에서 위 검증 커맨드를 실행하여 실제 동작을 확인하세요. 각 direction 옵션에서 기대한 결과가 나오는지 확인하세요.

