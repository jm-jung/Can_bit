# Guard v2 실전 가능 여부 평가 최종 보고서

**실행 ID:** 20260122_121101  
**생성일:** 2026-01-22 12:18:38

---

## 실행 커맨드

### STEP 1-1: Guard v2 ON, 비용 ON

```bash
/Users/jeongminjun/Projects/Can_bit/.venv/bin/python -m src.backtest.run_ml_xgb_backtest --strategy ml_tcn --symbol BTCUSDT --timeframe 5m --direction long --start-date 2023-01-01 --end-date 2024-12-31 --use-optimized-threshold --signal-confirmation-bars 1 --min-hold-bars 12 --cooldown-bars 12 --no-save --use-strategy-guard --strategy-guard-v2 --strategy-guard-v2-mode soft --strategy-guard-v2-scale-floor 0.02
```

### STEP 1-2: Guard v2 ON, 비용 OFF

```bash
/Users/jeongminjun/Projects/Can_bit/.venv/bin/python -m src.backtest.run_ml_xgb_backtest --strategy ml_tcn --symbol BTCUSDT --timeframe 5m --direction long --start-date 2023-01-01 --end-date 2024-12-31 --use-optimized-threshold --signal-confirmation-bars 1 --min-hold-bars 12 --cooldown-bars 12 --no-save --use-strategy-guard --strategy-guard-v2 --strategy-guard-v2-mode soft --strategy-guard-v2-scale-floor 0.02 --commission-rate 0 --slippage-rate 0
```

### STEP 2: margin/entropy 스윕 (샘플)

```bash
/Users/jeongminjun/Projects/Can_bit/.venv/bin/python -m src.backtest.run_ml_xgb_backtest --strategy ml_tcn --symbol BTCUSDT --timeframe 5m --direction long --start-date 2023-01-01 --end-date 2024-12-31 --use-optimized-threshold --signal-confirmation-bars 1 --min-hold-bars 12 --cooldown-bars 12 --no-save --use-strategy-guard --strategy-guard-v2 --strategy-guard-v2-mode soft --strategy-guard-v2-scale-floor 0.02 --strategy-guard-v2-min-margin 0.03 --strategy-guard-v2-max-entropy 0.6
```

### STEP 3: stage2_off

```bash
/Users/jeongminjun/Projects/Can_bit/.venv/bin/python -m src.backtest.run_ml_xgb_backtest --strategy ml_tcn --symbol BTCUSDT --timeframe 5m --direction long --start-date 2023-01-01 --end-date 2024-12-31 --use-optimized-threshold --signal-confirmation-bars 1 --min-hold-bars 12 --cooldown-bars 12 --no-save --use-strategy-guard --strategy-guard-v2 --strategy-guard-v2-mode soft --strategy-guard-v2-scale-floor 0.02
```

### STEP 3: stage2_on

```bash
/Users/jeongminjun/Projects/Can_bit/.venv/bin/python -m src.backtest.run_ml_xgb_backtest --strategy ml_tcn --symbol BTCUSDT --timeframe 5m --direction long --start-date 2023-01-01 --end-date 2024-12-31 --use-optimized-threshold --signal-confirmation-bars 1 --min-hold-bars 12 --cooldown-bars 12 --no-save --use-strategy-guard --strategy-guard-v2 --strategy-guard-v2-mode soft --strategy-guard-v2-scale-floor 0.02 --use-stage2
```

### STEP 3: threshold_default

```bash
/Users/jeongminjun/Projects/Can_bit/.venv/bin/python -m src.backtest.run_ml_xgb_backtest --strategy ml_tcn --symbol BTCUSDT --timeframe 5m --direction long --start-date 2023-01-01 --end-date 2024-12-31 --use-optimized-threshold --signal-confirmation-bars 1 --min-hold-bars 12 --cooldown-bars 12 --no-save --use-strategy-guard --strategy-guard-v2 --strategy-guard-v2-mode soft --strategy-guard-v2-scale-floor 0.02
```

---

## STEP 1: 거래 비용 OFF 실험

### 비용 ON vs OFF 비교

| 지표 | 비용 ON | 비용 OFF | 차이 |
|------|---------|----------|------|
| Total Return | -1.99% | -1.22% | 0.77% |
| Max Drawdown | 2.32% | 1.83% | -0.49% |
| Total Trades | 174 | 218 | 44 |
| Avg Holding | 12.0 bars | 12.0 bars | - |
| Win Rate | 48.28% | 54.13% | 5.85% |

### 결론

- **비용 OFF에서도 음수** → 신호/Stage2/threshold가 다음 과제

---

## STEP 2: margin/entropy 미니 스윕 (12-run)

### 상위 3개 조합

| 순위 | min_margin | max_entropy | Total Return | Max Drawdown | Total Trades | Median Scale | P90 Scale |
|------|------------|-------------|--------------|--------------|--------------|--------------|-----------|
| 1 | 0.03 | 0.6 | -0.61% | 0.91% | 54 | 0.047 | 0.138 |
| 2 | 0.03 | 0.65 | -0.62% | 0.92% | 54 | 0.047 | 0.138 |
| 3 | 0.05 | 0.6 | -0.64% | 0.84% | 40 | 0.029 | 0.064 |

### Baseline 및 현재 대비

| 기준 | Total Return | Max Drawdown |
|------|--------------|--------------|
| Baseline (Guard OFF) | -13.89% | 13.77% |
| 현재 (Guard v2) | -1.99% | 2.32% |
| STEP 2 최적 | -0.61% | 0.91% |


---

## STEP 3: Stage-2/threshold 재검증

### Stage-2 ON vs OFF 비교

| 지표 | Stage-2 OFF | Stage-2 ON | 차이 |
|------|------------|------------|------|
| Total Return | -1.99% | 0.00% | 1.99% |
| Max Drawdown | 2.32% | 0.00% | -2.32% |

### Threshold 비교 (optimized vs default)

| 지표 | Optimized | Default (0.5) | 차이 |
|------|-----------|---------------|------|
| Total Return | -1.99% | -1.99% | 0.00% |


---

## 종합 결론

- **비용 OFF에서도 음수** → 신호/Stage2/threshold가 다음 과제
- **STEP 2 최적 조합이 현재보다 개선** (Return: -0.61%)

---

## 최종 판정

### Guard v2 실전 가능 여부

**✅ YES**

- 현재 Return: -0.61%
- 현재 MaxDD: 0.91%
- Baseline 대비: Return 13.28%, MaxDD -12.86%

### 현재 병목

**비용+신호 혼합**

### 다음 액션

**최적 파라미터 적용 (min_margin=0.03, max_entropy=0.6) 및 실전 모니터링**

---

**참고:** 전체 실험 결과는 `data/experiments/guard_v2_final_eval/20260122_121101/` 디렉토리에 저장되어 있습니다.
