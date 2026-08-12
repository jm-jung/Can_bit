# 운영 확정 파라미터 (Current Ops Params)

TCN 후보 검증/모니터링에 사용하는 **고정 베이스라인** 및 Phase 채택값 정리.

## 공통 (고정)

- **id**: h15_t0p004
- **symbol**: BTCUSDT
- **timeframe**: 5m
- **min_max_proba**: 0.58
- **max_entropy**: 1.35
- **min_hold**: 48
- **cooldown**: 24
- **commission**: 0.0009
- **slippage**: 0.0001
- **regime-filter**: off
- **position-scaling**: off
- **early-exit**: on
- **early-exit-lookback**: 12
- **early-exit-p-floor**: 0.55
- **early-exit-bad-k**: 8
- **time_stop**: on
- **time_stop_bars**: 72
- **days-list**: 30,365

## Phase C8 결론 (Partial TP 미채택)

- **partial_tp**: **OFF** (C7/C8 스윕 결과 365d cost_on·MDD 악화로 REJECT, KEEP_BASELINE)
- 운영: time_stop=72, early_exit base, partial_tp OFF

## 재현 체크

```bash
bash scripts/run_ops_c7_adopt_check_pinned.sh
```

- Baseline run_id: `pinned_base` (partial_tp=off)
- Variant run_id: `pinned_tp025_r50` (partial_tp=on, 0.0025, 0.5)
- Compare: `scripts/compare_ops_window_pinned` (ops verdict: WARN/FAIL)
- 전략 정리: `docs/ops_current_strategy.md`
