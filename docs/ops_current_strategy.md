# 운영 전략 (Current Strategy)

Phase C8 공식 결론 반영. **KEEP_BASELINE**: time_stop=72, early_exit base, **partial_tp=OFF**.

## 모델 / 시장

- **model**: TCN
- **symbol**: BTCUSDT
- **timeframe**: 5m

## Entry 필터

- **min_max_proba**: 0.58
- **max_entropy**: 1.35
- **min_hold**: 48
- **cooldown**: 24

## Exit 규칙

- **time_stop**: 72 bars
- **early_exit**: ON (lookback=12, p_floor=0.55, bad_k=8)
- **partial_tp**: **OFF**
- **break_even**: **OFF**

## 비용 모델

- **commission**: 0.0009
- **slippage**: 0.0001

## Research notes

- **Phase C7/C8**: Partial TP는 365d pinned window 기준 cost_on 및 MDD 모두 크게 악화되어 **REJECT**. 운영은 baseline 유지 (partial_tp OFF).
- **Phase C9**: Break-even stop (be002/be003/be004) 365d cost_on·MDD 모두 악화 → **REJECT**. KEEP_BASELINE, BE 추가 튜닝 중단.

## Exit 실험 종료 (Phase C4~C9)

- Partial TP (C7/C8), Break-even stop (C9) 추가 탐색 **중단**. 운영 기본값 확정.
- time_stop=72, early_exit=ON (lookback=12, p_floor=0.55, bad_k=8), **partial_tp=OFF**, **break_even=OFF**.

## 재현 / ops 체크

- 기간 pinning: `bash scripts/run_ops_c7_adopt_check_pinned.sh`
- Phase C9 스윕: `bash scripts/run_phase_c9_be_sweep.sh`
- C9 sanity check: `--emit-trade-log on` 으로 phase_c9_base, phase_c9_be004 실행 후 `python -m scripts.analyze_exit_reason_shift --baseline-run-id phase_c9_base --other-run-id phase_c9_be004`
