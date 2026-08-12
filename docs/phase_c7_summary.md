# Phase C7 Partial Take-Profit — ADOPT 확정 요약

## 왜 ADOPT인지

- C7 sweep(threshold/ratio 변형) 및 스팟체크에서 **BEST가 partial_tp=on(0.0025, 0.5)** 설정으로 선정됨.
- Baseline(partial_tp=off) 대비 365d cost_on 개선(>= +0.001), MDD 악화 <= +0.005, trades 감소 <= 5% 조건을 만족하는 run이 있어 **ADOPT 확정**.
- 운영 세트에는 **partial_tp=on, partial_tp_threshold=0.0025, partial_tp_ratio=0.5** 반영.

## 기준 run_id / JSON

- **Baseline**: run_id=`ops_base_c7` → `data/diagnostics/tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_ops_base_c7.json`
- **C7 Adopt**: run_id=`ops_c7_tp025_r50` → `data/diagnostics/tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_ops_c7_tp025_r50.json`

## 재현 커맨드

```bash
cd /Users/jeongminjun/Projects/Can_bit && source .venv/bin/activate
export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# (A) Baseline
python -X faulthandler -m scripts.run_tcn_candidate_validation \
  --id h15_t0p004 --symbol BTCUSDT --timeframe 5m \
  --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24 \
  --commission 0.0009 --slippage 0.0001 \
  --regime-filter off --position-scaling off \
  --early-exit on --early-exit-lookback 12 --early-exit-p-floor 0.55 --early-exit-bad-k 8 \
  --time-stop on --time-stop-bars 72 \
  --partial-tp off --run-id ops_base_c7 --days-list 30,365

# (B) C7 Adopt
python -X faulthandler -m scripts.run_tcn_candidate_validation \
  --id h15_t0p004 --symbol BTCUSDT --timeframe 5m \
  --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24 \
  --commission 0.0009 --slippage 0.0001 \
  --regime-filter off --position-scaling off \
  --early-exit on --early-exit-lookback 12 --early-exit-p-floor 0.55 --early-exit-bad-k 8 \
  --time-stop on --time-stop-bars 72 \
  --partial-tp on --partial-tp-threshold 0.0025 --partial-tp-ratio 0.5 \
  --run-id ops_c7_tp025_r50 --days-list 30,365

# (C) Compare (verdict=ADOPT/FAIL)
python -m scripts.compare_baseline_vs_partial_tp \
  --baseline-run-id ops_base_c7 --other-run-ids ops_c7_tp025_r50
```

한 번에 실행:

```bash
bash scripts/run_ops_c7_adopt_check.sh
```

## JSON partial_tp 타입 정규화

run 결과 JSON에서 partial_tp 관련 필드는 아래와 같이 일관되게 기록된다.

| 설정 | partial_tp_count | pct_partial_tp | partial_tp_avg_pnl |
|------|-------------------|----------------|---------------------|
| partial_tp=on  | int >= 0 (항상) | float (항상) | float 또는 0.0 (count==0이면 0.0) |
| partial_tp=off | 0 (항상)        | 0.0 (항상)   | None                |

- partial_tp=off여도 results 각 span에 **partial_tp_count=0, pct_partial_tp=0.0** 는 반드시 기록됨.
- partial_tp_avg_pnl은 off일 때 **None** (의미 없음). on이고 count==0이면 **0.0**.
