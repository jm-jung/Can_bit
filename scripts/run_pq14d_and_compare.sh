#!/usr/bin/env bash
# 1) PQ-14d (proba_quantile q_window=4032) 실행
# 2) Baseline / PQ-30d / PQ-14d 비교 MD 생성
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "=== (1) PQ-14d run (q_window=4032, q=0.95) ==="
python -X faulthandler -m scripts.run_tcn_candidate_validation \
  --id h15_t0p004 --symbol BTCUSDT --timeframe 5m \
  --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24 \
  --regime-filter proba_quantile --q-window 4032 --q 0.95 --p-floor 0.55 \
  --position-scaling off \
  --days-list 30,365

echo ""
echo "=== (2) Compare Baseline / PQ-30d / PQ-14d → MD ==="
python -m scripts.compare_baseline_pq30d_pq14d

echo ""
echo "Done. See data/diagnostics/COMPARE_BASELINE_PQ30D_PQ14D.md"
