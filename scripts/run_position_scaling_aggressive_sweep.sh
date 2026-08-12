#!/usr/bin/env bash
# TCN Position Scaling 공격적 튜닝: Baseline + Agg-1/2/3 (30d/365d) 실행 후 compare.
# 사용법: 프로젝트 루트에서
#   cd /Users/jeongminjun/Projects/Can_bit && source .venv/bin/activate && bash scripts/run_position_scaling_aggressive_sweep.sh

set -e
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

BASE="python -X faulthandler -m scripts.run_tcn_candidate_validation"
COMMON="--id h15_t0p004 --symbol BTCUSDT --timeframe 5m --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24 --regime-filter off --days-list 30,365"

echo "=== (1) Torch import test ==="
python -X faulthandler -c "import torch; print('torch', torch.__version__); print('threads', torch.get_num_threads())"

echo ""
echo "=== (2) BASELINE (position_scaling=off) ==="
$BASE $COMMON --position-scaling off

echo ""
echo "=== (3) Agg-1 (p_floor=0.53, p_full=0.62, size_min=0.10) ==="
$BASE $COMMON --position-scaling linear --position-p-floor 0.53 --position-p-full 0.62 --position-size-min 0.10 --position-size-max 1.0

echo ""
echo "=== (4) Agg-2 (p_floor=0.52, p_full=0.60, size_min=0.05) ==="
$BASE $COMMON --position-scaling linear --position-p-floor 0.52 --position-p-full 0.60 --position-size-min 0.05 --position-size-max 1.0

echo ""
echo "=== (5) Agg-3 (p_floor=0.52, p_full=0.58, size_min=0.03) ==="
$BASE $COMMON --position-scaling linear --position-p-floor 0.52 --position-p-full 0.58 --position-size-min 0.03 --position-size-max 1.0

echo ""
echo "=== (6) Compare baseline vs latest scaled ==="
python -m scripts.compare_baseline_vs_position_scaling

echo ""
echo "=== Done. Check data/diagnostics/ for JSON/MD. ==="
