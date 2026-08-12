#!/usr/bin/env bash
# Phase C7: Partial Take-Profit sweep. 완전 자동: baseline + 4 sweep + compare + BEST spot x2 + compare + summary.
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIAG="${PROJECT_ROOT}/data/diagnostics"
PHASE_RUNS="${DIAG}/phase_runs"
mkdir -p "${PHASE_RUNS}"

export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${PROJECT_ROOT}" || exit 1
if [[ ! -d ".venv" ]]; then
  echo "ERROR: .venv not found. Run: cd ${PROJECT_ROOT} && source .venv/bin/activate"
  exit 1
fi
source .venv/bin/activate

LOG_SUFFIX=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PHASE_RUNS}/phase_c7_run_${LOG_SUFFIX}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Phase C7 log: ${LOG_FILE}"

# 고정 베이스라인: id=h15_t0p004, time_stop=72, early_exit=on (lookback=12, p_floor=0.55, bad_k=8), days=30,365
COMMON=(
  --id h15_t0p004 --symbol BTCUSDT --timeframe 5m
  --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24
  --regime-filter off --position-scaling off
  --time-stop on --time-stop-bars 72
  --early-exit on --early-exit-lookback 12 --early-exit-p-floor 0.55 --early-exit-bad-k 8
  --days-list 30,365
)

run() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation "${COMMON[@]}" "$@"
}

echo "[C7-1] Baseline phase_c7_base (partial_tp=off)"
run --run-id phase_c7_base --partial-tp off

echo "[C7-2] Sweep 1: phase_c7_tp025_r50 (threshold=0.0025, ratio=0.5)"
run --run-id phase_c7_tp025_r50 --partial-tp on --partial-tp-threshold 0.0025 --partial-tp-ratio 0.5

echo "[C7-2] Sweep 2: phase_c7_tp030_r50 (threshold=0.003, ratio=0.5)"
run --run-id phase_c7_tp030_r50 --partial-tp on --partial-tp-threshold 0.003 --partial-tp-ratio 0.5

echo "[C7-2] Sweep 3: phase_c7_tp040_r50 (threshold=0.004, ratio=0.5)"
run --run-id phase_c7_tp040_r50 --partial-tp on --partial-tp-threshold 0.004 --partial-tp-ratio 0.5

echo "[C7-2] Sweep 4: phase_c7_tp030_r33 (threshold=0.003, ratio=0.33)"
run --run-id phase_c7_tp030_r33 --partial-tp on --partial-tp-threshold 0.003 --partial-tp-ratio 0.33

echo "[C7-3] Compare baseline vs sweep"
COMPARE_OUT=$(python -m scripts.compare_baseline_vs_partial_tp_sweep \
  --baseline-run-id phase_c7_base \
  --other-run-ids phase_c7_tp025_r50,phase_c7_tp030_r50,phase_c7_tp040_r50,phase_c7_tp030_r33 \
  --summary-md phase_c7_partial_tp_summary.md 2>&1)
echo "$COMPARE_OUT"

BEST=$(echo "$COMPARE_OUT" | sed -n 's/^BEST run_id: *//p' | head -1)
if [[ -z "$BEST" ]]; then
  BEST="phase_c7_base"
fi
echo "[C7-4] BEST=${BEST} → spot runs phase_c7_best_spot1, phase_c7_best_spot2"

# Spot: BEST와 동일 설정으로 run_id만 spot1/spot2
case "$BEST" in
  phase_c7_tp025_r50)
    run --run-id phase_c7_best_spot1 --partial-tp on --partial-tp-threshold 0.0025 --partial-tp-ratio 0.5
    run --run-id phase_c7_best_spot2 --partial-tp on --partial-tp-threshold 0.0025 --partial-tp-ratio 0.5
    ;;
  phase_c7_tp030_r50)
    run --run-id phase_c7_best_spot1 --partial-tp on --partial-tp-threshold 0.003 --partial-tp-ratio 0.5
    run --run-id phase_c7_best_spot2 --partial-tp on --partial-tp-threshold 0.003 --partial-tp-ratio 0.5
    ;;
  phase_c7_tp040_r50)
    run --run-id phase_c7_best_spot1 --partial-tp on --partial-tp-threshold 0.004 --partial-tp-ratio 0.5
    run --run-id phase_c7_best_spot2 --partial-tp on --partial-tp-threshold 0.004 --partial-tp-ratio 0.5
    ;;
  phase_c7_tp030_r33)
    run --run-id phase_c7_best_spot1 --partial-tp on --partial-tp-threshold 0.003 --partial-tp-ratio 0.33
    run --run-id phase_c7_best_spot2 --partial-tp on --partial-tp-threshold 0.003 --partial-tp-ratio 0.33
    ;;
  *)
    run --run-id phase_c7_best_spot1 --partial-tp off
    run --run-id phase_c7_best_spot2 --partial-tp off
    ;;
esac

echo "[C7-5] Compare (with spot) and update summary"
python -m scripts.compare_baseline_vs_partial_tp_sweep \
  --baseline-run-id phase_c7_base \
  --other-run-ids phase_c7_tp025_r50,phase_c7_tp030_r50,phase_c7_tp040_r50,phase_c7_tp030_r33,phase_c7_best_spot1,phase_c7_best_spot2 \
  --summary-md phase_c7_partial_tp_summary.md

echo "Done. Summary: ${DIAG}/phase_c7_partial_tp_summary.md"
exit 0
