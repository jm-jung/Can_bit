#!/usr/bin/env bash
# Phase C8: 기간 pinning된 동일 구간에서 baseline + 3 variants + BEST spot 2회, compare, summary.
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
source .venv/bin/activate

# 기간 pinning (어제 기준)
END_DATE=$(date -v-1d +"%Y-%m-%d")
START_365=$(date -v-367d +"%Y-%m-%d")
START_30=$(date -v-31d +"%Y-%m-%d")

LOG_SUFFIX=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PHASE_RUNS}/phase_c8_combo_${LOG_SUFFIX}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Phase C8 combo log: ${LOG_FILE} (pinned end_date=${END_DATE})"

COMMON=(
  --id h15_t0p004 --symbol BTCUSDT --timeframe 5m
  --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24
  --commission 0.0009 --slippage 0.0001
  --regime-filter off --position-scaling off
  --time-stop on --time-stop-bars 72
  --early-exit on --early-exit-lookback 12 --early-exit-p-floor 0.55 --early-exit-bad-k 8
  --end-date "$END_DATE" --start-date-30 "$START_30" --start-date-365 "$START_365"
  --days-list 30,365
)

run_v() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation "${COMMON[@]}" "$@"
}

echo "[C8-1] Baseline phase_c8_base (partial_tp=off)"
run_v --partial-tp off --run-id phase_c8_base

echo "[C8-2a] phase_c8_tp025_r50"
run_v --partial-tp on --partial-tp-threshold 0.0025 --partial-tp-ratio 0.5 --run-id phase_c8_tp025_r50

echo "[C8-2b] phase_c8_tp030_r50"
run_v --partial-tp on --partial-tp-threshold 0.003 --partial-tp-ratio 0.5 --run-id phase_c8_tp030_r50

echo "[C8-2c] phase_c8_tp025_r33"
run_v --partial-tp on --partial-tp-threshold 0.0025 --partial-tp-ratio 0.33 --run-id phase_c8_tp025_r33

echo "[C8-3] Compare"
COMPARE_OUT=$(python -m scripts.compare_phase_c8_combo_sweep \
  --baseline-run-id phase_c8_base \
  --other-run-ids phase_c8_tp025_r50,phase_c8_tp030_r50,phase_c8_tp025_r33 \
  --summary-md phase_c8_combo_sweep_summary.md 2>&1)
echo "$COMPARE_OUT"

BEST=$(echo "$COMPARE_OUT" | sed -n 's/^BEST run_id: *//p' | head -1)
if [[ -z "$BEST" ]]; then
  BEST="phase_c8_base"
fi
echo "[C8-4] BEST=${BEST} → spot1, spot2"

case "$BEST" in
  phase_c8_tp025_r50)
    run_v --partial-tp on --partial-tp-threshold 0.0025 --partial-tp-ratio 0.5 --run-id phase_c8_best_spot1
    run_v --partial-tp on --partial-tp-threshold 0.0025 --partial-tp-ratio 0.5 --run-id phase_c8_best_spot2
    ;;
  phase_c8_tp030_r50)
    run_v --partial-tp on --partial-tp-threshold 0.003 --partial-tp-ratio 0.5 --run-id phase_c8_best_spot1
    run_v --partial-tp on --partial-tp-threshold 0.003 --partial-tp-ratio 0.5 --run-id phase_c8_best_spot2
    ;;
  phase_c8_tp025_r33)
    run_v --partial-tp on --partial-tp-threshold 0.0025 --partial-tp-ratio 0.33 --run-id phase_c8_best_spot1
    run_v --partial-tp on --partial-tp-threshold 0.0025 --partial-tp-ratio 0.33 --run-id phase_c8_best_spot2
    ;;
  *)
    run_v --partial-tp off --run-id phase_c8_best_spot1
    run_v --partial-tp off --run-id phase_c8_best_spot2
    ;;
esac

echo "[C8-5] Compare (with spots) and update summary"
python -m scripts.compare_phase_c8_combo_sweep \
  --baseline-run-id phase_c8_base \
  --other-run-ids phase_c8_tp025_r50,phase_c8_tp030_r50,phase_c8_tp025_r33,phase_c8_best_spot1,phase_c8_best_spot2 \
  --summary-md phase_c8_combo_sweep_summary.md

echo "Done. Summary: ${DIAG}/phase_c8_combo_sweep_summary.md"
exit 0
