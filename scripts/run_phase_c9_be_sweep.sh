#!/usr/bin/env bash
# Phase C9: Break-even stop sweep. 기간 pinning, baseline (BE off) + 3 variants (be 0.002/0.003/0.004), compare, summary.
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

END_DATE=$(date -v-1d +"%Y-%m-%d")
START_365=$(date -v-367d +"%Y-%m-%d")
START_30=$(date -v-31d +"%Y-%m-%d")

LOG_SUFFIX=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PHASE_RUNS}/phase_c9_be_${LOG_SUFFIX}.log"
echo "Phase C9 BE sweep log: ${LOG_FILE} (pinned end_date=${END_DATE})"
# 로그 파일에도 남기려면: bash "$0" 2>&1 | tee -a "${LOG_FILE}"

COMMON=(
  --id h15_t0p004 --symbol BTCUSDT --timeframe 5m
  --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24
  --commission 0.0009 --slippage 0.0001
  --regime-filter off --position-scaling off
  --time-stop on --time-stop-bars 72
  --early-exit on --early-exit-lookback 12 --early-exit-p-floor 0.55 --early-exit-bad-k 8
  --partial-tp off
  --end-date "$END_DATE" --start-date-30 "$START_30" --start-date-365 "$START_365"
  --days-list 30,365
)

run_v() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation "${COMMON[@]}" "$@"
}

echo "[C9-1] Baseline phase_c9_base (break_even=off)"
run_v --break-even-stop off --run-id phase_c9_base

echo "[C9-2a] phase_c9_be002 (be_threshold=0.002)"
run_v --break-even-stop on --be-threshold 0.002 --run-id phase_c9_be002

echo "[C9-2b] phase_c9_be003 (be_threshold=0.003)"
run_v --break-even-stop on --be-threshold 0.003 --run-id phase_c9_be003

echo "[C9-2c] phase_c9_be004 (be_threshold=0.004)"
run_v --break-even-stop on --be-threshold 0.004 --run-id phase_c9_be004

echo "[C9-3] Compare + summary"
python -m scripts.compare_phase_c9_be_sweep \
  --baseline-run-id phase_c9_base \
  --other-run-ids phase_c9_be002,phase_c9_be003,phase_c9_be004 \
  --summary-md phase_c9_be_sweep_summary.md

echo "Done. Summary: ${DIAG}/phase_c9_be_sweep_summary.md"
exit 0
