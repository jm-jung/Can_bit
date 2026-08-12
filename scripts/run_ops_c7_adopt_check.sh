#!/usr/bin/env bash
# C7 ADOPT 재현 체크: (A) Baseline ops_base_c7, (B) C7 Adopt ops_c7_tp025_r50, (C) Compare → verdict ADOPT/FAIL
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
LOG_FILE="${PHASE_RUNS}/ops_c7_${LOG_SUFFIX}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Ops C7 adopt check log: ${LOG_FILE}"

COMMON=(
  --id h15_t0p004 --symbol BTCUSDT --timeframe 5m
  --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24
  --commission 0.0009 --slippage 0.0001
  --regime-filter off --position-scaling off
  --early-exit on --early-exit-lookback 12 --early-exit-p-floor 0.55 --early-exit-bad-k 8
  --time-stop on --time-stop-bars 72
  --days-list 30,365
)

run_validation() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation "${COMMON[@]}" "$@"
}

echo "[A] Baseline run_id=ops_base_c7 (partial_tp=off)"
run_validation --partial-tp off --run-id ops_base_c7

echo "[B] C7 Adopt run_id=ops_c7_tp025_r50 (partial_tp=on, 0.0025, 0.5)"
run_validation --partial-tp on --partial-tp-threshold 0.0025 --partial-tp-ratio 0.5 --run-id ops_c7_tp025_r50

echo "[C] Compare Baseline vs C7 Adopt"
COMPARE_OUT=$(python -m scripts.compare_baseline_vs_partial_tp \
  --baseline-run-id ops_base_c7 \
  --other-run-ids ops_c7_tp025_r50 2>&1) || true
echo "$COMPARE_OUT"

VERDICT=$(echo "$COMPARE_OUT" | grep -E '^(ADOPT|FAIL)$' | tail -1)
if [[ -z "$VERDICT" ]]; then
  VERDICT="FAIL"
fi
echo ""
echo "--- verdict ---"
echo "$VERDICT"
if [[ "$VERDICT" != "ADOPT" ]]; then
  exit 1
fi
exit 0
