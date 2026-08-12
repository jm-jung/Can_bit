#!/usr/bin/env bash
# Ops C7 체크 (기간 pinning): 어제 기준 end_date 고정, baseline/variant 동일 기간 실행 후 compare.
# ENV: OPS_END_DATE, OPS_START_365, OPS_START_30 (optional)
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIAG="${PROJECT_ROOT}/data/diagnostics"
OPS_RUNS="${DIAG}/ops_runs"
mkdir -p "${OPS_RUNS}"

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

# 기간 pinning: 기본값 어제. macOS date -v-1d
if [[ -z "${OPS_END_DATE}" ]]; then
  OPS_END_DATE=$(date -v-1d +"%Y-%m-%d")
fi
if [[ -z "${OPS_START_365}" ]]; then
  OPS_START_365=$(date -v-367d +"%Y-%m-%d")
fi
if [[ -z "${OPS_START_30}" ]]; then
  OPS_START_30=$(date -v-31d +"%Y-%m-%d")
fi

LOG_SUFFIX=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${OPS_RUNS}/ops_run_${LOG_SUFFIX}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Ops pinned check log: ${LOG_FILE}"
echo "pinned: end_date=${OPS_END_DATE}, start_date_365=${OPS_START_365}, start_date_30=${OPS_START_30}"

COMMON=(
  --id h15_t0p004 --symbol BTCUSDT --timeframe 5m
  --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24
  --commission 0.0009 --slippage 0.0001
  --regime-filter off --position-scaling off
  --early-exit on --early-exit-lookback 12 --early-exit-p-floor 0.55 --early-exit-bad-k 8
  --time-stop on --time-stop-bars 72
  --end-date "$OPS_END_DATE"
  --start-date-30 "$OPS_START_30"
  --start-date-365 "$OPS_START_365"
  --days-list 30,365
)

run_validation() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation "${COMMON[@]}" "$@"
}

echo "[A] Baseline run_id=pinned_base (partial_tp=off)"
run_validation --partial-tp off --run-id pinned_base

echo "[B] Variant run_id=pinned_tp025_r50 (partial_tp=on, 0.0025, 0.5)"
run_validation --partial-tp on --partial-tp-threshold 0.0025 --partial-tp-ratio 0.5 --run-id pinned_tp025_r50

echo "[C] Compare (ops window pinned)"
COMPARE_OUT=$(python -m scripts.compare_ops_window_pinned \
  --baseline-run-id pinned_base \
  --other-run-ids pinned_tp025_r50 \
  --mode ops 2>&1) || true
echo "$COMPARE_OUT"

VERDICT=$(echo "$COMPARE_OUT" | grep -E '^(OK|WARN|FAIL|FLAG)$' | tail -1)
if [[ -z "$VERDICT" ]]; then
  VERDICT="FAIL"
fi
echo ""
echo "--- verdict (ops) ---"
echo "$VERDICT"
echo "JSON: ${DIAG}/tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_pinned_base.json"
echo "JSON: ${DIAG}/tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_pinned_tp025_r50.json"
if [[ "$VERDICT" == "FAIL" ]]; then
  exit 1
fi
exit 0
