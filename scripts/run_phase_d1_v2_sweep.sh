#!/usr/bin/env bash
# Phase D1_v2: min_max_proba × max_entropy sweep. 거래 수 감소로 cost_on 회복.
# baseline (0.55, 1.35) + 9 variants, compare, summary.
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
export OPENBLAS_NUM_THREADS=1

cd "${PROJECT_ROOT}" || exit 1
source .venv/bin/activate

# Pinned dates (Phase D1_v2와 동일)
END_DATE="2026-03-03"
START_30="2026-02-01"
START_365="2025-03-02"

LOG_SUFFIX=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PHASE_RUNS}/phase_d1v2_sweep_${LOG_SUFFIX}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Phase D1_v2 sweep log: ${LOG_FILE} (pinned end_date=${END_DATE})"

COMMON=(
  --id h15_t0p004 --symbol BTCUSDT --timeframe 5m
  --min-hold 36 --cooldown 12
  --commission 0.0009 --slippage 0.0001
  --regime-filter off --position-scaling off
  --time-stop on --time-stop-bars 72
  --early-exit on --early-exit-lookback 12 --early-exit-p-floor 0.55 --early-exit-bad-k 8
  --partial-tp off --break-even-stop off
  --end-date "$END_DATE" --start-date-30 "$START_30" --start-date-365 "$START_365"
  --days-list 30,365
)

run_v() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation "${COMMON[@]}" "$@"
}

echo "[D1v2-1] Baseline phase_d1v2_base (min_max_proba=0.55, max_entropy=1.35)"
run_v --min-max-proba 0.55 --max-entropy 1.35 --run-id phase_d1v2_base

echo "[D1v2-2] Variants: min_max_proba × max_entropy (9 runs)"
run_v --min-max-proba 0.58 --max-entropy 1.35 --run-id phase_d1v2_p058_e135
run_v --min-max-proba 0.58 --max-entropy 1.25 --run-id phase_d1v2_p058_e125
run_v --min-max-proba 0.58 --max-entropy 1.15 --run-id phase_d1v2_p058_e115
run_v --min-max-proba 0.60 --max-entropy 1.35 --run-id phase_d1v2_p060_e135
run_v --min-max-proba 0.60 --max-entropy 1.25 --run-id phase_d1v2_p060_e125
run_v --min-max-proba 0.60 --max-entropy 1.15 --run-id phase_d1v2_p060_e115
run_v --min-max-proba 0.62 --max-entropy 1.35 --run-id phase_d1v2_p062_e135
run_v --min-max-proba 0.62 --max-entropy 1.25 --run-id phase_d1v2_p062_e125
run_v --min-max-proba 0.62 --max-entropy 1.15 --run-id phase_d1v2_p062_e115

OTHER_IDS="phase_d1v2_p058_e135,phase_d1v2_p058_e125,phase_d1v2_p058_e115,phase_d1v2_p060_e135,phase_d1v2_p060_e125,phase_d1v2_p060_e115,phase_d1v2_p062_e135,phase_d1v2_p062_e125,phase_d1v2_p062_e115"

echo "[D1v2-3] Compare + summary"
python -m scripts.compare_phase_d1_v2_sweep \
  --baseline-run-id phase_d1v2_base \
  --other-run-ids "$OTHER_IDS" \
  --summary-md phase_d1v2_sweep_summary.md

echo "Done. Summary: ${DIAG}/phase_d1v2_sweep_summary.md"
exit 0
