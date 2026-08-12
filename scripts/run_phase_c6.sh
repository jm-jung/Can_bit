#!/usr/bin/env bash
# Phase C6: early_exit micro sweep (time_stop=72 고정). 완전 자동: baseline + 4 sweep + compare + BEST spot + compare.
set -e
PROJECT_ROOT="/Users/jeongminjun/Projects/Can_bit"
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
LOG_FILE="${PHASE_RUNS}/phase_c6_run_${LOG_SUFFIX}.log"
# exec > >(tee ...) 실패 시(예: /dev/fd 권한): ./scripts/run_phase_c6.sh 2>&1 | tee "${PHASE_RUNS}/phase_c6_run_${LOG_SUFFIX}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Phase C6 log: ${LOG_FILE}"

COMMON=(
  --id h15_t0p004 --symbol BTCUSDT --timeframe 5m
  --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24
  --regime-filter off --position-scaling off
  --time-stop on --time-stop-bars 72
  --days-list 30,365
)

run() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation "${COMMON[@]}" "$@"
}

echo "[3-1] Baseline phase_c6_base_ts72_ee_base"
run --run-id phase_c6_base_ts72_ee_base \
  --early-exit on --early-exit-lookback 12 --early-exit-p-floor 0.55 --early-exit-bad-k 8

echo "[3-2] Sweep (A) phase_c6_ee_k10"
run --run-id phase_c6_ee_k10 \
  --early-exit on --early-exit-lookback 12 --early-exit-p-floor 0.55 --early-exit-bad-k 10

echo "[3-2] Sweep (B) phase_c6_ee_k6"
run --run-id phase_c6_ee_k6 \
  --early-exit on --early-exit-lookback 12 --early-exit-p-floor 0.55 --early-exit-bad-k 6

echo "[3-2] Sweep (C) phase_c6_ee_l10"
run --run-id phase_c6_ee_l10 \
  --early-exit on --early-exit-lookback 10 --early-exit-p-floor 0.55 --early-exit-bad-k 8

echo "[3-2] Sweep (D) phase_c6_ee_p056"
run --run-id phase_c6_ee_p056 \
  --early-exit on --early-exit-lookback 12 --early-exit-p-floor 0.56 --early-exit-bad-k 8

echo "[3-3] Compare"
COMPARE_OUT=$(python -m scripts.compare_baseline_vs_early_exit_sweep \
  --baseline-run-id phase_c6_base_ts72_ee_base \
  --other-run-ids phase_c6_ee_k10,phase_c6_ee_k6,phase_c6_ee_l10,phase_c6_ee_p056 2>&1)
echo "$COMPARE_OUT"

BEST=$(echo "$COMPARE_OUT" | sed -n 's/^BEST run_id: *//p' | head -1)
if [[ -z "$BEST" ]]; then
  BEST="phase_c6_base_ts72_ee_base"
fi
echo "[3-4] BEST=${BEST} → spot run_id=${BEST}_spot"

# Spot: BEST와 동일 파라미터, run-id만 _spot
case "$BEST" in
  phase_c6_ee_k10)
    run --run-id phase_c6_ee_k10_spot --early-exit on --early-exit-lookback 12 --early-exit-p-floor 0.55 --early-exit-bad-k 10
    ;;
  phase_c6_ee_k6)
    run --run-id phase_c6_ee_k6_spot --early-exit on --early-exit-lookback 12 --early-exit-p-floor 0.55 --early-exit-bad-k 6
    ;;
  phase_c6_ee_l10)
    run --run-id phase_c6_ee_l10_spot --early-exit on --early-exit-lookback 10 --early-exit-p-floor 0.55 --early-exit-bad-k 8
    ;;
  phase_c6_ee_p056)
    run --run-id phase_c6_ee_p056_spot --early-exit on --early-exit-lookback 12 --early-exit-p-floor 0.56 --early-exit-bad-k 8
    ;;
  *)
    run --run-id phase_c6_base_ts72_ee_base_spot --early-exit on --early-exit-lookback 12 --early-exit-p-floor 0.55 --early-exit-bad-k 8
    ;;
esac

echo "[3-4] Compare (with spot)"
python -m scripts.compare_baseline_vs_early_exit_sweep \
  --baseline-run-id phase_c6_base_ts72_ee_base \
  --other-run-ids phase_c6_ee_k10,phase_c6_ee_k6,phase_c6_ee_l10,phase_c6_ee_p056,"${BEST}_spot" \
  --summary-md phase_c6_early_exit_sweep_summary.md

echo "Done. Summary: ${DIAG}/phase_c6_early_exit_sweep_summary.md"
