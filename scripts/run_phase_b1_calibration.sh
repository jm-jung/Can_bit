#!/usr/bin/env bash
# Phase B1: Probability Calibration (Temperature Scaling) 완전 자동 실행.
# 사용법: ./scripts/run_phase_b1_calibration.sh

set -e
PROJECT_ROOT="/Users/jeongminjun/Projects/Can_bit"
DIAG="${PROJECT_ROOT}/data/diagnostics"
PHASE_RUNS="${DIAG}/phase_runs"

run_env() {
  export PYTHONFAULTHANDLER=1
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
}

check_venv() {
  if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "ERROR: venv not active. Run: cd ${PROJECT_ROOT} && source .venv/bin/activate"
    exit 1
  fi
}

check_torch() {
  if [[ -n "${SKIP_TORCH_TEST}" ]]; then
    echo "SKIP_TORCH_TEST set: skipping torch import test."
    return 0
  fi
  if ! python -X faulthandler -c "import torch; print('torch', torch.__version__); print('threads', torch.get_num_threads())" 2>&1; then
    echo "ERROR: torch import failed (segfault 등). env 설정 확인 후 재시도. (로컬에서 segfault 시 SKIP_TORCH_TEST=1로 건너뛸 수 있음)"
    exit 1
  fi
}

# 고정 파라미터 (Phase B1)
COMMON_OPTS=(--id h15_t0p004 --symbol BTCUSDT --timeframe 5m --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24 --regime-filter off --position-scaling off --early-exit on --early-exit-lookback 12 --early-exit-p-floor 0.55 --early-exit-bad-k 8 --days-list 30,365)

run_phase_b1_base() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --calibration off \
    --run-id phase_b1_base
}

run_phase_b1_t11() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --calibration temp --calibration-T 1.1 \
    --run-id phase_b1_t11
}

run_phase_b1_t12() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --calibration temp --calibration-T 1.2 \
    --run-id phase_b1_t12
}

run_phase_b1_t14() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --calibration temp --calibration-T 1.4 \
    --run-id phase_b1_t14
}

main() {
  cd "${PROJECT_ROOT}" || exit 1
  if [[ ! -d ".venv" ]]; then
    echo "ERROR: .venv not found in ${PROJECT_ROOT}"
    exit 1
  fi
  source .venv/bin/activate
  check_venv
  run_env
  check_torch

  mkdir -p "${PHASE_RUNS}"
  LOG_SUFFIX=$(date +%Y%m%d_%H%M%S)
  LOG_FILE="${PHASE_RUNS}/phase_b1_${LOG_SUFFIX}.log"
  echo "Log: ${LOG_FILE}" 1>&2
  exec >> "${LOG_FILE}" 2>&1
  echo "Log: ${LOG_FILE}"

  echo "[Phase B1] phase_b1_base ..."
  run_phase_b1_base
  echo "[Phase B1] phase_b1_t11 (T=1.1) ..."
  run_phase_b1_t11
  echo "[Phase B1] phase_b1_t12 (T=1.2) ..."
  run_phase_b1_t12
  echo "[Phase B1] phase_b1_t14 (T=1.4) ..."
  run_phase_b1_t14
  echo "[Phase B1] compare_baseline_vs_calibration ..."
  python -m scripts.compare_baseline_vs_calibration --baseline-run-id phase_b1_base --other-run-ids phase_b1_t11,phase_b1_t12,phase_b1_t14

  echo "Done (exit 0)"
}

main "$@"
