#!/usr/bin/env bash
# Phase A (비용 시나리오) + Phase C (flat-proba gate) + Phase C2 (flat-gate sweep) 완전 자동 실행.
# 사용법: ./scripts/run_phase_checks.sh A | C | C2 | C3 | C4 | C5 | ALL

set -e
PROJECT_ROOT="/Users/jeongminjun/Projects/Can_bit"
DIAG="${PROJECT_ROOT}/data/diagnostics"
PHASE_RUNS="${DIAG}/phase_runs"
LOG_SUFFIX=""

run_env() {
  export PYTHONFAULTHANDLER=1
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
}

# 공통 옵션 (배열로 전달)
COMMON_OPTS=(--id h15_t0p004 --symbol BTCUSDT --timeframe 5m --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24 --regime-filter off --position-scaling off --early-exit on --early-exit-lookback 12 --early-exit-p-floor 0.55 --early-exit-bad-k 8 --days-list 30,365)

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
    echo "Torch import failed (exit 139 등). 재실행: SKIP_TORCH_TEST=1"
    exec env SKIP_TORCH_TEST=1 "$0" "$@"
  fi
}

run_phase_a_base() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --commission 0.0009 --slippage 0.0001 \
    --run-id phase_a_base
}

run_phase_a_conservative() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --commission 0.0012 --slippage 0.00015 \
    --run-id phase_a_conservative
}

run_phase_c_flatgate() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --entry-flat-gate on --max-flat-proba 0.45 \
    --run-id phase_c_flatgate_045
}

# Phase C2: flat-gate sweep (baseline + max_flat_proba 0.45, 0.40, 0.35, 0.30)
run_phase_c2_base() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --entry-flat-gate off \
    --run-id phase_c2_base
}

run_phase_c2_flatgate_045() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --entry-flat-gate on --max-flat-proba 0.45 \
    --run-id phase_c2_flatgate_045
}

run_phase_c2_flatgate_040() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --entry-flat-gate on --max-flat-proba 0.40 \
    --run-id phase_c2_flatgate_040
}

run_phase_c2_flatgate_035() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --entry-flat-gate on --max-flat-proba 0.35 \
    --run-id phase_c2_flatgate_035
}

run_phase_c2_flatgate_030() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --entry-flat-gate on --max-flat-proba 0.30 \
    --run-id phase_c2_flatgate_030
}

# Phase C3: flat-aware early exit (baseline + threshold 0.30/0.35, badk_delta=2)
run_phase_c3_base() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --flat-exit-aware off \
    --run-id phase_c3_base
}

run_phase_c3_flat_030_d2() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --flat-exit-aware on --flat-exit-threshold 0.30 --flat-exit-badk-delta 2 \
    --run-id phase_c3_flat_030_d2
}

run_phase_c3_flat_035_d2() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --flat-exit-aware on --flat-exit-threshold 0.35 --flat-exit-badk-delta 2 \
    --run-id phase_c3_flat_035_d2
}

run_phase_c3() {
  run_phase_c3_base
  run_phase_c3_flat_030_d2
  run_phase_c3_flat_035_d2
  python -m scripts.compare_baseline_vs_flat_exit_aware \
    --baseline-run-id phase_c3_base \
    --other-run-ids phase_c3_flat_030_d2,phase_c3_flat_035_d2
}

# Phase C4: time-stop sweep (baseline + ts72/ts96/ts144, optional combo)
run_phase_c4_base() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --time-stop off \
    --run-id phase_c4_base
}

run_phase_c4_ts72() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --time-stop on --time-stop-bars 72 \
    --run-id phase_c4_ts72
}

run_phase_c4_ts96() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --time-stop on --time-stop-bars 96 \
    --run-id phase_c4_ts96
}

run_phase_c4_ts144() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --time-stop on --time-stop-bars 144 \
    --run-id phase_c4_ts144
}

run_phase_c4_combo_ts96_flat035d2() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --flat-exit-aware on --flat-exit-threshold 0.35 --flat-exit-badk-delta 2 \
    --time-stop on --time-stop-bars 96 \
    --run-id phase_c4_combo_ts96_flat035d2
}

run_phase_c4() {
  run_phase_c4_base
  run_phase_c4_ts72
  run_phase_c4_ts96
  run_phase_c4_ts144
  run_phase_c4_combo_ts96_flat035d2
  python -m scripts.compare_baseline_vs_time_stop_sweep \
    --baseline-run-id phase_c4_base \
    --other-run-ids phase_c4_ts72,phase_c4_ts96,phase_c4_ts144,phase_c4_combo_ts96_flat035d2
}

# Phase C5: time-stop micro sweep (baseline=ts72 + ts60/ts84/ts96)
run_phase_c5_base_ts72() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --time-stop on --time-stop-bars 72 \
    --run-id phase_c5_base_ts72
}

run_phase_c5_ts60() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --time-stop on --time-stop-bars 60 \
    --run-id phase_c5_ts60
}

run_phase_c5_ts84() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --time-stop on --time-stop-bars 84 \
    --run-id phase_c5_ts84
}

run_phase_c5_ts96() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation \
    "${COMMON_OPTS[@]}" \
    --time-stop on --time-stop-bars 96 \
    --run-id phase_c5_ts96
}

run_phase_c5() {
  run_phase_c5_base_ts72
  run_phase_c5_ts60
  run_phase_c5_ts84
  run_phase_c5_ts96
  python -m scripts.compare_baseline_vs_time_stop_sweep \
    --baseline-run-id phase_c5_base_ts72 \
    --other-run-ids phase_c5_ts60,phase_c5_ts84,phase_c5_ts96 \
    --summary-md phase_c5_time_stop_sweep_summary.md \
    --max-30d-cost-degradation 0.2
}

run_phase_c2() {
  run_phase_c2_base
  run_phase_c2_flatgate_045
  run_phase_c2_flatgate_040
  run_phase_c2_flatgate_035
  run_phase_c2_flatgate_030
  python -m scripts.compare_baseline_vs_flatgate_sweep \
    --baseline-run-id phase_c2_base \
    --other-run-ids phase_c2_flatgate_045,phase_c2_flatgate_040,phase_c2_flatgate_035,phase_c2_flatgate_030
}

run_phase_a() {
  run_phase_a_base
  run_phase_a_conservative
  python -m scripts.compare_phase_a_costs --base-run-id phase_a_base --other-run-id phase_a_conservative
}

run_phase_c() {
  # phase_a_base JSON 없으면 먼저 실행
  if ! ls "${DIAG}"/tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_a_base.json 1>/dev/null 2>&1; then
    run_phase_a_base
  fi
  run_phase_c_flatgate
  python -m scripts.compare_baseline_vs_flatgate --baseline-run-id phase_a_base --flatgate-run-id phase_c_flatgate_045
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
  check_torch "$@"

  mkdir -p "${PHASE_RUNS}"
  LOG_SUFFIX=$(date +%Y%m%d_%H%M%S)
  LOG_FILE="${PHASE_RUNS}/phase_run_${LOG_SUFFIX}.log"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  echo "Log: ${LOG_FILE}"

  case "${1:-}" in
    A)
      run_phase_a
      ;;
    C)
      run_phase_c
      ;;
    C2)
      run_phase_c2
      ;;
    C3)
      run_phase_c3
      ;;
    C4)
      run_phase_c4
      ;;
    C5)
      run_phase_c5
      ;;
    ALL)
      run_phase_a
      run_phase_c
      ;;
    "")
      echo "Usage: $0 A | C | C2 | C3 | C4 | C5 | ALL"
      echo "  A   : Phase A (phase_a_base + phase_a_conservative + compare)"
      echo "  C   : Phase C (phase_a_base if missing + phase_c_flatgate_045 + compare)"
      echo "  C2  : Phase C2 (phase_c2_base + flatgate sweep 0.45/0.40/0.35/0.30 + compare)"
      echo "  C3  : Phase C3 (phase_c3_base + flat-aware early exit 0.30/0.35 d2 + compare)"
      echo "  C4  : Phase C4 (phase_c4_base + time-stop 72/96/144 + combo_ts96_flat035d2 + compare)"
      echo "  C5  : Phase C5 (phase_c5_base_ts72 + ts60/ts84/ts96 micro sweep + compare)"
      echo "  ALL : A then C"
      exit 0
      ;;
    *)
      echo "Unknown: $1. Use A | C | C2 | C3 | C4 | C5 | ALL"
      exit 1
      ;;
  esac
  echo "Done (exit 0)"
}

main "$@"
