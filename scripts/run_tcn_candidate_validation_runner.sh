#!/usr/bin/env bash
# TCN 후보 검증 실행기. OMP/MKL 스레드 제한으로 세그폴트(exit 139) 방지.
# 사용: ./scripts/run_tcn_candidate_validation_runner.sh --id h15_t0p004 --min-max-proba 0.55 ... [기타 인자]
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if [[ ! -d ".venv" ]]; then
  echo "ERROR: .venv not found. Run: cd ${PROJECT_ROOT} && python -m venv .venv && source .venv/bin/activate"
  exit 1
fi
source .venv/bin/activate
exec python -m scripts.run_tcn_candidate_validation "$@"
