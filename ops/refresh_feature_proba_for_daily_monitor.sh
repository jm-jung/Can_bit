#!/bin/bash
set -u

PROJECT_ROOT="/Users/jeongminjun/Projects/Can_bit"
REFRESH_ROOT="${PROJECT_ROOT}/data/diagnostics/feature_proba_refresh"
LOG_DIR="${REFRESH_ROOT}/logs"
FRESHNESS_DIR="${PROJECT_ROOT}/data/diagnostics/false_high_r7_daily_monitor/freshness"
mkdir -p "${LOG_DIR}" "${FRESHNESS_DIR}"

cd "${PROJECT_ROOT}" || exit 1

if [ -d "${PROJECT_ROOT}/.venv" ]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.venv/bin/activate"
elif [ -d "${PROJECT_ROOT}/venv" ]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/venv/bin/activate"
fi

load_env_file() {
  local env_file="$1"
  if [ -f "${env_file}" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
  fi
}

load_env_file "${PROJECT_ROOT}/.env"
load_env_file "${PROJECT_ROOT}/.env.local"
load_env_file "${PROJECT_ROOT}/config/.env"

REFRESH_EXIT=0
python "${PROJECT_ROOT}/scripts/diagnostics/refresh_daily_feature_proba_cache.py" \
  --lookback-days 30 \
  --json \
  >> "${LOG_DIR}/refresh_stdout.log" \
  2>> "${LOG_DIR}/refresh_stderr.log" || REFRESH_EXIT=$?

FRESHNESS_EXIT=0
python "${PROJECT_ROOT}/scripts/diagnostics/check_daily_data_freshness.py" \
  --max-age-hours 2 \
  --json \
  >> "${LOG_DIR}/refresh_stdout.log" \
  2>> "${LOG_DIR}/refresh_stderr.log" || FRESHNESS_EXIT=$?

if [ "${REFRESH_EXIT}" -ne 0 ]; then
  exit "${REFRESH_EXIT}"
fi
exit "${FRESHNESS_EXIT}"
