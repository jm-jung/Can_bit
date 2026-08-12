#!/bin/bash
# =============================================================================
# Can_bit Hybrid_A Candidate Daily Ops — launchd / 수동 실행 공통
# Diagnostic monitor only. No live orders, no paper/shadow state changes.
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/Users/jeongminjun/Projects/Can_bit"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/run_daily_hybrid_candidate_ops.sh"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
ENV_FILE="${PROJECT_ROOT}/.env"
LOG_DIR="${PROJECT_ROOT}/data/ops_logs"
LAUNCHD_STDOUT="${LOG_DIR}/launchd_hybrid_candidate_stdout.log"
LAUNCHD_STDERR="${LOG_DIR}/launchd_hybrid_candidate_stderr.log"

cd "${PROJECT_ROOT}" || {
    echo "FATAL: cannot cd to ${PROJECT_ROOT}" >&2
    exit 1
}

mkdir -p "${LOG_DIR}"

DATE_STR=$(date +%Y-%m-%d)
TIME_STR=$(date +%H%M%S)
LOG_FILE="${LOG_DIR}/hybrid_candidate_ops_${DATE_STR}_${TIME_STR}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}" | tee -a "${LAUNCHD_STDOUT}"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "${LOG_FILE}" | tee -a "${LAUNCHD_STDERR}" >&2
}

error_handler() {
    local exit_code=$?
    local line_number=$1
    local command=$2
    log_error "Hybrid Candidate Ops 실패: line=${line_number} cmd=${command} exit=${exit_code}"
    exit "${exit_code}"
}

trap 'error_handler ${LINENO} "${BASH_COMMAND}"' ERR

log "=== Can_bit Hybrid_A Candidate Daily Ops 시작 ==="
log "PROJECT_ROOT=${PROJECT_ROOT}"
log "SCRIPT=${SCRIPT_PATH}"
log "PID=$$ PPID=$PPID USER=${USER:-unknown}"

if [[ ! -x "${VENV_PYTHON}" ]]; then
    log_error ".venv python 없음: ${VENV_PYTHON}"
    exit 1
fi

if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
    set +a

# CANBIT_NOTIFICATION_CLEANUP_BEGIN
CANBIT_NOTIFICATION_MODE="${CANBIT_NOTIFICATION_MODE:-muted}"
if [[ "${CANBIT_NOTIFICATION_MODE}" == "muted" ]]; then
    unset DISCORD_WEBHOOK_URL
    unset CANBIT_DISCORD_WEBHOOK_URL
    unset WEBHOOK_URL
fi
# CANBIT_NOTIFICATION_CLEANUP_END
    if [[ -n "${DISCORD_WEBHOOK_URL:-}" ]]; then
        log "DISCORD_WEBHOOK_URL: loaded (set)"
    else
        log_error "DISCORD_WEBHOOK_URL: missing in .env"
    fi
else
    log_error ".env 없음: ${ENV_FILE}"
fi

log "=== Hybrid_A monitor (scripts.run_daily_hybrid_candidate) ==="
set +e
"${VENV_PYTHON}" -m scripts.run_daily_hybrid_candidate 2>&1 | tee -a "${LOG_FILE}" | tee -a "${LAUNCHD_STDOUT}"
HY_EXIT=${PIPESTATUS[0]}
set -e

if [[ "${HY_EXIT}" -eq 0 ]]; then
    log "=== Hybrid Candidate Ops 완료 (exit=0) ==="
else
    log_error "=== Hybrid Candidate Ops 실패 (exit=${HY_EXIT}) ==="
    exit "${HY_EXIT}"
fi

LATEST_JSON=$(ls -t "${PROJECT_ROOT}"/data/monitoring/hybrid_candidate_daily_report_*.json 2>/dev/null | head -1 || true)
if [[ -n "${LATEST_JSON}" ]]; then
    log "최신 JSON: ${LATEST_JSON}"
fi

log "=== Can_bit Hybrid_A Candidate Daily Ops DONE ==="
exit 0
