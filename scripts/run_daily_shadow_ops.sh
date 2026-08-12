#!/bin/bash
# =============================================================================
# Can_bit Shadow Daily Ops - 자동 실행 스크립트
#
# 기능:
#   1. 프로젝트 루트 이동 + .venv 활성화
#   2. Shadow 모드 1회 배치 실행 (run_daily_shadow.py)
#   3. 결과 리포트 생성 (JSON + MD)
#   4. Discord webhook으로 성공/실패 요약 전송
#   5. 에러 시 별도 에러 로그 저장
#
# 사용법:
#   bash scripts/run_daily_shadow_ops.sh
#
# 주의: live/paper 모드는 절대 실행하지 않음. shadow 전용.
# =============================================================================

set -euo pipefail

# 프로젝트 루트 (스크립트 위치 기준)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# .env 로드 (DISCORD_WEBHOOK_URL 등)
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# 가상환경 Python
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

# 로그 디렉토리
LOG_DIR="${PROJECT_ROOT}/data/ops_logs"
mkdir -p "${LOG_DIR}"

# 날짜/시각 기반 로그
DATE_STR=$(date +%Y-%m-%d)
TIME_STR=$(date +%H%M%S)
LOG_FILE="${LOG_DIR}/shadow_ops_${DATE_STR}_${TIME_STR}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "${LOG_FILE}" >&2
}

# 에러 핸들러
error_handler() {
    local exit_code=$?
    local line_number=$1
    local command=$2

    log_error "Shadow Ops 실패: line ${line_number}, command: ${command}, exit code: ${exit_code}"

    # Discord 에러 알림 (best effort)
    if [[ -n "${VENV_PYTHON:-}" ]]; then
        "${VENV_PYTHON}" -c "
from src.monitoring.notify_discord import send_discord_message
send_discord_message(
    '【Shadow Ops】Shadow Daily Ops FAILED',
    '**실패 line**: ${line_number}\n**명령**: ${command}\n**exit code**: ${exit_code}\n**로그**: ${LOG_FILE}',
    'ERROR'
)
" 2>&1 | tee -a "${LOG_FILE}" || true
    fi

    exit ${exit_code}
}

trap 'error_handler ${LINENO} "${BASH_COMMAND}"' ERR

# ─── 실행 시작 ───

log "=== Can_bit Shadow Daily Ops 시작 ==="
log "프로젝트 루트: ${PROJECT_ROOT}"
log "로그 파일: ${LOG_FILE}"

# venv 확인
if [[ ! -f "${VENV_PYTHON}" ]]; then
    log_error ".venv/bin/python을 찾을 수 없습니다: ${VENV_PYTHON}"
    exit 1
fi
log "Python 확인: ${VENV_PYTHON}"

# 필수 모듈 확인
if ! "${VENV_PYTHON}" -c "import pandas, numpy" 2>&1 | tee -a "${LOG_FILE}"; then
    log_error "필수 모듈(pandas, numpy) 임포트 실패"
    exit 1
fi
log "필수 모듈 확인 완료"

# Shadow 배치 실행
log "=== Shadow 배치 실행 시작 ==="
"${VENV_PYTHON}" -m scripts.run_daily_shadow 2>&1 | tee -a "${LOG_FILE}"
SHADOW_EXIT=$?

if [[ "${SHADOW_EXIT}" -eq 0 ]]; then
    log "=== Shadow 배치 실행 완료 (성공) ==="
else
    log_error "Shadow 배치 실행 실패 (exit=${SHADOW_EXIT})"
    exit ${SHADOW_EXIT}
fi

# 결과 파일 확인
LATEST_JSON=$(ls -t "${PROJECT_ROOT}/data/monitoring/shadow_daily_report_"*.json 2>/dev/null | head -1 || echo "")
LATEST_MD=$(ls -t "${PROJECT_ROOT}/data/monitoring/shadow_daily_report_"*.md 2>/dev/null | head -1 || echo "")

if [[ -n "${LATEST_JSON}" ]]; then
    log "최신 JSON 리포트: ${LATEST_JSON}"
else
    log "경고: JSON 리포트를 찾을 수 없음"
fi

if [[ -n "${LATEST_MD}" ]]; then
    log "최신 MD 리포트: ${LATEST_MD}"
else
    log "경고: MD 리포트를 찾을 수 없음"
fi

log "=== Can_bit Shadow Daily Ops 완료 (DONE) ==="

exit 0
