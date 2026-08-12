#!/bin/bash
# Can_bit 일일 운영 자동화 스크립트
# 
# 실행 순서:
# 1. OHLCV 업데이트
# 2. FR2 meta_metrics (--force-latest, OHLCV 직후)
# 3. FR2 meta_layer → state_log_legacy_*.csv (다음 실행용 레거시 갱신; 운영 state_log.csv는 덮어쓰지 않음)
# 4. Paper/Shadow 실행
# 5. 주간 리포트 생성 (월요일만)
# 6. SIGNAL_EXIT_TH 리포트 등
# 7. Production Shadow Ops (확정 파이프라인 배치 + 별도 Discord; 같은 13시 job에서 연속 전송)
#
# 실패 시 즉시 종료 (exit code != 0)

set -euo pipefail

# 에러 발생 시 Discord 알림을 위한 trap 설정
error_handler() {
    local exit_code=$?
    local line_number=$1
    local command=$2
    
    log_error "스크립트 실패: line ${line_number}, command: ${command}, exit code: ${exit_code}"
    
    # Discord ERROR 알림 전송 (실패해도 스크립트 종료는 계속 진행)
    # VENV_PYTHON과 LOG_FILE이 정의되어 있는지 확인
    if [[ -n "${VENV_PYTHON:-}" && -n "${LOG_FILE:-}" ]]; then
        "${VENV_PYTHON}" -c "
from src.monitoring.notify_discord import send_discord_message
send_discord_message(
    '❌ Can_bit 일일 운영 실패',
    f'**실패 단계**: line ${line_number}\\n**실행 명령**: ${command}\\n**종료 코드**: ${exit_code}\\n**로그 파일**: ${LOG_FILE}\\n\\n로그 파일을 확인하여 상세 원인을 파악하세요.',
    'ERROR'
)
" 2>&1 | tee -a "${LOG_FILE}" || true
    fi
    
    exit ${exit_code}
}

trap 'error_handler ${LINENO} "${BASH_COMMAND}"' ERR

# 프로젝트 루트 경로 (스크립트 위치 기준)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# .env 파일 로드 (Discord Webhook URL 등)
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a  # 자동으로 export
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# 가상환경 Python 경로
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

# 로그 디렉토리
LOG_DIR="${PROJECT_ROOT}/data/ops_logs"
mkdir -p "${LOG_DIR}"

# 날짜 기반 로그 파일
DATE_STR=$(date +%Y-%m-%d)
LOG_FILE="${LOG_DIR}/daily_run_${DATE_STR}.log"

# 로그 함수
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "${LOG_FILE}" >&2
}

# 실행 전 점검
log "=== Can_bit 일일 운영 시작 ==="
log "프로젝트 루트: ${PROJECT_ROOT}"
log "로그 파일: ${LOG_FILE}"

# .venv/bin/python 존재 확인
if [[ ! -f "${VENV_PYTHON}" ]]; then
    log_error ".venv/bin/python이 존재하지 않습니다: ${VENV_PYTHON}"
    exit 1
fi
log "Python 경로 확인: ${VENV_PYTHON}"

# 필수 모듈 점검
log "필수 모듈 점검 중..."
if ! "${VENV_PYTHON}" -c "import pandas, ccxt" 2>&1 | tee -a "${LOG_FILE}"; then
    log_error "필수 모듈(pandas, ccxt) 임포트 실패"
    exit 1
fi
log "필수 모듈 점검 완료"

# (1) OHLCV 업데이트
log "=== [1/7] OHLCV 업데이트 시작 ==="
UPDATE_OUTPUT=$("${VENV_PYTHON}" -m src.data.update_ohlcv \
    --symbol BTCUSDT \
    --timeframe 5m \
    --end now \
    2>&1) || UPDATE_EXIT=$?
echo "${UPDATE_OUTPUT}" | tee -a "${LOG_FILE}"

# update_ohlcv는 이미 최신 데이터인 경우에도 정상 종료하므로,
# 에러가 발생했는지 확인 (실제 문제인 경우만 실패 처리)
if [[ -n "${UPDATE_EXIT:-}" ]] && [[ "${UPDATE_EXIT}" != "0" ]]; then
    # 에러 메시지에 "Fetched data but no new rows"가 있고 "all are duplicates"가 없으면 실제 문제
    if echo "${UPDATE_OUTPUT}" | grep -q "Fetched data but no new rows" && \
       ! echo "${UPDATE_OUTPUT}" | grep -q "all are duplicates"; then
        log_error "OHLCV 업데이트 실패 (실제 문제 감지)"
        exit 1
    else
        # 중복 데이터는 정상 상황
        log "OHLCV 업데이트: 모든 데이터가 중복 (이미 최신 상태)"
    fi
fi
log "=== [1/7] OHLCV 업데이트 완료 ==="

# (2) FR2 meta_metrics: OHLCV 직후 (실패해도 파이프라인 계속)
log "=== [2/7] FR2 meta_metrics refresh (--force-latest) 시작 ==="
set +e
"${VENV_PYTHON}" scripts/refresh_fr2_meta_metrics.py --force-latest 2>&1 | tee -a "${LOG_FILE}"
META_REFRESH_EXIT=$?
set -e
if [[ "${META_REFRESH_EXIT}" -eq 0 ]]; then
    log "FR2 meta_metrics refresh 완료"
else
    log_error "FR2 meta_metrics refresh 실패 (exit=${META_REFRESH_EXIT}, 파이프라인 계속)"
fi
log "=== [2/7] FR2 meta_metrics refresh 종료 ==="

# (3) FR2 meta_layer 오프라인 CSV (legacy; 이후 일정용 rolling 소스)
log "=== [3/7] FR2 meta_layer (legacy CSV) 시작 ==="
META_LEGACY_OUT="${PROJECT_ROOT}/data/diagnostics/fr2/state_log_legacy_${DATE_STR}.csv"
set +e
"${VENV_PYTHON}" scripts/run_fr2_meta_layer.py --output "${META_LEGACY_OUT}" 2>&1 | tee -a "${LOG_FILE}"
META_LAYER_EXIT=$?
set -e
if [[ "${META_LAYER_EXIT}" -eq 0 ]]; then
    log "FR2 meta_layer legacy 작성: ${META_LEGACY_OUT}"
else
    log_error "FR2 meta_layer 실패 (exit=${META_LAYER_EXIT}, 파이프라인 계속)"
fi
log "=== [3/7] FR2 meta_layer 종료 ==="

# (4) Paper/Shadow 실행
log "=== [4/7] Paper/Shadow 실행 시작 ==="
if ! "${VENV_PYTHON}" -m src.monitoring.run_paper_shadow \
    --symbol BTCUSDT \
    --timeframe 5m \
    --lookback-days 30 \
    --update-data 0 \
    --force-refresh-proba 1 \
    2>&1 | tee -a "${LOG_FILE}"; then
    log_error "Paper/Shadow 실행 실패"
    exit 1
fi
log "=== [4/7] Paper/Shadow 실행 완료 ==="

# (5) 주간 리포트 생성 (월요일만)
DAY_OF_WEEK=$(date +%u)  # 1=월요일, 7=일요일
if [[ "${DAY_OF_WEEK}" == "1" ]]; then
    log "=== [5/7] 주간 리포트 생성 시작 (월요일) ==="
    
    # 주간 범위 계산: 지난 주 월요일 ~ 지난 주 일요일 (또는 오늘이 월요일이면 지난 주 전체)
    TODAY=$(date +%Y-%m-%d)
    
    # 지난 주 월요일 계산 (오늘에서 7일 빼고, 그 주의 월요일)
    # macOS date 명령어 사용: -v-7d로 7일 전으로 가고, 그 주의 월요일 계산
    # 또는 간단히: 오늘이 월요일이면 지난 주 월요일은 7일 전
    START_DATE=$(date -v-7d +%Y-%m-%d)
    # 지난 주 일요일은 어제 (오늘이 월요일이므로)
    END_DATE=$(date -v-1d +%Y-%m-%d)
    
    log "주간 리포트 범위: ${START_DATE} ~ ${END_DATE} (지난 주)"
    
    if ! "${VENV_PYTHON}" -m src.monitoring.generate_weekly_report \
        --start "${START_DATE}" \
        --end "${END_DATE}" \
        2>&1 | tee -a "${LOG_FILE}"; then
        log_error "주간 리포트 생성 실패"
        exit 1
    fi
    log "=== [5/7] 주간 리포트 생성 완료 ==="
else
    log "=== [5/7] 주간 리포트 생성 건너뜀 (월요일 아님, 요일: ${DAY_OF_WEEK}) ==="
fi

# (6) SIGNAL_EXIT_TH conditional/deployment 리포트 생성 (daily, non-blocking)
log "=== [6/7] SIGNAL_EXIT_TH 리포트 생성 시작 ==="
set +e
"${VENV_PYTHON}" scripts/signal_exit_th_conditional_deployment_report.py 2>&1 | tee -a "${LOG_FILE}"
COND_EXIT=$?
if [[ "${COND_EXIT}" == "0" ]]; then
    log "conditional deployment report 생성 성공"
else
    log_error "conditional deployment report 생성 실패 (exit=${COND_EXIT})"
fi

"${VENV_PYTHON}" scripts/signal_exit_th_deployment_decision.py 2>&1 | tee -a "${LOG_FILE}"
DEC_EXIT=$?
if [[ "${DEC_EXIT}" == "0" ]]; then
    log "deployment decision report 생성 성공"
else
    log_error "deployment decision report 생성 실패 (exit=${DEC_EXIT})"
fi
set -e
log "=== [6/7] SIGNAL_EXIT_TH 리포트 생성 완료 ==="

log "=== Can_bit 일일 운영 완료 (DONE) ==="

# 정상 종료 시 Discord 일일 리포트 전송
LATEST_SUMMARY=$(ls -t "${PROJECT_ROOT}/data/monitoring/monitor_guard_stage2_summary_"*.json 2>/dev/null | head -1 || echo "")

# OHLCV 정보 수집
OHLCV_INFO=$("${VENV_PYTHON}" -c "
import json
import sys
from pathlib import Path
from datetime import datetime

try:
    # OHLCV CSV에서 최신 timestamp 추출
    ohlcv_path = Path('${PROJECT_ROOT}') / 'data' / 'ohlcv' / 'BTCUSDT_5m_full.csv'
    if ohlcv_path.exists():
        import pandas as pd
        df = pd.read_csv(ohlcv_path)
        if len(df) > 0 and 'timestamp' in df.columns:
            # 최신 timestamp
            latest_ts = df['timestamp'].iloc[-1]
            
            # 오늘 날짜 기준으로 필터링하여 오늘 추가된 캔들 수 추정
            today_str = datetime.now().strftime('%Y-%m-%d')
            today_df = df[df['timestamp'].str.contains(today_str, na=False)]
            new_candles = len(today_df) if len(today_df) > 0 else 0
        else:
            latest_ts = 'N/A'
            new_candles = 0
    else:
        latest_ts = 'N/A'
        new_candles = 0
    
    info = {
        'symbol': 'BTCUSDT',
        'timeframe': '5m',
        'latest_ts': str(latest_ts),
        'new_candles': new_candles
    }
    print(json.dumps(info))
except Exception as e:
    print(json.dumps({'symbol': 'BTCUSDT', 'timeframe': '5m', 'latest_ts': 'N/A', 'new_candles': 0}))
" 2>/dev/null || echo '{"symbol":"BTCUSDT","timeframe":"5m","latest_ts":"N/A","new_candles":0}')

# 주간 리포트 경로 (월요일인 경우)
WEEKLY_REPORT=""
if [[ "${DAY_OF_WEEK}" == "1" ]]; then
    WEEKLY_REPORT=$(ls -t "${PROJECT_ROOT}/data/monitoring_reports/weekly_report_"*.md 2>/dev/null | head -1 || echo "")
fi

# 예측 캐시 경로 계산 (현재 전략: ml_tcn)
CACHE_PATH="${PROJECT_ROOT}/data/cache/ml_predictions/ml_tcn_BTCUSDT_5m_proba.parquet"
if [[ ! -f "${CACHE_PATH}" ]]; then
    log "경고: 예측 캐시 파일이 없습니다: ${CACHE_PATH}"
fi

# Discord 일일 리포트 전송 (실패해도 스크립트는 정상 종료)
"${VENV_PYTHON}" -c "
import json
import sys
from src.monitoring.notify_discord import send_daily_report

summary_path = '${LATEST_SUMMARY}' if '${LATEST_SUMMARY}' else None
ohlcv_info_str = '''${OHLCV_INFO}'''
weekly_report_path = '${WEEKLY_REPORT}' if '${WEEKLY_REPORT}' else None
cache_path = '${CACHE_PATH}'

try:
    ohlcv_info = json.loads(ohlcv_info_str)
except:
    ohlcv_info = {}

send_daily_report(
    summary_json_path=summary_path,
    ohlcv_info=ohlcv_info,
    weekly_report_path=weekly_report_path,
    pipeline_status='정상 완료',
    cache_path=cache_path
)
" 2>&1 | tee -a "${LOG_FILE}" || true

# (7) Production Shadow Ops — 일일 파이프라인과 별도 측정, 같은 웹훅으로 두 번째 embed 전송
log "=== [7/7] Production Shadow Ops (배치 + Discord) 시작 ==="
set +e
bash "${PROJECT_ROOT}/scripts/run_daily_shadow_ops.sh" 2>&1 | tee -a "${LOG_FILE}"
SHADOW_OPS_EXIT=$?
set -e
if [[ "${SHADOW_OPS_EXIT}" -eq 0 ]]; then
    log "=== [7/7] Production Shadow Ops 완료 ==="
else
    log_error "Production Shadow Ops 실패 (exit=${SHADOW_OPS_EXIT}, 일일 파이프라인은 이미 완료됨)"
fi

exit 0
