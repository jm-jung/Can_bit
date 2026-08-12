#!/bin/bash
set -u

PROJECT_ROOT="/Users/jeongminjun/Projects/Can_bit"
OUTPUT_ROOT="${PROJECT_ROOT}/data/diagnostics/false_high_r7_daily_monitor"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

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

FRESHNESS_DIR="${OUTPUT_ROOT}/freshness"
FRESHNESS_JSON="${FRESHNESS_DIR}/data_freshness_latest.json"
mkdir -p "${FRESHNESS_DIR}"

python "${PROJECT_ROOT}/scripts/diagnostics/check_daily_data_freshness.py" \
  --max-age-hours 2 \
  --json \
  >> "${LOG_DIR}/r7_daily_monitor_stdout.log" \
  2>> "${LOG_DIR}/r7_daily_monitor_stderr.log" || true

FRESHNESS_STATUS="$(python - <<'PY'
import json
from pathlib import Path
p = Path("data/diagnostics/false_high_r7_daily_monitor/freshness/data_freshness_latest.json")
if not p.exists():
    print("MISSING")
else:
    try:
        print(json.loads(p.read_text()).get("status", "UNKNOWN"))
    except Exception:
        print("ERROR")
PY
)"
OHLCV_STATUS="$(python - <<'PY'
import json
from pathlib import Path
p = Path("data/diagnostics/false_high_r7_daily_monitor/freshness/data_freshness_latest.json")
if not p.exists():
    print("MISSING")
else:
    try:
        d = json.loads(p.read_text())
        print(d.get("ohlcv_freshness_status", d.get("status", "UNKNOWN")))
    except Exception:
        print("ERROR")
PY
)"

SYNC_EXECUTED="false"
SYNC_STATUS="not_needed"
POST_SYNC_FRESHNESS_STATUS="${FRESHNESS_STATUS}"
if [ "${OHLCV_STATUS}" = "STALE" ] || [ "${OHLCV_STATUS}" = "MISSING" ] || [ "${FRESHNESS_STATUS}" = "OHLCV_STALE" ]; then
  SYNC_EXECUTED="true"
  SYNC_STATUS="ran:ops/sync_btcusdt_ohlcv_for_daily_monitor.sh"
  bash "${PROJECT_ROOT}/ops/sync_btcusdt_ohlcv_for_daily_monitor.sh" \
    >> "${LOG_DIR}/r7_daily_monitor_stdout.log" \
    2>> "${LOG_DIR}/r7_daily_monitor_stderr.log" || SYNC_STATUS="failed:ops/sync_btcusdt_ohlcv_for_daily_monitor.sh"

  python "${PROJECT_ROOT}/scripts/diagnostics/check_daily_data_freshness.py" \
    --max-age-hours 2 \
    --json \
    >> "${LOG_DIR}/r7_daily_monitor_stdout.log" \
    2>> "${LOG_DIR}/r7_daily_monitor_stderr.log" || true
  POST_SYNC_FRESHNESS_STATUS="$(python - <<'PY'
import json
from pathlib import Path
p = Path("data/diagnostics/false_high_r7_daily_monitor/freshness/data_freshness_latest.json")
if not p.exists():
    print("MISSING")
else:
    try:
        print(json.loads(p.read_text()).get("status", "UNKNOWN"))
    except Exception:
        print("ERROR")
PY
)"
fi

REFRESH_EXECUTED="false"
REFRESH_STATUS="not_needed"
FEATURE_STATUS="$(python - <<'PY'
import json
from pathlib import Path
p = Path("data/diagnostics/false_high_r7_daily_monitor/freshness/data_freshness_latest.json")
if not p.exists():
    print("MISSING")
else:
    try:
        d = json.loads(p.read_text())
        vals = [
            d.get("feature_freshness_status", "UNKNOWN"),
            d.get("proba_freshness_status", "UNKNOWN"),
            d.get("q2_freshness_status", "UNKNOWN"),
            d.get("r7_input_freshness_status", "UNKNOWN"),
            d.get("overall_status", d.get("status", "UNKNOWN")),
        ]
        stale = {"STALE", "MISSING", "ERROR", "FEATURE_CACHE_STALE", "PROBA_CACHE_MISSING", "Q2_CACHE_MISSING", "R7_INPUT_STALE"}
        print("STALE" if any(v in stale for v in vals) else "FRESH")
    except Exception:
        print("ERROR")
PY
)"
POST_REFRESH_FRESHNESS_STATUS="${POST_SYNC_FRESHNESS_STATUS}"
if [ "${FEATURE_STATUS}" = "STALE" ] || [ "${FEATURE_STATUS}" = "MISSING" ] || [ "${FEATURE_STATUS}" = "ERROR" ]; then
  REFRESH_EXECUTED="true"
  REFRESH_STATUS="ran:ops/refresh_feature_proba_for_daily_monitor.sh"
  bash "${PROJECT_ROOT}/ops/refresh_feature_proba_for_daily_monitor.sh" \
    >> "${LOG_DIR}/r7_daily_monitor_stdout.log" \
    2>> "${LOG_DIR}/r7_daily_monitor_stderr.log" || REFRESH_STATUS="failed:ops/refresh_feature_proba_for_daily_monitor.sh"

  python "${PROJECT_ROOT}/scripts/diagnostics/check_daily_data_freshness.py" \
    --max-age-hours 2 \
    --json \
    >> "${LOG_DIR}/r7_daily_monitor_stdout.log" \
    2>> "${LOG_DIR}/r7_daily_monitor_stderr.log" || true
  POST_REFRESH_FRESHNESS_STATUS="$(python - <<'PY'
import json
from pathlib import Path
p = Path("data/diagnostics/false_high_r7_daily_monitor/freshness/data_freshness_latest.json")
if not p.exists():
    print("MISSING")
else:
    try:
        print(json.loads(p.read_text()).get("status", "UNKNOWN"))
    except Exception:
        print("ERROR")
PY
)"
fi

cat > "${FRESHNESS_DIR}/sync_guard_latest.json" <<EOF
{
  "freshness_status_before": "${FRESHNESS_STATUS}",
  "sync_executed": ${SYNC_EXECUTED},
  "sync_status": "${SYNC_STATUS}",
  "post_sync_freshness_status": "${POST_SYNC_FRESHNESS_STATUS}",
  "refresh_executed": ${REFRESH_EXECUTED},
  "refresh_status": "${REFRESH_STATUS}",
  "post_refresh_freshness_status": "${POST_REFRESH_FRESHNESS_STATUS}",
  "production_changed": false,
  "q2_changed": false,
  "r7_action": "none"
}
EOF

if [ -n "${DISCORD_WEBHOOK_URL:-}" ] || [ -n "${CANBIT_DISCORD_WEBHOOK_URL:-}" ]; then
  DISCORD_FLAG="--discord"
else
  DISCORD_FLAG="--no-discord"
fi

python "${PROJECT_ROOT}/scripts/diagnostics/run_false_high_r7_daily_monitor.py" \
  ${DISCORD_FLAG} \
  --freshness-json "${FRESHNESS_JSON}" \
  --output-root "data/diagnostics/false_high_r7_daily_monitor" \
  >> "${LOG_DIR}/r7_daily_monitor_stdout.log" \
  2>> "${LOG_DIR}/r7_daily_monitor_stderr.log"
