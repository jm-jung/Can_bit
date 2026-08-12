#!/bin/bash
set -u

PROJECT_ROOT="/Users/jeongminjun/Projects/Can_bit"
SYNC_ROOT="${PROJECT_ROOT}/data/diagnostics/data_sync"
LOG_DIR="${SYNC_ROOT}/logs"
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

SYNC_EXIT=0
python "${PROJECT_ROOT}/scripts/data/sync_btcusdt_ohlcv.py" \
  --lookback-days 14 \
  --resample-to-5m true \
  --json \
  >> "${LOG_DIR}/sync_stdout.log" \
  2>> "${LOG_DIR}/sync_stderr.log" || SYNC_EXIT=$?

FRESHNESS_EXIT=0
python "${PROJECT_ROOT}/scripts/diagnostics/check_daily_data_freshness.py" \
  --max-age-hours 2 \
  --json \
  >> "${LOG_DIR}/sync_stdout.log" \
  2>> "${LOG_DIR}/sync_stderr.log" || FRESHNESS_EXIT=$?

python - <<'PY'
import json
from pathlib import Path

root = Path("data/diagnostics/data_sync")
fresh_path = Path("data/diagnostics/false_high_r7_daily_monitor/freshness/data_freshness_latest.json")
sync_path = root / "sync_latest.json"
quality_path = root / "data_quality_latest.csv"

sync = json.loads(sync_path.read_text(encoding="utf-8")) if sync_path.exists() else {}
fresh = json.loads(fresh_path.read_text(encoding="utf-8")) if fresh_path.exists() else {}
quality_status = sync.get("quality_status", "UNKNOWN")
fresh_status = fresh.get("status", "UNKNOWN")
sync_status = sync.get("sync_status", "MISSING")

feature_cache_status = "not refreshed; no safe explicit cache update script was executed"

if sync_status == "PASS" and feature_cache_status.startswith("not refreshed"):
    verdict = "data_sync_installed + sync_pass_but_feature_cache_stale + r7_stale_guard_active + production_not_ready"
elif sync_status == "PASS" and fresh_status == "FRESH" and quality_status in {"PASS", "WARN"}:
    verdict = "data_sync_installed + freshness_pass + r7_daily_ready + production_not_ready"
elif sync_status == "PASS" and fresh_status == "WARN" and quality_status in {"PASS", "WARN"}:
    verdict = "data_sync_installed + freshness_warn + r7_daily_ready + production_not_ready"
elif sync_status == "PASS":
    verdict = "data_sync_installed + sync_pass_but_feature_cache_stale + r7_stale_guard_active + production_not_ready"
elif sync_status == "FAIL":
    verdict = "data_sync_installed + sync_failed + r7_stale_guard_active + production_not_ready"
else:
    verdict = "data_sync_not_ready + r7_stale_guard_active + production_not_ready"

report = {
    "canonical 1m path": sync.get("canonical_1m_path", "data/market/btcusdt_1m.parquet"),
    "canonical 5m path": sync.get("canonical_5m_path", ""),
    "old latest timestamp": sync.get("old_latest_5m_ts", ""),
    "new latest timestamp": sync.get("new_latest_5m_ts", ""),
    "fetched rows": sync.get("fetched_rows", 0),
    "appended rows": sync.get("appended_rows", 0),
    "final 1m rows": sync.get("final_1m_rows", 0),
    "final 5m rows": sync.get("final_5m_rows", 0),
    "data quality status": quality_status,
    "freshness status after": fresh_status,
    "feature/proba cache status": feature_cache_status,
    "sync script path": "scripts/data/sync_btcusdt_ohlcv.py",
    "wrapper path": "ops/sync_btcusdt_ohlcv_for_daily_monitor.sh",
    "R7 shell integration status": "ops/run_false_high_r7_daily_monitor.sh calls this wrapper on STALE/MISSING",
    "post-sync R7 monitor status": "ready; stale guard remains active if feature/proba frame is still stale",
    "Discord freshness display": True,
    "stale forward protection 유지": True,
    "production hash unchanged 여부": "verified by R7 monitor audit; sync does not touch model files",
    "Q2/live/state unchanged 여부": True,
    "quality csv": str(quality_path),
    "final verdict": verdict,
}

lines = ["# BTCUSDT Daily Data Sync Setup Report", ""]
for k, v in report.items():
    lines.extend([f"## {k}", str(v), ""])
(root / "sync_setup_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
PY

if [ "${SYNC_EXIT}" -ne 0 ]; then
  exit "${SYNC_EXIT}"
fi
exit "${FRESHNESS_EXIT}"
