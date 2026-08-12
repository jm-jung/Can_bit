#!/bin/bash
set -u

PROJECT_ROOT="/Users/jeongminjun/Projects/Can_bit"
LABEL="com.canbit.false_high_r7_daily_monitor"
SRC_PLIST="${PROJECT_ROOT}/ops/launchd/${LABEL}.plist"
DEST_DIR="${HOME}/Library/LaunchAgents"
DEST_PLIST="${DEST_DIR}/${LABEL}.plist"
REPORT_DIR="${PROJECT_ROOT}/data/diagnostics/false_high_r7_daily_monitor/launchd"
REPORT_PATH="${REPORT_DIR}/launchd_install_report.md"

mkdir -p "${DEST_DIR}" "${REPORT_DIR}" "${PROJECT_ROOT}/data/diagnostics/false_high_r7_daily_monitor/logs"

STATUS="PASS"
DETAIL="installed"

if [ ! -f "${SRC_PLIST}" ]; then
  STATUS="FAIL"
  DETAIL="source plist missing"
else
  cp "${SRC_PLIST}" "${DEST_PLIST}" || { STATUS="FAIL"; DETAIL="copy failed"; }
fi

if [ "${STATUS}" = "PASS" ]; then
  launchctl unload "${DEST_PLIST}" >/dev/null 2>&1 || true
  if ! launchctl load "${DEST_PLIST}" >/dev/null 2>&1; then
    STATUS="WARN"
    DETAIL="copied but launchctl load failed"
  fi
fi

LIST_RESULT="$(launchctl list 2>/dev/null | grep "${LABEL}" || true)"

cat > "${REPORT_PATH}" <<EOF
# R7 Daily Monitor Launchd Install Report

## status
${STATUS}

## detail
${DETAIL}

## label
${LABEL}

## source_plist
${SRC_PLIST}

## installed_plist
${DEST_PLIST}

## launchctl_list_contains_label
$(if [ -n "${LIST_RESULT}" ]; then echo "true"; else echo "false"; fi)

## schedule
13:00 local macOS time / intended Asia-Seoul operational schedule

## diagnostics_only
true

## production_launchd_changed
false

## manual_test_commands
\`\`\`bash
bash ops/run_false_high_r7_daily_monitor.sh
python scripts/diagnostics/run_false_high_r7_daily_monitor.py --dry-run --no-discord
python scripts/diagnostics/run_false_high_r7_daily_monitor.py --discord --force-send
\`\`\`
EOF

echo "launchd install status: ${STATUS} (${DETAIL})"
echo "report: ${REPORT_PATH}"

if [ "${STATUS}" = "FAIL" ]; then
  exit 1
fi
exit 0
