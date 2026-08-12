#!/bin/bash
set -u

LABEL="com.canbit.false_high_r7_daily_monitor"
DEST_PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"

if [ -f "${DEST_PLIST}" ]; then
  launchctl unload "${DEST_PLIST}" >/dev/null 2>&1 || true
  rm -f "${DEST_PLIST}"
  echo "uninstalled ${LABEL}"
else
  echo "not installed: ${LABEL}"
fi
