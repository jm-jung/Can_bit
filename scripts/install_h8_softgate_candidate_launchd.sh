#!/bin/bash
# H8 SoftGate G30 candidate launchd 등록 (기존 shadow/paper/h8-hard plist 변경 없음)
set -euo pipefail
PROJECT_ROOT="/Users/jeongminjun/Projects/Can_bit"
PLIST_SRC="${PROJECT_ROOT}/scripts/launchd/com.canbit.daily-h8-softgate-candidate-ops.plist"
PLIST_DST="${HOME}/Library/LaunchAgents/com.canbit.daily-h8-softgate-candidate-ops.plist"

chmod +x "${PROJECT_ROOT}/scripts/run_daily_h8_softgate_candidate_ops.sh"
plutil -lint "${PLIST_SRC}"
launchctl bootout "gui/$(id -u)/com.canbit.daily-h8-softgate-candidate-ops" 2>/dev/null || \
    launchctl unload "${PLIST_DST}" 2>/dev/null || true
cp "${PLIST_SRC}" "${PLIST_DST}"
launchctl bootstrap "gui/$(id -u)" "${PLIST_DST}" 2>/dev/null || \
    launchctl load "${PLIST_DST}"
echo "=== launchctl list | grep canbit ==="
launchctl list | grep canbit || true
echo "H8 SoftGate candidate plist: ${PLIST_DST}"
