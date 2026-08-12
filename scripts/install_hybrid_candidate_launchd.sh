#!/bin/bash
# Hybrid_A candidate launchd 등록 (기존 shadow/paper/h8 plist 변경 없음)
set -euo pipefail
PROJECT_ROOT="/Users/jeongminjun/Projects/Can_bit"
PLIST_SRC="${PROJECT_ROOT}/scripts/launchd/com.canbit.daily-hybrid-candidate-ops.plist"
PLIST_DST="${HOME}/Library/LaunchAgents/com.canbit.daily-hybrid-candidate-ops.plist"

chmod +x "${PROJECT_ROOT}/scripts/run_daily_hybrid_candidate_ops.sh"
plutil -lint "${PLIST_SRC}"
launchctl bootout "gui/$(id -u)/com.canbit.daily-hybrid-candidate-ops" 2>/dev/null || \
    launchctl unload "${PLIST_DST}" 2>/dev/null || true
cp "${PLIST_SRC}" "${PLIST_DST}"
launchctl bootstrap "gui/$(id -u)" "${PLIST_DST}" 2>/dev/null || \
    launchctl load "${PLIST_DST}"
echo "=== launchctl list | grep canbit ==="
launchctl list | grep canbit || true
echo "Hybrid candidate plist: ${PLIST_DST}"
