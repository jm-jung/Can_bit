#!/bin/bash
# Paper launchd job 등록 (Shadow plist / com.canbit.daily 건드리지 않음)
set -euo pipefail
PROJECT_ROOT="/Users/jeongminjun/Projects/Can_bit"
PLIST_SRC="${PROJECT_ROOT}/scripts/launchd/com.canbit.daily-paper-ops.plist"
PLIST_DST="${HOME}/Library/LaunchAgents/com.canbit.daily-paper-ops.plist"

plutil -lint "${PLIST_SRC}"
launchctl bootout "gui/$(id -u)/com.canbit.daily-paper-ops" 2>/dev/null || \
    launchctl unload "${PLIST_DST}" 2>/dev/null || true
cp "${PLIST_SRC}" "${PLIST_DST}"
launchctl bootstrap "gui/$(id -u)" "${PLIST_DST}" 2>/dev/null || \
    launchctl load "${PLIST_DST}"
echo "=== launchctl list | grep canbit ==="
launchctl list | grep canbit || true
echo "Paper plist: ${PLIST_DST}"
