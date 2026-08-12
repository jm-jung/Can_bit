#!/bin/bash
# Meta research daily launchd 등록 (기존 production/paper/quality plist 변경 없음)
set -euo pipefail
PROJECT_ROOT="/Users/jeongminjun/Projects/Can_bit"
PLIST_SRC="${PROJECT_ROOT}/scripts/launchd/com.canbit.meta_research_daily.plist"
PLIST_DST="${HOME}/Library/LaunchAgents/com.canbit.meta_research_daily.plist"

chmod +x "${PROJECT_ROOT}/scripts/run_daily_meta_research_ops.sh"
plutil -lint "${PLIST_SRC}"
launchctl bootout "gui/$(id -u)/com.canbit.meta_research_daily" 2>/dev/null || \
    launchctl unload "${PLIST_DST}" 2>/dev/null || true
cp "${PLIST_SRC}" "${PLIST_DST}"
launchctl bootstrap "gui/$(id -u)" "${PLIST_DST}" 2>/dev/null || \
    launchctl load "${PLIST_DST}"
echo "=== launchctl list | grep meta_research ==="
launchctl list | grep meta_research || true
echo "Meta research plist: ${PLIST_DST}"
echo "Schedule: daily 13:00 local — research-only pipeline"
