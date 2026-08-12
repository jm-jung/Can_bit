#!/usr/bin/env bash
set -euo pipefail

LABEL="com.canbit.forward_research_v4_orderflow_relative_logger"
DST_PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"

launchctl bootout "gui/$(id -u)" "$DST_PLIST" >/dev/null 2>&1 || true
rm -f "$DST_PLIST"
echo "uninstalled ${LABEL}"
