#!/usr/bin/env bash
set -euo pipefail

LABEL="com.canbit.forward_btc_centric_v4_lite_logger"
DST_PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"

launchctl bootout "gui/$(id -u)" "$DST_PLIST" >/dev/null 2>&1 || true
rm -f "$DST_PLIST"
echo "uninstalled ${LABEL}"
