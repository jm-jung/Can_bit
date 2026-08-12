#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.canbit.forward_btc_centric_v4_lite_logger"
SRC_PLIST="$ROOT_DIR/ops/launchd/${LABEL}.plist"
DST_PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"

cd "$ROOT_DIR"
python scripts/diagnostics/run_forward_btc_centric_v4_lite_logger.py --dry-run --json
python scripts/diagnostics/run_forward_btc_centric_v4_lite_logger.py --once --json

mkdir -p "$HOME/Library/LaunchAgents"
cp "$SRC_PLIST" "$DST_PLIST"
launchctl bootout "gui/$(id -u)" "$DST_PLIST" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$(id -u)" "$DST_PLIST"
launchctl enable "gui/$(id -u)/$LABEL"
launchctl print "gui/$(id -u)/$LABEL"
