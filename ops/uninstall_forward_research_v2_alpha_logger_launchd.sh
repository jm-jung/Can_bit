#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.canbit.forward_research_v2_alpha_logger"
DEST="$HOME/Library/LaunchAgents/${LABEL}.plist"
REPORT_DIR="$ROOT_DIR/data/diagnostics/forward_research_v2_alpha_logger/install"
mkdir -p "$REPORT_DIR"

launchctl bootout "gui/$(id -u)" "$DEST" >/dev/null 2>&1 || true
rm -f "$DEST"
launchctl print "gui/$(id -u)/$LABEL" > "$REPORT_DIR/uninstall_status.txt" 2>&1 || true

cat > "$REPORT_DIR/uninstall_summary.md" <<EOF
# Forward Research V2 Alpha Logger Uninstall

- label: $LABEL
- removed plist: $DEST
- production launchd modified: false
EOF
