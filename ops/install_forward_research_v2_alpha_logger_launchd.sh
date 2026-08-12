#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.canbit.forward_research_v2_alpha_logger"
SRC="$ROOT_DIR/ops/launchd/${LABEL}.plist"
DEST="$HOME/Library/LaunchAgents/${LABEL}.plist"
REPORT_DIR="$ROOT_DIR/data/diagnostics/forward_research_v2_alpha_logger/install"
mkdir -p "$REPORT_DIR" "$HOME/Library/LaunchAgents"

cd "$ROOT_DIR"
python -m py_compile scripts/diagnostics/run_forward_research_v2_alpha_logger.py
python scripts/diagnostics/run_forward_research_v2_alpha_logger.py --dry-run --json > "$REPORT_DIR/install_dry_run.json"
python scripts/diagnostics/run_forward_research_v2_alpha_logger.py --once --json > "$REPORT_DIR/install_one_shot.json"

cp "$SRC" "$DEST"
launchctl bootout "gui/$(id -u)" "$DEST" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$(id -u)" "$DEST"
launchctl enable "gui/$(id -u)/$LABEL" || true
launchctl print "gui/$(id -u)/$LABEL" > "$REPORT_DIR/launchctl_status.txt" 2>&1 || true

cat > "$REPORT_DIR/install_summary.md" <<EOF
# Forward Research V2 Alpha Logger Install

- label: $LABEL
- plist: $DEST
- production_action: none
- private_api: false
- order_endpoint: false
- production launchd modified: false

Manual status:
\`\`\`bash
launchctl print "gui/$(id -u)/$LABEL"
\`\`\`
EOF
