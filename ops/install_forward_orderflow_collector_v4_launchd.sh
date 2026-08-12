#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data/diagnostics/proxy_cvd_expansion_and_forward_orderflow_collector/ops
mkdir -p data/diagnostics/forward_orderflow_collector_v4/logs

PLIST_SRC="ops/launchd/com.canbit.forward_orderflow_collector_v4.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.canbit.forward_orderflow_collector_v4.plist"
REPORT="data/diagnostics/proxy_cvd_expansion_and_forward_orderflow_collector/ops/launchd_install_report.md"
STATUS_JSON="data/diagnostics/proxy_cvd_expansion_and_forward_orderflow_collector/ops/launchd_status_after_install.json"

python scripts/diagnostics/run_forward_orderflow_collector_v4.py --dry-run --production-action none --json >/tmp/canbit_forward_orderflow_dry_run.json
python scripts/diagnostics/audit_forward_orderflow_collector_v4.py --json >/tmp/canbit_forward_orderflow_audit.json

mkdir -p "$HOME/Library/LaunchAgents"
cp "$PLIST_SRC" "$PLIST_DST"
launchctl unload "$PLIST_DST" >/dev/null 2>&1 || true
launchctl load "$PLIST_DST"

launchctl list | awk '/com.canbit.forward_orderflow_collector_v4/ {print}' > /tmp/canbit_forward_orderflow_launchctl.txt || true
python - <<'PY'
import json, pathlib
status = pathlib.Path('/tmp/canbit_forward_orderflow_launchctl.txt').read_text().strip()
out = {
  "label": "com.canbit.forward_orderflow_collector_v4",
  "installed": bool(status),
  "launchctl_line": status,
  "start_interval_seconds": 300,
  "production_action": "none",
}
pathlib.Path("data/diagnostics/proxy_cvd_expansion_and_forward_orderflow_collector/ops/launchd_status_after_install.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
PY

{
  echo "# Forward Orderflow Collector V4 Launchd Install Report"
  echo
  echo "- label: com.canbit.forward_orderflow_collector_v4"
  echo "- plist: $PLIST_DST"
  echo "- interval: 300 seconds"
  echo "- output root: data/diagnostics/forward_orderflow_collector_v4/"
  echo "- production_action: none"
  echo "- private/order/account/balance/position endpoints: forbidden"
  echo
  echo "Manual uninstall:"
  echo "\`bash ops/uninstall_forward_orderflow_collector_v4_launchd.sh\`"
} > "$REPORT"

cat "$STATUS_JSON"
