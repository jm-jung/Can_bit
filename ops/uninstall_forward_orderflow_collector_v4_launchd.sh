#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data/diagnostics/proxy_cvd_expansion_and_forward_orderflow_collector/ops
PLIST_DST="$HOME/Library/LaunchAgents/com.canbit.forward_orderflow_collector_v4.plist"
launchctl unload "$PLIST_DST" >/dev/null 2>&1 || true
rm -f "$PLIST_DST"
cat > data/diagnostics/proxy_cvd_expansion_and_forward_orderflow_collector/ops/launchd_uninstall_instructions.md <<'EOF'
# Forward Orderflow Collector V4 Uninstall

The launchd job was unloaded and the plist was removed if it existed.

Manual command:
`launchctl unload ~/Library/LaunchAgents/com.canbit.forward_orderflow_collector_v4.plist`
EOF
echo "uninstalled_or_not_present com.canbit.forward_orderflow_collector_v4"
