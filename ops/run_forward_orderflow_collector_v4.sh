#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source ".venv/bin/activate"
fi

python scripts/diagnostics/run_forward_orderflow_collector_v4.py \
  --once \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT \
  --include-orderbook \
  --orderbook-depth-limit 100 \
  --include-liquidation-if-available \
  --include-recent-aggtrades \
  --production-action none \
  --output-root data/diagnostics/forward_orderflow_collector_v4/ \
  --json
