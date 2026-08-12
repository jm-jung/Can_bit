#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
exec python scripts/diagnostics/run_forward_research_v4_orderflow_relative_logger.py --once --json
