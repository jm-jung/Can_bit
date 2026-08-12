#!/bin/bash
set -euo pipefail
REPO_ROOT="/Users/jeongminjun/Projects/Can_bit"
PYTHON="${REPO_ROOT}/.venv/bin/python"
LOG_DIR="${REPO_ROOT}/data/diagnostics/equity_etf_qqq_regime_prospective/launchd"
mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Idle-sleep prevention for this job only (not a BTC collector change)
exec /usr/bin/caffeinate -i "${PYTHON}" "${REPO_ROOT}/scripts/equity_etf/run_qqq_regime_prospective_daily.py" --update --json
