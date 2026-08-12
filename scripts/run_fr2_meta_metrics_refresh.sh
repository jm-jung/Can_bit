#!/usr/bin/env bash
set -euo pipefail

# Wrapper for cron: refresh FR2 meta rolling metrics source snapshot.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Load .env if present (for future extensibility; refresh script mostly deterministic)
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

VENV_PY="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "[ERROR] venv python not found: $VENV_PY" >&2
  exit 1
fi

exec "$VENV_PY" scripts/refresh_fr2_meta_metrics.py --force-latest

