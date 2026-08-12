#!/usr/bin/env bash
set -euo pipefail

# Run B_with_meta daily report + Discord send.
# Intended to be called by cron at 12:00 KST.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Load .env (export variables for child process)
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

VENV_PY="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "[ERROR] venv python not found at: $VENV_PY" >&2
  exit 1
fi

exec "$VENV_PY" scripts/send_b_with_meta_daily_discord.py

