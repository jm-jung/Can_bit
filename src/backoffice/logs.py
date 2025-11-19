"""Logging helpers for backoffice."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

TRADES_LOG = LOG_DIR / "trades.log"
ERRORS_LOG = LOG_DIR / "errors.log"
RISK_LOG = LOG_DIR / "risk.log"


def _append_json_line(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as file:
        file.write(line + "\n")


def _with_timestamp(event: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(event)
    data.setdefault("timestamp", datetime.utcnow().isoformat())
    return data


def log_trade_event(event: Dict[str, Any]) -> None:
    _append_json_line(TRADES_LOG, _with_timestamp(event))


def log_error_event(event: Dict[str, Any]) -> None:
    _append_json_line(ERRORS_LOG, _with_timestamp(event))


def log_risk_event(event: Dict[str, Any]) -> None:
    _append_json_line(RISK_LOG, _with_timestamp(event))
