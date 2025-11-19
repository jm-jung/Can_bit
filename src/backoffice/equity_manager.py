"""Manage equity curve storage."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .logs import LOG_DIR

EQUITY_FILE = LOG_DIR / "equity_curve.json"


def load_equity_curve() -> List[Dict]:
    if not EQUITY_FILE.exists():
        return []
    try:
        with open(EQUITY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return []


def append_equity_point(equity: float, timestamp: str | None = None) -> None:
    data = load_equity_curve()
    point = {
        "timestamp": timestamp or datetime.utcnow().isoformat(),
        "equity": float(equity),
    }
    data.append(point)
    EQUITY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EQUITY_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
