"""Backoffice utility helpers."""
from __future__ import annotations

import json
from collections import deque
from datetime import datetime, date
from pathlib import Path
from typing import Deque, Dict, List, Optional

from .logs import TRADES_LOG, ERRORS_LOG, RISK_LOG


def _read_json_lines(path: Path, limit: int) -> List[Dict]:
    if not path.exists() or limit <= 0:
        return []
    buffer: Deque[Dict] = deque(maxlen=limit)
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                buffer.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(buffer)


def read_trade_logs(limit: int = 100) -> List[Dict]:
    return _read_json_lines(TRADES_LOG, limit)


def read_error_logs(limit: int = 100) -> List[Dict]:
    return _read_json_lines(ERRORS_LOG, limit)


def read_risk_logs(limit: int = 100) -> List[Dict]:
    return _read_json_lines(RISK_LOG, limit)


def _parse_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1]
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def compute_daily_report(trade_logs: List[Dict], target_date: Optional[date] = None) -> Dict:
    target_date = target_date or datetime.utcnow().date()
    exits = []
    for event in trade_logs:
        if event.get("type") != "exit":
            continue
        ts = _parse_timestamp(event.get("timestamp", ""))
        if ts is None or ts.date() != target_date:
            continue
        exits.append(event)

    if not exits:
        return {
            "date": target_date.isoformat(),
            "num_trades": 0,
            "win_rate": 0.0,
            "total_pnl_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "best_trade_pct": None,
            "worst_trade_pct": None,
        }

    total_pnl_pct = 0.0
    wins = 0
    best = float("-inf")
    worst = float("inf")
    equity = 1.0
    running_max = 1.0
    max_drawdown = 0.0

    for event in exits:
        pnl_pct = float(event.get("pnl_pct", 0) or 0)
        total_pnl_pct += pnl_pct
        if pnl_pct > 0:
            wins += 1
        best = max(best, pnl_pct)
        worst = min(worst, pnl_pct)
        equity *= (1 + pnl_pct)
        running_max = max(running_max, equity)
        drawdown = (equity - running_max) / running_max if running_max else 0.0
        max_drawdown = min(max_drawdown, drawdown)

    win_rate = wins / len(exits)

    return {
        "date": target_date.isoformat(),
        "num_trades": len(exits),
        "win_rate": win_rate,
        "total_pnl_pct": total_pnl_pct,
        "max_drawdown_pct": max_drawdown,
        "best_trade_pct": best if best != float("-inf") else None,
        "worst_trade_pct": worst if worst != float("inf") else None,
    }
