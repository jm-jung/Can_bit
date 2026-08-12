"""
Monitor: log every tick/trade decision to JSONL and optionally alert via Discord.

Output: data/monitoring/trading_log_YYYYMMDD.jsonl
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

MONITORING_DIR = Path("data/monitoring")


class Monitor:
    """Append structured events to daily JSONL log files."""

    def __init__(self, monitoring_dir: Path = MONITORING_DIR, enable_discord: bool = False):
        self._dir = monitoring_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._enable_discord = enable_discord
        self._current_date: Optional[str] = None
        self._file_handle = None

    def _get_log_path(self) -> Path:
        return self._dir / f"trading_log_{date.today().strftime('%Y%m%d')}.jsonl"

    def _ensure_file(self) -> None:
        today = date.today().isoformat()
        if self._current_date != today:
            if self._file_handle:
                self._file_handle.close()
            self._current_date = today
            self._file_handle = open(self._get_log_path(), "a", encoding="utf-8")

    def log_event(self, event: Dict[str, Any]) -> None:
        """Log a structured event to the daily JSONL file."""
        event["ts"] = datetime.utcnow().isoformat()
        try:
            self._ensure_file()
            assert self._file_handle is not None
            self._file_handle.write(json.dumps(event, default=str) + "\n")
            self._file_handle.flush()
        except Exception as e:
            logger.error(f"[Monitor] Write failed: {e}")

    def log_tick(
        self,
        mode: str,
        strategy: str,
        activation: bool,
        entropy: float,
        vol_bucket: str,
        trend_state: str,
        decision: str,
        reason: str = "",
        pnl: float = 0.0,
        price: float = 0.0,
    ) -> None:
        """Log a single tick decision."""
        self.log_event({
            "type": "tick",
            "mode": mode,
            "strategy": strategy,
            "activation": activation,
            "entropy": entropy,
            "vol_bucket": vol_bucket,
            "trend_state": trend_state,
            "decision": decision,
            "reason": reason,
            "pnl": pnl,
            "price": price,
        })

    def log_trade(
        self,
        mode: str,
        strategy: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        reason: str = "",
    ) -> None:
        """Log a completed trade."""
        event = {
            "type": "trade",
            "mode": mode,
            "strategy": strategy,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "reason": reason,
        }
        self.log_event(event)

        if self._enable_discord and abs(pnl) > 0:
            self._send_discord_alert(f"Trade closed: {direction} pnl={pnl:.4f} ({reason})")

    def log_kill_switch(self, reason: str) -> None:
        """Log kill switch activation."""
        self.log_event({"type": "kill_switch", "reason": reason})
        if self._enable_discord:
            self._send_discord_alert(f"🚨 KILL SWITCH: {reason}")

    def log_summary(
        self,
        mode: str,
        total_return: float,
        daily_return: float,
        trade_count: int,
        win_rate: float,
        avg_profit: float,
        activation_ratio: float,
        entropy_mean: float,
        current_strategy: str,
    ) -> None:
        """Log periodic performance summary."""
        self.log_event({
            "type": "summary",
            "mode": mode,
            "total_return": total_return,
            "daily_return": daily_return,
            "trade_count": trade_count,
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "activation_ratio": activation_ratio,
            "entropy_mean": entropy_mean,
            "current_strategy": current_strategy,
        })

    def log_mode_transition(self, from_mode: str, to_mode: str, reason: str) -> None:
        """Log trading mode transition suggestion."""
        self.log_event({
            "type": "mode_transition",
            "from": from_mode,
            "to": to_mode,
            "reason": reason,
        })
        if self._enable_discord:
            self._send_discord_alert(f"Mode transition suggested: {from_mode} → {to_mode} ({reason})")

    def _send_discord_alert(self, message: str) -> None:
        """Send alert via Discord webhook (best effort)."""
        try:
            webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
            if not webhook_url:
                return
            import urllib.request
            data = json.dumps({"content": f"[CanBit] {message}"}).encode()
            req = urllib.request.Request(
                webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.debug(f"[Monitor] Discord alert failed: {e}")

    def close(self) -> None:
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
