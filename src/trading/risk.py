"""
Risk management module for trading engine.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import Dict, Optional, Tuple


@dataclass
class RiskConfig:
    max_position_size: float = 0.002
    risk_per_trade_pct: float = 0.01
    max_daily_loss_pct: float = 0.05
    min_seconds_between_trades: int = 60


class RiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config
        self.equity: float = 1.0
        self.start_equity_today: float = 1.0
        self.today: date = datetime.utcnow().date()
        self.last_trade_time: Optional[datetime] = None
        self.last_trade_pnl_pct: Optional[float] = None
        self.trading_disabled_reason: Optional[str] = None

    def _reset_day_if_needed(self, now: datetime) -> None:
        if now.date() != self.today:
            self.today = now.date()
            self.start_equity_today = self.equity
            self.trading_disabled_reason = None

    def can_open_new_position(self, now: Optional[datetime] = None) -> Tuple[bool, str]:
        now = now or datetime.utcnow()
        self._reset_day_if_needed(now)

        if self.trading_disabled_reason:
            return False, self.trading_disabled_reason

        if self.last_trade_time is not None:
            delta = (now - self.last_trade_time).total_seconds()
            if delta < self.config.min_seconds_between_trades:
                remain = int(self.config.min_seconds_between_trades - delta)
                return False, f"cooldown: wait {remain}s"

        dd_today = (self.equity - self.start_equity_today) / self.start_equity_today
        if dd_today <= -self.config.max_daily_loss_pct:
            self.trading_disabled_reason = "max_daily_loss_reached"
            return False, self.trading_disabled_reason

        return True, "ok"

    def calculate_position_size(self, price: float) -> float:
        if price <= 0:
            return 0.0
        usd_equity = self.equity
        risk_capital = usd_equity * self.config.risk_per_trade_pct
        if risk_capital <= 0:
            return 0.0
        amount = risk_capital / price
        return min(amount, self.config.max_position_size)

    def on_open_position(self, now: Optional[datetime] = None) -> None:
        self.last_trade_time = now or datetime.utcnow()

    def on_close_position(self, trade_result: Optional[Dict], now: Optional[datetime] = None) -> None:
        if not trade_result:
            return

        now = now or datetime.utcnow()
        self.last_trade_time = now

        entry = trade_result.get("entry") or {}
        entry_price = float(entry.get("price", 0) or 0)
        exit_price = float(trade_result.get("exit_price", 0) or 0)
        side = entry.get("side")

        if entry_price > 0 and exit_price > 0 and side in ("BUY", "SELL"):
            if side == "BUY":
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price
            self.equity *= (1 + pnl_pct)
            self.last_trade_pnl_pct = pnl_pct

    def status(self) -> Dict:
        now = datetime.utcnow()
        self._reset_day_if_needed(now)
        dd_today = (self.equity - self.start_equity_today) / self.start_equity_today
        return {
            "config": asdict(self.config),
            "equity": self.equity,
            "start_equity_today": self.start_equity_today,
            "drawdown_today": dd_today,
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None,
            "last_trade_pnl_pct": self.last_trade_pnl_pct,
            "trading_disabled_reason": self.trading_disabled_reason,
        }


risk_config = RiskConfig()
risk_manager = RiskManager(risk_config)

