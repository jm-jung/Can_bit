"""
State Manager: persistent trading state (equity, PnL, trade history).

Stores to data/state/trading_state.json with atomic writes for crash safety.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

STATE_DIR = Path("data/state")
STATE_FILE = STATE_DIR / "trading_state.json"


@dataclass
class TradeRecord:
    entry_time: str
    exit_time: Optional[str] = None
    direction: str = "LONG"
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl: float = 0.0
    strategy: str = "S1"
    reason: str = ""


@dataclass
class TradingState:
    mode: str = "shadow"
    initial_equity: float = 1.0
    equity: float = 1.0
    daily_pnl: float = 0.0
    daily_date: str = ""
    peak_equity: float = 1.0
    total_trades: int = 0
    total_wins: int = 0
    consecutive_losses: int = 0
    last_trade_time: str = ""
    activation_on_count: int = 0
    activation_off_count: int = 0
    kill_switch_active: bool = False
    kill_switch_reason: str = ""
    shadow_start_date: str = ""
    paper_start_date: str = ""
    trades: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class StateManager:
    """Persist and manage trading state with crash-safe writes."""

    def __init__(self, state_path: Path = STATE_FILE):
        self._path = state_path
        self._state: TradingState = self._load()

    @property
    def state(self) -> TradingState:
        return self._state

    def _load(self) -> TradingState:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                st = TradingState(**{k: v for k, v in data.items() if k in TradingState.__dataclass_fields__})
                logger.info(f"[StateManager] Loaded state: equity={st.equity:.6f}, trades={st.total_trades}")
                return st
            except Exception as e:
                logger.error(f"[StateManager] Failed to load state: {e}. Starting fresh.")
        return TradingState()

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(asdict(self._state), indent=2, default=str), encoding="utf-8")
            shutil.move(str(tmp), str(self._path))
        except Exception as e:
            logger.error(f"[StateManager] Failed to save state: {e}")

    def reset_daily(self) -> None:
        today = date.today().isoformat()
        if self._state.daily_date != today:
            self._state.daily_pnl = 0.0
            self._state.daily_date = today
            self.save()

    def record_trade(self, trade: TradeRecord) -> None:
        self._state.trades.append(asdict(trade))
        self._state.total_trades += 1
        if trade.pnl > 0:
            self._state.total_wins += 1
            self._state.consecutive_losses = 0
        elif trade.pnl < 0:
            self._state.consecutive_losses += 1
        self._state.equity += trade.pnl
        self._state.daily_pnl += trade.pnl
        if self._state.equity > self._state.peak_equity:
            self._state.peak_equity = self._state.equity
        self._state.last_trade_time = trade.exit_time or datetime.utcnow().isoformat()
        if len(self._state.trades) > 500:
            self._state.trades = self._state.trades[-500:]
        self.save()

    def record_activation(self, is_on: bool) -> None:
        if is_on:
            self._state.activation_on_count += 1
        else:
            self._state.activation_off_count += 1

    def activate_kill_switch(self, reason: str) -> None:
        self._state.kill_switch_active = True
        self._state.kill_switch_reason = reason
        logger.warning(f"[StateManager] KILL SWITCH ACTIVATED: {reason}")
        self.save()

    def deactivate_kill_switch(self) -> None:
        self._state.kill_switch_active = False
        self._state.kill_switch_reason = ""
        self.save()

    def record_error(self, msg: str) -> None:
        ts = datetime.utcnow().isoformat()
        self._state.errors.append(f"{ts}: {msg}")
        if len(self._state.errors) > 100:
            self._state.errors = self._state.errors[-100:]
        self.save()

    @property
    def drawdown(self) -> float:
        if self._state.peak_equity <= 0:
            return 0.0
        return (self._state.equity - self._state.peak_equity) / self._state.peak_equity

    @property
    def win_rate(self) -> float:
        if self._state.total_trades == 0:
            return 0.0
        return self._state.total_wins / self._state.total_trades

    @property
    def activation_ratio(self) -> float:
        total = self._state.activation_on_count + self._state.activation_off_count
        if total == 0:
            return 0.0
        return self._state.activation_on_count / total
