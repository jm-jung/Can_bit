"""
Execution Router: route orders based on trading mode (shadow/paper/live).

- shadow: log only, no execution
- paper: simulate with balance tracking
- live: real API call to exchange
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from src.ops.state_manager import StateManager, TradeRecord

logger = logging.getLogger(__name__)

TradingMode = Literal["shadow", "paper", "live"]


@dataclass
class Order:
    side: Literal["BUY", "SELL"]
    price: float
    amount: float = 1.0
    strategy: str = "S1"
    reason: str = ""


@dataclass
class OrderResult:
    status: Literal["executed", "simulated", "logged", "rejected"]
    mode: str = ""
    order: Optional[Order] = None
    pnl: float = 0.0
    message: str = ""


class ExecutionRouter:
    """Route trade decisions based on current trading mode."""

    def __init__(self, mode: TradingMode, state_mgr: StateManager):
        self._mode = mode
        self._state = state_mgr
        self._open_position: Optional[Dict[str, Any]] = None
        logger.info(f"[ExecutionRouter] Initialized in '{mode}' mode")

    @property
    def mode(self) -> TradingMode:
        return self._mode

    @property
    def has_position(self) -> bool:
        return self._open_position is not None

    @property
    def open_position(self) -> Optional[Dict[str, Any]]:
        return self._open_position

    def enter(self, order: Order) -> OrderResult:
        """Route an entry order."""
        if self._open_position is not None:
            return OrderResult(status="rejected", mode=self._mode, message="already_in_position")

        ts = datetime.utcnow().isoformat()

        if self._mode == "shadow":
            logger.info(f"[SHADOW] Entry signal: {order.side} @ {order.price:.2f} ({order.strategy})")
            return OrderResult(status="logged", mode="shadow", order=order)

        if self._mode == "paper":
            self._open_position = {
                "side": order.side,
                "price": order.price,
                "amount": order.amount,
                "strategy": order.strategy,
                "entry_time": ts,
            }
            logger.info(f"[PAPER] Entered: {order.side} @ {order.price:.2f} ({order.strategy})")
            return OrderResult(status="simulated", mode="paper", order=order)

        if self._mode == "live":
            try:
                from src.trading.binance_client import trader
                result = trader.create_order(order.side, order.price, order.amount)
                self._open_position = {
                    "side": order.side,
                    "price": order.price,
                    "amount": order.amount,
                    "strategy": order.strategy,
                    "entry_time": ts,
                }
                logger.info(f"[LIVE] Order executed: {order.side} @ {order.price:.2f}")
                return OrderResult(status="executed", mode="live", order=order)
            except Exception as e:
                msg = f"Live order failed: {e}"
                logger.error(f"[LIVE] {msg}")
                self._state.record_error(msg)
                return OrderResult(status="rejected", mode="live", message=msg)

        return OrderResult(status="rejected", message="unknown_mode")

    def exit(self, exit_price: float, reason: str = "") -> OrderResult:
        """Route an exit order."""
        if self._open_position is None:
            return OrderResult(status="rejected", mode=self._mode, message="no_position")

        pos = self._open_position
        ts = datetime.utcnow().isoformat()

        if pos["side"] == "BUY":
            pnl = (exit_price - pos["price"]) / pos["price"]
        else:
            pnl = (pos["price"] - exit_price) / pos["price"]

        trade = TradeRecord(
            entry_time=pos["entry_time"],
            exit_time=ts,
            direction="LONG" if pos["side"] == "BUY" else "SHORT",
            entry_price=pos["price"],
            exit_price=exit_price,
            pnl=pnl,
            strategy=pos["strategy"],
            reason=reason,
        )

        if self._mode == "shadow":
            logger.info(f"[SHADOW] Exit signal: @ {exit_price:.2f}, pnl={pnl:.6f} ({reason})")
            self._open_position = None
            return OrderResult(status="logged", mode="shadow", pnl=pnl)

        if self._mode == "paper":
            self._state.record_trade(trade)
            self._open_position = None
            logger.info(f"[PAPER] Exited: @ {exit_price:.2f}, pnl={pnl:.6f} ({reason})")
            return OrderResult(status="simulated", mode="paper", pnl=pnl)

        if self._mode == "live":
            try:
                from src.trading.binance_client import trader
                trader.close_position(exit_price)
                self._state.record_trade(trade)
                self._open_position = None
                logger.info(f"[LIVE] Position closed: @ {exit_price:.2f}, pnl={pnl:.6f}")
                return OrderResult(status="executed", mode="live", pnl=pnl)
            except Exception as e:
                msg = f"Live exit failed: {e}"
                logger.error(f"[LIVE] {msg}")
                self._state.record_error(msg)
                return OrderResult(status="rejected", mode="live", message=msg)

        return OrderResult(status="rejected", message="unknown_mode")
