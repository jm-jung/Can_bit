"""Simulated Binance trading client."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Literal, Optional

OrderSide = Literal["BUY", "SELL"]


@dataclass
class DummyTrader:
    """A placeholder trader that stores a single open position in memory."""

    position: Optional[Dict] = field(default=None)

    def get_position(self) -> Optional[Dict]:
        """Return the currently held position, if any."""
        return self.position

    def create_order(self, side: OrderSide, price: float, amount: float) -> Dict:
        """Simulate opening a new position."""
        entry_time = datetime.utcnow().isoformat()
        order = {
            "id": str(uuid.uuid4()),
            "side": side,
            "price": float(price),
            "amount": float(amount),
            "entry_time": entry_time,
        }
        self.position = order
        return {
            "status": "opened",
            "order": order,
        }

    def close_position(self, exit_price: float) -> Optional[Dict]:
        """Close the current position and calculate PnL."""
        if not self.position:
            return None

        entry = self.position
        entry_price = entry["price"]
        side: OrderSide = entry["side"]

        if side == "BUY":
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price

        result = {
            "status": "closed",
            "entry": entry,
            "exit_price": float(exit_price),
            "exit_time": datetime.utcnow().isoformat(),
            "pnl": float(pnl),
        }

        self.position = None
        return result

    def cancel_order(self) -> Optional[Dict]:
        """Cancel the current position (if any) without PnL."""
        if not self.position:
            return None

        cancelled = {
            "status": "cancelled",
            "order": self.position,
        }
        self.position = None
        return cancelled


trader = DummyTrader()
