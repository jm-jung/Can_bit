"""Safe Binance real-client wrapper (dry run by default)."""
import logging
import os
import uuid
from typing import Dict, Literal


OrderSide = Literal["BUY", "SELL"]


class BinanceRealClient:
    """Structure for real Binance trading (currently dry-run only)."""

    def __init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.live_mode = False  # Keep live trading disabled by default

    def enable_live_mode(self) -> None:
        """Enable live trading (still guarded by NotImplemented)."""
        self.live_mode = True

    def disable_live_mode(self) -> None:
        self.live_mode = False

    def get_position(self):
        """Placeholder for interface compatibility."""
        # Real client would fetch from account; here we have no state.
        return None

    def create_order(self, side: OrderSide, price: float, amount: float) -> Dict:
        """Simulate creating an order (dry run)."""
        if not self.live_mode:
            logging.warning("[DRY RUN] No real order sent ??%s %s @ %s", side, amount, price)
            return {
                "id": str(uuid.uuid4()),
                "side": side,
                "price": float(price),
                "amount": float(amount),
                "executed": False,
                "mode": "dry-run",
            }

        raise NotImplementedError(
            "Live mode not enabled. Implement actual Binance order logic here."
        )

    def close_position(self, price: float):
        logging.warning("[DRY RUN] close_position called at price %s (no effect).", price)
        return {"status": "dry-run", "price": price}


binance_real = BinanceRealClient()
