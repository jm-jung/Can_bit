"""Trading engine that consumes strategy output."""
from __future__ import annotations

from typing import Literal

from src.strategies.basic import simple_ema_rsi_strategy
from src.trading.router import trading_router
from src.trading.risk import risk_manager

MIN_ORDER_AMOUNT = 0.0005  # BTC minimum for safety
SLIPPAGE_BPS = 0.0005  # 5 bps


def _apply_slippage(price: float, side: Literal["BUY", "SELL"]) -> float:
    slip = price * SLIPPAGE_BPS
    if side == "BUY":
        return price + slip
    return max(price - slip, 0)


def trading_step():
    """Execute a single trading step based on current strategy signal."""
    strat = simple_ema_rsi_strategy()
    signal = strat["signal"]
    price = float(strat["close"])

    client = trading_router.get_client()
    position = client.get_position()

    # Entry logic
    if position is None:
        if signal not in ("LONG", "SHORT"):
            return {"status": "no position and no entry"}

        allowed, reason = risk_manager.can_open_new_position()
        if not allowed:
            return {"status": "blocked_by_risk", "reason": reason, "risk": risk_manager.status()}

        amount = risk_manager.calculate_position_size(price)
        if amount <= 0:
            return {"status": "blocked_by_risk", "reason": "calculated amount <= 0", "risk": risk_manager.status()}
        if amount < MIN_ORDER_AMOUNT:
            return {
                "status": "blocked_by_risk",
                "reason": f"amount {amount:.6f} below minimum {MIN_ORDER_AMOUNT}",
                "risk": risk_manager.status(),
            }

        side: Literal["BUY", "SELL"] = "BUY" if signal == "LONG" else "SELL"
        exec_price = _apply_slippage(price, side)

        try:
            order = client.create_order(side, exec_price, amount=amount)
            risk_manager.on_open_position()
            return {"status": "opened", "order": order, "risk": risk_manager.status()}
        except Exception as exc:  # pragma: no cover
            return {"error": "order_failed", "details": str(exc)}

    # Exit logic
    side = position.get("side")
    should_close_long = side == "BUY" and signal == "SHORT"
    should_close_short = side == "SELL" and signal == "LONG"

    if should_close_long or should_close_short:
        close_side: Literal["BUY", "SELL"] = "SELL" if side == "BUY" else "BUY"
        exit_price = _apply_slippage(price, close_side)
        try:
            result = client.close_position(exit_price)
            risk_manager.on_close_position(result)
            return {"status": "closed", "trade": result, "risk": risk_manager.status()}
        except Exception as exc:  # pragma: no cover
            return {"error": "close_failed", "details": str(exc)}

    return {"status": "position held", "position": position, "risk": risk_manager.status()}
