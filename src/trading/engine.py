"""Trading engine that consumes strategy output."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Literal, Optional

from src.backoffice.equity_manager import append_equity_point
from src.backoffice.logs import log_error_event, log_risk_event, log_trade_event
from src.strategies.basic import simple_ema_rsi_strategy
from src.strategies.ml_xgb import ml_xgb_strategy
from src.trading.router import trading_router
from src.trading.risk import risk_manager

MIN_ORDER_AMOUNT = 0.0005  # BTC minimum for safety
SLIPPAGE_BPS = 0.0005  # 5 bps

_last_signal: str = "HOLD"
_last_trade: Optional[Dict[str, Any]] = None

# Strategy type selection
class StrategyType(str, Enum):
    RULE = "RULE"  # EMA + RSI rule-based
    ML = "ML"  # XGBoost ML strategy
    HYBRID = "HYBRID"  # Both strategies must agree


# Global strategy type (default: RULE)
_current_strategy_type: StrategyType = StrategyType.RULE


def _apply_slippage(price: float, side: Literal["BUY", "SELL"]) -> float:
    slip = price * SLIPPAGE_BPS
    if side == "BUY":
        return price + slip
    return max(price - slip, 0)


def set_strategy_type(strategy_type: StrategyType) -> None:
    """Set the strategy type for trading engine."""
    global _current_strategy_type
    _current_strategy_type = strategy_type


def get_strategy_type() -> StrategyType:
    """Get current strategy type."""
    return _current_strategy_type


def _get_strategy_signal() -> tuple[str, float]:
    """
    Get signal from current strategy type.

    Returns:
        Tuple of (signal, price)
    """
    global _current_strategy_type

    if _current_strategy_type == StrategyType.RULE:
        strat = simple_ema_rsi_strategy()
        return strat["signal"], float(strat["close"])

    elif _current_strategy_type == StrategyType.ML:
        # ML strategy - warn if in REAL mode
        if trading_router.mode == "REAL":
            log_risk_event(
                {
                    "event": "ml_strategy_warning",
                    "message": "ML strategy used in REAL mode - use with caution",
                }
            )
        strat = ml_xgb_strategy()
        if strat["proba_up"] is None:
            # Model not available, fallback to HOLD
            return "HOLD", float(strat["close"])
        return strat["signal"], float(strat["close"])

    elif _current_strategy_type == StrategyType.HYBRID:
        rule_strat = simple_ema_rsi_strategy()
        ml_strat = ml_xgb_strategy()

        rule_signal = rule_strat["signal"]
        ml_signal = ml_strat["signal"] if ml_strat["proba_up"] is not None else "HOLD"
        price = float(rule_strat["close"])

        # Both must agree for LONG or SHORT
        if rule_signal == "LONG" and ml_signal == "LONG":
            return "LONG", price
        elif rule_signal == "SHORT" and ml_signal == "SHORT":
            return "SHORT", price
        else:
            return "HOLD", price

    else:
        # Fallback to RULE
        strat = simple_ema_rsi_strategy()
        return strat["signal"], float(strat["close"])


def trading_step():
    """Execute a single trading step based on current strategy signal."""
    global _last_signal, _last_trade

    signal, price = _get_strategy_signal()
    _last_signal = signal

    client = trading_router.get_client()
    position = client.get_position()

    # Entry logic
    if position is None:
        if signal not in ("LONG", "SHORT"):
            return {"status": "no position and no entry", "risk": risk_manager.status()}

        allowed, reason = risk_manager.can_open_new_position()
        if not allowed:
            log_risk_event(
                {
                    "event": "blocked",
                    "reason": reason,
                    "mode": trading_router.mode,
                }
            )
            return {"status": "blocked_by_risk", "reason": reason, "risk": risk_manager.status()}

        amount = risk_manager.calculate_position_size(price)
        if amount <= 0:
            reason = "calculated amount <= 0"
            log_risk_event({"event": "blocked", "reason": reason, "mode": trading_router.mode})
            return {"status": "blocked_by_risk", "reason": reason, "risk": risk_manager.status()}
        if amount < MIN_ORDER_AMOUNT:
            reason = f"amount {amount:.6f} below minimum {MIN_ORDER_AMOUNT}"
            log_risk_event({"event": "blocked", "reason": reason, "mode": trading_router.mode})
            return {"status": "blocked_by_risk", "reason": reason, "risk": risk_manager.status()}

        side: Literal["BUY", "SELL"] = "BUY" if signal == "LONG" else "SELL"
        exec_price = _apply_slippage(price, side)

        try:
            order = client.create_order(side, exec_price, amount=amount)
            risk_manager.on_open_position()
            log_trade_event(
                {
                    "type": "entry",
                    "side": side,
                    "entry_price": exec_price,
                    "amount": amount,
                    "signal": signal,
                    "mode": trading_router.mode,
                    "entry_time": order.get("entry_time"),
                }
            )
            _last_trade = None
            return {"status": "opened", "order": order, "risk": risk_manager.status()}
        except Exception as exc:  # pragma: no cover
            log_error_event({"event": "order_failed", "details": str(exc)})
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

            if result:
                entry = result.get("entry") or {}
                entry_price = float(entry.get("price", 0) or 0)
                exit_exec_price = float(result.get("exit_price", 0) or 0)
                trade_side = entry.get("side")
                amount = entry.get("amount")
                pnl_pct: Optional[float] = None
                if entry_price > 0 and exit_exec_price > 0 and trade_side in ("BUY", "SELL"):
                    if trade_side == "BUY":
                        pnl_pct = (exit_exec_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - exit_exec_price) / entry_price
                    result["pnl_pct"] = pnl_pct

                log_trade_event(
                    {
                        "type": "exit",
                        "side": trade_side,
                        "entry_price": entry_price,
                        "entry_time": entry.get("entry_time"),
                        "exit_price": exit_exec_price,
                        "exit_time": result.get("exit_time"),
                        "amount": amount,
                        "pnl_pct": pnl_pct,
                        "mode": trading_router.mode,
                    }
                )
                append_equity_point(risk_manager.equity, timestamp=result.get("exit_time"))
                _last_trade = {
                    **result,
                    "mode": trading_router.mode,
                    "recorded_at": datetime.utcnow().isoformat(),
                }

            return {"status": "closed", "trade": result, "risk": risk_manager.status()}
        except Exception as exc:  # pragma: no cover
            log_error_event({"event": "close_failed", "details": str(exc)})
            return {"error": "close_failed", "details": str(exc)}

    return {"status": "position held", "position": position, "risk": risk_manager.status()}


def get_last_signal() -> str:
    return _last_signal


def get_last_trade() -> Optional[Dict[str, Any]]:
    return _last_trade
