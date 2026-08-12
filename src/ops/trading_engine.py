"""
Trading Engine: orchestrates the confirmed strategy pipeline.

Pipeline (FIXED - DO NOT MODIFY):
  1. Activation Filter: vol_bucket in ("mid", "high")
  2. Strategy Switching (rule_C_trend): trend != sideways → S2, else → S1
  3. Signal Quality Filter: entropy <= 1.0 (natural log)
  4. Entry Execution

Shadow → Paper → Live transition logic with safety checks.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from src.ops.execution_router import ExecutionRouter, Order, TradingMode
from src.ops.monitor import Monitor
from src.ops.risk_manager import Decision, RiskManager
from src.ops.state_manager import StateManager

logger = logging.getLogger(__name__)

ENTROPY_THRESHOLD = 1.0
SHADOW_PROMOTION_DAYS = 14
PAPER_MDD_THRESHOLD = -0.03


@dataclass
class TickContext:
    """All signals/features needed for one decision tick."""
    price: float
    vol_bucket: str  # "low" | "mid" | "high"
    trend_label: str  # "sideways" | "up" | "down"
    p_long: float = 0.0
    p_short: float = 0.0
    p_flat: float = 0.0
    signal: Optional[Literal["LONG", "SHORT"]] = None
    exit_signal: bool = False


class TradingEngine:
    """
    Core engine coordinating strategy pipeline, risk checks, and execution.
    Designed for fault tolerance and crash recovery.
    """

    def __init__(
        self,
        mode: TradingMode = "shadow",
        state_mgr: Optional[StateManager] = None,
        risk_mgr: Optional[RiskManager] = None,
        router: Optional[ExecutionRouter] = None,
        monitor: Optional[Monitor] = None,
        enable_discord: bool = False,
    ):
        self._state = state_mgr or StateManager()
        self._risk = risk_mgr or RiskManager(self._state)
        self._router = router or ExecutionRouter(mode, self._state)
        self._monitor = monitor or Monitor(enable_discord=enable_discord)
        self._mode = mode
        self._entropy_accumulator: list[float] = []

        if self._mode == "shadow" and not self._state.state.shadow_start_date:
            self._state.state.shadow_start_date = date.today().isoformat()
            self._state.save()
        elif self._mode == "paper" and not self._state.state.paper_start_date:
            self._state.state.paper_start_date = date.today().isoformat()
            self._state.save()

        self._state.reset_daily()
        logger.info(f"[TradingEngine] Initialized: mode={mode}")

    @property
    def mode(self) -> TradingMode:
        return self._mode

    def process_tick(self, ctx: TickContext) -> Dict[str, Any]:
        """
        Process a single tick through the full strategy pipeline.
        Returns a dict describing the decision made.
        """
        result: Dict[str, Any] = {
            "decision": "skip",
            "reason": "",
            "strategy": "",
            "pnl": 0.0,
        }

        # --- Handle exit signal for open position ---
        if ctx.exit_signal and self._router.has_position:
            exit_result = self._router.exit(ctx.price, reason="exit_signal")
            result["decision"] = "exit"
            result["reason"] = "exit_signal"
            result["pnl"] = exit_result.pnl
            self._log_tick(ctx, result)
            return result

        # --- Risk check for open position (force exit) ---
        if self._router.has_position:
            pos = self._router.open_position
            assert pos is not None
            if pos["side"] == "BUY":
                unrealized = (ctx.price - pos["price"]) / pos["price"]
            else:
                unrealized = (pos["price"] - ctx.price) / pos["price"]

            risk_decision = self._risk.check_open_position(unrealized)
            if risk_decision == "FORCE_EXIT":
                exit_result = self._router.exit(ctx.price, reason="risk_force_exit")
                result["decision"] = "exit"
                result["reason"] = "risk_force_exit"
                result["pnl"] = exit_result.pnl
                self._log_tick(ctx, result)
                return result

            result["decision"] = "hold"
            result["reason"] = "in_position"
            self._log_tick(ctx, result)
            return result

        # --- Pipeline: only runs when no position is open ---

        # Step 1: Activation Filter (vol_bucket)
        if ctx.vol_bucket not in ("mid", "high"):
            self._state.record_activation(False)
            result["decision"] = "skip"
            result["reason"] = "activation_off_low_vol"
            self._log_tick(ctx, result)
            return result

        self._state.record_activation(True)

        # Step 2: Strategy Switching (rule_C_trend)
        if ctx.trend_label != "sideways":
            strategy = "S2"
        else:
            strategy = "S1"
        result["strategy"] = strategy

        # Step 3: Signal Quality Filter (entropy <= 1.0, natural log)
        probs = [ctx.p_long, ctx.p_short, ctx.p_flat]
        entropy = self._compute_entropy(probs)
        self._entropy_accumulator.append(entropy)

        if strategy == "S2" and entropy > ENTROPY_THRESHOLD:
            result["decision"] = "skip"
            result["reason"] = f"entropy_filter ({entropy:.4f} > {ENTROPY_THRESHOLD})"
            self._log_tick(ctx, result)
            return result

        # Step 4: Entry (if signal present)
        if ctx.signal is None:
            result["decision"] = "skip"
            result["reason"] = "no_signal"
            self._log_tick(ctx, result)
            return result

        # Pre-entry risk check
        risk_decision = self._risk.check_pre_entry()
        if risk_decision == "STOP_TRADING":
            result["decision"] = "skip"
            result["reason"] = "risk_stop_trading"
            self._monitor.log_kill_switch(self._state.state.kill_switch_reason)
            self._log_tick(ctx, result)
            return result

        # Execute entry
        side = "BUY" if ctx.signal == "LONG" else "SELL"
        order = Order(side=side, price=ctx.price, strategy=strategy, reason=f"signal_{ctx.signal}")
        entry_result = self._router.enter(order)

        result["decision"] = "enter"
        result["reason"] = f"{ctx.signal}_{strategy}"
        self._log_tick(ctx, result)
        return result

    def check_mode_transition(self) -> Optional[str]:
        """
        Check if a mode transition should be suggested.
        Returns a suggestion message or None.
        """
        st = self._state.state

        if self._mode == "shadow" and st.shadow_start_date:
            start = date.fromisoformat(st.shadow_start_date)
            days = (date.today() - start).days
            has_errors = len(st.errors) > 0
            if days >= SHADOW_PROMOTION_DAYS and not has_errors:
                msg = f"Shadow running {days}d with 0 errors. Ready for paper mode."
                self._monitor.log_mode_transition("shadow", "paper", msg)
                return msg

        if self._mode == "paper" and st.paper_start_date:
            start = date.fromisoformat(st.paper_start_date)
            days = (date.today() - start).days
            total_return = st.equity - st.initial_equity
            dd = self._state.drawdown
            if days >= 7 and total_return > 0 and dd > PAPER_MDD_THRESHOLD:
                msg = (
                    f"Paper running {days}d: return={total_return:.4f}, "
                    f"MDD={dd:.4f}. Ready for live mode."
                )
                self._monitor.log_mode_transition("paper", "live", msg)
                return msg

        return None

    def log_periodic_summary(self) -> None:
        """Log a summary event (call at end of session or daily)."""
        st = self._state.state
        total_return = st.equity - st.initial_equity
        entropy_mean = float(np.mean(self._entropy_accumulator)) if self._entropy_accumulator else 0.0

        self._monitor.log_summary(
            mode=self._mode,
            total_return=total_return,
            daily_return=st.daily_pnl,
            trade_count=st.total_trades,
            win_rate=self._state.win_rate,
            avg_profit=total_return / max(st.total_trades, 1),
            activation_ratio=self._state.activation_ratio,
            entropy_mean=entropy_mean,
            current_strategy="S2" if self._state.activation_ratio > 0.5 else "S1",
        )

    def shutdown(self) -> None:
        """Graceful shutdown: persist state, close files."""
        self.log_periodic_summary()
        self._state.save()
        self._monitor.close()
        logger.info("[TradingEngine] Shutdown complete")

    @staticmethod
    def _compute_entropy(probs: list[float]) -> float:
        """Compute Shannon entropy in natural log scale."""
        entropy = 0.0
        for p in probs:
            if p > 1e-10:
                entropy -= p * math.log(p)
        return entropy

    def _log_tick(self, ctx: TickContext, result: Dict[str, Any]) -> None:
        probs = [ctx.p_long, ctx.p_short, ctx.p_flat]
        self._monitor.log_tick(
            mode=self._mode,
            strategy=result.get("strategy", ""),
            activation=ctx.vol_bucket in ("mid", "high"),
            entropy=self._compute_entropy(probs),
            vol_bucket=ctx.vol_bucket,
            trend_state=ctx.trend_label,
            decision=result["decision"],
            reason=result.get("reason", ""),
            pnl=result.get("pnl", 0.0),
            price=ctx.price,
        )
