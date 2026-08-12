"""
Risk Manager: daily loss limit, drawdown limit, kill switch.

All thresholds are conservative by default. Overridable via constructor.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from src.ops.state_manager import StateManager

logger = logging.getLogger(__name__)

Decision = Literal["ALLOW", "STOP_TRADING", "FORCE_EXIT"]


@dataclass
class RiskLimits:
    max_daily_loss: float = -0.02
    max_trade_loss: float = -0.01
    max_drawdown: float = -0.05
    max_consecutive_losses: int = 5


class RiskManager:
    """Evaluate risk conditions and enforce kill switch."""

    def __init__(self, state_mgr: StateManager, limits: RiskLimits | None = None):
        self._state = state_mgr
        self._limits = limits or RiskLimits()

    @property
    def limits(self) -> RiskLimits:
        return self._limits

    def check_pre_entry(self) -> Decision:
        """Check BEFORE entering a new trade. Returns ALLOW or STOP_TRADING."""
        st = self._state.state

        if st.kill_switch_active:
            return "STOP_TRADING"

        if st.daily_pnl <= self._limits.max_daily_loss:
            self._state.activate_kill_switch(
                f"daily_loss_limit: {st.daily_pnl:.4f} <= {self._limits.max_daily_loss}"
            )
            logger.warning(f"[RiskManager] STOP: daily loss {st.daily_pnl:.4f}")
            return "STOP_TRADING"

        dd = self._state.drawdown
        if dd <= self._limits.max_drawdown:
            self._state.activate_kill_switch(
                f"max_drawdown: {dd:.4f} <= {self._limits.max_drawdown}"
            )
            logger.warning(f"[RiskManager] STOP: drawdown {dd:.4f}")
            return "STOP_TRADING"

        if st.consecutive_losses >= self._limits.max_consecutive_losses:
            self._state.activate_kill_switch(
                f"consecutive_losses: {st.consecutive_losses} >= {self._limits.max_consecutive_losses}"
            )
            logger.warning(f"[RiskManager] STOP: {st.consecutive_losses} consecutive losses")
            return "STOP_TRADING"

        return "ALLOW"

    def check_open_position(self, unrealized_pnl: float) -> Decision:
        """Check while holding a position. Returns ALLOW or FORCE_EXIT."""
        if unrealized_pnl <= self._limits.max_trade_loss:
            logger.warning(f"[RiskManager] FORCE_EXIT: unrealized {unrealized_pnl:.4f}")
            return "FORCE_EXIT"
        return "ALLOW"

    def reset_kill_switch(self) -> None:
        """Manual reset after operator review."""
        self._state.deactivate_kill_switch()
        logger.info("[RiskManager] Kill switch manually reset")
