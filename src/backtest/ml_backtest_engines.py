"""
Strategy-aware ML backtest engines.

This module provides abstract and concrete implementations of backtest engines
for different ML strategies (XGBoost, LSTM-Attention).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.ml_backtest_types import BacktestResult, Signal, Trade
from src.core.config import settings

logger = logging.getLogger(__name__)

# Default commission and slippage rates
DEFAULT_COMMISSION_RATE = getattr(settings, "COMMISSION_RATE", 0.0004)
DEFAULT_SLIPPAGE_RATE = getattr(settings, "SLIPPAGE_RATE", 0.0005)

# Cache directory for prediction probabilities
CACHE_DIR = Path("data/cache/ml_predictions")


class MLBacktestEngine(ABC):
    """
    Abstract base class for ML strategy backtest engines.
    
    Each concrete implementation handles strategy-specific:
    - Probability loading/caching
    - Signal generation from probabilities
    - Trade execution logic
    """
    
    def __init__(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        feature_preset: str = "extended_safe",
    ):
        """
        Initialize backtest engine.
        
        Args:
            strategy_name: Strategy identifier (e.g., "ml_xgb", "ml_lstm_attn")
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Timeframe (e.g., "5m")
            feature_preset: Feature preset (for XGBoost, ignored for LSTM)
        """
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.timeframe = timeframe
        self.feature_preset = feature_preset
        self.log_prefix = f"[ML Backtest][{self.get_engine_name()}]"
    
    @abstractmethod
    def get_engine_name(self) -> str:
        """Return engine name for logging (e.g., 'XGBoost', 'LSTM-Attn')."""
        pass
    
    @abstractmethod
    def load_predictions(
        self,
        proba_long_cache: Optional[np.ndarray] = None,
        proba_short_cache: Optional[np.ndarray] = None,
        df_with_proba: Optional[pd.DataFrame] = None,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load or compute prediction probabilities.
        
        Args:
            proba_long_cache: Optional pre-computed LONG probabilities
            proba_short_cache: Optional pre-computed SHORT probabilities
            df_with_proba: Optional DataFrame aligned with probabilities
        
        Returns:
            Tuple of (proba_long_arr, proba_short_arr, df_aligned)
        """
        pass
    
    @abstractmethod
    def generate_signals(
        self,
        proba_long_arr: np.ndarray,
        proba_short_arr: np.ndarray,
        df: pd.DataFrame,
        long_threshold: float,
        short_threshold: Optional[float],
        long_only: bool = False,
        short_only: bool = False,
        signal_confirmation_bars: int = 1,
        use_trend_filter: bool = False,
        trend_ema_window: int = 200,
        flat_threshold: Optional[float] = None,
        confidence_margin: float = 0.0,
        min_proba_dominance: float = 0.0,
    ) -> pd.DataFrame:
        """
        Generate trading signals from probabilities.
        
        Args:
            proba_long_arr: LONG probabilities array
            proba_short_arr: SHORT probabilities array
            df: DataFrame with OHLCV data
            long_threshold: Threshold for LONG signals
            short_threshold: Threshold for SHORT signals
            long_only: If True, only generate LONG signals
            short_only: If True, only generate SHORT signals
            signal_confirmation_bars: Number of consecutive bars for signal confirmation
            use_trend_filter: Whether to apply EMA trend filter
            trend_ema_window: EMA window for trend filter
        
        Returns:
            DataFrame with 'signal' column added
        """
        pass
    
    def execute_trades(
        self,
        df: pd.DataFrame,
        commission_rate: Optional[float] = None,
        slippage_rate: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        max_holding_bars: Optional[int] = None,
        use_confidence_filter: bool = False,
        confidence_quantile: float = 0.85,
        daily_loss_limit: Optional[float] = None,
    ) -> BacktestResult:
        """
        Execute trades based on signals in DataFrame.
        
        This is a generic implementation that works for both XGBoost and LSTM.
        Strategy-specific position management can be overridden if needed.
        
        Args:
            df: DataFrame with 'signal' column
            commission_rate: Override commission rate
            slippage_rate: Override slippage rate
            take_profit_pct: Take profit percentage
            stop_loss_pct: Stop loss percentage
            max_holding_bars: Maximum bars to hold a position
            use_confidence_filter: Whether to use confidence filter
            confidence_quantile: Confidence quantile threshold
            daily_loss_limit: Daily loss limit (kill switch)
        
        Returns:
            BacktestResult
        """
        from src.backtest.engine import _compute_trade_stats
        
        trades: list[Trade] = []
        equity_curve: list[float] = []
        position: dict | None = None
        balance = 1.0
        entries_attempted = 0
        exits_executed = 0
        tp_exits = 0
        sl_exits = 0
        
        # Additional tracking
        position_entry_bar_index: int | None = None
        daily_balance_start: float = 1.0
        current_date: str | None = None
        trading_disabled: bool = False
        
        # Confidence filter tracking
        confidence_proba_history: list[float] = []
        confidence_window = 100
        
        effective_commission_rate = commission_rate if commission_rate is not None else DEFAULT_COMMISSION_RATE
        effective_slippage_rate = slippage_rate if slippage_rate is not None else DEFAULT_SLIPPAGE_RATE
        
        logger.debug(f"{self.log_prefix} Starting trade execution loop: {len(df)} rows to process")
        
        for i, row in enumerate(df.itertuples()):
            signal: Signal = getattr(row, "signal")
            current_price = float(getattr(row, "close"))
            current_high = float(getattr(row, "high")) if hasattr(row, "high") else current_price
            current_low = float(getattr(row, "low")) if hasattr(row, "low") else current_price
            row_timestamp = getattr(row, "timestamp")
            
            # Daily loss limit (kill switch)
            if daily_loss_limit is not None:
                row_date = str(row_timestamp).split(" ")[0] if isinstance(row_timestamp, str) else str(row_timestamp).split("T")[0]
                if current_date is None:
                    current_date = row_date
                    daily_balance_start = balance
                elif row_date != current_date:
                    current_date = row_date
                    daily_balance_start = balance
                    trading_disabled = False
                
                daily_return = (balance - daily_balance_start) / daily_balance_start
                if daily_return <= -daily_loss_limit:
                    trading_disabled = True
                    if position is not None:
                        # Force close position
                        exit_price = current_price
                        entry_price = position["entry_price"]
                        direction = position["side"]
                        
                        entry_cost = entry_price * (effective_commission_rate + effective_slippage_rate)
                        exit_cost = exit_price * (effective_commission_rate + effective_slippage_rate)
                        
                        if direction == "LONG":
                            effective_entry = entry_price + entry_cost
                            effective_exit = exit_price - exit_cost
                            profit = (effective_exit - effective_entry) / effective_entry
                        else:
                            effective_entry = entry_price - entry_cost
                            effective_exit = exit_price + exit_cost
                            profit = (effective_entry - effective_exit) / effective_entry
                        
                        balance *= 1 + profit
                        trades.append(
                            Trade(
                                entry_time=position["entry_time"],
                                exit_time=str(row_timestamp),
                                entry_price=entry_price,
                                exit_price=exit_price,
                                direction=direction,
                                profit=profit,
                            )
                        )
                        equity_curve.append(balance)
                        exits_executed += 1
                        position = None
                        position_entry_bar_index = None
                    continue
            
            # Check for exit conditions if position exists
            exit_reason: str | None = None
            exit_price: float | None = None
            if position is not None:
                entry_price = position["entry_price"]
                direction = position["side"]
                
                # Max holding bars
                if max_holding_bars is not None and position_entry_bar_index is not None:
                    bars_held = i - position_entry_bar_index
                    if bars_held >= max_holding_bars:
                        exit_reason = "max_holding"
                        exit_price = current_price
                
                # Calculate returns for TP/SL (only if not already exiting)
                if exit_reason is None:
                    if direction == "LONG":
                        current_return_tp = (current_high / entry_price) - 1
                        current_return_sl = (current_low / entry_price) - 1
                    else:
                        current_return_tp = (entry_price / current_low) - 1
                        current_return_sl = (entry_price / current_high) - 1
                    
                    # Check TP
                    if take_profit_pct is not None and current_return_tp >= take_profit_pct:
                        exit_reason = "tp"
                        exit_price = current_high if direction == "LONG" else current_low
                    
                    # Check SL
                    if exit_reason is None:
                        if stop_loss_pct is not None and current_return_sl <= -stop_loss_pct:
                            exit_reason = "sl"
                            exit_price = current_low if direction == "LONG" else current_high
                
                # Check signal-based exit (strategy-specific)
                if exit_reason is None:
                    exit_reason = self._should_exit_position(position, signal)
                    # If signal-based exit, use current price (close price)
                    if exit_reason is not None and exit_price is None:
                        exit_price = current_price
                
                # Execute exit if needed
                if exit_reason is not None:
                    # Safety check: exit_price must be assigned
                    if exit_price is None:
                        logger.warning(
                            f"{self.log_prefix} exit_reason={exit_reason} but exit_price is None. "
                            f"Using current_price={current_price} as fallback."
                        )
                        exit_price = current_price
                    
                    entry_cost = entry_price * (effective_commission_rate + effective_slippage_rate)
                    exit_cost = exit_price * (effective_commission_rate + effective_slippage_rate)
                    
                    if direction == "LONG":
                        effective_entry = entry_price + entry_cost
                        effective_exit = exit_price - exit_cost
                        profit = (effective_exit - effective_entry) / effective_entry
                    else:
                        effective_entry = entry_price - entry_cost
                        effective_exit = exit_price + exit_cost
                        profit = (effective_entry - effective_exit) / effective_entry
                    
                    balance *= 1 + profit
                    trades.append(
                        Trade(
                            entry_time=position["entry_time"],
                            exit_time=str(row_timestamp),
                            entry_price=entry_price,
                            exit_price=exit_price,
                            direction=direction,
                            profit=profit,
                        )
                    )
                    equity_curve.append(balance)
                    exits_executed += 1
                    if exit_reason == "tp":
                        tp_exits += 1
                    elif exit_reason == "sl":
                        sl_exits += 1
                    position = None
                    position_entry_bar_index = None
            
            # Entry logic
            if position is None and not trading_disabled:
                if signal in ("LONG", "SHORT"):
                    # Confidence filter check
                    if use_confidence_filter:
                        # This would need proba arrays - for now, skip if not available
                        # Can be enhanced later
                        pass
                    
                    position = {
                        "side": signal,
                        "entry_price": current_price,
                        "entry_time": str(row_timestamp),
                    }
                    position_entry_bar_index = i
                    entries_attempted += 1
        
        # Close remaining position at end (forced EOD close)
        if position is not None:
            last_row = df.iloc[-1]
            forced_exit_price = float(last_row["close"])
            entry_price = position["entry_price"]
            direction = position["side"]
            
            logger.info(
                f"{self.log_prefix} Forced EOD close: position={direction}, "
                f"entry_price={entry_price:.2f}, exit_price={forced_exit_price:.2f}, "
                f"entry_bar={position_entry_bar_index}, exit_bar={len(df)-1}"
            )
            
            entry_cost = entry_price * (effective_commission_rate + effective_slippage_rate)
            exit_cost = forced_exit_price * (effective_commission_rate + effective_slippage_rate)
            
            if direction == "LONG":
                effective_entry = entry_price + entry_cost
                effective_exit = forced_exit_price - exit_cost
                profit = (effective_exit - effective_entry) / effective_entry
            else:
                effective_entry = entry_price - entry_cost
                effective_exit = forced_exit_price + exit_cost
                profit = (effective_entry - effective_exit) / effective_entry
            
            balance *= 1 + profit
            trades.append(
                Trade(
                    entry_time=position["entry_time"],
                    exit_time=str(last_row["timestamp"]),
                    entry_price=entry_price,
                    exit_price=forced_exit_price,
                    direction=direction,
                    profit=profit,
                )
            )
            equity_curve.append(balance)
            exits_executed += 1
        
        # Compute statistics
        from src.backtest.engine import _compute_trade_stats
        stats = _compute_trade_stats(trades)
        
        # Calculate max drawdown
        if equity_curve:
            peak = equity_curve[0]
            max_dd = 0.0
            for value in equity_curve:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
        else:
            max_dd = 0.0
        
        # Calculate win rate
        if trades:
            winning_trades = [t for t in trades if t["profit"] is not None and t["profit"] > 0]
            win_rate = len(winning_trades) / len(trades)
        else:
            win_rate = 0.0
        
        total_return = balance - 1.0
        
        # Log summary with special handling for zero trades
        if stats['total_trades'] == 0:
            logger.warning(
                f"{self.log_prefix} Trade execution complete: NO TRADES EXECUTED. "
                f"entries_attempted={entries_attempted}, signals may be too restrictive."
            )
        else:
            logger.info(
                f"{self.log_prefix} Trade execution complete: "
                f"total_trades={stats['total_trades']}, entries={entries_attempted}, "
                f"exits={exits_executed}, tp={tp_exits}, sl={sl_exits}, "
                f"total_return={total_return:.4f}, win_rate={win_rate:.4f}"
            )
        
        return BacktestResult(
            total_return=total_return,
            win_rate=win_rate,
            max_drawdown=max_dd,
            trades=trades,
            equity_curve=equity_curve if equity_curve else [1.0],
            total_trades=stats["total_trades"],
            long_trades=stats["long_trades"],
            short_trades=stats["short_trades"],
            avg_profit=stats["avg_profit"],
            median_profit=stats["median_profit"],
            avg_win=stats["avg_win"],
            avg_loss=stats["avg_loss"],
            max_consecutive_wins=stats["max_consecutive_wins"],
            max_consecutive_losses=stats["max_consecutive_losses"],
        )
    
    def _should_exit_position(self, position: dict, signal: Signal) -> Optional[str]:
        """
        Determine if position should be exited based on signal.
        
        Default implementation: exit if signal is HOLD or opposite direction.
        Can be overridden for strategy-specific logic.
        
        Args:
            position: Current position dict
            signal: Current signal
        
        Returns:
            Exit reason string or None
        """
        position_side = position["side"]
        
        if signal == "HOLD":
            return "signal_hold"
        elif signal != position_side:
            return "signal_opposite"
        
        return None
    
    @abstractmethod
    def run_backtest(
        self,
        long_threshold: float,
        short_threshold: Optional[float],
        use_optimized_threshold: bool = False,
        proba_long_cache: Optional[np.ndarray] = None,
        proba_short_cache: Optional[np.ndarray] = None,
        df_with_proba: Optional[pd.DataFrame] = None,
        index_mask: Optional[np.ndarray] = None,
        commission_rate: Optional[float] = None,
        slippage_rate: Optional[float] = None,
        long_only: bool = False,
        short_only: bool = False,
        signal_confirmation_bars: int = 1,
        use_trend_filter: bool = False,
        trend_ema_window: int = 200,
        take_profit_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        max_holding_bars: Optional[int] = None,
        use_confidence_filter: bool = False,
        confidence_quantile: float = 0.85,
        daily_loss_limit: Optional[float] = None,
        flat_threshold: Optional[float] = None,
        confidence_margin: float = 0.0,
        min_proba_dominance: float = 0.0,
    ) -> BacktestResult:
        """
        Run complete backtest.
        
        Args:
            long_threshold: Threshold for LONG signals
            short_threshold: Threshold for SHORT signals
            use_optimized_threshold: Whether to load optimized thresholds
            proba_long_cache: Optional pre-computed LONG probabilities
            proba_short_cache: Optional pre-computed SHORT probabilities
            df_with_proba: Optional DataFrame aligned with probabilities
            index_mask: Optional boolean mask for in-sample/out-of-sample splits
            commission_rate: Override commission rate
            slippage_rate: Override slippage rate
            long_only: If True, only execute LONG trades
            short_only: If True, only execute SHORT trades
            signal_confirmation_bars: Number of consecutive bars for signal confirmation
            use_trend_filter: Whether to apply EMA trend filter
            trend_ema_window: EMA window for trend filter
            take_profit_pct: Take profit percentage
            stop_loss_pct: Stop loss percentage
            max_holding_bars: Maximum bars to hold a position
            use_confidence_filter: Whether to use confidence filter
            confidence_quantile: Confidence quantile threshold
            daily_loss_limit: Daily loss limit (kill switch)
        
        Returns:
            BacktestResult
        """
        pass

