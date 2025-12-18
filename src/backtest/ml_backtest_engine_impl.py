"""
Concrete implementations of ML backtest engines.

This module contains XgbBacktestEngine and LstmAttnBacktestEngine implementations.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.ml_backtest_engines import MLBacktestEngine
from src.backtest.ml_backtest_types import BacktestResult
from src.core.config import settings
from src.dl.data.labels import LstmClassIndex
from src.indicators.basic import add_basic_indicators
from src.optimization.ml_proba_cache import _get_cache_path, get_or_build_predictions
from src.services.ohlcv_service import load_ohlcv_df
from src.strategies.ml_thresholds import resolve_ml_thresholds

logger = logging.getLogger(__name__)


class XgbBacktestEngine(MLBacktestEngine):
    """
    XGBoost backtest engine (binary classification).
    
    This engine wraps the existing XGBoost backtest logic from engine.py
    but provides strategy-specific logging.
    """
    
    def get_engine_name(self) -> str:
        return "XGBoost"
    
    def load_predictions(
        self,
        proba_long_cache: Optional[np.ndarray] = None,
        proba_short_cache: Optional[np.ndarray] = None,
        df_with_proba: Optional[pd.DataFrame] = None,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load XGBoost predictions.
        
        For XGBoost, we can use the existing run_backtest_with_ml logic
        or load from cache if provided.
        """
        # If cache is provided, use it
        if proba_long_cache is not None and proba_short_cache is not None and df_with_proba is not None:
            logger.info(
                f"{self.log_prefix} Using cached predictions: "
                f"df_rows={len(df_with_proba)}, proba_long_len={len(proba_long_cache)}"
            )
            return proba_long_cache, proba_short_cache, df_with_proba
        
        # Otherwise, predictions will be computed in run_backtest
        # This is handled by the existing run_backtest_with_ml logic
        # Return None to indicate predictions need to be computed
        raise ValueError(
            "XgbBacktestEngine.load_predictions: proba_long_cache/proba_short_cache/df_with_proba "
            "must be provided. For on-the-fly computation, use run_backtest() which delegates to "
            "run_backtest_with_ml."
        )
    
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
        Generate XGBoost signals (binary threshold logic).
        
        Note: flat_threshold, confidence_margin, min_proba_dominance are ignored for XGBoost
        (only used for LSTM 3-class model).
        """
        n = len(df)
        signals = np.full(n, "HOLD", dtype=object)
        
        # Ensure arrays are aligned
        if len(proba_long_arr) < n:
            proba_long_padded = np.pad(proba_long_arr, (0, n - len(proba_long_arr)), constant_values=0.0)
        else:
            proba_long_padded = proba_long_arr[:n]
        
        if len(proba_short_arr) < n:
            proba_short_padded = np.pad(proba_short_arr, (0, n - len(proba_short_arr)), constant_values=0.0)
        else:
            proba_short_padded = proba_short_arr[:n]
        
        # Binary threshold logic for XGBoost
        is_long_mask = (
            (proba_long_padded >= long_threshold) if long_threshold is not None
            else np.zeros(n, dtype=bool)
        )
        is_short_mask = (
            (proba_short_padded >= short_threshold) if short_threshold is not None
            else np.zeros(n, dtype=bool)
        )
        
        # Conflict resolution: if both are true, choose the one with larger margin
        conflict_mask = is_long_mask & is_short_mask
        if np.any(conflict_mask):
            margin_long = proba_long_padded[conflict_mask] - long_threshold
            margin_short = proba_short_padded[conflict_mask] - short_threshold
            prefer_short = margin_short > margin_long
            is_long_mask[conflict_mask] = ~prefer_short
            is_short_mask[conflict_mask] = prefer_short
        
        # Apply long_only / short_only filters
        if long_only:
            is_short_mask.fill(False)
        if short_only:
            is_long_mask.fill(False)
        
        # Set signals
        signals[is_long_mask] = "LONG"
        signals[is_short_mask] = "SHORT"
        
        df = df.copy()
        df["signal"] = signals
        
        # Apply signal confirmation
        if signal_confirmation_bars > 1:
            confirmed_signals = []
            raw_signal_list = df["signal"].tolist()
            
            for i in range(len(df)):
                if i < signal_confirmation_bars - 1:
                    confirmed_signals.append(raw_signal_list[i])
                else:
                    recent_signals = raw_signal_list[i - signal_confirmation_bars + 1 : i + 1]
                    if all(s == "LONG" for s in recent_signals):
                        confirmed_signals.append("LONG")
                    elif all(s == "SHORT" for s in recent_signals):
                        confirmed_signals.append("SHORT")
                    else:
                        confirmed_signals.append("HOLD")
            
            df["signal"] = confirmed_signals
        
        # Apply trend filter
        if use_trend_filter:
            df["trend_ema"] = df["close"].ewm(span=trend_ema_window, adjust=False).mean()
            filtered_signals = []
            for row in df.itertuples():
                signal_val = getattr(row, "signal")
                close_val = float(getattr(row, "close"))
                ema_val = float(getattr(row, "trend_ema"))
                
                if signal_val == "LONG" and close_val < ema_val:
                    filtered_signals.append("HOLD")
                elif signal_val == "SHORT" and close_val > ema_val:
                    filtered_signals.append("HOLD")
                else:
                    filtered_signals.append(signal_val)
            
            df["signal"] = filtered_signals
        
        return df
    
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
        Run XGBoost backtest using existing logic but with fixed logging.
        
        Note: flat_threshold, confidence_margin, min_proba_dominance are ignored for XGBoost
        (only used for LSTM 3-class model).
        """
        # Delegate to existing run_backtest_with_ml but ensure logging is correct
        from src.backtest.engine import run_backtest_with_ml
        
        # Resolve thresholds
        if use_optimized_threshold:
            resolved_long, resolved_short = resolve_ml_thresholds(
                long_threshold=long_threshold,
                short_threshold=short_threshold,
                use_optimized_thresholds=True,
                strategy_name=self.strategy_name,
                symbol=self.symbol,
                timeframe=self.timeframe,
                default_long=0.5,
                default_short=None,
            )
            long_threshold = resolved_long
            short_threshold = resolved_short
            logger.info(
                f"{self.log_prefix} Using optimized thresholds: "
                f"long={long_threshold:.3f}, short={short_threshold}"
            )
        else:
            if long_threshold is None:
                long_threshold = 0.5
            logger.info(
                f"{self.log_prefix} Using thresholds: "
                f"long={long_threshold:.3f}, short={short_threshold}"
            )
        
        # Call existing function (it will handle XGBoost-specific logic)
        result = run_backtest_with_ml(
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            use_optimized_threshold=False,  # Already resolved above
            strategy_name=self.strategy_name,
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_preset=self.feature_preset,
            proba_long_cache=proba_long_cache,
            proba_short_cache=proba_short_cache,
            df_with_proba=df_with_proba,
            index_mask=index_mask,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            long_only=long_only,
            short_only=short_only,
            signal_confirmation_bars=signal_confirmation_bars,
            use_trend_filter=use_trend_filter,
            trend_ema_window=trend_ema_window,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            max_holding_bars=max_holding_bars,
            use_confidence_filter=use_confidence_filter,
            confidence_quantile=confidence_quantile,
            daily_loss_limit=daily_loss_limit,
        )
        
        return result


class LstmAttnBacktestEngine(MLBacktestEngine):
    """
    LSTM-Attention backtest engine (3-class classification).
    
    This engine implements proper 3-class position management:
    - FLAT (0): No position
    - LONG (1): Long position
    - SHORT (2): Short position
    
    Position transitions follow the 3-class logic described in requirements.
    """
    
    def get_engine_name(self) -> str:
        return "LSTM-Attn"
    
    def load_predictions(
        self,
        proba_long_cache: Optional[np.ndarray] = None,
        proba_short_cache: Optional[np.ndarray] = None,
        df_with_proba: Optional[pd.DataFrame] = None,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load LSTM predictions from cache or compute them.
        
        For LSTM, we load from the probability cache parquet file.
        """
        # If cache is provided, use it
        if proba_long_cache is not None and proba_short_cache is not None and df_with_proba is not None:
            logger.info(
                f"{self.log_prefix} Using cached predictions: "
                f"df_rows={len(df_with_proba)}, proba_long_len={len(proba_long_cache)}, "
                f"proba_short_len={len(proba_short_cache)}"
            )
            return proba_long_cache, proba_short_cache, df_with_proba
        
        # Otherwise, load from cache file
        cache_path = _get_cache_path(
            strategy_name=self.strategy_name,
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_preset="base",  # LSTM doesn't use feature_preset
        )
        
        if not cache_path.exists():
            raise FileNotFoundError(
                f"{self.log_prefix} Probability cache not found: {cache_path}. "
                f"Please run: python -m src.optimization.ml_proba_cache "
                f"--strategy {self.strategy_name} --symbol {self.symbol} --timeframe {self.timeframe}"
            )
        
        logger.info(f"{self.log_prefix} Loading predictions from cache: {cache_path}")
        cache_df = pd.read_parquet(cache_path)
        
        # Validate cache structure
        required_cols = ["proba_long", "proba_short"]
        if not all(col in cache_df.columns for col in required_cols):
            raise ValueError(
                f"{self.log_prefix} Cache file {cache_path} missing required columns: {required_cols}"
            )
        
        proba_long_arr = cache_df["proba_long"].values.astype(np.float32)
        proba_short_arr = cache_df["proba_short"].values.astype(np.float32)
        df_aligned = cache_df.drop(columns=["proba_long", "proba_short"])
        
        logger.info(
            f"{self.log_prefix} Loaded {len(proba_long_arr)} predictions from cache. "
            f"3-class model: FLAT={LstmClassIndex.FLAT}, LONG={LstmClassIndex.LONG}, SHORT={LstmClassIndex.SHORT}"
        )
        
        return proba_long_arr, proba_short_arr, df_aligned
    
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
        Generate LSTM signals using 3-class logic.
        
        For each bar:
        - Compute proba_flat = 1 - proba_long - proba_short
        - Determine desired_direction using argmax and thresholds
        - Generate signal based on 3-class position management rules
        """
        n = len(df)
        signals = np.full(n, "HOLD", dtype=object)
        
        # Ensure arrays are aligned
        if len(proba_long_arr) < n:
            proba_long_padded = np.pad(proba_long_arr, (0, n - len(proba_long_arr)), constant_values=0.0)
        else:
            proba_long_padded = proba_long_arr[:n]
        
        if len(proba_short_arr) < n:
            proba_short_padded = np.pad(proba_short_arr, (0, n - len(proba_short_arr)), constant_values=0.0)
        else:
            proba_short_padded = proba_short_arr[:n]
        
        # Compute proba_flat (3-class: p_flat + p_long + p_short = 1)
        proba_flat_padded = 1.0 - proba_long_padded - proba_short_padded
        proba_flat_padded = np.clip(proba_flat_padded, 0.0, 1.0)
        
        # Determine desired_direction using 3-class logic with optional filters
        # Rule: if p_long >= T_long and p_long >= p_short: LONG
        #       elif p_short >= T_short and p_short > p_long: SHORT
        #       else: FLAT
        # Optional filters:
        #   - flat_threshold: if p_flat >= flat_threshold, force HOLD
        #   - confidence_margin: require (p_long - p_short) >= margin for LONG
        #   - min_proba_dominance: require min dominance over opposite direction
        
        # Use provided filter parameters (defaults handled in function signature)
        
        for i in range(n):
            p_long = proba_long_padded[i]
            p_short = proba_short_padded[i]
            p_flat = proba_flat_padded[i]
            
            # Step 1: Check flat_threshold (if set, high uncertainty forces HOLD)
            if flat_threshold is not None and p_flat >= flat_threshold:
                desired_direction = "HOLD"
            else:
                # Step 2: Determine desired direction with confidence filters
                long_margin = p_long - p_short
                short_margin = p_short - p_long
                
                # LONG signal: threshold + dominance + confidence margin
                if (p_long >= long_threshold and 
                    p_long >= p_short and
                    long_margin >= confidence_margin and
                    long_margin >= min_proba_dominance):
                    desired_direction = "LONG"
                # SHORT signal: threshold + dominance + confidence margin
                elif (short_threshold is not None and
                      p_short >= short_threshold and
                      p_short > p_long and
                      short_margin >= confidence_margin and
                      short_margin >= min_proba_dominance):
                    desired_direction = "SHORT"
                else:
                    desired_direction = "HOLD"  # FLAT
            
            # Apply long_only / short_only filters
            if long_only and desired_direction == "SHORT":
                desired_direction = "HOLD"
            if short_only and desired_direction == "LONG":
                desired_direction = "HOLD"
            
            signals[i] = desired_direction
        
        df = df.copy()
        df["signal"] = signals
        
        # Apply signal confirmation
        if signal_confirmation_bars > 1:
            confirmed_signals = []
            raw_signal_list = df["signal"].tolist()
            
            for i in range(len(df)):
                if i < signal_confirmation_bars - 1:
                    confirmed_signals.append(raw_signal_list[i])
                else:
                    recent_signals = raw_signal_list[i - signal_confirmation_bars + 1 : i + 1]
                    if all(s == "LONG" for s in recent_signals):
                        confirmed_signals.append("LONG")
                    elif all(s == "SHORT" for s in recent_signals):
                        confirmed_signals.append("SHORT")
                    else:
                        confirmed_signals.append("HOLD")
            
            df["signal"] = confirmed_signals
        
        # Apply trend filter
        if use_trend_filter:
            df["trend_ema"] = df["close"].ewm(span=trend_ema_window, adjust=False).mean()
            filtered_signals = []
            for row in df.itertuples():
                signal_val = getattr(row, "signal")
                close_val = float(getattr(row, "close"))
                ema_val = float(getattr(row, "trend_ema"))
                
                if signal_val == "LONG" and close_val < ema_val:
                    filtered_signals.append("HOLD")
                elif signal_val == "SHORT" and close_val > ema_val:
                    filtered_signals.append("HOLD")
                else:
                    filtered_signals.append(signal_val)
            
            df["signal"] = filtered_signals
        
        # Log signal distribution
        signal_counts = {
            "LONG": int(np.sum(signals == "LONG")),
            "SHORT": int(np.sum(signals == "SHORT")),
            "HOLD": int(np.sum(signals == "HOLD")),
        }
        total_signals = sum(signal_counts.values())
        if total_signals > 0:
            long_pct = 100 * signal_counts['LONG'] / total_signals
            short_pct = 100 * signal_counts['SHORT'] / total_signals
            hold_pct = 100 * signal_counts['HOLD'] / total_signals
            
            logger.info(
                f"{self.log_prefix} Signal generation (3-class): "
                f"LONG={signal_counts['LONG']} ({long_pct:.1f}%), "
                f"SHORT={signal_counts['SHORT']} ({short_pct:.1f}%), "
                f"HOLD={signal_counts['HOLD']} ({hold_pct:.1f}%)"
            )
            
            # Warn if signal rate is extremely low (likely threshold issue)
            active_signal_pct = long_pct + short_pct
            if active_signal_pct < 0.01:
                logger.warning(
                    f"{self.log_prefix} Extremely low active signal rate ({active_signal_pct:.3f}%). "
                    f"Thresholds may be too strict: long={long_threshold:.3f}, short={short_threshold}. "
                    f"Consider re-optimizing with relaxed constraints."
                )
        
        return df
    
    def _should_exit_position(self, position: dict, signal: Signal) -> Optional[str]:
        """
        LSTM-specific position exit logic (3-class).
        
        For LSTM with 3-class:
        - If current_position == LONG and desired_direction == FLAT: exit
        - If current_position == LONG and desired_direction == SHORT: exit (will flip)
        - If current_position == SHORT and desired_direction == FLAT: exit
        - If current_position == SHORT and desired_direction == LONG: exit (will flip)
        """
        position_side = position["side"]
        
        if signal == "HOLD":
            return "signal_hold"
        elif signal != position_side:
            # Opposite direction: exit current position (will enter new one after)
            return "signal_opposite"
        
        return None
    
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
        Run LSTM backtest with 3-class logic.
        """
        # Resolve thresholds
        if use_optimized_threshold:
            resolved_long, resolved_short = resolve_ml_thresholds(
                long_threshold=long_threshold,
                short_threshold=short_threshold,
                use_optimized_thresholds=True,
                strategy_name=self.strategy_name,
                symbol=self.symbol,
                timeframe=self.timeframe,
                default_long=settings.LSTM_ATTN_THRESHOLD_UP,
                default_short=settings.LSTM_ATTN_THRESHOLD_DOWN,
            )
            long_threshold = resolved_long
            short_threshold = resolved_short
            logger.info(
                f"{self.log_prefix} Using optimized thresholds: "
                f"long={long_threshold:.3f}, short={short_threshold}"
            )
        else:
            if long_threshold is None:
                long_threshold = settings.LSTM_ATTN_THRESHOLD_UP
            if short_threshold is None:
                short_threshold = settings.LSTM_ATTN_THRESHOLD_DOWN
            logger.info(
                f"{self.log_prefix} Using thresholds: "
                f"long={long_threshold:.3f}, short={short_threshold}"
            )
        
        # Load predictions
        proba_long_arr, proba_short_arr, df = self.load_predictions(
            proba_long_cache=proba_long_cache,
            proba_short_cache=proba_short_cache,
            df_with_proba=df_with_proba,
        )
        
        # Apply index_mask if provided
        if index_mask is not None:
            if len(index_mask) != len(df):
                logger.warning(
                    f"{self.log_prefix} index_mask length ({len(index_mask)}) != df length ({len(df)}). Ignoring mask."
                )
            else:
                df = df[index_mask].reset_index(drop=True)
                proba_long_arr = proba_long_arr[index_mask]
                proba_short_arr = proba_short_arr[index_mask]
                logger.debug(f"{self.log_prefix} Applied index_mask: filtered to {len(df)} rows")
        
        # Generate signals
        df = self.generate_signals(
            proba_long_arr=proba_long_arr,
            proba_short_arr=proba_short_arr,
            df=df,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            long_only=long_only,
            short_only=short_only,
            signal_confirmation_bars=signal_confirmation_bars,
            use_trend_filter=use_trend_filter,
            trend_ema_window=trend_ema_window,
            flat_threshold=flat_threshold,
            confidence_margin=confidence_margin,
            min_proba_dominance=min_proba_dominance,
        )
        
        # Execute trades
        result = self.execute_trades(
            df=df,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            max_holding_bars=max_holding_bars,
            use_confidence_filter=use_confidence_filter,
            confidence_quantile=confidence_quantile,
            daily_loss_limit=daily_loss_limit,
        )
        
        return result


def get_ml_backtest_engine(
    strategy_name: str,
    symbol: str,
    timeframe: str,
    feature_preset: str = "extended_safe",
) -> MLBacktestEngine:
    """
    Factory function to create appropriate backtest engine.
    
    Args:
        strategy_name: Strategy identifier ("ml_xgb" or "ml_lstm_attn")
        symbol: Trading symbol
        timeframe: Timeframe
        feature_preset: Feature preset (for XGBoost)
    
    Returns:
        MLBacktestEngine instance
    """
    if strategy_name == "ml_xgb":
        return XgbBacktestEngine(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            feature_preset=feature_preset,
        )
    elif strategy_name == "ml_lstm_attn":
        return LstmAttnBacktestEngine(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            feature_preset="base",  # LSTM doesn't use feature_preset
        )
    else:
        raise ValueError(f"Unsupported ML strategy: {strategy_name}")

