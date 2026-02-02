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
from src.strategies.ml_signal_policy import (
    ActionDecisionConfig,
    ActionDecisionState,
    decide_action_3class,
)

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
        
        # 진단 로그: 캐시 데이터 범위 확인
        if "timestamp" in cache_df.columns:
            cache_ts_max = cache_df["timestamp"].max()
            cache_ts_min = cache_df["timestamp"].min()
            logger.info(
                f"{self.log_prefix}[CACHE DEBUG] Cache DataFrame timestamp range: "
                f"{cache_ts_min} ~ {cache_ts_max}"
            )
        
        # Validate cache structure
        required_cols = ["proba_long", "proba_short"]
        if not all(col in cache_df.columns for col in required_cols):
            raise ValueError(
                f"{self.log_prefix} Cache file {cache_path} missing required columns: {required_cols}"
            )
        
        proba_long_arr = cache_df["proba_long"].values.astype(np.float32)
        proba_short_arr = cache_df["proba_short"].values.astype(np.float32)
        df_aligned = cache_df.drop(columns=["proba_long", "proba_short"])
        
        # ======================================================================
        # [CACHE DEBUG] 백테스트 캐시 로드 정합성 검사
        # ======================================================================
        # Validate probability sum (3-class: p_long + p_flat + p_short = 1)
        proba_flat_arr = 1.0 - proba_long_arr - proba_short_arr
        proba_sum_arr = proba_long_arr + proba_flat_arr + proba_short_arr
        sum_deviation = np.abs(proba_sum_arr - 1.0)
        mean_sum_dev = float(np.mean(sum_deviation))
        max_sum_dev = float(np.max(sum_deviation))
        
        # Check for NaN/Inf
        nan_count = int(np.sum(np.isnan(proba_long_arr) | np.isnan(proba_short_arr)))
        inf_count = int(np.sum(np.isinf(proba_long_arr) | np.isinf(proba_short_arr)))
        
        # Check index alignment (if timestamp column exists)
        ts_mismatch_count = 0
        if "timestamp" in df_aligned.columns:
            # Check if timestamps are sorted (basic sanity check)
            ts_sorted = df_aligned["timestamp"].is_monotonic_increasing
            if not ts_sorted:
                logger.warning(
                    f"{self.log_prefix} Timestamps are not sorted. "
                    "This may indicate cache alignment issues."
                )
        
        logger.info(
            f"{self.log_prefix} Loaded {len(proba_long_arr)} predictions from cache. "
            f"3-class model: FLAT={LstmClassIndex.FLAT}, LONG={LstmClassIndex.LONG}, SHORT={LstmClassIndex.SHORT}"
        )
        logger.debug(
            f"{self.log_prefix}[CACHE DEBUG] Probability sum check: "
            f"mean(|sum-1|)={mean_sum_dev:.6f}, max(|sum-1|)={max_sum_dev:.6f}, "
            f"NaN={nan_count}, Inf={inf_count}"
        )
        
        if max_sum_dev > 0.01:
            logger.warning(
                f"{self.log_prefix}[CACHE DEBUG] WARNING: Large probability sum deviation! "
                f"max(|sum-1|)={max_sum_dev:.6f}"
            )
        
        if nan_count > 0 or inf_count > 0:
            logger.error(
                f"{self.log_prefix}[CACHE DEBUG] ERROR: Found NaN/Inf in cache! "
                f"NaN={nan_count}, Inf={inf_count}"
            )
            raise ValueError(
                f"{self.log_prefix} Cache contains NaN/Inf predictions. "
                f"Cannot proceed with backtest."
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
        # Anti-overtrading parameters
        flat_max_th: Optional[float] = None,
        margin_th: Optional[float] = None,
        apply_confirmation_to_flips: bool = True,
        # Stage-2 parameters
        use_stage2: bool = False,
        stage2_trade_th: Optional[float] = None,
        stage2_min_edge: float = 0.0,
        stage2_exit_on_flat: bool = False,
        stage2_allow_flip: bool = True,
        stage2_cooldown_bars: int = 0,
    ) -> pd.DataFrame:
        """
        Generate LSTM signals using 3-class logic (SSOT).
        
        Uses decide_action_3class() as the single source of truth for signal generation.
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
        
        # Create policy config
        config = ActionDecisionConfig(
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            margin=confidence_margin,  # Use confidence_margin as margin
            min_confidence=min_proba_dominance,  # Use min_proba_dominance as min_confidence
            flat_threshold=flat_threshold,
            cooldown_bars=0,  # Cooldown handled separately if needed
            require_flip_via_flat=False,  # Can be made configurable later
            long_only=long_only,
            short_only=short_only,
        )
        
        # Track state for cooldown/flip rules (if needed in future)
        state = ActionDecisionState()
        
        # Track reason codes for logging
        reason_code_counts: dict[str, int] = {}
        action_counts: dict[str, int] = {"LONG": 0, "SHORT": 0, "FLAT": 0}
        
        # Sample logging: first 5, random 5, and trade events
        sample_indices = set(range(min(5, n)))
        if n > 10:
            # Add random samples (fixed seed for reproducibility)
            np.random.seed(42)
            random_indices = np.random.choice(range(5, n), size=min(5, n - 5), replace=False)
            sample_indices.update(random_indices)
        
        # Generate signals using SSOT function
        for i in range(n):
            p_long = proba_long_padded[i]
            p_short = proba_short_padded[i]
            p_flat = proba_flat_padded[i]
            
            # Get timestamp for logging
            ts = None
            if "timestamp" in df.columns:
                ts = str(df.iloc[i]["timestamp"])
            
            # ======================================================================
            # [ANTI-OVERTRADING] No-trade zone 확대
            # ======================================================================
            # flat_max_th: max(proba_long, proba_short) < flat_max_th 이면 FLAT
            if flat_max_th is not None:
                max_proba = max(p_long, p_short)
                if max_proba < flat_max_th:
                    signals[i] = "HOLD"
                    reason_code_counts["FLAT_MAX_TH"] = reason_code_counts.get("FLAT_MAX_TH", 0) + 1
                    action_counts["FLAT"] = action_counts.get("FLAT", 0) + 1
                    continue
            
            # margin_th: |proba_long - proba_short| < margin_th 이면 FLAT
            if margin_th is not None:
                proba_diff = abs(p_long - p_short)
                if proba_diff < margin_th:
                    signals[i] = "HOLD"
                    reason_code_counts["MARGIN_TH"] = reason_code_counts.get("MARGIN_TH", 0) + 1
                    action_counts["FLAT"] = action_counts.get("FLAT", 0) + 1
                    continue
            
            # Update state
            state.current_idx = i
            
            # Decide action using SSOT
            result = decide_action_3class(
                proba={"p_long": p_long, "p_flat": p_flat, "p_short": p_short},
                config=config,
                state=state,
                idx=i,
                ts=ts,
            )
            
            # Map FLAT to HOLD for compatibility
            action = result.action
            if action == "FLAT":
                action = "HOLD"
            
            signals[i] = action
            
            # Track counts
            reason_code = result.reason_code
            reason_code_counts[reason_code] = reason_code_counts.get(reason_code, 0) + 1
            action_counts[action] = action_counts.get(action, 0) + 1
            
            # Sample logging (DEBUG level)
            if i in sample_indices:
                logger.debug(
                    f"{self.log_prefix}[POLICY] idx={i} ts={ts} "
                    f"pL={p_long:.4f} pF={p_flat:.4f} pS={p_short:.4f} "
                    f"top1={result.debug_info['top1_label']}({result.debug_info['top1_prob']:.4f}) "
                    f"top2={result.debug_info['top2_label']}({result.debug_info['top2_prob']:.4f}) "
                    f"margin={result.debug_info['margin']:.4f} "
                    f"min_conf={result.debug_info['min_confidence']:.4f} "
                    f"cooldown={result.debug_info['cooldown_left']} "
                    f"reason={reason_code} action={action}"
                )
        
        # Log summary
        total_signals = sum(action_counts.values())
        if total_signals > 0:
            long_pct = 100 * action_counts["LONG"] / total_signals
            short_pct = 100 * action_counts["SHORT"] / total_signals
            hold_pct = 100 * action_counts["FLAT"] / total_signals
            
            logger.info(
                f"{self.log_prefix} Signal generation (3-class SSOT): "
                f"LONG={action_counts['LONG']} ({long_pct:.1f}%), "
                f"SHORT={action_counts['SHORT']} ({short_pct:.1f}%), "
                f"HOLD={action_counts['FLAT']} ({hold_pct:.1f}%)"
            )
            
            # Log reason code distribution
            logger.debug(
                f"{self.log_prefix} Reason code distribution: {reason_code_counts}"
            )
            
            # Warn if signal rate is extremely low
            active_signal_pct = long_pct + short_pct
            if active_signal_pct < 0.01:
                logger.warning(
                    f"{self.log_prefix} Extremely low active signal rate ({active_signal_pct:.3f}%). "
                    f"Thresholds may be too strict: long={long_threshold:.3f}, short={short_threshold}. "
                    f"Consider re-optimizing with relaxed constraints."
                )
        
        df = df.copy()
        df["raw_signal"] = signals  # Stage-1 결과
        
        # ======================================================================
        # [STAGE-2] 2단계 게이팅: 거래 여부 이진 판정
        # ======================================================================
        if use_stage2:
            stage2_trade = np.full(n, False, dtype=bool)
            stage2_reason = np.full(n, "", dtype=object)
            
            for i in range(n):
                raw_sig = signals[i]
                p_long = proba_long_padded[i]
                p_short = proba_short_padded[i]
                p_flat = proba_flat_padded[i]
                
                # Stage-1이 FLAT/HOLD면 기본적으로 Trade=False (포지션 유지)
                if raw_sig == "HOLD" or raw_sig == "FLAT":
                    if stage2_exit_on_flat:
                        # 강제 청산 옵션이 켜져 있으면 Trade=True (Exit 신호)
                        stage2_trade[i] = True
                        stage2_reason[i] = "exit_on_flat"
                    else:
                        stage2_trade[i] = False
                        stage2_reason[i] = "flat_candidate"
                    continue
                
                # LONG/SHORT 후보에 대해 Stage-2 판정
                trade_ok_abs = False
                trade_ok_edge = False
                
                if raw_sig == "LONG":
                    # (A) Absolute threshold gate
                    if stage2_trade_th is not None:
                        trade_ok_abs = p_long >= stage2_trade_th
                    
                    # (B) Edge gate
                    if stage2_min_edge > 0.0:
                        trade_ok_edge = (p_long - p_short) >= stage2_min_edge
                    
                    # 둘 중 하나라도 만족하면 Trade=True
                    if trade_ok_abs or trade_ok_edge:
                        stage2_trade[i] = True
                        if trade_ok_abs and trade_ok_edge:
                            stage2_reason[i] = "trade_ok_abs_edge"
                        elif trade_ok_abs:
                            stage2_reason[i] = "trade_ok_abs"
                        else:
                            stage2_reason[i] = "trade_ok_edge"
                    else:
                        stage2_trade[i] = False
                        if stage2_trade_th is not None and p_long < stage2_trade_th:
                            stage2_reason[i] = f"no_trade_low_conf_abs(p_long={p_long:.4f}<{stage2_trade_th:.4f})"
                        elif stage2_min_edge > 0.0 and (p_long - p_short) < stage2_min_edge:
                            stage2_reason[i] = f"no_trade_low_edge(p_diff={(p_long-p_short):.4f}<{stage2_min_edge:.4f})"
                        else:
                            stage2_reason[i] = "no_trade_low_conf"
                
                elif raw_sig == "SHORT":
                    # (A) Absolute threshold gate
                    if stage2_trade_th is not None:
                        trade_ok_abs = p_short >= stage2_trade_th
                    
                    # (B) Edge gate
                    if stage2_min_edge > 0.0:
                        trade_ok_edge = (p_short - p_long) >= stage2_min_edge
                    
                    # 둘 중 하나라도 만족하면 Trade=True
                    if trade_ok_abs or trade_ok_edge:
                        stage2_trade[i] = True
                        if trade_ok_abs and trade_ok_edge:
                            stage2_reason[i] = "trade_ok_abs_edge"
                        elif trade_ok_abs:
                            stage2_reason[i] = "trade_ok_abs"
                        else:
                            stage2_reason[i] = "trade_ok_edge"
                    else:
                        stage2_trade[i] = False
                        if stage2_trade_th is not None and p_short < stage2_trade_th:
                            stage2_reason[i] = f"no_trade_low_conf_abs(p_short={p_short:.4f}<{stage2_trade_th:.4f})"
                        elif stage2_min_edge > 0.0 and (p_short - p_long) < stage2_min_edge:
                            stage2_reason[i] = f"no_trade_low_edge(p_diff={(p_short-p_long):.4f}<{stage2_min_edge:.4f})"
                        else:
                            stage2_reason[i] = "no_trade_low_conf"
                
                else:
                    # 예외: 알 수 없는 raw_sig면 안전하게 no_trade 처리
                    stage2_trade[i] = False
                    stage2_reason[i] = f"unknown_raw_sig({raw_sig})"
            
            # Final signal: Stage-2가 Trade=False면 HOLD (포지션 유지)
            final_signals = np.where(stage2_trade, signals, "HOLD")
            df["stage2_trade"] = stage2_trade
            df["stage2_reason"] = stage2_reason
            df["signal"] = final_signals
            
            # Stage-2 통계 로깅
            stage2_trade_count = np.sum(stage2_trade)
            stage2_no_trade_count = n - stage2_trade_count
            logger.info(
                f"{self.log_prefix} Stage-2 statistics: "
                f"trade={stage2_trade_count} ({100*stage2_trade_count/n:.1f}%), "
                f"no_trade={stage2_no_trade_count} ({100*stage2_no_trade_count/n:.1f}%)"
            )
            
            # Stage-2 reason 분포
            reason_counts = {}
            for reason in stage2_reason:
                if reason:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            logger.debug(
                f"{self.log_prefix} Stage-2 reason distribution: {reason_counts}"
            )
        else:
            # Stage-2 비활성: 기존과 동일
            df["stage2_trade"] = np.full(n, True, dtype=bool)  # 모두 Trade=True로 처리
            df["stage2_reason"] = np.full(n, "stage2_disabled", dtype=object)
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
        
        # Log signal distribution (final signal after Stage-2)
        final_signal_counts = {
            "LONG": int(np.sum(df["signal"] == "LONG")),
            "SHORT": int(np.sum(df["signal"] == "SHORT")),
            "HOLD": int(np.sum(df["signal"] == "HOLD")),
        }
        total_signals = sum(final_signal_counts.values())
        if total_signals > 0:
            long_pct = 100 * final_signal_counts['LONG'] / total_signals
            short_pct = 100 * final_signal_counts['SHORT'] / total_signals
            hold_pct = 100 * final_signal_counts['HOLD'] / total_signals
            
            logger.info(
                f"{self.log_prefix} Final signal distribution (after Stage-2): "
                f"LONG={final_signal_counts['LONG']} ({long_pct:.1f}%), "
                f"SHORT={final_signal_counts['SHORT']} ({short_pct:.1f}%), "
                f"HOLD={final_signal_counts['HOLD']} ({hold_pct:.1f}%)"
            )
            
            # Raw signal distribution (Stage-1) 로깅
            if "raw_signal" in df.columns:
                raw_signal_counts = {
                    "LONG": int(np.sum(df["raw_signal"] == "LONG")),
                    "SHORT": int(np.sum(df["raw_signal"] == "SHORT")),
                    "HOLD": int(np.sum(df["raw_signal"] == "HOLD")),
                }
                raw_total = sum(raw_signal_counts.values())
                if raw_total > 0:
                    raw_long_pct = 100 * raw_signal_counts['LONG'] / raw_total
                    raw_short_pct = 100 * raw_signal_counts['SHORT'] / raw_total
                    raw_hold_pct = 100 * raw_signal_counts['HOLD'] / raw_total
                    logger.info(
                        f"{self.log_prefix} Raw signal distribution (Stage-1): "
                        f"LONG={raw_signal_counts['LONG']} ({raw_long_pct:.1f}%), "
                        f"SHORT={raw_signal_counts['SHORT']} ({raw_short_pct:.1f}%), "
                        f"HOLD={raw_signal_counts['HOLD']} ({raw_hold_pct:.1f}%)"
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
        # Anti-overtrading parameters
        enter_long_th: Optional[float] = None,
        exit_long_th: Optional[float] = None,
        enter_short_th: Optional[float] = None,
        exit_short_th: Optional[float] = None,
        min_hold_bars: Optional[int] = None,
        cooldown_bars: Optional[int] = None,
        flat_max_th: Optional[float] = None,
        margin_th: Optional[float] = None,
        apply_confirmation_to_flips: bool = True,
        # Stage-2 parameters
        use_stage2: bool = False,
        stage2_trade_th: Optional[float] = None,
        stage2_min_edge: float = 0.0,
        stage2_exit_on_flat: bool = False,
        stage2_allow_flip: bool = True,
        stage2_cooldown_bars: int = 0,
        # StrategyGuard
        use_strategy_guard: bool = False,
        # StrategyGuard Phase-2 options
        strategy_guard_min_win_rate: float | None = None,
        strategy_guard_min_avg_return: float | None = None,
        strategy_guard_unblock_win_rate: float | None = None,
        strategy_guard_unblock_avg_return: float | None = None,
        strategy_guard_min_block_trades: int | None = None,
        strategy_guard_recent_trades_window: int | None = None,
        strategy_guard_insufficient_sample_policy: str | None = None,
        # StrategyGuard v2 options
        use_strategy_guard_v2: bool = False,
        strategy_guard_v2_mode: str | None = None,
        strategy_guard_v2_window_signal_stats: int | None = None,
        strategy_guard_v2_min_margin: float | None = None,
        strategy_guard_v2_max_entropy: float | None = None,
        strategy_guard_v2_scale_floor: float | None = None,
        strategy_guard_v2_block_if_scale_below: float | None = None,
        # SHORT Strategy MVP
        enable_short_strategy: bool = False,
        # Trade dump
        dump_trades_path: str | None = None,
        # Stage-2 v2.2 options
        stage2_block_if_final_scale_below: float = 0.0,
        # Stage-2 CAP 임계값 (스윕용)
        stage2_cap_entropy_high_th: float = 0.64,  # 완화: 0.66 → 0.64
        stage2_cap_entropy_mid_th: float = 0.62,   # 완화: 0.64 → 0.62
        stage2_cap_pdiff_tiny_th: float = 0.002,
        stage2_cap_pdiff_small_th: float = 0.005,  # 기본값 승격: pdiff_small_005 (기존 0.004)
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
        
        # ======================================================================
        # [CACHE DEBUG] 백테스트 데이터와 캐시 정합성 검사
        # ======================================================================
        logger.info("=" * 60)
        logger.info(f"{self.log_prefix}[CACHE DEBUG] Backtest Data-Cache Alignment Check")
        logger.info("=" * 60)
        logger.info(
            f"{self.log_prefix}[CACHE DEBUG] Data shapes: df={len(df)}, "
            f"proba_long={len(proba_long_arr)}, proba_short={len(proba_short_arr)}"
        )
        
        # Check alignment
        if len(proba_long_arr) != len(df):
            logger.warning(
                f"{self.log_prefix}[CACHE DEBUG] WARNING: Length mismatch! "
                f"df={len(df)}, proba_long={len(proba_long_arr)}. "
                "This may cause alignment issues."
            )
        else:
            logger.info(
                f"{self.log_prefix}[CACHE DEBUG] Length match: ✓ "
                f"All arrays have {len(df)} elements"
            )
        
        # Check timestamp alignment if available
        if "timestamp" in df.columns:
            df_ts_min = str(df["timestamp"].min())
            df_ts_max = str(df["timestamp"].max())
            logger.info(
                f"{self.log_prefix}[CACHE DEBUG] DataFrame timestamp range: "
                f"{df_ts_min} ~ {df_ts_max}"
            )
        
        logger.info("=" * 60)
        
        # Apply index_mask if provided
        if index_mask is not None:
            if len(index_mask) != len(df):
                logger.warning(
                    f"{self.log_prefix}[CACHE DEBUG] index_mask length ({len(index_mask)}) != df length ({len(df)}). Ignoring mask."
                )
            else:
                rows_before = len(df)
                df = df[index_mask].reset_index(drop=True)
                proba_long_arr = proba_long_arr[index_mask]
                proba_short_arr = proba_short_arr[index_mask]
                rows_after = len(df)
                logger.info(
                    f"{self.log_prefix}[CACHE DEBUG] Applied index_mask: "
                    f"{rows_before} -> {rows_after} rows "
                    f"({rows_after/rows_before*100:.1f}% retained)"
                )
        
        # Resolve enter/exit thresholds (히스테리시스)
        # If not provided, use long_threshold/short_threshold as defaults
        if enter_long_th is None:
            enter_long_th = long_threshold
        if exit_long_th is None:
            # Default: exit threshold is lower than enter (hysteresis)
            exit_long_th = enter_long_th * 0.95  # 5% lower
        if enter_short_th is None:
            enter_short_th = short_threshold if short_threshold is not None else (1.0 - long_threshold)
        if exit_short_th is None:
            # Default: exit threshold is lower than enter (hysteresis)
            exit_short_th = enter_short_th * 0.95  # 5% lower
        
        logger.info(
            f"{self.log_prefix} Hysteresis thresholds: "
            f"enter_long={enter_long_th:.3f}, exit_long={exit_long_th:.3f}, "
            f"enter_short={enter_short_th:.3f}, exit_short={exit_short_th:.3f}"
        )
        if min_hold_bars is not None:
            logger.info(f"{self.log_prefix} Min hold bars: {min_hold_bars}")
        if cooldown_bars is not None:
            logger.info(f"{self.log_prefix} Cooldown bars: {cooldown_bars}")
        if flat_max_th is not None:
            logger.info(f"{self.log_prefix} Flat max threshold: {flat_max_th:.3f}")
        if margin_th is not None:
            logger.info(f"{self.log_prefix} Margin threshold: {margin_th:.3f}")
        logger.info(
            f"{self.log_prefix} Apply confirmation to flips: {apply_confirmation_to_flips}"
        )
        
        # Generate signals
        # CRITICAL: Signal generation should use enter thresholds to match execution logic
        # This ensures signal=LONG means we can actually enter LONG position
        signal_gen_long_th = enter_long_th if enter_long_th is not None else long_threshold
        signal_gen_short_th = enter_short_th if enter_short_th is not None else short_threshold
        
        logger.info(
            f"{self.log_prefix} Signal generation thresholds: "
            f"long={signal_gen_long_th:.3f}, short={signal_gen_short_th:.3f} "
            f"(using enter thresholds to match execution)"
        )
        
        # Stage-2 설정 로깅
        if use_stage2:
            logger.info(
                f"{self.log_prefix} Stage-2 enabled: "
                f"trade_th={stage2_trade_th}, min_edge={stage2_min_edge}, "
                f"exit_on_flat={stage2_exit_on_flat}, allow_flip={stage2_allow_flip}, "
                f"cooldown_bars={stage2_cooldown_bars}"
            )
        else:
            logger.info(f"{self.log_prefix} Stage-2 disabled (using Stage-1 only)")
        
        df = self.generate_signals(
            proba_long_arr=proba_long_arr,
            proba_short_arr=proba_short_arr,
            df=df,
            long_threshold=signal_gen_long_th,  # Use enter threshold for signal generation
            short_threshold=signal_gen_short_th,  # Use enter threshold for signal generation
            long_only=long_only,
            short_only=short_only,
            signal_confirmation_bars=signal_confirmation_bars,
            use_trend_filter=use_trend_filter,
            trend_ema_window=trend_ema_window,
            flat_threshold=flat_threshold,
            confidence_margin=confidence_margin,
            min_proba_dominance=min_proba_dominance,
            flat_max_th=flat_max_th,
            margin_th=margin_th,
            apply_confirmation_to_flips=apply_confirmation_to_flips,
            # Stage-2 parameters
            use_stage2=use_stage2,
            stage2_trade_th=stage2_trade_th,
            stage2_min_edge=stage2_min_edge,
            stage2_exit_on_flat=stage2_exit_on_flat,
            stage2_allow_flip=stage2_allow_flip,
            stage2_cooldown_bars=stage2_cooldown_bars,
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
            enter_long_th=enter_long_th,
            exit_long_th=exit_long_th,
            enter_short_th=enter_short_th,
            exit_short_th=exit_short_th,
            min_hold_bars=min_hold_bars,
            cooldown_bars=cooldown_bars,
            proba_long_arr=proba_long_arr,
            proba_short_arr=proba_short_arr,
            # Direction filter
            long_only=long_only,
            short_only=short_only,
            # StrategyGuard
            use_strategy_guard=use_strategy_guard,
            # StrategyGuard Phase-2 options
            strategy_guard_min_win_rate=strategy_guard_min_win_rate,
            strategy_guard_min_avg_return=strategy_guard_min_avg_return,
            strategy_guard_unblock_win_rate=strategy_guard_unblock_win_rate,
            strategy_guard_unblock_avg_return=strategy_guard_unblock_avg_return,
            strategy_guard_min_block_trades=strategy_guard_min_block_trades,
            strategy_guard_recent_trades_window=strategy_guard_recent_trades_window,
            strategy_guard_insufficient_sample_policy=strategy_guard_insufficient_sample_policy,
            # StrategyGuard v2
            use_strategy_guard_v2=use_strategy_guard_v2,
            strategy_guard_v2_mode=strategy_guard_v2_mode,
            strategy_guard_v2_window_signal_stats=strategy_guard_v2_window_signal_stats,
            strategy_guard_v2_min_margin=strategy_guard_v2_min_margin,
            strategy_guard_v2_max_entropy=strategy_guard_v2_max_entropy,
            strategy_guard_v2_scale_floor=strategy_guard_v2_scale_floor,
            strategy_guard_v2_block_if_scale_below=strategy_guard_v2_block_if_scale_below,
            # SHORT Strategy MVP
            enable_short_strategy=enable_short_strategy,
            # Trade dump
            dump_trades_path=dump_trades_path,
            # Stage-2 (for dump accuracy)
            use_stage2=use_stage2,
            stage2_block_if_final_scale_below=stage2_block_if_final_scale_below,
            stage2_cap_entropy_high_th=stage2_cap_entropy_high_th,
            stage2_cap_entropy_mid_th=stage2_cap_entropy_mid_th,
            stage2_cap_pdiff_tiny_th=stage2_cap_pdiff_tiny_th,
            stage2_cap_pdiff_small_th=stage2_cap_pdiff_small_th,
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
    elif strategy_name == "ml_tcn":
        # TCN uses the same engine as LSTM-Attn (both are 3-class models)
        return LstmAttnBacktestEngine(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            feature_preset="base",  # TCN doesn't use feature_preset
        )
    else:
        raise ValueError(f"Unsupported ML strategy: {strategy_name}")

