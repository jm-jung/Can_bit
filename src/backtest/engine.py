"""
Simple backtesting engine for EMA + RSI strategy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Literal, TypedDict

import numpy as np
import pandas as pd

from src.core.config import settings
from src.indicators.basic import get_df_with_indicators, add_basic_indicators
from src.ml.xgb_model import get_xgb_model, MLXGBModel
from src.dl.lstm_attn_model import get_lstm_attn_model
from src.strategies.ml_thresholds import resolve_ml_thresholds
from src.services.ohlcv_service import load_ohlcv_df
from src.ml.features import build_feature_frame

logger = logging.getLogger(__name__)

# 전역 기본 수수료/슬리피지 설정 (settings 값이 있으면 우선 사용)
DEFAULT_COMMISSION_RATE = getattr(settings, "COMMISSION_RATE", 0.0004)
DEFAULT_SLIPPAGE_RATE = getattr(settings, "SLIPPAGE_RATE", 0.0005)


Signal = Literal["LONG", "SHORT", "HOLD"]


def _compute_trade_stats(trades: List[Trade]) -> dict:
    """
    trades 리스트로부터 다양한 통계를 계산해서 dict로 반환.
    
    Returns:
        dict with keys: total_trades, long_trades, short_trades, avg_profit,
        median_profit, avg_win, avg_loss, max_consecutive_wins, max_consecutive_losses
    """
    if not trades:
        return {
            "total_trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "avg_profit": 0.0,
            "median_profit": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
        }

    # profit 리스트 추출 (None 방지)
    profits = [t["profit"] for t in trades if t["profit"] is not None]
    if not profits:
        # profit 정보가 없는 경우도 위와 동일 처리
        return {
            "total_trades": len(trades),
            "long_trades": sum(1 for t in trades if t["direction"] == "LONG"),
            "short_trades": sum(1 for t in trades if t["direction"] == "SHORT"),
            "avg_profit": 0.0,
            "median_profit": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
        }

    total_trades = len(trades)
    long_trades = sum(1 for t in trades if t["direction"] == "LONG")
    short_trades = sum(1 for t in trades if t["direction"] == "SHORT")

    avg_profit = float(np.mean(profits))
    median_profit = float(np.median(profits))

    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p <= 0]

    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0

    # 연속 승/패 계산
    max_consec_wins = 0
    max_consec_losses = 0
    current_wins = 0
    current_losses = 0

    for p in profits:
        if p > 0:
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)

    return {
        "total_trades": total_trades,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "avg_profit": avg_profit,
        "median_profit": median_profit,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
    }


class Trade(TypedDict):
    entry_time: str
    exit_time: str | None
    entry_price: float
    exit_price: float | None
    direction: Literal["LONG", "SHORT"]
    profit: float | None


class BacktestResult(TypedDict):
    total_return: float
    win_rate: float
    max_drawdown: float
    trades: List[Trade]
    equity_curve: List[float]
    # Extended statistics
    total_trades: int
    long_trades: int
    short_trades: int
    avg_profit: float  # 전체 trade profit 평균
    median_profit: float  # 전체 trade profit 중앙값
    avg_win: float  # 이긴 trade 들의 평균 profit
    avg_loss: float  # 진 trade 들의 평균 profit
    max_consecutive_wins: int
    max_consecutive_losses: int


@dataclass
class MLBacktestAdapter:
    """Defines how a specific ML strategy plugs into the generic backtester."""

    name: str
    strategy_name: str
    get_model: Callable[[], Any]
    min_history_provider: Callable[[Any], int]
    default_long: float
    default_short: float | None


def _get_ml_adapter(strategy_name: str) -> MLBacktestAdapter:
    """
    Return adapter configuration for a given ML strategy identifier.
    """
    if strategy_name == "ml_lstm_attn":
        return MLBacktestAdapter(
            name="LSTM-Attn",
            strategy_name=strategy_name,
            get_model=get_lstm_attn_model,
            min_history_provider=lambda model: getattr(model, "window_size", 60),
            default_long=settings.LSTM_ATTN_THRESHOLD_UP,
            default_short=settings.LSTM_ATTN_THRESHOLD_DOWN,
        )

    if strategy_name == "ml_xgb":
        return MLBacktestAdapter(
            name="XGBoost",
            strategy_name="ml_xgb",
            get_model=get_xgb_model,
            min_history_provider=lambda _: 20,
            default_long=0.5,
            default_short=None,
        )

    raise ValueError(f"Unsupported ML strategy: {strategy_name}")


def run_backtest() -> BacktestResult:
    """
    Run a simple EMA + RSI backtest across the entire dataset.
    """

    df: pd.DataFrame = get_df_with_indicators().copy()

    # Strategy signals
    df["signal"] = "HOLD"
    df.loc[(df["close"] > df["ema_20"]) & (df["rsi_14"] < 70), "signal"] = "LONG"
    df.loc[(df["close"] < df["ema_20"]) & (df["rsi_14"] > 30), "signal"] = "SHORT"

    trades: List[Trade] = []
    equity_curve: List[float] = []

    position: dict | None = None
    balance = 1.0  # Start with normalized capital of 1.0

    for row in df.itertuples():
        signal: Signal = getattr(row, "signal")

        if position is None:
            if signal in ("LONG", "SHORT"):
                position = {
                    "side": signal,
                    "entry_price": float(getattr(row, "close")),
                    "entry_time": str(getattr(row, "timestamp")),
                }
        else:
            if signal == "HOLD" or signal != position["side"]:
                exit_price = float(getattr(row, "close"))
                entry_price = position["entry_price"]
                direction: Literal["LONG", "SHORT"] = position["side"]

                if direction == "LONG":
                    profit = (exit_price - entry_price) / entry_price
                else:
                    profit = (entry_price - exit_price) / entry_price

                balance *= 1 + profit

                trades.append(
                    Trade(
                        entry_time=position["entry_time"],
                        exit_time=str(getattr(row, "timestamp")),
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction=direction,
                        profit=profit,
                    )
                )

                equity_curve.append(balance)
                position = None

    # Close remaining position at last candle if open
    if position is not None:
        last_row = df.iloc[-1]
        last_price = float(last_row["close"])
        entry_price = position["entry_price"]
        direction: Literal["LONG", "SHORT"] = position["side"]

        if direction == "LONG":
            profit = (last_price - entry_price) / entry_price
        else:
            profit = (entry_price - last_price) / entry_price

        balance *= 1 + profit

        trades.append(
            Trade(
                entry_time=position["entry_time"],
                exit_time=str(last_row["timestamp"]),
                entry_price=entry_price,
                exit_price=last_price,
                direction=direction,
                profit=profit,
            )
        )
        equity_curve.append(balance)

    if trades:
        wins = sum(1 for trade in trades if trade["profit"] is not None and trade["profit"] > 0)
        win_rate = wins / len(trades)
    else:
        win_rate = 0.0

    # Max Drawdown calculation
    max_drawdown = 0.0
    running_max = float("-inf")

    for value in equity_curve:
        running_max = max(running_max, value)
        if running_max == 0:
            continue
        drawdown = (value - running_max) / running_max
        max_drawdown = min(max_drawdown, drawdown)

    # Compute extended statistics
    stats = _compute_trade_stats(trades)

    return BacktestResult(
        total_return=balance - 1.0,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
        trades=trades,
        equity_curve=equity_curve,
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


def run_backtest_compare(
    threshold_up: float | None = None,
    threshold_down: float | None = None,
) -> dict:
    """
    simple / XGB-ML / DL-LSTM-Attn 3가지 전략을 한 번에 백테스트하고
    핵심 메트릭만 요약해서 반환한다.
    
    Args:
        threshold_up: Optional override for LSTM up-threshold (default: from settings)
        threshold_down: Optional override for LSTM down-threshold (default: from settings)
    
    Returns:
        dict with keys: simple, xgb_ml, dl_lstm_attn
        Each contains summarized backtest metrics.
    """
    simple_res = run_backtest()
    ml_res = run_backtest_with_ml()
    dl_res = run_backtest_with_dl_lstm_attn(
        threshold_up=threshold_up,
        threshold_down=threshold_down,
    )

    def summarize(name: str, res: BacktestResult) -> dict:
        return {
            "strategy": name,
            "total_return": res["total_return"],
            "win_rate": res["win_rate"],
            "max_drawdown": res["max_drawdown"],
            "total_trades": res["total_trades"],
            "long_trades": res["long_trades"],
            "short_trades": res["short_trades"],
            "avg_profit": res["avg_profit"],
            "median_profit": res["median_profit"],
            "avg_win": res["avg_win"],
            "avg_loss": res["avg_loss"],
            "max_consecutive_wins": res["max_consecutive_wins"],
            "max_consecutive_losses": res["max_consecutive_losses"],
        }

    return {
        "simple": summarize("simple_ema_rsi", simple_res),
        "xgb_ml": summarize("xgb_ml", ml_res),
        "dl_lstm_attn": summarize("dl_lstm_attn", dl_res),
    }

def run_backtest_with_ml(
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    *,
    use_optimized_threshold: bool = False,
    strategy_name: str = "ml_xgb",
    symbol: str | None = None,
    timeframe: str | None = None,
    feature_preset: str = "extended_safe",
    proba_up_cache: np.ndarray | None = None,
    proba_long_cache: np.ndarray | None = None,  # Step A: LONG/SHORT split + proba cache patch
    proba_short_cache: np.ndarray | None = None,  # Step A: LONG/SHORT split + proba cache patch
    df_with_proba: pd.DataFrame | None = None,
    index_mask: np.ndarray | None = None,
    commission_rate: float | None = None,
    slippage_rate: float | None = None,
    long_only: bool = False,
    short_only: bool = False,
    signal_confirmation_bars: int = 1,
    use_trend_filter: bool = False,
    trend_ema_window: int = 200,
    take_profit_pct: float | None = None,
    stop_loss_pct: float | None = None,
    # Phase E: New trading logic parameters
    max_holding_bars: int | None = None,
    use_confidence_filter: bool = False,
    confidence_quantile: float = 0.85,
    daily_loss_limit: float | None = None,
) -> BacktestResult:
    """
    Run backtest using a pluggable ML strategy.
    
    Args:
        long_threshold: Probability threshold for LONG signal
        short_threshold: Probability threshold for SHORT signal
        use_optimized_threshold: Whether to use optimized thresholds from JSON
        strategy_name: Strategy identifier (e.g., "ml_xgb", "ml_lstm_attn")
        symbol: Optional symbol override for threshold lookup
        timeframe: Optional timeframe override for threshold lookup
        proba_up_cache: Optional pre-computed probability array (for optimization, backward compatibility)
        proba_long_cache: Optional pre-computed LONG probability array (for optimization, Step A patch)
        proba_short_cache: Optional pre-computed SHORT probability array (for optimization, Step A patch)
        df_with_proba: Optional DataFrame aligned with proba_long_cache/proba_short_cache (for optimization)
        index_mask: Optional boolean mask to filter rows (for in-sample/out-of-sample splits)
        commission_rate: Override commission rate (default: from settings)
        slippage_rate: Override slippage rate (default: from settings)
        long_only: If True, only execute LONG trades (ignore SHORT signals)
        short_only: If True, only execute SHORT trades (ignore LONG signals)
        signal_confirmation_bars: Number of consecutive bars required to confirm signal (default: 1, no smoothing)
        use_trend_filter: If True, apply EMA-based trend filter (default: False)
        trend_ema_window: EMA window for trend filter (default: 200)
        take_profit_pct: Take profit percentage (e.g., 0.003 = 0.3%), None to disable (default: None)
        stop_loss_pct: Stop loss percentage (e.g., 0.002 = 0.2%), None to disable (default: None)
    
    Returns:
        BacktestResult
    """
    adapter = _get_ml_adapter(strategy_name)
    
    # Resolve symbol and timeframe for threshold lookup
    resolved_symbol = symbol or getattr(settings, "BINANCE_SYMBOL", "BTC/USDT").replace("/", "").upper()
    resolved_timeframe = timeframe or getattr(settings, "THRESHOLD_TIMEFRAME", "1m")
    
    # Load optimized thresholds if requested
    if use_optimized_threshold:
        from src.optimization.threshold_loader import load_optimized_thresholds
        try:
            opt_long, opt_short, threshold_path = load_optimized_thresholds(
                strategy_name, resolved_symbol, resolved_timeframe
            )
            long_threshold = opt_long
            short_threshold = opt_short
            logger.info(
                "[ML Backtest][XGBoost] Using optimized thresholds from %s: long=%.3f, short=%s",
                threshold_path,
                long_threshold,
                "None" if short_threshold is None else f"{short_threshold:.3f}",
            )
        except Exception as e:
            logger.warning(
                "[ML Backtest][XGBoost] Failed to load optimized thresholds (%s). Falling back to defaults.",
                str(e)
            )
            # Fall back to default thresholds
            long_threshold = adapter.default_long
            short_threshold = adapter.default_short
    else:
        # Use provided thresholds or defaults
        if long_threshold is None:
            long_threshold = adapter.default_long
        if short_threshold is None:
            short_threshold = adapter.default_short
        
        logger.info(
            "[ML Backtest][XGBoost] Using default thresholds: long=%.3f, short=%s",
            long_threshold,
            "None" if short_threshold is None else f"{short_threshold:.3f}",
        )
    
    # Log final thresholds before signal generation
    logger.info(
        "[ML Backtest][XGBoost] Final thresholds applied: long=%.3f, short=%s",
        long_threshold,
        "None" if short_threshold is None else f"{short_threshold:.3f}",
    )
    
    # Initialize proba arrays to None (will be set from cache or computation)
    proba_long_arr = None
    proba_short_arr = None
    
    # Use cached predictions if provided, otherwise compute on-the-fly
    # Priority: proba_long_cache/proba_short_cache > proba_up_cache (backward compatibility)
    use_cached = (
        (proba_long_cache is not None and proba_short_cache is not None and df_with_proba is not None) or
        (proba_up_cache is not None and df_with_proba is not None)
    )
    
    if use_cached:
        # OPTIMIZATION: Avoid unnecessary copy - use view if index_mask is None
        # Only copy if we need to modify the DataFrame
        df = df_with_proba.copy() if index_mask is None else df_with_proba
        # Use separate caches if available, otherwise fall back to proba_up_cache
        if proba_long_cache is not None and proba_short_cache is not None:
            # OPTIMIZATION: Avoid copy - use views (arrays are read-only in backtest)
            proba_long_arr = proba_long_cache  # No copy needed - read-only
            proba_short_arr = proba_short_cache  # No copy needed - read-only
            logger.debug(
                f"[ML Backtest][{adapter.name}] Using cached LONG/SHORT predictions: "
                f"df_rows={len(df)}, proba_long_len={len(proba_long_arr)}, proba_short_len={len(proba_short_arr)}"
            )
        elif proba_up_cache is not None:
            # Backward compatibility: use proba_up_cache as proba_long
            proba_long_arr = proba_up_cache.copy()
            # Approximate SHORT proba as 1 - LONG proba (for backward compatibility)
            proba_short_arr = 1.0 - proba_long_arr
            logger.debug(
                f"[ML Backtest][{adapter.name}] Using cached predictions (backward compatibility): "
                f"df_rows={len(df)}, proba_len={len(proba_long_arr)}"
            )
        else:
            proba_long_arr = None
            proba_short_arr = None
    else:
        # Phase E: Load OHLCV data with timeframe support
        backtest_symbol = symbol or getattr(settings, "BINANCE_SYMBOL", "BTC/USDT").replace("/", "").upper()
        backtest_timeframe = timeframe or getattr(settings, "THRESHOLD_TIMEFRAME", "1m")
        
        # Load OHLCV data with specified timeframe
        # [ML] Pass symbol to use pre-resampled long-run CSV for 5m
        df_raw = load_ohlcv_df(timeframe=backtest_timeframe, symbol=backtest_symbol)
        df = add_basic_indicators(df_raw.copy())
        
        logger.info(
            f"[ML Backtest][{adapter.name}] Loaded OHLCV data: "
            f"symbol={backtest_symbol}, timeframe={backtest_timeframe}, rows={len(df)}"
        )
        
        # Phase E: Use MLXGBModel for ml_xgb strategy
        if strategy_name == "ml_xgb":
            try:
                ml_model = MLXGBModel(
                    strategy=strategy_name,
                    symbol=backtest_symbol,
                    timeframe=backtest_timeframe,
                    feature_preset=feature_preset,
                )
                logger.info(
                    f"[ML Backtest][{adapter.name}] Using MLXGBModel: "
                    f"long={ml_model.long_model_path.name}, short={ml_model.short_model_path.name}, "
                    f"scaler={ml_model.scaler_path.name if ml_model.scaler_path and ml_model.scaler_path.exists() else 'None'}, "
                    f"label_mode={ml_model.label_mode}, feature_preset={feature_preset}"
                )
                logger.info(
                    f"[ML Backtest][{adapter.name}] Model metadata: "
                    f"timeframe={ml_model.meta_data.get('timeframe', 'unknown')}, "
                    f"label_thresholds={ml_model.label_thresholds}, "
                    f"scaler_type={ml_model.scaler_type}"
                )
                # Use MLXGBModel for predictions
                model = ml_model
                has_separate_models = True
            except Exception as e:
                logger.warning(
                    f"[ML Backtest][{adapter.name}] Failed to load MLXGBModel ({e}). "
                    "Falling back to legacy model loader."
                )
                model = adapter.get_model()
                has_separate_models = getattr(model, "has_separate_models", lambda: False)() if model is not None else False
        else:
            model = adapter.get_model()
            has_separate_models = getattr(model, "has_separate_models", lambda: False)() if model is not None else False

        def _empty_result(message: str) -> BacktestResult:
            logger.error(message)
            empty_stats = _compute_trade_stats([])
            return BacktestResult(
                total_return=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                trades=[],
                equity_curve=[1.0],
                total_trades=empty_stats["total_trades"],
                long_trades=empty_stats["long_trades"],
                short_trades=empty_stats["short_trades"],
                avg_profit=empty_stats["avg_profit"],
                median_profit=empty_stats["median_profit"],
                avg_win=empty_stats["avg_win"],
                avg_loss=empty_stats["avg_loss"],
                max_consecutive_wins=empty_stats["max_consecutive_wins"],
                max_consecutive_losses=empty_stats["max_consecutive_losses"],
            )

        if model is None:
            return _empty_result(f"{adapter.name} model instance is None. Cannot run ML backtest.")

        # Check if model is loaded (MLXGBModel or legacy XGBSignalModel)
        if isinstance(model, MLXGBModel):
            if not model.is_loaded():
                return _empty_result(
                    f"{adapter.name} MLXGBModel not loaded. "
                    f"long={model.long_model_path}, short={model.short_model_path}"
                )
        elif not getattr(model, "is_loaded", lambda: False)():
            model_path = getattr(model, "model_path", None)
            exists = model_path.exists() if model_path is not None else False
            return _empty_result(
                f"{adapter.name} model not loaded. Model path: {model_path}, exists={exists}"
            )

        logger.info(
            f"[ML Backtest][{adapter.name}] Computing predictions: "
            f"long_threshold={long_threshold}, short_threshold={short_threshold}, "
            f"total_rows={len(df)}"
        )

        try:
            # Get min history requirement
            if isinstance(model, MLXGBModel):
                min_rows_for_prediction = 20  # Default for XGBoost
            else:
                min_rows_for_prediction = max(0, adapter.min_history_provider(model))
            start_idx = min_rows_for_prediction

            # Extract symbol and timeframe for event features
            # Use provided symbol/timeframe or fall back to settings
            backtest_symbol = symbol or getattr(settings, "BINANCE_SYMBOL", "BTC/USDT").replace("/", "").upper()
            backtest_timeframe = timeframe or getattr(settings, "THRESHOLD_TIMEFRAME", "1m")
            
            # 🔥 FIX: Generate features once for entire dataset to avoid repeated event feature calculation
            # Extract features for the entire dataset (event features calculated once)
            logger.info(
                f"[ML Backtest][{adapter.name}] Extracting features for entire dataset "
                f"(rows={len(df)}, start_idx={start_idx})..."
            )
            logger.debug(
                f"[ML Backtest][{adapter.name}] This should trigger build_event_feature_frame only ONCE "
                f"for the entire dataset, not per-row."
            )
            
            # Phase E: Use MLXGBModel's prediction methods if available
            if isinstance(model, MLXGBModel):
                # MLXGBModel: Extract features once and use batch prediction
                try:
                    from src.features.ml_feature_config import MLFeatureConfig
                    feature_config = MLFeatureConfig.from_preset(feature_preset)
                    
                    # Extract features for entire dataset
                    full_features = build_feature_frame(
                        df,
                        symbol=backtest_symbol,
                        timeframe=backtest_timeframe,
                        use_events=settings.EVENTS_ENABLED,
                        feature_config=feature_config,
                    )
                    
                    proba_long_values: list[float] = []
                    proba_short_values: list[float] = []
                    prediction_errors = 0
                    
                    for i in range(start_idx, len(df)):
                        # Use features up to index i (sliding window)
                        features_slice = full_features.iloc[: i + 1]
                        last_features = features_slice.iloc[[-1]].copy()
                        
                        try:
                            # MLXGBModel handles scaling internally
                            proba_long = model.predict_proba_long(last_features)
                            proba_short = model.predict_proba_short(last_features)
                            proba_long_values.append(float(proba_long[0]) if isinstance(proba_long, np.ndarray) else float(proba_long))
                            proba_short_values.append(float(proba_short[0]) if isinstance(proba_short, np.ndarray) else float(proba_short))
                        except Exception as exc:
                            prediction_errors += 1
                            if prediction_errors <= 5:
                                logger.warning(
                                    f"[ML Backtest][{adapter.name}] Prediction failed at index {i}: "
                                    f"{type(exc).__name__}: {exc}"
                                )
                            continue
                    
                    # Check if we got any successful predictions
                    if not proba_long_values:
                        raise RuntimeError(
                            f"[ML Backtest][{adapter.name}] MLXGBModel prediction failed: "
                            f"No successful predictions out of {len(df) - start_idx} attempts. "
                            f"All predictions failed."
                        )
                    
                    # Convert to numpy arrays
                    proba_long_arr = np.array(proba_long_values, dtype=np.float32)
                    proba_short_arr = np.array(proba_short_values, dtype=np.float32)
                    
                    # Align df with proba arrays
                    df = df.iloc[start_idx:start_idx + len(proba_long_arr)].reset_index(drop=True)
                    
                    logger.info(
                        f"[ML Backtest][{adapter.name}] Completed ML predictions: "
                        f"successful={len(proba_long_values)}, errors={prediction_errors}, "
                        f"mean_proba_long={proba_long_arr.mean():.4f}, mean_proba_short={proba_short_arr.mean():.4f}"
                    )
                except Exception as e:
                    logger.exception(
                        f"[ML Backtest][{adapter.name}] Failed to compute ML predictions with MLXGBModel",
                        exc_info=e
                    )
                    raise RuntimeError(
                        f"[ML Backtest][{adapter.name}] MLXGBModel prediction failed in run_backtest_with_ml"
                    ) from e
            elif hasattr(model, "_extract_features"):
                # Legacy XGBSignalModel: Use existing _extract_features method
                full_features = model._extract_features(df, symbol=backtest_symbol, timeframe=backtest_timeframe)
                
                # Use batch prediction with full features
                # For each prediction point, use features up to that point (sliding window)
                # But event features are already calculated for the full dataset
                proba_long_values: list[float] = []
                proba_short_values: list[float] = []
                prediction_errors = 0
                
                # Get model's expected feature names (use long_model if available, else model)
                ref_model = getattr(model, "long_model", None) or getattr(model, "model", None)
                if ref_model and hasattr(ref_model, "get_booster"):
                    model_feature_names = ref_model.get_booster().feature_names
                    if model_feature_names is None:
                        model_feature_names = [f"f{i}" for i in range(len(ref_model.feature_importances_))]
                else:
                    model_feature_names = None
                
                for i in range(start_idx, len(df)):
                    # Use features up to index i (sliding window)
                    features_slice = full_features.iloc[: i + 1]
                    last_features = features_slice.iloc[[-1]].copy()
                    
                    # Align with model's expected feature order
                    if model_feature_names:
                        missing_features = set(model_feature_names) - set(last_features.columns)
                        for feat in missing_features:
                            last_features[feat] = 0.0
                        last_features = last_features.reindex(columns=model_feature_names, fill_value=0.0)
                    
                    try:
                        if has_separate_models:
                            # Predict with both LONG and SHORT models
                            proba_long = model.long_model.predict_proba(last_features)[0]
                            proba_short = model.short_model.predict_proba(last_features)[0]
                            proba_long_values.append(float(proba_long[1]))
                            proba_short_values.append(float(proba_short[1]))
                        else:
                            # Single model (backward compatibility)
                            proba = model.model.predict_proba(last_features)[0]
                            proba_up = float(proba[1])  # proba[1] = probability of class 1 (up)
                            proba_long_values.append(proba_up)
                            proba_short_values.append(1.0 - proba_up)  # Approximate SHORT proba
                    except Exception as exc:
                        prediction_errors += 1
                        if prediction_errors <= 5:
                            logger.warning(
                                f"[ML Backtest][{adapter.name}] Prediction failed at index {i}: "
                                f"{type(exc).__name__}: {exc}"
                            )
                        continue
                
                if not proba_long_values:
                    return _empty_result(
                        f"[ML Backtest][{adapter.name}] No successful predictions!"
                    )
                
                proba_long_arr = np.array(proba_long_values, dtype=np.float32)
                proba_short_arr = np.array(proba_short_values, dtype=np.float32)
                # Align df with proba arrays
                df = df.iloc[start_idx:start_idx + len(proba_long_arr)].reset_index(drop=True)
                
                logger.info(
                    f"[ML Backtest][{adapter.name}] Completed predictions: "
                    f"successful={len(proba_long_values)}, errors={prediction_errors}, "
                    f"mean_proba_long={proba_long_arr.mean():.4f}, mean_proba_short={proba_short_arr.mean():.4f}"
                )
            else:
                # Fallback: use predict_proba_latest for each slice (slower but compatible)
                logger.warning(
                    f"[ML Backtest][{adapter.name}] Model does not have _extract_features. "
                    f"Using per-slice prediction (may be slower)."
                )
                proba_long_values: list[float] = []
                proba_short_values: list[float] = []
                prediction_errors = 0
                for i in range(start_idx, len(df)):
                    df_slice = df.iloc[: i + 1]
                    try:
                        if has_separate_models:
                            # Use return_both=True to get both proba_long and proba_short
                            proba_long, proba_short = model.predict_proba_latest(
                                df_slice, 
                                symbol=backtest_symbol, 
                                timeframe=backtest_timeframe,
                                return_both=True
                            )
                            proba_long_values.append(float(proba_long))
                            proba_short_values.append(float(proba_short))
                        else:
                            # Single model (backward compatibility)
                            proba_up = float(model.predict_proba_latest(df_slice, symbol=backtest_symbol, timeframe=backtest_timeframe))
                            proba_long_values.append(proba_up)
                            proba_short_values.append(1.0 - proba_up)  # Approximate SHORT proba
                    except Exception as exc:
                        prediction_errors += 1
                        if prediction_errors <= 5:
                            logger.warning(
                                f"[ML Backtest][{adapter.name}] Prediction failed at index {i}: "
                                f"{type(exc).__name__}: {exc}"
                            )
                        continue
                
                if not proba_long_values:
                    return _empty_result(
                        f"[ML Backtest][{adapter.name}] No successful predictions!"
                    )
                
                proba_long_arr = np.array(proba_long_values, dtype=np.float32)
                proba_short_arr = np.array(proba_short_values, dtype=np.float32)
                df = df.iloc[start_idx:start_idx + len(proba_long_arr)].reset_index(drop=True)
                
        except Exception as exc:
            return _empty_result(
                f"[ML Backtest][{adapter.name}] Prediction loop failed: {type(exc).__name__}: {exc}"
            )

    # Apply index_mask if provided (for in-sample/out-of-sample splits)
    if index_mask is not None:
        if len(index_mask) != len(df):
            logger.warning(
                f"[ML Backtest][{adapter.name}] index_mask length ({len(index_mask)}) "
                f"!= df length ({len(df)}). Ignoring mask."
            )
        else:
            df = df[index_mask].reset_index(drop=True)
            if 'proba_long_arr' in locals():
                proba_long_arr = proba_long_arr[index_mask]
            if 'proba_short_arr' in locals():
                proba_short_arr = proba_short_arr[index_mask]
            logger.debug(
                f"[ML Backtest][{adapter.name}] Applied index_mask: "
                f"filtered to {len(df)} rows"
            )
    
    # Ensure proba arrays exist (from cache or computation)
    # For ML strategies, proba_long_arr and proba_short_arr should be set by now
    # If they're still None, it's an error for ML strategies
    if strategy_name in ["ml_xgb", "ml_lstm_attn"]:
        if proba_long_arr is None:
            raise ValueError(
                f"[ML Backtest][{adapter.name}] proba_long_arr not available for ML strategy. "
                f"Either provide proba_long_cache/proba_short_cache or ensure predictions were computed successfully."
            )
        
        # Ensure proba_short_arr exists (should be set by computation or backward compatibility fallback)
        if proba_short_arr is None:
            # Fallback: approximate SHORT proba as 1 - LONG proba (for backward compatibility)
            proba_short_arr = 1.0 - proba_long_arr
            logger.warning(
                f"[ML Backtest][{adapter.name}] proba_short_arr not available. "
                f"Using approximate: proba_short = 1 - proba_long (backward compatibility)"
            )
    else:
        # Non-ML strategies: proba arrays may not be needed
        if proba_long_arr is None:
            raise ValueError(
                f"[ML Backtest][{adapter.name}] proba_long_arr not available. "
                f"This should not happen for non-ML strategies."
            )

    # Generate signals from probabilities (using separate LONG/SHORT proba)
    # OPTIMIZATION: Vectorized signal generation (10-50x faster than loop-based)
    n = len(df)
    signals = np.full(n, "HOLD", dtype=object)
    
    # Ensure arrays are aligned
    if len(proba_long_arr) < n:
        # Pad with None/0 if needed
        proba_long_padded = np.pad(proba_long_arr, (0, n - len(proba_long_arr)), constant_values=0.0)
    else:
        proba_long_padded = proba_long_arr[:n]
    
    if len(proba_short_arr) < n:
        proba_short_padded = np.pad(proba_short_arr, (0, n - len(proba_short_arr)), constant_values=0.0)
    else:
        proba_short_padded = proba_short_arr[:n]
    
    # Vectorized threshold checks
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
        # Prefer LONG if margin_long >= margin_short, else prefer SHORT
        prefer_short = margin_short > margin_long
        is_long_mask[conflict_mask] = ~prefer_short
        is_short_mask[conflict_mask] = prefer_short
    
    # Apply long_only / short_only filters
    if long_only:
        is_short_mask.fill(False)
    if short_only:
        is_long_mask.fill(False)
    
    # Set signals (vectorized)
    signals[is_long_mask] = "LONG"
    signals[is_short_mask] = "SHORT"
    
    # Count signals
    signal_counts = {
        "LONG": int(np.sum(is_long_mask)),
        "SHORT": int(np.sum(is_short_mask)),
        "HOLD": int(n - np.sum(is_long_mask) - np.sum(is_short_mask)),
    }
    
    # Assign to DataFrame
    df["signal"] = signals
    
    if not use_cached:
        logger.info(
            f"[ML Backtest][{adapter.name}] Starting signal generation: "
            f"long_threshold={long_threshold}, short_threshold={short_threshold}, "
            f"total_rows={len(df)}"
        )
        total_signals = sum(signal_counts.values())
        hold_pct = signal_counts["HOLD"] / total_signals * 100.0 if total_signals > 0 else 0.0
        long_pct = signal_counts["LONG"] / total_signals * 100.0 if total_signals > 0 else 0.0
        short_pct = signal_counts["SHORT"] / total_signals * 100.0 if total_signals > 0 else 0.0
        logger.info(
            f"[ML Backtest][{adapter.name}] Signal generation summary: "
            f"LONG={signal_counts['LONG']} ({long_pct:.2f}%), "
            f"SHORT={signal_counts['SHORT']} ({short_pct:.2f}%), "
            f"HOLD={signal_counts['HOLD']} ({hold_pct:.2f}%)"
        )

    # Log proba statistics if not using cached (to avoid duplicate logs)
    if not use_cached and len(proba_long_arr) > 0:
        logger.info(
            f"[ML Backtest][{adapter.name}] Proba statistics (LONG model): "
            f"mean={proba_long_arr.mean():.4f}, min={proba_long_arr.min():.4f}, "
            f"max={proba_long_arr.max():.4f}, std={proba_long_arr.std():.4f}"
        )
        if proba_short_arr is not None and len(proba_short_arr) > 0:
            logger.info(
                f"[ML Backtest][{adapter.name}] Proba statistics (SHORT model): "
                f"mean={proba_short_arr.mean():.4f}, min={proba_short_arr.min():.4f}, "
                f"max={proba_short_arr.max():.4f}, std={proba_short_arr.std():.4f}"
            )
        logger.info(
            f"[ML Backtest][{adapter.name}] Proba distribution vs thresholds: "
            f"above_long_threshold={np.sum(proba_long_arr >= long_threshold)}, "
            f"above_short_threshold={np.sum(proba_short_arr >= short_threshold) if (proba_short_arr is not None and short_threshold is not None) else 0}, "
            f"in_between={np.sum((proba_long_arr < long_threshold) & ((proba_short_arr < short_threshold) if (proba_short_arr is not None and short_threshold is not None) else True))}"
        )

    # Apply signal smoothing (confirmation bars)
    df["raw_signal"] = df["signal"].copy()
    if signal_confirmation_bars > 1:
        confirmed_signals = []
        raw_signal_list = df["raw_signal"].tolist()
        
        for i in range(len(df)):
            if i < signal_confirmation_bars - 1:
                # Not enough history, use raw signal
                confirmed_signals.append(raw_signal_list[i])
            else:
                # Check last N signals
                recent_signals = raw_signal_list[i - signal_confirmation_bars + 1 : i + 1]
                if all(s == "LONG" for s in recent_signals):
                    confirmed_signals.append("LONG")
                elif all(s == "SHORT" for s in recent_signals):
                    confirmed_signals.append("SHORT")
                else:
                    confirmed_signals.append("HOLD")
        
        df["signal"] = confirmed_signals
    else:
        # signal_confirmation_bars == 1, no smoothing needed
        pass
    
    # Apply trend filter (EMA-based)
    if use_trend_filter:
        # Calculate EMA
        df["trend_ema"] = df["close"].ewm(span=trend_ema_window, adjust=False).mean()
        
        # Filter signals based on trend
        filtered_signals = []
        for row in df.itertuples():
            signal_val = getattr(row, "signal")
            close_val = float(getattr(row, "close"))
            ema_val = float(getattr(row, "trend_ema"))
            
            if signal_val == "LONG" and close_val < ema_val:
                # LONG signal but price below EMA (downtrend), filter out
                filtered_signals.append("HOLD")
            elif signal_val == "SHORT" and close_val > ema_val:
                # SHORT signal but price above EMA (uptrend), filter out
                filtered_signals.append("HOLD")
            else:
                filtered_signals.append(signal_val)
        
        df["signal"] = filtered_signals
    else:
        # No trend filter, use confirmed signal as-is
        pass
    
    # Log strategy options
    if not use_cached:
        logger.info(
            f"[ML Backtest][{adapter.name}] Signal options: "
            f"signal_confirmation_bars={signal_confirmation_bars}, "
            f"use_trend_filter={use_trend_filter}, "
            f"trend_ema_window={trend_ema_window if use_trend_filter else 'N/A'}, "
            f"take_profit_pct={take_profit_pct if take_profit_pct is not None else 'None'}, "
            f"stop_loss_pct={stop_loss_pct if stop_loss_pct is not None else 'None'}"
        )
        if use_trend_filter:
            logger.info(
                f"[ML Backtest][{adapter.name}] Trend filter: use_trend_filter=True, ema_window={trend_ema_window}"
            )

    trades: List[Trade] = []
    equity_curve: List[float] = []
    position: dict | None = None
    balance = 1.0
    entries_attempted = 0
    exits_executed = 0
    tp_exits = 0
    sl_exits = 0
    
    # Phase E: Additional tracking
    position_entry_bar_index: int | None = None  # Track when position was opened
    daily_balance_start: float = 1.0  # Track daily balance for kill switch
    current_date: str | None = None  # Track current date for daily reset
    trading_disabled: bool = False  # Kill switch flag
    
    # Phase E: Confidence filter - compute rolling percentile if enabled
    confidence_proba_history: list[float] = []  # Store recent probabilities for confidence filter
    confidence_window = 100  # Window size for confidence percentile
    
    if not use_cached:
        logger.debug(
            f"[ML Backtest][{adapter.name}] Starting trade execution loop: {len(df)} rows to process"
        )
        if max_holding_bars is not None:
            logger.info(f"[ML Backtest][{adapter.name}] Max holding bars: {max_holding_bars}")
        if use_confidence_filter:
            logger.info(f"[ML Backtest][{adapter.name}] Confidence filter: enabled (quantile={confidence_quantile})")
        if daily_loss_limit is not None:
            logger.info(f"[ML Backtest][{adapter.name}] Daily loss limit: {daily_loss_limit}")

    for i, row in enumerate(df.itertuples()):
        signal: Signal = getattr(row, "signal")
        current_price = float(getattr(row, "close"))
        current_high = float(getattr(row, "high")) if hasattr(row, "high") else current_price
        current_low = float(getattr(row, "low")) if hasattr(row, "low") else current_price
        row_timestamp = getattr(row, "timestamp")
        
        # Phase E: Daily loss limit (kill switch) - reset daily balance at start of new day
        if daily_loss_limit is not None:
            row_date = str(row_timestamp).split(" ")[0] if isinstance(row_timestamp, str) else str(row_timestamp).split("T")[0]
            if current_date is None:
                current_date = row_date
                daily_balance_start = balance
            elif row_date != current_date:
                # New day - reset daily tracking
                current_date = row_date
                daily_balance_start = balance
                trading_disabled = False  # Re-enable trading for new day
            
            # Check if daily loss limit exceeded
            daily_return = (balance - daily_balance_start) / daily_balance_start
            if daily_return <= -daily_loss_limit:
                trading_disabled = True
                if position is not None:
                    # Force close position if kill switch triggered
                    exit_reason = "kill_switch"
                    exit_price = current_price
                else:
                    # Skip entry if kill switch active
                    continue

        # Phase E: Check max holding bars
        exit_reason: str | None = None
        if position is not None:
            # Phase E: Check max holding bars
            if max_holding_bars is not None and position_entry_bar_index is not None:
                bars_held = i - position_entry_bar_index
                if bars_held >= max_holding_bars:
                    exit_reason = "max_holding"
                    exit_price = current_price
            entry_price = position["entry_price"]
            direction: Literal["LONG", "SHORT"] = position["side"]
            
            # Calculate current return
            if direction == "LONG":
                # For LONG: use high price for TP check, low price for SL check
                current_return_tp = (current_high / entry_price) - 1
                current_return_sl = (current_low / entry_price) - 1
            else:
                # For SHORT: use low price for TP check, high price for SL check
                current_return_tp = (entry_price / current_low) - 1
                current_return_sl = (entry_price / current_high) - 1
            
            # Check TP first (only if not already exiting due to max_holding)
            if exit_reason is None:
                if take_profit_pct is not None and current_return_tp >= take_profit_pct:
                    exit_reason = "tp"
                    exit_price = current_high if direction == "LONG" else current_low
                # Check SL
                elif stop_loss_pct is not None and current_return_sl <= -stop_loss_pct:
                    exit_reason = "sl"
                    exit_price = current_low if direction == "LONG" else current_high

        # Execute exit if TP/SL triggered
        if exit_reason is not None:
            entry_price = position["entry_price"]
            direction: Literal["LONG", "SHORT"] = position["side"]

            # Apply commission and slippage (use override if provided)
            effective_commission_rate = commission_rate if commission_rate is not None else DEFAULT_COMMISSION_RATE
            effective_slippage_rate = slippage_rate if slippage_rate is not None else DEFAULT_SLIPPAGE_RATE
            
            # Entry: apply commission + slippage
            entry_cost = entry_price * (effective_commission_rate + effective_slippage_rate)
            # Exit: apply commission + slippage
            exit_cost = exit_price * (effective_commission_rate + effective_slippage_rate)
            
            if direction == "LONG":
                # Long: buy at entry_price + costs, sell at exit_price - costs
                effective_entry = entry_price + entry_cost
                effective_exit = exit_price - exit_cost
                profit = (effective_exit - effective_entry) / effective_entry
            else:
                # Short: sell at entry_price - costs, buy at exit_price + costs
                effective_entry = entry_price - entry_cost
                effective_exit = exit_price + exit_cost
                profit = (effective_entry - effective_exit) / effective_entry

            balance *= 1 + profit

            trades.append(
                Trade(
                    entry_time=position["entry_time"],
                    exit_time=str(getattr(row, "timestamp")),
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
            
            # After exit, check if we should enter new position based on signal
            if signal in ("LONG", "SHORT"):
                position = {
                    "side": signal,
                    "entry_price": current_price,
                    "entry_time": str(getattr(row, "timestamp")),
                }
                position_entry_bar_index = i  # Phase E: Track entry bar index
                entries_attempted += 1
        
        # Entry logic (if no exit was triggered)
        if exit_reason is None:
            if position is None:
                # No position, check for entry
                # Phase E: Skip entry if kill switch is active
                if trading_disabled:
                    continue
                
                if signal not in ("LONG", "SHORT"):
                    continue
                
                # Phase E: Confidence filter - only enter if probability is in top X percentile
                if use_confidence_filter:
                    # Get current probability (use LONG or SHORT based on signal)
                    current_proba = None
                    if signal == "LONG" and proba_long_arr is not None and i < len(proba_long_arr):
                        current_proba = proba_long_arr[i]
                    elif signal == "SHORT" and proba_short_arr is not None and i < len(proba_short_arr):
                        current_proba = proba_short_arr[i]
                    
                    if current_proba is not None:
                        # Update history
                        confidence_proba_history.append(current_proba)
                        if len(confidence_proba_history) > confidence_window:
                            confidence_proba_history.pop(0)
                        
                        # Check if we have enough history
                        if len(confidence_proba_history) >= 10:  # Minimum history required
                            percentile_threshold = np.percentile(confidence_proba_history, confidence_quantile * 100)
                            if current_proba < percentile_threshold:
                                # Probability too low, skip entry
                                continue
                
                # Enter position
                if signal in ("LONG", "SHORT"):
                    position = {
                        "side": signal,
                        "entry_price": current_price,
                        "entry_time": str(getattr(row, "timestamp")),
                    }
                    position_entry_bar_index = i  # Phase E: Track entry bar index
                    entries_attempted += 1
            else:
                # Position exists, check for signal-based exit
                if signal == "HOLD" or signal != position["side"]:
                    exit_price = current_price
                    entry_price = position["entry_price"]
                    direction: Literal["LONG", "SHORT"] = position["side"]

                    # Apply commission and slippage (use override if provided)
                    effective_commission_rate = commission_rate if commission_rate is not None else DEFAULT_COMMISSION_RATE
                    effective_slippage_rate = slippage_rate if slippage_rate is not None else DEFAULT_SLIPPAGE_RATE
                    
                    # Entry: apply commission + slippage
                    entry_cost = entry_price * (effective_commission_rate + effective_slippage_rate)
                    # Exit: apply commission + slippage
                    exit_cost = exit_price * (effective_commission_rate + effective_slippage_rate)
                    
                    if direction == "LONG":
                        # Long: buy at entry_price + costs, sell at exit_price - costs
                        effective_entry = entry_price + entry_cost
                        effective_exit = exit_price - exit_cost
                        profit = (effective_exit - effective_entry) / effective_entry
                    else:
                        # Short: sell at entry_price - costs, buy at exit_price + costs
                        effective_entry = entry_price - entry_cost
                        effective_exit = exit_price + exit_cost
                        profit = (effective_entry - effective_exit) / effective_entry

                    balance *= 1 + profit

                    trades.append(
                        Trade(
                            entry_time=position["entry_time"],
                            exit_time=str(getattr(row, "timestamp")),
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

    logger.info(
        f"[ML Backtest][{adapter.name}] Trade execution summary: "
        f"entries_attempted={entries_attempted}, exits_executed={exits_executed}, "
        f"final_trades={len(trades)}, final_position={'OPEN' if position is not None else 'CLOSED'}"
    )
    if take_profit_pct is not None or stop_loss_pct is not None:
        logger.info(
            f"[ML Backtest][{adapter.name}] TP/SL exits: tp={tp_exits}, sl={sl_exits}, "
            f"signal_exits={exits_executed - tp_exits - sl_exits}"
        )

    if position is not None:
        last_row = df.iloc[-1]
        last_price = float(last_row["close"])
        entry_price = position["entry_price"]
        direction: Literal["LONG", "SHORT"] = position["side"]

        # Apply commission and slippage for closing position (use override if provided)
        effective_commission_rate = commission_rate if commission_rate is not None else DEFAULT_COMMISSION_RATE
        effective_slippage_rate = slippage_rate if slippage_rate is not None else DEFAULT_SLIPPAGE_RATE
        
        entry_cost = entry_price * (effective_commission_rate + effective_slippage_rate)
        exit_cost = last_price * (effective_commission_rate + effective_slippage_rate)
        
        if direction == "LONG":
            effective_entry = entry_price + entry_cost
            effective_exit = last_price - exit_cost
            profit = (effective_exit - effective_entry) / effective_entry
        else:
            effective_entry = entry_price - entry_cost
            effective_exit = last_price + exit_cost
            profit = (effective_entry - effective_exit) / effective_entry

        balance *= 1 + profit

        trades.append(
            Trade(
                entry_time=position["entry_time"],
                exit_time=str(last_row["timestamp"]),
                entry_price=entry_price,
                exit_price=last_price,
                direction=direction,
                profit=profit,
            )
        )
        equity_curve.append(balance)

    if trades:
        wins = sum(1 for trade in trades if trade["profit"] is not None and trade["profit"] > 0)
        win_rate = wins / len(trades)
    else:
        win_rate = 0.0

    max_drawdown = 0.0
    running_max = float("-inf")

    for value in equity_curve:
        running_max = max(running_max, value)
        if running_max == 0:
            continue
        drawdown = (value - running_max) / running_max
        max_drawdown = min(max_drawdown, drawdown)

    stats = _compute_trade_stats(trades)

    if not use_cached:
        # Log commission and slippage settings
        logger.info(
            f"[ML Backtest][{adapter.name}] Completed: "
            f"total_trades={stats['total_trades']}, long={stats['long_trades']}, short={stats['short_trades']}, "
            f"win_rate={win_rate:.2%}, total_return={balance - 1.0:.2%}, "
            f"max_drawdown={max_drawdown:.2%}"
        )
        logger.info(
            f"[ML Backtest][{adapter.name}] Commission rate: {DEFAULT_COMMISSION_RATE:.4f} ({DEFAULT_COMMISSION_RATE*100:.2f}%), "
            f"Slippage rate: {DEFAULT_SLIPPAGE_RATE:.4f} ({DEFAULT_SLIPPAGE_RATE*100:.2f}%)"
        )

    return BacktestResult(
        total_return=balance - 1.0,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
        trades=trades,
        equity_curve=equity_curve,
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


def run_backtest_compare(
    threshold_up: float | None = None,
    threshold_down: float | None = None,
) -> dict:
    """
    simple / XGB-ML / DL-LSTM-Attn 3가지 전략을 한 번에 백테스트하고
    핵심 메트릭만 요약해서 반환한다.
    
    Args:
        threshold_up: Optional override for LSTM up-threshold (default: from settings)
        threshold_down: Optional override for LSTM down-threshold (default: from settings)
    
    Returns:
        dict with keys: simple, xgb_ml, dl_lstm_attn
        Each contains summarized backtest metrics.
    """
    simple_res = run_backtest()
    ml_res = run_backtest_with_ml()
    dl_res = run_backtest_with_dl_lstm_attn(
        threshold_up=threshold_up,
        threshold_down=threshold_down,
    )

    def summarize(name: str, res: BacktestResult) -> dict:
        return {
            "strategy": name,
            "total_return": res["total_return"],
            "win_rate": res["win_rate"],
            "max_drawdown": res["max_drawdown"],
            "total_trades": res["total_trades"],
            "long_trades": res["long_trades"],
            "short_trades": res["short_trades"],
            "avg_profit": res["avg_profit"],
            "median_profit": res["median_profit"],
            "avg_win": res["avg_win"],
            "avg_loss": res["avg_loss"],
            "max_consecutive_wins": res["max_consecutive_wins"],
            "max_consecutive_losses": res["max_consecutive_losses"],
        }

    return {
        "simple": summarize("simple_ema_rsi", simple_res),
        "xgb_ml": summarize("xgb_ml", ml_res),
        "dl_lstm_attn": summarize("dl_lstm_attn", dl_res),
    }


def run_backtest_with_dl_lstm_attn(
    threshold_up: float | None = None,
    threshold_down: float | None = None,
    *,
    use_optimized_thresholds: bool | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> BacktestResult:
    """
    Compatibility wrapper that routes to the generic ML backtester using the LSTM adapter.
    """
    # Convert use_optimized_thresholds to use_optimized_threshold for compatibility
    use_optimized_threshold = use_optimized_thresholds if use_optimized_thresholds is not None else False
    return run_backtest_with_ml(
        long_threshold=threshold_up,
        short_threshold=threshold_down,
        use_optimized_threshold=use_optimized_threshold,
        strategy_name="ml_lstm_attn",
        symbol=symbol,
        timeframe=timeframe,
    )


def run_backtest_compare(
    threshold_up: float | None = None,
    threshold_down: float | None = None,
) -> dict:
    """
    simple / XGB-ML / DL-LSTM-Attn 3가지 전략을 한 번에 백테스트하고
    핵심 메트릭만 요약해서 반환한다.
    
    Args:
        threshold_up: Optional override for LSTM up-threshold (default: from settings)
        threshold_down: Optional override for LSTM down-threshold (default: from settings)
    
    Returns:
        dict with keys: simple, xgb_ml, dl_lstm_attn
        Each contains summarized backtest metrics.
    """
    simple_res = run_backtest()
    ml_res = run_backtest_with_ml()
    dl_res = run_backtest_with_dl_lstm_attn(
        threshold_up=threshold_up,
        threshold_down=threshold_down,
    )

    def summarize(name: str, res: BacktestResult) -> dict:
        return {
            "strategy": name,
            "total_return": res["total_return"],
            "win_rate": res["win_rate"],
            "max_drawdown": res["max_drawdown"],
            "total_trades": res["total_trades"],
            "long_trades": res["long_trades"],
            "short_trades": res["short_trades"],
            "avg_profit": res["avg_profit"],
            "median_profit": res["median_profit"],
            "avg_win": res["avg_win"],
            "avg_loss": res["avg_loss"],
            "max_consecutive_wins": res["max_consecutive_wins"],
            "max_consecutive_losses": res["max_consecutive_losses"],
        }

    return {
        "simple": summarize("simple_ema_rsi", simple_res),
        "xgb_ml": summarize("xgb_ml", ml_res),
        "dl_lstm_attn": summarize("dl_lstm_attn", dl_res),
    }
