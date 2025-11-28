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
from src.indicators.basic import get_df_with_indicators
from src.ml.xgb_model import get_xgb_model
from src.dl.lstm_attn_model import get_lstm_attn_model
from src.strategies.ml_thresholds import resolve_ml_thresholds

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
    use_optimized_thresholds: bool | None = None,
    strategy_name: str = "ml_xgb",
    symbol: str | None = None,
    timeframe: str | None = None,
    proba_up_cache: np.ndarray | None = None,
    df_with_proba: pd.DataFrame | None = None,
    index_mask: np.ndarray | None = None,
) -> BacktestResult:
    """
    Run backtest using a pluggable ML strategy.
    
    Args:
        long_threshold: Probability threshold for LONG signal
        short_threshold: Probability threshold for SHORT signal
        use_optimized_thresholds: Whether to use optimized thresholds from JSON
        strategy_name: Strategy identifier (e.g., "ml_xgb", "ml_lstm_attn")
        symbol: Optional symbol override for threshold lookup
        timeframe: Optional timeframe override for threshold lookup
        proba_up_cache: Optional pre-computed probability array (for optimization)
        df_with_proba: Optional DataFrame aligned with proba_up_cache (for optimization)
        index_mask: Optional boolean mask to filter rows (for in-sample/out-of-sample splits)
    
    Returns:
        BacktestResult
    """
    if use_optimized_thresholds is None:
        use_optimized_thresholds = getattr(settings, "USE_OPTIMIZED_THRESHOLDS", False)

    adapter = _get_ml_adapter(strategy_name)
    
    # Use cached predictions if provided, otherwise compute on-the-fly
    use_cached = proba_up_cache is not None and df_with_proba is not None

    long_threshold, short_threshold = resolve_ml_thresholds(
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        use_optimized_thresholds=use_optimized_thresholds,
        strategy_name=adapter.strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        default_long=adapter.default_long,
        default_short=adapter.default_short,
    )
    
    # Use cached predictions if provided, otherwise compute on-the-fly
    use_cached = proba_up_cache is not None and df_with_proba is not None
    
    if use_cached:
        df = df_with_proba.copy()
        proba_arr = proba_up_cache.copy()
        logger.debug(
            f"[ML Backtest][{adapter.name}] Using cached predictions: "
            f"df_rows={len(df)}, proba_len={len(proba_arr)}"
        )
    else:
        df: pd.DataFrame = get_df_with_indicators().copy()
        model = adapter.get_model()

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

        if not getattr(model, "is_loaded", lambda: False)():
            model_path = getattr(model, "model_path", None)
            exists = model_path.exists() if model_path is not None else False
            return _empty_result(
                f"{adapter.name} model not loaded. Model path: {model_path}, exists={exists}"
            )

        proba_values: list[float] = []
        prediction_errors = 0

        logger.info(
            f"[ML Backtest][{adapter.name}] Computing predictions: "
            f"long_threshold={long_threshold}, short_threshold={short_threshold}, "
            f"total_rows={len(df)}"
        )

        try:
            min_rows_for_prediction = max(0, adapter.min_history_provider(model))
            start_idx = min_rows_for_prediction

            # Extract symbol and timeframe for event features
            # Use provided symbol/timeframe or fall back to settings
            backtest_symbol = symbol or getattr(settings, "BINANCE_SYMBOL", "BTC/USDT").replace("/", "").upper()
            backtest_timeframe = timeframe or getattr(settings, "THRESHOLD_TIMEFRAME", "1m")
            
            for i in range(start_idx, len(df)):
                df_slice = df.iloc[: i + 1]
                try:
                    proba_up = float(model.predict_proba_latest(df_slice, symbol=backtest_symbol, timeframe=backtest_timeframe))
                    proba_values.append(proba_up)
                except Exception as exc:
                    prediction_errors += 1
                    if prediction_errors <= 5:
                        logger.warning(
                            f"[ML Backtest][{adapter.name}] Prediction failed at index {i}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    continue
            
            if not proba_values:
                return _empty_result(
                    f"[ML Backtest][{adapter.name}] No successful predictions!"
                )
            
            proba_arr = np.array(proba_values, dtype=np.float32)
            # Align df with proba_arr (remove rows where prediction failed)
            df = df.iloc[start_idx:start_idx + len(proba_arr)].reset_index(drop=True)
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
            proba_arr = proba_arr[index_mask]
            logger.debug(
                f"[ML Backtest][{adapter.name}] Applied index_mask: "
                f"filtered to {len(df)} rows"
            )

    # Generate signals from probabilities
    df["signal"] = "HOLD"
    signal_counts = {"LONG": 0, "SHORT": 0, "HOLD": 0}
    
    for i in range(len(df)):
        proba_up = proba_arr[i]
        if proba_up >= long_threshold:
            df.loc[df.index[i], "signal"] = "LONG"
            signal_counts["LONG"] += 1
        elif short_threshold is not None and proba_up <= short_threshold:
            df.loc[df.index[i], "signal"] = "SHORT"
            signal_counts["SHORT"] += 1
        else:
            signal_counts["HOLD"] += 1
    
    if not use_cached:
        logger.info(
            f"[ML Backtest][{adapter.name}] Starting signal generation: "
            f"long_threshold={long_threshold}, short_threshold={short_threshold}, "
            f"total_rows={len(df)}"
        )
        logger.info(
            f"[ML Backtest][{adapter.name}] Signal generation summary: "
            f"signals={signal_counts}"
        )

    # Log proba statistics if not using cached (to avoid duplicate logs)
    if not use_cached and len(proba_arr) > 0:
        logger.info(
            f"[ML Backtest][{adapter.name}] Proba statistics: "
            f"mean={proba_arr.mean():.4f}, min={proba_arr.min():.4f}, "
            f"max={proba_arr.max():.4f}, std={proba_arr.std():.4f}"
        )
        lower_bound = short_threshold if short_threshold is not None else -1.0
        logger.info(
            f"[ML Backtest][{adapter.name}] Proba distribution vs thresholds: "
            f"above_long_threshold={np.sum(proba_arr >= long_threshold)}, "
            f"below_short_threshold={np.sum(proba_arr <= short_threshold) if short_threshold is not None else 0}, "
            f"in_middle={np.sum((proba_arr < long_threshold) & (proba_arr > lower_bound))}"
        )

    trades: List[Trade] = []
    equity_curve: List[float] = []
    position: dict | None = None
    balance = 1.0
    entries_attempted = 0
    exits_executed = 0

    if not use_cached:
        logger.debug(
            f"[ML Backtest][{adapter.name}] Starting trade execution loop: {len(df)} rows to process"
        )

    for row in df.itertuples():
        signal: Signal = getattr(row, "signal")

        if position is None:
            if signal in ("LONG", "SHORT"):
                position = {
                    "side": signal,
                    "entry_price": float(getattr(row, "close")),
                    "entry_time": str(getattr(row, "timestamp")),
                }
                entries_attempted += 1
        else:
            if signal == "HOLD" or signal != position["side"]:
                exit_price = float(getattr(row, "close"))
                entry_price = position["entry_price"]
                direction: Literal["LONG", "SHORT"] = position["side"]

                # Apply commission and slippage
                commission_rate = DEFAULT_COMMISSION_RATE
                slippage_rate = DEFAULT_SLIPPAGE_RATE
                
                # Entry: apply commission + slippage
                entry_cost = entry_price * (commission_rate + slippage_rate)
                # Exit: apply commission + slippage
                exit_cost = exit_price * (commission_rate + slippage_rate)
                
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

    logger.info(
        f"[ML Backtest][{adapter.name}] Trade execution summary: "
        f"entries_attempted={entries_attempted}, exits_executed={exits_executed}, "
        f"final_trades={len(trades)}, final_position={'OPEN' if position is not None else 'CLOSED'}"
    )

    if position is not None:
        last_row = df.iloc[-1]
        last_price = float(last_row["close"])
        entry_price = position["entry_price"]
        direction: Literal["LONG", "SHORT"] = position["side"]

        # Apply commission and slippage for closing position
        commission_rate = DEFAULT_COMMISSION_RATE
        slippage_rate = DEFAULT_SLIPPAGE_RATE
        
        entry_cost = entry_price * (commission_rate + slippage_rate)
        exit_cost = last_price * (commission_rate + slippage_rate)
        
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
    return run_backtest_with_ml(
        long_threshold=threshold_up,
        short_threshold=threshold_down,
        use_optimized_thresholds=use_optimized_thresholds,
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
