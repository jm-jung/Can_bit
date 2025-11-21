"""
Simple backtesting engine for EMA + RSI strategy.
"""
from __future__ import annotations

import logging
from typing import List, Literal, TypedDict

import numpy as np
import pandas as pd

from src.core.config import settings
from src.indicators.basic import get_df_with_indicators
from src.ml.xgb_model import get_xgb_model
from src.dl.lstm_attn_model import get_lstm_attn_model

logger = logging.getLogger(__name__)


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

def run_backtest_with_ml() -> BacktestResult:
    """
    Run backtest using XGBoost ML strategy.
    """
    df: pd.DataFrame = get_df_with_indicators().copy()

    # Get ML model
    model = get_xgb_model()
    if model is None or not model.is_loaded():
        # Return empty result if model not available
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

    # Generate ML signals
    df["signal"] = "HOLD"
    try:
        for i in range(len(df)):
            df_slice = df.iloc[: i + 1]
            try:
                proba_up = model.predict_proba_latest(df_slice)
                if proba_up >= 0.55:
                    df.loc[df.index[i], "signal"] = "LONG"
                elif proba_up <= 0.45:
                    df.loc[df.index[i], "signal"] = "SHORT"
            except Exception:
                # Skip if prediction fails
                continue
    except Exception:
        # If ML prediction fails entirely, return empty result
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

    trades: List[Trade] = []
    equity_curve: List[float] = []

    position: dict | None = None
    balance = 1.0

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

    # Close remaining position
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


def run_backtest_with_dl_lstm_attn(
    threshold_up: float | None = None,
    threshold_down: float | None = None,
) -> BacktestResult:
    """
    Run backtest using LSTM + Attention deep learning strategy.
    
    NOTE: threshold_up/down can be overridden via API query params now.
    If not provided, uses settings.LSTM_ATTN_THRESHOLD_UP/DOWN.
    
    Args:
        threshold_up: Optional override for LSTM up-threshold (default: from settings)
        threshold_down: Optional override for LSTM down-threshold (default: from settings)
    """
    logger.info("[DEBUG] run_backtest_with_dl_lstm_attn actually executing")
    logger.info("[LSTM Backtest] >>> run_backtest_with_dl_lstm_attn called")
    df: pd.DataFrame = get_df_with_indicators().copy()

    # Get DL model
    model = get_lstm_attn_model()
    if model is None:
        logger.warning(
            "DL model instance is None. Cannot run DL backtest. "
            "Please check if the model file exists and can be loaded."
        )
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
    
    if not model.is_loaded():
        logger.warning(
            f"DL model not available (file not found or loading failed). "
            f"Expected path: {model.model_path.resolve()}. "
            f"Skipping DL backtest. Please train the model first."
        )
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
    
    logger.info("DL model loaded successfully, starting backtest...")

    # Use provided thresholds or fall back to settings
    up = threshold_up if threshold_up is not None else settings.LSTM_ATTN_THRESHOLD_UP
    down = threshold_down if threshold_down is not None else settings.LSTM_ATTN_THRESHOLD_DOWN
    logger.info(
        "[LSTM Backtest] Using thresholds: up=%.3f, down=%.3f",
        up,
        down,
    )

    # Generate DL signals
    df["signal"] = "HOLD"
    signal_counts = {"LONG": 0, "SHORT": 0, "HOLD": 0}
    
    # 통계 수집용 딕셔너리 초기화
    stats = {
        "long": 0,
        "short": 0,
        "hold": 0,
        "probs": [],
    }
    
    try:
        window_size = model.window_size
        logger.info(f"[LSTM Backtest] Window size: {window_size}, Total bars: {len(df)}")
        
        # Need at least window_size rows to make predictions
        for i in range(window_size, len(df)):
            df_slice = df.iloc[: i + 1]
            try:
                # prob_up 수집을 위해 predict_proba_latest 먼저 호출
                proba_up = model.predict_proba_latest(df_slice)
                stats["probs"].append(proba_up)
                
                # Use predict_label_latest for consistent signal generation
                signal = model.predict_label_latest(
                    df_slice,
                    threshold_up=up,
                    threshold_down=down,
                )
                df.loc[df.index[i], "signal"] = signal
                signal_counts[signal] += 1
                
                # stats 업데이트
                if signal == "LONG":
                    stats["long"] += 1
                elif signal == "SHORT":
                    stats["short"] += 1
                else:
                    stats["hold"] += 1
                
                # 디버그 로깅: 처음 200개 bar와 일정 간격으로 샘플링
                if i < window_size + 200 or (i - window_size) % 500 == 0:
                    logger.debug(
                        f"[LSTM Backtest] Bar {i}: prob_up={proba_up:.4f}, "
                        f"signal={signal}, close={df.iloc[i]['close']:.2f}"
                    )
            except Exception as e:
                # Skip if prediction fails
                logger.debug(f"[LSTM Backtest] Prediction failed at bar {i}: {e}")
                continue
    except Exception as e:
        logger.error(f"[LSTM Backtest] Signal generation failed: {e}")
        # If DL prediction fails entirely, return empty result
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
    
    # Signal 통계 로깅
    total_signals = sum(signal_counts.values())
    logger.info(
        f"[LSTM Backtest] Signal distribution: "
        f"LONG={signal_counts['LONG']}, SHORT={signal_counts['SHORT']}, "
        f"HOLD={signal_counts['HOLD']} (total={total_signals})"
    )
    
    # Prediction Stats 출력
    logger.info("---- Prediction Stats ----")
    logger.info("LONG : %d", stats["long"])
    logger.info("SHORT: %d", stats["short"])
    logger.info("HOLD : %d", stats["hold"])
    
    if len(stats["probs"]) > 0:
        probs = np.array(stats["probs"])
        logger.info("prob_up mean: %.5f", probs.mean())
        logger.info("prob_up min : %.5f", probs.min())
        logger.info("prob_up max : %.5f", probs.max())
        logger.info("prob_up std : %.5f", probs.std())
    else:
        logger.warning("No prob_up values collected")
    logger.info("--------------------------")

    trades: List[Trade] = []
    equity_curve: List[float] = []

    position: dict | None = None
    balance = 1.0

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

    # Close remaining position
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

    # 백테스트 결과 요약 로깅
    logger.info(
        f"[LSTM Backtest] Completed: "
        f"total_trades={stats['total_trades']}, long={stats['long_trades']}, short={stats['short_trades']}, "
        f"win_rate={win_rate:.2%}, total_return={balance - 1.0:.2%}, "
        f"max_drawdown={max_drawdown:.2%}"
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
