"""
Production Trading Loop (Scheduler).

Usage:
    python -m scripts.run_trading_loop --mode shadow
    python -m scripts.run_trading_loop --mode paper
    python -m scripts.run_trading_loop --mode live

This is the top-level entry point for operational trading.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ops.execution_router import ExecutionRouter, TradingMode
from src.ops.monitor import Monitor
from src.ops.risk_manager import RiskLimits, RiskManager
from src.ops.state_manager import StateManager
from src.ops.trading_engine import TickContext, TradingEngine
from src.strategy_filters.regime_ablation import compute_regime_components

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("trading_loop")

TICK_INTERVAL_SEC = 60
SUMMARY_INTERVAL_TICKS = 60
SHUTDOWN_REQUESTED = False


def handle_signal(signum, frame):
    global SHUTDOWN_REQUESTED
    logger.info(f"[Loop] Received signal {signum}, requesting graceful shutdown...")
    SHUTDOWN_REQUESTED = True


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def load_latest_data() -> Optional[pd.DataFrame]:
    """
    Load the latest OHLCV data for feature computation.
    Looks for the most recent parquet file in data/ directory.
    """
    data_paths = [
        Path("data/btc_1h.parquet"),
        Path("data/ohlcv_1h.parquet"),
        Path("data/btc_data.parquet"),
    ]
    for p in data_paths:
        if p.exists():
            try:
                df = pd.read_parquet(p)
                if "close" in df.columns and len(df) > 200:
                    return df
            except Exception as e:
                logger.warning(f"[Loop] Failed to load {p}: {e}")

    csv_paths = list(Path("data").glob("*.csv"))
    for p in sorted(csv_paths, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            df = pd.read_csv(p)
            if "close" in df.columns and len(df) > 200:
                return df
        except Exception:
            continue

    return None


def compute_features(df: pd.DataFrame) -> dict:
    """
    Compute all features needed for the strategy pipeline from OHLCV data.
    Returns dict with vol_bucket, trend_label, probabilities, signal, etc.
    """
    ema_above, is_sideways, is_high_vol, is_mid_vol, is_low_vol, trend_label, _ = (
        compute_regime_components(df)
    )

    last_idx = len(df) - 1
    current_trend = str(trend_label[last_idx])

    if is_high_vol[last_idx]:
        current_vol = "high"
    elif is_low_vol[last_idx]:
        current_vol = "low"
    else:
        current_vol = "mid"

    close = df["close"].iloc[-1]

    features = {
        "price": float(close),
        "vol_bucket": current_vol,
        "trend_label": current_trend,
        "p_long": 0.33,
        "p_short": 0.33,
        "p_flat": 0.34,
        "signal": None,
        "exit_signal": False,
    }
    return features


def run_loop(mode: TradingMode, enable_discord: bool = False) -> None:
    """Main trading loop."""
    logger.info(f"[Loop] Starting trading loop: mode={mode}")

    state_mgr = StateManager()
    risk_mgr = RiskManager(state_mgr)
    router = ExecutionRouter(mode, state_mgr)
    monitor = Monitor(enable_discord=enable_discord)

    engine = TradingEngine(
        mode=mode,
        state_mgr=state_mgr,
        risk_mgr=risk_mgr,
        router=router,
        monitor=monitor,
        enable_discord=enable_discord,
    )

    tick_count = 0

    try:
        while not SHUTDOWN_REQUESTED:
            try:
                state_mgr.reset_daily()

                df = load_latest_data()
                if df is None:
                    logger.warning("[Loop] No data available, sleeping...")
                    time.sleep(TICK_INTERVAL_SEC)
                    continue

                features = compute_features(df)

                ctx = TickContext(
                    price=features["price"],
                    vol_bucket=features["vol_bucket"],
                    trend_label=features["trend_label"],
                    p_long=features["p_long"],
                    p_short=features["p_short"],
                    p_flat=features["p_flat"],
                    signal=features["signal"],
                    exit_signal=features["exit_signal"],
                )

                result = engine.process_tick(ctx)
                tick_count += 1

                if tick_count % SUMMARY_INTERVAL_TICKS == 0:
                    engine.log_periodic_summary()
                    transition_msg = engine.check_mode_transition()
                    if transition_msg:
                        logger.info(f"[Loop] MODE TRANSITION SUGGESTION: {transition_msg}")

                time.sleep(TICK_INTERVAL_SEC)

            except KeyboardInterrupt:
                break
            except Exception as e:
                msg = f"Tick error: {traceback.format_exc()}"
                logger.error(f"[Loop] {msg}")
                state_mgr.record_error(str(e))
                time.sleep(TICK_INTERVAL_SEC * 2)

    finally:
        logger.info("[Loop] Shutting down...")
        engine.shutdown()
        logger.info("[Loop] Shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="Can_bit Production Trading Loop")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["shadow", "paper", "live"],
        default="shadow",
        help="Trading mode",
    )
    parser.add_argument(
        "--discord",
        action="store_true",
        help="Enable Discord notifications",
    )
    args = parser.parse_args()

    if args.mode == "live":
        logger.warning("=" * 60)
        logger.warning("  ⚠️  LIVE MODE - REAL MONEY AT RISK  ⚠️")
        logger.warning("=" * 60)
        confirm = input("Type 'CONFIRM_LIVE' to proceed: ")
        if confirm.strip() != "CONFIRM_LIVE":
            logger.info("Live mode cancelled.")
            sys.exit(0)

    run_loop(mode=args.mode, enable_discord=args.discord)


if __name__ == "__main__":
    main()
