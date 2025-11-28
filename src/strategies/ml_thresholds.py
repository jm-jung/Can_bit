"""
Shared helpers for ML-based strategy thresholds.
"""
from __future__ import annotations

import logging
from typing import Tuple

from src.core.config import settings
from src.optimization.threshold_optimizer import load_threshold_result_for

logger = logging.getLogger(__name__)


def _resolve_threshold_identity(
    symbol: str | None,
    timeframe: str | None,
) -> Tuple[str, str]:
    """
    Resolve symbol/timeframe identifiers used for threshold loading.
    """
    resolved_symbol = symbol or getattr(settings, "THRESHOLD_SYMBOL", None)
    if resolved_symbol is None:
        resolved_symbol = getattr(settings, "BINANCE_SYMBOL", "BTC/USDT")
    resolved_timeframe = timeframe or getattr(settings, "THRESHOLD_TIMEFRAME", "1m")
    return resolved_symbol, resolved_timeframe


def resolve_ml_thresholds(
    *,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    use_optimized_thresholds: bool = True,
    strategy_name: str = "ml_xgb",
    symbol: str | None = None,
    timeframe: str | None = None,
    default_long: float = 0.5,
    default_short: float | None = None,
) -> tuple[float, float | None]:
    """
    Resolve the effective long/short thresholds with optional auto-loading.
    """
    resolved_symbol, resolved_timeframe = _resolve_threshold_identity(symbol, timeframe)

    resolved_long = long_threshold
    resolved_short = short_threshold

    should_use_optimized = (
        use_optimized_thresholds
        and getattr(settings, "USE_OPTIMIZED_THRESHOLDS", False)
        and (resolved_long is None or resolved_short is None)
    )

    if should_use_optimized:
        loaded = load_threshold_result_for(strategy_name, resolved_symbol, resolved_timeframe)
        if loaded:
            result, path = loaded
            if resolved_long is None and result.best_long_threshold is not None:
                resolved_long = result.best_long_threshold
            if resolved_short is None and result.best_short_threshold is not None:
                resolved_short = result.best_short_threshold
            logger.info(
                "[ML Threshold] Loaded optimized thresholds from %s "
                "(strategy=%s, symbol=%s, timeframe=%s): long=%.3f, short=%s, metric=%.6f",
                path,
                strategy_name,
                resolved_symbol,
                resolved_timeframe,
                result.best_long_threshold,
                f"{result.best_short_threshold:.3f}"
                if result.best_short_threshold is not None
                else "None",
                result.best_metric_value,
            )
        else:
            logger.info(
                "[ML Threshold] No optimized thresholds for %s/%s/%s. Using provided/default values.",
                strategy_name,
                resolved_symbol,
                resolved_timeframe,
            )

    if resolved_long is None:
        resolved_long = default_long
        logger.debug(
            "[ML Threshold] Falling back to default long threshold=%.3f "
            "(strategy=%s, symbol=%s, timeframe=%s)",
            default_long,
            strategy_name,
            resolved_symbol,
            resolved_timeframe,
        )

    if resolved_short is None and default_short is not None:
        resolved_short = default_short
        logger.debug(
            "[ML Threshold] Falling back to default short threshold=%.3f "
            "(strategy=%s, symbol=%s, timeframe=%s)",
            default_short,
            strategy_name,
            resolved_symbol,
            resolved_timeframe,
        )

    logger.debug(
        "[ML Threshold] Final thresholds (strategy=%s, symbol=%s, timeframe=%s): "
        "long=%.3f, short=%s",
        strategy_name,
        resolved_symbol,
        resolved_timeframe,
        resolved_long,
        f"{resolved_short:.3f}" if resolved_short is not None else "None",
    )

    return resolved_long, resolved_short


