"""
Thin CLI wrapper for XGBoost threshold optimization.

This allows the following command to work:

    python -m src.optimization.optimize_xgb_thresholds --strategy ml_xgb --symbol BTCUSDT --timeframe 1m

Internally, it just delegates to `src.optimization.optimize_ml_threshold` module's CLI entry point.
"""

from __future__ import annotations

import logging

# Import the module that contains the CLI logic
from . import optimize_ml_threshold


def main() -> None:
    """Delegate CLI execution to `optimize_ml_threshold` module's __main__ block."""
    # The optimize_ml_threshold module has __main__ block that handles argparse and execution
    # We replicate that behavior here by calling the same function
    import argparse
    
    from src.optimization.optimize_ml_threshold import (
        STRATEGY_REGISTRY,
        run_threshold_optimization_for_ml_strategy,
        _parse_args,
    )
    
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    run_threshold_optimization_for_ml_strategy(
        strategy_name=args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        metric_name=args.metric,
        long_threshold_min=args.long_min,
        long_threshold_max=args.long_max,
        long_threshold_step=args.long_step,
        short_threshold_min=args.short_min,
        short_threshold_max=args.short_max,
        short_threshold_step=args.short_step,
        save_result=not args.no_save,
    )


if __name__ == "__main__":
    main()

