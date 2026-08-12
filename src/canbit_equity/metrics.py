"""Comparison metrics vs buy-and-hold."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .backtest import run_backtest, summarize_equity, signals_to_positions
from .strategies import signal_buy_and_hold


def compare_to_buyhold(df: pd.DataFrame, signal: pd.Series, cost_bps: float) -> Dict[str, Any]:
    strat = run_backtest(df, signal, cost_bps_per_side=cost_bps)
    bh = run_backtest(df, signal_buy_and_hold(df), cost_bps_per_side=cost_bps)
    s = summarize_equity(strat)
    b = summarize_equity(bh)
    cagr_pres = (s["CAGR"] / b["CAGR"] * 100.0) if b["CAGR"] not in (0, None) and np.isfinite(b["CAGR"]) else None
    mdd_imp = None
    if b["max_drawdown"] < 0:
        # improvement: strategy less negative than BH
        mdd_imp = (abs(b["max_drawdown"]) - abs(s["max_drawdown"])) / abs(b["max_drawdown"]) * 100.0
    return {
        "strategy": s,
        "buy_hold": b,
        "cagr_preservation_pct": cagr_pres,
        "mdd_improvement_pct": mdd_imp,
        "strategy_equity": strat["equity"],
        "buyhold_equity": bh["equity"],
        "strategy_position": strat["position"],
        "session_date": strat["session_date"],
    }
