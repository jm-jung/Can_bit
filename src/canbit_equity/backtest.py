"""Next-open execution backtest for LONG/FLAT equity strategies."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def signals_to_positions(signal: pd.Series) -> pd.Series:
    """Signal at t close → position from t+1 open (shift by 1)."""
    return signal.fillna(0.0).clip(0.0, 1.0).shift(1).fillna(0.0)


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    cost_bps_per_side: float = 5.0,
    initial_equity: float = 1.0,
) -> Dict[str, Any]:
    d = df.copy().reset_index(drop=True)
    # Align by position so iloc-sliced frames with non-zero indexes still work
    sig = pd.Series(np.asarray(signal, dtype=float), index=d.index).fillna(0.0).clip(0.0, 1.0)
    pos = signals_to_positions(sig)
    open_ = d["open_adj"].astype(float)
    close = d["close_adj"].astype(float)
    # open-to-close return while in position; plus overnight from previous close to open when already in position
    # To avoid double counting: use close-to-close when holding continuously, with cost on turnover.
    # Spec: entry next open. Simple consistent approach:
    # daily_ret = position_{t} * (close_t / open_t - 1)  for days entered same morning
    # + position continuity overnight: if position_t == position_{t-1} == 1, include open_t/close_{t-1}-1
    # Equivalent close-to-close when always long after first entry, with first day open-to-close.
    prev_close = close.shift(1)
    overnight = open_ / prev_close - 1.0
    intraday = close / open_ - 1.0
    held_prev = pos.shift(1).fillna(0.0)
    day_ret = pos * intraday + np.minimum(pos, held_prev) * overnight.fillna(0.0)

    turnover = (pos - held_prev).abs()
    cost = turnover * (cost_bps_per_side / 10000.0)
    net_ret = day_ret - cost

    equity = (1.0 + net_ret.fillna(0.0)).cumprod() * initial_equity
    gross_ret = day_ret.fillna(0.0)
    equity_gross = (1.0 + gross_ret).cumprod() * initial_equity

    trades = int((turnover > 0).sum())
    long_exposure = float(pos.mean())
    return {
        "session_date": d["session_date"],
        "signal": sig,
        "position": pos,
        "net_ret": net_ret.fillna(0.0),
        "gross_ret": gross_ret,
        "equity": equity,
        "equity_gross": equity_gross,
        "turnover": turnover,
        "trade_count": trades,
        "long_exposure": long_exposure,
        "cost_bps_per_side": cost_bps_per_side,
    }


def summarize_equity(bt: Dict[str, Any], ann_factor: float = 252.0) -> Dict[str, Any]:
    rets = bt["net_ret"].astype(float)
    eq = bt["equity"].astype(float)
    n = max(len(rets), 1)
    years = n / ann_factor
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) else 0.0
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) if years > 0 and eq.iloc[0] > 0 else 0.0
    vol = float(rets.std() * np.sqrt(ann_factor)) if n > 2 else 0.0
    sharpe = float(rets.mean() / rets.std() * np.sqrt(ann_factor)) if rets.std() > 0 else 0.0
    downside = rets.clip(upper=0.0)
    sortino = float(rets.mean() / downside.std() * np.sqrt(ann_factor)) if downside.std() > 0 else 0.0
    peak = eq.cummax()
    dd = eq / peak - 1.0
    mdd = float(dd.min()) if len(dd) else 0.0
    # duration
    under = dd < 0
    dur = 0
    max_dur = 0
    for x in under:
        if x:
            dur += 1
            max_dur = max(max_dur, dur)
        else:
            dur = 0
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0
    return {
        "total_return": total,
        "CAGR": cagr,
        "annualized_volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "max_drawdown": mdd,
        "max_drawdown_duration": max_dur,
        "Calmar": calmar,
        "long_exposure": float(bt["long_exposure"]),
        "turnover": float(bt["turnover"].sum()),
        "trade_count": int(bt["trade_count"]),
        "sessions": int(n),
    }
