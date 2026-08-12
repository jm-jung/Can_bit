"""
Common utilities for lifecycle diagnostics (analysis-only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = REPO_ROOT / "data" / "state" / "paper_trading_state.json"
MONITORING_DIR = REPO_ROOT / "data" / "monitoring"
OUT_DIR = REPO_ROOT / "data" / "diagnostics" / "lifecycle"

DEFAULT_POSITION_SIZE = 0.05
DEFAULT_FEE_RATE = 0.0004
DEFAULT_SLIPPAGE_RATE = 0.0002
KS_STREAK_THRESHOLD = 5


@dataclass
class TickEvent:
    idx: int
    ts: str
    price: float
    decision: str
    reason: str
    strategy: str
    entropy: Optional[float]
    activation: Optional[bool]
    vol_bucket: Optional[str]
    trend_state: Optional[str]
    enter_direction: Optional[str]


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        lv = v.strip().lower()
        if lv in {"true", "1", "yes"}:
            return True
        if lv in {"false", "0", "no"}:
            return False
    return None


def _iter_log_paths() -> List[Path]:
    return sorted(MONITORING_DIR.glob("paper_trading_log_*.jsonl"))


def load_state() -> Dict[str, Any]:
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def load_state_trades() -> List[Dict[str, Any]]:
    st = load_state()
    tr = st.get("trades") or st.get("trade_history") or []
    return tr if isinstance(tr, list) else []


def load_tick_events() -> List[TickEvent]:
    events: List[TickEvent] = []
    idx = 0
    for p in _iter_log_paths():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") != "tick":
                    continue
                price = _safe_float(ev.get("price"))
                if price is None:
                    continue
                reason = str(ev.get("reason") or "")
                enter_dir = None
                if str(ev.get("decision") or "") == "enter":
                    enter_dir = "SHORT" if reason.startswith("SHORT") else "LONG"
                events.append(
                    TickEvent(
                        idx=idx,
                        ts=str(ev.get("ts") or ""),
                        price=float(price),
                        decision=str(ev.get("decision") or ""),
                        reason=reason,
                        strategy=str(ev.get("strategy") or ""),
                        entropy=_safe_float(ev.get("entropy")),
                        activation=_safe_bool(ev.get("activation")),
                        vol_bucket=(str(ev.get("vol_bucket")) if ev.get("vol_bucket") is not None else None),
                        trend_state=(str(ev.get("trend_state")) if ev.get("trend_state") is not None else None),
                        enter_direction=enter_dir,
                    )
                )
                idx += 1
    return events


def _select_latest_window_by_state(
    trades_df: pd.DataFrame,
    state_trades: List[Dict[str, Any]],
    raw_col: str = "raw_return",
) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df
    n = len(state_trades)
    if n <= 0 or len(trades_df) < n:
        return trades_df.copy()
    target = np.array([float(t.get("pnl", 0.0)) for t in state_trades], dtype=float)
    arr = trades_df[raw_col].to_numpy(dtype=float)
    best_start = len(trades_df) - n
    best_err = float("inf")
    for i in range(0, len(trades_df) - n + 1):
        err = float(np.mean(np.abs(arr[i : i + n] - target)))
        if err < best_err:
            best_err = err
            best_start = i
    return trades_df.iloc[best_start : best_start + n].reset_index(drop=True)


def reconstruct_actual_trades(events: List[TickEvent]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    open_pos: Optional[Dict[str, Any]] = None

    for ev in events:
        if open_pos is not None:
            open_pos["hold_bars"] += 1
            if open_pos["direction"] == "LONG":
                r = (ev.price - open_pos["entry_price"]) / open_pos["entry_price"]
            else:
                r = (open_pos["entry_price"] - ev.price) / open_pos["entry_price"]
            open_pos["mfe_approx"] = max(open_pos["mfe_approx"], r)
            open_pos["mae_approx"] = min(open_pos["mae_approx"], r)

        if ev.decision == "enter" and open_pos is None and ev.enter_direction is not None:
            open_pos = {
                "entry_idx": ev.idx,
                "entry_time": ev.ts,
                "entry_price": ev.price,
                "direction": ev.enter_direction,
                "strategy_used": ev.reason.rsplit("_", 1)[-1] if "_" in ev.reason else (ev.strategy or "UNKNOWN"),
                "entropy": ev.entropy,
                "activation_state": ev.activation,
                "vol_bucket": ev.vol_bucket,
                "trend_state": ev.trend_state,
                "hold_bars": 0,
                "mfe_approx": 0.0,
                "mae_approx": 0.0,
            }
        elif ev.decision == "exit" and open_pos is not None:
            if open_pos["direction"] == "LONG":
                raw = (ev.price - open_pos["entry_price"]) / open_pos["entry_price"]
            else:
                raw = (open_pos["entry_price"] - ev.price) / open_pos["entry_price"]
            rows.append(
                {
                    **open_pos,
                    "exit_idx": ev.idx,
                    "exit_time": ev.ts,
                    "exit_price": ev.price,
                    "exit_reason": ev.reason,
                    "raw_return": float(raw),
                }
            )
            open_pos = None

    return pd.DataFrame(rows)


def align_to_state_window(trades_df: pd.DataFrame) -> pd.DataFrame:
    return _select_latest_window_by_state(trades_df, load_state_trades(), raw_col="raw_return")


def compute_net_return(raw_return: float, fee_rate: float, slippage_rate: float) -> float:
    return float(raw_return) - 2.0 * (float(fee_rate) + float(slippage_rate))


def build_equity_curve(
    returns: List[float],
    position_size: float = DEFAULT_POSITION_SIZE,
    initial_equity: float = 1.0,
) -> List[float]:
    eq = float(initial_equity)
    curve: List[float] = []
    for r in returns:
        eq *= 1.0 + float(r) * float(position_size)
        curve.append(eq)
    return curve


def max_drawdown_from_curve(curve: List[float]) -> float:
    if not curve:
        return 0.0
    peak = curve[0]
    mdd = 0.0
    for eq in curve:
        peak = max(peak, eq)
        dd = (eq - peak) / peak if peak > 0 else 0.0
        mdd = min(mdd, dd)
    return float(mdd)


def streak_stats(returns: List[float]) -> Dict[str, Any]:
    max_streak = 0
    current = 0
    streak_sizes: List[int] = []
    for r in returns:
        if r < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            if current > 0:
                streak_sizes.append(current)
            current = 0
    if current > 0:
        streak_sizes.append(current)
    return {
        "max_consecutive_losses": int(max_streak),
        "avg_losing_streak": float(mean(streak_sizes)) if streak_sizes else 0.0,
        "ks_like_triggers": int(sum(1 for s in streak_sizes if s >= KS_STREAK_THRESHOLD)),
    }


def expectancy_stats(returns: List[float]) -> Dict[str, float]:
    vals = [float(x) for x in returns]
    if not vals:
        return {
            "trades": 0,
            "wr": 0.0,
            "expectancy": 0.0,
            "pf": 0.0,
            "avg_return": 0.0,
        }
    wins = [x for x in vals if x > 0]
    losses = [x for x in vals if x < 0]
    wr = len(wins) / len(vals)
    avg_win = mean(wins) if wins else 0.0
    avg_loss = mean(losses) if losses else 0.0
    expectancy = (wr * avg_win) - ((1.0 - wr) * abs(avg_loss))
    gross_profit = sum(wins)
    gross_loss_abs = abs(sum(losses))
    pf = gross_profit / gross_loss_abs if gross_loss_abs > 0 else 0.0
    return {
        "trades": float(len(vals)),
        "wr": float(wr),
        "expectancy": float(expectancy),
        "pf": float(pf),
        "avg_return": float(mean(vals)),
    }


def summarize_case(
    case_name: str,
    returns: List[float],
    position_size: float = DEFAULT_POSITION_SIZE,
) -> Dict[str, Any]:
    exp = expectancy_stats(returns)
    eq = build_equity_curve(returns, position_size=position_size, initial_equity=1.0)
    mdd = max_drawdown_from_curve(eq)
    streak = streak_stats(returns)
    out: Dict[str, Any] = {
        "case": case_name,
        "trades": int(exp["trades"]),
        "wr": exp["wr"],
        "expectancy": exp["expectancy"],
        "pf": exp["pf"],
        "avg_return": exp["avg_return"],
        "equity_return": (eq[-1] - 1.0) if eq else 0.0,
        "mdd_proxy": mdd,
        **streak,
    }
    return out


def ensure_out_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR
