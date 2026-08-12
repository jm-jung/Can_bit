"""
Paper trade lifecycle diagnostics (analysis-only).

Usage:
  python -m scripts.diagnostics.analyze_paper_trade_lifecycle
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = REPO_ROOT / "data" / "state" / "paper_trading_state.json"
MONITORING_DIR = REPO_ROOT / "data" / "monitoring"
OUT_DIR = REPO_ROOT / "data" / "diagnostics" / "paper"

PAPER_FEE_RATE = 0.0004
PAPER_SLIPPAGE_RATE = 0.0002
ROUND_TRIP_FEE = 2.0 * PAPER_FEE_RATE
ROUND_TRIP_SLIP = 2.0 * PAPER_SLIPPAGE_RATE
ROUND_TRIP_COST = ROUND_TRIP_FEE + ROUND_TRIP_SLIP


@dataclass
class OpenTrade:
    entry_idx: int
    entry_time: str
    entry_price: float
    direction: str
    strategy_used: str
    entropy: Optional[float]
    activation_state: Optional[bool]
    vol_bucket: Optional[str]
    trend_state: Optional[str]
    hold_bars: int = 0
    mfe: float = 0.0
    mae: float = 0.0


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


def _load_state_trades() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    trades = state.get("trades") or state.get("trade_history") or []
    if not isinstance(trades, list):
        trades = []
    return state, trades


def _iter_log_paths() -> List[Path]:
    return sorted(MONITORING_DIR.glob("paper_trading_log_*.jsonl"))


def _update_mfe_mae(open_trade: OpenTrade, current_price: float) -> None:
    if open_trade.entry_price <= 0:
        return
    if open_trade.direction == "LONG":
        ret = (current_price - open_trade.entry_price) / open_trade.entry_price
    else:
        ret = (open_trade.entry_price - current_price) / open_trade.entry_price
    open_trade.mfe = max(open_trade.mfe, ret)
    open_trade.mae = min(open_trade.mae, ret)


def _to_direction(reason: str) -> str:
    if reason.startswith("SHORT"):
        return "SHORT"
    return "LONG"


def _to_strategy(strategy_field: Any, reason: str) -> str:
    if isinstance(strategy_field, str) and strategy_field:
        return strategy_field
    if "_" in reason:
        return reason.rsplit("_", 1)[-1]
    return "UNKNOWN"


def _build_lifecycle_trades() -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    open_trade: Optional[OpenTrade] = None
    global_idx = 0

    for path in _iter_log_paths():
        with path.open("r", encoding="utf-8") as f:
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

                decision = str(ev.get("decision") or "")
                price = _safe_float(ev.get("price"))
                ts = str(ev.get("ts") or "")
                reason = str(ev.get("reason") or "")
                entropy = _safe_float(ev.get("entropy"))
                activation = _safe_bool(ev.get("activation"))
                strategy_field = ev.get("strategy")
                vol_bucket = ev.get("vol_bucket")
                trend_state = ev.get("trend_state")

                if open_trade is not None:
                    open_trade.hold_bars += 1
                    if price is not None:
                        _update_mfe_mae(open_trade, price)

                if decision == "enter" and price is not None:
                    open_trade = OpenTrade(
                        entry_idx=global_idx,
                        entry_time=ts,
                        entry_price=price,
                        direction=_to_direction(reason),
                        strategy_used=_to_strategy(strategy_field, reason),
                        entropy=entropy,
                        activation_state=activation,
                        vol_bucket=str(vol_bucket) if vol_bucket is not None else None,
                        trend_state=str(trend_state) if trend_state is not None else None,
                    )
                elif decision == "exit" and open_trade is not None and price is not None:
                    raw_return = _safe_float(ev.get("pnl"))
                    if raw_return is None:
                        if open_trade.direction == "LONG":
                            raw_return = (price - open_trade.entry_price) / open_trade.entry_price
                        else:
                            raw_return = (open_trade.entry_price - price) / open_trade.entry_price
                    fee_cost = ROUND_TRIP_FEE
                    slip_cost = ROUND_TRIP_SLIP
                    net_return = raw_return - (fee_cost + slip_cost)
                    records.append(
                        {
                            "entry_idx": open_trade.entry_idx,
                            "exit_idx": global_idx,
                            "entry_time": open_trade.entry_time,
                            "exit_time": ts,
                            "entry_price": open_trade.entry_price,
                            "exit_price": price,
                            "direction": open_trade.direction,
                            "strategy_used": open_trade.strategy_used,
                            "activation_state": open_trade.activation_state,
                            "entropy": open_trade.entropy,
                            "vol_bucket": open_trade.vol_bucket,
                            "trend_state": open_trade.trend_state,
                            "exit_reason": reason,
                            "hold_bars": int(max(open_trade.hold_bars, 1)),
                            "raw_return": float(raw_return),
                            "fee": fee_cost,
                            "slippage": slip_cost,
                            "net_return": float(net_return),
                            "mfe_approx": float(open_trade.mfe),
                            "mae_approx": float(open_trade.mae),
                        }
                    )
                    open_trade = None
                global_idx += 1

    return pd.DataFrame(records)


def _select_latest_session(all_trades: pd.DataFrame, state_trades: List[Dict[str, Any]]) -> pd.DataFrame:
    if all_trades.empty:
        return all_trades
    n = len(state_trades)
    if n <= 0 or len(all_trades) < n:
        return all_trades.copy()
    state_pnls = [float(t.get("pnl", 0.0)) for t in state_trades]
    arr = all_trades["raw_return"].to_numpy(dtype=float)
    target = np.array(state_pnls, dtype=float)
    best_start = len(all_trades) - n
    best_err = float("inf")
    for i in range(0, len(all_trades) - n + 1):
        err = float(np.mean(np.abs(arr[i : i + n] - target)))
        if err < best_err:
            best_err = err
            best_start = i
    return all_trades.iloc[best_start : best_start + n].reset_index(drop=True)


def _expectancy_stats(returns: Iterable[float]) -> Dict[str, float]:
    vals = [float(x) for x in returns]
    if not vals:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy_per_trade": 0.0,
            "median_return": 0.0,
            "p25_return": 0.0,
            "p75_return": 0.0,
        }
    wins = [x for x in vals if x > 0]
    losses = [x for x in vals if x < 0]
    win_rate = len(wins) / len(vals)
    avg_win = mean(wins) if wins else 0.0
    avg_loss = mean(losses) if losses else 0.0
    gross_profit = sum(wins)
    gross_loss_abs = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss_abs) if gross_loss_abs > 0 else 0.0
    expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * abs(avg_loss))
    return {
        "total_trades": len(vals),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy_per_trade": expectancy,
        "median_return": float(median(vals)),
        "p25_return": float(np.percentile(vals, 25)),
        "p75_return": float(np.percentile(vals, 75)),
    }


def _streaks(returns: List[float]) -> Tuple[int, float, List[List[float]]]:
    max_streak = 0
    current = 0
    streak_sizes: List[int] = []
    streak_returns: List[List[float]] = []
    current_returns: List[float] = []
    for r in returns:
        if r < 0:
            current += 1
            current_returns.append(r)
            max_streak = max(max_streak, current)
        else:
            if current > 0:
                streak_sizes.append(current)
                streak_returns.append(current_returns.copy())
            current = 0
            current_returns = []
    if current > 0:
        streak_sizes.append(current)
        streak_returns.append(current_returns.copy())
    avg_streak = float(mean(streak_sizes)) if streak_sizes else 0.0
    return max_streak, avg_streak, streak_returns


def _drawdown_episodes(equity_curve: List[float]) -> Tuple[int, int]:
    in_dd = False
    episodes = 0
    underwater_len = 0
    peak = equity_curve[0] if equity_curve else 1.0
    for eq in equity_curve:
        peak = max(peak, eq)
        underwater = eq < peak
        if underwater:
            underwater_len += 1
            if not in_dd:
                in_dd = True
                episodes += 1
        else:
            in_dd = False
    return underwater_len, episodes


def _format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"paper_trade_lifecycle_analysis_{timestamp}.csv"
    md_path = OUT_DIR / f"paper_trade_lifecycle_analysis_{timestamp}.md"

    state, state_trades = _load_state_trades()
    lifecycle_all = _build_lifecycle_trades()
    trades_df = _select_latest_session(lifecycle_all, state_trades)

    if trades_df.empty:
        raise RuntimeError("No paper lifecycle trades found from paper_trading_log_*.jsonl")

    trades_df["gross_return"] = trades_df["raw_return"]
    trades_df["is_win"] = trades_df["net_return"] > 0
    trades_df["equity_step"] = 1.0 + trades_df["net_return"] * 0.05
    trades_df["equity"] = trades_df["equity_step"].cumprod()
    trades_df["peak_equity"] = trades_df["equity"].cummax()
    trades_df["drawdown"] = (trades_df["equity"] - trades_df["peak_equity"]) / trades_df["peak_equity"]

    exp = _expectancy_stats(trades_df["net_return"].tolist())
    long_df = trades_df[trades_df["direction"] == "LONG"]
    short_df = trades_df[trades_df["direction"] == "SHORT"]

    def _group_perf(df: pd.DataFrame, key: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=[key, "trade_count", "avg_return", "win_rate", "avg_hold_bars"])
        out = (
            df.groupby(key, dropna=False)
            .agg(
                trade_count=("net_return", "count"),
                avg_return=("net_return", "mean"),
                win_rate=("is_win", "mean"),
                avg_hold_bars=("hold_bars", "mean"),
            )
            .reset_index()
            .sort_values("trade_count", ascending=False)
        )
        return out

    exit_perf = _group_perf(trades_df, "exit_reason")

    bins = [0, 3, 6, 9, 12, 10**9]
    labels = ["1~3", "4~6", "7~9", "10~12", ">12"]
    trades_df["hold_bucket"] = pd.cut(
        trades_df["hold_bars"], bins=bins, labels=labels, include_lowest=True, right=True
    )
    hold_perf = _group_perf(trades_df, "hold_bucket")

    gross_return_sum = float(trades_df["gross_return"].sum())
    net_return_sum = float(trades_df["net_return"].sum())
    total_fee_cost = float(trades_df["fee"].sum())
    total_slippage_cost = float(trades_df["slippage"].sum())
    edge_after_cost = net_return_sum - gross_return_sum

    returns = trades_df["net_return"].tolist()
    max_streak, avg_streak, streak_returns = _streaks(returns)
    ks_threshold = 5
    ks_idx = None
    consec = 0
    for i, r in enumerate(returns):
        if r < 0:
            consec += 1
            if consec >= ks_threshold:
                ks_idx = i
                break
        else:
            consec = 0
    ks_sequence = []
    if ks_idx is not None:
        ks_sequence = returns[max(0, ks_idx - 4) : ks_idx + 1]

    final_equity = float(trades_df["equity"].iloc[-1])
    max_drawdown = float(trades_df["drawdown"].min())
    underwater_duration, dd_episodes = _drawdown_episodes(trades_df["equity"].tolist())
    worst_trade = float(trades_df["net_return"].min())
    best_trade = float(trades_df["net_return"].max())

    win_ent = trades_df.loc[trades_df["net_return"] > 0, "entropy"].dropna().astype(float)
    loss_ent = trades_df.loc[trades_df["net_return"] < 0, "entropy"].dropna().astype(float)
    win_ent_mean = float(win_ent.mean()) if len(win_ent) else float("nan")
    loss_ent_mean = float(loss_ent.mean()) if len(loss_ent) else float("nan")

    strat_perf = _group_perf(trades_df, "strategy_used")
    strategy_expectancy: Dict[str, float] = {}
    for s in trades_df["strategy_used"].dropna().unique().tolist():
        s_df = trades_df[trades_df["strategy_used"] == s]
        strategy_expectancy[s] = _expectancy_stats(s_df["net_return"].tolist())["expectancy_per_trade"]

    long_share = len(long_df) / max(len(trades_df), 1)
    short_share = len(short_df) / max(len(trades_df), 1)
    long_expectancy = _expectancy_stats(long_df["net_return"].tolist())["expectancy_per_trade"] if len(long_df) else 0.0
    short_expectancy = _expectancy_stats(short_df["net_return"].tolist())["expectancy_per_trade"] if len(short_df) else 0.0

    if gross_return_sum > 0 and net_return_sum <= 0:
        verdict = "A"
    elif abs(exp["expectancy_per_trade"]) <= 0.0005:
        verdict = "B"
    elif gross_return_sum <= 0 and strategy_expectancy.get("S2", 0.0) <= 0 and long_expectancy <= 0 and short_expectancy <= 0:
        verdict = "C"
    else:
        # 조건이 깔끔히 맞지 않는 경우는 expectancy 기반으로 보수적 분류
        verdict = "B" if abs(exp["expectancy_per_trade"]) < 0.001 else "A"

    trades_df.to_csv(csv_path, index=False)

    md_lines: List[str] = []
    md_lines.append("# Paper Trade Lifecycle Analysis")
    md_lines.append("")
    md_lines.append(f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append(f"- source_state: `{STATE_PATH}`")
    md_lines.append(f"- source_logs: `{MONITORING_DIR}`")
    md_lines.append(f"- trades_used: {len(trades_df)} (state_trades={len(state_trades)}, lifecycle_all={len(lifecycle_all)})")
    md_lines.append("")
    md_lines.append("## [EXPECTANCY]")
    md_lines.append(f"- total_trades: {exp['total_trades']}")
    md_lines.append(f"- win_rate: {_format_pct(exp['win_rate'])}")
    md_lines.append(f"- avg_win: {exp['avg_win']:.6f}")
    md_lines.append(f"- avg_loss: {exp['avg_loss']:.6f}")
    md_lines.append(f"- profit_factor: {exp['profit_factor']:.4f}")
    md_lines.append(f"- expectancy_per_trade: {exp['expectancy_per_trade']:.6f}")
    md_lines.append(f"- median_return: {exp['median_return']:.6f}")
    md_lines.append(f"- p25/p75_return: {exp['p25_return']:.6f} / {exp['p75_return']:.6f}")
    md_lines.append("")
    md_lines.append("## [LONG SHORT]")
    md_lines.append(f"- long_count: {len(long_df)} | short_count: {len(short_df)}")
    md_lines.append(f"- long_share: {_format_pct(long_share)} | short_share: {_format_pct(short_share)}")
    md_lines.append(f"- long_win_rate: {_format_pct((long_df['net_return'] > 0).mean() if len(long_df) else 0.0)}")
    md_lines.append(f"- short_win_rate: {_format_pct((short_df['net_return'] > 0).mean() if len(short_df) else 0.0)}")
    md_lines.append(f"- long_avg_return: {long_df['net_return'].mean() if len(long_df) else 0.0:.6f}")
    md_lines.append(f"- short_avg_return: {short_df['net_return'].mean() if len(short_df) else 0.0:.6f}")
    md_lines.append(f"- long_expectancy: {long_expectancy:.6f}")
    md_lines.append(f"- short_expectancy: {short_expectancy:.6f}")
    md_lines.append("")
    md_lines.append("## [EXIT ANALYSIS]")
    if exit_perf.empty:
        md_lines.append("- no exit_reason data")
    else:
        for _, row in exit_perf.iterrows():
            md_lines.append(
                f"- {row['exit_reason']}: trades={int(row['trade_count'])}, avg_return={row['avg_return']:.6f}, "
                f"win_rate={_format_pct(float(row['win_rate']))}, avg_hold_bars={row['avg_hold_bars']:.2f}"
            )
    md_lines.append("")
    md_lines.append("## [FEE IMPACT]")
    md_lines.append(f"- gross_return_sum: {gross_return_sum:.6f}")
    md_lines.append(f"- net_return_sum: {net_return_sum:.6f}")
    md_lines.append(f"- total_fee_cost: {total_fee_cost:.6f}")
    md_lines.append(f"- total_slippage_cost: {total_slippage_cost:.6f}")
    md_lines.append(f"- edge_after_cost (net-gross): {edge_after_cost:.6f}")
    md_lines.append("")
    md_lines.append("## [HOLD ANALYSIS]")
    if hold_perf.empty:
        md_lines.append("- no hold_bars data")
    else:
        for _, row in hold_perf.iterrows():
            md_lines.append(
                f"- {row['hold_bucket']}: trades={int(row['trade_count'])}, avg_return={row['avg_return']:.6f}, "
                f"win_rate={_format_pct(float(row['win_rate']))}"
            )
    md_lines.append("")
    md_lines.append("## [KILL SWITCH]")
    md_lines.append(f"- max_consecutive_losses: {max_streak}")
    md_lines.append(f"- average_losing_streak: {avg_streak:.2f}")
    md_lines.append(f"- state_kill_switch_active: {state.get('kill_switch_active')}")
    md_lines.append(f"- state_kill_switch_reason: {state.get('kill_switch_reason')}")
    md_lines.append(f"- ks_trigger_trade_index(estimated): {ks_idx}")
    md_lines.append(f"- ks_recent_sequence_net_returns: {[round(x, 6) for x in ks_sequence]}")
    md_lines.append("")
    md_lines.append("## [EQUITY]")
    md_lines.append(f"- final_equity: {final_equity:.6f}")
    md_lines.append(f"- max_drawdown: {max_drawdown:.6f}")
    md_lines.append(f"- underwater_duration: {underwater_duration}")
    md_lines.append(f"- drawdown_episodes: {dd_episodes}")
    md_lines.append(f"- worst_trade: {worst_trade:.6f}")
    md_lines.append(f"- best_trade: {best_trade:.6f}")
    md_lines.append("")
    md_lines.append("## [ENTROPY]")
    md_lines.append(f"- winning_entropy_mean: {win_ent_mean:.6f}")
    md_lines.append(f"- losing_entropy_mean: {loss_ent_mean:.6f}")
    md_lines.append("")
    md_lines.append("## [STRATEGY]")
    if strat_perf.empty:
        md_lines.append("- no strategy_used data")
    else:
        for _, row in strat_perf.iterrows():
            s_name = str(row["strategy_used"])
            md_lines.append(
                f"- {s_name}: trades={int(row['trade_count'])}, avg_return={row['avg_return']:.6f}, "
                f"win_rate={_format_pct(float(row['win_rate']))}, expectancy={strategy_expectancy.get(s_name, 0.0):.6f}"
            )
    md_lines.append("")
    md_lines.append("## [FINAL VERDICT]")
    md_lines.append(f"- {verdict}")
    md_lines.append("")
    md_lines.append("## [NEXT ACTION]")
    md_lines.append("- Exit ablation 필요 (exit_reason별 성과 차이 확인 후 분리 실험)")
    md_lines.append("- fee/slippage 민감도 점검 필요 (gross/net 괴리 추적)")
    md_lines.append("- hold shortening/extension 실험 필요 (1~3 / 10~12 성과 대비)")
    md_lines.append("- LONG/SHORT 비중 불균형 시 SHORT 강화 실험 필요")

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"CSV: {csv_path}")
    print(f"MD:  {md_path}")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
