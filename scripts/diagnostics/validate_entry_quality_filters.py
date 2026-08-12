"""
Can_bit Entry Quality Filter Validation (diagnostics only).

Usage:
  python -m scripts.diagnostics.validate_entry_quality_filters --lookback-list 14,30,60,90
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import timedelta
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.run_daily_paper import PAPER_MAX_HOLDING_BARS, load_ohlcv, simulate_signals_paper


OUT_DIR = Path("data/diagnostics/entry_quality")
MAX_TRADE_LOSS = -0.01
MAX_DAILY_LOSS = -0.02
MAX_DRAWDOWN = -0.05
MAX_CONSEC_LOSSES = 5


@dataclass
class FilterSpec:
    name: str
    group: str
    params: Dict[str, Any]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("validate_entry_quality_filters")
    p.add_argument("--lookback-list", default="14,30,60,90")
    p.add_argument("--position-size", type=float, default=0.05)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0002)
    p.add_argument("--stride-days", type=int, default=7)
    p.add_argument("--max-windows-per-lookback", type=int, default=24)
    return p.parse_args()


def _entropy(t: Dict[str, Any]) -> float:
    e = 0.0
    for p in (float(t["p_long"]), float(t["p_short"]), float(t["p_flat"])):
        if p > 1e-10:
            e -= p * math.log(p)
    return e


def _margin(t: Dict[str, Any]) -> float:
    return abs(float(t["p_long"]) - float(t["p_short"]))


def _slice_ticks_by_range(ticks: List[Dict[str, Any]], start: pd.Timestamp, end: pd.Timestamp) -> List[Dict[str, Any]]:
    out = []
    for t in ticks:
        ts = t.get("timestamp")
        if not ts:
            continue
        tts = pd.Timestamp(ts)
        if start <= tts <= end:
            out.append(t)
    return out


def _window_endpoints(
    ticks: List[Dict[str, Any]],
    lookback_days: int,
    stride_days: int,
    max_windows: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    ts = sorted([pd.Timestamp(t["timestamp"]) for t in ticks if t.get("timestamp")])
    if not ts:
        return []
    tmin, tmax = ts[0], ts[-1]
    windows = []
    cur_end = tmin + timedelta(days=lookback_days)
    while cur_end <= tmax:
        cur_start = cur_end - timedelta(days=lookback_days)
        windows.append((cur_start, cur_end))
        cur_end += timedelta(days=stride_days)
    if windows and windows[-1][1] < tmax:
        windows.append((tmax - timedelta(days=lookback_days), tmax))
    if len(windows) > max_windows:
        windows = windows[-max_windows:]
    return windows


def _build_specs() -> List[FilterSpec]:
    specs: List[FilterSpec] = [FilterSpec("baseline_current", "baseline", {})]

    for eub in [1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.85]:
        specs.append(FilterSpec(f"entropy_le_{eub:.2f}", "entropy_sweep", {"entropy_ub": eub}))
    for mf in [0.00, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
        specs.append(FilterSpec(f"margin_ge_{mf:.2f}", "margin_sweep", {"margin_floor": mf}))

    # combined
    specs.extend(
        [
            FilterSpec("combo_A", "combined", {"entropy_ub": 0.95, "margin_floor": 0.04}),
            FilterSpec("combo_B", "combined", {"entropy_ub": 0.92, "margin_floor": 0.05}),
            FilterSpec("combo_C", "combined", {"entropy_ub": 0.90, "margin_floor": 0.06}),
            FilterSpec("combo_D", "combined", {"entropy_ub": 0.92, "exclude_high_vol": True}),
            FilterSpec("combo_E", "combined", {"entropy_ub": 0.90, "long_only": True, "margin_floor": 0.05}),
            FilterSpec("combo_F", "combined", {"entropy_ub": 0.95, "long_only": True, "exclude_long_down": True}),
            FilterSpec("combo_G", "combined", {"entropy_ub": 0.90, "long_margin_top_pct": 0.30}),
            FilterSpec("combo_H", "combined", {"entropy_bottom_pct": 0.30, "margin_top_pct": 0.30}),
        ]
    )

    # adaptive
    specs.extend(
        [
            FilterSpec("adaptive_A_entropy_bottom20", "adaptive", {"entropy_bottom_pct": 0.20}),
            FilterSpec("adaptive_B_entropy_bottom30", "adaptive", {"entropy_bottom_pct": 0.30}),
            FilterSpec("adaptive_C_margin_top20", "adaptive", {"margin_top_pct": 0.20}),
            FilterSpec("adaptive_D_margin_top30", "adaptive", {"margin_top_pct": 0.30}),
            FilterSpec("adaptive_E_entropy_bottom30_margin_top30", "adaptive", {"entropy_bottom_pct": 0.30, "margin_top_pct": 0.30}),
        ]
    )
    return specs


def _prepare_dynamic_thresholds(ticks: List[Dict[str, Any]]) -> Dict[str, float]:
    ent = np.array([_entropy(t) for t in ticks], dtype=float)
    mar = np.array([_margin(t) for t in ticks], dtype=float)
    return {
        "entropy_p20": float(np.quantile(ent, 0.20)) if len(ent) else 0.0,
        "entropy_p30": float(np.quantile(ent, 0.30)) if len(ent) else 0.0,
        "margin_p70": float(np.quantile(mar, 0.70)) if len(mar) else 0.0,
        "margin_p80": float(np.quantile(mar, 0.80)) if len(mar) else 0.0,
    }


def _allow_entry(spec: FilterSpec, t: Dict[str, Any], signal: str, dyn: Dict[str, float]) -> bool:
    ent = _entropy(t)
    mar = _margin(t)
    vol = str(t.get("vol_bucket") or "")
    trend = str(t.get("trend_label") or "")

    p = spec.params
    if "entropy_ub" in p and ent > float(p["entropy_ub"]):
        return False
    if "margin_floor" in p and mar < float(p["margin_floor"]):
        return False
    if p.get("exclude_high_vol") and vol == "high":
        return False
    if p.get("long_only") and signal != "LONG":
        return False
    if p.get("exclude_long_down") and signal == "LONG" and trend == "down":
        return False

    if "long_margin_top_pct" in p and signal == "LONG":
        # top30% => >= p70
        if mar < dyn["margin_p70"]:
            return False

    if "entropy_bottom_pct" in p:
        if p["entropy_bottom_pct"] == 0.20 and ent > dyn["entropy_p20"]:
            return False
        if p["entropy_bottom_pct"] == 0.30 and ent > dyn["entropy_p30"]:
            return False

    if "margin_top_pct" in p:
        if p["margin_top_pct"] == 0.20 and mar < dyn["margin_p80"]:
            return False
        if p["margin_top_pct"] == 0.30 and mar < dyn["margin_p70"]:
            return False

    return True


def _simulate(
    ticks: List[Dict[str, Any]],
    spec: FilterSpec,
    position_size: float,
    fee_rate: float,
    slippage_rate: float,
) -> Tuple[Dict[str, Any], pd.DataFrame, List[int], List[int]]:
    dyn = _prepare_dynamic_thresholds(ticks)
    trades: List[Dict[str, Any]] = []
    open_pos: Optional[Dict[str, Any]] = None

    eq = 1.0
    peak = 1.0
    daily_pnl = 0.0
    consec_losses = 0
    ks_triggered = False
    ks_count = 0
    risk_force = 0
    exit_sig = 0
    blocked_entries: List[int] = []
    entered_indices: List[int] = []

    for i, t in enumerate(ticks):
        px = float(t["price"])
        signal = t.get("signal")
        vol = str(t.get("vol_bucket") or "")
        trend = str(t.get("trend_label") or "")

        # explicit exit signals
        exit_signal = False
        reason = ""
        if ks_triggered and open_pos is not None:
            exit_signal = True
            reason = "kill_switch_close"
        if open_pos is not None:
            open_pos["hold_bars"] += 1
            if signal is not None and ((open_pos["side"] == "BUY" and signal == "SHORT") or (open_pos["side"] == "SELL" and signal == "LONG")):
                exit_signal = True
                reason = reason or "opposite_signal"
            elif open_pos["hold_bars"] >= PAPER_MAX_HOLDING_BARS:
                exit_signal = True
                reason = reason or "max_holding_bars"
        if exit_signal and open_pos is not None:
            raw = (px - open_pos["entry_price"]) / open_pos["entry_price"] if open_pos["side"] == "BUY" else (open_pos["entry_price"] - px) / open_pos["entry_price"]
            net = raw - 2.0 * (fee_rate + slippage_rate)
            eq *= 1.0 + net * position_size
            peak = max(peak, eq)
            daily_pnl = eq - 1.0
            consec_losses = consec_losses + 1 if net < 0 else 0
            trades.append(
                {
                    "entry_idx": open_pos["entry_idx"],
                    "direction": open_pos["direction"],
                    "hold_bars": open_pos["hold_bars"],
                    "exit_reason": reason or "exit_signal",
                    "gross_return": raw,
                    "net_return": net,
                }
            )
            exit_sig += 1
            open_pos = None
            continue

        # risk_force_exit while holding
        if open_pos is not None:
            unreal = (px - open_pos["entry_price"]) / open_pos["entry_price"] if open_pos["side"] == "BUY" else (open_pos["entry_price"] - px) / open_pos["entry_price"]
            if unreal <= MAX_TRADE_LOSS:
                raw = unreal
                net = raw - 2.0 * (fee_rate + slippage_rate)
                eq *= 1.0 + net * position_size
                peak = max(peak, eq)
                daily_pnl = eq - 1.0
                consec_losses = consec_losses + 1 if net < 0 else 0
                trades.append(
                    {
                        "entry_idx": open_pos["entry_idx"],
                        "direction": open_pos["direction"],
                        "hold_bars": open_pos["hold_bars"],
                        "exit_reason": "risk_force_exit",
                        "gross_return": raw,
                        "net_return": net,
                    }
                )
                risk_force += 1
                open_pos = None
                continue
            else:
                continue

        # no-position pipeline
        if vol not in ("mid", "high"):
            continue
        strategy = "S2" if trend != "sideways" else "S1"
        ent = _entropy(t)
        if strategy == "S2" and ent > 1.0:
            continue
        if signal is None:
            continue

        # additional entry-quality filter (diagnostic only)
        if not _allow_entry(spec, t, signal, dyn):
            blocked_entries.append(i)
            continue

        # pre-entry risk check
        drawdown = (eq - peak) / peak if peak > 0 else 0.0
        stop = ks_triggered or daily_pnl <= MAX_DAILY_LOSS or drawdown <= MAX_DRAWDOWN or consec_losses >= MAX_CONSEC_LOSSES
        if stop:
            if not ks_triggered:
                ks_count += 1
            ks_triggered = True
            continue

        side = "BUY" if signal == "LONG" else "SELL"
        open_pos = {
            "entry_idx": i,
            "entry_price": px,
            "direction": "LONG" if side == "BUY" else "SHORT",
            "side": side,
            "hold_bars": 0,
        }
        entered_indices.append(i)

    tdf = pd.DataFrame(trades)
    vals = tdf["net_return"].tolist() if not tdf.empty else []
    gross = tdf["gross_return"].tolist() if not tdf.empty else []
    wins = [x for x in vals if x > 0]
    losses = [x for x in vals if x < 0]
    wr = len(wins) / len(vals) if vals else 0.0
    avg_w = mean(wins) if wins else 0.0
    avg_l = mean(losses) if losses else 0.0
    expectancy = (wr * avg_w) - ((1.0 - wr) * abs(avg_l)) if vals else 0.0
    pf = (sum(wins) / abs(sum(losses))) if losses else 0.0

    eq2 = 1.0
    peak2 = 1.0
    mdd = 0.0
    max_consec = 0
    c = 0
    for r in vals:
        eq2 *= 1.0 + r * position_size
        peak2 = max(peak2, eq2)
        mdd = min(mdd, (eq2 - peak2) / peak2 if peak2 > 0 else 0.0)
        if r < 0:
            c += 1
            max_consec = max(max_consec, c)
        else:
            c = 0

    metrics = {
        "total_trades": len(tdf),
        "win_rate": wr,
        "avg_return": float(np.mean(vals)) if vals else 0.0,
        "median_return": float(np.median(vals)) if vals else 0.0,
        "expectancy": expectancy,
        "profit_factor": pf,
        "gross_return": float(np.sum(gross)) if gross else 0.0,
        "net_return": float(np.sum(vals)) if vals else 0.0,
        "equity_return": eq2 - 1.0,
        "max_drawdown": mdd,
        "max_consecutive_losses": max_consec,
        "kill_switch_triggered": ks_triggered,
        "kill_switch_count": ks_count,
        "risk_force_exit_count": risk_force,
        "avg_hold_bars": float(tdf["hold_bars"].mean()) if not tdf.empty else 0.0,
        "avg_hold_winners": float(tdf[tdf["net_return"] > 0]["hold_bars"].mean()) if not tdf.empty else 0.0,
        "avg_hold_losers": float(tdf[tdf["net_return"] < 0]["hold_bars"].mean()) if not tdf.empty else 0.0,
    }
    return metrics, tdf, blocked_entries, entered_indices


def _trade_quality_from_baseline(base_trades: pd.DataFrame, blocked_entries: List[int], filtered_trades: pd.DataFrame) -> Dict[str, Any]:
    removed = base_trades[base_trades["entry_idx"].isin(blocked_entries)] if not base_trades.empty else pd.DataFrame()
    bad_removed = int((removed["net_return"] < 0).sum()) if not removed.empty else 0
    total_removed = int(len(removed))
    good_removed = int((removed["net_return"] > 0).sum()) if not removed.empty else 0
    precision = (bad_removed / total_removed) if total_removed > 0 else 0.0

    vals = filtered_trades["net_return"].tolist() if not filtered_trades.empty else []
    wins = [x for x in vals if x > 0]
    losses = [x for x in vals if x < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else 0.0
    exp = (len(wins) / len(vals) * mean(wins) - (1 - len(wins) / len(vals)) * abs(mean(losses))) if losses and wins else (mean(wins) if wins else 0.0)
    return {
        "trades_removed": total_removed,
        "trade_reduction_ratio": (total_removed / max(len(base_trades), 1)),
        "bad_trades_removed": bad_removed,
        "good_trades_removed": good_removed,
        "removal_precision": precision,
        "survivor_expectancy": exp,
        "survivor_PF": pf,
    }


def _consistency(sub: pd.DataFrame) -> Dict[str, Any]:
    improved = int((sub["net_return_delta"] > 0).sum())
    degraded = int((sub["net_return_delta"] < 0).sum())
    arr = sub["net_return_delta"].tolist()
    avg_d = float(mean(arr)) if arr else 0.0
    std_d = float(pstdev(arr)) if len(arr) > 1 else 0.0
    score = avg_d / (std_d + 1e-9)
    return {
        "improved_periods": improved,
        "degraded_periods": degraded,
        "avg_delta": avg_d,
        "std_delta": std_d,
        "consistency_score": score,
    }


def _grade(row: pd.Series) -> str:
    if row["improved_periods"] >= 4 and row["expectancy_delta_avg"] > 0 and row["PF_delta_avg"] > 0 and row["KS_delta_avg"] <= 0 and row["MDD_delta_avg"] >= 0:
        return "A"
    if row["improved_periods"] >= 2 and row["net_return_delta_avg"] > 0:
        return "B"
    return "C"


def main() -> None:
    args = _parse_args()
    lookbacks = [int(x.strip()) for x in args.lookback_list.split(",") if x.strip()]
    specs = _build_specs()

    df = load_ohlcv()
    ticks, _ = simulate_signals_paper(df)

    rows: List[Dict[str, Any]] = []
    quality_rows: List[Dict[str, Any]] = []
    for lb in lookbacks:
        wins = _window_endpoints(ticks, lb, args.stride_days, args.max_windows_per_lookback)
        for wid, (ws, we) in enumerate(wins):
            w_ticks = _slice_ticks_by_range(ticks, ws, we)
            if len(w_ticks) < 30:
                continue
            base_metrics = None
            base_trades = None
            base_blocked: List[int] = []
            for spec in specs:
                m, tdf, blocked, _ = _simulate(
                    w_ticks,
                    spec,
                    position_size=args.position_size,
                    fee_rate=args.fee_rate,
                    slippage_rate=args.slippage_rate,
                )
                rec = {
                    "lookback_days": lb,
                    "window_id": wid,
                    "window_start": ws,
                    "window_end": we,
                    "filter_name": spec.name,
                    "filter_group": spec.group,
                    **m,
                }
                rows.append(rec)
                if spec.name == "baseline_current":
                    base_metrics = m
                    base_trades = tdf
                    base_blocked = blocked
                else:
                    if base_trades is not None:
                        q = _trade_quality_from_baseline(base_trades, blocked, tdf)
                        quality_rows.append(
                            {
                                "lookback_days": lb,
                                "window_id": wid,
                                "filter_name": spec.name,
                                **q,
                            }
                        )

            # deltas
            if base_metrics is not None:
                for r in rows:
                    if r["lookback_days"] == lb and r["window_id"] == wid:
                        r["expectancy_delta"] = r["expectancy"] - base_metrics["expectancy"]
                        r["PF_delta"] = r["profit_factor"] - base_metrics["profit_factor"]
                        r["net_return_delta"] = r["net_return"] - base_metrics["net_return"]
                        r["equity_return_delta"] = r["equity_return"] - base_metrics["equity_return"]
                        r["MDD_delta"] = r["max_drawdown"] - base_metrics["max_drawdown"]
                        r["KS_delta"] = int(r["kill_switch_triggered"]) - int(base_metrics["kill_switch_triggered"])
                        r["risk_force_exit_delta"] = r["risk_force_exit_count"] - base_metrics["risk_force_exit_count"]

    df_all = pd.DataFrame(rows)
    qdf = pd.DataFrame(quality_rows)

    # summary per filter
    sum_rows = []
    for fname in sorted(df_all["filter_name"].unique().tolist()):
        if fname == "baseline_current":
            continue
        sub = df_all[df_all["filter_name"] == fname]
        cons = _consistency(sub)
        qsub = qdf[qdf["filter_name"] == fname]
        sum_rows.append(
            {
                "filter_name": fname,
                "filter_group": sub["filter_group"].iloc[0],
                "expectancy_delta_avg": float(sub["expectancy_delta"].mean()),
                "PF_delta_avg": float(sub["PF_delta"].mean()),
                "net_return_delta_avg": float(sub["net_return_delta"].mean()),
                "equity_return_delta_avg": float(sub["equity_return_delta"].mean()),
                "MDD_delta_avg": float(sub["MDD_delta"].mean()),
                "KS_delta_avg": float(sub["KS_delta"].mean()),
                "risk_force_exit_delta_avg": float(sub["risk_force_exit_delta"].mean()),
                **cons,
                "trade_reduction_ratio_avg": float(qsub["trade_reduction_ratio"].mean()) if len(qsub) else 0.0,
                "removal_precision_avg": float(qsub["removal_precision"].mean()) if len(qsub) else 0.0,
                "survivor_expectancy_avg": float(qsub["survivor_expectancy"].mean()) if len(qsub) else 0.0,
                "survivor_PF_avg": float(qsub["survivor_PF"].mean()) if len(qsub) else 0.0,
            }
        )
    summary = pd.DataFrame(sum_rows)
    summary["grade"] = summary.apply(_grade, axis=1)
    summary = summary.sort_values(
        ["grade", "consistency_score", "net_return_delta_avg", "expectancy_delta_avg"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    # winners by group
    def _best(group: str) -> pd.Series:
        s = summary[summary["filter_group"] == group]
        return s.iloc[0] if len(s) else pd.Series(dtype=float)

    best_entropy = _best("entropy_sweep")
    best_margin = _best("margin_sweep")
    best_combined = _best("combined")
    best_adaptive = _best("adaptive")
    overall = summary.iloc[0] if len(summary) else pd.Series(dtype=float)
    consistency_winner = summary.sort_values("consistency_score", ascending=False).iloc[0] if len(summary) else pd.Series(dtype=float)

    recommendation = "reject"
    if len(summary):
        if overall["grade"] == "A":
            recommendation = "apply_candidate"
        elif overall["grade"] == "B":
            recommendation = "forward_monitor"
        else:
            recommendation = "reject"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"entry_quality_filter_validation_{ts}.csv"
    md_path = OUT_DIR / f"entry_quality_filter_validation_{ts}.md"

    # write combined csv with sections
    tbls = []
    tbls.append(df_all.assign(section="rolling_detail"))
    tbls.append(summary.assign(section="filter_summary"))
    if len(qdf):
        tbls.append(qdf.assign(section="trade_quality"))
    pd.concat(tbls, ignore_index=True, sort=False).to_csv(csv_path, index=False)

    md: List[str] = []
    md.append("# Can_bit Entry Quality Filter Validation Report")
    md.append("")
    md.append("## 1. 목적")
    md.append("- Entry quality filter가 rolling OOS에서 일관되게 개선되는지 검증")
    md.append("")
    md.append("## 2. baseline 결과")
    base = df_all[df_all["filter_name"] == "baseline_current"]
    md.append(
        f"- samples={len(base)}, avg_expectancy={base['expectancy'].mean():.6f}, avg_net={base['net_return'].mean():.6f}, "
        f"avg_PF={base['profit_factor'].mean():.4f}, avg_MDD={base['max_drawdown'].mean():.6f}"
    )
    md.append("")
    md.append("## 3. entropy sweep")
    for _, r in summary[summary["filter_group"] == "entropy_sweep"].head(5).iterrows():
        md.append(
            f"- {r['filter_name']}: grade={r['grade']}, netΔ={r['net_return_delta_avg']:+.6f}, expΔ={r['expectancy_delta_avg']:+.6f}, "
            f"consistency={r['consistency_score']:.4f}"
        )
    md.append("")
    md.append("## 4. margin sweep")
    for _, r in summary[summary["filter_group"] == "margin_sweep"].head(5).iterrows():
        md.append(
            f"- {r['filter_name']}: grade={r['grade']}, netΔ={r['net_return_delta_avg']:+.6f}, expΔ={r['expectancy_delta_avg']:+.6f}, "
            f"precision={r['removal_precision_avg']:.2%}"
        )
    md.append("")
    md.append("## 5. combined filters")
    for _, r in summary[summary["filter_group"] == "combined"].head(8).iterrows():
        md.append(
            f"- {r['filter_name']}: grade={r['grade']}, netΔ={r['net_return_delta_avg']:+.6f}, expΔ={r['expectancy_delta_avg']:+.6f}, "
            f"MDDΔ={r['MDD_delta_avg']:+.6f}"
        )
    md.append("")
    md.append("## 6. adaptive filters")
    for _, r in summary[summary["filter_group"] == "adaptive"].head(5).iterrows():
        md.append(
            f"- {r['filter_name']}: grade={r['grade']}, netΔ={r['net_return_delta_avg']:+.6f}, expΔ={r['expectancy_delta_avg']:+.6f}, "
            f"consistency={r['consistency_score']:.4f}"
        )
    md.append("")
    md.append("## 7. rolling OOS summary")
    md.append(f"- lookbacks={sorted(df_all['lookback_days'].unique().tolist())}, windows={df_all[['lookback_days','window_id']].drop_duplicates().shape[0]}")
    md.append("")
    md.append("## 8. trade quality analysis")
    for _, r in summary.head(5).iterrows():
        md.append(
            f"- {r['filter_name']}: reduction={r['trade_reduction_ratio_avg']:.2%}, precision={r['removal_precision_avg']:.2%}, "
            f"survivor_exp={r['survivor_expectancy_avg']:.6f}, survivor_pf={r['survivor_PF_avg']:.4f}"
        )
    md.append("")
    md.append("## 9. consistency analysis")
    for _, r in summary.head(5).iterrows():
        md.append(
            f"- {r['filter_name']}: improved={int(r['improved_periods'])}, degraded={int(r['degraded_periods'])}, "
            f"avgΔ={r['avg_delta']:+.6f}, stdΔ={r['std_delta']:.6f}, score={r['consistency_score']:.4f}"
        )
    md.append("")
    md.append("## 10. best candidate")
    if len(summary):
        md.append(
            f"- {overall['filter_name']} (group={overall['filter_group']}, grade={overall['grade']}), "
            f"netΔ={overall['net_return_delta_avg']:+.6f}, expΔ={overall['expectancy_delta_avg']:+.6f}"
        )
    md.append("")
    md.append("## 11. 운영 적용 가능 여부")
    md.append(f"- recommendation: {recommendation}")
    md.append("")
    md.append("## 12. 위험 요소")
    md.append("- trade_reduction_ratio 과도 상승 시 실전 적용 위험")
    md.append("- 단기 윈도우 과적합 가능성")
    md.append("")
    md.append("## 13. next step")
    md.append("- 상위 2개 필터만 별도 walk-forward / blocked OOS로 재검증")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("[ENTRY QUALITY FILTER VALIDATION SUMMARY]")
    print("")
    if len(best_entropy):
        print(f"best_entropy_filter:\n- {best_entropy['filter_name']} (grade={best_entropy['grade']}, netΔ={best_entropy['net_return_delta_avg']:+.6f})")
    else:
        print("best_entropy_filter:\n- n/a")
    if len(best_margin):
        print(f"\nbest_margin_filter:\n- {best_margin['filter_name']} (grade={best_margin['grade']}, netΔ={best_margin['net_return_delta_avg']:+.6f})")
    else:
        print("\nbest_margin_filter:\n- n/a")
    if len(best_combined):
        print(f"\nbest_combined_filter:\n- {best_combined['filter_name']} (grade={best_combined['grade']}, netΔ={best_combined['net_return_delta_avg']:+.6f})")
    else:
        print("\nbest_combined_filter:\n- n/a")
    if len(best_adaptive):
        print(f"\nbest_adaptive_filter:\n- {best_adaptive['filter_name']} (grade={best_adaptive['grade']}, netΔ={best_adaptive['net_return_delta_avg']:+.6f})")
    else:
        print("\nbest_adaptive_filter:\n- n/a")
    if len(summary):
        print(f"\noverall_best_candidate:\n- {overall['filter_name']} (grade={overall['grade']}, netΔ={overall['net_return_delta_avg']:+.6f})")
        print(f"\nconsistency_winner:\n- {consistency_winner['filter_name']} (score={consistency_winner['consistency_score']:.4f})")
    else:
        print("\noverall_best_candidate:\n- n/a\n\nconsistency_winner:\n- n/a")
    print(f"\nfinal_recommendation:\n- {recommendation}")
    print(f"\ncreated:\n- csv: {csv_path}\n- markdown: {md_path}")


if __name__ == "__main__":
    main()
