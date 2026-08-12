"""
Conditional risk_force_exit replay validation (diagnostics only).

Usage:
  python -m scripts.diagnostics.validate_conditional_risk_force_exit_replay --lookback-list 14,30,60,90
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from scripts.run_daily_paper import PAPER_MAX_HOLDING_BARS, load_ohlcv, simulate_signals_paper


OUT_DIR = Path("data/diagnostics/lifecycle")
MAX_TRADE_LOSS = -0.01
MAX_DAILY_LOSS = -0.02
MAX_DRAWDOWN = -0.05
MAX_CONSEC_LOSSES = 5


@dataclass
class Variant:
    name: str


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("validate_conditional_risk_force_exit_replay")
    p.add_argument("--lookback-list", default="14,30,60,90")
    p.add_argument("--position-size", type=float, default=0.05)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0002)
    return p.parse_args()


def _variants() -> List[Variant]:
    return [
        Variant("current"),
        Variant("global_grace_1"),
        Variant("confirm_2_consecutive"),
        Variant("long_only_grace_1"),
        Variant("long_entropy_095_100_grace_1"),
        Variant("long_entropy_095_100_confirm_2"),
        Variant("long_hold_lt6_confirm_2"),
        Variant("long_hold_4_6_grace_1"),
        Variant("long_down_regime_strict_exit"),
        Variant("long_down_regime_block_force_grace"),
        Variant("entropy_safe_zone_no_force_exit"),
        Variant("entropy_danger_zone_strict"),
        Variant("composite_candidate_1"),
        Variant("composite_candidate_2_conservative"),
    ]


def _slice_ticks(ticks: List[Dict[str, Any]], days: int) -> List[Dict[str, Any]]:
    ts_vals = [pd.Timestamp(t["timestamp"]) for t in ticks if t.get("timestamp")]
    if not ts_vals:
        return ticks
    cutoff = max(ts_vals) - pd.Timedelta(days=days)
    return [t for t in ticks if t.get("timestamp") and pd.Timestamp(t["timestamp"]) >= cutoff]


def _unrealized(side: str, entry_price: float, px: float) -> float:
    if side == "BUY":
        return (px - entry_price) / entry_price
    return (entry_price - px) / entry_price


def _entropy_of_tick(t: Dict[str, Any]) -> float:
    ent = 0.0
    for p in (float(t["p_long"]), float(t["p_short"]), float(t["p_flat"])):
        if p > 1e-10:
            ent -= p * math.log(p)
    return ent


def _opposite_signal(pos_side: str, tick_signal: Optional[str]) -> bool:
    if tick_signal is None:
        return False
    return (pos_side == "BUY" and tick_signal == "SHORT") or (pos_side == "SELL" and tick_signal == "LONG")


def _loss_clusters(net_returns: List[float]) -> Tuple[int, float, float]:
    clusters: List[List[float]] = []
    cur: List[float] = []
    for r in net_returns:
        if r < 0:
            cur.append(r)
        else:
            if cur:
                clusters.append(cur[:])
            cur = []
    if cur:
        clusters.append(cur[:])
    if not clusters:
        return 0, 0.0, 0.0
    worst = min(clusters, key=sum)
    top5 = sorted([sum(c) for c in clusters])[:5]
    return len(worst), float(sum(worst)), float(sum(top5))


def _ks_prev5_reproduced(net_returns: List[float]) -> bool:
    consec = 0
    idx = -1
    for i, r in enumerate(net_returns):
        if r < 0:
            consec += 1
            if consec >= 5:
                idx = i
                break
        else:
            consec = 0
    return idx >= 4


def _policy_action(
    variant: str,
    direction: str,
    entropy: float,
    hold_bars: int,
    trend_state: str,
    recent_return_24: float,
    risk_consec: int,
    grace_left: int,
) -> str:
    """
    Return one of: immediate_exit / grace / confirm / ignore
    """
    in_danger_entropy = 0.95 <= entropy <= 1.00
    in_safe_entropy = entropy < 0.90
    is_down = (trend_state == "down") or (recent_return_24 < 0)

    if variant == "current":
        return "immediate_exit"
    if variant == "global_grace_1":
        return "grace"
    if variant == "confirm_2_consecutive":
        return "confirm2"

    if variant == "long_only_grace_1":
        return "grace" if direction == "LONG" else "immediate_exit"

    if variant == "long_entropy_095_100_grace_1":
        if direction == "LONG" and in_danger_entropy:
            return "grace"
        return "immediate_exit"

    if variant == "long_entropy_095_100_confirm_2":
        if direction == "LONG" and in_danger_entropy:
            return "confirm2"
        return "immediate_exit"

    if variant == "long_hold_lt6_confirm_2":
        if direction == "LONG" and hold_bars < 6:
            return "confirm2"
        return "immediate_exit"

    if variant == "long_hold_4_6_grace_1":
        if direction == "LONG" and 4 <= hold_bars <= 6:
            return "grace"
        return "immediate_exit"

    if variant == "long_down_regime_strict_exit":
        if direction == "LONG" and is_down:
            return "immediate_exit"
        return "grace"

    if variant == "long_down_regime_block_force_grace":
        if direction == "LONG" and is_down:
            return "immediate_exit"
        if direction == "LONG":
            return "confirm2"
        return "immediate_exit"

    if variant == "entropy_safe_zone_no_force_exit":
        if in_safe_entropy:
            return "ignore"
        return "immediate_exit"

    if variant == "entropy_danger_zone_strict":
        if in_danger_entropy:
            return "immediate_exit"
        return "grace"

    if variant == "composite_candidate_1":
        if direction == "LONG" and is_down:
            return "immediate_exit"
        if direction == "LONG" and in_danger_entropy and hold_bars < 6:
            return "confirm2"
        if in_safe_entropy:
            return "ignore"
        return "grace"

    if variant == "composite_candidate_2_conservative":
        if direction == "SHORT":
            return "immediate_exit"
        if direction == "LONG" and is_down:
            return "immediate_exit"
        if direction == "LONG" and hold_bars < 3:
            return "immediate_exit"
        if direction == "LONG" and 3 <= hold_bars <= 6:
            return "confirm2"
        return "grace"

    return "immediate_exit"


def _simulate_variant(
    ticks: List[Dict[str, Any]],
    variant_name: str,
    position_size: float,
    fee_rate: float,
    slippage_rate: float,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    trades: List[Dict[str, Any]] = []
    open_pos: Optional[Dict[str, Any]] = None

    equity = 1.0
    peak = 1.0
    daily_pnl = 0.0
    consec_losses = 0
    ks_triggered = False
    ks_trade_idx = -1

    risk_exit_count = 0
    exit_signal_count = 0
    long_count = 0
    short_count = 0

    recent_net_returns: List[float] = []

    for i, t in enumerate(ticks):
        price = float(t["price"])
        signal = t.get("signal")
        vol_bucket = str(t.get("vol_bucket") or "")
        trend_state = str(t.get("trend_label") or "")
        entropy = _entropy_of_tick(t)

        # explicit lifecycle exits
        pre_exit = ""
        exit_signal = False
        if ks_triggered and open_pos is not None:
            exit_signal = True
            pre_exit = "kill_switch_close"
        if open_pos is not None:
            open_pos["hold_bars"] += 1
            if _opposite_signal(open_pos["side"], signal):
                exit_signal = True
                pre_exit = pre_exit or "opposite_signal"
            elif open_pos["hold_bars"] >= PAPER_MAX_HOLDING_BARS:
                exit_signal = True
                pre_exit = pre_exit or "max_holding_bars"

        if exit_signal and open_pos is not None:
            raw = _unrealized(open_pos["side"], open_pos["entry_price"], price)
            net = raw - 2.0 * (fee_rate + slippage_rate)
            equity *= 1.0 + net * position_size
            peak = max(peak, equity)
            daily_pnl = equity - 1.0
            consec_losses = consec_losses + 1 if net < 0 else 0
            recent_net_returns.append(net)
            trades.append(
                {
                    "direction": open_pos["direction"],
                    "entropy": open_pos["entropy"],
                    "trend_state": open_pos["trend_state"],
                    "hold_bars": open_pos["hold_bars"],
                    "exit_reason": pre_exit or "exit_signal",
                    "raw_return": raw,
                    "net_return": net,
                    "is_win": net > 0,
                    "failure_signature": int(
                        open_pos["direction"] == "LONG"
                        and 0.95 <= open_pos["entropy"] <= 1.00
                        and open_pos["hold_bars"] < 6
                        and (pre_exit or "exit_signal") == "risk_force_exit"
                    ),
                }
            )
            exit_signal_count += 1
            open_pos = None
            continue

        # risk_force_exit branch while holding
        if open_pos is not None:
            raw_u = _unrealized(open_pos["side"], open_pos["entry_price"], price)
            risk_cond = raw_u <= MAX_TRADE_LOSS
            if risk_cond:
                open_pos["risk_consec"] += 1
            else:
                open_pos["risk_consec"] = 0
                open_pos["grace_left"] = 0

            recent_24 = mean(recent_net_returns[-24:]) if recent_net_returns else 0.0
            action = _policy_action(
                variant_name,
                open_pos["direction"],
                open_pos["entropy"],
                open_pos["hold_bars"],
                open_pos["trend_state"],
                recent_24,
                open_pos["risk_consec"],
                open_pos["grace_left"],
            )

            do_exit = False
            if risk_cond:
                if action == "immediate_exit":
                    do_exit = True
                elif action == "grace":
                    if open_pos["grace_left"] <= 0:
                        open_pos["grace_left"] = 1
                    else:
                        open_pos["grace_left"] -= 1
                        if open_pos["grace_left"] <= 0:
                            do_exit = True
                elif action == "confirm2":
                    if open_pos["risk_consec"] >= 2:
                        do_exit = True
                elif action == "ignore":
                    do_exit = False

            if do_exit:
                raw = raw_u
                net = raw - 2.0 * (fee_rate + slippage_rate)
                equity *= 1.0 + net * position_size
                peak = max(peak, equity)
                daily_pnl = equity - 1.0
                consec_losses = consec_losses + 1 if net < 0 else 0
                recent_net_returns.append(net)
                trades.append(
                    {
                        "direction": open_pos["direction"],
                        "entropy": open_pos["entropy"],
                        "trend_state": open_pos["trend_state"],
                        "hold_bars": open_pos["hold_bars"],
                        "exit_reason": "risk_force_exit",
                        "raw_return": raw,
                        "net_return": net,
                        "is_win": net > 0,
                        "failure_signature": int(
                            open_pos["direction"] == "LONG"
                            and 0.95 <= open_pos["entropy"] <= 1.00
                            and open_pos["hold_bars"] < 6
                            and True
                        ),
                    }
                )
                risk_exit_count += 1
                open_pos = None
                continue
            # hold
            continue

        # no-position: pipeline
        if vol_bucket not in ("mid", "high"):
            continue
        strategy = "S2" if trend_state != "sideways" else "S1"
        if strategy == "S2" and entropy > 1.0:
            continue
        if signal is None:
            continue

        drawdown = (equity - peak) / peak if peak > 0 else 0.0
        stop = (
            ks_triggered
            or daily_pnl <= MAX_DAILY_LOSS
            or drawdown <= MAX_DRAWDOWN
            or consec_losses >= MAX_CONSEC_LOSSES
        )
        if stop:
            ks_triggered = True
            if ks_trade_idx < 0:
                ks_trade_idx = max(0, len(trades) - 1)
            continue

        side = "BUY" if signal == "LONG" else "SELL"
        direction = "LONG" if side == "BUY" else "SHORT"
        if direction == "LONG":
            long_count += 1
        else:
            short_count += 1
        open_pos = {
            "side": side,
            "direction": direction,
            "entry_price": price,
            "entropy": entropy,
            "trend_state": trend_state,
            "hold_bars": 0,
            "risk_consec": 0,
            "grace_left": 0,
        }

    tdf = pd.DataFrame(trades)
    returns = tdf["net_return"].tolist() if not tdf.empty else []
    gross = tdf["raw_return"].tolist() if not tdf.empty else []
    wins = [x for x in returns if x > 0]
    losses = [x for x in returns if x < 0]
    wr = len(wins) / len(returns) if returns else 0.0
    avg_win = mean(wins) if wins else 0.0
    avg_loss = mean(losses) if losses else 0.0
    expectancy = (wr * avg_win) - ((1.0 - wr) * abs(avg_loss)) if returns else 0.0
    pf = (sum(wins) / abs(sum(losses))) if losses else 0.0

    eq = 1.0
    peak2 = 1.0
    mdd = 0.0
    max_consec = 0
    cons = 0
    for r in returns:
        eq *= 1.0 + r * position_size
        peak2 = max(peak2, eq)
        mdd = min(mdd, (eq - peak2) / peak2 if peak2 > 0 else 0.0)
        if r < 0:
            cons += 1
            max_consec = max(max_consec, cons)
        else:
            cons = 0

    long_rets = tdf[tdf["direction"] == "LONG"]["net_return"].tolist() if not tdf.empty else []
    short_rets = tdf[tdf["direction"] == "SHORT"]["net_return"].tolist() if not tdf.empty else []
    long_exp = mean(long_rets) if long_rets else 0.0
    short_exp = mean(short_rets) if short_rets else 0.0

    worst_len, worst_sum, top5_sum = _loss_clusters(returns)
    failure_sig_count = int(tdf["failure_signature"].sum()) if not tdf.empty else 0

    metrics = {
        "total_trades": len(tdf),
        "win_rate": wr,
        "avg_return": mean(returns) if returns else 0.0,
        "median_return": median(returns) if returns else 0.0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "profit_factor": pf,
        "gross_return_sum": sum(gross) if gross else 0.0,
        "net_return_sum": sum(returns) if returns else 0.0,
        "equity_return": eq - 1.0,
        "max_drawdown": mdd,
        "max_consecutive_losses": max_consec,
        "kill_switch_triggered": ks_triggered,
        "kill_switch_trade_index": ks_trade_idx,
        "risk_force_exit_count": risk_exit_count,
        "exit_signal_count": exit_signal_count,
        "avg_hold_bars": float(tdf["hold_bars"].mean()) if not tdf.empty else 0.0,
        "avg_hold_bars_winners": float(tdf[tdf["net_return"] > 0]["hold_bars"].mean()) if not tdf.empty else 0.0,
        "avg_hold_bars_losers": float(tdf[tdf["net_return"] < 0]["hold_bars"].mean()) if not tdf.empty else 0.0,
        "fee_cost_sum": len(tdf) * 2.0 * fee_rate,
        "slippage_cost_sum": len(tdf) * 2.0 * slippage_rate,
        "long_trade_count": long_count,
        "short_trade_count": short_count,
        "long_expectancy": long_exp,
        "short_expectancy": short_exp,
        "worst_loss_cluster_length": worst_len,
        "worst_loss_cluster_sum": worst_sum,
        "top5_loss_cluster_sum": top5_sum,
        "ks_prev5_loss_reproduced": _ks_prev5_reproduced(returns),
        "failure_signature_count": failure_sig_count,
    }
    return metrics, tdf


def _grade_variant(row: pd.Series) -> str:
    cond_a = (
        row["improved_periods"] >= 3
        and row["expectancy_delta_avg"] > 0
        and row["net_return_delta_avg"] > 0
        and row["profit_factor_delta_avg"] > 0
        and row["max_drawdown_delta_avg"] >= 0
        and row["kill_switch_reduction_count"] >= 0
        and row["worst_loss_cluster_sum_delta_avg"] > 0
        and row["failure_signature_reduction_avg"] > 0
    )
    if cond_a:
        return "A"
    cond_b = row["improved_periods"] >= 2 and row["net_return_delta_avg"] > 0
    if cond_b:
        return "B"
    return "C"


def main() -> None:
    args = _parse_args()
    lookbacks = [int(x.strip()) for x in args.lookback_list.split(",") if x.strip()]
    variants = _variants()

    df = load_ohlcv()
    ticks, _ = simulate_signals_paper(df)

    rows: List[Dict[str, Any]] = []
    for lb in lookbacks:
        lb_ticks = _slice_ticks(ticks, lb)
        per_case: Dict[str, Dict[str, Any]] = {}
        for v in variants:
            met, _ = _simulate_variant(
                lb_ticks,
                v.name,
                position_size=args.position_size,
                fee_rate=args.fee_rate,
                slippage_rate=args.slippage_rate,
            )
            per_case[v.name] = met
            rows.append({"lookback_days": lb, "variant": v.name, **met})

        base = per_case["current"]
        for r in rows:
            if r["lookback_days"] != lb:
                continue
            r["expectancy_delta"] = r["expectancy"] - base["expectancy"]
            r["net_return_delta"] = r["net_return_sum"] - base["net_return_sum"]
            r["equity_return_delta"] = r["equity_return"] - base["equity_return"]
            r["MDD_delta"] = r["max_drawdown"] - base["max_drawdown"]
            r["PF_delta"] = r["profit_factor"] - base["profit_factor"]
            r["KS_avoided"] = bool(base["kill_switch_triggered"] and (not r["kill_switch_triggered"]))
            r["risk_force_exit_reduction"] = base["risk_force_exit_count"] - r["risk_force_exit_count"]
            r["max_consecutive_losses_delta"] = r["max_consecutive_losses"] - base["max_consecutive_losses"]
            r["long_expectancy_delta"] = r["long_expectancy"] - base["long_expectancy"]
            r["short_expectancy_delta"] = r["short_expectancy"] - base["short_expectancy"]
            r["worst_loss_cluster_sum_delta"] = base["worst_loss_cluster_sum"] - r["worst_loss_cluster_sum"]
            r["failure_signature_reduction"] = base["failure_signature_count"] - r["failure_signature_count"]
            r["improved"] = (
                r["variant"] != "current"
                and r["net_return_delta"] > 0
                and r["expectancy_delta"] > 0
                and r["PF_delta"] > 0
            )

    out = pd.DataFrame(rows).sort_values(["lookback_days", "variant"]).reset_index(drop=True)

    # variant summary vs current
    sum_rows = []
    for v in [x.name for x in variants if x.name != "current"]:
        sub = out[out["variant"] == v]
        sum_rows.append(
            {
                "variant": v,
                "improved_periods": int(sub["improved"].sum()),
                "expectancy_delta_avg": float(sub["expectancy_delta"].mean()),
                "net_return_delta_avg": float(sub["net_return_delta"].mean()),
                "equity_return_delta_avg": float(sub["equity_return_delta"].mean()),
                "max_drawdown_delta_avg": float(sub["MDD_delta"].mean()),
                "profit_factor_delta_avg": float(sub["PF_delta"].mean()),
                "kill_switch_reduction_count": int(sub["KS_avoided"].sum()),
                "max_consecutive_losses_delta_avg": float(sub["max_consecutive_losses_delta"].mean()),
                "risk_force_exit_reduction_avg": float(sub["risk_force_exit_reduction"].mean()),
                "long_expectancy_delta_avg": float(sub["long_expectancy_delta"].mean()),
                "short_expectancy_delta_avg": float(sub["short_expectancy_delta"].mean()),
                "worst_loss_cluster_sum_delta_avg": float(sub["worst_loss_cluster_sum_delta"].mean()),
                "failure_signature_reduction_avg": float(sub["failure_signature_reduction"].mean()),
            }
        )
    summary = pd.DataFrame(sum_rows)
    summary["verdict"] = summary.apply(_grade_variant, axis=1)
    summary = summary.sort_values(
        ["verdict", "improved_periods", "net_return_delta_avg", "expectancy_delta_avg"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    best = summary.iloc[0]

    recommendation = "hold"
    if best["verdict"] == "A":
        recommendation = "apply_candidate"
    elif best["verdict"] == "B":
        recommendation = "forward_monitor"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"conditional_risk_force_exit_replay_{ts}.csv"
    md_path = OUT_DIR / f"conditional_risk_force_exit_replay_{ts}.md"

    out.to_csv(csv_path, index=False)

    md: List[str] = []
    md.append("# Can_bit Conditional Risk Force Exit Replay Report")
    md.append("")
    md.append("## 1. 목적")
    md.append("- 손실 signature 조건에서만 risk_force_exit 정책을 다르게 적용했을 때 개선 여부 검증")
    md.append("")
    md.append("## 2. 실험 조건")
    md.append(f"- lookback_list: {lookbacks}")
    md.append(f"- position_size: {args.position_size}")
    md.append(f"- fee_rate/slippage_rate: {args.fee_rate}/{args.slippage_rate}")
    md.append("")
    md.append("## 3. baseline current 결과")
    for lb in lookbacks:
        r = out[(out["lookback_days"] == lb) & (out["variant"] == "current")].iloc[0]
        md.append(
            f"- {lb}d: net={r['net_return_sum']:.6f}, exp={r['expectancy']:.6f}, PF={r['profit_factor']:.4f}, "
            f"MDD={r['max_drawdown']:.6f}, KS={bool(r['kill_switch_triggered'])}, failure_sig={int(r['failure_signature_count'])}"
        )
    md.append("")
    md.append("## 4. global 후보 결과")
    for v in ["global_grace_1", "confirm_2_consecutive"]:
        s = summary[summary["variant"] == v]
        if len(s):
            rr = s.iloc[0]
            md.append(
                f"- {v}: verdict={rr['verdict']}, improved={int(rr['improved_periods'])}, "
                f"netΔavg={rr['net_return_delta_avg']:+.6f}, expΔavg={rr['expectancy_delta_avg']:+.6f}"
            )
    md.append("")
    md.append("## 5. 조건부 variant 결과표")
    for _, r in summary.iterrows():
        md.append(
            f"- {r['variant']}: verdict={r['verdict']}, improved={int(r['improved_periods'])}, "
            f"netΔavg={r['net_return_delta_avg']:+.6f}, expΔavg={r['expectancy_delta_avg']:+.6f}, "
            f"PFΔavg={r['profit_factor_delta_avg']:+.4f}, MDDΔavg={r['max_drawdown_delta_avg']:+.6f}"
        )
    md.append("")
    md.append("## 6. 기간별 결과")
    for lb in lookbacks:
        sub = out[(out["lookback_days"] == lb) & (out["variant"] != "current")].sort_values(
            ["net_return_delta", "expectancy_delta"], ascending=[False, False]
        )
        if len(sub):
            w = sub.iloc[0]
            md.append(
                f"- {lb}d winner: {w['variant']} (netΔ={w['net_return_delta']:+.6f}, expΔ={w['expectancy_delta']:+.6f})"
            )
    md.append("")
    md.append("## 7. consistency summary")
    md.append("- improved_periods 기준으로 4개 기간 중 개선 빈도 평가")
    md.append("")
    md.append("## 8. loss cluster reduction 분석")
    for _, r in summary.head(5).iterrows():
        md.append(
            f"- {r['variant']}: worst_cluster_sumΔavg={r['worst_loss_cluster_sum_delta_avg']:+.6f}, "
            f"failure_signature_reduction_avg={r['failure_signature_reduction_avg']:+.2f}"
        )
    md.append("")
    md.append("## 9. KS 변화")
    for _, r in summary.head(5).iterrows():
        md.append(f"- {r['variant']}: KS_reduction_count={int(r['kill_switch_reduction_count'])}")
    md.append("")
    md.append("## 10. MDD 변화")
    for _, r in summary.head(5).iterrows():
        md.append(f"- {r['variant']}: MDD_delta_avg={r['max_drawdown_delta_avg']:+.6f}")
    md.append("")
    md.append("## 11. final candidate")
    md.append(
        f"- {best['variant']} (verdict={best['verdict']}): improved={int(best['improved_periods'])}, "
        f"netΔavg={best['net_return_delta_avg']:+.6f}, expΔavg={best['expectancy_delta_avg']:+.6f}"
    )
    md.append("")
    md.append("## 12. 운영 적용 가능 여부")
    md.append("- diagnostics 결과이며 운영 로직/기본값 변경 없음")
    md.append(f"- recommendation: {recommendation}")
    md.append("")
    md.append("## 13. next action")
    md.append("- best variant를 rolling window OOS로 재검증하고, KS/cluster 지표를 함께 모니터링")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("[CONDITIONAL RISK FORCE EXIT REPLAY SUMMARY]")
    print("")
    print("best_variant:")
    print(f"- name: {best['variant']}")
    print(f"- verdict: {best['verdict']}")
    print(f"- improved_periods: {int(best['improved_periods'])}")
    print(f"- expectancy_delta_avg: {best['expectancy_delta_avg']:+.6f}")
    print(f"- net_delta_avg: {best['net_return_delta_avg']:+.6f}")
    print(f"- PF_delta_avg: {best['profit_factor_delta_avg']:+.6f}")
    print(f"- MDD_delta_avg: {best['max_drawdown_delta_avg']:+.6f}")
    print(f"- KS_reduction: {int(best['kill_switch_reduction_count'])}")
    print(f"- failure_signature_reduction: {best['failure_signature_reduction_avg']:+.2f}")
    print("")
    print("final_recommendation:")
    print(f"- {recommendation}")
    print("")
    print("created:")
    print(f"- csv: {csv_path}")
    print(f"- markdown: {md_path}")


if __name__ == "__main__":
    main()
