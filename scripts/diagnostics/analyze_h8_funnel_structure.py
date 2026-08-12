"""
H8 funnel structure diagnostics (diagnostics only).

Goal:
- Determine whether H8 (entropy<=0.96 AND trend==up) is selective or over-filtered.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from scripts.diagnostics.validate_entry_quality_filters import _slice_ticks_by_range, _window_endpoints
from scripts.diagnostics.validate_final_hypotheses import _simulate
from scripts.run_daily_paper import load_ohlcv, simulate_signals_paper


OUT_DIR = Path("data/diagnostics/h8_funnel")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("analyze_h8_funnel_structure")
    p.add_argument("--lookback-list", default="14,30,60,90,180")
    p.add_argument("--stride-days", type=int, default=7)
    p.add_argument("--max-windows-per-lookback", type=int, default=9999)
    p.add_argument("--position-size", type=float, default=0.05)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0002)
    return p.parse_args()


def _entropy(t: Dict[str, Any]) -> float:
    e = 0.0
    for p in (float(t["p_long"]), float(t["p_short"]), float(t["p_flat"])):
        if p > 1e-10:
            e -= p * math.log(p)
    return e


def _expectancy(vals: List[float]) -> float:
    if not vals:
        return 0.0
    wins = [x for x in vals if x > 0]
    losses = [x for x in vals if x < 0]
    wr = len(wins) / len(vals)
    avg_w = mean(wins) if wins else 0.0
    avg_l = mean(losses) if losses else 0.0
    return (wr * avg_w) - ((1.0 - wr) * abs(avg_l))


def _pf(vals: List[float]) -> float:
    wins = sum(x for x in vals if x > 0)
    losses = abs(sum(x for x in vals if x < 0))
    return wins / losses if losses > 0 else 0.0


def _entropy_bucket(ent: float) -> str:
    if ent < 0.90:
        return "<0.90"
    if ent < 0.96:
        return "0.90~0.96"
    if ent < 1.00:
        return "0.96~1.00"
    return ">=1.00"


def _funnel_counts(ticks: List[Dict[str, Any]]) -> Dict[str, Any]:
    activation_pass = 0
    production_candidates = 0
    entropy_le_1_pass = 0
    entropy_le_096_pass = 0
    trend_up_pass = 0
    h8_intersection = 0
    blocked_only_entropy = 0
    blocked_only_trend = 0
    blocked_both = 0

    survive_regs: Counter[str] = Counter()
    fail_regs: Counter[str] = Counter()

    for t in ticks:
        vol = str(t.get("vol_bucket") or "")
        sig = t.get("signal")
        trend = str(t.get("trend_label") or "")
        ent = _entropy(t)
        if vol in ("mid", "high"):
            activation_pass += 1
        else:
            continue

        if sig is None:
            continue
        production_candidates += 1

        ent1 = ent <= 1.0
        ent096 = ent <= 0.96
        tr_up = trend == "up"

        if ent1:
            entropy_le_1_pass += 1
        if ent096:
            entropy_le_096_pass += 1
        if tr_up:
            trend_up_pass += 1
        if ent096 and tr_up:
            h8_intersection += 1
            survive_regs[f"{trend}|{vol}|{_entropy_bucket(ent)}"] += 1
        else:
            fail_regs[f"{trend}|{vol}|{_entropy_bucket(ent)}"] += 1
            if (not ent096) and tr_up:
                blocked_only_entropy += 1
            elif ent096 and (not tr_up):
                blocked_only_trend += 1
            elif (not ent096) and (not tr_up):
                blocked_both += 1

    denom = max(production_candidates, 1)
    return {
        "activation_pass": activation_pass,
        "production_candidates": production_candidates,
        "entropy_le_1_pass": entropy_le_1_pass,
        "entropy_le_0_96_pass": entropy_le_096_pass,
        "trend_up_pass": trend_up_pass,
        "h8_intersection_pass": h8_intersection,
        "blocked_only_by_entropy": blocked_only_entropy,
        "blocked_only_by_trend": blocked_only_trend,
        "blocked_by_both": blocked_both,
        "entropy_survival_ratio": entropy_le_096_pass / denom,
        "trend_survival_ratio": trend_up_pass / denom,
        "intersection_ratio": h8_intersection / denom,
        "most_common_h8_survival_regime": survive_regs.most_common(1)[0][0] if survive_regs else "n/a",
        "most_common_h8_failure_regime": fail_regs.most_common(1)[0][0] if fail_regs else "n/a",
    }


def _sim_metrics(
    ticks: List[Dict[str, Any]],
    params: Dict[str, Any],
    position_size: float,
    fee_rate: float,
    slippage_rate: float,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    m, tdf = _simulate(ticks, params, position_size, fee_rate, slippage_rate)
    vals = tdf["net_return"].tolist() if not tdf.empty else []
    m2 = {
        "trades": int(m["total_trades"]),
        "win_rate": float(m["win_rate"]),
        "expectancy": float(m["expectancy"]),
        "profit_factor": float(m["profit_factor"]),
        "net_return": float(m["net_return"]),
        "MDD": float(m["max_drawdown"]),
        "risk_force_exit_count": int(m["risk_force_exit_count"]),
        "KS_triggered": int(bool(m["kill_switch_triggered"])),
        "avg_hold": float(tdf["hold_bars"].mean()) if not tdf.empty else 0.0,
        "vals": vals,
    }
    return m2, tdf


def _no_trade_reason(row: pd.Series) -> str:
    ir = float(row["intersection_ratio"])
    tr = float(row["trend_survival_ratio"])
    er = float(row["entropy_survival_ratio"])
    if ir < 0.03 and tr > 0.15 and er > 0.15:
        return "intersection collapse"
    if er < 0.05:
        return "entropy overfilter"
    if tr < 0.05:
        return "trend overfilter"
    if ir < 0.05 and tr < 0.10 and er < 0.10:
        return "market regime incompatible"
    return "healthy/neutral"


def _grade(final_row: pd.Series) -> str:
    if bool(final_row["is_no_trade_filter"]):
        return "D"
    if float(final_row["H8_precision_score"]) >= 0.65 and float(final_row["good_trade_preservation_rate"]) >= 0.55:
        return "A"
    if float(final_row["H8_precision_score"]) >= 0.55:
        return "B"
    return "C"


def main() -> None:
    args = _parse_args()
    lookbacks = [int(x.strip()) for x in args.lookback_list.split(",") if x.strip()]

    df = load_ohlcv()
    ticks, _ = simulate_signals_paper(df)

    detail_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for lb in lookbacks:
        wins = _window_endpoints(ticks, lb, args.stride_days, args.max_windows_per_lookback)
        for wid, (ws, we) in enumerate(wins):
            w_ticks = _slice_ticks_by_range(ticks, ws, we)
            if len(w_ticks) < 30:
                continue

            funnel = _funnel_counts(w_ticks)

            base_m, base_tdf = _sim_metrics(w_ticks, {}, args.position_size, args.fee_rate, args.slippage_rate)
            ent_m, ent_tdf = _sim_metrics(w_ticks, {"entropy_ub": 0.96}, args.position_size, args.fee_rate, args.slippage_rate)
            tr_m, tr_tdf = _sim_metrics(w_ticks, {"trend_up_only": True}, args.position_size, args.fee_rate, args.slippage_rate)
            h8_m, h8_tdf = _sim_metrics(w_ticks, {"entropy_ub": 0.96, "trend_up_only": True}, args.position_size, args.fee_rate, args.slippage_rate)

            trade_denom = max(base_m["trades"], 1)
            h8_removed = set(base_tdf["entry_idx"].astype(int).tolist()) - set(h8_tdf["entry_idx"].astype(int).tolist())
            removed_df = base_tdf[base_tdf["entry_idx"].isin(h8_removed)] if not base_tdf.empty else pd.DataFrame()
            bad_removed = int((removed_df["net_return"] < 0).sum()) if not removed_df.empty else 0
            good_removed = int((removed_df["net_return"] > 0).sum()) if not removed_df.empty else 0
            base_good = int((base_tdf["net_return"] > 0).sum()) if not base_tdf.empty else 0
            base_bad = int((base_tdf["net_return"] < 0).sum()) if not base_tdf.empty else 0

            row = {
                "lookback_days": lb,
                "window_id": wid,
                "window_start": ws,
                "window_end": we,
                **funnel,
                "no_trade_reason": _no_trade_reason(pd.Series(funnel)),
                # A/B/C/D metrics
                "baseline_trades": base_m["trades"],
                "baseline_win_rate": base_m["win_rate"],
                "baseline_expectancy": base_m["expectancy"],
                "baseline_profit_factor": base_m["profit_factor"],
                "baseline_net_return": base_m["net_return"],
                "baseline_MDD": base_m["MDD"],
                "baseline_risk_force_exit_count": base_m["risk_force_exit_count"],
                "baseline_KS_triggered": base_m["KS_triggered"],
                "baseline_avg_hold": base_m["avg_hold"],
                "entropy_only_trades": ent_m["trades"],
                "entropy_only_expectancy": ent_m["expectancy"],
                "entropy_only_profit_factor": ent_m["profit_factor"],
                "entropy_only_net_return": ent_m["net_return"],
                "entropy_only_MDD": ent_m["MDD"],
                "trend_only_trades": tr_m["trades"],
                "trend_only_expectancy": tr_m["expectancy"],
                "trend_only_profit_factor": tr_m["profit_factor"],
                "trend_only_net_return": tr_m["net_return"],
                "trend_only_MDD": tr_m["MDD"],
                "h8_trades": h8_m["trades"],
                "h8_win_rate": h8_m["win_rate"],
                "h8_expectancy": h8_m["expectancy"],
                "h8_profit_factor": h8_m["profit_factor"],
                "h8_net_return": h8_m["net_return"],
                "h8_MDD": h8_m["MDD"],
                "h8_risk_force_exit_count": h8_m["risk_force_exit_count"],
                "h8_KS_triggered": h8_m["KS_triggered"],
                "h8_avg_hold": h8_m["avg_hold"],
                "h8_trade_reduction_pct": (1.0 - h8_m["trades"] / trade_denom) * 100.0,
                "h8_expectancy_delta": h8_m["expectancy"] - base_m["expectancy"],
                "h8_pf_delta": h8_m["profit_factor"] - base_m["profit_factor"],
                "h8_net_return_delta": h8_m["net_return"] - base_m["net_return"],
                "h8_mdd_delta": h8_m["MDD"] - base_m["MDD"],
                "h8_ks_delta": h8_m["KS_triggered"] - base_m["KS_triggered"],
                "h8_force_exit_delta": h8_m["risk_force_exit_count"] - base_m["risk_force_exit_count"],
                "entropy_survive_trade_pct": ent_m["trades"] / trade_denom,
                "trend_survive_trade_pct": tr_m["trades"] / trade_denom,
                "h8_survive_trade_pct": h8_m["trades"] / trade_denom,
                "bad_trade_removed_proxy": bad_removed,
                "good_trade_removed_proxy": good_removed,
                "removal_precision": bad_removed / max(len(removed_df), 1),
                "good_trade_removed_pct": good_removed / max(base_good, 1),
                "bad_trade_removed_pct": bad_removed / max(base_bad, 1),
            }
            detail_rows.append(row)

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        print("best explanation of no-trade behavior: insufficient data")
        print("which condition is the bottleneck: unknown")
        print("whether H8 is: regime-sensitive")
        print("whether H8 should: continue monitor")
        print("FINAL VERDICT: D")
        print("top 3 actionable next steps: collect more windows; verify data alignment; rerun diagnostics")
        return

    for lb, sub in detail.groupby("lookback_days"):
        improved = int((sub["h8_net_return_delta"] > 0).sum())
        degraded = int((sub["h8_net_return_delta"] < 0).sum())
        arr = sub["h8_net_return_delta"].tolist()
        avg_d = float(mean(arr)) if arr else 0.0
        std_d = float(pstdev(arr)) if len(arr) > 1 else 0.0
        summary_rows.append(
            {
                "lookback_days": lb,
                "windows": len(sub),
                "improved_windows": improved,
                "degraded_windows": degraded,
                "improved_ratio": improved / max(len(sub), 1),
                "avg_delta": avg_d,
                "std_delta": std_d,
                "entropy_survival_ratio_mean": float(sub["entropy_survival_ratio"].mean()),
                "trend_survival_ratio_mean": float(sub["trend_survival_ratio"].mean()),
                "intersection_ratio_mean": float(sub["intersection_ratio"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    # aggregate final
    reason_counts = Counter(detail["no_trade_reason"].tolist())
    dominant_reason = reason_counts.most_common(1)[0][0]
    bottleneck_counts = {
        "entropy<=0.96": float(detail["blocked_only_by_entropy"].mean()),
        "trend==up": float(detail["blocked_only_by_trend"].mean()),
        "both_intersection": float(detail["blocked_by_both"].mean()),
    }
    bottleneck = max(bottleneck_counts.items(), key=lambda x: x[1])[0]

    avg_intersection_ratio = float(detail["intersection_ratio"].mean())
    good_pres = 1.0 - float(detail["good_trade_removed_pct"].mean())
    bad_rem = float(detail["bad_trade_removed_pct"].mean())
    precision = float(detail["removal_precision"].mean())
    overfilter = float(detail["h8_trade_reduction_pct"].mean()) / 100.0
    no_trade_flag = bool((detail["intersection_ratio"] < 0.03).mean() > 0.5)

    intersection_health_score = max(0.0, min(1.0, 0.5 * avg_intersection_ratio * 10 + 0.3 * good_pres + 0.2 * (1.0 - overfilter)))
    h8_precision_score = max(0.0, min(1.0, 0.6 * precision + 0.4 * bad_rem))
    h8_overfilter_score = max(0.0, min(1.0, 0.6 * overfilter + 0.4 * (1.0 - good_pres)))

    if no_trade_flag and dominant_reason == "intersection collapse":
        action = "split H8 into soft risk gate"
    elif dominant_reason == "entropy overfilter":
        action = "relax entropy"
    elif dominant_reason == "trend overfilter":
        action = "relax trend"
    elif h8_precision_score >= 0.60 and h8_overfilter_score <= 0.55:
        action = "continue monitor"
    else:
        action = "reject H8"

    if action == "continue monitor" and h8_precision_score >= 0.65:
        h8_type = "selective"
    elif h8_overfilter_score >= 0.65:
        h8_type = "overfiltered"
    else:
        h8_type = "regime-sensitive"

    final_row = pd.Series(
        {
            "largest_filter_bottleneck": bottleneck,
            "intersection_health_score": intersection_health_score,
            "good_trade_preservation_rate": good_pres,
            "bad_trade_removal_rate": bad_rem,
            "H8_precision_score": h8_precision_score,
            "H8_overfilter_score": h8_overfilter_score,
            "is_no_trade_filter": no_trade_flag,
            "recommended_action": action,
            "FINAL_VERDICT": "",  # filled below
            "dominant_no_trade_reason": dominant_reason,
            "h8_type": h8_type,
        }
    )
    final_row["FINAL_VERDICT"] = _grade(final_row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"h8_funnel_structure_{ts}.csv"
    md_path = OUT_DIR / f"h8_funnel_structure_{ts}.md"

    out = pd.concat(
        [
            detail.assign(section="rolling_detail"),
            summary.assign(section="lookback_summary"),
            pd.DataFrame([final_row]).assign(section="final_verdict"),
        ],
        ignore_index=True,
        sort=False,
    )
    out.to_csv(csv_path, index=False)

    md_lines = [
        "# H8 Funnel Structure Diagnostics",
        "",
        f"- windows: {len(detail)}",
        f"- largest_filter_bottleneck: {final_row['largest_filter_bottleneck']}",
        f"- dominant_no_trade_reason: {final_row['dominant_no_trade_reason']}",
        f"- intersection_health_score: {final_row['intersection_health_score']:.4f}",
        f"- good_trade_preservation_rate: {final_row['good_trade_preservation_rate']:.4f}",
        f"- bad_trade_removal_rate: {final_row['bad_trade_removal_rate']:.4f}",
        f"- H8_precision_score: {final_row['H8_precision_score']:.4f}",
        f"- H8_overfilter_score: {final_row['H8_overfilter_score']:.4f}",
        f"- is_no_trade_filter: {final_row['is_no_trade_filter']}",
        f"- recommended_action: {final_row['recommended_action']}",
        f"- FINAL_VERDICT: {final_row['FINAL_VERDICT']}",
        "",
        "## Bottleneck means",
        f"- entropy<=0.96 blocked mean: {bottleneck_counts['entropy<=0.96']:.2f}",
        f"- trend==up blocked mean: {bottleneck_counts['trend==up']:.2f}",
        f"- both_intersection blocked mean: {bottleneck_counts['both_intersection']:.2f}",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # required final print-only blocks
    print(f"best explanation of no-trade behavior: {final_row['dominant_no_trade_reason']}")
    print(f"which condition is the bottleneck: {final_row['largest_filter_bottleneck']}")
    print(f"whether H8 is: {final_row['h8_type']}")
    print(f"whether H8 should: {final_row['recommended_action']}")
    print(f"FINAL VERDICT: {final_row['FINAL_VERDICT']}")
    print("top 3 actionable next steps: "
          "1) compare H8 vs entropy-only blocked windows; "
          "2) test soft-gate weighting (not hard intersection) in diagnostics; "
          "3) run monthly forward monitor with minimum-trade floor")


if __name__ == "__main__":
    main()

