"""
Validate H8 soft risk gate (diagnostics replay only).

Hard H8:
  - block entries if H8 fails
Soft H8:
  - keep entries, scale position size when H8 fails
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.diagnostics.validate_entry_quality_filters import _slice_ticks_by_range, _window_endpoints
from scripts.run_daily_paper import PAPER_MAX_HOLDING_BARS, load_ohlcv, simulate_signals_paper


OUT_DIR = Path("data/diagnostics/h8_soft_gate")
MAX_TRADE_LOSS = -0.01
MAX_DAILY_LOSS = -0.02
MAX_DRAWDOWN = -0.05
MAX_CONSEC_LOSSES = 5


@dataclass
class Variant:
    name: str
    fail_scale: float
    hard_block: bool = False


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("validate_h8_soft_risk_gate")
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


def _h8_pass(t: Dict[str, Any]) -> bool:
    return (_entropy(t) <= 0.96) and (str(t.get("trend_label") or "") == "up")


def _is_loss_cluster_trade(tr: Dict[str, Any], t: Dict[str, Any]) -> bool:
    if tr.get("exit_reason") == "risk_force_exit":
        return True
    ent = _entropy(t)
    trend = str(t.get("trend_label") or "")
    vol = str(t.get("vol_bucket") or "")
    return (
        str(tr.get("direction")) == "LONG"
        and 0.90 <= ent <= 0.95
        and vol == "high"
        and trend == "down"
        and int(tr.get("hold_bars", 0)) < 6
    )


def _expectancy(vals: List[float]) -> float:
    if not vals:
        return 0.0
    wins = [x for x in vals if x > 0]
    losses = [x for x in vals if x < 0]
    wr = len(wins) / len(vals)
    avg_w = mean(wins) if wins else 0.0
    avg_l = mean(losses) if losses else 0.0
    return (wr * avg_w) - ((1 - wr) * abs(avg_l))


def _simulate_variant(
    ticks: List[Dict[str, Any]],
    variant: Variant,
    position_size: float,
    fee_rate: float,
    slippage_rate: float,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    trades: List[Dict[str, Any]] = []
    open_pos: Optional[Dict[str, Any]] = None

    eq = 1.0
    peak = 1.0
    daily_pnl = 0.0
    consec_losses = 0
    ks_triggered = False
    risk_force = 0

    for i, t in enumerate(ticks):
        px = float(t["price"])
        sig = t.get("signal")
        vol = str(t.get("vol_bucket") or "")
        trend = str(t.get("trend_label") or "")

        exit_signal = False
        reason = ""
        if ks_triggered and open_pos is not None:
            exit_signal = True
            reason = "kill_switch_close"
        if open_pos is not None:
            open_pos["hold_bars"] += 1
            if sig is not None and (
                (open_pos["side"] == "BUY" and sig == "SHORT")
                or (open_pos["side"] == "SELL" and sig == "LONG")
            ):
                exit_signal = True
                reason = reason or "opposite_signal"
            elif open_pos["hold_bars"] >= PAPER_MAX_HOLDING_BARS:
                exit_signal = True
                reason = reason or "max_holding_bars"

        if exit_signal and open_pos is not None:
            raw = (
                (px - open_pos["entry_price"]) / open_pos["entry_price"]
                if open_pos["side"] == "BUY"
                else (open_pos["entry_price"] - px) / open_pos["entry_price"]
            )
            net = raw - 2.0 * (fee_rate + slippage_rate)
            scale = float(open_pos["scale"])
            eq *= 1.0 + net * position_size * scale
            peak = max(peak, eq)
            daily_pnl = eq - 1.0
            consec_losses = consec_losses + 1 if net * scale < 0 else 0
            trades.append(
                {
                    "entry_idx": open_pos["entry_idx"],
                    "direction": open_pos["direction"],
                    "hold_bars": open_pos["hold_bars"],
                    "exit_reason": reason or "exit_signal",
                    "gross_return": raw,
                    "net_return": net,
                    "scaled_return": net * scale,
                    "scale": scale,
                    "h8_pass": bool(open_pos["h8_pass"]),
                }
            )
            open_pos = None
            continue

        if open_pos is not None:
            unreal = (
                (px - open_pos["entry_price"]) / open_pos["entry_price"]
                if open_pos["side"] == "BUY"
                else (open_pos["entry_price"] - px) / open_pos["entry_price"]
            )
            if unreal <= MAX_TRADE_LOSS:
                net = unreal - 2.0 * (fee_rate + slippage_rate)
                scale = float(open_pos["scale"])
                eq *= 1.0 + net * position_size * scale
                peak = max(peak, eq)
                daily_pnl = eq - 1.0
                consec_losses = consec_losses + 1 if net * scale < 0 else 0
                trades.append(
                    {
                        "entry_idx": open_pos["entry_idx"],
                        "direction": open_pos["direction"],
                        "hold_bars": open_pos["hold_bars"],
                        "exit_reason": "risk_force_exit",
                        "gross_return": unreal,
                        "net_return": net,
                        "scaled_return": net * scale,
                        "scale": scale,
                        "h8_pass": bool(open_pos["h8_pass"]),
                    }
                )
                risk_force += 1
                open_pos = None
            continue

        # same production pipeline
        if vol not in ("mid", "high"):
            continue
        strategy = "S2" if trend != "sideways" else "S1"
        if strategy == "S2" and _entropy(t) > 1.0:
            continue
        if sig is None:
            continue

        drawdown = (eq - peak) / peak if peak > 0 else 0.0
        if ks_triggered or daily_pnl <= MAX_DAILY_LOSS or drawdown <= MAX_DRAWDOWN or consec_losses >= MAX_CONSEC_LOSSES:
            ks_triggered = True
            continue

        h8 = _h8_pass(t)
        if variant.hard_block and (not h8):
            continue
        scale = 1.0 if h8 else float(variant.fail_scale)
        side = "BUY" if sig == "LONG" else "SELL"
        open_pos = {
            "entry_idx": i,
            "entry_price": px,
            "direction": "LONG" if side == "BUY" else "SHORT",
            "side": side,
            "hold_bars": 0,
            "scale": scale,
            "h8_pass": h8,
        }

    tdf = pd.DataFrame(trades)
    vals = tdf["scaled_return"].tolist() if not tdf.empty else []
    wins = [x for x in vals if x > 0]
    losses = [x for x in vals if x < 0]
    wr = len(wins) / len(vals) if vals else 0.0
    exp = _expectancy(vals)
    pf = (sum(wins) / abs(sum(losses))) if losses else 0.0
    mdd = 0.0
    eq2 = 1.0
    peak2 = 1.0
    for r in vals:
        eq2 *= 1.0 + r * position_size
        peak2 = max(peak2, eq2)
        mdd = min(mdd, (eq2 - peak2) / peak2 if peak2 > 0 else 0.0)

    metrics = {
        "trades": int(len(tdf)),
        "win_rate": wr,
        "expectancy": exp,
        "profit_factor": pf,
        "net_return": float(np.sum(vals)) if vals else 0.0,
        "MDD": float(mdd),
        "risk_force_exit_count": int((tdf["exit_reason"] == "risk_force_exit").sum()) if not tdf.empty else 0,
        "KS_triggered": int(bool(ks_triggered)),
        "avg_hold": float(tdf["hold_bars"].mean()) if not tdf.empty else 0.0,
        "h8_fail_trade_ratio": float((tdf["h8_pass"] == False).mean()) if not tdf.empty else 0.0,
    }
    return metrics, tdf


def _consistency(sub: pd.DataFrame, col: str) -> Dict[str, Any]:
    improved = int((sub[col] > 0).sum())
    degraded = int((sub[col] < 0).sum())
    arr = sub[col].tolist()
    avg_d = float(mean(arr)) if arr else 0.0
    std_d = float(pstdev(arr)) if len(arr) > 1 else 0.0
    return {
        "improved_windows": improved,
        "degraded_windows": degraded,
        "consistency_score": avg_d / (std_d + 1e-9),
    }


def _grade(row: pd.Series) -> str:
    if (
        row["avg_net_delta"] > 0
        and row["avg_mdd_delta"] >= 0
        and row["avg_ks_delta"] <= 0
        and row["avg_rfe_delta"] <= 0
        and row["trade_preservation_score"] >= 0.45
        and row["consistency_score"] > 0
    ):
        return "A"
    if row["avg_net_delta"] > 0 and row["risk_adjusted_score"] > 0:
        return "B"
    if row["risk_adjusted_score"] > -0.01:
        return "C"
    return "D"


def main() -> None:
    args = _parse_args()
    lookbacks = [int(x.strip()) for x in args.lookback_list.split(",") if x.strip()]
    variants = [
        Variant("Baseline", fail_scale=1.0, hard_block=False),
        Variant("Variant_G30", fail_scale=0.30, hard_block=False),
        Variant("Variant_G50", fail_scale=0.50, hard_block=False),
        Variant("Variant_G70", fail_scale=0.70, hard_block=False),
        Variant("Variant_G85", fail_scale=0.85, hard_block=False),
        Variant("Variant_Hard", fail_scale=0.0, hard_block=True),
    ]

    df = load_ohlcv()
    ticks, _ = simulate_signals_paper(df)

    detail_rows: List[Dict[str, Any]] = []
    for lb in lookbacks:
        wins = _window_endpoints(ticks, lb, args.stride_days, args.max_windows_per_lookback)
        for wid, (ws, we) in enumerate(wins):
            w_ticks = _slice_ticks_by_range(ticks, ws, we)
            if len(w_ticks) < 30:
                continue

            base_m, base_tdf = _simulate_variant(
                w_ticks,
                variants[0],
                args.position_size,
                args.fee_rate,
                args.slippage_rate,
            )

            base_trade_set = set(base_tdf["entry_idx"].astype(int).tolist()) if not base_tdf.empty else set()
            base_good = int((base_tdf["scaled_return"] > 0).sum()) if not base_tdf.empty else 0
            base_bad = int((base_tdf["scaled_return"] < 0).sum()) if not base_tdf.empty else 0
            base_loss_cluster = 0
            if not base_tdf.empty:
                for _, tr in base_tdf.iterrows():
                    i = int(tr["entry_idx"])
                    if _is_loss_cluster_trade(tr.to_dict(), w_ticks[i]):
                        base_loss_cluster += 1

            detail_rows.append(
                {
                    "lookback_days": lb,
                    "window_id": wid,
                    "window_start": ws,
                    "window_end": we,
                    "variant": "Baseline",
                    **base_m,
                    "trade_reduction_pct": 0.0,
                    "net_delta": 0.0,
                    "expectancy_delta": 0.0,
                    "pf_delta": 0.0,
                    "mdd_delta": 0.0,
                    "ks_delta": 0.0,
                    "rfe_delta": 0.0,
                    "loss_cluster_reduction": 0.0,
                    "trade_preservation_pct": 1.0,
                    "good_trade_preservation_pct": 1.0,
                    "bad_trade_removal_pct": 0.0,
                    "efficiency_net_per_1pct_removed": 0.0,
                    "risk_reduction_per_scale_reduction": 0.0,
                }
            )

            for v in variants[1:]:
                m, tdf = _simulate_variant(w_ticks, v, args.position_size, args.fee_rate, args.slippage_rate)
                trade_set = set(tdf["entry_idx"].astype(int).tolist()) if not tdf.empty else set()
                removed = base_trade_set - trade_set
                removed_df = base_tdf[base_tdf["entry_idx"].isin(removed)] if not base_tdf.empty else pd.DataFrame()
                good_removed = int((removed_df["scaled_return"] > 0).sum()) if not removed_df.empty else 0
                bad_removed = int((removed_df["scaled_return"] < 0).sum()) if not removed_df.empty else 0

                loss_cluster = 0
                if not tdf.empty:
                    for _, tr in tdf.iterrows():
                        i = int(tr["entry_idx"])
                        if _is_loss_cluster_trade(tr.to_dict(), w_ticks[i]):
                            loss_cluster += 1

                red_pct = (1.0 - m["trades"] / max(base_m["trades"], 1)) * 100.0
                scale_reduction = (1.0 - v.fail_scale) if (not v.hard_block) else 1.0
                rfe_red = max(0.0, (base_m["risk_force_exit_count"] - m["risk_force_exit_count"]) / max(base_m["risk_force_exit_count"], 1))

                detail_rows.append(
                    {
                        "lookback_days": lb,
                        "window_id": wid,
                        "window_start": ws,
                        "window_end": we,
                        "variant": v.name,
                        **m,
                        "trade_reduction_pct": red_pct,
                        "net_delta": m["net_return"] - base_m["net_return"],
                        "expectancy_delta": m["expectancy"] - base_m["expectancy"],
                        "pf_delta": m["profit_factor"] - base_m["profit_factor"],
                        "mdd_delta": m["MDD"] - base_m["MDD"],
                        "ks_delta": m["KS_triggered"] - base_m["KS_triggered"],
                        "rfe_delta": m["risk_force_exit_count"] - base_m["risk_force_exit_count"],
                        "loss_cluster_reduction": (base_loss_cluster - loss_cluster) / max(base_loss_cluster, 1),
                        "trade_preservation_pct": m["trades"] / max(base_m["trades"], 1),
                        "good_trade_preservation_pct": 1.0 - (good_removed / max(base_good, 1)),
                        "bad_trade_removal_pct": bad_removed / max(base_bad, 1),
                        "efficiency_net_per_1pct_removed": (m["net_return"] - base_m["net_return"]) / max(red_pct, 1e-9),
                        "risk_reduction_per_scale_reduction": rfe_red / max(scale_reduction, 1e-9),
                    }
                )

    detail = pd.DataFrame(detail_rows)
    sub = detail[detail["variant"] != "Baseline"].copy()

    summary_rows: List[Dict[str, Any]] = []
    for v, g in sub.groupby("variant"):
        c = _consistency(g, "net_delta")
        avg_net = float(g["net_delta"].mean())
        avg_exp = float(g["expectancy_delta"].mean())
        avg_pf = float(g["pf_delta"].mean())
        avg_mdd = float(g["mdd_delta"].mean())
        avg_ks = float(g["ks_delta"].mean())
        avg_rfe = float(g["rfe_delta"].mean())
        risk_adjusted = avg_net + 0.3 * avg_mdd - 0.15 * max(avg_ks, 0) - 0.15 * max(avg_rfe, 0)
        trade_pres_score = float(g["trade_preservation_pct"].mean())
        shadow_score = 0.4 * max(c["consistency_score"], 0) + 0.3 * max(avg_exp, 0) + 0.2 * max(avg_mdd, 0) + 0.1 * trade_pres_score
        paper_score = 0.3 * max(avg_net, 0) + 0.3 * max(avg_exp, 0) + 0.2 * max(avg_mdd, 0) + 0.2 * max(-avg_ks, 0)
        row = {
            "variant": v,
            "avg_trades": float(g["trades"].mean()),
            "avg_trade_reduction_pct": float(g["trade_reduction_pct"].mean()),
            "avg_net_delta": avg_net,
            "avg_expectancy_delta": avg_exp,
            "avg_pf_delta": avg_pf,
            "avg_mdd_delta": avg_mdd,
            "avg_ks_delta": avg_ks,
            "avg_rfe_delta": avg_rfe,
            "avg_loss_cluster_reduction": float(g["loss_cluster_reduction"].mean()),
            "avg_trade_preservation_pct": trade_pres_score,
            "avg_good_trade_preservation_pct": float(g["good_trade_preservation_pct"].mean()),
            "avg_bad_trade_removal_pct": float(g["bad_trade_removal_pct"].mean()),
            "avg_efficiency_net_per_1pct_removed": float(g["efficiency_net_per_1pct_removed"].mean()),
            "avg_risk_reduction_per_scale_reduction": float(g["risk_reduction_per_scale_reduction"].mean()),
            "shadow_score": shadow_score,
            "paper_score": paper_score,
            "risk_adjusted_score": risk_adjusted,
            "consistency_score": c["consistency_score"],
            "trade_preservation_score": trade_pres_score,
        }
        row["grade"] = _grade(pd.Series(row))
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["grade", "risk_adjusted_score", "paper_score", "shadow_score"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    best_soft = summary[summary["variant"] != "Variant_Hard"].iloc[0]
    best_overall = summary.iloc[0]
    hard = summary[summary["variant"] == "Variant_Hard"].iloc[0]
    hard_vs_soft = "soft" if float(best_soft["risk_adjusted_score"]) > float(hard["risk_adjusted_score"]) else "hard"
    optimal_scale = {
        "Variant_G30": 0.30,
        "Variant_G50": 0.50,
        "Variant_G70": 0.70,
        "Variant_G85": 0.85,
        "Variant_Hard": 0.0,
    }.get(str(best_soft["variant"]), 1.0)

    if hard_vs_soft == "soft" and float(best_soft["avg_trade_preservation_pct"]) >= 0.5:
        evo = "soft gate"
        next_step = "soft gate candidate"
    elif hard_vs_soft == "hard" and float(hard["grade"] in ("A", "B")):
        evo = "hard filter"
        next_step = "continue monitor"
    else:
        evo = "hybrid"
        next_step = "hybrid hard+soft"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"h8_soft_gate_{ts}.csv"
    md_path = OUT_DIR / f"h8_soft_gate_{ts}.md"

    out = pd.concat(
        [
            detail.assign(section="rolling_detail"),
            summary.assign(section="variant_summary"),
            pd.DataFrame(
                [
                    {
                        "best_soft_variant": best_soft["variant"],
                        "best_overall_policy": best_overall["variant"],
                        "hard_vs_soft_winner": hard_vs_soft,
                        "optimal_scale": optimal_scale,
                        "recommended_next_step": next_step,
                        "evolve_to": evo,
                    }
                ]
            ).assign(section="final_policy"),
        ],
        ignore_index=True,
        sort=False,
    )
    out.to_csv(csv_path, index=False)

    md = [
        "# H8 Soft Gate Validation",
        f"- best_soft_variant: {best_soft['variant']}",
        f"- best_overall_policy: {best_overall['variant']}",
        f"- hard_vs_soft_winner: {hard_vs_soft}",
        f"- optimal_scale: {optimal_scale}",
        f"- recommended_next_step: {next_step}",
        f"- evolve_to: {evo}",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    # print-only output block
    print(f"best soft gate variant: {best_soft['variant']}")
    print(f"whether hard or soft H8 is superior: {hard_vs_soft}")
    print(f"optimal scale for H8-fail trades: {optimal_scale:.2f}")
    print(
        "trade preservation vs risk reduction summary: "
        f"preservation={best_soft['avg_trade_preservation_pct']:.2%}, "
        f"bad_removal={best_soft['avg_bad_trade_removal_pct']:.2%}, "
        f"RFE_delta={best_soft['avg_rfe_delta']:+.4f}, "
        f"MDD_delta={best_soft['avg_mdd_delta']:+.6f}"
    )
    print(f"whether H8 should evolve into: {evo}")
    print(f"FINAL VERDICT: {best_overall['grade']}")
    print(
        "top 3 next actions: "
        "1) monitor best soft variant in replay-forward weekly; "
        "2) evaluate hybrid gate (soft scale + hard cap on extreme entropy); "
        "3) add minimum trade-floor constraint to avoid hidden no-trade drift"
    )


if __name__ == "__main__":
    main()

