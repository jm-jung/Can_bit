"""
Final hypothesis tournament (diagnostics only).

Compares H1–H13 entry-filter hypotheses vs baseline on rolling OOS.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.diagnostics.validate_entry_quality_filters import (
    MAX_CONSEC_LOSSES,
    MAX_DAILY_LOSS,
    MAX_DRAWDOWN,
    MAX_TRADE_LOSS,
    FilterSpec,
    _entropy,
    _margin,
    _prepare_dynamic_thresholds,
    _slice_ticks_by_range,
    _window_endpoints,
)
from scripts.run_daily_paper import PAPER_MAX_HOLDING_BARS, load_ohlcv, simulate_signals_paper


from pathlib import Path

OUT_DIR = Path("data/diagnostics/final_hypothesis")

EXPLANATORY_FORMULA = (
    "explanatory_power = 0.25*loss_cluster_removed_pct + 0.20*ks_reduction_pct + "
    "0.20*force_exit_reduction_pct + 0.25*expectancy_improvement_pct + 0.10*trade_preservation_pct"
)


@dataclass
class Hypothesis:
    id: str
    name: str
    description: str
    params: Dict[str, Any]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("validate_final_hypotheses")
    p.add_argument("--lookback-list", default="14,30,60,90,180")
    p.add_argument("--stride-days", type=int, default=7)
    p.add_argument("--max-windows-per-lookback", type=int, default=9999)
    p.add_argument("--position-size", type=float, default=0.05)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0002)
    return p.parse_args()


def _build_hypotheses() -> List[Hypothesis]:
    return [
        Hypothesis("H1", "entropy", "entropy<=0.90", {"entropy_ub": 0.90}),
        Hypothesis("H2", "trend", "trend_state==up", {"trend_up_only": True}),
        Hypothesis("H3", "trend", "trend_state!=down", {"exclude_down": True}),
        Hypothesis("H4", "vol", "vol!=high", {"exclude_high_vol": True}),
        Hypothesis("H5", "margin", "margin>=0.07", {"margin_floor": 0.07}),
        Hypothesis("H6", "direction", "LONG only", {"long_only": True}),
        Hypothesis("H7", "direction", "SHORT removed", {"long_only": True}),
        Hypothesis("H8", "combo", "entropy<=0.96 AND trend==up", {"entropy_ub": 0.96, "trend_up_only": True}),
        Hypothesis("H9", "combo", "entropy<=0.96 AND margin>=0.07", {"entropy_ub": 0.96, "margin_floor": 0.07}),
        Hypothesis("H10", "combo", "entropy<=0.96 AND trend!=down", {"entropy_ub": 0.96, "exclude_down": True}),
        Hypothesis("H11", "combo", "entropy<=0.96 AND trend==up AND margin>=0.07", {"entropy_ub": 0.96, "trend_up_only": True, "margin_floor": 0.07}),
        Hypothesis("H12", "combo", "entropy<=0.96 AND trend!=down AND margin>=0.07", {"entropy_ub": 0.96, "exclude_down": True, "margin_floor": 0.07}),
        Hypothesis("H13", "combo", "combo_F: entropy<=0.95 LONG trend!=down", {"entropy_ub": 0.95, "long_only": True, "exclude_long_down": True}),
    ]


def _allow_entry(params: Dict[str, Any], t: Dict[str, Any], signal: str, dyn: Dict[str, float]) -> bool:
    ent = _entropy(t)
    mar = _margin(t)
    vol = str(t.get("vol_bucket") or "")
    trend = str(t.get("trend_label") or "")

    if "entropy_ub" in params and ent > float(params["entropy_ub"]):
        return False
    if "margin_floor" in params and mar < float(params["margin_floor"]):
        return False
    if params.get("exclude_high_vol") and vol == "high":
        return False
    if params.get("long_only") and signal != "LONG":
        return False
    if params.get("exclude_long_down") and signal == "LONG" and trend == "down":
        return False
    if params.get("exclude_down") and trend == "down":
        return False
    if params.get("trend_up_only") and trend != "up":
        return False

    if "long_margin_top_pct" in params and signal == "LONG":
        if mar < dyn["margin_p70"]:
            return False
    if "entropy_bottom_pct" in params:
        if params["entropy_bottom_pct"] == 0.20 and ent > dyn["entropy_p20"]:
            return False
        if params["entropy_bottom_pct"] == 0.30 and ent > dyn["entropy_p30"]:
            return False
    if "margin_top_pct" in params:
        if params["margin_top_pct"] == 0.20 and mar < dyn["margin_p80"]:
            return False
        if params["margin_top_pct"] == 0.30 and mar < dyn["margin_p70"]:
            return False
    return True


def _simulate(
    ticks: List[Dict[str, Any]],
    params: Dict[str, Any],
    position_size: float,
    fee_rate: float,
    slippage_rate: float,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
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

    for i, t in enumerate(ticks):
        px = float(t["price"])
        signal = t.get("signal")
        vol = str(t.get("vol_bucket") or "")
        trend = str(t.get("trend_label") or "")

        exit_signal = False
        reason = ""
        if ks_triggered and open_pos is not None:
            exit_signal = True
            reason = "kill_switch_close"
        if open_pos is not None:
            open_pos["hold_bars"] += 1
            if signal is not None and (
                (open_pos["side"] == "BUY" and signal == "SHORT")
                or (open_pos["side"] == "SELL" and signal == "LONG")
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
                        "gross_return": unreal,
                        "net_return": net,
                    }
                )
                risk_force += 1
                open_pos = None
            continue

        if vol not in ("mid", "high"):
            continue
        strategy = "S2" if trend != "sideways" else "S1"
        if strategy == "S2" and _entropy(t) > 1.0:
            continue
        if signal is None:
            continue
        if not _allow_entry(params, t, signal, dyn):
            continue

        drawdown = (eq - peak) / peak if peak > 0 else 0.0
        if (
            ks_triggered
            or daily_pnl <= MAX_DAILY_LOSS
            or drawdown <= MAX_DRAWDOWN
            or consec_losses >= MAX_CONSEC_LOSSES
        ):
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

    tdf = pd.DataFrame(trades)
    vals = tdf["net_return"].tolist() if not tdf.empty else []
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
    for r in vals:
        eq2 *= 1.0 + r * position_size
        peak2 = max(peak2, eq2)
        mdd = min(mdd, (eq2 - peak2) / peak2 if peak2 > 0 else 0.0)

    metrics = {
        "total_trades": len(tdf),
        "win_rate": wr,
        "expectancy": expectancy,
        "profit_factor": pf,
        "net_return": float(np.sum(vals)) if vals else 0.0,
        "max_drawdown": mdd,
        "kill_switch_triggered": ks_triggered,
        "risk_force_exit_count": risk_force,
    }
    return metrics, tdf


def _is_loss_cluster_trade(trade: pd.Series, tick: Dict[str, Any]) -> bool:
    if str(trade.get("exit_reason")) == "risk_force_exit":
        return True
    ent = _entropy(tick)
    trend = str(tick.get("trend_label") or "")
    vol = str(tick.get("vol_bucket") or "")
    if (
        str(trade.get("direction")) == "LONG"
        and 0.90 <= ent <= 0.95
        and vol == "high"
        and trend == "down"
    ):
        return True
    return False


def _survivor_stats(base_trades: pd.DataFrame, filt_trades: pd.DataFrame) -> Dict[str, float]:
    if filt_trades.empty:
        return {"survivor_expectancy": 0.0, "survivor_pf": 0.0, "trade_reduction": 1.0}
    vals = filt_trades["net_return"].tolist()
    wins = [x for x in vals if x > 0]
    losses = [x for x in vals if x < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else 0.0
    exp = (
        (len(wins) / len(vals)) * mean(wins) - (1 - len(wins) / len(vals)) * abs(mean(losses))
        if losses and wins
        else (mean(wins) if wins else 0.0)
    )
    red = 1.0 - len(filt_trades) / max(len(base_trades), 1)
    return {"survivor_expectancy": float(exp), "survivor_pf": float(pf), "trade_reduction": float(red)}


def _loss_cluster_quality(
    base_trades: pd.DataFrame, filt_trades: pd.DataFrame, ticks: List[Dict[str, Any]]
) -> Dict[str, float]:
    if base_trades.empty:
        return {"loss_cluster_removed_pct": 0.0, "baseline_loss_cluster_count": 0.0}
    base_lc = 0
    removed_lc = 0
    surv_idx = set(filt_trades["entry_idx"].astype(int).tolist()) if not filt_trades.empty else set()
    for _, tr in base_trades.iterrows():
        i = int(tr["entry_idx"])
        tick = ticks[i]
        if _is_loss_cluster_trade(tr, tick):
            base_lc += 1
            if i not in surv_idx:
                removed_lc += 1
    pct = removed_lc / base_lc if base_lc > 0 else 0.0
    return {"loss_cluster_removed_pct": float(pct), "baseline_loss_cluster_count": float(base_lc)}


def _consistency(sub: pd.DataFrame) -> Dict[str, Any]:
    improved = int((sub["net_return_delta"] > 0).sum())
    degraded = int((sub["net_return_delta"] < 0).sum())
    arr = sub["net_return_delta"].tolist()
    avg_d = float(mean(arr)) if arr else 0.0
    std_d = float(pstdev(arr)) if len(arr) > 1 else 0.0
    score = avg_d / (std_d + 1e-9)
    return {
        "improved_windows": improved,
        "degraded_windows": degraded,
        "avg_net_delta": avg_d,
        "std_net_delta": std_d,
        "consistency_score": score,
    }


def _compute_explanatory(row: pd.Series, base_exp: float, base_ks: float, base_rfe: float) -> float:
    lc = float(row.get("loss_cluster_removed_pct", 0.0))
    ks_red = max(0.0, -float(row.get("ks_delta_avg", 0.0)) / max(base_ks, 1e-9))
    rfe_red = max(0.0, -float(row.get("force_exit_delta_avg", 0.0)) / max(base_rfe, 1e-9))
    exp_imp = max(0.0, float(row.get("avg_expectancy_delta", 0.0)) / max(abs(base_exp), 1e-9))
    trade_pres = max(0.0, 1.0 - float(row.get("trade_reduction_avg", 0.0)))
    ks_red = min(ks_red, 1.0)
    rfe_red = min(rfe_red, 1.0)
    exp_imp = min(exp_imp, 1.0)
    return 0.25 * lc + 0.20 * ks_red + 0.20 * rfe_red + 0.25 * exp_imp + 0.10 * trade_pres


def _tradeoff_score(row: pd.Series) -> float:
    """Higher = better balance of net improvement vs trade preservation."""
    net = float(row.get("avg_net_delta", 0.0))
    pres = 1.0 - float(row.get("trade_reduction_avg", 0.0))
    cons = float(row.get("consistency_score", 0.0))
    return net * pres * (1.0 + max(cons, 0.0))


def main() -> None:
    args = _parse_args()
    lookbacks = [int(x.strip()) for x in args.lookback_list.split(",") if x.strip()]
    hyps = _build_hypotheses()
    baseline = FilterSpec("baseline_current", "baseline", {})

    df = load_ohlcv()
    ticks, _ = simulate_signals_paper(df)

    detail_rows: List[Dict[str, Any]] = []
    quality_rows: List[Dict[str, Any]] = []
    total_jobs = 0
    done_jobs = 0

    for lb in lookbacks:
        wins = _window_endpoints(ticks, lb, args.stride_days, args.max_windows_per_lookback)
        total_jobs += len(wins) * (len(hyps) + 1)
        print(f"[lookback={lb}d] windows={len(wins)}", flush=True)

    for lb in lookbacks:
        wins = _window_endpoints(ticks, lb, args.stride_days, args.max_windows_per_lookback)
        for wid, (ws, we) in enumerate(wins):
            w_ticks = _slice_ticks_by_range(ticks, ws, we)
            if len(w_ticks) < 30:
                continue

            base_m, base_tdf = _simulate(
                w_ticks, baseline.params, args.position_size, args.fee_rate, args.slippage_rate
            )
            base_m["hypothesis_id"] = "baseline"
            base_m["lookback_days"] = lb
            base_m["window_id"] = wid
            detail_rows.append({**base_m, "net_return_delta": 0.0, "expectancy_delta": 0.0, "PF_delta": 0.0, "MDD_delta": 0.0, "ks_delta": 0, "force_exit_delta": 0})

            for h in hyps:
                m, tdf = _simulate(
                    w_ticks, h.params, args.position_size, args.fee_rate, args.slippage_rate
                )
                surv = _survivor_stats(base_tdf, tdf)
                lc = _loss_cluster_quality(base_tdf, tdf, w_ticks)
                rec = {
                    "hypothesis_id": h.id,
                    "hypothesis_name": h.name,
                    "description": h.description,
                    "lookback_days": lb,
                    "window_id": wid,
                    "window_start": ws,
                    "window_end": we,
                    **m,
                    **surv,
                    **lc,
                    "net_return_delta": m["net_return"] - base_m["net_return"],
                    "expectancy_delta": m["expectancy"] - base_m["expectancy"],
                    "PF_delta": m["profit_factor"] - base_m["profit_factor"],
                    "MDD_delta": m["max_drawdown"] - base_m["max_drawdown"],
                    "ks_delta": int(m["kill_switch_triggered"]) - int(base_m["kill_switch_triggered"]),
                    "force_exit_delta": m["risk_force_exit_count"] - base_m["risk_force_exit_count"],
                }
                detail_rows.append(rec)
                quality_rows.append(rec)

            done_jobs += len(hyps) + 1
            if done_jobs % 50 == 0 or wid == len(wins) - 1:
                print(f"  progress {done_jobs}/{total_jobs} (lb={lb} wid={wid})", flush=True)

    detail = pd.DataFrame(detail_rows)
    base_sub = detail[detail["hypothesis_id"] == "baseline"]
    base_exp = float(base_sub["expectancy"].mean()) if len(base_sub) else 1e-9
    base_ks = float(base_sub["kill_switch_triggered"].mean()) if len(base_sub) else 1e-9
    base_rfe = float(base_sub["risk_force_exit_count"].mean()) if len(base_sub) else 1e-9

    summary_rows: List[Dict[str, Any]] = []
    for h in hyps:
        sub = detail[detail["hypothesis_id"] == h.id]
        if sub.empty:
            continue
        cons = _consistency(sub)
        summary_rows.append(
            {
                "rank": 0,
                "hypothesis_id": h.id,
                "hypothesis_name": h.name,
                "description": h.description,
                "avg_trades": float(sub["total_trades"].mean()),
                "trade_reduction_avg": float(sub["trade_reduction"].mean()),
                "avg_expectancy": float(sub["expectancy"].mean()),
                "avg_profit_factor": float(sub["profit_factor"].mean()),
                "avg_net_return": float(sub["net_return"].mean()),
                "avg_max_drawdown": float(sub["max_drawdown"].mean()),
                "avg_win_rate": float(sub["win_rate"].mean()),
                "survivor_expectancy_avg": float(sub["survivor_expectancy"].mean()),
                "survivor_pf_avg": float(sub["survivor_pf"].mean()),
                "ks_delta_avg": float(sub["ks_delta"].mean()),
                "force_exit_delta_avg": float(sub["force_exit_delta"].mean()),
                "avg_expectancy_delta": float(sub["expectancy_delta"].mean()),
                "avg_pf_delta": float(sub["PF_delta"].mean()),
                "avg_mdd_delta": float(sub["MDD_delta"].mean()),
                "loss_cluster_removed_pct": float(sub["loss_cluster_removed_pct"].mean()),
                **cons,
                "over_filtered": bool(float(sub["trade_reduction"].mean()) > 0.80),
            }
        )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        print("No hypothesis results.")
        return

    summary["explanatory_power"] = summary.apply(
        lambda r: _compute_explanatory(r, base_exp, base_ks, base_rfe), axis=1
    )
    summary["tradeoff_score"] = summary.apply(_tradeoff_score, axis=1)
    summary["final_score"] = (
        0.35 * summary["explanatory_power"]
        + 0.25 * summary["consistency_score"].clip(lower=0)
        + 0.25 * summary["avg_net_delta"].clip(lower=0)
        + 0.15 * (1.0 - summary["trade_reduction_avg"])
    )
    summary = summary.sort_values("final_score", ascending=False).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)

    # Factor ranking for Q1 (single-axis hypotheses); trend = max(H2, H3)
    trend_score = max(
        float(summary[summary["hypothesis_id"] == "H2"]["explanatory_power"].iloc[0]) if len(summary[summary["hypothesis_id"] == "H2"]) else 0,
        float(summary[summary["hypothesis_id"] == "H3"]["explanatory_power"].iloc[0]) if len(summary[summary["hypothesis_id"] == "H3"]) else 0,
    )
    q1_ranking = sorted(
        [
            ("entropy", float(summary[summary["hypothesis_id"] == "H1"]["explanatory_power"].iloc[0]) if len(summary[summary["hypothesis_id"] == "H1"]) else 0),
            ("trend", trend_score),
            ("margin", float(summary[summary["hypothesis_id"] == "H5"]["explanatory_power"].iloc[0]) if len(summary[summary["hypothesis_id"] == "H5"]) else 0),
            ("vol", float(summary[summary["hypothesis_id"] == "H4"]["explanatory_power"].iloc[0]) if len(summary[summary["hypothesis_id"] == "H4"]) else 0),
            ("direction", float(summary[summary["hypothesis_id"] == "H6"]["explanatory_power"].iloc[0]) if len(summary[summary["hypothesis_id"] == "H6"]) else 0),
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    winner = summary.iloc[0]
    runner = summary.iloc[1] if len(summary) > 1 else winner
    worst = summary.iloc[-1]

    deploy_ok = summary[(~summary["over_filtered"]) & (summary["avg_net_delta"] > 0)].copy()
    deploy_ok = deploy_ok.sort_values("final_score", ascending=False)
    deploy_no = summary[summary["over_filtered"] | (summary["avg_net_delta"] <= 0)].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"final_hypothesis_tournament_{ts}.csv"
    md_path = OUT_DIR / f"final_hypothesis_tournament_{ts}.md"

    out_parts = [
        detail.assign(section="rolling_detail"),
        summary.assign(section="hypothesis_summary"),
    ]
    pd.concat(out_parts, ignore_index=True, sort=False).to_csv(csv_path, index=False)

    md: List[str] = []
    md.append("# Final Hypothesis Tournament")
    md.append("")
    md.append(f"- windows: {detail[['lookback_days','window_id']].drop_duplicates().shape[0]}")
    md.append(f"- baseline avg expectancy: {base_exp:.6f}")
    md.append(f"- explanatory formula: `{EXPLANATORY_FORMULA}`")
    md.append("")
    md.append("## Ranking (1–13)")
    for _, r in summary.iterrows():
        md.append(
            f"{int(r['rank'])}. {r['hypothesis_id']} {r['description']}: "
            f"final_score={r['final_score']:.4f}, expΔ={r['avg_expectancy_delta']:+.6f}, "
            f"netΔ={r['avg_net_delta']:+.6f}, over_filtered={r['over_filtered']}"
        )
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    best_exp = summary.sort_values("avg_expectancy", ascending=False).iloc[0]
    best_pf = summary.sort_values("avg_profit_factor", ascending=False).iloc[0]
    lowest_red = summary.sort_values("trade_reduction_avg", ascending=True).iloc[0]
    most_over = summary.sort_values("trade_reduction_avg", ascending=False).iloc[0]
    best_expl = summary.sort_values("explanatory_power", ascending=False).iloc[0]
    best_tradeoff = summary.sort_values("tradeoff_score", ascending=False).iloc[0]

    shadow_candidate = deploy_ok.iloc[0] if len(deploy_ok) else best_tradeoff

    print("[FINAL HYPOTHESIS TOURNAMENT]")
    print("")
    print(f"explanatory_power_formula:\n- {EXPLANATORY_FORMULA}")
    print(f"\n1. best_hypothesis:\n- {winner['hypothesis_id']} ({winner['description']})")
    print(f"\n2. runner_up:\n- {runner['hypothesis_id']} ({runner['description']})")
    print(f"\n3. worst_hypothesis:\n- {worst['hypothesis_id']} ({worst['description']})")
    print(f"\n4. highest_explanatory_power:\n- {best_expl['hypothesis_id']} (score={best_expl['explanatory_power']:.4f})")
    print(f"\n5. best_tradeoff:\n- {best_tradeoff['hypothesis_id']} (score={best_tradeoff['tradeoff_score']:.6f})")
    print(f"\n6. best_expectancy:\n- {best_exp['hypothesis_id']} (exp={best_exp['avg_expectancy']:.6f})")
    print(f"\n7. best_pf:\n- {best_pf['hypothesis_id']} (PF={best_pf['avg_profit_factor']:.4f})")
    print(f"\n8. lowest_trade_reduction:\n- {lowest_red['hypothesis_id']} (reduction={lowest_red['trade_reduction_avg']:.2%})")
    print(f"\n9. most_overfiltered:\n- {most_over['hypothesis_id']} (reduction={most_over['trade_reduction_avg']:.2%})")
    print(f"\n10. FINAL WINNER:\n- {winner['hypothesis_id']}")
    print(f"\n11. deploy_ok:\n- {', '.join(deploy_ok['hypothesis_id'].tolist()[:5]) or 'none'}")
    print(f"\n12. deploy_no:\n- {', '.join(deploy_no['hypothesis_id'].tolist()[:5]) or 'none'}")
    print(f"\n13. FINAL VERDICT:\n- primary_leak={q1_ranking[0][0]}; winner={winner['hypothesis_id']}")
    print("")
    print("Q1 factor ranking (explanatory_power):")
    for i, (fac, sc) in enumerate(q1_ranking, 1):
        print(f"  {i}. {fac} ({sc:.4f})")
    print(f"\nQ2 single largest cause:\n- {q1_ranking[0][0]}")
    print(f"\nQ3 shadow/paper candidate:\n- {shadow_candidate['hypothesis_id']} ({shadow_candidate['description']})")
    print("")
    print("FINAL WINNER:")
    print(f"{winner['hypothesis_id']} — {winner['description']}")
    print("")
    print("WHY:")
    print(
        f"final_score={winner['final_score']:.4f}, netΔ={winner['avg_net_delta']:+.6f}, "
        f"expΔ={winner['avg_expectancy_delta']:+.6f}, explanatory={winner['explanatory_power']:.4f}, "
        f"consistency={winner['consistency_score']:.4f}, over_filtered={winner['over_filtered']}"
    )
    print("")
    print("SHADOW/PAPER CANDIDATE:")
    print(f"{shadow_candidate['hypothesis_id']} — {shadow_candidate['description']}")
    print("")
    print("DO NOT DEPLOY:")
    over_list = deploy_no[deploy_no["over_filtered"]]["hypothesis_id"].tolist()
    print(", ".join(over_list[:6]) if over_list else "none marked over_filtered")
    print(f"\ncreated:\n- {csv_path}\n- {md_path}")


if __name__ == "__main__":
    main()
