"""
Can_bit risk_force_exit forward replay validation (diagnostics only).

Usage:
  python -m scripts.diagnostics.validate_risk_force_exit_forward_replay --lookback-list 14,30,60,90
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional

import pandas as pd

from scripts.run_daily_paper import PAPER_MAX_HOLDING_BARS, load_ohlcv, simulate_signals_paper


OUT_DIR = Path("data/diagnostics/lifecycle")
MAX_TRADE_LOSS = -0.01
MAX_DAILY_LOSS = -0.02
MAX_DRAWDOWN = -0.05
MAX_CONSEC_LOSSES = 5


@dataclass
class VariantConfig:
    name: str
    disable_risk: bool = False
    grace_bars: int = 0
    confirm_n: int = 1
    hybrid: bool = False


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("validate_risk_force_exit_forward_replay")
    p.add_argument("--lookback-list", default="14,30,60,90")
    p.add_argument("--position-size", type=float, default=0.05)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0002)
    return p.parse_args()


def _net_from_raw(raw: float, fee_rate: float, slippage_rate: float) -> float:
    return float(raw) - 2.0 * (fee_rate + slippage_rate)


def _unrealized(side: str, entry: float, current: float) -> float:
    if side == "BUY":
        return (current - entry) / entry
    return (entry - current) / entry


def _raw_return(side: str, entry: float, exit_price: float) -> float:
    return _unrealized(side, entry, exit_price)


def _opposite_signal(pos_side: str, tick_signal: Optional[str]) -> bool:
    if tick_signal is None:
        return False
    return (pos_side == "BUY" and tick_signal == "SHORT") or (pos_side == "SELL" and tick_signal == "LONG")


def _build_variants() -> List[VariantConfig]:
    return [
        VariantConfig("current"),
        VariantConfig("disable_risk_force_exit", disable_risk=True),
        VariantConfig("grace_1_bar", grace_bars=1),
        VariantConfig("grace_3_bars", grace_bars=3),
        VariantConfig("confirm_2_consecutive", confirm_n=2),
        VariantConfig("confirm_3_consecutive", confirm_n=3),
        VariantConfig("hybrid_grace3_confirm2", grace_bars=3, confirm_n=2, hybrid=True),
    ]


def _slice_ticks_by_lookback(ticks: List[Dict[str, Any]], days: int) -> List[Dict[str, Any]]:
    ts_vals = [pd.Timestamp(t["timestamp"]) for t in ticks if t.get("timestamp")]
    if not ts_vals:
        return ticks
    tmax = max(ts_vals)
    cutoff = tmax - pd.Timedelta(days=days)
    out = []
    for t in ticks:
        ts = t.get("timestamp")
        if ts and pd.Timestamp(ts) >= cutoff:
            out.append(t)
    return out


def _simulate_variant(
    ticks: List[Dict[str, Any]],
    variant: VariantConfig,
    position_size: float,
    fee_rate: float,
    slippage_rate: float,
) -> Dict[str, Any]:
    equity = 1.0
    peak = 1.0
    daily_pnl = 0.0
    consecutive_losses = 0
    kill_switch_active = False
    kill_switch_trade_idx = -1

    open_pos: Optional[Dict[str, Any]] = None
    hold_bars = 0
    trades: List[Dict[str, Any]] = []
    risk_force_exit_count = 0
    exit_signal_count = 0
    enter_count = 0

    for i, t in enumerate(ticks):
        price = float(t["price"])
        signal = t.get("signal")
        vol_bucket = t.get("vol_bucket")
        trend_label = t.get("trend_label")
        p_long, p_short, p_flat = float(t["p_long"]), float(t["p_short"]), float(t["p_flat"])
        entropy = 0.0
        for p in (p_long, p_short, p_flat):
            if p > 1e-10:
                entropy -= p * math.log(p)

        # ----- exit priority path (same structure as run_daily_paper/trading_engine) -----
        pre_exit = ""
        exit_signal = False
        if kill_switch_active and open_pos is not None:
            exit_signal = True
            pre_exit = "kill_switch_close"
        if open_pos is not None:
            hold_bars += 1
            if _opposite_signal(open_pos["side"], signal):
                exit_signal = True
                pre_exit = pre_exit or "opposite_signal"
            elif hold_bars >= PAPER_MAX_HOLDING_BARS:
                exit_signal = True
                pre_exit = pre_exit or "max_holding_bars"

        if exit_signal and open_pos is not None:
            raw = _raw_return(open_pos["side"], open_pos["entry_price"], price)
            net = _net_from_raw(raw, fee_rate, slippage_rate)
            equity *= 1.0 + net * position_size
            daily_pnl = equity - 1.0
            peak = max(peak, equity)
            consecutive_losses = consecutive_losses + 1 if net < 0 else 0
            trades.append(
                {
                    "net_return": net,
                    "raw_return": raw,
                    "hold_bars": hold_bars,
                    "exit_reason": pre_exit or "exit_signal",
                }
            )
            exit_signal_count += 1
            open_pos = None
            hold_bars = 0
            continue

        # risk exit (variant logic)
        if open_pos is not None:
            u = _unrealized(open_pos["side"], open_pos["entry_price"], price)
            risk_cond = u <= MAX_TRADE_LOSS
            if "risk_consec" not in open_pos:
                open_pos["risk_consec"] = 0
                open_pos["grace_left"] = 0
            if risk_cond:
                open_pos["risk_consec"] += 1
                if open_pos["grace_left"] > 0:
                    open_pos["grace_left"] -= 1
                elif open_pos["grace_left"] == 0 and variant.grace_bars > 0:
                    open_pos["grace_left"] = variant.grace_bars
            else:
                open_pos["risk_consec"] = 0
                open_pos["grace_left"] = 0

            do_risk_exit = False
            if risk_cond and not variant.disable_risk:
                if variant.hybrid:
                    # grace 3 + confirm 2
                    if open_pos["grace_left"] <= 0 and open_pos["risk_consec"] >= 2:
                        do_risk_exit = True
                else:
                    grace_ok = open_pos["grace_left"] <= 0
                    confirm_ok = open_pos["risk_consec"] >= max(1, variant.confirm_n)
                    if grace_ok and confirm_ok:
                        do_risk_exit = True

            if do_risk_exit:
                raw = _raw_return(open_pos["side"], open_pos["entry_price"], price)
                net = _net_from_raw(raw, fee_rate, slippage_rate)
                equity *= 1.0 + net * position_size
                daily_pnl = equity - 1.0
                peak = max(peak, equity)
                consecutive_losses = consecutive_losses + 1 if net < 0 else 0
                trades.append(
                    {
                        "net_return": net,
                        "raw_return": raw,
                        "hold_bars": hold_bars,
                        "exit_reason": "risk_force_exit",
                    }
                )
                risk_force_exit_count += 1
                open_pos = None
                hold_bars = 0
                continue

            # holding
            continue

        # ----- no-position pipeline -----
        if vol_bucket not in ("mid", "high"):
            continue
        strategy = "S2" if trend_label != "sideways" else "S1"
        if strategy == "S2" and entropy > 1.0:
            continue
        if signal is None:
            continue

        # pre-entry risk check
        drawdown = (equity - peak) / peak if peak > 0 else 0.0
        stop = (
            kill_switch_active
            or daily_pnl <= MAX_DAILY_LOSS
            or drawdown <= MAX_DRAWDOWN
            or consecutive_losses >= MAX_CONSEC_LOSSES
        )
        if stop:
            kill_switch_active = True
            if kill_switch_trade_idx < 0:
                kill_switch_trade_idx = len(trades) - 1
            continue

        side = "BUY" if signal == "LONG" else "SELL"
        open_pos = {"side": side, "entry_price": price}
        hold_bars = 0
        enter_count += 1

    returns = [float(t["net_return"]) for t in trades]
    gross = [float(t["raw_return"]) for t in trades]
    wins = [x for x in returns if x > 0]
    losses = [x for x in returns if x < 0]
    wr = len(wins) / len(returns) if returns else 0.0
    avg_win = mean(wins) if wins else 0.0
    avg_loss = mean(losses) if losses else 0.0
    expectancy = (wr * avg_win) - ((1.0 - wr) * abs(avg_loss)) if returns else 0.0
    pf = (sum(wins) / abs(sum(losses))) if losses else 0.0
    mdd = 0.0
    eq = 1.0
    peak2 = 1.0
    max_consec = 0
    consec = 0
    for r in returns:
        eq *= 1.0 + r * position_size
        peak2 = max(peak2, eq)
        mdd = min(mdd, (eq - peak2) / peak2 if peak2 > 0 else 0.0)
        if r < 0:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0

    return {
        "total_trades": len(trades),
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
        "kill_switch_triggered": bool(kill_switch_trade_idx >= 0),
        "risk_force_exit_count": risk_force_exit_count,
        "exit_signal_count": exit_signal_count,
        "avg_hold_bars": mean([t["hold_bars"] for t in trades]) if trades else 0.0,
        "daily_trade_density": (len(trades) / max(1.0, (len(ticks) / 288.0))),  # 5m bars -> ~288/day
        "fee_cost_sum": len(trades) * 2.0 * fee_rate,
        "slippage_cost_sum": len(trades) * 2.0 * slippage_rate,
        "kill_switch_trade_index": kill_switch_trade_idx,
    }


def _grade_variant(rows: pd.DataFrame) -> str:
    improved_periods = int(rows["improved"].sum())
    avg_exp = float(rows["expectancy_delta"].mean())
    avg_net = float(rows["net_return_delta"].mean())
    avg_mdd = float(rows["MDD_delta"].mean())
    ks_ok = bool((rows["KS_delta"] <= 0).all())
    pf_ok = bool((rows["PF_delta"].mean() > 0))
    if improved_periods >= 3 and avg_exp > 0 and avg_net > 0 and avg_mdd >= 0 and ks_ok and pf_ok:
        return "A"
    if improved_periods >= 2 and avg_net > 0:
        return "B"
    return "C"


def main() -> None:
    args = _parse_args()
    lookbacks = [int(x.strip()) for x in args.lookback_list.split(",") if x.strip()]
    variants = _build_variants()

    df = load_ohlcv()
    ticks, _ = simulate_signals_paper(df)

    rows: List[Dict[str, Any]] = []
    for lb in lookbacks:
        lb_ticks = _slice_ticks_by_lookback(ticks, lb)
        by_case: Dict[str, Dict[str, Any]] = {}
        for v in variants:
            res = _simulate_variant(
                lb_ticks,
                v,
                position_size=args.position_size,
                fee_rate=args.fee_rate,
                slippage_rate=args.slippage_rate,
            )
            by_case[v.name] = res
            rows.append({"lookback_days": lb, "case_name": v.name, **res})

        base = by_case["current"]
        for r in rows:
            if r["lookback_days"] != lb:
                continue
            r["expectancy_delta"] = r["expectancy"] - base["expectancy"]
            r["net_return_delta"] = r["net_return_sum"] - base["net_return_sum"]
            r["equity_return_delta"] = r["equity_return"] - base["equity_return"]
            r["MDD_delta"] = r["max_drawdown"] - base["max_drawdown"]
            r["KS_avoided"] = base["kill_switch_triggered"] and (not r["kill_switch_triggered"])
            r["PF_delta"] = r["profit_factor"] - base["profit_factor"]
            r["risk_force_exit_reduction"] = base["risk_force_exit_count"] - r["risk_force_exit_count"]
            r["KS_delta"] = int(r["kill_switch_triggered"]) - int(base["kill_switch_triggered"])
            r["improved"] = (
                (r["case_name"] != "current")
                and (r["expectancy_delta"] > 0)
                and (r["net_return_delta"] > 0)
                and (r["MDD_delta"] >= 0)
                and (r["PF_delta"] > 0)
                and (r["KS_delta"] <= 0)
            )

    out = pd.DataFrame(rows).sort_values(["lookback_days", "case_name"]).reset_index(drop=True)

    # stability summary per variant
    summary_rows = []
    for v in [x.name for x in variants if x.name != "current"]:
        sub = out[out["case_name"] == v].copy()
        grade = _grade_variant(sub)
        summary_rows.append(
            {
                "variant": v,
                "improved_periods": int(sub["improved"].sum()),
                "avg_expectancy_delta": float(sub["expectancy_delta"].mean()),
                "avg_net_delta": float(sub["net_return_delta"].mean()),
                "avg_MDD_delta": float(sub["MDD_delta"].mean()),
                "KS_reduction": int(sub["KS_avoided"].sum()),
                "verdict": grade,
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["verdict", "improved_periods", "avg_expectancy_delta", "avg_net_delta"],
        ascending=[True, False, False, False],
    )
    best = summary_df.iloc[0]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"risk_force_exit_forward_replay_{ts}.csv"
    md_path = OUT_DIR / f"risk_force_exit_forward_replay_{ts}.md"
    out.to_csv(csv_path, index=False)

    # markdown
    md: List[str] = []
    md.append("# Can_bit Risk Force Exit Forward Replay Report")
    md.append("")
    md.append("## 1. 목적")
    md.append("- 14/30/60/90일 구간에서 risk_force_exit 완화 variant의 일관성 검증")
    md.append("")
    md.append("## 2. 실험 조건")
    md.append(f"- lookback_list: {lookbacks}")
    md.append(f"- position_size: {args.position_size}")
    md.append(f"- fee_rate: {args.fee_rate}")
    md.append(f"- slippage_rate: {args.slippage_rate}")
    md.append("- pipeline: activation -> rule_C_trend -> entropy<=1.0 -> paper lifecycle")
    md.append("")
    md.append("## 3. current 결과")
    for lb in lookbacks:
        r = out[(out["lookback_days"] == lb) & (out["case_name"] == "current")].iloc[0]
        md.append(
            f"- {lb}d: trades={int(r['total_trades'])}, exp={r['expectancy']:.6f}, net={r['net_return_sum']:.6f}, "
            f"PF={r['profit_factor']:.4f}, MDD={r['max_drawdown']:.6f}, KS={bool(r['kill_switch_triggered'])}"
        )
    md.append("")
    md.append("## 4. variant별 결과표")
    for _, r in out.iterrows():
        if r["case_name"] == "current":
            continue
        md.append(
            f"- {int(r['lookback_days'])}d / {r['case_name']}: expΔ={r['expectancy_delta']:+.6f}, "
            f"netΔ={r['net_return_delta']:+.6f}, PFΔ={r['PF_delta']:+.4f}, MDDΔ={r['MDD_delta']:+.6f}, "
            f"KS_avoided={bool(r['KS_avoided'])}"
        )
    md.append("")
    md.append("## 5. 기간별 winner")
    for lb in lookbacks:
        sub = out[(out["lookback_days"] == lb) & (out["case_name"] != "current")].sort_values(
            ["expectancy_delta", "net_return_delta"], ascending=[False, False]
        )
        if len(sub):
            w = sub.iloc[0]
            md.append(
                f"- {lb}d winner: {w['case_name']} (expΔ={w['expectancy_delta']:+.6f}, netΔ={w['net_return_delta']:+.6f})"
            )
    md.append("")
    md.append("## 6. consistency summary")
    for _, r in summary_df.iterrows():
        md.append(
            f"- {r['variant']}: improved_periods={int(r['improved_periods'])}, avg_expΔ={r['avg_expectancy_delta']:+.6f}, "
            f"avg_netΔ={r['avg_net_delta']:+.6f}, avg_MDDΔ={r['avg_MDD_delta']:+.6f}, KS_reduction={int(r['KS_reduction'])}, verdict={r['verdict']}"
        )
    md.append("")
    md.append("## 7. risk analysis")
    md.append("- MDD_delta < 0 인 구간은 drawdown 악화로 해석")
    md.append("- KS_avoided가 많을수록 연속손실 클러스터 완화 가능성")
    md.append("")
    md.append("## 8. 최종 후보")
    md.append(
        f"- {best['variant']} (verdict={best['verdict']}, improved_periods={int(best['improved_periods'])}, "
        f"avg_expΔ={best['avg_expectancy_delta']:+.6f}, avg_netΔ={best['avg_net_delta']:+.6f})"
    )
    md.append("")
    md.append("## 9. 운영 적용 가능 여부")
    md.append("- 본 리포트는 diagnostics 결과이며 운영 기본값 변경 없음")
    md.append("- verdict A일 때도 forward OOS 재검증 후 적용 검토")
    md.append("")
    md.append("## 10. 다음 단계")
    md.append("- scripts/diagnostics/validate_risk_force_exit_forward_replay.py를 기간 확장(14/30/60/90 rolling)으로 고도화")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("[RISK FORCE EXIT FORWARD REPLAY SUMMARY]")
    print("")
    print("best_variant:")
    print(f"- name: {best['variant']}")
    print(f"- improved_periods: {int(best['improved_periods'])}")
    print(f"- avg_expectancy_delta: {best['avg_expectancy_delta']:+.6f}")
    print(f"- avg_net_delta: {best['avg_net_delta']:+.6f}")
    print(f"- avg_MDD_delta: {best['avg_MDD_delta']:+.6f}")
    print(f"- KS_reduction: {int(best['KS_reduction'])}")
    print(f"- verdict: {best['verdict']}")
    print("")
    print("created:")
    print(f"- csv: {csv_path}")
    print(f"- markdown: {md_path}")


if __name__ == "__main__":
    main()
