"""
Validate risk_force_exit ablation variants (analysis-only).

Outputs:
  - data/diagnostics/lifecycle/risk_force_exit_ablation_<timestamp>.csv
  - data/diagnostics/lifecycle/risk_force_exit_ablation_<timestamp>.md
"""

from __future__ import annotations

from datetime import datetime
from statistics import mean, median
from typing import Any, Dict, List, Optional

import pandas as pd

from scripts.diagnostics.lifecycle_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_POSITION_SIZE,
    DEFAULT_SLIPPAGE_RATE,
    KS_STREAK_THRESHOLD,
    align_to_state_window,
    build_equity_curve,
    ensure_out_dir,
    load_tick_events,
    max_drawdown_from_curve,
    reconstruct_actual_trades,
)


def _direction_from_reason(reason: str) -> str:
    return "SHORT" if str(reason).startswith("SHORT") else "LONG"


def _raw_return(direction: str, entry_price: float, exit_price: float) -> float:
    if direction == "LONG":
        return (exit_price - entry_price) / entry_price
    return (entry_price - exit_price) / entry_price


def _calc_metrics(case_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "case_name": case_name,
            "trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "median_return": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "gross_return_sum": 0.0,
            "net_return_sum": 0.0,
            "estimated_fee_cost": 0.0,
            "estimated_slippage_cost": 0.0,
            "max_consecutive_losses": 0,
            "kill_switch_triggered": False,
            "kill_switch_trade_index": -1,
            "max_drawdown_proxy": 0.0,
            "equity_return_proxy": 0.0,
            "risk_force_exit_count": 0,
            "exit_signal_count": 0,
            "avg_hold_bars": 0.0,
            "avg_hold_bars_winners": 0.0,
            "avg_hold_bars_losers": 0.0,
        }

    returns = df["net_return"].astype(float).tolist()
    gross = df["raw_return"].astype(float).tolist()
    wins = [x for x in returns if x > 0]
    losses = [x for x in returns if x < 0]
    wr = len(wins) / len(returns)
    avg_win = mean(wins) if wins else 0.0
    avg_loss = mean(losses) if losses else 0.0
    expectancy = (wr * avg_win) - ((1.0 - wr) * abs(avg_loss))
    gross_profit = sum(x for x in returns if x > 0)
    gross_loss_abs = abs(sum(x for x in returns if x < 0))
    pf = gross_profit / gross_loss_abs if gross_loss_abs > 0 else 0.0

    # KS proxy
    consec = 0
    max_consec = 0
    ks_idx = -1
    for i, r in enumerate(returns):
        if r < 0:
            consec += 1
            max_consec = max(max_consec, consec)
            if ks_idx < 0 and consec >= KS_STREAK_THRESHOLD:
                ks_idx = i
        else:
            consec = 0

    curve = build_equity_curve(returns, position_size=DEFAULT_POSITION_SIZE, initial_equity=1.0)
    mdd = max_drawdown_from_curve(curve)
    eq_ret = (curve[-1] - 1.0) if curve else 0.0

    winners = df[df["net_return"] > 0]
    losers = df[df["net_return"] < 0]

    return {
        "case_name": case_name,
        "trades": int(len(df)),
        "win_rate": float(wr),
        "avg_return": float(mean(returns)),
        "median_return": float(median(returns)),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "expectancy": float(expectancy),
        "profit_factor": float(pf),
        "gross_return_sum": float(sum(gross)),
        "net_return_sum": float(sum(returns)),
        "estimated_fee_cost": float(len(df) * 2.0 * DEFAULT_FEE_RATE),
        "estimated_slippage_cost": float(len(df) * 2.0 * DEFAULT_SLIPPAGE_RATE),
        "max_consecutive_losses": int(max_consec),
        "kill_switch_triggered": bool(ks_idx >= 0),
        "kill_switch_trade_index": int(ks_idx),
        "max_drawdown_proxy": float(mdd),
        "equity_return_proxy": float(eq_ret),
        "risk_force_exit_count": int((df["exit_reason"] == "risk_force_exit").sum()),
        "exit_signal_count": int((df["exit_reason"] == "exit_signal").sum()),
        "avg_hold_bars": float(df["hold_bars"].mean()),
        "avg_hold_bars_winners": float(winners["hold_bars"].mean()) if len(winners) else 0.0,
        "avg_hold_bars_losers": float(losers["hold_bars"].mean()) if len(losers) else 0.0,
    }


def _simulate_case(
    events,
    case_name: str,
    max_hold_bars: int = 12,
    disable_risk_force_exit: bool = False,
    grace_bars: int = 0,
    confirm_n: int = 1,
    delayed_exit_to: Optional[int] = None,
    hybrid: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    pos: Optional[Dict[str, Any]] = None

    for ev in events:
        if pos is not None:
            pos["hold_bars"] += 1
            pos["risk_seen_consecutive"] = 0 if ev.reason != "risk_force_exit" else pos["risk_seen_consecutive"] + 1
            if pos["risk_grace_left"] > 0:
                pos["risk_grace_left"] -= 1

        if pos is None:
            if ev.decision == "enter":
                pos = {
                    "entry_idx": ev.idx,
                    "entry_time": ev.ts,
                    "entry_price": ev.price,
                    "direction": _direction_from_reason(ev.reason),
                    "hold_bars": 0,
                    "risk_seen_consecutive": 0,
                    "risk_grace_left": 0,
                }
            continue

        should_exit = False
        exit_reason = ""

        if ev.decision == "exit":
            if ev.reason == "exit_signal":
                should_exit = True
                exit_reason = "exit_signal"
            elif ev.reason == "risk_force_exit":
                if disable_risk_force_exit:
                    pass
                else:
                    if hybrid:
                        if pos["risk_grace_left"] == 0:
                            pos["risk_grace_left"] = 3
                        if pos["risk_grace_left"] <= 0 and pos["risk_seen_consecutive"] >= 2:
                            should_exit = True
                            exit_reason = "risk_force_exit_hybrid"
                    else:
                        if grace_bars > 0 and pos["risk_grace_left"] == 0:
                            pos["risk_grace_left"] = grace_bars
                        grace_ok = pos["risk_grace_left"] <= 0
                        confirm_ok = pos["risk_seen_consecutive"] >= max(1, confirm_n)
                        delay_ok = True
                        if delayed_exit_to is not None and pos["hold_bars"] < delayed_exit_to:
                            delay_ok = False
                        if grace_ok and confirm_ok and delay_ok:
                            should_exit = True
                            exit_reason = "risk_force_exit"

        if not should_exit and pos["hold_bars"] >= max_hold_bars:
            should_exit = True
            exit_reason = "max_hold_exit"

        if should_exit:
            raw = _raw_return(pos["direction"], pos["entry_price"], ev.price)
            net = raw - 2.0 * (DEFAULT_FEE_RATE + DEFAULT_SLIPPAGE_RATE)
            rows.append(
                {
                    "case_name": case_name,
                    "entry_idx": pos["entry_idx"],
                    "exit_idx": ev.idx,
                    "entry_time": pos["entry_time"],
                    "exit_time": ev.ts,
                    "direction": pos["direction"],
                    "exit_reason": exit_reason,
                    "hold_bars": pos["hold_bars"],
                    "raw_return": raw,
                    "net_return": net,
                }
            )
            pos = None

    return pd.DataFrame(rows)


def _judge_case(row: pd.Series) -> str:
    cond_a = (
        row["expectancy"] > 0
        and row["net_return_sum"] > 0
        and row["profit_factor"] > 1.0
        and row["max_consecutive_losses_delta"] < 0
        and bool(row["kill_switch_avoided"])
        and row["max_drawdown_proxy_delta"] >= -1e-9
    )
    if cond_a:
        return "A"
    cond_b = (
        row["expectancy_delta"] > 0
        and (row["max_consecutive_losses_delta"] < 0 or bool(row["kill_switch_avoided"]))
    )
    if cond_b:
        return "B"
    return "C"


def main() -> None:
    out_dir = ensure_out_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"risk_force_exit_ablation_{ts}.csv"
    md_path = out_dir / f"risk_force_exit_ablation_{ts}.md"

    events_all = load_tick_events()
    baseline_df = align_to_state_window(reconstruct_actual_trades(events_all))
    baseline_df["net_return"] = baseline_df["raw_return"] - 2.0 * (DEFAULT_FEE_RATE + DEFAULT_SLIPPAGE_RATE)

    start_idx = int(baseline_df["entry_idx"].min()) if not baseline_df.empty else 0
    end_idx = int(baseline_df["exit_idx"].max()) if not baseline_df.empty else (events_all[-1].idx if events_all else 0)
    events = [e for e in events_all if start_idx <= e.idx <= end_idx]

    cases_df: Dict[str, pd.DataFrame] = {
        "current": baseline_df.copy(),
        "disable_risk_force_exit": _simulate_case(events, "disable_risk_force_exit", disable_risk_force_exit=True),
        "grace_1_bar": _simulate_case(events, "grace_1_bar", grace_bars=1),
        "grace_3_bars": _simulate_case(events, "grace_3_bars", grace_bars=3),
        "confirm_2_consecutive": _simulate_case(events, "confirm_2_consecutive", confirm_n=2),
        "confirm_3_consecutive": _simulate_case(events, "confirm_3_consecutive", confirm_n=3),
        "delayed_exit_to_6_bars": _simulate_case(events, "delayed_exit_to_6_bars", delayed_exit_to=6),
        "delayed_exit_to_9_bars": _simulate_case(events, "delayed_exit_to_9_bars", delayed_exit_to=9),
        "hybrid_best_candidate": _simulate_case(events, "hybrid_best_candidate", hybrid=True),
    }

    rows = []
    for name, cdf in cases_df.items():
        rows.append(_calc_metrics(name, cdf))
    out = pd.DataFrame(rows)

    base = out[out["case_name"] == "current"].iloc[0]
    for col in ["expectancy", "net_return_sum", "profit_factor", "max_consecutive_losses", "risk_force_exit_count", "max_drawdown_proxy"]:
        out[f"{col}_delta"] = out[col] - base[col]
    out["kill_switch_avoided"] = base["kill_switch_triggered"] & (~out["kill_switch_triggered"])
    out["grade"] = out.apply(_judge_case, axis=1)

    out = out.sort_values(["grade", "expectancy_delta"], ascending=[True, False]).reset_index(drop=True)
    out.to_csv(csv_path, index=False)

    candidates = out[out["case_name"] != "current"].copy()
    best = candidates.sort_values(
        ["grade", "expectancy_delta", "net_return_sum_delta"], ascending=[True, False, False]
    ).iloc[0]
    top3 = candidates.sort_values(["expectancy_delta", "net_return_sum_delta"], ascending=[False, False]).head(3)
    final_verdict = best["grade"]

    md: List[str] = []
    md.append("# Can_bit Risk Force Exit Ablation Report")
    md.append("")
    md.append("## 1. 목적")
    md.append("- risk_force_exit 완화 variants를 replay/hypothetical로 비교해 lifecycle 병목 완화 가능성 검증")
    md.append("")
    md.append("## 2. 입력 데이터")
    md.append("- data/state/paper_trading_state.json")
    md.append("- data/monitoring/paper_trading_log_*.jsonl")
    md.append("- data/diagnostics/lifecycle/LIFECYCLE_MASTER_REPORT.md (참조)")
    md.append("")
    md.append("## 3. Current baseline 재현")
    md.append(
        f"- trades={int(base['trades'])}, win_rate={base['win_rate']*100:.2f}%, expectancy={base['expectancy']:.6f}, "
        f"PF={base['profit_factor']:.4f}, gross={base['gross_return_sum']:.6f}, net={base['net_return_sum']:.6f}, "
        f"KS={bool(base['kill_switch_triggered'])}"
    )
    md.append("")
    md.append("## 4. Case별 결과표")
    for _, r in out.iterrows():
        md.append(
            f"- {r['case_name']}: grade={r['grade']} exp={r['expectancy']:.6f} (Δ{r['expectancy_delta']:+.6f}), "
            f"net={r['net_return_sum']:.6f} (Δ{r['net_return_sum_delta']:+.6f}), PF={r['profit_factor']:.4f} "
            f"(Δ{r['profit_factor_delta']:+.4f}), KS={bool(r['kill_switch_triggered'])}, "
            f"MDD={r['max_drawdown_proxy']:.6f} (Δ{r['max_drawdown_proxy_delta']:+.6f})"
        )
    md.append("")
    md.append("## 5. Top 3 후보")
    for _, r in top3.iterrows():
        md.append(
            f"- {r['case_name']}: grade={r['grade']} expΔ={r['expectancy_delta']:+.6f}, "
            f"netΔ={r['net_return_sum_delta']:+.6f}, PFΔ={r['profit_factor_delta']:+.4f}, "
            f"KS_avoided={bool(r['kill_switch_avoided'])}"
        )
    md.append("")
    md.append("## 6. Case별 위험 분석")
    md.append("- A: expectancy 양수+PF>1+KS 완화+MDD 비악화")
    md.append("- B: 개선은 있으나 순이익/안정성 미흡")
    md.append("- C: 기대값 악화 또는 리스크 악화")
    md.append("")
    md.append("## 7. KS 회피 여부")
    for _, r in out.iterrows():
        if r["case_name"] == "current":
            continue
        md.append(
            f"- {r['case_name']}: KS_triggered={bool(r['kill_switch_triggered'])}, "
            f"kill_switch_avoided={bool(r['kill_switch_avoided'])}, max_consec_loss={int(r['max_consecutive_losses'])}"
        )
    md.append("")
    md.append("## 8. 비용 영향")
    md.append("- estimated_fee_cost = trades * 2 * fee_rate")
    md.append("- estimated_slippage_cost = trades * 2 * slippage_rate")
    md.append("- gross/net 동시 비교로 비용 훼손 정도 추적")
    md.append("")
    md.append("## 9. 최종 후보 1개")
    md.append(
        f"- {best['case_name']} (grade={best['grade']}): exp={best['expectancy']:.6f}, "
        f"net={best['net_return_sum']:.6f}, PF={best['profit_factor']:.4f}, KS={bool(best['kill_switch_triggered'])}"
    )
    md.append("")
    md.append("## 10. 운영 적용 전 forward validation 조건")
    md.append("- 최근 14/30/60/90일 구간에서 current vs best_case 고정 비교")
    md.append("- 동일한 fee/slippage/position_size 조건 유지")
    md.append("- KS, MDD, expectancy 동시 통과 시에만 운영 적용 검토")

    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("[RISK FORCE EXIT ABLATION SUMMARY]")
    print("")
    print("current:")
    print(f"- expectancy: {base['expectancy']:.6f}")
    print(f"- net_return: {base['net_return_sum']:.6f}")
    print(f"- PF: {base['profit_factor']:.4f}")
    print(f"- KS: {bool(base['kill_switch_triggered'])}")
    print("")
    print("best_case:")
    print(f"- name: {best['case_name']}")
    print(f"- expectancy: {best['expectancy']:.6f}")
    print(f"- net_return: {best['net_return_sum']:.6f}")
    print(f"- PF: {best['profit_factor']:.4f}")
    print(f"- KS: {bool(best['kill_switch_triggered'])}")
    print(
        f"- reason: grade={best['grade']}, expectancy_delta={best['expectancy_delta']:+.6f}, "
        f"net_return_delta={best['net_return_sum_delta']:+.6f}"
    )
    print("")
    print("final_verdict:")
    print(f"- {final_verdict}")
    print("")
    print("created:")
    print(f"- csv: {csv_path}")
    print(f"- markdown: {md_path}")


if __name__ == "__main__":
    main()
