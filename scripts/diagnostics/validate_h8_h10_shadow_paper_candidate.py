"""
H8 / H10 shadow-paper candidate validation (diagnostics only).

Compares baseline vs H8 vs H10 on rolling OOS; no operational code changes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from scripts.diagnostics.validate_final_hypotheses import _simulate
from scripts.diagnostics.validate_entry_quality_filters import (
    _slice_ticks_by_range,
    _window_endpoints,
)
from scripts.run_daily_paper import load_ohlcv, simulate_signals_paper


OUT_DIR = Path("data/diagnostics/h8_h10_validation")

SHADOW_SCORE_FORMULA = (
    "shadow_score = 0.40*consistency_norm + 0.30*expectancy_improvement_norm + "
    "0.20*mdd_improvement_norm + 0.10*trade_preservation_norm"
)
PAPER_SCORE_FORMULA = (
    "paper_score = 0.30*net_improvement_norm + 0.30*expectancy_norm + "
    "0.20*mdd_norm + 0.20*ks_avoidance_norm"
)


@dataclass
class Candidate:
    id: str
    description: str
    params: Dict[str, Any]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("validate_h8_h10_shadow_paper_candidate")
    p.add_argument("--lookback-list", default="14,30,60,90,180")
    p.add_argument("--stride-days", type=int, default=7)
    p.add_argument("--max-windows-per-lookback", type=int, default=9999)
    p.add_argument("--position-size", type=float, default=0.05)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0002)
    return p.parse_args()


def _candidates() -> List[Candidate]:
    return [
        Candidate("baseline", "production-equivalent replay", {}),
        Candidate("H8", "entropy<=0.96 AND trend==up", {"entropy_ub": 0.96, "trend_up_only": True}),
        Candidate("H10", "entropy<=0.96 AND trend!=down", {"entropy_ub": 0.96, "exclude_down": True}),
    ]


def _trade_reduction_pct(base_trades: int, cand_trades: int) -> float:
    if base_trades <= 0:
        return 0.0
    return (1.0 - cand_trades / base_trades) * 100.0


def _consistency_metrics(sub: pd.DataFrame) -> Dict[str, float]:
    improved = int((sub["net_return_delta"] > 0).sum())
    degraded = int((sub["net_return_delta"] < 0).sum())
    flat = int((sub["net_return_delta"] == 0).sum())
    total = improved + degraded + flat
    arr = sub["net_return_delta"].tolist()
    avg_d = float(mean(arr)) if arr else 0.0
    std_d = float(pstdev(arr)) if len(arr) > 1 else 0.0
    ratio = improved / total if total > 0 else 0.0
    return {
        "improved_windows": float(improved),
        "degraded_windows": float(degraded),
        "flat_windows": float(flat),
        "improved_ratio": ratio,
        "avg_net_delta": avg_d,
        "std_net_delta": std_d,
        "consistency_score": avg_d / (std_d + 1e-9),
    }


def _norm01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _shadow_paper_scores(row: pd.Series, base_exp: float, base_ks: float) -> Dict[str, float]:
    imp_ratio = float(row["improved_ratio"])
    cons_norm = _norm01(float(row["consistency_score"]), -0.5, 2.0)
    exp_imp = float(row["avg_expectancy_delta"])
    exp_norm = _norm01(exp_imp, -0.01, 0.01)
    mdd_imp = float(row["avg_mdd_delta"])
    mdd_norm = _norm01(mdd_imp, -0.01, 0.01)
    pres = 1.0 - float(row["trade_reduction_pct_avg"]) / 100.0
    pres_norm = float(np.clip(pres, 0.0, 1.0))

    net_imp = float(row["avg_net_delta"])
    net_norm = _norm01(net_imp, -0.05, 0.05)
    exp_lvl = float(row["avg_expectancy"])
    exp_lvl_norm = _norm01(exp_lvl, base_exp, 0.01)
    ks_avoid = max(0.0, -float(row["ks_delta_avg"]))
    ks_norm = _norm01(ks_avoid, 0.0, max(base_ks, 0.1))

    shadow = (
        0.40 * _norm01(imp_ratio, 0.4, 0.8)
        + 0.30 * exp_norm
        + 0.20 * mdd_norm
        + 0.10 * pres_norm
    )
    paper = 0.30 * net_norm + 0.30 * exp_lvl_norm + 0.20 * mdd_norm + 0.20 * ks_norm
    return {"shadow_score": shadow, "paper_score": paper}


def _deploy_grade(row: pd.Series) -> str:
    net = float(row["avg_net_delta"])
    imp = float(row["improved_ratio"])
    shadow = float(row["shadow_score"])
    paper = float(row["paper_score"])
    over = float(row["trade_reduction_pct_avg"]) > 80.0
    if net > 0 and imp >= 0.55 and shadow >= 0.55 and paper >= 0.50 and not over:
        return "A"
    if net > 0 and imp >= 0.50 and shadow >= 0.45:
        return "B"
    if net > 0 or imp >= 0.52:
        return "C"
    return "D"


def _h2h(h8: pd.DataFrame, h10: pd.DataFrame) -> Dict[str, Any]:
    keys = ["lookback_days", "window_id"]
    m = h8.merge(h10, on=keys, suffixes=("_h8", "_h10"))
    if m.empty:
        return {}
    h8_better_net = int((m["net_return_delta_h8"] > m["net_return_delta_h10"]).sum())
    h10_better_net = int((m["net_return_delta_h10"] > m["net_return_delta_h8"]).sum())
    h8_better_exp = int((m["expectancy_h8"] > m["expectancy_h10"]).sum())
    h8_better_mdd = int((m["max_drawdown_h8"] > m["max_drawdown_h10"]).sum())
    h8_less_reduction = int((m["trade_reduction_pct_h8"] < m["trade_reduction_pct_h10"]).sum())
    return {
        "paired_windows": len(m),
        "h8_more_net_improvement_windows": h8_better_net,
        "h10_more_net_improvement_windows": h10_better_net,
        "h8_higher_expectancy_windows": h8_better_exp,
        "h8_better_mdd_windows": h8_better_mdd,
        "h8_less_trade_reduction_windows": h8_less_reduction,
        "avg_net_delta_h8": float(m["net_return_delta_h8"].mean()),
        "avg_net_delta_h10": float(m["net_return_delta_h10"].mean()),
        "avg_expectancy_h8": float(m["expectancy_h8"].mean()),
        "avg_expectancy_h10": float(m["expectancy_h10"].mean()),
        "avg_mdd_h8": float(m["max_drawdown_h8"].mean()),
        "avg_mdd_h10": float(m["max_drawdown_h10"].mean()),
        "avg_trade_reduction_pct_h8": float(m["trade_reduction_pct_h8"].mean()),
        "avg_trade_reduction_pct_h10": float(m["trade_reduction_pct_h10"].mean()),
    }


def main() -> None:
    args = _parse_args()
    lookbacks = [int(x.strip()) for x in args.lookback_list.split(",") if x.strip()]
    cands = _candidates()

    df = load_ohlcv()
    ticks, _ = simulate_signals_paper(df)

    detail_rows: List[Dict[str, Any]] = []

    for lb in lookbacks:
        wins = _window_endpoints(ticks, lb, args.stride_days, args.max_windows_per_lookback)
        print(f"[lookback={lb}d] windows={len(wins)}", flush=True)
        for wid, (ws, we) in enumerate(wins):
            w_ticks = _slice_ticks_by_range(ticks, ws, we)
            if len(w_ticks) < 30:
                continue

            base_m, base_tdf = _simulate(
                w_ticks, {}, args.position_size, args.fee_rate, args.slippage_rate
            )
            base_trades = int(base_m["total_trades"])

            for cand in cands:
                if cand.id == "baseline":
                    m = base_m
                    tdf = base_tdf
                else:
                    m, tdf = _simulate(
                        w_ticks, cand.params, args.position_size, args.fee_rate, args.slippage_rate
                    )

                cand_trades = int(m["total_trades"])
                tr_red = 0.0 if cand.id == "baseline" else _trade_reduction_pct(base_trades, cand_trades)

                rec: Dict[str, Any] = {
                    "candidate_id": cand.id,
                    "description": cand.description,
                    "lookback_days": lb,
                    "window_id": wid,
                    "window_start": ws,
                    "window_end": we,
                    "trades": cand_trades,
                    "win_rate": m["win_rate"],
                    "expectancy": m["expectancy"],
                    "profit_factor": m["profit_factor"],
                    "net_return": m["net_return"],
                    "max_drawdown": m["max_drawdown"],
                    "kill_switch_triggered": m["kill_switch_triggered"],
                    "risk_force_exit_count": m["risk_force_exit_count"],
                    "trade_reduction_pct": tr_red,
                }
                if cand.id != "baseline":
                    rec["expectancy_delta"] = m["expectancy"] - base_m["expectancy"]
                    rec["net_return_delta"] = m["net_return"] - base_m["net_return"]
                    rec["pf_delta"] = m["profit_factor"] - base_m["profit_factor"]
                    rec["mdd_delta"] = m["max_drawdown"] - base_m["max_drawdown"]
                    rec["ks_delta"] = int(m["kill_switch_triggered"]) - int(base_m["kill_switch_triggered"])
                    rec["force_exit_delta"] = m["risk_force_exit_count"] - base_m["risk_force_exit_count"]
                else:
                    rec["expectancy_delta"] = 0.0
                    rec["net_return_delta"] = 0.0
                    rec["pf_delta"] = 0.0
                    rec["mdd_delta"] = 0.0
                    rec["ks_delta"] = 0
                    rec["force_exit_delta"] = 0
                detail_rows.append(rec)

    detail = pd.DataFrame(detail_rows)
    base_line = detail[detail["candidate_id"] == "baseline"]
    base_exp = float(base_line["expectancy"].mean()) if len(base_line) else -0.002
    base_ks = float(base_line["kill_switch_triggered"].mean()) if len(base_line) else 0.1

    summary_rows: List[Dict[str, Any]] = []
    for cand in cands:
        if cand.id == "baseline":
            sub = detail[detail["candidate_id"] == "baseline"]
            cons = {"improved_windows": 0.0, "degraded_windows": 0.0, "improved_ratio": 0.0, "avg_net_delta": 0.0, "std_net_delta": 0.0, "consistency_score": 0.0}
        else:
            sub = detail[detail["candidate_id"] == cand.id]
            cons = _consistency_metrics(sub)

        row = {
            "candidate_id": cand.id,
            "description": cand.description,
            "windows": len(sub),
            "avg_trades": float(sub["trades"].mean()),
            "avg_win_rate": float(sub["win_rate"].mean()),
            "avg_expectancy": float(sub["expectancy"].mean()),
            "avg_profit_factor": float(sub["profit_factor"].mean()),
            "avg_net_return": float(sub["net_return"].mean()),
            "avg_max_drawdown": float(sub["max_drawdown"].mean()),
            "trade_reduction_pct_avg": float(sub["trade_reduction_pct"].mean()) if cand.id != "baseline" else 0.0,
            "avg_expectancy_delta": float(sub["expectancy_delta"].mean()) if cand.id != "baseline" else 0.0,
            "avg_net_delta": float(sub["net_return_delta"].mean()) if cand.id != "baseline" else 0.0,
            "avg_pf_delta": float(sub["pf_delta"].mean()) if cand.id != "baseline" else 0.0,
            "avg_mdd_delta": float(sub["mdd_delta"].mean()) if cand.id != "baseline" else 0.0,
            "ks_delta_avg": float(sub["ks_delta"].mean()) if cand.id != "baseline" else 0.0,
            "force_exit_delta_avg": float(sub["force_exit_delta"].mean()) if cand.id != "baseline" else 0.0,
            **cons,
        }
        if cand.id != "baseline":
            scores = _shadow_paper_scores(pd.Series(row), base_exp, base_ks)
            row.update(scores)
            row["deploy_readiness"] = _deploy_grade(pd.Series({**row, **scores}))
            row["over_filtered"] = row["trade_reduction_pct_avg"] > 80.0
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    h8_det = detail[detail["candidate_id"] == "H8"].copy()
    h10_det = detail[detail["candidate_id"] == "H10"].copy()
    h2h = _h2h(h8_det, h10_det)

    # pick winner among H8/H10
    h8s = summary[summary["candidate_id"] == "H8"].iloc[0]
    h10s = summary[summary["candidate_id"] == "H10"].iloc[0]
    winner = "H8"
    if float(h10s["paper_score"]) > float(h8s["paper_score"]) + 0.02:
        winner = "H10"
    elif float(h10s["shadow_score"]) > float(h8s["shadow_score"]) + 0.03:
        winner = "H10"
    elif float(h8s["avg_net_delta"]) >= float(h10s["avg_net_delta"]):
        winner = "H8"
    else:
        winner = "H10"

    shadow_cand = winner
    winner_grade = str(summary[summary["candidate_id"] == winner]["deploy_readiness"].iloc[0])
    paper_cand = winner if winner_grade in ("A", "B") else ("H10" if winner == "H8" else "H8")

    grade = winner_grade
    if grade in ("A", "B"):
        verdict = "운영 후보 가능 (shadow/paper forward monitor)"
    elif grade == "C":
        verdict = "추가 검증 필요 (blocked walk-forward / 최소 trade floor)"
    else:
        verdict = "추가 검증 필요 (현 단계 paper 반영 비권장)"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"h8_h10_candidate_validation_{ts}.csv"
    md_path = OUT_DIR / f"h8_h10_candidate_validation_{ts}.md"

    parts = [
        detail.assign(section="rolling_detail"),
        summary.assign(section="candidate_summary"),
    ]
    if h2h:
        parts.append(pd.DataFrame([{**h2h, "section": "h8_vs_h10"}]))
    pd.concat(parts, ignore_index=True, sort=False).to_csv(csv_path, index=False)

    md: List[str] = [
        "# H8 / H10 Shadow-Paper Candidate Validation",
        "",
        f"- generated: {ts}",
        f"- total windows (per candidate): {int(h8s.get('windows', 0))}",
        f"- baseline avg expectancy: {base_exp:.6f}",
        "",
        "## Score formulas",
        f"- {SHADOW_SCORE_FORMULA}",
        f"- {PAPER_SCORE_FORMULA}",
        "",
        "## Candidate summary",
    ]
    for _, r in summary.iterrows():
        md.append(
            f"- **{r['candidate_id']}**: exp={r['avg_expectancy']:.6f}, net={r['avg_net_return']:.6f}, "
            f"netΔ={r.get('avg_net_delta', 0):+.6f}, improved_ratio={r.get('improved_ratio', 0):.2%}, "
            f"shadow={r.get('shadow_score', 0):.4f}, paper={r.get('paper_score', 0):.4f}, "
            f"grade={r.get('deploy_readiness', '-')}"
        )
    if h2h:
        md.extend(
            [
                "",
                "## H8 vs H10 head-to-head",
                f"- paired_windows: {h2h['paired_windows']}",
                f"- H8 more net-improvement windows: {h2h['h8_more_net_improvement_windows']}",
                f"- H10 more net-improvement windows: {h2h['h10_more_net_improvement_windows']}",
                f"- H8 higher expectancy windows: {h2h['h8_higher_expectancy_windows']}",
                f"- H8 better MDD windows: {h2h['h8_better_mdd_windows']}",
                f"- H8 less trade reduction windows: {h2h['h8_less_trade_reduction_windows']}",
            ]
        )
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    ws = h8s if winner == "H8" else h10s
    why = (
        f"shadow_score={float(ws['shadow_score']):.4f}, paper_score={float(ws['paper_score']):.4f}, "
        f"netΔ={float(ws['avg_net_delta']):+.6f}, improved_ratio={float(ws['improved_ratio']):.2%}, "
        f"trade_reduction={float(ws['trade_reduction_pct_avg']):.1f}%"
    )

    print("[H8 / H10 CANDIDATE VALIDATION]")
    print("")
    print(f"FINAL WINNER:\n{winner}")
    print(f"\nWHY:\n{why}")
    print(f"\nSHADOW CANDIDATE:\n{shadow_cand}")
    print(f"\nPAPER CANDIDATE:\n{paper_cand}")
    print(f"\nDEPLOY READINESS:\nH8={h8s.get('deploy_readiness','-')}, H10={h10s.get('deploy_readiness','-')}")
    print(f"\nFINAL VERDICT:\n{verdict}")
    print(f"\nscore_formulas:\n- {SHADOW_SCORE_FORMULA}\n- {PAPER_SCORE_FORMULA}")
    if h2h:
        print(
            f"\nH8 vs H10:\n"
            f"- net wins H8/H10: {h2h['h8_more_net_improvement_windows']}/{h2h['h10_more_net_improvement_windows']}\n"
            f"- less trade reduction windows H8: {h2h['h8_less_trade_reduction_windows']}"
        )
    print(f"\ncreated:\n- {csv_path}\n- {md_path}")


if __name__ == "__main__":
    main()
