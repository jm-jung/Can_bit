"""
Full-history structural replay: Production baseline vs H8 SoftGate G30.

Diagnostics only — no live orders, no state mutation, no deployment.
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

from scripts.diagnostics.validate_h8_soft_risk_gate import (
    Variant,
    _entropy,
    _h8_pass,
    _is_loss_cluster_trade,
    _simulate_variant,
)
from scripts.run_daily_paper import load_ohlcv, simulate_signals_paper


OUT_DIR = Path("data/diagnostics/full_history")
POSITION_SIZE = 0.05
FEE_RATE = 0.0004
SLIPPAGE_RATE = 0.0002
FAIL_SCALE = 0.30

BASELINE = Variant("Baseline", fail_scale=1.0, hard_block=False)
G30 = Variant("Variant_G30", fail_scale=FAIL_SCALE, hard_block=False)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("replay_h8_softgate_full_history")
    p.add_argument("--position-size", type=float, default=POSITION_SIZE)
    p.add_argument("--fee-rate", type=float, default=FEE_RATE)
    p.add_argument("--slippage-rate", type=float, default=SLIPPAGE_RATE)
    p.add_argument("--rolling-window-trades", type=int, default=20)
    return p.parse_args()


def _enrich_trades(tdf: pd.DataFrame, ticks: List[Dict[str, Any]]) -> pd.DataFrame:
    if tdf.empty:
        return tdf.copy()
    rows = []
    for _, tr in tdf.iterrows():
        i = int(tr["entry_idx"])
        t = ticks[i]
        exit_i = min(i + int(tr["hold_bars"]), len(ticks) - 1)
        entry_ts = pd.Timestamp(t.get("timestamp")) if t.get("timestamp") else pd.NaT
        exit_ts = (
            pd.Timestamp(ticks[exit_i].get("timestamp"))
            if ticks[exit_i].get("timestamp")
            else entry_ts
        )
        trend = str(t.get("trend_label") or "unknown")
        vol = str(t.get("vol_bucket") or "unknown")
        ent = _entropy(t)
        pnl = float(tr["scaled_return"]) * POSITION_SIZE
        rows.append(
            {
                **tr.to_dict(),
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "trend_label": trend,
                "vol_bucket": vol,
                "entropy": ent,
                "h8_pass_entry": _h8_pass(t),
                "market_regime": _market_regime(trend),
                "vol_regime": "high_vol" if vol == "high" else "low_vol",
                "trend_regime": "trend_up" if trend == "up" else ("trend_down" if trend == "down" else "sideways"),
                "pnl_contribution": pnl,
                "loss_cluster_flag": _is_loss_cluster_trade(tr.to_dict(), t),
            }
        )
    out = pd.DataFrame(rows)
    out["exit_ts"] = pd.to_datetime(out["exit_ts"], errors="coerce")
    out["entry_ts"] = pd.to_datetime(out["entry_ts"], errors="coerce")
    return out


def _market_regime(trend: str) -> str:
    if trend == "up":
        return "bull"
    if trend == "down":
        return "bear"
    return "sideways"


def _equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["exit_ts", "equity", "underwater", "drawdown"])
    eq = 1.0
    peak = 1.0
    pts = []
    for _, tr in trades.sort_values("exit_ts").iterrows():
        eq *= 1.0 + float(tr["pnl_contribution"])
        peak = max(peak, eq)
        dd = (eq - peak) / peak if peak > 0 else 0.0
        pts.append({"exit_ts": tr["exit_ts"], "equity": eq, "drawdown": dd, "underwater": dd < 0})
    return pd.DataFrame(pts)


def _daily_equity(curve: pd.DataFrame) -> pd.Series:
    if curve.empty:
        return pd.Series(dtype=float)
    s = curve.set_index("exit_ts")["equity"].sort_index()
    daily = s.resample("D").last().ffill()
    daily.iloc[0] = 1.0
    return daily.ffill()


def _core_metrics(trades: pd.DataFrame, curve: pd.DataFrame, sim_m: Dict[str, Any]) -> Dict[str, Any]:
    vals = trades["scaled_return"].tolist() if not trades.empty else []
    span_days = 1.0
    if not curve.empty and curve["exit_ts"].notna().any():
        t0 = curve["exit_ts"].min()
        t1 = curve["exit_ts"].max()
        span_days = max((t1 - t0).total_seconds() / 86400.0, 1.0)
    final_eq = float(curve["equity"].iloc[-1]) if not curve.empty else 1.0
    cagr = (final_eq ** (365.0 / span_days)) - 1.0 if final_eq > 0 else 0.0
    mdd = float(sim_m["MDD"])
    sharpe = 0.0
    if len(vals) > 1:
        sharpe = float(np.mean(vals) / (np.std(vals) + 1e-9) * math.sqrt(len(vals)))
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    return {
        "trades": int(sim_m["trades"]),
        "win_rate": float(sim_m["win_rate"]),
        "expectancy": float(sim_m["expectancy"]),
        "profit_factor": float(sim_m["profit_factor"]),
        "net_return": float(sim_m["net_return"]),
        "cagr_proxy": float(cagr),
        "MDD": mdd,
        "sharpe_proxy": sharpe,
        "calmar_proxy": calmar,
        "avg_hold": float(sim_m["avg_hold"]),
        "RFE_count": int(sim_m["risk_force_exit_count"]),
        "KS_triggered": bool(sim_m["KS_triggered"]),
        "span_days": span_days,
        "final_equity": final_eq,
    }


def _equity_quality(curve: pd.DataFrame, daily: pd.Series) -> Dict[str, Any]:
    if curve.empty:
        return {
            "max_underwater_duration_days": 0,
            "avg_recovery_days": 0.0,
            "equity_smoothness": 0.0,
            "worst_1d_loss": 0.0,
            "worst_7d_loss": 0.0,
            "worst_30d_loss": 0.0,
            "tail_risk_top5pct": 0.0,
        }
    dd = curve["drawdown"].tolist()
    # longest underwater stretch (trade-level approx)
    max_uw = 0
    cur_uw = 0
    for d in dd:
        if d < -1e-9:
            cur_uw += 1
            max_uw = max(max_uw, cur_uw)
        else:
            cur_uw = 0

    rets = daily.pct_change().dropna() if len(daily) > 1 else pd.Series(dtype=float)
    w1 = float(rets.min()) if not rets.empty else 0.0
    roll7 = daily.pct_change(7).dropna()
    roll30 = daily.pct_change(30).dropna()
    tail = float(rets.quantile(0.05)) if len(rets) >= 20 else w1

    return {
        "max_underwater_duration_days": int(max_uw),
        "avg_recovery_days": float(max_uw * 0.5),
        "equity_smoothness": float(1.0 / (rets.std() + 1e-9)) if not rets.empty else 0.0,
        "worst_1d_loss": w1,
        "worst_7d_loss": float(roll7.min()) if not roll7.empty else 0.0,
        "worst_30d_loss": float(roll30.min()) if not roll30.empty else 0.0,
        "tail_risk_top5pct": tail,
    }


def _period_returns(trades: pd.DataFrame, freq: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["period", "pnl"])
    g = trades.dropna(subset=["exit_ts"]).copy()
    g["period"] = g["exit_ts"].dt.to_period(freq)
    return g.groupby("period")["pnl_contribution"].sum().reset_index(name="pnl")


def _regime_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for col, label in [
        ("market_regime", "market"),
        ("vol_regime", "volatility"),
        ("trend_regime", "trend"),
    ]:
        for key, g in trades.groupby(col):
            pnl = g["pnl_contribution"].sum()
            rows.append(
                {
                    "split_type": label,
                    "regime": key,
                    "trades": len(g),
                    "net_pnl": float(pnl),
                    "expectancy": float(g["scaled_return"].mean()),
                    "win_rate": float((g["scaled_return"] > 0).mean()),
                    "avg_scale": float(g["scale"].mean()) if "scale" in g.columns else 1.0,
                }
            )
    return pd.DataFrame(rows)


def _loss_cluster_stats(trades: pd.DataFrame) -> Dict[str, Any]:
    if trades.empty:
        return {
            "worst_losing_streak": 0,
            "RFE_count": 0,
            "RFE_cluster_max": 0,
            "loss_cluster_trades": 0,
            "catastrophic_sequences_3plus": 0,
        }
    streak = 0
    worst = 0
    for r in trades.sort_values("exit_ts")["scaled_return"]:
        if r < 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0

    rfe = trades[trades["exit_reason"] == "risk_force_exit"]
    rfe_cluster = 0
    cur = 0
    for reason in trades.sort_values("exit_ts")["exit_reason"]:
        if reason == "risk_force_exit":
            cur += 1
            rfe_cluster = max(rfe_cluster, cur)
        else:
            cur = 0

    cat = 0
    cur_loss = 0
    for r in trades.sort_values("exit_ts")["scaled_return"]:
        if r < -0.005:
            cur_loss += 1
            if cur_loss >= 3:
                cat += 1
        else:
            cur_loss = 0

    return {
        "worst_losing_streak": int(worst),
        "RFE_count": int(len(rfe)),
        "RFE_cluster_max": int(rfe_cluster),
        "loss_cluster_trades": int(trades["loss_cluster_flag"].sum()) if "loss_cluster_flag" in trades else 0,
        "catastrophic_sequences_3plus": int(cat),
    }


def _rolling_stability(trades: pd.DataFrame, window: int) -> Dict[str, Any]:
    if trades.empty or len(trades) < window:
        return {"rolling_expectancy_mean": 0.0, "rolling_pf_mean": 0.0, "rolling_mdd_min": 0.0}
    s = trades.sort_values("exit_ts")
    exps, pfs, mdds = [], [], []
    vals = s["scaled_return"].tolist()
    for i in range(window, len(vals) + 1):
        w = vals[i - window : i]
        wins = [x for x in w if x > 0]
        losses = [x for x in w if x < 0]
        wr = len(wins) / len(w)
        exp = (wr * mean(wins) if wins else 0) - ((1 - wr) * abs(mean(losses)) if losses else 0)
        pf = (sum(wins) / abs(sum(losses))) if losses else 0.0
        eq = 1.0
        peak = 1.0
        mdd = 0.0
        for r in w:
            eq *= 1.0 + r * POSITION_SIZE
            peak = max(peak, eq)
            mdd = min(mdd, (eq - peak) / peak if peak > 0 else 0.0)
        exps.append(exp)
        pfs.append(pf)
        mdds.append(mdd)
    return {
        "rolling_expectancy_mean": float(mean(exps)),
        "rolling_pf_mean": float(mean(pfs)),
        "rolling_mdd_min": float(min(mdds)),
    }


def _profit_concentration(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {"top_1pct": 0.0, "top_5pct": 0.0, "top_10pct": 0.0}
    total = trades["pnl_contribution"].sum()
    if abs(total) < 1e-12:
        return {"top_1pct": 0.0, "top_5pct": 0.0, "top_10pct": 0.0}
    ranked = trades["pnl_contribution"].sort_values(ascending=False)
    n = len(ranked)

    def _share(k: int) -> float:
        top_n = max(1, int(math.ceil(n * k / 100.0)))
        return float(ranked.head(top_n).sum() / total)

    return {"top_1pct": _share(1), "top_5pct": _share(5), "top_10pct": _share(10)}


def _monthly_consistency(prod_m: pd.DataFrame, sg_m: pd.DataFrame) -> Dict[str, Any]:
    if prod_m.empty and sg_m.empty:
        return {"months_total": 0, "months_g30_better": 0, "monthly_consistency_pct": 0.0}
    merged = prod_m.merge(sg_m, on="period", how="outer", suffixes=("_prod", "_sg")).fillna(0.0)
    better = int((merged["pnl_sg"] > merged["pnl_prod"]).sum())
    total = len(merged)
    return {
        "months_total": total,
        "months_g30_better": better,
        "monthly_consistency_pct": better / total if total else 0.0,
    }


def _quarterly_consistency(prod_q: pd.DataFrame, sg_q: pd.DataFrame) -> Dict[str, Any]:
    mc = _monthly_consistency(prod_q, sg_q)
    mc["quarters_total"] = mc.pop("months_total")
    mc["quarters_g30_better"] = mc.pop("months_g30_better")
    mc["quarterly_consistency_pct"] = mc.pop("monthly_consistency_pct")
    return mc


def _regime_delta(prod_rb: pd.DataFrame, sg_rb: pd.DataFrame) -> pd.DataFrame:
    if prod_rb.empty:
        return pd.DataFrame()
    m = prod_rb.merge(sg_rb, on=["split_type", "regime"], suffixes=("_prod", "_sg"))
    m["net_pnl_delta"] = m["net_pnl_sg"] - m["net_pnl_prod"]
    m["expectancy_delta"] = m["expectancy_sg"] - m["expectancy_prod"]
    return m.sort_values("net_pnl_delta", ascending=False)


def _low_trade_diagnosis(ticks: List[Dict[str, Any]], recent_days: int = 30) -> Dict[str, Any]:
    ts_list = []
    for t in ticks:
        if t.get("timestamp"):
            ts_list.append(pd.Timestamp(t["timestamp"]))
    if not ts_list:
        return {"diagnosis": "unknown", "recent_h8_pass_ratio": 0.0, "historical_h8_pass_ratio": 0.0}

    ts_max = max(ts_list)
    cutoff = ts_max - pd.Timedelta(days=recent_days)

    def _candidate_stats(subset: List[Dict[str, Any]]) -> Dict[str, float]:
        prod_c = h8_p = 0
        for t in subset:
            vol = str(t.get("vol_bucket") or "")
            sig = t.get("signal")
            ent = _entropy(t)
            if vol not in ("mid", "high") or sig is None:
                continue
            trend = str(t.get("trend_label") or "")
            if trend != "sideways" and ent > 1.0:
                continue
            prod_c += 1
            if _h8_pass(t):
                h8_p += 1
        return {
            "production_candidates": prod_c,
            "h8_pass_ratio": h8_p / max(prod_c, 1),
        }

    recent = [t for t, ts in zip(ticks, ts_list) if ts >= cutoff]
    hist = ticks
    r = _candidate_stats(recent)
    h = _candidate_stats(hist)
    cand_drop = (r["production_candidates"] / max(h["production_candidates"] / max(len(hist), 1) * len(recent), 1)) - 1.0

    if r["production_candidates"] < 5 and h["production_candidates"] > 50:
        diag = "filter-driven" if r["h8_pass_ratio"] < h["h8_pass_ratio"] * 0.7 else "market-driven"
    elif abs(cand_drop) < 0.3:
        diag = "market-driven"
    else:
        diag = "filter-driven" if r["h8_pass_ratio"] < h["h8_pass_ratio"] else "market-driven"

    return {
        "diagnosis": diag,
        "recent_production_candidates": r["production_candidates"],
        "historical_production_candidates_per_day": h["production_candidates"] / max((ts_max - min(ts_list)).days, 1),
        "recent_h8_pass_ratio": r["h8_pass_ratio"],
        "historical_h8_pass_ratio": h["h8_pass_ratio"],
    }


def _structural_verdict(
    prod_core: Dict[str, Any],
    sg_core: Dict[str, Any],
    monthly: Dict[str, Any],
    quarterly: Dict[str, Any],
    regime_delta: pd.DataFrame,
    prod_conc: Dict[str, float],
    sg_conc: Dict[str, float],
) -> Tuple[str, Dict[str, Any]]:
    net_delta = sg_core["net_return"] - prod_core["net_return"]
    mdd_delta = sg_core["MDD"] - prod_core["MDD"]
    rfe_delta = sg_core["RFE_count"] - prod_core["RFE_count"]
    eq_improved = sg_core["final_equity"] > prod_core["final_equity"]
    mdd_improved = mdd_delta >= 0
    month_pct = monthly.get("monthly_consistency_pct", 0.0)
    quarter_pct = quarterly.get("quarterly_consistency_pct", 0.0)

    top_regime = ""
    if not regime_delta.empty:
        top_regime = str(regime_delta.iloc[0]["regime"])

    conc_reduced = sg_conc["top_10pct"] < prod_conc["top_10pct"]

    struct_robust = net_delta > 0 and mdd_improved and month_pct >= 0.55
    regime_dependent = month_pct < 0.55 or (not regime_delta.empty and regime_delta["net_pnl_delta"].std() > abs(net_delta) * 2)
    overfit_risk = month_pct < 0.45 and net_delta > 0
    unstable = net_delta <= 0 or (not mdd_improved and net_delta < abs(prod_core["net_return"]) * 0.05)

    if struct_robust and not overfit_risk:
        grade = "A"
        promotion = "enter paper candidate stage"
    elif net_delta > 0 and mdd_improved and month_pct >= 0.45:
        grade = "B"
        promotion = "continue monitor"
    elif net_delta > 0 or mdd_improved:
        grade = "C"
        promotion = "continue monitor"
    else:
        grade = "D"
        promotion = "be abandoned"

    if grade in ("A", "B") and rfe_delta <= 0 and conc_reduced:
        evolution = "become hybrid risk gate"
    elif grade == "A":
        evolution = "enter paper candidate stage"
    else:
        evolution = promotion

    narrative = {
        "equity_quality_improved": eq_improved and mdd_improved,
        "robustness_improved": mdd_improved and rfe_delta <= 0,
        "primary_help_regime": top_regime or "n/a",
        "structurally_robust": struct_robust,
        "regime_dependent": regime_dependent,
        "overfit_risk": overfit_risk,
        "unstable": unstable,
        "recommended_action": evolution,
        "grade": grade,
        "net_delta": net_delta,
        "mdd_delta": mdd_delta,
        "monthly_g30_win_pct": month_pct,
        "quarterly_g30_win_pct": quarter_pct,
        "outlier_dependency_reduced": conc_reduced,
    }
    return grade, narrative


def _insights(narrative: Dict[str, Any], regime_delta: pd.DataFrame, low_trade: Dict[str, Any]) -> List[str]:
    insights = []
    if narrative["equity_quality_improved"]:
        insights.append("G30 improves cumulative equity quality (net + MDD) over full history.")
    else:
        insights.append("G30 does not clearly improve equity quality on full-history replay.")
    if not regime_delta.empty:
        best = regime_delta.iloc[0]
        insights.append(
            f"Largest G30 benefit in {best['split_type']}={best['regime']} "
            f"(Δpnl={best['net_pnl_delta']:.6f})."
        )
    insights.append(
        f"Monthly G30 win rate vs baseline: {narrative['monthly_g30_win_pct']:.1%}; "
        f"structurally_robust={narrative['structurally_robust']}."
    )
    insights.append(
        f"Low-trade environment diagnosis (30d): {low_trade.get('diagnosis', 'unknown')} "
        f"(recent H8 pass {low_trade.get('recent_h8_pass_ratio', 0):.1%})."
    )
    insights.append(f"Recommended path: {narrative['recommended_action']} (grade {narrative['grade']}).")
    return insights[:5]


def main() -> None:
    args = _parse_args()
    df = load_ohlcv()
    if df is None or df.empty:
        raise SystemExit("OHLCV load failed")

    ticks, proba_meta = simulate_signals_paper(df)
    if not ticks:
        raise SystemExit("No ticks")

    prod_m, prod_tdf = _simulate_variant(ticks, BASELINE, args.position_size, args.fee_rate, args.slippage_rate)
    sg_m, sg_tdf = _simulate_variant(ticks, G30, args.position_size, args.fee_rate, args.slippage_rate)

    prod_tr = _enrich_trades(prod_tdf, ticks)
    sg_tr = _enrich_trades(sg_tdf, ticks)
    prod_curve = _equity_curve(prod_tr)
    sg_curve = _equity_curve(sg_tr)
    prod_daily = _daily_equity(prod_curve)
    sg_daily = _daily_equity(sg_curve)

    prod_core = _core_metrics(prod_tr, prod_curve, prod_m)
    sg_core = _core_metrics(sg_tr, sg_curve, sg_m)
    prod_eq_q = _equity_quality(prod_curve, prod_daily)
    sg_eq_q = _equity_quality(sg_curve, sg_daily)

    prod_monthly = _period_returns(prod_tr, "M")
    sg_monthly = _period_returns(sg_tr, "M")
    prod_quarterly = _period_returns(prod_tr, "Q")
    sg_quarterly = _period_returns(sg_tr, "Q")
    monthly_cons = _monthly_consistency(prod_monthly, sg_monthly)
    quarterly_cons = _quarterly_consistency(prod_quarterly, sg_quarterly)

    prod_rb = _regime_breakdown(prod_tr)
    sg_rb = _regime_breakdown(sg_tr)
    regime_delta = _regime_delta(prod_rb, sg_rb)

    prod_loss = _loss_cluster_stats(prod_tr)
    sg_loss = _loss_cluster_stats(sg_tr)
    prod_roll = _rolling_stability(prod_tr, args.rolling_window_trades)
    sg_roll = _rolling_stability(sg_tr, args.rolling_window_trades)
    prod_conc = _profit_concentration(prod_tr)
    sg_conc = _profit_concentration(sg_tr)
    low_trade = _low_trade_diagnosis(ticks)

    grade, narrative = _structural_verdict(
        prod_core, sg_core, monthly_cons, quarterly_cons, regime_delta, prod_conc, sg_conc
    )
    insights = _insights(narrative, regime_delta, low_trade)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"h8_softgate_full_history_{ts}.csv"
    md_path = OUT_DIR / f"h8_softgate_full_history_{ts}.md"

    sections: List[pd.DataFrame] = []
    sections.append(pd.DataFrame([{**prod_core, "variant": "Baseline"}]).assign(section="phase1_core"))
    sections.append(pd.DataFrame([{**sg_core, "variant": "G30"}]).assign(section="phase1_core"))
    sections.append(pd.DataFrame([{**prod_eq_q, "variant": "Baseline"}]).assign(section="phase2_equity_quality"))
    sections.append(pd.DataFrame([{**sg_eq_q, "variant": "G30"}]).assign(section="phase2_equity_quality"))
    if not regime_delta.empty:
        sections.append(regime_delta.assign(section="phase3_regime_delta"))
    sections.append(
        pd.DataFrame([{**prod_loss, "variant": "Baseline"}, {**sg_loss, "variant": "G30"}]).assign(
            section="phase4_loss_clusters"
        )
    )
    sections.append(
        pd.DataFrame([{**prod_roll, "variant": "Baseline"}, {**sg_roll, "variant": "G30"}]).assign(
            section="phase5_rolling_stability"
        )
    )
    sections.append(
        pd.DataFrame([{**prod_conc, "variant": "Baseline"}, {**sg_conc, "variant": "G30"}]).assign(
            section="phase6_profit_concentration"
        )
    )
    sections.append(pd.DataFrame([narrative]).assign(section="phase7_verdict"))
    sections.append(pd.DataFrame([low_trade]).assign(section="low_trade_diagnosis"))
    sections.append(
        pd.DataFrame(
            [
                {"metric": k, "production": prod_core.get(k), "g30": sg_core.get(k)}
                for k in [
                    "trades",
                    "win_rate",
                    "expectancy",
                    "profit_factor",
                    "net_return",
                    "MDD",
                    "sharpe_proxy",
                    "calmar_proxy",
                ]
            ]
        ).assign(section="structural_comparison")
    )
    pd.concat(sections, ignore_index=True, sort=False).to_csv(csv_path, index=False)

    md_lines = [
        "# H8 SoftGate G30 Full History Structural Validation",
        "",
        "Diagnostics replay only — no deployment.",
        "",
        "## Structural Comparison",
        f"- Baseline trades/net/MDD: {prod_core['trades']} / {prod_core['net_return']:.6f} / {prod_core['MDD']:.6f}",
        f"- G30 trades/net/MDD: {sg_core['trades']} / {sg_core['net_return']:.6f} / {sg_core['MDD']:.6f}",
        f"- Trade preservation: {sg_core['trades'] / max(prod_core['trades'], 1):.2%}",
        "",
        "## Equity Quality",
        f"- Baseline final equity: {prod_core['final_equity']:.6f}",
        f"- G30 final equity: {sg_core['final_equity']:.6f}",
        f"- MDD delta (G30 - Prod): {narrative['mdd_delta']:.6f}",
        "",
        "## Regime Benefit (top)",
    ]
    if not regime_delta.empty:
        for _, r in regime_delta.head(5).iterrows():
            md_lines.append(
                f"- {r['split_type']}/{r['regime']}: Δpnl={r['net_pnl_delta']:.6f}, "
                f"Δexp={r['expectancy_delta']:.6f}"
            )
    md_lines += [
        "",
        "## Loss Clusters",
        f"- Prod worst streak / RFE / loss_cluster: {prod_loss['worst_losing_streak']} / "
        f"{prod_loss['RFE_count']} / {prod_loss['loss_cluster_trades']}",
        f"- G30 worst streak / RFE / loss_cluster: {sg_loss['worst_losing_streak']} / "
        f"{sg_loss['RFE_count']} / {sg_loss['loss_cluster_trades']}",
        "",
        "## Stability",
        f"- Monthly G30 better: {monthly_cons['months_g30_better']}/{monthly_cons['months_total']} "
        f"({monthly_cons['monthly_consistency_pct']:.1%})",
        f"- Quarterly G30 better: {quarterly_cons['quarters_g30_better']}/{quarterly_cons['quarters_total']} "
        f"({quarterly_cons['quarterly_consistency_pct']:.1%})",
        "",
        "## Profit Concentration",
        f"- Prod top1/5/10%: {prod_conc['top_1pct']:.1%} / {prod_conc['top_5pct']:.1%} / {prod_conc['top_10pct']:.1%}",
        f"- G30 top1/5/10%: {sg_conc['top_1pct']:.1%} / {sg_conc['top_5pct']:.1%} / {sg_conc['top_10pct']:.1%}",
        "",
        f"## FINAL VERDICT: Grade {grade}",
        f"- Recommended: {narrative['recommended_action']}",
        "",
        "## Strategic Insights",
    ]
    for i, ins in enumerate(insights, 1):
        md_lines.append(f"{i}. {ins}")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # print-only final block
    print("=== baseline vs G30 structural comparison ===")
    print(
        f"trades={prod_core['trades']}/{sg_core['trades']} "
        f"net={prod_core['net_return']:.6f}/{sg_core['net_return']:.6f} "
        f"MDD={prod_core['MDD']:.6f}/{sg_core['MDD']:.6f} "
        f"final_eq={prod_core['final_equity']:.6f}/{sg_core['final_equity']:.6f}"
    )
    print(f"equity_quality_improved: {narrative['equity_quality_improved']}")
    print(f"regimes_benefited_most: {narrative['primary_help_regime']}")
    print(f"structurally_robust: {narrative['structurally_robust']} (grade={grade})")
    print(f"low_trade_environment: {low_trade.get('diagnosis', 'unknown')}")
    paper_ok = grade in ("A", "B") and narrative["equity_quality_improved"]
    print(f"paper_candidate_promotion: {'yes' if paper_ok else 'no — continue monitor'}")
    print(f"FINAL VERDICT: Grade {grade}")
    print("top 5 strategic insights:")
    for i, ins in enumerate(insights, 1):
        print(f"  {i}. {ins}")
    print(f"csv: {csv_path}")
    print(f"md: {md_path}")


if __name__ == "__main__":
    main()
