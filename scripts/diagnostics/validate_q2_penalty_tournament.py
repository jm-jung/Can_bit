"""
Q2 penalty tournament — fix false high-score inflation without hard blocks.

Diagnostics only. Base: Q2 score + M3 mapping.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from scripts.diagnostics.validate_h8_soft_risk_gate import (
    Variant,
    _entropy,
    _expectancy,
    _is_loss_cluster_trade,
    _simulate_variant,
)
from scripts.diagnostics.validate_hybrid_gate_replay import HYBRID_CANDIDATES, _simulate_hybrid
from scripts.diagnostics.validate_quality_score_replay import (
    FEE_RATE,
    POSITION_SIZE,
    SLIPPAGE_RATE,
    _risk_routing,
    _simulate_quality,
    map_m3,
    score_q2,
    _clamp,
)
from scripts.run_daily_paper import load_ohlcv, simulate_signals_paper


OUT_DIR = Path("data/diagnostics/q2_penalty")
BASELINE = Variant("Baseline", fail_scale=1.0, hard_block=False)
G30 = Variant("Variant_G30", fail_scale=0.30, hard_block=False)
HYBRID_A = HYBRID_CANDIDATES[0]


def _profit_factor(vals: List[float]) -> float:
    w = sum(x for x in vals if x > 0)
    l = abs(sum(x for x in vals if x < 0))
    return w / l if l > 0 else 0.0


def _ohlcv_features(df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    px = pd.to_numeric(df["close"], errors="coerce")
    ema = px.ewm(span=20, adjust=False).mean()
    rr = px.pct_change()
    rr24 = px.pct_change(24)
    rv = rr.rolling(24).std()
    ema_dist = ((px - ema) / ema.replace(0, np.nan)).abs()
    ts = rr.rolling(12).mean().abs()
    ema_p80 = float(ema_dist.quantile(0.80))
    ts_p30 = float(ts.quantile(0.30))
    out: Dict[int, Dict[str, Any]] = {}
    for i in range(len(df)):
        out[i] = {
            "recent_return_24": float(rr24.iloc[i]) if pd.notna(rr24.iloc[i]) else float("nan"),
            "ema_distance": float(ema_dist.iloc[i]) if pd.notna(ema_dist.iloc[i]) else float("nan"),
            "realized_vol": float(rv.iloc[i]) if pd.notna(rv.iloc[i]) else float("nan"),
            "trend_strength": float(ts.iloc[i]) if pd.notna(ts.iloc[i]) else float("nan"),
            "_ema_p80": ema_p80,
            "_ts_p30": ts_p30,
        }
    return out


def apply_penalties(t: Dict[str, Any], fm: Dict[int, Dict[str, Any]], active: Set[str]) -> float:
    s = score_q2(t, fm)
    ent = _entropy(t)
    vol = str(t.get("vol_bucket") or "")
    trend = str(t.get("trend_label") or "")
    is_long = t.get("signal") == "LONG"
    idx = int(t.get("df_idx", -1))
    ex = fm.get(idx, {})
    rr24 = float(ex.get("recent_return_24", 0) or 0) if pd.notna(ex.get("recent_return_24")) else 0.0
    ema_d = float(ex.get("ema_distance", 0) or 0) if pd.notna(ex.get("ema_distance")) else 0.0
    ts = float(ex.get("trend_strength", 0) or 0) if pd.notna(ex.get("trend_strength")) else 0.0

    if "A" in active and vol == "high" and ent > 0.95:
        s -= 0.20
    if "B" in active and is_long and trend == "up" and vol == "high":
        s -= 0.15
    if "C" in active and is_long and vol == "high":
        s -= 0.25
    if "D" in active and trend == "up" and vol == "high" and ent <= 0.90:
        s -= 0.20
    if "E" in active and rr24 > 0 and vol == "high":
        s -= 0.10
    if "F" in active and ema_d > float(ex.get("_ema_p80", 999)):
        s -= 0.15
    if "G" in active and ts < float(ex.get("_ts_p30", 0)):
        s -= 0.20
    if "H" in active and vol == "high":
        if ent <= 0.90:
            s -= 0.05
        elif ent <= 0.96:
            s -= 0.15
        else:
            s -= 0.25
    if "I" in active and is_long and vol == "high" and rr24 > 0 and trend == "up":
        s -= 0.25
    if "J" in active and is_long and vol == "high" and ent > 0.92 and rr24 > 0.005:
        s -= 0.30
    return _clamp(s)


def _calibration_coarse(tdf: pd.DataFrame) -> Tuple[bool, float, pd.DataFrame]:
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    rows = []
    exps = []
    for lo, hi in bins:
        sub = tdf[(tdf["quality_score"] >= lo) & (tdf["quality_score"] < hi)] if not tdf.empty else pd.DataFrame()
        vals = sub["scaled_return"].tolist() if not sub.empty else []
        exp = _expectancy(vals)
        if len(sub):
            exps.append(exp)
        rows.append({
            "bucket": f"{lo:.1f}~{hi:.1f}",
            "trade_count": len(sub),
            "expectancy": exp,
            "rfe_ratio": float((sub["exit_reason"] == "risk_force_exit").mean()) if len(sub) else 0.0,
        })
    mono = all(exps[i] <= exps[i + 1] for i in range(len(exps) - 1)) if len(exps) >= 2 else False
    smooth = float(pstdev(exps)) if len(exps) > 1 else 0.0
    return mono, smooth, pd.DataFrame(rows)


def _false_high_stats(tdf: pd.DataFrame) -> Dict[str, Any]:
    if tdf.empty:
        return {"false_high_count": 0, "false_high_expectancy": 0.0, "false_high_rfe_rate": 0.0}
    fh = tdf[(tdf["quality_score"] >= 0.80) & (tdf["scaled_return"] < 0)]
    return {
        "false_high_count": int(len(fh)),
        "false_high_expectancy": _expectancy(fh["scaled_return"].tolist()) if len(fh) else 0.0,
        "false_high_rfe_rate": float((fh["exit_reason"] == "risk_force_exit").mean()) if len(fh) else 0.0,
        "false_high_net_sum": float(fh["scaled_return"].sum()) if len(fh) else 0.0,
    }


def _loss_cluster_count(tdf: pd.DataFrame, ticks: List[Dict[str, Any]]) -> int:
    if tdf.empty:
        return 0
    n = 0
    for _, tr in tdf.iterrows():
        if _is_loss_cluster_trade(tr.to_dict(), ticks[int(tr["entry_idx"])]):
            n += 1
    return n


def run_candidate(
    name: str,
    ticks: List[Dict[str, Any]],
    fm: Dict[int, Dict[str, Any]],
    active: Set[str],
    prod_trades: int,
    baseline_m: Dict[str, Any],
) -> Dict[str, Any]:
    active_set = set(active)

    def sc(t: Dict[str, Any], f: Dict[int, Dict[str, Any]]) -> float:
        return apply_penalties(t, f, active_set)

    m, tdf, _ = _simulate_quality(ticks, sc, map_m3, fm)
    vals = tdf["scaled_return"].tolist() if not tdf.empty else []
    rr = _risk_routing(tdf, ticks)
    mono, cal_smooth, cal_df = _calibration_coarse(tdf)
    fh = _false_high_stats(tdf)
    routing = rr["avg_scale_winners"] > rr["avg_scale_losers"] > rr["avg_scale_rfe"] if m["trades"] > 0 else False
    pres = m["trades"] / max(prod_trades, 1)
    overfiltered = pres < 0.85

    cal_improved = mono and (not baseline_m.get("calibration_monotonic", False) or fh["false_high_count"] < baseline_m.get("false_high_count", 999))
    mdd_improved = m["MDD"] >= baseline_m["MDD"]

    if routing and cal_improved and pres >= 0.85 and mdd_improved:
        grade = "A"
    elif (routing or fh["false_high_count"] < baseline_m.get("false_high_count", 999)) and pres >= 0.85:
        grade = "B"
    elif pres >= 0.70:
        grade = "C"
    else:
        grade = "D"

    routing_why = ""
    if not routing and m["trades"] > 0:
        routing_why = f"win={rr['avg_scale_winners']:.3f} loss={rr['avg_scale_losers']:.3f} rfe={rr['avg_scale_rfe']:.3f}"

    return {
        "candidate": name,
        "penalties": "+".join(sorted(active_set)) if active_set else "baseline",
        "trades": m["trades"],
        "trade_preservation": pres,
        "overfiltered": overfiltered,
        "expectancy": m["expectancy"],
        "net_return": m["net_return"],
        "profit_factor": m["profit_factor"],
        "MDD": m["MDD"],
        "rfe_count": m["risk_force_exit_count"],
        "ks_triggered": bool(m["KS_triggered"]),
        "loss_cluster_count": _loss_cluster_count(tdf, ticks),
        "calibration_monotonic": mono,
        "calibration_smoothness": cal_smooth,
        "routing_valid": routing,
        "routing_why": routing_why,
        "avg_scale_winners": rr["avg_scale_winners"],
        "avg_scale_losers": rr["avg_scale_losers"],
        "avg_scale_rfe": rr["avg_scale_rfe"],
        **fh,
        "grade": grade,
        "mdd_delta_vs_baseline": m["MDD"] - baseline_m["MDD"],
        "net_delta_vs_baseline": m["net_return"] - baseline_m["net_return"],
        "false_high_delta": fh["false_high_count"] - baseline_m.get("false_high_count", 0),
        "calibration_table": cal_df.to_dict(orient="records"),
    }


def _grade_score(row: pd.Series) -> float:
    sc = 0.0
    if row["routing_valid"]:
        sc += 3.0
    if row["calibration_monotonic"]:
        sc += 2.0
    sc += max(row["trade_preservation"], 0) * 2.0
    sc += max(row["mdd_delta_vs_baseline"], 0) * 50.0
    sc += max(row["net_delta_vs_baseline"], 0) * 10.0
    sc -= max(-row["false_high_delta"], 0) * 0.5
    if row["overfiltered"]:
        sc -= 2.0
    return sc


def main() -> None:
    df = load_ohlcv()
    if df is None or df.empty:
        raise SystemExit("OHLCV load failed")
    fm = _ohlcv_features(df)
    ticks, _ = simulate_signals_paper(df)
    if not ticks:
        raise SystemExit("No ticks")

    _, prod_tdf = _simulate_variant(ticks, BASELINE, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    prod_n = len(prod_tdf)

    g30_m, g30_tdf = _simulate_variant(ticks, G30, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    hyb_m, hyb_tdf, _ = _simulate_hybrid(ticks, HYBRID_A, fm, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)

    candidates: List[Tuple[str, Set[str]]] = [
        ("Q2_M3_baseline", set()),
        ("Penalty_A", {"A"}),
        ("Penalty_B", {"B"}),
        ("Penalty_C", {"C"}),
        ("Penalty_D", {"D"}),
        ("Penalty_E", {"E"}),
        ("Penalty_F", {"F"}),
        ("Penalty_G", {"G"}),
        ("Penalty_H", {"H"}),
        ("Penalty_I", {"I"}),
        ("Penalty_J", {"J"}),
        ("Combo_A+B", {"A", "B"}),
        ("Combo_B+D", {"B", "D"}),
        ("Combo_B+I", {"B", "I"}),
        ("Combo_D+I", {"D", "I"}),
        ("Combo_H+I", {"H", "I"}),
        ("Combo_B+D+I", {"B", "D", "I"}),
    ]

    rows: List[Dict[str, Any]] = []
    baseline_row: Optional[Dict[str, Any]] = None
    for name, active in candidates:
        if baseline_row is None:
            r = run_candidate(name, ticks, fm, active, prod_n, {"MDD": -1.0, "net_return": 0.0, "calibration_monotonic": False, "false_high_count": 999})
            baseline_row = r
            rows.append(r)
        else:
            rows.append(run_candidate(name, ticks, fm, active, prod_n, baseline_row))

    summary = pd.DataFrame(rows)
    summary["tournament_score"] = summary.apply(_grade_score, axis=1)
    summary = summary.sort_values("tournament_score", ascending=False).reset_index(drop=True)

    best = summary.iloc[0]
    combos = summary[summary["candidate"].str.startswith("Combo_")]
    singles = summary[~summary["candidate"].str.startswith("Combo_") & (summary["candidate"] != "Q2_M3_baseline")]
    best_single = singles.iloc[0] if not singles.empty else best
    best_combo = combos.iloc[0] if not combos.empty else best

    best_tradeoff = summary.sort_values(
        ["routing_valid", "trade_preservation", "mdd_delta_vs_baseline", "net_delta_vs_baseline"],
        ascending=[False, False, False, False],
    ).iloc[0]
    best_pres = summary.sort_values("trade_preservation", ascending=False).iloc[0]

    any_routing = bool(summary["routing_valid"].any())
    cal_improved_any = bool((summary["calibration_monotonic"] & (summary["candidate"] != "Q2_M3_baseline")).any()) or (
        summary.loc[summary["candidate"] != "Q2_M3_baseline", "false_high_count"].min() < baseline_row["false_high_count"]
    )

    q2_best = best
    vs_g30 = q2_best["net_return"] > g30_m["net_return"] and q2_best["MDD"] >= g30_m["MDD"]
    vs_hybrid = q2_best["net_return"] > hyb_m["net_return"] and q2_best["MDD"] >= hyb_m["MDD"]

    viable_count = int((summary["grade"].isin(["A", "B"])).sum())
    if viable_count >= 3 and any_routing:
        arch = "viable"
    elif viable_count >= 1:
        arch = "promising"
    elif summary["trade_preservation"].max() < 0.85:
        arch = "unstable"
    else:
        arch = "invalid"

    if best["grade"] == "A" and best["trade_preservation"] >= 0.90:
        rec = "replace Q2_M3"
    elif best["grade"] in ("A", "B"):
        rec = "become monitor candidate"
    else:
        rec = "remain forensic only"

    final_grade = best["grade"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"q2_penalty_tournament_{ts}.csv"
    md_path = OUT_DIR / f"q2_penalty_tournament_{ts}.md"

    export = summary.drop(columns=["calibration_table"], errors="ignore")
    export.to_csv(csv_path, index=False)

    md = [
        "# Q2 Penalty Tournament",
        f"- best: **{best['candidate']}** (grade {best['grade']})",
        f"- routing_valid any: {any_routing}",
        f"- recommendation: {rec}",
        "",
        "## Top 5",
    ]
    for _, r in summary.head(5).iterrows():
        md.append(
            f"- {r['candidate']}: grade={r['grade']} routing={r['routing_valid']} "
            f"mono={r['calibration_monotonic']} pres={r['trade_preservation']:.1%} "
            f"false_high={int(r['false_high_count'])} MDD={r['MDD']:.6f}"
        )
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    insights = [
        f"Best penalty: {best_single['candidate']} (routing={best_single['routing_valid']}, false_high={int(best_single['false_high_count'])}).",
        f"Best combo: {best_combo['candidate']} (grade={best_combo['grade']}).",
        f"Penalty_D targets false-high signature (trend_up+high_vol+low entropy).",
        f"Baseline false_high_count={int(baseline_row['false_high_count'])} → best={int(best['false_high_count'])}.",
        f"routing_valid achieved: {any_routing} (candidates: {', '.join(summary[summary['routing_valid']]['candidate'].tolist())}).",
        f"Trade preservation best: {best_pres['candidate']} at {best_pres['trade_preservation']:.1%}.",
        f"vs G30 superior: {vs_g30}; vs Hybrid_A superior: {vs_hybrid}.",
        f"Architecture status: {arch}.",
        f"Recommendation: {rec}.",
        "Combo B+D+I penalizes LONG trend_up high_vol without hard block.",
    ]

    print(f"best penalty candidate: {best_single['candidate']}")
    print(f"best combo candidate: {best_combo['candidate']}")
    print(f"whether routing_valid became TRUE: {any_routing}")
    print(f"whether calibration improved: {cal_improved_any}")
    print(f"best tradeoff candidate: {best_tradeoff['candidate']}")
    print(f"best preservation candidate: {best_pres['candidate']} ({best_pres['trade_preservation']:.1%})")
    print(f"Q2 architecture now: {arch}")
    print(f"continuous routing superior to G30: {vs_g30}")
    print(f"continuous routing superior to Hybrid_A: {vs_hybrid}")
    print(f"penalty system should: {rec}")
    print(f"FINAL VERDICT: Grade {final_grade}")
    print("top 10 strategic insights:")
    for i, s in enumerate(insights[:10], 1):
        print(f"  {i}. {s}")
    print(f"csv: {csv_path}")
    print(f"md: {md_path}")


if __name__ == "__main__":
    main()
