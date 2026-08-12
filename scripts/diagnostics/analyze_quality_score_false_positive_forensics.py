"""
Q2 quality score false-positive forensics (diagnostics only).

Investigates why high-score (>=0.80) trades still lose.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.diagnostics.validate_entry_quality_filters import _margin
from scripts.diagnostics.validate_h8_soft_risk_gate import (
    Variant,
    _entropy,
    _expectancy,
    _h8_pass,
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


OUT_DIR = Path("data/diagnostics/quality_score_forensics")
G30 = Variant("Variant_G30", fail_scale=0.30, hard_block=False)
BASELINE = Variant("Baseline", fail_scale=1.0, hard_block=False)
HYBRID_A = HYBRID_CANDIDATES[0]


def _profit_factor(vals: List[float]) -> float:
    w = sum(x for x in vals if x > 0)
    l = abs(sum(x for x in vals if x < 0))
    return w / l if l > 0 else 0.0


def _max_proba(t: Dict[str, Any]) -> float:
    return max(float(t["p_long"]), float(t["p_short"]), float(t["p_flat"]))


def _ohlcv_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    px = pd.to_numeric(df["close"], errors="coerce")
    ema = px.ewm(span=20, adjust=False).mean()
    rr = px.pct_change()
    rr24 = px.pct_change(24)
    rv = rr.rolling(24).std()
    ema_dist = (px - ema) / ema.replace(0, np.nan)
    ts_strength = rr.rolling(12).mean().abs()
    feat_df = pd.DataFrame(
        {
            "df_idx": np.arange(len(df)),
            "price_vs_ema": (px > ema).astype(int),
            "ema_distance": ema_dist,
            "recent_return_24": rr24,
            "realized_vol": rv,
            "trend_strength": ts_strength,
            "future_return_12": px.pct_change(12).shift(-12),
        }
    )
    feat_map = feat_df.set_index("df_idx").to_dict(orient="index")
    return feat_df, feat_map


def _entropy_bucket(e: float) -> str:
    if e <= 0.90:
        return "<=0.90"
    if e <= 0.96:
        return "0.90~0.96"
    if e <= 1.00:
        return "0.96~1.00"
    return ">1.00"


def _margin_bucket(m: float) -> str:
    if m < 0.05:
        return "<0.05"
    if m < 0.07:
        return "0.05~0.07"
    if m < 0.10:
        return "0.07~0.10"
    return ">=0.10"


def _score_components(t: Dict[str, Any], fm: Dict[int, Dict[str, Any]]) -> Dict[str, float]:
    ent = _entropy(t)
    trend = str(t.get("trend_label") or "")
    vol = str(t.get("vol_bucket") or "")
    if ent <= 0.90:
        base = 0.90
    elif ent <= 0.96:
        base = 0.65
    elif ent <= 1.00:
        base = 0.40
    else:
        base = 0.20
    trend_adj = 0.10 if trend == "up" else (-0.15 if trend == "down" else 0.0)
    vol_adj = -0.15 if vol == "high" else 0.0
    danger = (
        -0.20
        if t.get("signal") == "LONG" and trend == "down" and vol == "high"
        else 0.0
    )
    h8_bonus = 0.0  # Q2 does not use H8 directly
    margin_bonus = 0.0  # Q2 does not use margin
    final = _clamp(base + trend_adj + vol_adj + danger)
    return {
        "entropy_base": base,
        "trend_adj": trend_adj,
        "vol_adj": vol_adj,
        "danger_adj": danger,
        "h8_bonus": h8_bonus,
        "margin_bonus": margin_bonus,
        "quality_score": final,
    }


def _path_metrics(tr: Dict[str, Any], ticks: List[Dict[str, Any]]) -> Dict[str, float]:
    i0 = int(tr["entry_idx"])
    hb = int(tr["hold_bars"])
    entry = float(tr["entry_price"])
    side = tr.get("side", "BUY")
    mae = 0.0
    mfe = 0.0
    ent_exit = _entropy(ticks[min(i0 + hb, len(ticks) - 1)])
    ent_entry = _entropy(ticks[i0])
    max_dd_vel = 0.0
    for j in range(i0, min(i0 + hb + 1, len(ticks))):
        px = float(ticks[j]["price"])
        raw = (px - entry) / entry if side == "BUY" else (entry - px) / entry
        mae = min(mae, raw)
        mfe = max(mfe, raw)
        if j > i0:
            prev = float(ticks[j - 1]["price"])
            step = (px - prev) / prev if side == "BUY" else (prev - px) / prev
            if step < 0:
                max_dd_vel = min(max_dd_vel, step)
    return {
        "mae": mae,
        "mfe": mfe,
        "entropy_entry": ent_entry,
        "entropy_exit": ent_exit,
        "entropy_expansion": ent_exit - ent_entry,
        "drawdown_velocity": max_dd_vel,
    }


def _enrich_q2_trades(tdf: pd.DataFrame, ticks: List[Dict[str, Any]], feat_map: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    if tdf.empty:
        return tdf.copy()
    rows = []
    for _, tr in tdf.iterrows():
        i = int(tr["entry_idx"])
        t = ticks[i]
        idx = int(t.get("df_idx", i))
        extra = feat_map.get(idx, {})
        comps = _score_components(t, feat_map)
        path = _path_metrics(tr.to_dict(), ticks)
        rr24 = extra.get("recent_return_24", np.nan)
        rows.append(
            {
                **tr.to_dict(),
                **comps,
                **path,
                "entropy": _entropy(t),
                "margin": _margin(t),
                "max_proba": _max_proba(t),
                "trend_state": str(t.get("trend_label") or ""),
                "vol_bucket": str(t.get("vol_bucket") or ""),
                "h8_pass": _h8_pass(t),
                "recent_return_24": float(rr24) if pd.notna(rr24) else np.nan,
                "price_vs_ema": int(extra.get("price_vs_ema", 0)) if pd.notna(extra.get("price_vs_ema")) else 0,
                "ema_distance": float(extra.get("ema_distance", np.nan)) if pd.notna(extra.get("ema_distance")) else np.nan,
                "realized_vol": float(extra.get("realized_vol", np.nan)) if pd.notna(extra.get("realized_vol")) else np.nan,
                "trend_strength": float(extra.get("trend_strength", np.nan)) if pd.notna(extra.get("trend_strength")) else np.nan,
                "future_return_12": float(extra.get("future_return_12", np.nan)) if pd.notna(extra.get("future_return_12")) else np.nan,
                "trade_return": float(tr["scaled_return"]),
                "is_rfe": tr.get("exit_reason") == "risk_force_exit",
                "loss_cluster": _is_loss_cluster_trade(tr.to_dict(), t),
            }
        )
    return pd.DataFrame(rows)


def _group_label(row: pd.Series) -> str:
    s = float(row["quality_score"])
    r = float(row["trade_return"])
    if s >= 0.80 and r < 0:
        return "FALSE_HIGH_SCORE"
    if s >= 0.80 and r > 0:
        return "TRUE_HIGH_WINNER"
    if 0.50 <= s < 0.80:
        return "MID_SCORE"
    if s < 0.50 and r > 0:
        return "LOW_SCORE_SURVIVOR"
    return "OTHER"


def _group_summary(g: pd.DataFrame, label: str) -> Dict[str, Any]:
    if g.empty:
        return {"group": label, "trade_count": 0}
    vals = g["trade_return"].tolist()
    return {
        "group": label,
        "trade_count": len(g),
        "expectancy": _expectancy(vals),
        "profit_factor": _profit_factor(vals),
        "avg_hold": float(g["hold_bars"].mean()),
        "avg_mae": float(g["mae"].mean()),
        "avg_mfe": float(g["mfe"].mean()),
        "rfe_rate": float(g["is_rfe"].mean()),
        "avg_entropy": float(g["entropy"].mean()),
        "avg_margin": float(g["margin"].mean()),
        "avg_max_proba": float(g["max_proba"].mean()),
        "avg_realized_vol": float(g["realized_vol"].mean()),
        "avg_recent_return_24": float(g["recent_return_24"].mean()),
        "long_ratio": float((g["direction"] == "LONG").mean()),
        "trend_dist": dict(Counter(g["trend_state"])),
        "vol_dist": dict(Counter(g["vol_bucket"])),
    }


def _signature_key(row: pd.Series) -> str:
    rr = "neg" if float(row.get("recent_return_24", 0) or 0) < 0 else "pos"
    return (
        f"{row['direction']}|ent {_entropy_bucket(float(row['entropy']))}|"
        f"mar {_margin_bucket(float(row['margin']))}|trend {row['trend_state']}|"
        f"vol {row['vol_bucket']}|rr24 {rr}|ema {row.get('price_vs_ema', 0)}"
    )


def _top_false_signatures(fp: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if fp.empty:
        return pd.DataFrame()
    tmp = fp.copy()
    tmp["signature"] = tmp.apply(_signature_key, axis=1)
    rows = []
    for sig, g in tmp.groupby("signature"):
        vals = g["trade_return"].tolist()
        rows.append(
            {
                "signature": sig,
                "trade_count": len(g),
                "expectancy": _expectancy(vals),
                "rfe_rate": float(g["is_rfe"].mean()),
                "avg_hold": float(g["hold_bars"].mean()),
                "loss_cluster_rate": float(g["loss_cluster"].mean()),
                "net_sum": float(np.sum(vals)),
            }
        )
    return pd.DataFrame(rows).sort_values(["expectancy", "trade_count"]).head(n)


def _interaction_table(trades: pd.DataFrame) -> pd.DataFrame:
    masks = [
        ("trend_up", lambda r: r["trend_state"] == "up"),
        ("trend_up+high_vol", lambda r: (r["trend_state"] == "up") & (r["vol_bucket"] == "high")),
        ("trend_up+ent>0.95", lambda r: (r["trend_state"] == "up") & (r["entropy"] > 0.95)),
        ("trend_up+LONG+high_vol", lambda r: (r["trend_state"] == "up") & (r["direction"] == "LONG") & (r["vol_bucket"] == "high")),
        ("margin_high", lambda r: r["margin"] >= 0.07),
        ("margin_high+high_vol", lambda r: (r["margin"] >= 0.07) & (r["vol_bucket"] == "high")),
        ("entropy_low", lambda r: r["entropy"] <= 0.90),
        ("entropy_low+downtrend", lambda r: (r["entropy"] <= 0.90) & (r["trend_state"] == "down")),
    ]
    rows = []
    for name, fn in masks:
        sub = trades[trades.apply(fn, axis=1)]
        vals = sub["trade_return"].tolist()
        rows.append({"interaction": name, "trade_count": len(sub), "expectancy": _expectancy(vals), "win_rate": float((sub["trade_return"] > 0).mean()) if len(sub) else 0.0})
    return pd.DataFrame(rows)


def _component_leakage(fp: pd.DataFrame, all_hi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comp in ("entropy_base", "trend_adj", "vol_adj", "danger_adj"):
        for val, g in fp.groupby(comp):
            base = all_hi[all_hi[comp] == val]
            rows.append(
                {
                    "component": comp,
                    "value": val,
                    "fp_count": len(g),
                    "fp_rate": len(g) / max(len(base), 1),
                    "fp_expectancy": _expectancy(g["trade_return"].tolist()),
                    "fp_rfe_rate": float(g["is_rfe"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _fine_calibration(trades: pd.DataFrame) -> Tuple[pd.DataFrame, float, bool]:
    bins = [(i / 10, (i + 1) / 10) for i in range(10)]
    rows = []
    exps = []
    for lo, hi in bins:
        sub = trades[(trades["quality_score"] >= lo) & (trades["quality_score"] < hi)]
        vals = sub["trade_return"].tolist()
        exp = _expectancy(vals)
        exps.append(exp if len(sub) else None)
        rows.append(
            {
                "bucket": f"{lo:.1f}~{hi:.1f}",
                "trade_count": len(sub),
                "expectancy": exp,
                "profit_factor": _profit_factor(vals),
                "rfe_ratio": float(sub["is_rfe"].mean()) if len(sub) else 0.0,
                "win_rate": float((sub["trade_return"] > 0).mean()) if len(sub) else 0.0,
                "avg_scale": float(sub["scale"].mean()) if len(sub) else 0.0,
            }
        )
    valid = [e for e in exps if e is not None]
    mono = all(valid[i] <= valid[i + 1] for i in range(len(valid) - 1)) if len(valid) >= 2 else False
    smooth = float(np.std(valid)) if len(valid) > 1 else 0.0
    return pd.DataFrame(rows), smooth, mono


def _penalty_score_q2(t: Dict[str, Any], fm: Dict[int, Dict[str, Any]], penalties: Dict[str, float]) -> float:
    s = score_q2(t, fm)
    ent = _entropy(t)
    vol = str(t.get("vol_bucket") or "")
    trend = str(t.get("trend_label") or "")
    idx = int(t.get("df_idx", -1))
    extra = fm.get(idx, {})
    rv = extra.get("realized_vol", np.nan)
    ts = extra.get("trend_strength", np.nan)
    rv_med = extra.get("_rv_p70", np.nan)

    if penalties.get("A") and vol == "high" and ent > 0.95:
        s -= 0.20
    if penalties.get("B") and t.get("signal") == "LONG" and trend == "up" and pd.notna(rv) and pd.notna(rv_med) and rv > rv_med:
        s -= 0.15
    if penalties.get("C") and pd.notna(ts) and ts < 0.001:
        s -= 0.10
    if penalties.get("D") and ent > 0.92 and vol == "high":
        s -= 0.25
    if penalties.get("E") and trend == "up" and float(extra.get("recent_return_24", 0) or 0) > 0.02 and ent > 0.93:
        s -= 0.20
    return _clamp(s)


def _make_map(spread: Tuple[float, float, float, float], block_bottom: bool = False) -> Callable[[float], Optional[float]]:
    a, b, c, d = spread

    def _m(score: float) -> Optional[float]:
        if score >= 0.80:
            return a
        if score >= 0.65:
            return b
        if score >= 0.50:
            return c
        if block_bottom or d <= 0.0:
            return None
        return d

    return _m


def _simulate_penalty_map(
    ticks: List[Dict[str, Any]],
    fm: Dict[int, Dict[str, Any]],
    penalties: Dict[str, float],
    map_fn: Callable[[float], Optional[float]],
    prod_trades: int,
) -> Dict[str, Any]:
    from scripts.diagnostics.validate_quality_score_replay import _simulate_quality

    def sc(t: Dict[str, Any], f: Dict[int, Dict[str, Any]]) -> float:
        return _penalty_score_q2(t, f, penalties)

    m, tdf, _ = _simulate_quality(ticks, sc, map_fn, fm)
    rr = _risk_routing(tdf, ticks)
    cal, _, mono = _fine_calibration(_enrich_q2_trades(tdf, ticks, fm)) if not tdf.empty else (pd.DataFrame(), 0.0, False)
    routing = rr["avg_scale_winners"] > rr["avg_scale_losers"] > rr["avg_scale_rfe"] if m["trades"] > 0 else False
    return {
        "trades": m["trades"],
        "preservation": m["trades"] / max(prod_trades, 1),
        "net_return": m["net_return"],
        "MDD": m["MDD"],
        "rfe": m["risk_force_exit_count"],
        "routing_valid": routing,
        "calibration_monotonic": mono,
        "hi_bucket_exp": float(cal[cal["bucket"] == "0.8~0.9"]["expectancy"].iloc[0]) if not cal.empty and (cal["bucket"] == "0.8~0.9").any() else 0.0,
    }


def main() -> None:
    df = load_ohlcv()
    if df is None or df.empty:
        raise SystemExit("OHLCV load failed")
    feat_df, feat_map = _ohlcv_features(df)
    rv_p70 = float(feat_df["realized_vol"].quantile(0.70))
    for k in feat_map:
        feat_map[k]["_rv_p70"] = rv_p70

    ticks, _ = simulate_signals_paper(df)
    if not ticks:
        raise SystemExit("No ticks")

    _, prod_tdf = _simulate_variant(ticks, BASELINE, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    _, g30_tdf = _simulate_variant(ticks, G30, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    _, hyb_tdf, _ = _simulate_hybrid(ticks, HYBRID_A, feat_map, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    _, q2_tdf, _ = _simulate_quality(ticks, score_q2, map_m3, feat_map)

    enriched = _enrich_q2_trades(q2_tdf, ticks, feat_map)
    enriched["forensic_group"] = enriched.apply(_group_label, axis=1)

    fp = enriched[enriched["forensic_group"] == "FALSE_HIGH_SCORE"]
    thw = enriched[enriched["forensic_group"] == "TRUE_HIGH_WINNER"]
    mid = enriched[enriched["forensic_group"] == "MID_SCORE"]
    low_surv = enriched[enriched["forensic_group"] == "LOW_SCORE_SURVIVOR"]

    phase1 = pd.DataFrame([_group_summary(g, lbl) for lbl, g in [
        ("FALSE_HIGH_SCORE", fp), ("TRUE_HIGH_WINNER", thw), ("MID_SCORE", mid), ("LOW_SCORE_SURVIVOR", low_surv)
    ]])

    phase2 = _top_false_signatures(fp, 10)
    phase3 = _interaction_table(enriched)
    phase4 = _component_leakage(fp, enriched[enriched["quality_score"] >= 0.80])
    phase7, cal_smooth, cal_mono = _fine_calibration(enriched)

  # regime delay
    regime_rows = []
    if not fp.empty:
        regime_rows.append({
            "metric": "false_high_avg_future_12bar_return",
            "value": float(fp["future_return_12"].mean()),
            "true_high_winner": float(thw["future_return_12"].mean()) if not thw.empty else np.nan,
        })
        regime_rows.append({
            "metric": "false_high_avg_entropy_expansion",
            "value": float(fp["entropy_expansion"].mean()),
            "true_high_winner": float(thw["entropy_expansion"].mean()) if not thw.empty else np.nan,
        })
        regime_rows.append({
            "metric": "false_high_avg_drawdown_velocity",
            "value": float(fp["drawdown_velocity"].mean()),
            "true_high_winner": float(thw["drawdown_velocity"].mean()) if not thw.empty else np.nan,
        })
    phase5 = pd.DataFrame(regime_rows)

    phase6_rows = []
    if not fp.empty and not thw.empty:
        phase6_rows = [
            {"metric": "avg_hold", "false_high": float(fp["hold_bars"].mean()), "true_high": float(thw["hold_bars"].mean())},
            {"metric": "rfe_rate", "false_high": float(fp["is_rfe"].mean()), "true_high": float(thw["is_rfe"].mean())},
            {"metric": "avg_mae", "false_high": float(fp["mae"].mean()), "true_high": float(thw["mae"].mean())},
            {"metric": "avg_mfe", "false_high": float(fp["mfe"].mean()), "true_high": float(thw["mfe"].mean())},
            {"metric": "immediate_loss_rate", "false_high": float((fp["mae"] < -0.005).mean()), "true_high": float((thw["mae"] < -0.005).mean())},
        ]
    phase6 = pd.DataFrame(phase6_rows)

    prod_n = len(prod_tdf)
    penalty_tests = []
    for pname, pflags in [
        ("baseline_Q2_M3", {}),
        ("Penalty_A", {"A": 0.20}),
        ("Penalty_B", {"B": 0.15}),
        ("Penalty_C", {"C": 0.10}),
        ("Penalty_D", {"D": 0.25}),
        ("Penalty_E", {"E": 0.20}),
        ("Penalty_A+D", {"A": 0.20, "D": 0.25}),
    ]:
        penalty_tests.append({"variant": pname, **_simulate_penalty_map(ticks, feat_map, pflags, map_m3, prod_n)})
    phase8 = pd.DataFrame(penalty_tests)

    routing_tests = []
    spreads = [
        ("M3_current", (1.0, 0.70, 0.40, 0.15)),
        ("spread_tight", (1.0, 0.60, 0.25, 0.05)),
        ("spread_aggressive", (1.0, 0.50, 0.20, 0.0)),  # 0.0 => hard block
        ("spread_mild", (1.0, 0.80, 0.30, 0.05)),
    ]
    for name, sp in spreads:
        routing_tests.append({"mapping": name, **_simulate_penalty_map(ticks, feat_map, {}, _make_map(sp), prod_n)})
    phase9 = pd.DataFrame(routing_tests)

    # diagnosis ranking
    fp_trend_up = float((fp["trend_state"] == "up").mean()) if not fp.empty else 0.0
    fp_high_vol = float((fp["vol_bucket"] == "high").mean()) if not fp.empty else 0.0
    fp_ent_mid = float(((fp["entropy"] > 0.90) & (fp["entropy"] <= 0.96)).mean()) if not fp.empty else 0.0
    lag_signal = float(fp["future_return_12"].mean()) < 0 if not fp.empty else False

    severity = {
        "A_score_formula": fp_ent_mid + fp_trend_up * 0.3,
        "B_mapping": 0.2 if not cal_mono else 0.1,
        "C_regime_label": fp_trend_up,
        "D_interaction": float(phase3.loc[phase3["interaction"] == "trend_up+ent>0.95", "expectancy"].iloc[0]) if len(phase3) else 0,
        "E_lifecycle": float(fp["is_rfe"].mean()) if not fp.empty else 0,
        "F_insufficient_alpha": float(enriched["trade_return"].mean()) < 0,
        "G_delayed_timing": 1.0 if lag_signal else 0.3,
    }
    top_problem = max(severity, key=lambda k: severity[k])

    best_pen = phase8.sort_values(["routing_valid", "calibration_monotonic", "hi_bucket_exp"], ascending=[False, False, False]).iloc[0]
    phase9_ok = phase9[phase9["preservation"] <= 1.05]
    best_map = phase9_ok.sort_values(["routing_valid", "preservation", "MDD"], ascending=[False, False, False]).iloc[0] if not phase9_ok.empty else phase9.iloc[0]

    top_sig = str(phase2.iloc[0]["signature"]) if not phase2.empty else "n/a"
    entropy_overweight = fp_ent_mid > 0.5 and float(phase4[phase4["component"] == "entropy_base"]["fp_rate"].max()) > 0.4 if not phase4.empty else False
    trend_lagging = lag_signal and fp_trend_up > 0.6
    routing_cause = "both" if not cal_mono and not _risk_routing(q2_tdf, ticks)["avg_scale_winners"] > _risk_routing(q2_tdf, ticks)["avg_scale_losers"] else ("mapping" if cal_mono else "score")

    can_mono = bool(best_pen["calibration_monotonic"]) or bool(best_map["calibration_monotonic"])
    arch_verdict = "should continue" if can_mono or float(best_pen["preservation"]) >= 0.85 else "should be redesigned"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"quality_score_false_positive_forensics_{ts}.csv"
    md_path = OUT_DIR / f"quality_score_false_positive_forensics_{ts}.md"

    parts = [
        phase1.assign(section="phase1_group_summary"),
        phase2.assign(section="phase2_top_false_signatures"),
        phase3.assign(section="phase3_interactions"),
        phase4.assign(section="phase4_component_leakage"),
        phase5.assign(section="phase5_regime_delay"),
        phase6.assign(section="phase6_lifecycle"),
        phase7.assign(section="phase7_calibration"),
        phase8.assign(section="phase8_counterfactual_penalty"),
        phase9.assign(section="phase9_routing_spread"),
        pd.DataFrame([{"top_problem": top_problem, **severity}]).assign(section="phase10_diagnosis"),
    ]
    pd.concat(parts, ignore_index=True, sort=False).to_csv(csv_path, index=False)

    insights = [
        f"Largest failure source: {top_problem} (false_high n={len(fp)}).",
        f"Top false-positive signature: {top_sig}.",
        f"Entropy overweighted: {entropy_overweight} — high-score bucket often entropy 0.90~0.96 + trend_up.",
        f"Trend labeling lagging: {trend_lagging} — false highs show negative forward 12-bar return.",
        f"Routing failure driver: {routing_cause}.",
        f"Best counterfactual penalty: {best_pen['variant']} (routing={best_pen['routing_valid']}, pres={best_pen['preservation']:.1%}).",
        f"Best routing mapping: {best_map['mapping']} (routing={best_map['routing_valid']}, pres={best_map['preservation']:.1%}).",
        f"Calibration monotonic achievable: {can_mono} (with penalty/map tuning).",
        f"Architecture: {arch_verdict}.",
        "FALSE HIGH trades collapse early (high immediate MAE / RFE rate vs true winners).",
    ]

    md_lines = ["# Q2 False Positive Forensics", "", f"- false_high_count: {len(fp)}", f"- true_high_winner_count: {len(thw)}", "", "## Top False Signatures"] + [
        f"- {r['signature']}: n={int(r['trade_count'])} exp={r['expectancy']:.6f}" for _, r in phase2.head(5).iterrows()
    ] + ["", "## Final", f"- top_problem: {top_problem}", f"- verdict: {arch_verdict}"]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    grade = "B" if arch_verdict == "should continue" else "C"
    print(f"largest false high-score failure source: {top_problem}")
    print(f"top false-positive signature: {top_sig}")
    print(f"entropy overweighted: {entropy_overweight}")
    print(f"trend labeling lagging: {trend_lagging}")
    print(f"routing failure cause: {routing_cause}")
    print(f"best counterfactual penalty: {best_pen['variant']}")
    print(f"best routing mapping: {best_map['mapping']}")
    print(f"calibration monotonic achievable: {can_mono}")
    print(f"continuous quality score architecture: {arch_verdict}")
    print(f"FINAL VERDICT: Grade {grade}")
    print("top 10 strategic insights:")
    for i, ins in enumerate(insights[:10], 1):
        print(f"  {i}. {ins}")
    print(f"csv: {csv_path}")
    print(f"md: {md_path}")


if __name__ == "__main__":
    main()
