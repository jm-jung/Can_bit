"""
Continuous Quality Score → Position Scale diagnostic replay.

Compares Production, H8 Hard, G30, Hybrid_A vs Q×M quality-score variants.
Diagnostics only — no deployment, no operational changes.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.diagnostics.validate_entry_quality_filters import _margin, _slice_ticks_by_range, _window_endpoints
from scripts.diagnostics.validate_h8_soft_risk_gate import (
    Variant,
    _entropy,
    _expectancy,
    _h8_pass,
    _is_loss_cluster_trade,
    _simulate_variant,
)
from scripts.diagnostics.validate_hybrid_gate_replay import HYBRID_CANDIDATES, _ohlcv_features, _simulate_hybrid
from scripts.run_daily_paper import PAPER_MAX_HOLDING_BARS, load_ohlcv, simulate_signals_paper


OUT_DIR = Path("data/diagnostics/quality_score")
POSITION_SIZE = 0.05
FEE_RATE = 0.0004
SLIPPAGE_RATE = 0.0002
FAIL_SCALE = 0.30
MARGIN_HIGH = 0.07

MAX_TRADE_LOSS = -0.01
MAX_DAILY_LOSS = -0.02
MAX_DRAWDOWN = -0.05
MAX_CONSEC_LOSSES = 5

BASELINE = Variant("Baseline", fail_scale=1.0, hard_block=False)
G30 = Variant("Variant_G30", fail_scale=FAIL_SCALE, hard_block=False)
H8_HARD = Variant("H8_Hard", fail_scale=0.0, hard_block=True)
HYBRID_A = HYBRID_CANDIDATES[0]

ScoreFn = Callable[[Dict[str, Any], Dict[int, Dict[str, Any]]], float]
MapFn = Callable[[float], Optional[float]]


def _clamp(s: float) -> float:
    return max(0.0, min(1.0, s))


def _max_proba(t: Dict[str, Any]) -> float:
    return max(float(t["p_long"]), float(t["p_short"]), float(t["p_flat"]))


def _feat(t: Dict[str, Any], fm: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    return fm.get(int(t.get("df_idx", -1)), {})


def _is_long(t: Dict[str, Any]) -> bool:
    return t.get("signal") == "LONG"


def _danger_long_down_high(t: Dict[str, Any]) -> bool:
    return _is_long(t) and str(t.get("trend_label")) == "down" and str(t.get("vol_bucket")) == "high"


def score_q1(t: Dict[str, Any], fm: Dict[int, Dict[str, Any]]) -> float:
    ent = _entropy(t)
    trend = str(t.get("trend_label") or "")
    vol = str(t.get("vol_bucket") or "")
    s = 0.50
    if ent <= 0.96:
        s += 0.20
    if trend == "up":
        s += 0.15
    if _margin(t) >= MARGIN_HIGH:
        s += 0.10
    if vol != "high":
        s += 0.05
    if trend == "down":
        s -= 0.20
    if vol == "high":
        s -= 0.15
    if _danger_long_down_high(t):
        s -= 0.25
    if ent > 1.00:
        s -= 0.10
    return _clamp(s)


def score_q2(t: Dict[str, Any], fm: Dict[int, Dict[str, Any]]) -> float:
    ent = _entropy(t)
    trend = str(t.get("trend_label") or "")
    vol = str(t.get("vol_bucket") or "")
    if ent <= 0.90:
        s = 0.90
    elif ent <= 0.96:
        s = 0.65
    elif ent <= 1.00:
        s = 0.40
    else:
        s = 0.20
    if trend == "up":
        s += 0.10
    if trend == "down":
        s -= 0.15
    if vol == "high":
        s -= 0.15
    if _danger_long_down_high(t):
        s -= 0.20
    return _clamp(s)


def score_q3(t: Dict[str, Any], fm: Dict[int, Dict[str, Any]]) -> float:
    ent = _entropy(t)
    trend = str(t.get("trend_label") or "")
    vol = str(t.get("vol_bucket") or "")
    s = 0.50
    if trend == "up":
        s += 0.25
    elif trend == "sideways":
        s += 0.00
    elif trend == "down":
        s -= 0.25
    if ent <= 0.96:
        s += 0.15
    if _margin(t) >= MARGIN_HIGH:
        s += 0.10
    if vol == "high":
        s -= 0.10
    if _is_long(t) and trend == "down":
        s -= 0.20
    if _danger_long_down_high(t):
        s -= 0.20
    return _clamp(s)


def score_q4(t: Dict[str, Any], fm: Dict[int, Dict[str, Any]]) -> float:
    ent = _entropy(t)
    s = 0.50
    if _h8_pass(t):
        s += 0.35
    else:
        s -= 0.15
    if _danger_long_down_high(t):
        s -= 0.45
    if _margin(t) >= MARGIN_HIGH:
        s += 0.10
    if ent <= 0.90:
        s += 0.10
    if ent > 1.00:
        s -= 0.15
    return _clamp(s)


def score_q5(t: Dict[str, Any], fm: Dict[int, Dict[str, Any]]) -> float:
    ent = _entropy(t)
    trend = str(t.get("trend_label") or "")
    vol = str(t.get("vol_bucket") or "")
    s = 0.40
    if _h8_pass(t):
        s += 0.25
    if _margin(t) >= MARGIN_HIGH:
        s += 0.15
    if vol != "high":
        s += 0.10
    if trend == "up":
        s += 0.10
    if _is_long(t) and trend == "down":
        s -= 0.30
    if vol == "high" and ent > 0.96:
        s -= 0.25
    if _danger_long_down_high(t):
        s -= 0.50
    return _clamp(s)


SCORE_VARIANTS: Dict[str, Tuple[ScoreFn, int]] = {
    "Q1": (score_q1, 8),
    "Q2": (score_q2, 6),
    "Q3": (score_q3, 7),
    "Q4": (score_q4, 6),
    "Q5": (score_q5, 8),
}


def map_m1(score: float) -> Optional[float]:
    if score >= 0.80:
        return 1.00
    if score >= 0.65:
        return 0.70
    if score >= 0.50:
        return 0.30
    return None


def map_m2(score: float) -> Optional[float]:
    if score >= 0.85:
        return 1.00
    if score >= 0.70:
        return 0.75
    if score >= 0.55:
        return 0.50
    if score >= 0.40:
        return 0.25
    return None


def map_m3(score: float) -> Optional[float]:
    if score >= 0.80:
        return 1.00
    if score >= 0.65:
        return 0.70
    if score >= 0.50:
        return 0.40
    return 0.15


def map_m4(score: float) -> Optional[float]:
    if score >= 0.85:
        return 1.00
    if score >= 0.70:
        return 0.50
    if score >= 0.55:
        return 0.25
    return None


MAP_VARIANTS: Dict[str, MapFn] = {
    "M1": map_m1,
    "M2": map_m2,
    "M3": map_m3,
    "M4": map_m4,
}


def _simulate_quality(
    ticks: List[Dict[str, Any]],
    score_fn: ScoreFn,
    map_fn: MapFn,
    feat_map: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[str, Any], pd.DataFrame, int]:
    trades: List[Dict[str, Any]] = []
    open_pos: Optional[Dict[str, Any]] = None
    eq = peak = 1.0
    daily_pnl = 0.0
    consec_losses = 0
    ks_triggered = False
    risk_force = blocks = 0

    for i, t in enumerate(ticks):
        px = float(t["price"])
        sig = t.get("signal")
        vol = str(t.get("vol_bucket") or "")
        trend = str(t.get("trend_label") or "")

        exit_signal = False
        reason = ""
        if ks_triggered and open_pos is not None:
            exit_signal, reason = True, "kill_switch_close"
        if open_pos is not None:
            open_pos["hold_bars"] += 1
            if sig is not None and (
                (open_pos["side"] == "BUY" and sig == "SHORT")
                or (open_pos["side"] == "SELL" and sig == "LONG")
            ):
                exit_signal, reason = True, reason or "opposite_signal"
            elif open_pos["hold_bars"] >= PAPER_MAX_HOLDING_BARS:
                exit_signal, reason = True, reason or "max_holding_bars"

        if exit_signal and open_pos is not None:
            raw = (
                (px - open_pos["entry_price"]) / open_pos["entry_price"]
                if open_pos["side"] == "BUY"
                else (open_pos["entry_price"] - px) / open_pos["entry_price"]
            )
            net = raw - 2.0 * (FEE_RATE + SLIPPAGE_RATE)
            scale = float(open_pos["scale"])
            eq *= 1.0 + net * POSITION_SIZE * scale
            peak = max(peak, eq)
            daily_pnl = eq - 1.0
            consec_losses = consec_losses + 1 if net * scale < 0 else 0
            trades.append({**open_pos, "exit_reason": reason or "exit_signal", "gross_return": raw, "net_return": net, "scaled_return": net * scale})
            open_pos = None
            continue

        if open_pos is not None:
            unreal = (
                (px - open_pos["entry_price"]) / open_pos["entry_price"]
                if open_pos["side"] == "BUY"
                else (open_pos["entry_price"] - px) / open_pos["entry_price"]
            )
            if unreal <= MAX_TRADE_LOSS:
                net = unreal - 2.0 * (FEE_RATE + SLIPPAGE_RATE)
                scale = float(open_pos["scale"])
                eq *= 1.0 + net * POSITION_SIZE * scale
                peak = max(peak, eq)
                daily_pnl = eq - 1.0
                consec_losses = consec_losses + 1 if net * scale < 0 else 0
                trades.append({**open_pos, "exit_reason": "risk_force_exit", "gross_return": unreal, "net_return": net, "scaled_return": net * scale})
                risk_force += 1
                open_pos = None
            continue

        if vol not in ("mid", "high"):
            continue
        if (trend != "sideways" and _entropy(t) > 1.0) or sig is None:
            continue
        dd = (eq - peak) / peak if peak > 0 else 0.0
        if ks_triggered or daily_pnl <= MAX_DAILY_LOSS or dd <= MAX_DRAWDOWN or consec_losses >= MAX_CONSEC_LOSSES:
            ks_triggered = True
            continue

        qscore = score_fn(t, feat_map)
        scale = map_fn(qscore)
        if scale is None:
            blocks += 1
            continue

        side = "BUY" if sig == "LONG" else "SELL"
        open_pos = {
            "entry_idx": i,
            "entry_price": px,
            "direction": "LONG" if side == "BUY" else "SHORT",
            "side": side,
            "hold_bars": 0,
            "scale": scale,
            "quality_score": qscore,
            "h8_pass": _h8_pass(t),
        }

    tdf = pd.DataFrame(trades)
    vals = tdf["scaled_return"].tolist() if not tdf.empty else []
    wins = [x for x in vals if x > 0]
    losses = [x for x in vals if x < 0]
    wr = len(wins) / len(vals) if vals else 0.0
    pf = (sum(wins) / abs(sum(losses))) if losses else 0.0
    mdd = eq_f = peak_f = 1.0
    mdd_min = 0.0
    for r in vals:
        eq_f *= 1.0 + r * POSITION_SIZE
        peak_f = max(peak_f, eq_f)
        mdd_min = min(mdd_min, (eq_f - peak_f) / peak_f if peak_f > 0 else 0.0)

    worst_streak = cur = 0
    loss_cluster = 0
    if not tdf.empty:
        for _, tr in tdf.iterrows():
            if _is_loss_cluster_trade(tr.to_dict(), ticks[int(tr["entry_idx"])]):
                loss_cluster += 1
            r = float(tr["scaled_return"])
            if r < 0:
                cur += 1
                worst_streak = max(worst_streak, cur)
            else:
                cur = 0

    scales = tdf["scale"].tolist() if not tdf.empty else []
    metrics = {
        "trades": int(len(tdf)),
        "win_rate": wr,
        "expectancy": _expectancy(vals),
        "profit_factor": pf,
        "net_return": float(np.sum(vals)) if vals else 0.0,
        "final_equity": float(eq_f),
        "MDD": float(mdd_min),
        "risk_force_exit_count": int(risk_force),
        "KS_triggered": int(bool(ks_triggered)),
        "avg_hold": float(tdf["hold_bars"].mean()) if not tdf.empty else 0.0,
        "worst_losing_streak": int(worst_streak),
        "loss_cluster_count": int(loss_cluster),
        "avg_scale": float(np.mean(scales)) if scales else 0.0,
        "scale_distribution": dict(Counter(scales)),
        "blocks_count": int(blocks),
    }
    return metrics, tdf, blocks


def _equity_quality(tdf: pd.DataFrame) -> Dict[str, float]:
    if tdf.empty:
        return {"equity_smoothness": 0.0, "max_underwater_bars": 0, "worst_1d": 0.0, "worst_7d": 0.0, "worst_30d": 0.0, "tail_5pct": 0.0}
    eq = 1.0
    peak = 1.0
    dds = []
    for r in tdf["scaled_return"]:
        eq *= 1.0 + float(r) * POSITION_SIZE
        peak = max(peak, eq)
        dds.append((eq - peak) / peak if peak > 0 else 0.0)
    rets = tdf["scaled_return"].tolist()
    tail = float(np.quantile(rets, 0.05)) if len(rets) >= 20 else (min(rets) if rets else 0.0)
    uw = cur = 0
    for d in dds:
        if d < -1e-9:
            cur += 1
            uw = max(uw, cur)
        else:
            cur = 0
    std = float(np.std(rets)) if len(rets) > 1 else 1.0
    return {
        "equity_smoothness": 1.0 / (std + 1e-9),
        "max_underwater_bars": float(uw),
        "worst_1d": float(min(rets)) if rets else 0.0,
        "worst_7d": 0.0,
        "worst_30d": 0.0,
        "tail_5pct": tail,
    }


def _risk_routing(tdf: pd.DataFrame, ticks: List[Dict[str, Any]]) -> Dict[str, float]:
    if tdf.empty:
        return {"avg_scale_winners": 0.0, "avg_scale_losers": 0.0, "avg_scale_rfe": 0.0, "avg_scale_cluster": 0.0}
    wins = tdf[tdf["scaled_return"] > 0]["scale"]
    losses = tdf[tdf["scaled_return"] < 0]["scale"]
    rfe = tdf[tdf["exit_reason"] == "risk_force_exit"]["scale"]
    cluster_idx = [int(tr["entry_idx"]) for _, tr in tdf.iterrows() if _is_loss_cluster_trade(tr.to_dict(), ticks[int(tr["entry_idx"])])]
    cluster = tdf[tdf["entry_idx"].isin(cluster_idx)]["scale"] if cluster_idx else pd.Series(dtype=float)
    return {
        "avg_scale_winners": float(wins.mean()) if len(wins) else 0.0,
        "avg_scale_losers": float(losses.mean()) if len(losses) else 0.0,
        "avg_scale_rfe": float(rfe.mean()) if len(rfe) else 0.0,
        "avg_scale_cluster": float(cluster.mean()) if len(cluster) else 0.0,
    }


def _score_calibration(tdf: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    if tdf.empty or "quality_score" not in tdf.columns:
        return pd.DataFrame(), False
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    rows = []
    exps = []
    for lo, hi in bins:
        sub = tdf[(tdf["quality_score"] >= lo) & (tdf["quality_score"] < hi)]
        if sub.empty:
            rows.append({"bucket": f"{lo:.1f}~{hi:.1f}", "trade_count": 0, "avg_return": 0.0, "win_rate": 0.0, "expectancy": 0.0})
            exps.append(0.0)
            continue
        vals = sub["scaled_return"].tolist()
        exp = _expectancy(vals)
        exps.append(exp)
        rfe_ratio = float((sub["exit_reason"] == "risk_force_exit").mean())
        rows.append({
            "bucket": f"{lo:.1f}~{hi:.1f}",
            "trade_count": len(sub),
            "avg_return": float(np.mean(vals)),
            "win_rate": float((sub["scaled_return"] > 0).mean()),
            "expectancy": exp,
            "profit_factor": float(sub[sub["scaled_return"] > 0]["scaled_return"].sum() / abs(sub[sub["scaled_return"] < 0]["scaled_return"].sum())) if (sub["scaled_return"] < 0).any() else 0.0,
            "rfe_ratio": rfe_ratio,
        })
    nonempty = [e for i, e in enumerate(exps) if rows[i]["trade_count"] > 0]
    monotonic = all(nonempty[i] <= nonempty[i + 1] for i in range(len(nonempty) - 1)) if len(nonempty) >= 2 else False
    return pd.DataFrame(rows), monotonic


def _run_baselines(ticks: List[Dict[str, Any]], feat_map: Dict[int, Dict[str, Any]]) -> Dict[str, Tuple[Dict[str, Any], pd.DataFrame]]:
    out: Dict[str, Tuple[Dict[str, Any], pd.DataFrame]] = {}
    prod_m, prod_tdf = _simulate_variant(ticks, BASELINE, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    out["Production"] = (prod_m, prod_tdf)
    h8_m, h8_tdf = _simulate_variant(ticks, H8_HARD, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    out["H8_Hard"] = (h8_m, h8_tdf)
    g30_m, g30_tdf = _simulate_variant(ticks, G30, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    out["G30"] = (g30_m, g30_tdf)
    hyb_m, hyb_tdf, _ = _simulate_hybrid(ticks, HYBRID_A, feat_map, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    out["Hybrid_A"] = (hyb_m, hyb_tdf)
    return out


def _enrich_metrics(m: Dict[str, Any], tdf: pd.DataFrame, ticks: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(m)
    if "final_equity" not in out:
        eq = 1.0
        for r in tdf["scaled_return"] if not tdf.empty else []:
            eq *= 1.0 + float(r) * POSITION_SIZE
        out["final_equity"] = eq
    if "loss_cluster_count" not in out:
        out["loss_cluster_count"] = sum(
            1 for _, tr in tdf.iterrows() if _is_loss_cluster_trade(tr.to_dict(), ticks[int(tr["entry_idx"])])
        ) if not tdf.empty else 0
    if "avg_scale" not in out:
        out["avg_scale"] = float(tdf["scale"].mean()) if not tdf.empty and "scale" in tdf.columns else 1.0
    return out


def _metrics_row(
    candidate: str,
    m: Dict[str, Any],
    tdf: pd.DataFrame,
    ticks: List[Dict[str, Any]],
    base_trades: int,
    g30_m: Dict[str, Any],
    hyb_m: Dict[str, Any],
    complexity: int = 0,
    scope: str = "full_history",
    lb: int = 0,
    wid: int = 0,
) -> Dict[str, Any]:
    m = _enrich_metrics(m, tdf, ticks)
    pres = m["trades"] / max(base_trades, 1)
    rr = _risk_routing(tdf, ticks)
    eq_q = _equity_quality(tdf)
    routing_valid = rr["avg_scale_winners"] > rr["avg_scale_losers"] > rr["avg_scale_rfe"] if m["trades"] > 0 else False
    return {
        "scope": scope,
        "lookback_days": lb,
        "window_id": wid,
        "candidate": candidate,
        "trades": m["trades"],
        "trade_preservation_pct": pres,
        "avg_scale": m.get("avg_scale", 1.0),
        "win_rate": m["win_rate"],
        "expectancy": m["expectancy"],
        "profit_factor": m["profit_factor"],
        "net_return": m["net_return"],
        "final_equity": m.get("final_equity", 1.0),
        "MDD": m["MDD"],
        "risk_force_exit_count": m["risk_force_exit_count"],
        "KS_triggered": m["KS_triggered"],
        "loss_cluster_count": m.get("loss_cluster_count", 0),
        "avg_hold": m["avg_hold"],
        "net_delta_vs_g30": m["net_return"] - g30_m["net_return"],
        "mdd_delta_vs_g30": m["MDD"] - g30_m["MDD"],
        "rfe_delta_vs_g30": m["risk_force_exit_count"] - g30_m["risk_force_exit_count"],
        "ks_delta_vs_g30": m["KS_triggered"] - g30_m["KS_triggered"],
        "net_delta_vs_hybrid": m["net_return"] - hyb_m["net_return"],
        "mdd_delta_vs_hybrid": m["MDD"] - hyb_m["MDD"],
        "rfe_delta_vs_hybrid": m["risk_force_exit_count"] - hyb_m["risk_force_exit_count"],
        "complexity_score": complexity,
        **eq_q,
        **rr,
        "routing_valid": routing_valid,
    }


def _tournament_score(g: pd.DataFrame) -> Dict[str, float]:
    c = g["net_delta_vs_g30"].tolist()
    avg_net = float(g["net_delta_vs_g30"].mean())
    avg_mdd = float(g["mdd_delta_vs_g30"].mean())
    avg_rfe = float(g["rfe_delta_vs_g30"].mean())
    pres = float(g["trade_preservation_pct"].mean())
    std_n = float(pstdev(c)) if len(c) > 1 else 0.0
    stability = avg_net / (std_n + 1e-9)
    complexity = float(g["complexity_score"].iloc[0]) if "complexity_score" in g.columns else 5.0
    risk_adj = avg_net + 0.35 * avg_mdd - 0.15 * max(avg_rfe, 0) - 0.02 * complexity
    final = 0.35 * max(risk_adj, 0) + 0.25 * max(stability, 0) + 0.2 * pres + 0.1 * max(-avg_rfe, 0) + 0.1 * max(avg_mdd, 0)
    return {"stability_score": stability, "risk_adjusted_score": risk_adj, "final_score": final, "avg_preservation": pres}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("validate_quality_score_replay")
    p.add_argument("--lookback-list", default="14,30,60,90,180")
    p.add_argument("--stride-days", type=int, default=7)
    p.add_argument("--max-windows-per-lookback", type=int, default=9999)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    lookbacks = [int(x.strip()) for x in args.lookback_list.split(",") if x.strip()]

    df = load_ohlcv()
    if df is None or df.empty:
        raise SystemExit("OHLCV load failed")
    feat_map = _ohlcv_features(df)
    ticks, _ = simulate_signals_paper(df)
    if not ticks:
        raise SystemExit("No ticks")

    rows: List[Dict[str, Any]] = []
    quality_combos = [(qn, mn, SCORE_VARIANTS[qn][0], MAP_VARIANTS[mn], SCORE_VARIANTS[qn][1]) for qn in SCORE_VARIANTS for mn in MAP_VARIANTS]

    def process_scope(scope: str, w_ticks: List[Dict[str, Any]], lb: int = 0, wid: int = 0) -> None:
        bases = _run_baselines(w_ticks, feat_map)
        base_trades = bases["Production"][0]["trades"]
        g30_m = bases["G30"][0]
        hyb_m = bases["Hybrid_A"][0]

        for name, (m, tdf) in bases.items():
            rows.append(_metrics_row(name, m, tdf, w_ticks, base_trades, g30_m, hyb_m, complexity=0, scope=scope, lb=lb, wid=wid))

        for qn, mn, sfn, mfn, cx in quality_combos:
            cand = f"{qn}_{mn}"
            m, tdf, _ = _simulate_quality(w_ticks, sfn, mfn, feat_map)
            rows.append(_metrics_row(cand, m, tdf, w_ticks, base_trades, g30_m, hyb_m, complexity=cx, scope=scope, lb=lb, wid=wid))

    for lb in lookbacks:
        for wid, (ws, we) in enumerate(_window_endpoints(ticks, lb, args.stride_days, args.max_windows_per_lookback)):
            w_ticks = _slice_ticks_by_range(ticks, ws, we)
            if len(w_ticks) < 30:
                continue
            process_scope("rolling_oos", w_ticks, lb, wid)

    process_scope("full_history", ticks, 0, 0)

    detail = pd.DataFrame(rows)
    roll = detail[detail["scope"] == "rolling_oos"]
    full = detail[detail["scope"] == "full_history"]

    summary_rows = []
    for cand, g in roll.groupby("candidate"):
        ts = _tournament_score(g)
        fh = full[full["candidate"] == cand]
        fh_r = fh.iloc[0].to_dict() if not fh.empty else {}
        summary_rows.append({"candidate": cand, **ts, **{f"fh_{k}": v for k, v in fh_r.items() if k in ("net_return", "MDD", "trades", "risk_force_exit_count", "trade_preservation_pct", "routing_valid")}})

    summary = pd.DataFrame(summary_rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    best = summary.iloc[0]
    best_name = str(best["candidate"])
    g30_fh = full[full["candidate"] == "G30"].iloc[0]
    hyb_fh = full[full["candidate"] == "Hybrid_A"].iloc[0]
    best_fh = full[full["candidate"] == best_name].iloc[0]

    # calibration on best quality candidate (full history)
    cal_mono = False
    cal_df = pd.DataFrame()
    if best_name not in ("Production", "H8_Hard", "G30", "Hybrid_A"):
        qn, mn = best_name.split("_", 1)
        _, tdf_best, _ = _simulate_quality(ticks, SCORE_VARIANTS[qn][0], MAP_VARIANTS[mn], feat_map)
        cal_df, cal_mono = _score_calibration(tdf_best)

    qs_gt_hybrid = float(best_fh["net_return"]) > float(hyb_fh["net_return"]) and float(best_fh["MDD"]) >= float(hyb_fh["MDD"])
    pres_ok = float(best_fh["trade_preservation_pct"]) >= float(hyb_fh["trade_preservation_pct"])
    routing_valid = bool(best_fh.get("routing_valid", False))
    hyb_routing = bool(hyb_fh.get("routing_valid", False))

    if qs_gt_hybrid and pres_ok and routing_valid and cal_mono:
        grade, rec = "A", "replace Hybrid_A"
    elif qs_gt_hybrid and float(best["final_score"]) > summary[summary["candidate"] == "Hybrid_A"]["final_score"].iloc[0]:
        grade, rec = "B", "become daily monitor"
    elif float(best["final_score"]) > summary[summary["candidate"] == "G30"]["final_score"].iloc[0]:
        grade, rec = "C", "continue as diagnostic"
    else:
        grade, rec = "D", "be rejected"

    if best_name in ("G30", "Hybrid_A", "Production", "H8_Hard"):
        rec = "continue as diagnostic — rule baselines still competitive"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"quality_score_replay_{ts}.csv"
    md_path = OUT_DIR / f"quality_score_replay_{ts}.md"

    parts = [detail.assign(section="detail"), summary.assign(section="tournament")]
    if not cal_df.empty:
        parts.append(cal_df.assign(section="calibration", candidate=best_name))
    parts.append(pd.DataFrame([{"grade": grade, "recommendation": rec, "best_candidate": best_name}]).assign(section="verdict"))
    pd.concat(parts, ignore_index=True, sort=False).to_csv(csv_path, index=False)

    md = [
        "# Quality Score Replay",
        f"- best: **{best_name}** (grade {grade})",
        f"- recommendation: {rec}",
        "",
        "## Full History",
        f"| candidate | trades | net | MDD | RFE | preservation |",
        f"|-----------|--------|-----|-----|-----|--------------|",
    ]
    for c in ["Production", "G30", "Hybrid_A", best_name]:
        r = full[full["candidate"] == c].iloc[0]
        md.append(f"| {c} | {int(r['trades'])} | {r['net_return']:.6f} | {r['MDD']:.6f} | {int(r['risk_force_exit_count'])} | {r['trade_preservation_pct']:.1%} |")
    if not cal_df.empty:
        md.append("\n## Score Calibration (best)")
        for _, row in cal_df.iterrows():
            md.append(f"- {row['bucket']}: n={int(row['trade_count'])}, exp={row.get('expectancy', 0):.6f}")
        md.append(f"- monotonic: {cal_mono}")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    best_q, best_m = (best_name.split("_", 1) if "_" in best_name and best_name not in ("H8_Hard",) else (best_name, "n/a"))

    insights = [
        f"Best tournament candidate: {best_name} (final_score={best['final_score']:.4f}).",
        f"Quality score > Hybrid_A (full net/MDD): {qs_gt_hybrid}.",
        f"Trade preservation vs production: {best_fh['trade_preservation_pct']:.1%} (Hybrid {hyb_fh['trade_preservation_pct']:.1%}).",
        f"Score calibration monotonic: {cal_mono}; risk routing valid (win>loss>RFE): {routing_valid} (Hybrid_A: {hyb_routing}).",
        f"Recommendation: {rec} (grade {grade}).",
    ]

    print(f"best quality score candidate: {best_name}")
    print(f"best mapping: {best_m}")
    print(f"whether quality score > Hybrid_A: {qs_gt_hybrid}")
    print(f"whether quality score improves trade preservation: {pres_ok}")
    print(f"whether score calibration is monotonic: {cal_mono}")
    print(f"whether risk routing is valid: {routing_valid}")
    print(f"whether this should: {rec}")
    print(f"FINAL VERDICT: Grade {grade}")
    print("top 5 strategic insights:")
    for i, s in enumerate(insights, 1):
        print(f"  {i}. {s}")
    print(f"csv: {csv_path}")
    print(f"md: {md_path}")


if __name__ == "__main__":
    main()
