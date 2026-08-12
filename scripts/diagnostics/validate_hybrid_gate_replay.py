"""
Hybrid gate forensic replay: Production vs G30 vs G30 + extreme hard block.

Diagnostics only — no deployment, no operational code changes.
"""

from __future__ import annotations

import argparse
import math
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
from scripts.run_daily_paper import PAPER_MAX_HOLDING_BARS, load_ohlcv, simulate_signals_paper


OUT_DIR = Path("data/diagnostics/hybrid_gate")
MAX_TRADE_LOSS = -0.01
MAX_DAILY_LOSS = -0.02
MAX_DRAWDOWN = -0.05
MAX_CONSEC_LOSSES = 5
FAIL_SCALE = 0.30
MARGIN_HIGH = 0.07

BASELINE = Variant("Baseline", fail_scale=1.0, hard_block=False)
G30 = Variant("Variant_G30", fail_scale=FAIL_SCALE, hard_block=False)


@dataclass(frozen=True)
class HybridCandidate:
    name: str
    block_fn: Callable[[Dict[str, Any], Dict[int, Dict[str, Any]]], bool]


def _ohlcv_features(df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    px = pd.to_numeric(df["close"], errors="coerce")
    rr24 = px.pct_change(24)
    out: Dict[int, Dict[str, Any]] = {}
    for i in range(len(df)):
        v = rr24.iloc[i]
        out[i] = {
            "recent_return_24": float(v) if pd.notna(v) else float("nan"),
        }
    return out


def _feat(t: Dict[str, Any], feat_map: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    idx = int(t.get("df_idx", -1))
    return feat_map.get(idx, {})


def _is_long(t: Dict[str, Any]) -> bool:
    return t.get("signal") == "LONG"


def _block_A(t: Dict[str, Any], fm: Dict[int, Dict[str, Any]]) -> bool:
    return _is_long(t) and str(t.get("trend_label")) == "down" and str(t.get("vol_bucket")) == "high"


def _block_B(t: Dict[str, Any], fm: Dict[int, Dict[str, Any]]) -> bool:
    ent = _entropy(t)
    return _is_long(t) and str(t.get("trend_label")) == "down" and 0.90 <= ent <= 0.96


def _block_C(t: Dict[str, Any], fm: Dict[int, Dict[str, Any]]) -> bool:
    ent = _entropy(t)
    return _is_long(t) and 0.90 <= ent <= 0.96 and str(t.get("vol_bucket")) == "high"


def _block_D(t: Dict[str, Any], fm: Dict[int, Dict[str, Any]]) -> bool:
    ent = _entropy(t)
    return (
        _is_long(t)
        and str(t.get("trend_label")) == "down"
        and str(t.get("vol_bucket")) == "high"
        and 0.90 <= ent <= 0.96
    )


def _block_E(t: Dict[str, Any], fm: Dict[int, Dict[str, Any]]) -> bool:
    return _is_long(t) and _margin(t) >= MARGIN_HIGH and str(t.get("trend_label")) == "down"


def _block_F(t: Dict[str, Any], fm: Dict[int, Dict[str, Any]]) -> bool:
    rr = _feat(t, fm).get("recent_return_24", float("nan"))
    return _is_long(t) and pd.notna(rr) and float(rr) < 0 and str(t.get("trend_label")) == "down"


def _block_G(t: Dict[str, Any], fm: Dict[int, Dict[str, Any]]) -> bool:
    ent = _entropy(t)
    return (
        _is_long(t)
        and str(t.get("trend_label")) == "down"
        and str(t.get("vol_bucket")) == "high"
        and 0.90 <= ent <= 0.96
        and _margin(t) >= MARGIN_HIGH
    )


HYBRID_CANDIDATES: List[HybridCandidate] = [
    HybridCandidate("Hybrid_A", _block_A),
    HybridCandidate("Hybrid_B", _block_B),
    HybridCandidate("Hybrid_C", _block_C),
    HybridCandidate("Hybrid_D", _block_D),
    HybridCandidate("Hybrid_E", _block_E),
    HybridCandidate("Hybrid_F", _block_F),
    HybridCandidate("Hybrid_G", _block_G),
]


def _simulate_hybrid(
    ticks: List[Dict[str, Any]],
    hybrid: HybridCandidate,
    feat_map: Dict[int, Dict[str, Any]],
    position_size: float,
    fee_rate: float,
    slippage_rate: float,
) -> Tuple[Dict[str, Any], pd.DataFrame, int]:
    """G30 soft gate + extreme hard block. Returns metrics, trades, blocks_count."""
    trades: List[Dict[str, Any]] = []
    open_pos: Optional[Dict[str, Any]] = None
    eq = 1.0
    peak = 1.0
    daily_pnl = 0.0
    consec_losses = 0
    ks_triggered = False
    risk_force = 0
    blocks = 0

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
                    "hybrid_blocked": False,
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
                        "hybrid_blocked": False,
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
        if sig is None:
            continue

        drawdown = (eq - peak) / peak if peak > 0 else 0.0
        if ks_triggered or daily_pnl <= MAX_DAILY_LOSS or drawdown <= MAX_DRAWDOWN or consec_losses >= MAX_CONSEC_LOSSES:
            ks_triggered = True
            continue

        h8 = _h8_pass(t)
        if hybrid.block_fn(t, feat_map):
            blocks += 1
            continue

        scale = 1.0 if h8 else FAIL_SCALE
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
    pf = (sum(wins) / abs(sum(losses))) if losses else 0.0
    mdd = 0.0
    eq2 = 1.0
    peak2 = 1.0
    for r in vals:
        eq2 *= 1.0 + r * position_size
        peak2 = max(peak2, eq2)
        mdd = min(mdd, (eq2 - peak2) / peak2 if peak2 > 0 else 0.0)

    loss_cluster = 0
    worst_streak = 0
    cur_streak = 0
    cat_loss = 0
    cur_cat = 0
    if not tdf.empty:
        for _, tr in tdf.iterrows():
            i = int(tr["entry_idx"])
            if _is_loss_cluster_trade(tr.to_dict(), ticks[i]):
                loss_cluster += 1
            r = float(tr["scaled_return"])
            if r < 0:
                cur_streak += 1
                worst_streak = max(worst_streak, cur_streak)
                if r < -0.005:
                    cur_cat += 1
                    if cur_cat >= 3:
                        cat_loss += 1
            else:
                cur_streak = 0
                cur_cat = 0

    metrics = {
        "trades": int(len(tdf)),
        "win_rate": wr,
        "expectancy": _expectancy(vals),
        "profit_factor": pf,
        "net_return": float(np.sum(vals)) if vals else 0.0,
        "MDD": float(mdd),
        "risk_force_exit_count": int(risk_force),
        "KS_triggered": int(bool(ks_triggered)),
        "avg_hold": float(tdf["hold_bars"].mean()) if not tdf.empty else 0.0,
        "loss_cluster_count": int(loss_cluster),
        "worst_losing_streak": int(worst_streak),
        "catastrophic_sequences": int(cat_loss),
        "tail_worst_5pct": float(np.quantile(vals, 0.05)) if len(vals) >= 20 else (min(vals) if vals else 0.0),
        "blocks_count": int(blocks),
    }
    return metrics, tdf, blocks


def _catastrophic_sequences(tdf: pd.DataFrame) -> int:
    if tdf.empty:
        return 0
    cat = 0
    cur = 0
    for r in tdf.sort_values("entry_idx")["scaled_return"]:
        if float(r) < -0.005:
            cur += 1
            if cur >= 3:
                cat += 1
        else:
            cur = 0
    return cat


def _loss_cluster_count(tdf: pd.DataFrame, ticks: List[Dict[str, Any]]) -> int:
    if tdf.empty:
        return 0
    n = 0
    for _, tr in tdf.iterrows():
        if _is_loss_cluster_trade(tr.to_dict(), ticks[int(tr["entry_idx"])]):
            n += 1
    return n


def _removal_cost(
    g30_tdf: pd.DataFrame,
    hyb_tdf: pd.DataFrame,
    ticks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    g30_idx = set(g30_tdf["entry_idx"].astype(int).tolist()) if not g30_tdf.empty else set()
    hyb_idx = set(hyb_tdf["entry_idx"].astype(int).tolist()) if not hyb_tdf.empty else set()
    removed = g30_idx - hyb_idx
    if not removed or g30_tdf.empty:
        return {
            "good_trade_removed": 0,
            "bad_trade_removed": 0,
            "removal_precision": 0.0,
            "trade_opportunity_destruction_pct": 0.0,
        }
    rem_df = g30_tdf[g30_tdf["entry_idx"].isin(removed)]
    good = int((rem_df["scaled_return"] > 0).sum())
    bad = int((rem_df["scaled_return"] < 0).sum())
    total = len(rem_df)
    prec = bad / total if total else 0.0
    dest = (1.0 - len(hyb_idx) / max(len(g30_idx), 1)) * 100.0
    return {
        "good_trade_removed": good,
        "bad_trade_removed": bad,
        "removal_precision": round(prec, 4),
        "trade_opportunity_destruction_pct": round(dest, 2),
    }


def _regime_pnl(tdf: pd.DataFrame, ticks: List[Dict[str, Any]]) -> Dict[str, float]:
    if tdf.empty:
        return {}
    pnl: Dict[str, float] = {}
    for _, tr in tdf.iterrows():
        t = ticks[int(tr["entry_idx"])]
        trend = str(t.get("trend_label") or "")
        vol = str(t.get("vol_bucket") or "")
        mreg = "bull" if trend == "up" else ("bear" if trend == "down" else "sideways")
        for key in (mreg, "high_vol" if vol == "high" else "low_vol", f"trend_{trend}"):
            pnl[key] = pnl.get(key, 0.0) + float(tr["scaled_return"]) * 0.05
    return pnl


def _profit_concentration(tdf: pd.DataFrame) -> Dict[str, float]:
    if tdf.empty:
        return {"top_1pct": 0.0, "top_5pct": 0.0, "top_10pct": 0.0}
    contrib = tdf["scaled_return"] * 0.05
    total = float(contrib.sum())
    if abs(total) < 1e-12:
        return {"top_1pct": 0.0, "top_5pct": 0.0, "top_10pct": 0.0}
    ranked = contrib.sort_values(ascending=False)
    n = len(ranked)

    def share(pct: int) -> float:
        k = max(1, int(math.ceil(n * pct / 100.0)))
        return float(ranked.head(k).sum() / total)

    return {"top_1pct": share(1), "top_5pct": share(5), "top_10pct": share(10)}


def _consistency(sub: pd.DataFrame, col: str) -> Dict[str, float]:
    if sub.empty:
        return {"improved_windows": 0, "degraded_windows": 0, "consistency_score": 0.0}
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


def _scores(g: pd.DataFrame) -> Dict[str, float]:
    avg_net_d = float(g["net_delta_vs_g30"].mean())
    avg_mdd_d = float(g["mdd_delta_vs_g30"].mean())
    avg_rfe_red = float(g["rfe_reduction_pct"].mean())
    avg_ks_red = float(g["ks_reduction_pct"].mean())
    trade_pres = float(g["trade_preservation_vs_g30"].mean())
    removal_prec = float(g["removal_precision"].mean()) if "removal_precision" in g.columns else 0.0
    c = _consistency(g, "net_delta_vs_g30")
    risk_score = avg_net_d + 0.35 * avg_mdd_d + 0.2 * avg_rfe_red + 0.15 * avg_ks_red
    stability_score = c["consistency_score"]
    hybrid_eff = (avg_rfe_red + avg_ks_red) * removal_prec / max(100.0 - trade_pres * 100.0, 1.0)
    shadow_score = 0.35 * max(stability_score, 0) + 0.25 * max(avg_mdd_d, 0) + 0.2 * max(avg_rfe_red, 0) + 0.2 * trade_pres
    paper_score = 0.3 * max(avg_net_d, 0) + 0.25 * max(avg_mdd_d, 0) + 0.2 * max(avg_rfe_red, 0) + 0.15 * trade_pres + 0.1 * removal_prec
    return {
        "shadow_score": shadow_score,
        "paper_score": paper_score,
        "risk_score": risk_score,
        "stability_score": stability_score,
        "trade_preservation_score": trade_pres,
        "hybrid_efficiency_score": hybrid_eff,
        "removal_precision": removal_prec,
        "avg_net_delta_vs_g30": avg_net_d,
        "avg_mdd_delta_vs_g30": avg_mdd_d,
        "avg_rfe_reduction_pct": avg_rfe_red,
        "avg_ks_reduction_pct": avg_ks_red,
    }


def _grade_hybrid(row: pd.Series) -> str:
    if (
        row["avg_rfe_reduction_pct"] > 0.5
        and row["trade_preservation_score"] >= 0.92
        and row["avg_net_delta_vs_g30"] >= -0.002
        and row["avg_mdd_delta_vs_g30"] >= 0
        and row["stability_score"] > 0
    ):
        return "A"
    if row["avg_rfe_reduction_pct"] > 0 and row["trade_preservation_score"] >= 0.85 and row["risk_score"] > 0:
        return "B"
    if row["risk_score"] > -0.01 or row["avg_rfe_reduction_pct"] > 0:
        return "C"
    return "D"


def _recommendation(grade: str, row: pd.Series, hybrid_gt_g30: bool) -> str:
    if grade == "D" or not hybrid_gt_g30:
        return "be rejected"
    if grade == "A" and row["trade_preservation_score"] >= 0.95:
        return "become monitor candidate"
    if grade in ("A", "B") and row["avg_rfe_reduction_pct"] > 2 and row["avg_net_delta_vs_g30"] > 0:
        return "replace G30"
    if grade in ("B", "C"):
        return "continue forensic replay"
    return "be rejected"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("validate_hybrid_gate_replay")
    p.add_argument("--lookback-list", default="14,30,60,90,180")
    p.add_argument("--stride-days", type=int, default=7)
    p.add_argument("--max-windows-per-lookback", type=int, default=9999)
    p.add_argument("--position-size", type=float, default=0.05)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0002)
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

    def _run_scope(scope: str, w_ticks: List[Dict[str, Any]], lb: int = 0, wid: int = 0) -> None:
        base_m, base_tdf = _simulate_variant(w_ticks, BASELINE, args.position_size, args.fee_rate, args.slippage_rate)
        g30_m, g30_tdf = _simulate_variant(w_ticks, G30, args.position_size, args.fee_rate, args.slippage_rate)

        for policy, m, tdf, blocks in [
            ("Baseline", base_m, base_tdf, 0),
            ("G30", g30_m, g30_tdf, 0),
        ]:
            pres = m["trades"] / max(base_m["trades"], 1)
            rows.append(
                {
                    "scope": scope,
                    "lookback_days": lb,
                    "window_id": wid,
                    "policy": policy,
                    "hybrid_candidate": policy,
                    **m,
                    "trade_preservation_vs_prod": pres,
                    "trade_preservation_vs_g30": 1.0 if policy == "G30" else pres,
                    "rfe_reduction_pct_vs_prod": 0.0,
                    "ks_reduction_pct_vs_prod": 0.0,
                    "net_delta_vs_g30": 0.0,
                    "mdd_delta_vs_g30": 0.0,
                    "blocks_count": blocks,
                }
            )

        for hc in HYBRID_CANDIDATES:
            h_m, h_tdf, blocks = _simulate_hybrid(
                w_ticks, hc, feat_map, args.position_size, args.fee_rate, args.slippage_rate
            )
            rem = _removal_cost(g30_tdf, h_tdf, w_ticks)
            rfe_red = (1.0 - h_m["risk_force_exit_count"] / max(g30_m["risk_force_exit_count"], 1)) * 100.0
            ks_red = (1.0 - h_m["KS_triggered"] / max(g30_m["KS_triggered"], 1)) * 100.0 if g30_m["KS_triggered"] else (
                100.0 if not h_m["KS_triggered"] else 0.0
            )
            rfe_red_prod = (1.0 - h_m["risk_force_exit_count"] / max(base_m["risk_force_exit_count"], 1)) * 100.0
            g30_lc = _loss_cluster_count(g30_tdf, w_ticks)
            cluster_red = (1.0 - h_m["loss_cluster_count"] / max(g30_lc, 1)) * 100.0
            g30_cat = _catastrophic_sequences(g30_tdf)
            cat_red = (1.0 - h_m["catastrophic_sequences"] / max(g30_cat, 1)) * 100.0

            rows.append(
                {
                    "scope": scope,
                    "lookback_days": lb,
                    "window_id": wid,
                    "policy": "Hybrid",
                    "hybrid_candidate": hc.name,
                    **h_m,
                    "trade_preservation_vs_prod": h_m["trades"] / max(base_m["trades"], 1),
                    "trade_preservation_vs_g30": h_m["trades"] / max(g30_m["trades"], 1),
                    "rfe_reduction_pct_vs_prod": rfe_red_prod,
                    "rfe_reduction_pct": rfe_red,
                    "ks_reduction_pct": ks_red,
                    "loss_cluster_reduction_pct": cluster_red,
                    "catastrophic_reduction_pct": cat_red,
                    "tail_risk_improvement": float(
                        h_m["tail_worst_5pct"]
                        - (
                            float(np.quantile(g30_tdf["scaled_return"], 0.05))
                            if len(g30_tdf) >= 20
                            else 0.0
                        )
                    ),
                    "net_delta_vs_g30": h_m["net_return"] - g30_m["net_return"],
                    "mdd_delta_vs_g30": h_m["MDD"] - g30_m["MDD"],
                    "expectancy_delta_vs_g30": h_m["expectancy"] - g30_m["expectancy"],
                    "blocks_count": blocks,
                    **rem,
                }
            )

    # rolling OOS
    for lb in lookbacks:
        wins = _window_endpoints(ticks, lb, args.stride_days, args.max_windows_per_lookback)
        for wid, (ws, we) in enumerate(wins):
            w_ticks = _slice_ticks_by_range(ticks, ws, we)
            if len(w_ticks) < 30:
                continue
            _run_scope("rolling_oos", w_ticks, lb, wid)

    # full history
    _run_scope("full_history", ticks, 0, 0)

    detail = pd.DataFrame(rows)
    hyb = detail[detail["policy"] == "Hybrid"].copy()
    g30_ref = detail[detail["policy"] == "G30"].copy()

    # tournament summary per hybrid candidate (rolling only for consistency)
    roll = hyb[hyb["scope"] == "rolling_oos"]
    full = hyb[hyb["scope"] == "full_history"]
    summary_rows: List[Dict[str, Any]] = []
    for name, g in roll.groupby("hybrid_candidate"):
        sc = _scores(g)
        fh = full[full["hybrid_candidate"] == name]
        fh_row = fh.iloc[0] if not fh.empty else {}
        row = {"hybrid_candidate": name, **sc}
        if not fh.empty:
            row["full_history_net"] = fh_row["net_return"]
            row["full_history_mdd"] = fh_row["MDD"]
            row["full_history_rfe"] = fh_row["risk_force_exit_count"]
            row["full_history_trades"] = fh_row["trades"]
            row["full_history_preservation_vs_g30"] = fh_row["trade_preservation_vs_g30"]
        row["grade"] = _grade_hybrid(pd.Series(row))
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise SystemExit("No hybrid results")

    summary = summary.sort_values(
        ["grade", "hybrid_efficiency_score", "paper_score", "risk_score"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    best = summary.iloc[0]
    g30_full = detail[(detail["policy"] == "G30") & (detail["scope"] == "full_history")].iloc[0]
    prod_full = detail[(detail["policy"] == "Baseline") & (detail["scope"] == "full_history")].iloc[0]
    best_full = full[full["hybrid_candidate"] == best["hybrid_candidate"]].iloc[0] if not full.empty else best

    hybrid_gt_g30 = (
        float(best["avg_rfe_reduction_pct"]) > 0
        or float(best["avg_ks_reduction_pct"]) > 0
    ) and float(best["avg_net_delta_vs_g30"]) >= -0.005 and float(best["trade_preservation_score"]) >= 0.85

    struct_robust = best["grade"] in ("A", "B") and float(best["stability_score"]) > 0
    overfit = float(best["stability_score"]) < 0 and float(best["avg_net_delta_vs_g30"]) > 0
    overfiltered = float(best["trade_preservation_score"]) < 0.85
    practical = (
        not overfiltered
        and float(best["removal_precision"]) >= 0.5
        and (float(best["avg_rfe_reduction_pct"]) > 0 or float(best["avg_ks_reduction_pct"]) > 0)
    )

    rec = _recommendation(str(best["grade"]), best, hybrid_gt_g30)
    final_grade = str(best["grade"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"hybrid_gate_replay_{ts}.csv"
    md_path = OUT_DIR / f"hybrid_gate_replay_{ts}.md"

    out = pd.concat(
        [
            detail.assign(section="detail"),
            summary.assign(section="tournament_summary"),
            pd.DataFrame(
                [
                    {
                        "best_hybrid": best["hybrid_candidate"],
                        "hybrid_gt_g30": hybrid_gt_g30,
                        "recommendation": rec,
                        "final_grade": final_grade,
                    }
                ]
            ).assign(section="final_policy"),
        ],
        ignore_index=True,
        sort=False,
    )
    out.to_csv(csv_path, index=False)

    md_lines = [
        "# Hybrid Gate Forensic Replay",
        "",
        f"- best_hybrid: **{best['hybrid_candidate']}**",
        f"- grade: {final_grade}",
        f"- recommendation: {rec}",
        "",
        "## Full History Comparison",
        f"| Policy | trades | net | MDD | RFE | KS |",
        f"|--------|--------|-----|-----|-----|-----|",
        f"| Production | {int(prod_full['trades'])} | {prod_full['net_return']:.6f} | {prod_full['MDD']:.6f} | {int(prod_full['risk_force_exit_count'])} | {bool(prod_full['KS_triggered'])} |",
        f"| G30 | {int(g30_full['trades'])} | {g30_full['net_return']:.6f} | {g30_full['MDD']:.6f} | {int(g30_full['risk_force_exit_count'])} | {bool(g30_full['KS_triggered'])} |",
        f"| {best['hybrid_candidate']} | {int(best_full['trades'])} | {best_full['net_return']:.6f} | {best_full['MDD']:.6f} | {int(best_full['risk_force_exit_count'])} | {bool(best_full['KS_triggered'])} |",
        "",
        "## Best Hybrid vs G30 (rolling avg)",
        f"- RFE reduction: {best['avg_rfe_reduction_pct']:.2f}%",
        f"- KS reduction: {best['avg_ks_reduction_pct']:.2f}%",
        f"- trade preservation vs G30: {best['trade_preservation_score']:.2%}",
        f"- removal precision: {best.get('removal_precision', best_full.get('removal_precision', 0)):.2%}",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    insights = [
        f"Best hybrid {best['hybrid_candidate']} (grade {final_grade}): RFE↓{best['avg_rfe_reduction_pct']:.1f}%, "
        f"preservation {best['trade_preservation_score']:.1%} vs G30.",
        "G30 reduces loss size; hybrid hard-block targets RFE/KS event frequency only if blocks fire pre-entry.",
        f"Hybrid > G30: {hybrid_gt_g30} — net Δ vs G30 rolling avg {best['avg_net_delta_vs_g30']:.6f}.",
        f"Practical={practical}, overfiltered={overfiltered}, structurally_robust={struct_robust}.",
        f"Action: {rec}.",
    ]

    print(f"best hybrid candidate: {best['hybrid_candidate']}")
    print(f"whether hybrid > G30: {hybrid_gt_g30}")
    print(f"RFE reduction result: {best['avg_rfe_reduction_pct']:.2f}% (rolling avg vs G30)")
    print(f"KS reduction result: {best['avg_ks_reduction_pct']:.2f}% (rolling avg vs G30)")
    print(f"trade preservation impact: {best['trade_preservation_score']:.2%} vs G30")
    print(
        f"hybrid character: structurally_robust={struct_robust}, overfit={overfit}, "
        f"overfiltered={overfiltered}, practical={practical}"
    )
    print(f"hybrid should: {rec}")
    print(f"FINAL VERDICT: Grade {final_grade}")
    print("top 5 strategic insights:")
    for i, s in enumerate(insights[:5], 1):
        print(f"  {i}. {s}")
    print(f"csv: {csv_path}")
    print(f"md: {md_path}")


if __name__ == "__main__":
    main()
