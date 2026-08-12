"""
Quality Score Candidate Daily Monitor (diagnostics only).

Compares Production, G30, Hybrid_A, Q2_M3, Q2_PD, Q2_BDI, G30+Q2 overlay.
No live orders, no state mutation, no TradingEngine execution.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.diagnostics.validate_h8_soft_risk_gate import Variant, _entropy, _expectancy, _h8_pass, _simulate_variant
from scripts.diagnostics.validate_hybrid_gate_replay import HYBRID_CANDIDATES, _ohlcv_features, _simulate_hybrid
from scripts.diagnostics.validate_q2_penalty_tournament import apply_penalties
from scripts.diagnostics.validate_quality_score_replay import (
    FAIL_SCALE,
    FEE_RATE,
    POSITION_SIZE,
    SLIPPAGE_RATE,
    _risk_routing,
    _score_calibration,
    _simulate_quality,
    map_m3,
    score_q2,
)
from scripts.run_daily_paper import PAPER_MAX_HOLDING_BARS, load_ohlcv, simulate_signals_paper
from scripts.run_daily_shadow import RECENT_WINDOW_HOURS, diagnose_proba_source, _parse_tick_timestamp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("quality_score_candidate")

MONITORING_DIR = Path("data/monitoring")
MAX_TRADE_LOSS = -0.01
MAX_DAILY_LOSS = -0.02
MAX_DRAWDOWN = -0.05
MAX_CONSEC_LOSSES = 5

DISCLAIMER = (
    "진단 후보 모니터입니다. 실제 주문 없음. 운영 로직 미변경. "
    "Q2 continuous quality score / routing calibration monitor."
)

BASELINE = Variant("Baseline", fail_scale=1.0, hard_block=False)
G30 = Variant("Variant_G30", fail_scale=FAIL_SCALE, hard_block=False)
HYBRID_A = HYBRID_CANDIDATES[0]

PENALTY_D = frozenset({"D"})
PENALTY_BDI = frozenset({"B", "D", "I"})


def _penalty_scorer(active: frozenset[str]):
    def _sc(t: Dict[str, Any], fm: Dict[int, Dict[str, Any]]) -> float:
        return apply_penalties(t, fm, set(active))

    return _sc


SCORE_Q2_PD = _penalty_scorer(PENALTY_D)
SCORE_Q2_BDI = _penalty_scorer(PENALTY_BDI)


def _false_high_count(tdf: pd.DataFrame) -> int:
    if tdf.empty:
        return 0
    return int(((tdf["quality_score"] >= 0.80) & (tdf["scaled_return"] < 0)).sum())


def _calibration_summary(tdf: pd.DataFrame) -> Dict[str, Any]:
    cal_df, mono = _score_calibration(tdf)
    hi = cal_df[cal_df["bucket"].astype(str).str.startswith("0.8")] if not cal_df.empty else pd.DataFrame()
    hi_exp = float(hi["expectancy"].iloc[0]) if not hi.empty else 0.0
    return {
        "calibration_monotonic": bool(mono),
        "false_high_count": _false_high_count(tdf),
        "false_high_eliminated": _false_high_count(tdf) == 0,
        "hi_bucket_expectancy": hi_exp,
        "calibration_buckets": cal_df.to_dict(orient="records") if not cal_df.empty else [],
    }


def _q2_overlay_mult(score: float) -> float:
    if score >= 0.80:
        return 1.00
    if score >= 0.65:
        return 0.80
    if score >= 0.50:
        return 0.60
    return 0.40


def _simulate_g30_q2_overlay(
    ticks: List[Dict[str, Any]],
    feat_map: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    trades: List[Dict[str, Any]] = []
    open_pos: Optional[Dict[str, Any]] = None
    eq = peak = 1.0
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

        h8 = _h8_pass(t)
        qscore = score_q2(t, feat_map)
        g30_scale = 1.0 if h8 else FAIL_SCALE
        scale = g30_scale * _q2_overlay_mult(qscore)
        side = "BUY" if sig == "LONG" else "SELL"
        open_pos = {
            "entry_idx": i,
            "entry_price": px,
            "direction": "LONG" if side == "BUY" else "SHORT",
            "side": side,
            "hold_bars": 0,
            "scale": scale,
            "quality_score": qscore,
            "h8_pass": h8,
            "g30_base_scale": g30_scale,
        }

    tdf = pd.DataFrame(trades)
    vals = tdf["scaled_return"].tolist() if not tdf.empty else []
    wins = [x for x in vals if x > 0]
    losses = [x for x in vals if x < 0]
    wr = len(wins) / len(vals) if vals else 0.0
    pf = (sum(wins) / abs(sum(losses))) if losses else 0.0
    eq_f = peak_f = 1.0
    mdd_min = 0.0
    for r in vals:
        eq_f *= 1.0 + r * POSITION_SIZE
        peak_f = max(peak_f, eq_f)
        mdd_min = min(mdd_min, (eq_f - peak_f) / peak_f if peak_f > 0 else 0.0)

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
        "avg_scale": float(tdf["scale"].mean()) if not tdf.empty else 0.0,
    }
    return metrics, tdf


def _slice_recent_ticks(ticks: List[Dict[str, Any]], hours: float = RECENT_WINDOW_HOURS) -> List[Dict[str, Any]]:
    ts_list = [(i, _parse_tick_timestamp(t)) for i, t in enumerate(ticks)]
    valid = [(i, ts) for i, ts in ts_list if ts is not None]
    if not valid:
        return []
    ts_max = max(ts for _, ts in valid)
    cutoff = ts_max - pd.Timedelta(hours=hours)
    return [ticks[i] for i, ts in valid if ts >= cutoff]


def _metrics_pack(
    name: str,
    m: Dict[str, Any],
    tdf: pd.DataFrame,
    ticks: List[Dict[str, Any]],
    prod_trades: int,
    g30_trades: int,
    with_calibration: bool = False,
) -> Dict[str, Any]:
    rr = _risk_routing(tdf, ticks)
    routing_valid = rr["avg_scale_winners"] > rr["avg_scale_losers"] > rr["avg_scale_rfe"] if m["trades"] > 0 else False
    eq_smooth = 0.0
    if not tdf.empty and len(tdf) > 1:
        eq_smooth = 1.0 / (float(np.std(tdf["scaled_return"])) + 1e-9)
    pack: Dict[str, Any] = {
        "name": name,
        "trade_count": int(m["trades"]),
        "win_rate": float(m["win_rate"]),
        "expectancy": float(m["expectancy"]),
        "net_return": float(m["net_return"]),
        "final_equity": float(m.get("final_equity", 1.0)),
        "max_drawdown": float(m["MDD"]),
        "risk_force_exit_count": int(m["risk_force_exit_count"]),
        "kill_switch_triggered": bool(m["KS_triggered"]),
        "avg_hold_bars": float(m["avg_hold"]),
        "avg_scale": float(m.get("avg_scale", 1.0)),
        "trade_preservation_vs_prod": m["trades"] / max(prod_trades, 1),
        "trade_preservation_vs_g30": m["trades"] / max(g30_trades, 1),
        "equity_smoothness": eq_smooth,
        **rr,
        "routing_valid": routing_valid,
    }
    if with_calibration:
        pack.update(_calibration_summary(tdf))
    return pack


def _daily_verdict(v24: Dict[str, Dict[str, Any]]) -> str:
    keys = ("G30", "Q2_M3", "Q2_PD", "Q2_BDI", "G30_Q2")
    if sum(v24[k]["trade_count"] for k in keys if k in v24) < 5:
        return "INSUFFICIENT_SAMPLE"
    g30 = v24["G30"]
    candidates = [
        ("Q2_M3", v24.get("Q2_M3", {})),
        ("Q2_PD", v24.get("Q2_PD", {})),
        ("Q2_BDI", v24.get("Q2_BDI", {})),
        ("G30_Q2", v24.get("G30_Q2", {})),
    ]
    scored = []
    for name, v in candidates:
        if not v:
            continue
        scored.append(
            (
                name,
                v["net_return"] > g30["net_return"] and v["max_drawdown"] >= g30["max_drawdown"],
                v["net_return"],
                v["max_drawdown"],
            )
        )
    if not scored:
        return "QUALITY_UNCLEAR"
    best = max(scored, key=lambda x: (x[1], x[2], x[3]))
    if best[0] == "Q2_PD" and best[1]:
        return "QUALITY_PD_BETTER"
    if best[0] == "Q2_BDI" and best[1]:
        return "QUALITY_BDI_BETTER"
    if best[0] in ("Q2_M3", "G30_Q2") and best[1]:
        return "QUALITY_BETTER"
    if any(x[1] for x in scored):
        return "QUALITY_BETTER"
    return "QUALITY_UNCLEAR"


def _structural_notes(vf: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    q2 = vf["Q2_M3"]
    pd_v = vf["Q2_PD"]
    bdi = vf["Q2_BDI"]
    ov = vf["G30_Q2"]
    hyb = vf["Hybrid_A"]
    improving = pd_v["net_return"] > hyb["net_return"] or bdi["net_return"] > hyb["net_return"]
    stable = all(vf[k]["trade_preservation_vs_g30"] >= 0.90 for k in ("Q2_M3", "Q2_PD", "Q2_BDI"))
    overfiltering = any(vf[k]["trade_preservation_vs_prod"] < 0.85 for k in ("Q2_M3", "Q2_PD", "Q2_BDI"))
    return {
        "improving": improving,
        "stable": stable,
        "overfiltering": overfiltering,
        "unstable": False,
        "calibration_valid": q2.get("calibration_monotonic", False) or pd_v.get("calibration_monotonic", False),
        "routing_valid_q2_m3": q2["routing_valid"],
        "routing_valid_q2_pd": pd_v["routing_valid"],
        "routing_valid_q2_bdi": bdi["routing_valid"],
        "routing_valid_g30_q2": ov["routing_valid"],
        "false_high_eliminated_pd": pd_v.get("false_high_eliminated", False),
        "false_high_eliminated_bdi": bdi.get("false_high_eliminated", False),
        "calibration_monotonic_pd": pd_v.get("calibration_monotonic", False),
        "calibration_monotonic_bdi": bdi.get("calibration_monotonic", False),
    }


def _strategic_notes(report: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    vf = report["variants_full"]
    q2 = vf["Q2_M3"]
    pd_v = vf["Q2_PD"]
    bdi = vf["Q2_BDI"]
    hyb = vf["Hybrid_A"]
    ov = vf["G30_Q2"]
    sn = report["structural_notes"]

    if pd_v.get("false_high_eliminated"):
        notes.append("Penalty_D successfully suppressing false high-score losers.")
    if bdi.get("calibration_monotonic"):
        notes.append("Combo_B+D+I improving calibration monotonicity vs Q2_M3.")
    if pd_v["routing_valid"] or bdi["routing_valid"]:
        notes.append("Routing quality now valid on penalized Q2 variants (win>loss>RFE).")
    if bdi["net_return"] > hyb["net_return"] and bdi["max_drawdown"] >= hyb["max_drawdown"]:
        notes.append("Continuous routing (Q2_BDI) outperforming Hybrid_A on net/MDD.")
    if pd_v["trade_preservation_vs_prod"] >= 0.95:
        notes.append("Trade preservation remains healthy after penalty layer.")
    if q2.get("false_high_count", 0) > 0 and pd_v.get("false_high_count", 0) < q2["false_high_count"]:
        notes.append("Score inflation reduced — false high count dropped with Penalty_D.")
    if q2.get("false_high_count", 0) > 0:
        notes.append("Q2_M3 still shows high_vol LONG overconfidence without penalties.")
    if not q2.get("calibration_monotonic") and bdi.get("calibration_monotonic"):
        notes.append("Penalty combo fixes calibration where base Q2_M3 fails.")
    if ov["net_return"] > q2["net_return"]:
        notes.append("G30+Q2 overlay still competitive vs base Q2_M3 on net.")
    return notes[:8]


def build_report(ticks: List[Dict[str, Any]], proba_meta: Dict[str, Any], feat_map: Dict[int, Dict[str, Any]], dry_run: bool = False) -> Dict[str, Any]:
    ts_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    proba_diag = diagnose_proba_source(None, proba_meta)
    recent = _slice_recent_ticks(ticks)

    prod_m, prod_tdf = _simulate_variant(ticks, BASELINE, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    g30_m, g30_tdf = _simulate_variant(ticks, G30, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    hyb_m, hyb_tdf, _ = _simulate_hybrid(ticks, HYBRID_A, feat_map, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    q2_m, q2_tdf, _ = _simulate_quality(ticks, score_q2, map_m3, feat_map)
    pd_m, pd_tdf, _ = _simulate_quality(ticks, SCORE_Q2_PD, map_m3, feat_map)
    bdi_m, bdi_tdf, _ = _simulate_quality(ticks, SCORE_Q2_BDI, map_m3, feat_map)
    ov_m, ov_tdf = _simulate_g30_q2_overlay(ticks, feat_map)

    prod_tr = prod_m["trades"]
    g30_tr = g30_m["trades"]

    variants_full = {
        "Production": _metrics_pack("Production", prod_m, prod_tdf, ticks, prod_tr, g30_tr),
        "G30": _metrics_pack("G30", g30_m, g30_tdf, ticks, prod_tr, g30_tr),
        "Hybrid_A": _metrics_pack("Hybrid_A", hyb_m, hyb_tdf, ticks, prod_tr, g30_tr),
        "Q2_M3": _metrics_pack("Q2_M3", q2_m, q2_tdf, ticks, prod_tr, g30_tr, with_calibration=True),
        "Q2_PD": _metrics_pack("Q2_PD", pd_m, pd_tdf, ticks, prod_tr, g30_tr, with_calibration=True),
        "Q2_BDI": _metrics_pack("Q2_BDI", bdi_m, bdi_tdf, ticks, prod_tr, g30_tr, with_calibration=True),
        "G30_Q2": _metrics_pack("G30_Q2", ov_m, ov_tdf, ticks, prod_tr, g30_tr),
    }

    cal_mono = any(
        variants_full[k].get("calibration_monotonic", False) for k in ("Q2_M3", "Q2_PD", "Q2_BDI")
    )

    def _run24(w_ticks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        p_m, p_t = _simulate_variant(w_ticks, BASELINE, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
        g_m, g_t = _simulate_variant(w_ticks, G30, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
        pt, gt = p_m["trades"], g_m["trades"]
        q_m, q_t, _ = _simulate_quality(w_ticks, score_q2, map_m3, feat_map)
        pd_m24, pd_t24, _ = _simulate_quality(w_ticks, SCORE_Q2_PD, map_m3, feat_map)
        bdi_m24, bdi_t24, _ = _simulate_quality(w_ticks, SCORE_Q2_BDI, map_m3, feat_map)
        o_m, o_t = _simulate_g30_q2_overlay(w_ticks, feat_map)
        return {
            "G30": _metrics_pack("G30", g_m, g_t, w_ticks, pt, gt),
            "Q2_M3": _metrics_pack("Q2_M3", q_m, q_t, w_ticks, pt, gt, with_calibration=True),
            "Q2_PD": _metrics_pack("Q2_PD", pd_m24, pd_t24, w_ticks, pt, gt, with_calibration=True),
            "Q2_BDI": _metrics_pack("Q2_BDI", bdi_m24, bdi_t24, w_ticks, pt, gt, with_calibration=True),
            "G30_Q2": _metrics_pack("G30_Q2", o_m, o_t, w_ticks, pt, gt),
        }

    v24 = _run24(recent) if recent else {k: variants_full[k] for k in ("G30", "Q2_M3", "Q2_PD", "Q2_BDI", "G30_Q2")}
    verdict = _daily_verdict(v24)
    structural = _structural_notes(variants_full)

    cont_keys = ("Hybrid_A", "Q2_M3", "Q2_PD", "Q2_BDI")
    hybrid_vs = {
        "lowest_rfe": min(
            [(k, variants_full[k]["risk_force_exit_count"]) for k in cont_keys],
            key=lambda x: x[1],
        )[0],
        "best_preservation": max(
            [(k, variants_full[k]["trade_preservation_vs_prod"]) for k in cont_keys],
            key=lambda x: x[1],
        )[0],
        "best_mdd": max(
            [(k, variants_full[k]["max_drawdown"]) for k in cont_keys],
            key=lambda x: x[1],
        )[0],
        "smoothest_equity": max(
            [(k, variants_full[k]["equity_smoothness"]) for k in cont_keys],
            key=lambda x: x[1],
        )[0],
        "best_calibration": max(
            [(k, int(variants_full[k].get("calibration_monotonic", False))) for k in ("Q2_M3", "Q2_PD", "Q2_BDI")],
            key=lambda x: x[1],
        )[0],
        "best_routing": max(
            [(k, int(variants_full[k]["routing_valid"])) for k in ("Q2_M3", "Q2_PD", "Q2_BDI")],
            key=lambda x: x[1],
        )[0],
    }

    report: Dict[str, Any] = {
        "mode": "quality_score_monitor",
        "disclaimer": DISCLAIMER,
        "run_timestamp": ts_run,
        "dry_run": dry_run,
        "proba_health": {
            "proba_loaded": proba_diag.get("proba_loaded"),
            "fallback_used": proba_diag.get("fallback", {}).get("fallback_used"),
            "aligned_ratio": proba_diag.get("alignment", {}).get("aligned_ratio"),
            "ohlcv_ts_max": proba_diag.get("alignment", {}).get("ohlcv_ts_max"),
            "proba_ts_max": proba_diag.get("alignment", {}).get("proba_ts_max"),
        },
        "variants_full": variants_full,
        "compare_24h": v24,
        "calibration_q2_m3": variants_full["Q2_M3"].get("calibration_buckets", []),
        "calibration_q2_pd": variants_full["Q2_PD"].get("calibration_buckets", []),
        "calibration_q2_bdi": variants_full["Q2_BDI"].get("calibration_buckets", []),
        "calibration_monotonic": cal_mono,
        "hybrid_vs_continuous": hybrid_vs,
        "structural_notes": structural,
        "daily_verdict": verdict,
        "proba_source_diagnostics": proba_diag,
    }
    report["strategic_notes"] = _strategic_notes(report)
    return report


def _write_markdown(report: Dict[str, Any], path: Path) -> None:
    vf = report["variants_full"]
    lines = [
        "# Quality Score Candidate Daily Monitor",
        "",
        report["disclaimer"],
        "",
        f"- Run: {report['run_timestamp']}",
        "",
        "## Full Replay Summary",
        "| Variant | trades | net | MDD | RFE | final_eq |",
        "|---------|--------|-----|-----|-----|----------|",
    ]
    for k in ("Production", "G30", "Hybrid_A", "Q2_M3", "Q2_PD", "Q2_BDI", "G30_Q2"):
        v = vf[k]
        lines.append(
            f"| {k} | {v['trade_count']} | {v['net_return']:.6f} | {v['max_drawdown']:.6f} | "
            f"{v['risk_force_exit_count']} | {v['final_equity']:.6f} |"
        )
    lines += [
        "",
        "## Calibration (Q2 variants)",
        f"- Q2_M3: mono={vf['Q2_M3'].get('calibration_monotonic')} false_high={vf['Q2_M3'].get('false_high_count', 0)}",
        f"- Q2_PD: mono={vf['Q2_PD'].get('calibration_monotonic')} false_high={vf['Q2_PD'].get('false_high_count', 0)}",
        f"- Q2_BDI: mono={vf['Q2_BDI'].get('calibration_monotonic')} false_high={vf['Q2_BDI'].get('false_high_count', 0)}",
        "",
        f"## Daily Verdict (24h): **{report['daily_verdict']}**",
        f"- calibration_monotonic: {report['calibration_monotonic']}",
        "",
        "## Strategic Notes",
    ]
    for n in report["strategic_notes"]:
        lines.append(f"- {n}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def send_discord(report: Dict[str, Any], json_path: str, md_path: str, dry_run: bool = False) -> bool:
    import os

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except Exception:
        pass

    webhook = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook and not dry_run:
        logger.warning("[Discord] DISCORD_WEBHOOK_URL missing")
        return False

    ph = report["proba_health"]
    vf = report["variants_full"]
    sn = report["structural_notes"]
    hvc = report["hybrid_vs_continuous"]

    def _row(k: str) -> str:
        v = vf[k]
        return f"{k}: tr={v['trade_count']} net={v['net_return']:.6f} MDD={v['max_drawdown']:.4f} RFE={v['risk_force_exit_count']} eq={v['final_equity']:.6f}"

    def _cal_block(label: str, key: str) -> str:
        v = vf[key]
        rows = report.get(f"calibration_{key.lower()}", report.get("calibration_q2_m3", []))
        if not rows:
            rows = v.get("calibration_buckets", [])
        lines = [
            f"**{label}** mono={v.get('calibration_monotonic')} false_high={v.get('false_high_count', 0)}"
        ]
        for row in rows[:5]:
            lines.append(f"  {row.get('bucket')}: n={row.get('trade_count')} exp={row.get('expectancy', 0):.6f}")
        return "\n".join(lines)

    cal_side = "\n\n".join([
        _cal_block("Q2_M3", "Q2_M3"),
        _cal_block("Q2_PD", "Q2_PD"),
        _cal_block("Q2_BDI", "Q2_BDI"),
    ])

    strat = "\n".join(f"• {n}" for n in report.get("strategic_notes", [])[:6])

    fields = [
        {
            "name": "📋 [0] 실행 정보",
            "value": (
                f"**mode**: quality_score_monitor\n"
                f"**{report['disclaimer']}**\n"
                f"**timestamp**: {report['run_timestamp']}\n"
                f"diagnostic only | no real order | production untouched"
            ),
            "inline": False,
        },
        {
            "name": "📊 [1] Data / Proba Health",
            "value": (
                f"proba_loaded: {ph['proba_loaded']}\n"
                f"fallback: {ph['fallback_used']}\n"
                f"aligned_ratio: {ph.get('aligned_ratio')}\n"
                f"ohlcv_ts_max: {ph.get('ohlcv_ts_max')}\n"
                f"proba_ts_max: {ph.get('proba_ts_max')}"
            ),
            "inline": False,
        },
        {
            "name": "⚖️ [2] Full Replay Summary",
            "value": "\n".join(_row(k) for k in ("Production", "G30", "Hybrid_A", "Q2_M3", "Q2_PD", "Q2_BDI", "G30_Q2")),
            "inline": False,
        },
        {
            "name": "📈 [3] Trade Preservation",
            "value": (
                f"Q2_M3: {vf['Q2_M3']['trade_preservation_vs_prod']:.1%} / {vf['Q2_M3']['trade_preservation_vs_g30']:.1%}\n"
                f"Q2_PD: {vf['Q2_PD']['trade_preservation_vs_prod']:.1%} / {vf['Q2_PD']['trade_preservation_vs_g30']:.1%}\n"
                f"Q2_BDI: {vf['Q2_BDI']['trade_preservation_vs_prod']:.1%} / {vf['Q2_BDI']['trade_preservation_vs_g30']:.1%}\n"
                f"Hybrid_A: {vf['Hybrid_A']['trade_preservation_vs_prod']:.1%}"
            ),
            "inline": False,
        },
        {
            "name": "🔀 [4] Routing Quality",
            "value": (
                f"Q2_M3: win={vf['Q2_M3']['avg_scale_winners']:.3f} loss={vf['Q2_M3']['avg_scale_losers']:.3f} "
                f"RFE={vf['Q2_M3']['avg_scale_rfe']:.3f} valid={vf['Q2_M3']['routing_valid']}\n"
                f"Q2_PD: win={vf['Q2_PD']['avg_scale_winners']:.3f} loss={vf['Q2_PD']['avg_scale_losers']:.3f} "
                f"RFE={vf['Q2_PD']['avg_scale_rfe']:.3f} valid={vf['Q2_PD']['routing_valid']}\n"
                f"Q2_BDI: win={vf['Q2_BDI']['avg_scale_winners']:.3f} loss={vf['Q2_BDI']['avg_scale_losers']:.3f} "
                f"RFE={vf['Q2_BDI']['avg_scale_rfe']:.3f} valid={vf['Q2_BDI']['routing_valid']}\n"
                f"G30+Q2: valid={vf['G30_Q2']['routing_valid']}"
            ),
            "inline": False,
        },
        {
            "name": "📉 [5] Score Calibration",
            "value": cal_side,
            "inline": False,
        },
        {
            "name": "🆚 [6] Hybrid vs Continuous",
            "value": (
                f"lowest RFE: {hvc['lowest_rfe']}\n"
                f"best preservation: {hvc['best_preservation']}\n"
                f"best MDD: {hvc['best_mdd']}\n"
                f"smoothest equity: {hvc['smoothest_equity']}\n"
                f"best calibration: {hvc['best_calibration']}\n"
                f"best routing: {hvc['best_routing']}"
            ),
            "inline": False,
        },
        {
            "name": "🔍 [7] Structural Diagnostics",
            "value": (
                f"improving={sn['improving']} stable={sn['stable']} overfiltering={sn['overfiltering']}\n"
                f"Q2_PD: false_high_elim={sn['false_high_eliminated_pd']} "
                f"routing={sn['routing_valid_q2_pd']} mono={sn['calibration_monotonic_pd']}\n"
                f"Q2_BDI: false_high_elim={sn['false_high_eliminated_bdi']} "
                f"routing={sn['routing_valid_q2_bdi']} mono={sn['calibration_monotonic_bdi']}\n"
                f"Q2_M3 routing={sn['routing_valid_q2_m3']}"
            ),
            "inline": False,
        },
        {
            "name": "✅ [8] Final Daily Verdict",
            "value": f"**{report['daily_verdict']}**",
            "inline": True,
        },
        {
            "name": "💡 [9] Strategic Notes",
            "value": strat or "n/a",
            "inline": False,
        },
        {
            "name": "📁 Logs",
            "value": f"JSON: `{json_path}`\nMD: `{md_path}`",
            "inline": False,
        },
    ]

    verdict = report["daily_verdict"]
    color = 0x2ecc71 if verdict in ("QUALITY_BETTER", "QUALITY_PD_BETTER", "QUALITY_BDI_BETTER") else 0xf1c40f
    embed = {
        "title": "【Quality Score Ops】Q2 Continuous Routing 일일 요약",
        "description": report["disclaimer"],
        "color": color,
        "fields": fields,
        "footer": {"text": "Can_bit Quality Score Monitor (diagnostic only)"},
        "timestamp": datetime.utcnow().isoformat(),
    }
    payload = {"embeds": [embed]}

    if dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return True

    try:
        import requests

        resp = requests.post(webhook, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("[Discord] Quality score candidate report sent")
        return True
    except Exception as e:
        logger.error("[Discord] send failed: %s", e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality Score candidate daily monitor")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-discord", action="store_true")
    args = parser.parse_args()

    df = load_ohlcv()
    if df is None or df.empty:
        logger.error("OHLCV load failed")
        sys.exit(1)

    feat_map = _ohlcv_features(df)
    ticks, proba_meta = simulate_signals_paper(df)
    if not ticks:
        logger.error("No ticks")
        sys.exit(1)

    report = build_report(ticks, proba_meta, feat_map, dry_run=args.dry_run)

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = MONITORING_DIR / f"quality_score_candidate_daily_report_{ts_file}.json"
    md_path = MONITORING_DIR / f"quality_score_candidate_daily_report_{ts_file}.md"
    log_path = MONITORING_DIR / f"quality_score_candidate_log_{datetime.now().strftime('%Y%m%d')}.jsonl"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _write_markdown(report, md_path)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": report["run_timestamp"], "verdict": report["daily_verdict"], "report": report}, default=str) + "\n")

    discord_ok = False
    if not args.no_discord:
        discord_ok = send_discord(report, str(json_path), str(md_path), dry_run=args.dry_run)
    elif args.dry_run:
        discord_ok = send_discord(report, str(json_path), str(md_path), dry_run=True)

    q2 = report["variants_full"]["Q2_M3"]
    pd_v = report["variants_full"]["Q2_PD"]
    bdi = report["variants_full"]["Q2_BDI"]
    sn = report["structural_notes"]

    print("[QUALITY SCORE CANDIDATE VALIDATION]")
    print(f"monitor_update_succeeded: True")
    print(f"discord_payload_updated: {discord_ok or args.no_discord}")
    print(
        f"Q2_M3: trades={q2['trade_count']} net={q2['net_return']:.6f} MDD={q2['max_drawdown']:.6f} "
        f"RFE={q2['risk_force_exit_count']} routing={q2['routing_valid']} false_high={q2.get('false_high_count', 0)}"
    )
    print(
        f"Q2_PD: trades={pd_v['trade_count']} net={pd_v['net_return']:.6f} MDD={pd_v['max_drawdown']:.6f} "
        f"RFE={pd_v['risk_force_exit_count']} routing={pd_v['routing_valid']} false_high={pd_v.get('false_high_count', 0)}"
    )
    print(
        f"Q2_BDI: trades={bdi['trade_count']} net={bdi['net_return']:.6f} MDD={bdi['max_drawdown']:.6f} "
        f"RFE={bdi['risk_force_exit_count']} routing={bdi['routing_valid']} false_high={bdi.get('false_high_count', 0)}"
    )
    print(
        f"false_high_eliminated: PD={sn['false_high_eliminated_pd']} BDI={sn['false_high_eliminated_bdi']} | "
        f"routing_valid: PD={sn['routing_valid_q2_pd']} BDI={sn['routing_valid_q2_bdi']} | "
        f"calibration_monotonic: PD={sn['calibration_monotonic_pd']} BDI={sn['calibration_monotonic_bdi']}"
    )
    print(
        "next_monitoring_focus: track Q2_PD/Q2_BDI false_high, routing_valid, calibration_monotonic daily; "
        "compare vs Hybrid_A RFE/MDD"
    )
    print(f"files: {json_path}, {md_path}, {log_path}, data/ops_logs/launchd_quality_score_candidate_stdout.log")


if __name__ == "__main__":
    main()
