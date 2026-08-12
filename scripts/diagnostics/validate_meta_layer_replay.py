"""
Validate Meta Layer V2 replay vs handcrafted baselines (diagnostics only).

Outputs monitor-candidate reports — no live routing changes.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from scripts.diagnostics.build_meta_label_dataset import _entry_features_v2, _ohlcv_features, _rolling_context, load_v2_dataset
from scripts.diagnostics.validate_h8_soft_risk_gate import Variant, _entropy, _expectancy, _is_loss_cluster_trade, _simulate_variant
from scripts.diagnostics.validate_hybrid_gate_replay import HYBRID_CANDIDATES, _simulate_hybrid
from scripts.diagnostics.validate_q2_penalty_tournament import apply_penalties
from scripts.diagnostics.validate_quality_score_replay import (
    FEE_RATE,
    POSITION_SIZE,
    SLIPPAGE_RATE,
    _risk_routing,
    _simulate_quality,
    map_m3,
)
from scripts.run_daily_paper import PAPER_MAX_HOLDING_BARS, load_ohlcv, simulate_signals_paper

OUT_DIR = Path("data/diagnostics/meta_layer")
BASELINE = Variant("Baseline", fail_scale=1.0, hard_block=False)
G30 = Variant("Variant_G30", fail_scale=0.30, hard_block=False)
HYBRID_A = HYBRID_CANDIDATES[0]
PENALTY_D = frozenset({"D"})
PENALTY_BDI = frozenset({"B", "D", "I"})
SCORE_Q2_PD = lambda t, fm: apply_penalties(t, fm, set(PENALTY_D))
SCORE_Q2_BDI = lambda t, fm: apply_penalties(t, fm, set(PENALTY_BDI))

MAX_TRADE_LOSS = -0.01
MAX_DAILY_LOSS = -0.02
MAX_DRAWDOWN = -0.05
MAX_CONSEC_LOSSES = 5


def _load_artifact() -> Dict[str, Any]:
    pointer = OUT_DIR / "meta_layer_model_latest.json"
    if not pointer.exists():
        cands = sorted(OUT_DIR.glob("meta_layer_v2_model_*.joblib"))
        if not cands:
            raise SystemExit("No V2 model. Run train_meta_layer_model first.")
        return joblib.load(cands[-1])
    path = Path(json.loads(pointer.read_text(encoding="utf-8"))["artifact_path"])
    return joblib.load(path)


def _predict_meta_v2(artifact: Dict[str, Any], feats: Dict[str, Any]) -> Tuple[float, float]:
    fcols = artifact["feature_cols"]
    row = {c: feats.get(c, 0) for c in fcols}
    x = pd.DataFrame([row], columns=fcols).fillna(0)
    for col in x.columns:
        if x[col].dtype == bool:
            x[col] = x[col].astype(int)

    raw_prob = float(artifact["calibration_model"].predict_proba(x)[0, 1])
    cal_prob = float(artifact["isotonic_calibrator"].predict([raw_prob])[0])

    scale = float(artifact["scale_regressor"].predict(x)[0])
    # Risk-aware blend: downscale when calibration fail likely
    scale = scale * (1.0 - 0.85 * cal_prob)
    scale = max(0.05, min(1.0, scale))
    return cal_prob, scale


def _simulate_meta_v2(
    ticks: List[Dict[str, Any]],
    feat_map: Dict[int, Dict[str, Any]],
    artifact: Dict[str, Any],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    trades: List[Dict[str, Any]] = []
    closed: List[Dict[str, Any]] = []
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
                (open_pos["side"] == "BUY" and sig == "SHORT") or (open_pos["side"] == "SELL" and sig == "LONG")
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
            trades.append({
                **open_pos, "exit_reason": reason or "exit_signal",
                "net_return": net, "scaled_return": net * scale,
            })
            closed.append({"net_return": net, "false_high_signature_flag": open_pos.get("false_high_signature_flag", 0)})
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
                trades.append({**open_pos, "exit_reason": "risk_force_exit", "net_return": net, "scaled_return": net * scale})
                risk_force += 1
                closed.append({"net_return": net})
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

        direction = "LONG" if sig == "LONG" else "SHORT"
        ctx = _rolling_context(closed)
        feats = _entry_features_v2(t, ticks, i, feat_map, direction, ctx)
        cal_prob, scale = _predict_meta_v2(artifact, feats)
        if scale <= 0.05 and cal_prob >= 0.75:
            continue

        side = "BUY" if sig == "LONG" else "SELL"
        open_pos = {
            "entry_idx": i, "entry_price": px, "direction": direction, "side": side,
            "hold_bars": 0, "scale": scale, "cal_prob": cal_prob,
            "false_high_signature_flag": feats.get("false_high_signature_flag", 0),
        }

    tdf = pd.DataFrame(trades)
    vals = tdf["scaled_return"].tolist() if not tdf.empty else []
    eq_f = peak_f = 1.0
    mdd_min = 0.0
    for r in vals:
        eq_f *= 1.0 + r * POSITION_SIZE
        peak_f = max(peak_f, eq_f)
        mdd_min = min(mdd_min, (eq_f - peak_f) / peak_f if peak_f > 0 else 0.0)

    return {
        "trades": int(len(tdf)),
        "net_return": float(np.sum(vals)) if vals else 0.0,
        "final_equity": float(eq_f),
        "MDD": float(mdd_min),
        "risk_force_exit_count": int(risk_force),
        "KS_triggered": int(bool(ks_triggered)),
        "avg_scale": float(tdf["scale"].mean()) if not tdf.empty else 0.0,
        "expectancy": _expectancy(vals),
    }, tdf


def _false_high_suppression(tdf: pd.DataFrame, ticks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if tdf.empty:
        return {"false_high_count": 0, "false_high_suppression_rate": 0.0}
    fh = 0
    for _, tr in tdf.iterrows():
        t = ticks[int(tr["entry_idx"])]
        is_fh = (
            tr.get("direction", "LONG") == "LONG"
            and str(t.get("trend_label")) == "up"
            and str(t.get("vol_bucket")) == "high"
            and _entropy(t) <= 0.90
            and float(tr["scaled_return"]) < 0
        )
        fh += int(is_fh)
    return {"false_high_count": fh, "false_high_suppression_rate": 1.0 - fh / max(len(tdf), 1)}


def _routing_monotonicity(tdf: pd.DataFrame, ticks: List[Dict[str, Any]]) -> Dict[str, Any]:
    rr = _risk_routing(tdf, ticks)
    valid = rr["avg_scale_winners"] > rr["avg_scale_losers"] > rr["avg_scale_rfe"] if not tdf.empty else False
    return {**rr, "routing_valid": valid, "routing_monotonicity": valid}


def _top_bucket_expectancy(tdf: pd.DataFrame) -> float:
    if tdf.empty or "cal_prob" not in tdf.columns:
        return 0.0
    top = tdf[tdf["cal_prob"] <= tdf["cal_prob"].quantile(0.25)]
    if top.empty:
        return 0.0
    return float(top["scaled_return"].mean())


def _metrics_row(name: str, m: Dict[str, Any], tdf: pd.DataFrame, ticks: List[Dict[str, Any]], prod_trades: int, prod_mdd: float) -> Dict[str, Any]:
    rr = _routing_monotonicity(tdf, ticks)
    fh = _false_high_suppression(tdf, ticks)
    return {
        "candidate": name,
        "trade_count": m["trades"],
        "net_return": m["net_return"],
        "max_drawdown": m["MDD"],
        "risk_force_exit_count": m["risk_force_exit_count"],
        "final_equity": m.get("final_equity", 1.0),
        "avg_scale": m.get("avg_scale", 1.0),
        "trade_preservation_vs_prod": m["trades"] / max(prod_trades, 1),
        "preservation_ratio": m["trades"] / max(prod_trades, 1),
        "drawdown_reduction_vs_prod": m["MDD"] - prod_mdd,
        "top_bucket_expectancy": _top_bucket_expectancy(tdf),
        **rr,
        **fh,
    }


def _deployment_verdict(meta_row: Dict[str, Any], artifact: Dict[str, Any], q2_pd: Dict[str, Any]) -> str:
    diag = artifact.get("model_diagnostics", {})
    test_auc = artifact.get("test_metrics", {}).get("auc", 0)
    if diag.get("sample_size_warning", True) or test_auc < 0.52:
        return "reject"
    if test_auc < 0.58 or diag.get("overfit_risk", True):
        return "monitor_only"
    if meta_row["routing_valid"] and meta_row["max_drawdown"] >= q2_pd["max_drawdown"] and test_auc >= 0.62:
        return "candidate"
    return "monitor_only"


def run_replay() -> Tuple[pd.DataFrame, Dict[str, Any], Path, Path]:
    artifact = _load_artifact()
    pointer = OUT_DIR / "meta_layer_model_latest.json"
    ptr = json.loads(pointer.read_text(encoding="utf-8")) if pointer.exists() else {}

    ohlcv = load_ohlcv()
    if ohlcv is None or ohlcv.empty:
        raise SystemExit("OHLCV load failed")
    feat_map, _ = _ohlcv_features(ohlcv)
    ticks, _ = simulate_signals_paper(ohlcv)
    if not ticks:
        raise SystemExit("No ticks")

    prod_m, prod_tdf = _simulate_variant(ticks, BASELINE, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    g30_m, g30_tdf = _simulate_variant(ticks, G30, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    hyb_m, hyb_tdf, _ = _simulate_hybrid(ticks, HYBRID_A, feat_map, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    pd_m, pd_tdf, _ = _simulate_quality(ticks, SCORE_Q2_PD, map_m3, feat_map)
    bdi_m, bdi_tdf, _ = _simulate_quality(ticks, SCORE_Q2_BDI, map_m3, feat_map)
    meta_m, meta_tdf = _simulate_meta_v2(ticks, feat_map, artifact)

    prod_tr = prod_m["trades"]
    prod_mdd = prod_m["MDD"]
    rows = [
        _metrics_row("Production", prod_m, prod_tdf, ticks, prod_tr, prod_mdd),
        _metrics_row("G30", g30_m, g30_tdf, ticks, prod_tr, prod_mdd),
        _metrics_row("Hybrid_A", hyb_m, hyb_tdf, ticks, prod_tr, prod_mdd),
        _metrics_row("Q2_PD", pd_m, pd_tdf, ticks, prod_tr, prod_mdd),
        _metrics_row("Q2_BDI", bdi_m, bdi_tdf, ticks, prod_tr, prod_mdd),
        _metrics_row("Meta_V2_candidate", meta_m, meta_tdf, ticks, prod_tr, prod_mdd),
    ]
    result_df = pd.DataFrame(rows)

    meta_row = next(r for r in rows if r["candidate"] == "Meta_V2_candidate")
    q2_pd = next(r for r in rows if r["candidate"] == "Q2_PD")
    q2_bdi = next(r for r in rows if r["candidate"] == "Q2_BDI")

    summary = {
        "meta_v2_version": "2.0",
        "artifact_model": artifact.get("best_model_key"),
        "meta_routing_valid": meta_row["routing_valid"],
        "meta_routing_monotonicity": meta_row["routing_monotonicity"],
        "false_high_count_meta": meta_row["false_high_count"],
        "false_high_count_q2_pd": _false_high_suppression(pd_tdf, ticks)["false_high_count"],
        "q2_pd_routing_valid": q2_pd["routing_valid"],
        "q2_bdi_routing_valid": q2_bdi["routing_valid"],
        "deployment_verdict": _deployment_verdict(meta_row, artifact, q2_pd),
        "test_auc": ptr.get("test_auc", artifact.get("test_metrics", {}).get("auc")),
        "economic_utility_oos_note": "full-history replay; OOS economic utility requires walk-forward scale application",
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"meta_layer_replay_{ts}.csv"
    md_path = OUT_DIR / f"meta_layer_replay_{ts}.md"
    monitor_path = OUT_DIR / f"meta_monitor_candidate_{ts}.md"
    routing_path = OUT_DIR / f"meta_routing_monotonicity_{ts}.md"

    result_df.to_csv(csv_path, index=False)
    _write_reports(md_path, monitor_path, routing_path, result_df, summary, artifact, ts)
    return result_df, summary, csv_path, md_path


def _write_reports(
    md_path: Path, monitor_path: Path, routing_path: Path,
    df: pd.DataFrame, summary: Dict[str, Any], artifact: Dict[str, Any], ts: str,
) -> None:
    lines = [
        "# Meta Layer V2 Replay Report",
        "",
        f"- deployment_verdict: **{summary['deployment_verdict']}**",
        f"- model: {summary.get('artifact_model')}",
        "",
        "## Economic Comparison",
        "| candidate | trades | net | MDD | RFE | routing_valid | false_high | preservation |",
        "|-----------|--------|-----|-----|-----|---------------|------------|--------------|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['candidate']} | {int(r['trade_count'])} | {r['net_return']:.6f} | {r['max_drawdown']:.6f} | "
            f"{int(r['risk_force_exit_count'])} | {r['routing_valid']} | {int(r.get('false_high_count', 0))} | {r['preservation_ratio']:.2%} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    monitor_lines = [
        "# Meta Monitor Candidate (diagnostic only — NOT live)",
        "",
        "Compare: Production | G30 | Hybrid_A | Q2_PD | Q2_BDI | Meta_V2_candidate",
        "",
        df.to_string(index=False),
        "",
        f"**Verdict:** {summary['deployment_verdict']}",
    ]
    monitor_path.write_text("\n".join(monitor_lines) + "\n", encoding="utf-8")

    routing_lines = [
        "# Routing Monotonicity Analysis",
        "",
        json.dumps({r["candidate"]: {
            "routing_valid": r["routing_valid"],
            "avg_scale_winners": r["avg_scale_winners"],
            "avg_scale_losers": r["avg_scale_losers"],
            "avg_scale_rfe": r["avg_scale_rfe"],
        } for _, r in df.iterrows()}, indent=2),
    ]
    routing_path.write_text("\n".join(routing_lines) + "\n", encoding="utf-8")

    overfit_path = OUT_DIR / f"meta_overfit_leakage_{ts}.md"
    overfit_path.write_text(
        "# Overfit / Leakage Diagnosis\n\n"
        + json.dumps({
            "model_diagnostics": artifact.get("model_diagnostics", {}),
            "test_metrics": artifact.get("test_metrics", {}),
            "deployment_verdict": summary["deployment_verdict"],
        }, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    df, summary, csv_path, md_path = run_replay()
    print(f"Meta_V2_routing_valid: {summary['meta_routing_valid']}")
    print(f"false_high_meta: {summary['false_high_count_meta']}")
    print(f"deployment_verdict: {summary['deployment_verdict']}")
    print(f"csv: {csv_path}")
    print(f"md: {md_path}")


if __name__ == "__main__":
    main()
