"""
Meta_V2 recalibration & preservation recovery (diagnostics only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd

from scripts.diagnostics.build_meta_label_dataset import (
    _entry_features_v2,
    _ohlcv_features,
    _rolling_context,
    load_v2_dataset,
)
from scripts.diagnostics.validate_h8_soft_risk_gate import Variant, _entropy, _simulate_variant
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
RECAL_DIR = OUT_DIR / "recalibration"
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
MIN_SCALE = 0.05
HARD_REJECT_CAL = 0.75
CAL_MULTS = [0.15, 0.30, 0.45, 0.55, 0.70, 0.85]
PRESERVATION_TARGETS = [0.70, 0.80, 0.85, 0.90]


def _load_artifact() -> Dict[str, Any]:
    pointer = OUT_DIR / "meta_layer_model_latest.json"
    path = Path(json.loads(pointer.read_text(encoding="utf-8"))["artifact_path"])
    return joblib.load(path)


def _ctx_ablated(closed: List[Dict[str, Any]], ablate: Set[str]) -> Dict[str, float]:
    ctx = _rolling_context(closed)
    if "danger_cluster_count" in ablate:
        ctx["danger_cluster_count"] = 0.0
    if "rolling_false_positive_density" in ablate:
        ctx["rolling_false_positive_density"] = 0.0
    return ctx


def _predict_scale(artifact: Dict[str, Any], feats: Dict[str, Any], cal_mult: float) -> Tuple[float, float, float]:
    fcols = artifact["feature_cols"]
    row = {c: feats.get(c, 0) for c in fcols}
    x = pd.DataFrame([row], columns=fcols).fillna(0)
    raw_prob = float(artifact["calibration_model"].predict_proba(x)[0, 1])
    iso = artifact.get("isotonic_calibrator")
    cal_prob = float(iso.predict([raw_prob])[0]) if iso is not None else raw_prob
    scale_raw = float(artifact["scale_regressor"].predict(x)[0])
    scale = max(MIN_SCALE, min(1.0, scale_raw * (1.0 - cal_mult * cal_prob)))
    return cal_prob, scale_raw, scale


def simulate_meta_counterfactual(
    ticks: List[Dict[str, Any]],
    feat_map: Dict[int, Dict[str, Any]],
    artifact: Dict[str, Any],
    dataset: pd.DataFrame,
    *,
    cal_mult: float = 0.30,
    hard_reject: bool = False,
    ablate_ctx: Optional[Set[str]] = None,
    min_scale: float = MIN_SCALE,
) -> pd.DataFrame:
    """Apply meta scale to production-equivalent entries only (fair preservation compare)."""
    ablate_ctx = ablate_ctx or set()
    idx_to_tick = {int(t.get("df_idx", i)): (i, t) for i, t in enumerate(ticks)}
    closed: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for _, prod in dataset.sort_values("df_idx").iterrows():
        df_idx = int(prod["df_idx"])
        if df_idx not in idx_to_tick:
            continue
        i, t = idx_to_tick[df_idx]
        direction = str(prod["direction"])
        ctx = _ctx_ablated(closed, ablate_ctx)
        feats = _entry_features_v2(t, ticks, i, feat_map, direction, ctx)
        cal_prob, scale_raw, scale = _predict_scale(artifact, feats, cal_mult)
        scale = max(min_scale, min(1.0, scale))

        if hard_reject and scale <= min_scale and cal_prob >= HARD_REJECT_CAL:
            closed.append({"net_return": float(prod["net_return"]), "false_high_signature_flag": feats.get("false_high_signature_flag", 0), "hybrid_a_danger": feats.get("hybrid_a_danger", 0), "q2_score": feats.get("q2_score", 0)})
            continue

        net = float(prod["net_return"])
        rows.append({
            "entry_idx": i, "df_idx": df_idx, "direction": direction,
            "scale": scale, "cal_prob": cal_prob, "scale_raw": scale_raw,
            "net_return": net, "scaled_return": net * scale,
            "exit_reason": prod.get("exit_reason", ""),
            "false_high_signature_flag": feats.get("false_high_signature_flag", 0),
        })
        closed.append({"net_return": net, "false_high_signature_flag": feats.get("false_high_signature_flag", 0), "hybrid_a_danger": feats.get("hybrid_a_danger", 0), "q2_score": feats.get("q2_score", 0)})

    return pd.DataFrame(rows)


def simulate_meta_recal(
    ticks: List[Dict[str, Any]],
    feat_map: Dict[int, Dict[str, Any]],
    artifact: Dict[str, Any],
    *,
    cal_mult: float = 0.85,
    hard_reject: bool = True,
    ablate_ctx: Optional[Set[str]] = None,
) -> pd.DataFrame:
    ablate_ctx = ablate_ctx or set()
    trades: List[Dict[str, Any]] = []
    closed: List[Dict[str, Any]] = []
    open_pos: Optional[Dict[str, Any]] = None
    eq = peak = 1.0
    daily_pnl = consec_losses = 0
    ks_triggered = False

    for i, t in enumerate(ticks):
        px = float(t["price"])
        sig = t.get("signal")
        vol = str(t.get("vol_bucket") or "")
        trend = str(t.get("trend_label") or "")

        exit_signal, reason = False, ""
        if ks_triggered and open_pos is not None:
            exit_signal, reason = True, "kill_switch_close"
        if open_pos is not None:
            open_pos["hold_bars"] += 1
            if sig and ((open_pos["side"] == "BUY" and sig == "SHORT") or (open_pos["side"] == "SELL" and sig == "LONG")):
                exit_signal, reason = True, reason or "opposite_signal"
            elif open_pos["hold_bars"] >= PAPER_MAX_HOLDING_BARS:
                exit_signal, reason = True, reason or "max_holding_bars"

        if exit_signal and open_pos is not None:
            raw = (px - open_pos["entry_price"]) / open_pos["entry_price"] if open_pos["side"] == "BUY" else (open_pos["entry_price"] - px) / open_pos["entry_price"]
            net = raw - 2.0 * (FEE_RATE + SLIPPAGE_RATE)
            scale = float(open_pos["scale"])
            eq *= 1.0 + net * POSITION_SIZE * scale
            peak = max(peak, eq)
            daily_pnl = eq - 1.0
            consec_losses = consec_losses + 1 if net * scale < 0 else 0
            trades.append({**open_pos, "exit_reason": reason or "exit_signal", "net_return": net, "scaled_return": net * scale})
            closed.append({"net_return": net, "false_high_signature_flag": open_pos.get("false_high_signature_flag", 0), "hybrid_a_danger": open_pos.get("hybrid_a_danger", 0), "q2_score": open_pos.get("q2_score", 0)})
            open_pos = None
            continue

        if open_pos is not None:
            unreal = (px - open_pos["entry_price"]) / open_pos["entry_price"] if open_pos["side"] == "BUY" else (open_pos["entry_price"] - px) / open_pos["entry_price"]
            if unreal <= MAX_TRADE_LOSS:
                net = unreal - 2.0 * (FEE_RATE + SLIPPAGE_RATE)
                scale = float(open_pos["scale"])
                eq *= 1.0 + net * POSITION_SIZE * scale
                trades.append({**open_pos, "exit_reason": "risk_force_exit", "net_return": net, "scaled_return": net * scale})
                closed.append({"net_return": net, "false_high_signature_flag": open_pos.get("false_high_signature_flag", 0)})
                open_pos = None
            continue

        if vol not in ("mid", "high") or ((trend != "sideways" and _entropy(t) > 1.0) or sig is None):
            continue
        dd = (eq - peak) / peak if peak > 0 else 0.0
        if ks_triggered or daily_pnl <= MAX_DAILY_LOSS or dd <= MAX_DRAWDOWN or consec_losses >= MAX_CONSEC_LOSSES:
            ks_triggered = True
            continue

        direction = "LONG" if sig == "LONG" else "SHORT"
        feats = _entry_features_v2(t, ticks, i, feat_map, direction, _ctx_ablated(closed, ablate_ctx))
        cal_prob, scale_raw, scale = _predict_scale(artifact, feats, cal_mult)
        if hard_reject and scale <= MIN_SCALE and cal_prob >= HARD_REJECT_CAL:
            continue

        open_pos = {
            "entry_idx": i, "entry_price": px, "direction": direction,
            "side": "BUY" if sig == "LONG" else "SELL", "hold_bars": 0,
            "scale": scale, "cal_prob": cal_prob, "scale_raw": scale_raw,
            "df_idx": int(t.get("df_idx", i)),
            "false_high_signature_flag": feats.get("false_high_signature_flag", 0),
            "hybrid_a_danger": feats.get("hybrid_a_danger", 0), "q2_score": feats.get("q2_score", 0),
        }
    return pd.DataFrame(trades)


def _mdd(vals: List[float]) -> float:
    eq = peak = 1.0
    mdd = 0.0
    for r in vals:
        eq *= 1.0 + r * POSITION_SIZE
        peak = max(peak, eq)
        mdd = min(mdd, (eq - peak) / peak if peak > 0 else 0)
    return float(mdd)


def _false_high_count(tdf: pd.DataFrame, ticks: List[Dict[str, Any]]) -> int:
    n = 0
    for _, tr in tdf.iterrows():
        tt = ticks[int(tr["entry_idx"])]
        if tr.get("direction") == "LONG" and str(tt.get("trend_label")) == "up" and str(tt.get("vol_bucket")) == "high" and _entropy(tt) <= 0.90 and float(tr.get("scaled_return", 0)) < 0:
            n += 1
    return n


def _tdf_df_idx(tdf: pd.DataFrame, ticks: List[Dict[str, Any]]) -> Set[int]:
    if tdf.empty:
        return set()
    if "df_idx" in tdf.columns:
        return set(tdf["df_idx"].astype(int).tolist())
    if "entry_idx" in tdf.columns:
        return {int(ticks[int(i)].get("df_idx", i)) for i in tdf["entry_idx"].astype(int)}
    return set()


def _good_reject_rate(tdf: pd.DataFrame, dataset: pd.DataFrame, ticks: List[Dict[str, Any]]) -> float:
    taken = _tdf_df_idx(tdf, ticks)
    good = dataset[dataset["trade_quality_label"] == 2]
    return float((~good["df_idx"].astype(int).isin(taken)).sum() / max(len(good), 1))


def _metrics_row(name: str, tdf: pd.DataFrame, ticks: List[Dict[str, Any]], dataset: pd.DataFrame, prod_trades: int, extra: Optional[Dict] = None) -> Dict[str, Any]:
    vals = tdf["scaled_return"].tolist() if not tdf.empty else []
    rr = _risk_routing(tdf, ticks)
    routing_valid = rr["avg_scale_winners"] > rr["avg_scale_losers"] > rr["avg_scale_rfe"] if not tdf.empty else False
    cal_err = 0.0
    if not tdf.empty and "df_idx" in tdf.columns and "cal_prob" in tdf.columns:
        m = tdf.merge(dataset[["df_idx", "binary_bad_trade"]], on="df_idx", how="left")
        cal_err = float(np.mean(np.abs(m["cal_prob"] - m["binary_bad_trade"].fillna(0))))
    bad_rate = 0.0
    if not tdf.empty:
        taken = _tdf_df_idx(tdf, ticks)
        sub = dataset[dataset["df_idx"].astype(int).isin(taken)]
        bad_rate = float(sub["binary_bad_trade"].mean()) if len(sub) else 0.0
    scale_corr = 0.0
    if not tdf.empty and "df_idx" in tdf.columns:
        m = tdf.merge(dataset[["df_idx", "scale_target_label"]], on="df_idx", how="left")
        if m["scale_target_label"].notna().sum() > 2:
            scale_corr = float(np.corrcoef(m["scale"], m["scale_target_label"].fillna(0))[0, 1])
    row = {
        "candidate": name, "trades": len(tdf), "preservation": len(tdf) / max(prod_trades, 1),
        "net_return": float(np.sum(vals)) if vals else 0, "MDD": _mdd(vals), "routing_valid": routing_valid,
        "false_high": _false_high_count(tdf, ticks), "RFE": int((tdf["exit_reason"] == "risk_force_exit").sum()) if not tdf.empty else 0,
        "calibration_error": cal_err, "good_trade_false_rejection_rate": _good_reject_rate(tdf, dataset, ticks),
        "top_bucket_expectancy": float(tdf.nlargest(max(1, len(tdf) // 4), "scale")["scaled_return"].mean()) if len(tdf) > 3 else 0,
        "scale_quality_correlation": scale_corr,
        "bad_trade_rate": bad_rate,
        "avg_scale": float(tdf["scale"].mean()) if not tdf.empty else 0, **rr,
    }
    if extra:
        row.update(extra)
    return row


def _calibration_forensics(dataset: pd.DataFrame, artifact: Dict[str, Any]) -> Dict[str, Any]:
    fcols = artifact["feature_cols"]
    probs = []
    for _, row in dataset.iterrows():
        x = pd.DataFrame([{c: row.get(c, 0) for c in fcols}], columns=fcols).fillna(0)
        raw = float(artifact["calibration_model"].predict_proba(x)[0, 1])
        iso = artifact.get("isotonic_calibrator")
        probs.append(float(iso.predict([raw])[0]) if iso else raw)
    ds = dataset.copy()
    ds["meta_cal_prob"] = probs
    hi = ds[ds["meta_cal_prob"] >= 0.75]
    return {
        "calibration_label_counts": ds["calibration_label"].value_counts().to_dict(),
        "severe_imbalance": int(ds["calibration_label"].sum()) < 20,
        "high_cal_actual_bad_rate": float(hi["binary_bad_trade"].mean()) if len(hi) else 0,
        "cal_prob_std": float(ds["meta_cal_prob"].std()),
        "drawdown_proxy_correlation": float(np.corrcoef(ds["meta_cal_prob"], ds["drawdown_risk_label"])[0, 1]),
        "interpretation": "Model learns drawdown_risk proxy; calibration_label positive n=4 unusable for calibration learning",
    }


def _lh_forensics(tdf: pd.DataFrame, dataset: pd.DataFrame, ticks: List[Dict[str, Any]]) -> Dict[str, Any]:
    lh = dataset[(dataset["direction"] == "LONG") & (dataset["vol_bucket"] == "high")].copy()
    taken = _tdf_df_idx(tdf, ticks)
    lh["survived"] = lh["df_idx"].astype(int).isin(taken)
    surv, rej = lh[lh["survived"]], lh[~lh["survived"]]
    return {
        "long_highvol_total": len(lh), "survived": len(surv), "rejected": len(rej),
        "survived_expectancy": float(surv["net_return"].mean()) if len(surv) else 0,
        "rejected_expectancy": float(rej["net_return"].mean()) if len(rej) else 0,
        "survived_mae": float(surv["mae"].mean()) if len(surv) else 0, "rejected_mae": float(rej["mae"].mean()) if len(rej) else 0,
        "rejected_bad_rate": float(rej["binary_bad_trade"].mean()) if len(rej) else 0,
        "survived_bad_rate": float(surv["binary_bad_trade"].mean()) if len(surv) else 0,
        "verdict": "blanket LONG/high_vol rejection" if len(rej) > len(surv) * 2 and float(rej["binary_bad_trade"].mean()) < 0.52 else "partial selective removal",
    }


def _verdict(meta: Dict, q2: Dict, cal: Dict) -> str:
    if meta["preservation"] < 0.65:
        return "needs_recalibration"
    if meta["good_trade_false_rejection_rate"] > 0.45:
        return "feature_rebalance_needed"
    beats = meta["MDD"] >= q2["MDD"] and meta["routing_valid"] and meta["false_high"] <= q2["false_high"] + 1 and meta["preservation"] >= 0.70
    if beats and not cal.get("severe_imbalance"):
        return "monitor_only_candidate"
    if beats:
        return "Meta_V2 viable after recalibration"
    if meta["preservation"] >= 0.70:
        return "Q2_PD/Q2_BDI superior"
    return "needs_recalibration"


def run_recalibration() -> Dict[str, Any]:
    RECAL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = _load_artifact()
    dataset, _ = load_v2_dataset()
    ohlcv = load_ohlcv()
    feat_map, _ = _ohlcv_features(ohlcv)
    ticks, _ = simulate_signals_paper(ohlcv)
    prod_trades = len(dataset)
    cache: Dict[str, pd.DataFrame] = {}

    def sim(key: str, **kw) -> pd.DataFrame:
        if key not in cache:
            cache[key] = simulate_meta_counterfactual(ticks, feat_map, artifact, dataset, **kw)
        return cache[key]

    phase1 = pd.DataFrame([
        _metrics_row("hard_reject_ON", sim("p1_on", cal_mult=0.85, hard_reject=True), ticks, dataset, prod_trades, {"hard_reject": True}),
        _metrics_row("hard_reject_OFF", sim("p1_off", cal_mult=0.85, hard_reject=False), ticks, dataset, prod_trades, {"hard_reject": False}),
    ])
    phase2 = pd.DataFrame([_metrics_row(f"cal_mult_{cm}", sim(f"p2_{cm}", cal_mult=cm, hard_reject=False), ticks, dataset, prod_trades, {"cal_mult": cm}) for cm in CAL_MULTS])
    ablations = {"full": set(), "no_danger_cluster": {"danger_cluster_count"}, "no_rolling_fp": {"rolling_false_positive_density"}, "no_both": {"danger_cluster_count", "rolling_false_positive_density"}}
    phase3 = pd.DataFrame([_metrics_row(n, sim(f"p3_{n}", cal_mult=0.30, hard_reject=False, ablate_ctx=a), ticks, dataset, prod_trades, {"ablation": n}) for n, a in ablations.items()])

    phase5_rows = []
    for target in PRESERVATION_TARGETS:
        sub = phase2.copy()
        sub["diff"] = (sub["preservation"] - target).abs()
        best = sub.loc[sub["diff"].idxmin()].to_dict()
        best["candidate"] = f"Meta_pres_{int(target*100)}"
        best["preservation_target"] = target
        phase5_rows.append(best)
    phase5 = pd.DataFrame(phase5_rows)

    recal_tdf = sim("p3_no_both", cal_mult=0.30, hard_reject=False, ablate_ctx=ablations["no_both"])
    meta_row = _metrics_row("Meta_V2_Recalibrated", recal_tdf, ticks, dataset, prod_trades, {"cal_mult": 0.30, "hard_reject": False, "ablation": "no_both"})

    _, prod_tdf = _simulate_variant(ticks, BASELINE, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    _, g30_tdf = _simulate_variant(ticks, G30, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    hyb_m, hyb_tdf, _ = _simulate_hybrid(ticks, HYBRID_A, feat_map, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    _, pd_tdf, _ = _simulate_quality(ticks, SCORE_Q2_PD, map_m3, feat_map)
    _, bdi_tdf, _ = _simulate_quality(ticks, SCORE_Q2_BDI, map_m3, feat_map)

    compare = pd.DataFrame([
        _metrics_row("Production", prod_tdf, ticks, dataset, prod_trades),
        _metrics_row("G30", g30_tdf, ticks, dataset, prod_trades),
        _metrics_row("Hybrid_A", hyb_tdf, ticks, dataset, prod_trades),
        _metrics_row("Q2_PD", pd_tdf, ticks, dataset, prod_trades),
        _metrics_row("Q2_BDI", bdi_tdf, ticks, dataset, prod_trades),
        meta_row,
    ])
    cal_f = _calibration_forensics(dataset, artifact)
    lh_f = _lh_forensics(recal_tdf, dataset, ticks)
    q2 = compare[compare["candidate"] == "Q2_PD"].iloc[0].to_dict()
    verdict = _verdict(meta_row, q2, cal_f)

    phase2.to_csv(RECAL_DIR / "calibration_multiplier_tournament.csv", index=False)
    phase5.to_csv(RECAL_DIR / "preservation_recovery_curve.csv", index=False)
    compare.to_csv(RECAL_DIR / "meta_v2_recalibrated_comparison.csv", index=False)
    recal_tdf.to_csv(RECAL_DIR / "meta_v2_recalibrated_replay.csv", index=False)

    (RECAL_DIR / "meta_v2_recalibration_report.md").write_text(
        f"# Meta V2 Recalibration Report\n\n**Verdict:** {verdict}\n\n## Phase1\n{phase1.to_string(index=False)}\n\n## Phase2\n{phase2.to_string(index=False)}\n\n## Phase3\n{phase3.to_string(index=False)}\n\n## Phase5\n{phase5.to_string(index=False)}\n\n## Compare\n{compare.to_string(index=False)}\n",
        encoding="utf-8",
    )
    (RECAL_DIR / "path_feedback_ablation_report.md").write_text(f"# Ablation\n\n{phase3.to_string(index=False)}\n", encoding="utf-8")
    (RECAL_DIR / "long_highvol_rejection_forensics.md").write_text(f"# LONG/HV\n\n{json.dumps(lh_f, indent=2)}\n", encoding="utf-8")
    (RECAL_DIR / "calibration_label_forensics.md").write_text(f"# Cal Label\n\n{json.dumps(cal_f, indent=2)}\n", encoding="utf-8")
    (RECAL_DIR / "recommended_meta_v2_action.md").write_text(
        f"# Action\n\nVerdict: **{verdict}**\n\nRecal: cal_mult=0.30, hard_reject=OFF, ablation=no_both\n"
        f"preservation={meta_row['preservation']:.1%} MDD={meta_row['MDD']:.6f} routing={meta_row['routing_valid']}\n"
        f"good_reject={meta_row['good_trade_false_rejection_rate']:.1%}\n\nQ2_PD: MDD={q2['MDD']:.6f} false_high={q2['false_high']}\n",
        encoding="utf-8",
    )
    return {"verdict": verdict, "meta_recalibrated": meta_row, "compare": compare.to_dict(orient="records"), "paths": str(RECAL_DIR)}


def main() -> None:
    r = run_recalibration()
    m = r["meta_recalibrated"]
    print(f"verdict: {r['verdict']}")
    print(f"Meta_V2_Recalibrated: trades={m['trades']} preservation={m['preservation']:.1%} MDD={m['MDD']:.6f} routing={m['routing_valid']} false_high={m['false_high']} good_reject={m['good_trade_false_rejection_rate']:.1%}")
    print(f"output: {r['paths']}")


if __name__ == "__main__":
    main()
