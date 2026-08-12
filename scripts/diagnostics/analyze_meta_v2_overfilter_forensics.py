"""
Meta_V2 overfilter forensics — why preservation collapsed to 17% (diagnostics only).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from scripts.diagnostics.build_meta_label_dataset import (
    _entry_features_v2,
    _ohlcv_features,
    _path_metrics,
    _rolling_context,
    load_v2_dataset,
)
from scripts.diagnostics.validate_h8_soft_risk_gate import (
    Variant,
    _entropy,
    _expectancy,
    _h8_pass,
    _is_loss_cluster_trade,
)
from scripts.diagnostics.validate_quality_score_replay import FEE_RATE, POSITION_SIZE, SLIPPAGE_RATE, _risk_routing
from scripts.run_daily_paper import PAPER_MAX_HOLDING_BARS, load_ohlcv, simulate_signals_paper

OUT_DIR = Path("data/diagnostics/meta_layer")
FORENSIC_DIR = OUT_DIR / "overfilter_forensics"
BASELINE = Variant("Baseline", fail_scale=1.0, hard_block=False)

MAX_TRADE_LOSS = -0.01
MAX_DAILY_LOSS = -0.02
MAX_DRAWDOWN = -0.05
MAX_CONSEC_LOSSES = 5

CAL_MULT = 0.85
HARD_REJECT_CAL = 0.75
MIN_SCALE = 0.05
PRESERVATION_TARGETS = [0.20, 0.35, 0.50, 0.70, 0.85, 0.90]

COMPARE_COLS = [
    "direction", "trend_state", "vol_bucket", "entropy", "margin", "p_long", "p_short",
    "danger_cluster_count", "false_high_signature_flag", "q2_pd_penalty_score",
    "q2_bdi_penalty_score", "hybrid_a_danger_score", "recent_return_24", "ema_distance",
    "mae", "mfe", "trade_quality_label", "net_return", "rfe_flag",
]


def _load_artifact() -> Dict[str, Any]:
    pointer = OUT_DIR / "meta_layer_model_latest.json"
    path = Path(json.loads(pointer.read_text(encoding="utf-8"))["artifact_path"])
    return joblib.load(path)


def _predict_components(artifact: Dict[str, Any], feats: Dict[str, Any]) -> Dict[str, float]:
    fcols = artifact["feature_cols"]
    row = {c: feats.get(c, 0) for c in fcols}
    x = pd.DataFrame([row], columns=fcols).fillna(0)
    for col in x.columns:
        if x[col].dtype == bool:
            x[col] = x[col].astype(int)

    raw_prob = float(artifact["calibration_model"].predict_proba(x)[0, 1])
    iso = artifact["isotonic_calibrator"]
    cal_prob = float(iso.predict([raw_prob])[0]) if iso is not None else raw_prob
    scale_raw = float(artifact["scale_regressor"].predict(x)[0])
    cal_penalty = 1.0 - CAL_MULT * cal_prob
    scale_final = max(MIN_SCALE, min(1.0, scale_raw * cal_penalty))

    danger = (
        float(feats.get("false_high_signature_flag", 0))
        + float(feats.get("hybrid_a_danger_score", 0))
        + float(feats.get("q2_pd_penalty_score", 0)) * 2
        + float(feats.get("rolling_false_positive_density", 0))
    )

    return {
        "raw_risk_prob": raw_prob,
        "calibrated_risk_prob": cal_prob,
        "original_scale_prediction": scale_raw,
        "calibration_penalty_factor": cal_penalty,
        "final_scale_mapping": scale_final,
        "danger_score": danger,
        "quality_score": float(feats.get("q2_score", 0)),
    }


def _routing_bucket(scale: float) -> str:
    if scale >= 0.80:
        return "high"
    if scale >= 0.50:
        return "mid"
    if scale >= 0.20:
        return "low"
    return "floor"


def _rejection_reason(comp: Dict[str, float]) -> Tuple[str, str]:
    scale = comp["final_scale_mapping"]
    cal = comp["calibrated_risk_prob"]
    if scale <= MIN_SCALE and cal >= HARD_REJECT_CAL:
        return "hard_reject", f"scale<={MIN_SCALE}_AND_cal>={HARD_REJECT_CAL}"
    if scale <= MIN_SCALE:
        return "scale_floor", f"final_scale={scale:.4f}"
    if cal >= 0.90:
        return "calibration_collapse", f"cal_prob={cal:.4f}"
    if comp["calibration_penalty_factor"] < 0.25:
        return "cal_penalty_aggressive", f"penalty={comp['calibration_penalty_factor']:.4f}"
    return "accepted", ""


def _feature_contributors(artifact: Dict[str, Any], feats: Dict[str, Any]) -> str:
    fcols = artifact["feature_cols"]
    model = artifact["calibration_model"]
    vals = {c: float(feats.get(c, 0) or 0) for c in fcols}
    imp: Dict[str, float] = {}
    if hasattr(model, "feature_importances_"):
        for c, v in zip(fcols, model.feature_importances_):
            imp[c] = abs(float(v) * vals[c])
    elif hasattr(model, "named_steps"):
        coef = model.named_steps["clf"].coef_[0]
        for c, w in zip(fcols, coef):
            imp[c] = abs(float(w) * vals[c])
    top = sorted(imp.items(), key=lambda x: -x[1])[:5]
    return "|".join(f"{k}:{v:.4f}" for k, v in top)


def _trace_production_entries(
    ticks: List[Dict[str, Any]],
    feat_map: Dict[int, Dict[str, Any]],
    artifact: Dict[str, Any],
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Trace Meta_V2 decision at each production trade entry (sequential rolling context)."""
    idx_to_row = {int(r["df_idx"]): r for _, r in dataset.iterrows()}
    closed: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []

    for i, t in enumerate(ticks):
        sig = t.get("signal")
        vol = str(t.get("vol_bucket") or "")
        trend = str(t.get("trend_label") or "")
        if vol not in ("mid", "high"):
            continue
        if (trend != "sideways" and _entropy(t) > 1.0) or sig is None:
            continue

        df_idx = int(t.get("df_idx", i))
        if df_idx not in idx_to_row:
            continue

        direction = "LONG" if sig == "LONG" else "SHORT"
        ctx = _rolling_context(closed)
        feats = _entry_features_v2(t, ticks, i, feat_map, direction, ctx)
        comp = _predict_components(artifact, feats)
        reason, trigger = _rejection_reason(comp)
        hard_reject = reason == "hard_reject"
        accepted = not hard_reject and comp["final_scale_mapping"] > MIN_SCALE

        prod = idx_to_row[df_idx]
        row = {
            "entry_idx": i,
            "df_idx": df_idx,
            "trade_id": prod.get("trade_id", ""),
            "direction": direction,
            "timestamp": t.get("timestamp"),
            **comp,
            "rejection_reason": reason,
            "threshold_trigger": trigger,
            "hard_reject": hard_reject,
            "meta_accepted_counterfactual": accepted,
            "routing_bucket": _routing_bucket(comp["final_scale_mapping"]),
            "feature_contributors": _feature_contributors(artifact, feats),
            **{k: feats.get(k) for k in (
                "entropy", "margin", "p_long", "p_short", "trend_state", "vol_bucket",
                "danger_cluster_count", "false_high_signature_flag", "q2_pd_penalty_score",
                "q2_bdi_penalty_score", "hybrid_a_danger_score", "recent_return_24",
                "ema_distance", "confidence_overextension", "entropy_delta",
                "rolling_false_positive_density",
            )},
            "net_return": float(prod["net_return"]),
            "mae": float(prod["mae"]),
            "mfe": float(prod["mfe"]),
            "rfe_flag": bool(prod["rfe_flag"]),
            "trade_quality_label": int(prod["trade_quality_label"]),
            "binary_bad_trade": int(prod["binary_bad_trade"]),
            "binary_good_trade": int(prod["binary_good_trade"]),
            "scale_target_label": float(prod.get("scale_target_label", 0)),
        }
        traces.append(row)

        if accepted:
            closed.append({
                **feats,
                "net_return": float(prod["net_return"]),
                "false_high_signature_flag": feats.get("false_high_signature_flag", 0),
                "hybrid_a_danger": feats.get("hybrid_a_danger", 0),
                "q2_score": feats.get("q2_score", 0),
            })

    return pd.DataFrame(traces)


def _simulate_meta_path(
    ticks: List[Dict[str, Any]],
    feat_map: Dict[int, Dict[str, Any]],
    artifact: Dict[str, Any],
    cal_reject: float = HARD_REJECT_CAL,
    cal_mult: float = CAL_MULT,
    min_scale: float = MIN_SCALE,
    min_score_percentile: Optional[float] = None,
    score_lookup: Optional[Dict[int, float]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Path-dependent meta simulation with optional score percentile gate."""
    trades: List[Dict[str, Any]] = []
    closed: List[Dict[str, Any]] = []
    open_pos: Optional[Dict[str, Any]] = None
    eq = peak = 1.0
    daily_pnl = 0.0
    consec_losses = 0
    ks_triggered = False

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
            trades.append({**open_pos, "exit_reason": reason, "net_return": net, "scaled_return": net * scale})
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
        ctx = _rolling_context(closed)
        feats = _entry_features_v2(t, ticks, i, feat_map, direction, ctx)
        comp = _predict_components(artifact, feats)
        scale_raw = comp["original_scale_prediction"]
        cal_prob = comp["calibrated_risk_prob"]
        scale = max(min_scale, min(1.0, scale_raw * (1.0 - cal_mult * cal_prob)))

        df_idx = int(t.get("df_idx", i))
        if min_score_percentile is not None and score_lookup is not None:
            thr = score_lookup.get("_threshold", 0)
            if score_lookup.get(df_idx, 0) < thr:
                continue

        if scale <= min_scale and cal_prob >= cal_reject:
            continue

        side = "BUY" if sig == "LONG" else "SELL"
        open_pos = {
            "entry_idx": i, "entry_price": px, "direction": direction, "side": side,
            "hold_bars": 0, "scale": scale, "cal_prob": cal_prob,
            "false_high_signature_flag": feats.get("false_high_signature_flag", 0),
            "df_idx": df_idx,
        }

    return pd.DataFrame(trades), closed


def _compare_groups(survived: pd.DataFrame, rejected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in COMPARE_COLS:
        if col not in survived.columns:
            continue
        s = survived[col]
        r = rejected[col]
        if not pd.api.types.is_numeric_dtype(s):
            rows.append({
                "metric": col,
                "survived_mean": np.nan,
                "rejected_mean": np.nan,
                "survived_top": str(s.value_counts().index[0]) if len(s) else "",
                "rejected_top": str(r.value_counts().index[0]) if len(r) else "",
                "delta": np.nan,
            })
            continue
        if s.dtype in (bool, np.bool_) or r.dtype in (bool, np.bool_):
            s, r = s.astype(float), r.astype(float)
        rows.append({
            "metric": col,
            "survived_mean": float(s.mean()) if len(s) else 0,
            "rejected_mean": float(r.mean()) if len(r) else 0,
            "survived_pct_true": float(s.mean()) if len(s) and s.max() <= 1 and s.min() >= 0 else np.nan,
            "rejected_pct_true": float(r.mean()) if len(r) and r.max() <= 1 and r.min() >= 0 else np.nan,
            "delta": float(s.mean() - r.mean()) if len(s) and len(r) else 0,
        })
    return pd.DataFrame(rows)


def _good_trade_loss(rejected: pd.DataFrame, all_trades: pd.DataFrame) -> Dict[str, Any]:
    good = rejected[rejected["trade_quality_label"] == 2]
    profitable = rejected[rejected["net_return"] > 0]
    total_good = int((all_trades["trade_quality_label"] == 2).sum())
    return {
        "rejected_total": int(len(rejected)),
        "rejected_good_count": int(len(good)),
        "rejected_good_rate": float(len(good) / max(len(rejected), 1)),
        "rejected_good_expectancy": float(good["net_return"].mean()) if len(good) else 0.0,
        "rejected_profitable_count": int(len(profitable)),
        "good_trade_false_rejection_rate": float(len(good) / max(total_good, 1)),
        "rejected_good_long_pct": float((good["direction"] == "LONG").mean()) if len(good) else 0,
        "rejected_good_trend_up_pct": float((good["trend_state"] == "up").mean()) if len(good) else 0,
        "rejected_good_high_vol_pct": float((good["vol_bucket"] == "high").mean()) if len(good) else 0,
    }


def _false_high_analysis(survived: pd.DataFrame) -> Dict[str, Any]:
    if survived.empty:
        return {}
    fh = survived[
        (survived["direction"] == "LONG")
        & (survived["trend_state"] == "up")
        & (survived["vol_bucket"] == "high")
        & (survived["entropy"] <= 0.90)
    ]
    fh_losers = fh[fh["net_return"] < 0]
    return {
        "survived_false_high_cluster": int(len(fh)),
        "survived_false_high_losers": int(len(fh_losers)),
        "avg_confidence_overextension_fh": float(fh["confidence_overextension"].mean()) if len(fh) else 0,
        "avg_rolling_fp_density_fh": float(fh["rolling_false_positive_density"].mean()) if len(fh) else 0,
        "avg_entropy_delta_fh": float(fh["entropy_delta"].mean()) if len(fh) else 0,
    }


def _feature_forensics(trace: pd.DataFrame, artifact: Dict[str, Any]) -> Dict[str, Any]:
    surv = trace[trace["meta_accepted_counterfactual"]]
    rej = trace[~trace["meta_accepted_counterfactual"]]
    feat_cols = artifact["feature_cols"]
    model = artifact["calibration_model"]

    def _mean_feats(df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {}
        return {c: float(df[c].mean()) for c in feat_cols if c in df.columns}

    imp = {}
    if hasattr(model, "feature_importances_"):
        for c, v in zip(feat_cols, model.feature_importances_):
            imp[c] = float(v)

    top_imp = sorted(imp.items(), key=lambda x: -x[1])[:10]
    reject_feats = _mean_feats(rej)
    survive_feats = _mean_feats(surv)
    reject_rank = sorted(reject_feats.items(), key=lambda x: -x[1])[:10]
    survive_rank = sorted(survive_feats.items(), key=lambda x: -x[1])[:10]

    ent_share = imp.get("entropy", 0) / max(sum(imp.values()), 1e-9)
    return {
        "top_model_importance": top_imp,
        "top_reject_feature_means": reject_rank,
        "top_survive_feature_means": survive_rank,
        "entropy_overdominant": ent_share >= 0.20,
        "entropy_importance_share": ent_share,
        "danger_cluster_reject_mean": float(rej["danger_cluster_count"].mean()) if len(rej) else 0,
        "danger_cluster_survive_mean": float(surv["danger_cluster_count"].mean()) if len(surv) else 0,
        "false_high_reject_rate": float(rej["false_high_signature_flag"].mean()) if len(rej) else 0,
        "false_high_survive_rate": float(surv["false_high_signature_flag"].mean()) if len(surv) else 0,
        "rolling_fp_reject_mean": float(rej["rolling_false_positive_density"].mean()) if len(rej) else 0,
        "rolling_fp_survive_mean": float(surv["rolling_false_positive_density"].mean()) if len(surv) else 0,
    }


def _tournament_counterfactual(trace: pd.DataFrame, prod_trades: int) -> pd.DataFrame:
    """Counterfactual threshold relaxation using production outcomes (fast)."""
    trace = trace.copy()
    trace["meta_rank_score"] = trace["final_scale_mapping"] - trace["calibrated_risk_prob"] * 0.5
    trace = trace.sort_values("meta_rank_score", ascending=False)
    rows = []

    for target in PRESERVATION_TARGETS:
        n = max(1, int(round(len(trace) * target)))
        sub = trace.head(n)
        vals = (sub["net_return"] * sub["final_scale_mapping"]).tolist()
        scales = sub["final_scale_mapping"].tolist()
        wins = sub[sub["net_return"] > 0]
        losses = sub[sub["net_return"] < 0]
        rfe = sub[sub["rfe_flag"]]
        win_scale = float(wins["final_scale_mapping"].mean()) if len(wins) else 0
        loss_scale = float(losses["final_scale_mapping"].mean()) if len(losses) else 0
        rfe_scale = float(rfe["final_scale_mapping"].mean()) if len(rfe) else 0
        routing_valid = win_scale > loss_scale > rfe_scale if len(wins) and len(losses) else False

        fh = sub[
            (sub["direction"] == "LONG") & (sub["trend_state"] == "up")
            & (sub["vol_bucket"] == "high") & (sub["entropy"] <= 0.90) & (sub["net_return"] < 0)
        ]
        scale_corr = float(np.corrcoef(scales, sub["scale_target_label"])[0, 1]) if len(scales) > 2 else 0.0
        cal_err = float(np.mean(np.abs(sub["calibrated_risk_prob"] - sub["binary_bad_trade"])))

        rows.append({
            "preservation_target": target,
            "cal_reject_threshold": "rank_based",
            "cal_mult": "rank_based",
            "score_rank_cutoff": float(sub["meta_rank_score"].min()) if len(sub) else 0,
            "trades": n,
            "preservation": n / max(prod_trades, 1),
            "net_return": float(np.sum(vals)),
            "MDD": _mdd_from_returns(vals),
            "routing_valid": routing_valid,
            "false_high": int(len(fh)),
            "RFE": int(sub["rfe_flag"].sum()),
            "calibration_error": cal_err,
            "top_bucket_expectancy": float(sub.nlargest(max(1, n // 4), "meta_rank_score")["net_return"].mean()),
            "scale_quality_correlation": scale_corr,
            "bad_trade_rate": float(sub["binary_bad_trade"].mean()),
            "good_trade_rate": float(sub["binary_good_trade"].mean()),
        })
    return pd.DataFrame(rows)


def _mdd_from_returns(vals: List[float]) -> float:
    eq = peak = 1.0
    mdd = 0.0
    for r in vals:
        eq *= 1.0 + r * POSITION_SIZE
        peak = max(peak, eq)
        mdd = min(mdd, (eq - peak) / peak if peak > 0 else 0)
    return float(mdd)


def _root_cause(trace: pd.DataFrame, feat_f: Dict[str, Any], good_loss: Dict[str, Any]) -> Tuple[List[str], str]:
    causes = []
    rej = trace[~trace["meta_accepted_counterfactual"]]
    hard_pct = float(rej["hard_reject"].mean()) if len(rej) else 0
    scale_floor_pct = float((rej["rejection_reason"] == "scale_floor").mean()) if len(rej) else 0
    cal_high = float((rej["calibrated_risk_prob"] >= 0.75).mean()) if len(rej) else 0

    if hard_pct > 0.5:
        causes.append("A_threshold_aggressive")
        causes.append("I_scale_mapping_collapse")
    if cal_high > 0.6:
        causes.append("B_calibration_collapse")
    if good_loss.get("rejected_good_rate", 0) > 0.35:
        causes.append("good_trade_mass_rejection")
    if float(rej["direction"].eq("LONG").mean()) > 0.85 if len(rej) else False:
        causes.append("F_LONG_bias_rejection")
    if feat_f.get("entropy_overdominant"):
        causes.append("G_entropy_overreaction")
    if float(rej["false_high_signature_flag"].mean()) > 0.3 if len(rej) else False:
        causes.append("H_false_high_suppression_overapplied")
    if float(rej["vol_bucket"].eq("high").mean()) > 0.7 if len(rej) else False:
        causes.append("E_high_vol_overpenalty")
    causes.append("D_positive_sample_deficit")
    causes.append("C_class_imbalance")

    if hard_pct > 0.4 and good_loss.get("rejected_good_rate", 0) > 0.3:
        verdict = "threshold_overfilter"
    elif feat_f.get("entropy_overdominant") or feat_f.get("rolling_fp_reject_mean", 0) > 0.15:
        verdict = "feature_rebalance_needed"
    elif cal_high > 0.5:
        verdict = "needs_recalibration"
    elif good_loss.get("rejected_good_rate", 0) > 0.4:
        verdict = "label_restructure_needed"
    else:
        verdict = "reject"

    return causes, verdict


def run_forensics() -> Dict[str, Any]:
    FORENSIC_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    artifact = _load_artifact()
    dataset, _ = load_v2_dataset()
    ohlcv = load_ohlcv()
    feat_map, _ = _ohlcv_features(ohlcv)
    ticks, _ = simulate_signals_paper(ohlcv)

    trace = _trace_production_entries(ticks, feat_map, artifact, dataset)
    path_tdf, _ = _simulate_meta_path(ticks, feat_map, artifact)
    path_survived_idx = set(path_tdf["entry_idx"].astype(int).tolist()) if not path_tdf.empty else set()
    trace["meta_path_survived"] = trace["entry_idx"].isin(path_survived_idx)

    survived = trace[trace["meta_path_survived"]]
    rejected = trace[~trace["meta_path_survived"]]

    compare_df = _compare_groups(survived, rejected)
    good_loss = _good_trade_loss(rejected, trace)
    fh_analysis = _false_high_analysis(survived)
    feat_f = _feature_forensics(trace, artifact)
    tournament = _tournament_counterfactual(trace, prod_trades=len(dataset))
    causes, verdict = _root_cause(trace, feat_f, good_loss)

    rej_reasons = trace[~trace["meta_accepted_counterfactual"]]["rejection_reason"].value_counts().to_dict()
    pres = len(survived) / max(len(trace), 1)

    summary = {
        "production_trades": len(trace),
        "meta_path_survived": len(survived),
        "preservation": pres,
        "rejection_reasons": rej_reasons,
        "good_trade_loss": good_loss,
        "false_high_analysis": fh_analysis,
        "root_causes": causes,
        "verdict": verdict,
        "answer": (
            "Meta_V2 rejected most of the market — not selective danger removal. "
            if good_loss.get("rejected_good_rate", 0) > 0.25 and pres < 0.35
            else "Meta_V2 removed some danger clusters but with heavy collateral good-trade loss. "
        ),
    }

    # Deliverables
    compare_df.to_csv(FORENSIC_DIR / "survived_vs_rejected_analysis.csv", index=False)
    trace.to_csv(FORENSIC_DIR / f"meta_v2_filter_trace_{ts}.csv", index=False)
    tournament.to_csv(FORENSIC_DIR / "threshold_relaxation_tournament.csv", index=False)
    tournament.to_csv(FORENSIC_DIR / "meta_v2_preservation_curve.csv", index=False)

    _write_good_trade_report(FORENSIC_DIR / "good_trade_rejection_report.md", good_loss, rejected)
    _write_feature_report(FORENSIC_DIR / "feature_contribution_forensics.md", feat_f)
    _write_failure_signature(FORENSIC_DIR / "meta_v2_failure_signature_analysis.md", trace, rej_reasons, fh_analysis)
    _write_root_cause(FORENSIC_DIR / "meta_v2_overfilter_rootcause.md", summary, causes, verdict, rej_reasons, compare_df)
    _write_recommended(FORENSIC_DIR / "recommended_action.md", verdict, summary, tournament)

    summary["paths"] = {k: str(FORENSIC_DIR / v) for k, v in {
        "rootcause": "meta_v2_overfilter_rootcause.md",
        "survived_vs_rejected": "survived_vs_rejected_analysis.csv",
        "good_trade": "good_trade_rejection_report.md",
        "tournament": "threshold_relaxation_tournament.csv",
        "feature": "feature_contribution_forensics.md",
        "preservation_curve": "meta_v2_preservation_curve.csv",
        "failure_signature": "meta_v2_failure_signature_analysis.md",
        "recommended": "recommended_action.md",
    }.items()}
    return summary


def _write_good_trade_report(path: Path, good_loss: Dict[str, Any], rejected: pd.DataFrame) -> None:
    good = rejected[rejected["trade_quality_label"] == 2]
    lines = [
        "# Good Trade Rejection Report",
        "",
        f"- rejected total: {good_loss['rejected_total']}",
        f"- rejected good (quality=2): {good_loss['rejected_good_count']} ({good_loss['rejected_good_rate']:.1%})",
        f"- rejected profitable: {good_loss['rejected_profitable_count']}",
        f"- good trade false rejection rate: {good_loss['good_trade_false_rejection_rate']:.1%}",
        f"- rejected good expectancy: {good_loss['rejected_good_expectancy']:.6f}",
        "",
        "## Rejected Good Trade Regime",
        f"- LONG pct: {good_loss['rejected_good_long_pct']:.1%}",
        f"- trend_up pct: {good_loss['rejected_good_trend_up_pct']:.1%}",
        f"- high_vol pct: {good_loss['rejected_good_high_vol_pct']:.1%}",
        "",
        "## Common Signature (rejected good)",
    ]
    if len(good):
        lines.append(f"- mean entropy: {good['entropy'].mean():.4f}")
        lines.append(f"- mean q2_pd_penalty: {good['q2_pd_penalty_score'].mean():.4f}")
        lines.append(f"- false_high flag rate: {good['false_high_signature_flag'].mean():.1%}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_feature_report(path: Path, feat_f: Dict[str, Any]) -> None:
    lines = ["# Feature Contribution Forensics", ""]
    lines.append("## Top Model Importance")
    for k, v in feat_f.get("top_model_importance", []):
        lines.append(f"- {k}: {v:.6f}")
    lines.append(f"\n- entropy_overdominant: {feat_f.get('entropy_overdominant')}")
    lines.append(f"- entropy_importance_share: {feat_f.get('entropy_importance_share', 0):.3f}")
    lines.append(f"\n## Reject vs Survive")
    lines.append(f"- danger_cluster reject/survive: {feat_f.get('danger_cluster_reject_mean', 0):.3f} / {feat_f.get('danger_cluster_survive_mean', 0):.3f}")
    lines.append(f"- false_high reject/survive rate: {feat_f.get('false_high_reject_rate', 0):.3f} / {feat_f.get('false_high_survive_rate', 0):.3f}")
    lines.append(f"- rolling_fp_density reject/survive: {feat_f.get('rolling_fp_reject_mean', 0):.3f} / {feat_f.get('rolling_fp_survive_mean', 0):.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_failure_signature(path: Path, trace: pd.DataFrame, rej_reasons: Dict, fh: Dict) -> None:
    lines = [
        "# Meta V2 Failure Signature Analysis",
        "",
        "## Rejection Reason Breakdown",
        json.dumps(rej_reasons, indent=2),
        "",
        "## False High Cluster (survived)",
        json.dumps(fh, indent=2),
        "",
        "## Hard Reject Trigger Stats",
        f"- mean cal_prob rejected: {trace[~trace['meta_accepted_counterfactual']]['calibrated_risk_prob'].mean():.4f}",
        f"- mean final_scale rejected: {trace[~trace['meta_accepted_counterfactual']]['final_scale_mapping'].mean():.4f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_root_cause(path: Path, summary: Dict, causes: List[str], verdict: str, rej_reasons: Dict, compare: pd.DataFrame) -> None:
    lines = [
        "# Meta V2 Overfilter Root Cause",
        "",
        f"**Verdict:** {verdict}",
        "",
        f"- preservation: {summary['preservation']:.1%} ({summary['meta_path_survived']}/{summary['production_trades']})",
        f"- conclusion: {summary['answer']}",
        "",
        "## Root Cause Candidates",
    ]
    for c in causes:
        lines.append(f"- {c}")
    lines += ["", "## Rejection Reasons", json.dumps(rej_reasons, indent=2), "", "## Key Metric Deltas (survived - rejected)"]
    for _, r in compare.head(12).iterrows():
        lines.append(f"- {r['metric']}: delta={r['delta']:.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_recommended(path: Path, verdict: str, summary: Dict, tournament: pd.DataFrame) -> None:
    lines = [
        "# Recommended Action",
        "",
        f"**Verdict:** {verdict}",
        "",
        "## Immediate Actions",
        "1. Disable hard_reject gate (scale<=0.05 AND cal>=0.75) — primary overfilter driver",
        "2. Reduce cal_mult from 0.85 to 0.30–0.55 for preservation 70%+",
        "3. Do NOT trust net-positive replay at 17% preservation",
        "4. Recalibrate isotonic layer — drawdown_risk proxy miscalibrated",
        "5. Accumulate dataset to 300+ before any monitor_only promotion",
        "",
        "## Threshold Tournament Summary",
    ]
    if not tournament.empty:
        lines.append(tournament.to_string(index=False))
    else:
        lines.append("(no tournament rows)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary = run_forensics()
    print(f"preservation: {summary['preservation']:.1%}")
    print(f"verdict: {summary['verdict']}")
    print(f"rejection_reasons: {summary['rejection_reasons']}")
    print(f"rejected_good_rate: {summary['good_trade_loss']['rejected_good_rate']:.1%}")
    print(f"answer: {summary['answer']}")
    for k, v in summary["paths"].items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
