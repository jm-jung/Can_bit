"""
Forward Meta shadow monitor (diagnostics only).

This module does not change production routing, execution, launchd state, or
Q2_BDI. It appends shadow-only Meta diagnostics and regenerates forward
monitoring artifacts from the accumulated shadow log.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="An input array is constant; the correlation coefficient is not defined.",
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.diagnostics.analyze_counterfactual_label_quality import FEATURE_COLS
from scripts.diagnostics.analyze_executed_anchor_routing import (
    LABEL_COL,
    POSITION_SIZE,
    _bucket_stats,
    _build_anchor_dataset,
    _monotonicity_score,
    _risk_buckets,
    _risk_routing_valid,
    _train_meta_model,
)
from scripts.diagnostics.analyze_executed_first_meta_learning import _prepare_full_df
from scripts.diagnostics.analyze_meta_activation_controller import (
    _add_regime_confidence,
    _apply_hysteresis,
    _meta_activation_score,
)
from scripts.diagnostics.analyze_regime_conditioned_meta import _q2_scale, _segment_regimes
from scripts.diagnostics.train_meta_layer_model import _prepare_x, _time_split

OUT_DIR = Path("data/diagnostics/meta_shadow")
SHADOW_PARQUET = OUT_DIR / "meta_shadow_log.parquet"
SHADOW_CSV = OUT_DIR / "meta_shadow_log.csv"
ROLLING_DAYS = 30
MIN_BUCKET_ROWS = 16


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _date_series(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(df.get("timestamp", df.get("entry_ts")), errors="coerce")
    fallback = pd.to_datetime(df.get("entry_ts"), errors="coerce")
    ts = ts.fillna(fallback)
    return ts.dt.date.astype(str).fillna("unknown")


def _mdd(rets: pd.Series, scales: pd.Series) -> float:
    eq = peak = 1.0
    mdd = 0.0
    for r, s in zip(rets.fillna(0), scales.fillna(1)):
        eq *= 1.0 + float(r) * float(s) * POSITION_SIZE
        peak = max(peak, eq)
        mdd = min(mdd, (eq - peak) / peak if peak > 0 else 0.0)
    return float(mdd)


def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else 0.0


def _risk_bucket_labels(scores: np.ndarray) -> np.ndarray:
    if len(scores) < 4:
        return np.array(["Q1"] * len(scores), dtype=object)
    return _risk_buckets(scores)


def _routing_consistency(df: pd.DataFrame, scores: np.ndarray) -> bool:
    if len(df) < 10:
        return False
    return bool(_risk_routing_valid(df, scores))


def _bucket_summary(df: pd.DataFrame, scores: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if len(df) < 4:
        empty = pd.DataFrame()
        return empty, {"monotonic": False, "spearman": None, "violations": 0}
    tmp = df.reset_index(drop=True).copy()
    if LABEL_COL not in tmp.columns and "drawdown_risk" in tmp.columns:
        tmp[LABEL_COL] = tmp["drawdown_risk"]
    if "engine_ret" not in tmp.columns and "actual_realized_return" in tmp.columns:
        tmp["engine_ret"] = tmp["actual_realized_return"]
    bdf = _bucket_stats(tmp, scores)
    mono = _monotonicity_score(bdf, "bad_trade_rate") if not bdf.empty else {
        "monotonic": False,
        "spearman": None,
        "violations": 0,
    }
    return bdf, mono


def _bucket_metric(bdf: pd.DataFrame, bucket: str, metric: str) -> float:
    if bdf.empty or "bucket" not in bdf.columns or metric not in bdf.columns:
        return float("nan")
    vals = bdf.loc[bdf["bucket"] == bucket, metric]
    return float(vals.iloc[0]) if len(vals) else float("nan")


def _policy_scales(df: pd.DataFrame, policy: str) -> pd.Series:
    q2 = df["q2_bdi_scale"].astype(float).copy()
    risk = df["meta_raw_score"].astype(float).clip(0, 1)

    if policy == "q2_only":
        active = pd.Series(False, index=df.index)
    elif policy == "always_on_meta":
        active = pd.Series(True, index=df.index)
    elif policy == "activation_controlled_meta":
        active = df["activation_policy_result"].astype(str).eq("ON")
    elif policy == "transition_aware_meta":
        active = (
            df["activation_policy_result"].astype(str).eq("ON")
            & (df["transition_probability"].astype(float) <= 0.35)
        )
    elif policy == "false_high_off_meta":
        active = ~df["false_high_signature"].astype(bool)
    elif policy == "vol_expansion_only_meta":
        active = df["vol_regime"].astype(str).eq("vol_expansion")
    else:
        active = pd.Series(False, index=df.index)

    out = q2.copy()
    out.loc[active] = (q2.loc[active] * (1.0 - 0.35 * risk.loc[active])).clip(0.05, 1.0)
    return out


def _policy_metrics(df: pd.DataFrame, policy: str) -> Dict[str, Any]:
    scales = _policy_scales(df, policy)
    active = scales < df["q2_bdi_scale"].astype(float)
    scores = df["meta_raw_score"].to_numpy(dtype=float)
    active_df = df[active]
    active_scores = scores[active.to_numpy()]
    bdf, mono = _bucket_summary(active_df, active_scores) if len(active_df) >= 4 else (pd.DataFrame(), {"monotonic": False})
    false_high = df["false_high_signature"].astype(bool)
    fh_active = active[false_high].mean() if false_high.any() else 0.0
    return {
        "policy": policy,
        "rows": int(len(df)),
        "active_rows": int(active.sum()),
        "active_pct": float(active.mean()) if len(active) else 0.0,
        "mdd": _mdd(df["actual_realized_return"], scales),
        "monotonicity": bool(mono.get("monotonic", False)),
        "monotonic_spearman": mono.get("spearman"),
        "routing_consistency": _routing_consistency(active_df, active_scores) if len(active_df) >= 10 else False,
        "false_high_suppression": float(1.0 - fh_active),
        "preservation": 1.0,
        "bucket_stability": float(1.0 - (mono.get("violations", 3) or 0) / 3.0) if len(bdf) >= 4 else 0.0,
    }


def _build_shadow_batch() -> pd.DataFrame:
    full = _prepare_full_df()
    anchor, _, weights = _build_anchor_dataset(full)
    anchor = _segment_regimes(anchor)
    anchor = _add_regime_confidence(anchor)
    train_df, val_df, test_df = _time_split(anchor)
    eval_df = pd.concat([val_df, test_df], ignore_index=True)

    model = _train_meta_model(train_df, weights[: len(train_df)])
    scores = model.predict_proba(_prepare_x(eval_df, FEATURE_COLS))[:, 1]
    eval_df = eval_df.copy().reset_index(drop=True)
    eval_df["meta_raw_score"] = scores
    eval_df["meta_risk_bucket"] = _risk_bucket_labels(scores)
    eval_df["activation_score"] = _meta_activation_score(eval_df, scores)
    eval_df["activation_policy_result"] = np.where(eval_df["activation_score"] >= 0.55, "ON", "OFF")
    eval_df["q2_bdi_scale"] = eval_df.apply(_q2_scale, axis=1)
    eval_df["shadow_run_ts"] = _now_ts()
    eval_df["shadow_date"] = _date_series(eval_df)

    out = pd.DataFrame({
        "timestamp": eval_df.get("timestamp", eval_df.get("entry_ts")),
        "entry_ts": eval_df.get("entry_ts"),
        "exit_ts": eval_df.get("exit_ts"),
        "trade_id": eval_df.get("trade_id", pd.Series(range(len(eval_df)))),
        "df_idx": eval_df.get("df_idx"),
        "direction": eval_df.get("direction"),
        "q2_bdi_scale": eval_df["q2_bdi_scale"],
        "meta_raw_score": eval_df["meta_raw_score"],
        "meta_risk_bucket": eval_df["meta_risk_bucket"],
        "regime_label": eval_df["regime_label"],
        "regime_confidence": eval_df["regime_confidence_score"],
        "regime_stability_score": eval_df["regime_stability_score"],
        "regime_persistence": eval_df["regime_persistence"],
        "activation_score": eval_df["activation_score"],
        "activation_policy_result": eval_df["activation_policy_result"],
        "false_high_signature": eval_df.get("false_high_signature_flag", 0).astype(bool),
        "entropy": eval_df.get("entropy"),
        "confidence_overextension": eval_df.get("confidence_overextension"),
        "vol_regime": eval_df["vol_regime"],
        "trend_regime": eval_df["trend_regime"],
        "confidence_regime": eval_df["confidence_regime"],
        "transition_probability": eval_df["transition_probability"],
        "actual_outcome": eval_df.get("quality_class", eval_df.get("trade_quality_label")),
        "mae": eval_df.get("mae"),
        "mfe": eval_df.get("mfe"),
        "rfe_flag": eval_df.get("rfe_flag", False).astype(bool),
        "drawdown_risk": eval_df.get(LABEL_COL),
        "actual_realized_return": eval_df.get("engine_ret", eval_df.get("net_return")),
        "binary_bad_trade": eval_df.get("binary_bad_trade"),
        "binary_good_trade": eval_df.get("binary_good_trade"),
        "shadow_date": eval_df["shadow_date"],
        "shadow_run_ts": eval_df["shadow_run_ts"],
        "promotion_ready": False,
    })
    return out


def _append_shadow_log(batch: pd.DataFrame, *, dry_run: bool = False) -> Tuple[pd.DataFrame, int]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SHADOW_PARQUET.exists():
        existing = pd.read_parquet(SHADOW_PARQUET)
    else:
        existing = pd.DataFrame()

    if existing.empty:
        new_rows = batch.copy()
        combined = batch.copy()
    else:
        existing_ids = set(existing["trade_id"].astype(str))
        new_rows = batch[~batch["trade_id"].astype(str).isin(existing_ids)].copy()
        combined = pd.concat([existing, new_rows], ignore_index=True) if len(new_rows) else existing.copy()

    combined = combined.drop_duplicates("trade_id", keep="first").sort_values(
        ["timestamp", "trade_id"], na_position="last"
    ).reset_index(drop=True)

    if not dry_run:
        combined.to_parquet(SHADOW_PARQUET, index=False)
        combined.to_csv(SHADOW_CSV, index=False)
    return combined, int(len(new_rows))


def _window_for_day(log: pd.DataFrame, day: str) -> pd.DataFrame:
    dates = pd.to_datetime(log["shadow_date"], errors="coerce")
    end = pd.to_datetime(day, errors="coerce")
    if pd.isna(end):
        return log[log["shadow_date"] == day].copy()
    start = end - pd.Timedelta(days=ROLLING_DAYS - 1)
    return log[(dates >= start) & (dates <= end)].copy()


def _write_forward_monotonicity(log: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_rows: List[Dict[str, Any]] = []
    bucket_rows: List[Dict[str, Any]] = []
    routing_rows: List[Dict[str, Any]] = []
    for day in sorted(log["shadow_date"].dropna().unique()):
        win = _window_for_day(log, day)
        scores = win["meta_raw_score"].to_numpy(dtype=float)
        bdf, mono = _bucket_summary(win, scores) if len(win) >= MIN_BUCKET_ROWS else (pd.DataFrame(), {"monotonic": False, "spearman": None, "violations": 0})
        routing = _routing_consistency(win, scores)
        daily_rows.append({
            "date": day,
            "window_days": ROLLING_DAYS,
            "rows": int(len(win)),
            "risk_bucket_monotonicity": bool(mono.get("monotonic", False)),
            "monotonic_spearman": mono.get("spearman"),
            "bucket_inversions": int(mono.get("violations", 0) or 0),
            "routing_consistency": routing,
            "preservation": 1.0,
            "good_trade_rejection": float((win["binary_good_trade"].astype(bool) & win["meta_risk_bucket"].eq("Q4")).mean()) if len(win) else 0.0,
            "bad_trade_rate": _safe_mean(win["binary_bad_trade"]),
            "rfe_rate": _safe_mean(win["rfe_flag"].astype(float)),
            "drawdown_risk_rate": _safe_mean(win["drawdown_risk"]),
        })
        if not bdf.empty:
            for row in bdf.to_dict(orient="records"):
                bucket_rows.append({"date": day, **row})
        routing_rows.append({
            "date": day,
            "rows": int(len(win)),
            "routing_consistency": routing,
            "q1_bad_rate": _bucket_metric(bdf, "Q1", "bad_trade_rate"),
            "q4_bad_rate": _bucket_metric(bdf, "Q4", "bad_trade_rate"),
            "q4_minus_q1_bad_rate": _bucket_metric(bdf, "Q4", "bad_trade_rate") - _bucket_metric(bdf, "Q1", "bad_trade_rate"),
        })

    daily = pd.DataFrame(daily_rows)
    buckets = pd.DataFrame(bucket_rows)
    routing = pd.DataFrame(routing_rows)
    daily.to_csv(OUT_DIR / "forward_monotonicity_daily.csv", index=False)
    buckets.to_csv(OUT_DIR / "bucket_ordering_history.csv", index=False)
    routing.to_csv(OUT_DIR / "routing_consistency_history.csv", index=False)
    return daily, buckets, routing


def _write_regime_tracking(log: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for day in sorted(log["shadow_date"].dropna().unique()):
        win = _window_for_day(log, day)
        for regime_col in ["regime_label", "vol_regime", "trend_regime", "confidence_regime"]:
            for regime_value, sub in win.groupby(regime_col, dropna=False):
                scores = sub["meta_raw_score"].to_numpy(dtype=float)
                bdf, mono = _bucket_summary(sub, scores) if len(sub) >= MIN_BUCKET_ROWS else (pd.DataFrame(), {"monotonic": False})
                rows.append({
                    "date": day,
                    "regime_type": regime_col,
                    "regime_value": str(regime_value),
                    "rows": int(len(sub)),
                    "bad_trade_rate": _safe_mean(sub["binary_bad_trade"]),
                    "rfe_rate": _safe_mean(sub["rfe_flag"].astype(float)),
                    "drawdown_risk_rate": _safe_mean(sub["drawdown_risk"]),
                    "mean_meta_score": _safe_mean(sub["meta_raw_score"]),
                    "monotonicity": bool(mono.get("monotonic", False)),
                    "routing_consistency": _routing_consistency(sub, scores),
                    "false_high_events": int(sub["false_high_signature"].astype(bool).sum()),
                    "vol_expansion_rows": int((sub["vol_regime"] == "vol_expansion").sum()),
                })
                if len(sub) >= MIN_BUCKET_ROWS and not bool(mono.get("monotonic", False)):
                    failures.append({
                        "date": day,
                        "failure_type": "regime_bucket_inversion",
                        "regime_type": regime_col,
                        "regime_value": str(regime_value),
                        "rows": int(len(sub)),
                        "transition_probability_mean": _safe_mean(sub["transition_probability"]),
                        "entropy_mean": _safe_mean(sub["entropy"]),
                        "false_high_events": int(sub["false_high_signature"].astype(bool).sum()),
                    })

        transition = win[win["transition_probability"].astype(float) > 0.35]
        if len(transition) >= MIN_BUCKET_ROWS:
            scores = transition["meta_raw_score"].to_numpy(dtype=float)
            _, mono = _bucket_summary(transition, scores)
            if not bool(mono.get("monotonic", False)):
                failures.append({
                    "date": day,
                    "failure_type": "post_transition_ordering_break",
                    "regime_type": "transition",
                    "regime_value": "transition_probability_gt_0.35",
                    "rows": int(len(transition)),
                    "transition_probability_mean": _safe_mean(transition["transition_probability"]),
                    "entropy_mean": _safe_mean(transition["entropy"]),
                    "false_high_events": int(transition["false_high_signature"].astype(bool).sum()),
                })

    tracking = pd.DataFrame(rows)
    failure_df = pd.DataFrame(failures)
    tracking.to_csv(OUT_DIR / "regime_forward_tracking.csv", index=False)
    failure_df.to_csv(OUT_DIR / "regime_transition_failures.csv", index=False)
    return tracking, failure_df


def _write_false_high_monitor(log: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for day in sorted(log["shadow_date"].dropna().unique()):
        win = _window_for_day(log, day)
        fh = win[win["false_high_signature"].astype(bool)]
        scores = fh["meta_raw_score"].to_numpy(dtype=float)
        _, mono = _bucket_summary(fh, scores) if len(fh) >= MIN_BUCKET_ROWS else (pd.DataFrame(), {"monotonic": False, "spearman": None})
        rows.append({
            "date": day,
            "rows": int(len(win)),
            "false_high_events": int(len(fh)),
            "false_high_frequency": float(len(fh) / max(len(win), 1)),
            "bad_trade_rate": _safe_mean(fh["binary_bad_trade"]) if len(fh) else 0.0,
            "meta_ordering_inversion": not bool(mono.get("monotonic", False)) if len(fh) >= MIN_BUCKET_ROWS else False,
            "monotonic_spearman": mono.get("spearman"),
            "long_bias_intensity": float((fh["direction"] == "LONG").mean()) if len(fh) else 0.0,
            "near_transition_rate": float((fh["transition_probability"].astype(float) > 0.35).mean()) if len(fh) else 0.0,
            "entropy_spike_rate": float((fh["entropy"].astype(float).diff().abs() > 0.08).mean()) if len(fh) else 0.0,
            "vol_expansion_rate": float((fh["vol_regime"] == "vol_expansion").mean()) if len(fh) else 0.0,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "false_high_monitor.csv", index=False)
    latest = out.iloc[-1].to_dict() if len(out) else {}
    (OUT_DIR / "false_high_regime_daily.md").write_text(
        "# False High Regime Daily\n\n"
        "False-high remains shadow-only and hard promotion blocked.\n\n"
        f"```json\n{json.dumps(latest, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )
    return out


def _write_policy_compare(log: pd.DataFrame) -> pd.DataFrame:
    policies = [
        "always_on_meta",
        "activation_controlled_meta",
        "transition_aware_meta",
        "false_high_off_meta",
        "vol_expansion_only_meta",
    ]
    rows: List[Dict[str, Any]] = []
    for day in sorted(log["shadow_date"].dropna().unique()):
        win = _window_for_day(log, day)
        for policy in policies:
            rows.append({"date": day, **_policy_metrics(win, policy)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "activation_policy_shadow_compare.csv", index=False)

    latest_day = out["date"].max() if len(out) else ""
    latest = out[out["date"] == latest_day].copy() if latest_day else pd.DataFrame()
    best = latest.sort_values(["monotonicity", "routing_consistency", "mdd"], ascending=[False, False, False]).head(1)
    best_policy = best.iloc[0].to_dict() if len(best) else {}
    (OUT_DIR / "activation_policy_daily_summary.md").write_text(
        "# Activation Policy Daily Summary\n\n"
        "Shadow comparison only. Q2_BDI remains the production baseline.\n\n"
        f"**Best shadow policy:** `{best_policy.get('policy', 'N/A')}`\n\n"
        f"```json\n{json.dumps(best_policy, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )
    return out


def _write_robustness(log: pd.DataFrame, mono_daily: pd.DataFrame, routing_daily: pd.DataFrame, false_high: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for day in sorted(log["shadow_date"].dropna().unique()):
        win = _window_for_day(log, day)
        md = mono_daily[mono_daily["date"] == day].tail(1)
        rd = routing_daily[routing_daily["date"] == day].tail(1)
        fh = false_high[false_high["date"] == day].tail(1)
        monotonic = bool(md["risk_bucket_monotonicity"].iloc[0]) if len(md) else False
        bucket_stability = 1.0 - float(md["bucket_inversions"].iloc[0]) / 3.0 if len(md) else 0.0
        regime_stability = float(win["regime_stability_score"].mean()) if len(win) else 0.0
        transition_robustness = 1.0 - float(win["transition_probability"].mean()) if len(win) else 0.0
        fh_inversion = float(fh["meta_ordering_inversion"].iloc[0]) if len(fh) else 0.0
        routing = bool(rd["routing_consistency"].iloc[0]) if len(rd) else False
        recent = mono_daily[mono_daily["date"] <= day].tail(7)
        rolling_consistency = float(recent["risk_bucket_monotonicity"].mean()) if len(recent) else 0.0
        score = (
            25.0 * float(monotonic)
            + 15.0 * max(0.0, bucket_stability)
            + 15.0 * regime_stability
            + 15.0 * max(0.0, transition_robustness)
            + 15.0 * (1.0 - fh_inversion)
            + 10.0 * float(routing)
            + 5.0 * rolling_consistency
        )
        rows.append({
            "date": day,
            "rows": int(len(win)),
            "meta_forward_robustness_score": float(np.clip(score, 0, 100)),
            "recent_monotonicity": monotonic,
            "bucket_stability": float(max(0.0, bucket_stability)),
            "regime_stability": regime_stability,
            "transition_robustness": transition_robustness,
            "false_high_inversion_frequency": fh_inversion,
            "routing_consistency": routing,
            "fold_equivalent_rolling_consistency": rolling_consistency,
            "promotion_ready": False,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "meta_forward_robustness_history.csv", index=False)
    latest = out.iloc[-1].to_dict() if len(out) else {}
    verdict = _final_verdict(out, false_high)
    (OUT_DIR / "forward_robustness_report.md").write_text(
        "# Forward Robustness Report\n\n"
        f"**Verdict:** `{verdict}`\n\n"
        "**Production status:** Q2_BDI unchanged; Meta shadow-only; promotion_ready=false.\n\n"
        f"```json\n{json.dumps(latest, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )
    return out


def _final_verdict(robustness: pd.DataFrame, false_high: pd.DataFrame) -> str:
    if robustness.empty:
        return "monitor_only_shadow_system"
    recent = robustness.tail(min(7, len(robustness)))
    mono_rate = float(recent["recent_monotonicity"].mean())
    routing_rate = float(recent["routing_consistency"].mean())
    score_mean = float(recent["meta_forward_robustness_score"].mean())
    fh_recent = false_high.tail(min(7, len(false_high))) if len(false_high) else pd.DataFrame()
    fh_inversion = float(fh_recent["meta_ordering_inversion"].mean()) if len(fh_recent) else 0.0
    if mono_rate >= 0.8 and routing_rate >= 0.8:
        return "forward_monotonicity_persistent"
    if fh_inversion >= 0.5:
        return "false_high_inversion_persistent"
    if score_mean >= 70 and routing_rate >= 0.6:
        return "meta_signal_survives_forward"
    return "monitor_only_shadow_system"


def _discord_summary(
    log: pd.DataFrame,
    rows_added: int,
    mono_daily: pd.DataFrame,
    policy_compare: pd.DataFrame,
    robustness: pd.DataFrame,
    false_high: pd.DataFrame,
) -> Dict[str, Any]:
    latest_day = log["shadow_date"].max() if len(log) else ""
    mono = mono_daily[mono_daily["date"] == latest_day].tail(1)
    rob = robustness[robustness["date"] == latest_day].tail(1)
    fh = false_high[false_high["date"] == latest_day].tail(1)
    policies = policy_compare[policy_compare["date"] == latest_day].copy()
    best_policy = "N/A"
    if len(policies):
        best_policy = str(
            policies.sort_values(["monotonicity", "routing_consistency", "mdd"], ascending=[False, False, False]).iloc[0]["policy"]
        )
    return {
        "date": latest_day,
        "rows_added": rows_added,
        "regime": str(log[log["shadow_date"] == latest_day]["regime_label"].mode().iloc[0]) if latest_day and len(log[log["shadow_date"] == latest_day]) else "N/A",
        "meta_forward_score": float(rob["meta_forward_robustness_score"].iloc[0]) if len(rob) else 0.0,
        "bucket_monotonicity": bool(mono["risk_bucket_monotonicity"].iloc[0]) if len(mono) else False,
        "routing_consistency": bool(mono["routing_consistency"].iloc[0]) if len(mono) else False,
        "false_high_events": int(fh["false_high_events"].iloc[0]) if len(fh) else 0,
        "activation_policy_best": best_policy,
        "Q2_vs_meta_shadow_MDD": {
            "q2_only": float(_policy_metrics(log[log["shadow_date"] == latest_day], "q2_only")["mdd"]) if latest_day else 0.0,
            "always_on_meta": float(_policy_metrics(log[log["shadow_date"] == latest_day], "always_on_meta")["mdd"]) if latest_day else 0.0,
        },
        "promotion_ready": False,
    }


def run_shadow_monitor(*, dry_run: bool = False) -> Dict[str, Any]:
    batch = _build_shadow_batch()
    log, rows_added = _append_shadow_log(batch, dry_run=dry_run)
    mono_daily, _, routing_daily = _write_forward_monotonicity(log)
    _write_regime_tracking(log)
    false_high = _write_false_high_monitor(log)
    policy_compare = _write_policy_compare(log)
    robustness = _write_robustness(log, mono_daily, routing_daily, false_high)
    summary = _discord_summary(log, rows_added, mono_daily, policy_compare, robustness, false_high)
    summary.update({
        "status": "PASS",
        "shadow_rows_total": int(len(log)),
        "shadow_rows_added": rows_added,
        "output_dir": str(OUT_DIR),
        "verdict": _final_verdict(robustness, false_high),
        "promotion_ready": False,
    })
    (OUT_DIR / "forward_meta_shadow_monitor_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward Meta shadow monitor (diagnostics only)")
    parser.add_argument("--dry-run", action="store_true", help="Do not append shadow log")
    args = parser.parse_args()
    summary = run_shadow_monitor(dry_run=args.dry_run)
    print(f"status: {summary['status']}")
    print(f"rows_added: {summary['rows_added']}")
    print(f"meta_forward_score: {summary['meta_forward_score']:.2f}")
    print(f"verdict: {summary['verdict']}")


if __name__ == "__main__":
    main()
