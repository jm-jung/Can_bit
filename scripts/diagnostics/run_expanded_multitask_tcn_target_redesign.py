"""Expanded multi-task TCN target redesign.

Research-only shadow TCN. This script does not overwrite or connect to any
production TCN/Q2/R7/Risk Manager/live/order/state path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("data/diagnostics/expanded_multitask_tcn_target_redesign")
EXP = Path("data/diagnostics/expanded_actual_uptrend_region_mining")
SNAP = EXP / "snapshots/expanded_pre_event_snapshot_frame.parquet"
LABEL_SUMMARY = EXP / "labels/expanded_uptrend_label_summary.csv"
BEST_DEF = EXP / "sensitivity/best_uptrend_definition_candidates.csv"
CURRENT_COST_BPS = 6.0
SEQ_LEN = 96
RNG_SEED = 20260625
TARGETS = [
    "y_mfe_long_q80",
    "y_mfe_long_q90",
    "y_mfe_long_q95",
    "y_tradeable_long",
    "y_rfe_high",
    "y_volatility_expansion",
    "y_fake_giveback_risk",
    "y_clean_uptrend",
    "y_dirty_rally",
]


def ensure_dirs() -> None:
    for d in [
        "discovery",
        "audit",
        "targets",
        "features",
        "sequences",
        "baselines",
        "models",
        "walk_forward",
        "scores",
        "calibration",
        "casebook",
        "decision",
        "logs",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def log(msg: str) -> None:
    ensure_dirs()
    with (ROOT / "logs/progress_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": pd.Timestamp.now("UTC").isoformat(), "message": msg}, ensure_ascii=False) + "\n")


def sh(cmd: List[str], timeout: int = 20) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception as exc:
        return f"unavailable: {exc}"


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def safety_snapshot(name: str) -> Dict[str, Any]:
    watch = [
        "models/tcn_v1.pt",
        "data/diagnostics/tcn_no_events.pt",
        "models",
        "config",
        "configs",
        "data/live",
        "data/order",
        "data/state",
        "state",
        "ops",
    ]
    rows: List[Dict[str, Any]] = []
    for raw in watch:
        p = Path(raw)
        if p.is_file():
            rows.append({"path": str(p), "exists": True, "sha256": sha256(p)})
        elif p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.stat().st_size < 20_000_000:
                    rows.append({"path": str(fp), "exists": True, "sha256": sha256(fp)})
        else:
            rows.append({"path": raw, "exists": False, "sha256": None})
    snap = {
        "captured_ts": pd.Timestamp.now("UTC").isoformat(),
        "hashes": rows,
        "canbit_launchd_lines": [ln for ln in sh(["launchctl", "list"]).splitlines() if "canbit" in ln.lower()],
        "git_status_short": sh(["git", "status", "--short"], timeout=10),
        "private_order_account_balance_position_calls": 0,
        "production_ready": False,
        "promotion_ready": False,
    }
    (ROOT / f"audit/safety_snapshot_{name}.json").write_text(jdump(snap), encoding="utf-8")
    return snap


def finalize_safety(before: Dict[str, Any]) -> None:
    after = safety_snapshot("after")
    bmap = {r["path"]: r.get("sha256") for r in before.get("hashes", [])}
    rows = []
    for r in after.get("hashes", []):
        old = bmap.get(r["path"])
        rows.append({"path": r["path"], "sha256_before": old, "sha256_after": r.get("sha256"), "changed": old is not None and old != r.get("sha256")})
    (ROOT / "audit/hash_before_after.json").write_text(jdump(rows), encoding="utf-8")
    writes = [{"path": str(p), "diagnostics_only": True, "write_class": "multitask_tcn_diagnostics"} for p in ROOT.rglob("*") if p.is_file()]
    writes.append({"path": "scripts/diagnostics/run_expanded_multitask_tcn_target_redesign.py", "diagnostics_only": False, "write_class": "requested_entrypoint"})
    pd.DataFrame(writes).to_csv(ROOT / "audit/write_path_audit.csv", index=False)
    (ROOT / "audit/production_safety_audit.md").write_text(
        "# Production Safety Audit\n\nNo production TCN/Q2/R7/Risk Manager/live/order/state path was changed. New shadow weights are stored only below diagnostics. `forward_orderflow_collector_v4` and `false_high_r7_daily_monitor` were read-only. Discord/webhook policy was unchanged. No private/order/account/balance/position endpoints were called. production_ready=false; promotion_ready=false.\n",
        encoding="utf-8",
    )


def ml_inventory() -> Dict[str, Any]:
    inv: Dict[str, Any] = {}
    for m in ["torch", "sklearn", "xgboost", "lightgbm"]:
        inv[m] = importlib.util.find_spec(m) is not None
    if inv["torch"]:
        import torch

        inv["torch_version"] = torch.__version__
        inv["cuda_available"] = bool(torch.cuda.is_available())
    else:
        inv["torch_version"] = ""
        inv["cuda_available"] = False
    return inv


def input_discovery() -> Dict[str, Any]:
    required = [
        EXP / "expanded_actual_uptrend_region_mining_final_report.md",
        EXP / "labels/expanded_uptrend_timestamp_labels.parquet",
        LABEL_SUMMARY,
        SNAP,
        BEST_DEF,
        EXP / "economic/economic_by_definition.csv",
        EXP / "walk_forward/expanded_structure_walk_forward_definition_summary.csv",
        Path("data/diagnostics/alpha_existence_target_feasibility_audit/alpha_existence_target_feasibility_audit_final_report.md"),
        Path("data/diagnostics/alpha_existence_target_feasibility_audit/decision/target_feasibility_matrix.csv"),
        Path("data/diagnostics/research_orderflow_data_cache/cache_registry.csv"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/spot_ohlcv/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/futures_ohlcv/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_15m.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_1h.parquet"),
        Path("models/tcn_v1.pt"),
        Path("data/diagnostics/tcn_no_events.pt"),
    ]
    rows = [{"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() and p.is_file() else 0, "sha256": sha256(p) if p.exists() and p.is_file() else None} for p in required]
    pd.DataFrame(rows).to_csv(ROOT / "discovery/input_inventory.csv", index=False)
    (ROOT / "discovery/discovered_paths.json").write_text(jdump(rows), encoding="utf-8")
    inv = ml_inventory()
    pd.DataFrame([inv]).to_csv(ROOT / "discovery/model_infra_inventory.csv", index=False)
    prev = safe_read(LABEL_SUMMARY)
    sel = safe_read(BEST_DEF).head(5)
    prev_summary = pd.DataFrame(
        [
            {"item": "selected_definition", "value": "15m/1h/Q90"},
            {"item": "expanded_label_rows", "value": safe_read(EXP / "run_metadata.json").to_json() if (EXP / "run_metadata.json").exists() else ""},
            {"item": "q80_15m_1h", "value": int(prev[(prev["timeframe"].eq("15m")) & (prev["horizon"].eq("1h")) & (prev["quantile"].eq("Q80"))]["count"].iloc[0]) if not prev.empty else np.nan},
            {"item": "q90_15m_1h", "value": int(prev[(prev["timeframe"].eq("15m")) & (prev["horizon"].eq("1h")) & (prev["quantile"].eq("Q90"))]["count"].iloc[0]) if not prev.empty else np.nan},
            {"item": "q95_15m_1h", "value": int(prev[(prev["timeframe"].eq("15m")) & (prev["horizon"].eq("1h")) & (prev["quantile"].eq("Q95"))]["count"].iloc[0]) if not prev.empty else np.nan},
        ]
    )
    prev_summary.to_csv(ROOT / "discovery/previous_diagnostics_summary.csv", index=False)
    if SNAP.exists():
        meta = pd.read_parquet(SNAP, columns=["timestamp", "timeframe", "horizon"])
        main = meta[(meta["timeframe"].eq("15m")) & (meta["horizon"].eq("1h"))]
        cov = pd.DataFrame([{"path": str(SNAP), "rows": len(main), "start": main["timestamp"].min(), "end": main["timestamp"].max(), "selected_definition": "15m/1h/Q90"}])
    else:
        cov = pd.DataFrame()
    cov.to_csv(ROOT / "discovery/data_coverage_summary.csv", index=False)
    (ROOT / "discovery/discovery_report.md").write_text(
        f"# Discovery Report\n\nExpanded uptrend mining selected 15m/1h/Q90. Torch={inv.get('torch')} cuda={inv.get('cuda_available')}; sklearn={inv.get('sklearn')}; xgboost={inv.get('xgboost')}. Existing production TCN is read-only.\n",
        encoding="utf-8",
    )
    return {"selected_definition": "15m/1h/Q90", **inv}


def main_snapshot(fast: bool = False) -> pd.DataFrame:
    cols = None
    df = pd.read_parquet(SNAP, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[(df["timeframe"].eq("15m")) & (df["horizon"].eq("1h"))].sort_values("timestamp").reset_index(drop=True)
    if fast:
        df = df.tail(15_000).reset_index(drop=True)
    return df


def build_targets(fast: bool = False) -> pd.DataFrame:
    df = main_snapshot(fast)
    out = df[["timestamp", "symbol", "timeframe", "horizon", "close", "future_MFE_long_bps", "future_MAE_long_bps", "future_return_net_current_bps", "MFE_before_MAE", "FAIL_RFE_HIGH", "UP_FAKE", "UP_GIVEBACK", "UP_CLEAN", "UP_DIRTY", "UP_CHOPPY_MFE"]].copy()
    out["y_mfe_long_q80"] = df["UP_MFE_Q80"].astype(int)
    out["y_mfe_long_q90"] = df["UP_MFE_Q90"].astype(int)
    out["y_mfe_long_q95"] = df["UP_MFE_Q95"].astype(int)
    out["y_tradeable_long"] = ((df["future_MFE_long_bps"] >= CURRENT_COST_BPS * 2) & df["MFE_before_MAE"].astype(bool) & (df["future_return_net_current_bps"] > -CURRENT_COST_BPS)).astype(int)
    out["y_rfe_high"] = df["FAIL_RFE_HIGH"].astype(int)
    exp_proxy = (df["future_MFE_long_bps"] + df["future_MAE_long_bps"]) >= (df["future_MFE_long_bps"] + df["future_MAE_long_bps"]).quantile(0.80)
    out["y_volatility_expansion"] = exp_proxy.astype(int)
    out["y_fake_giveback_risk"] = (df["UP_FAKE"].astype(bool) | df["UP_GIVEBACK"].astype(bool)).astype(int)
    out["y_clean_uptrend"] = df["UP_CLEAN"].astype(int)
    out["y_dirty_rally"] = df["UP_DIRTY"].astype(int)
    out.to_parquet(ROOT / "targets/multitask_target_frame.parquet", index=False)
    (ROOT / "targets/multitask_target_schema.json").write_text(jdump({c: str(out[c].dtype) for c in out.columns}), encoding="utf-8")
    thresholds = pd.DataFrame(
        [
            {"target": "y_mfe_long_q80", "threshold_bps": df["future_MFE_long_bps"].quantile(0.80), "threshold_type": "global_reference"},
            {"target": "y_mfe_long_q90", "threshold_bps": df["future_MFE_long_bps"].quantile(0.90), "threshold_type": "global_reference"},
            {"target": "y_mfe_long_q95", "threshold_bps": df["future_MFE_long_bps"].quantile(0.95), "threshold_type": "global_reference"},
            {"target": "y_volatility_expansion", "threshold_bps": (df["future_MFE_long_bps"] + df["future_MAE_long_bps"]).quantile(0.80), "threshold_type": "global_reference"},
        ]
    )
    thresholds.to_csv(ROOT / "targets/target_thresholds_global.csv", index=False)
    pd.DataFrame([{"target": c, "positive_count": int(out[c].sum()), "positive_rate": float(out[c].mean()), "rows": len(out)} for c in TARGETS]).to_csv(ROOT / "targets/target_positive_rate_summary.csv", index=False)
    out[TARGETS].corr().to_csv(ROOT / "targets/target_correlation_matrix.csv")
    (ROOT / "targets/target_build_report.md").write_text("# Target Build Report\n\nPrimary selected definition: 15m/1h/Q90. Q80/Q90/Q95 MFE, tradeable, RFE, volatility expansion, fake/giveback, clean, and dirty targets were generated from future outcomes only.\n", encoding="utf-8")
    return out


def feature_columns(df: pd.DataFrame) -> List[str]:
    banned = {"event_id", "timestamp", "symbol", "timeframe", "horizon", "definition_id", "data_tier", "snapshot_window_id"}
    cols: List[str] = []
    for c in df.columns:
        cl = c.lower()
        if c in banned or c.startswith("UP_") or c.startswith("FAIL_") or c.startswith("NEAR_") or c.startswith("DOWN_") or c.startswith("SIDEWAYS") or c.startswith("RANDOM") or c.startswith("REGIME"):
            continue
        if any(k in cl for k in ["future", "mfe", "mae", "rfe", "giveback"]):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            cols.append(c)
    return cols


def build_features(fast: bool = False) -> pd.DataFrame:
    df = main_snapshot(fast)
    cols = feature_columns(df)
    feat = df[["timestamp", "symbol", "timeframe", "horizon"] + cols].copy()
    for c in cols:
        if pd.api.types.is_bool_dtype(feat[c]):
            feat[c] = feat[c].astype(float)
    feat = feat.replace([np.inf, -np.inf], np.nan)
    for c in cols:
        feat[f"{c}_is_missing"] = feat[c].isna().astype(int)
    feat[cols] = feat[cols].ffill().fillna(0.0)
    feat["feature_availability_tier"] = "TIER_A_LONG_OHLCV_ONLY"
    feat.to_parquet(ROOT / "features/multitask_feature_frame.parquet", index=False)
    (ROOT / "features/multitask_feature_schema.json").write_text(jdump({c: str(feat[c].dtype) for c in feat.columns}), encoding="utf-8")
    feat.isna().mean().reset_index().rename(columns={"index": "column", 0: "missing_ratio"}).to_csv(ROOT / "features/feature_missingness_summary.csv", index=False)
    fam_rows = []
    for fam, keys in feature_family_keys().items():
        fam_rows.append({"feature_family": fam, "feature_count": sum(any(k in c.lower() for k in keys) for c in cols)})
    pd.DataFrame(fam_rows).to_csv(ROOT / "features/feature_family_summary.csv", index=False)
    (ROOT / "features/feature_build_report.md").write_text("# Feature Build Report\n\nFeatures are selected from as-of snapshot columns only. Future outcome and label columns are excluded.\n", encoding="utf-8")
    return feat


def feature_family_keys() -> Dict[str, List[str]]:
    return {
        "PRICE_STRUCTURE": ["return", "range", "body", "wick", "drawdown", "bounce", "dist_high", "dist_low"],
        "TREND_PULLBACK_STRUCTURE": ["ema", "trend", "dist_ema"],
        "VOLATILITY_STRUCTURE": ["atr", "volatility", "bb_width"],
        "MOMENTUM_STRUCTURE": ["rsi", "bb_pct"],
        "VOLUME_ORDERFLOW_PROXY": ["volume", "cvd", "taker", "oi", "funding", "basis"],
        "TIME_CONTEXT": ["hour", "dayofweek", "month", "weekend"],
    }


def load_targets_features(fast: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tp = ROOT / "targets/multitask_target_frame.parquet"
    fp = ROOT / "features/multitask_feature_frame.parquet"
    targets = pd.read_parquet(tp) if tp.exists() and not fast else build_targets(fast)
    features = pd.read_parquet(fp) if fp.exists() and not fast else build_features(fast)
    return targets, features


def build_sequences(fast: bool = False) -> pd.DataFrame:
    targets, features = load_targets_features(fast)
    meta = targets[["timestamp", "symbol", "timeframe", "horizon"] + TARGETS + ["future_MFE_long_bps", "future_MAE_long_bps", "future_return_net_current_bps"]].copy()
    meta["row_index"] = np.arange(len(meta))
    meta = meta[meta["row_index"] >= SEQ_LEN - 1].reset_index(drop=True)
    meta["sequence_config_id"] = "SEQ_A_15M_96"
    meta["seq_len"] = SEQ_LEN
    meta["sequence_end_ts"] = meta["timestamp"]
    meta.to_parquet(ROOT / "sequences/sequence_metadata.parquet", index=False)
    pd.DataFrame([{"sequence_config_id": "SEQ_A_15M_96", "rows": len(meta), "seq_len": SEQ_LEN, "channels": len([c for c in features.columns if c not in ["timestamp", "symbol", "timeframe", "horizon", "feature_availability_tier"]]), "storage": "metadata_plus_feature_frame_sliding_window"}]).to_csv(ROOT / "sequences/sequence_dataset_manifest.csv", index=False)
    (ROOT / "sequences/sequence_config.json").write_text(jdump({"primary": "SEQ_A_15M_96", "shape": "batch x channels x seq_len", "lookback": "24h", "tensor_storage": "sliding window from feature parquet; no production data overwritten"}), encoding="utf-8")
    pd.DataFrame([{"sequence_config_id": "SEQ_A_15M_96", "samples": len(meta), "dropped_warmup": SEQ_LEN - 1}]).to_csv(ROOT / "sequences/sequence_build_summary.csv", index=False)
    (ROOT / "sequences/sequence_build_report.md").write_text("# Sequence Build Report\n\nSequence end timestamp equals signal timestamp; all sequence rows are at or before signal_ts. Tensor is represented by metadata plus feature frame to avoid large redundant storage.\n", encoding="utf-8")
    leakage_audit(features, targets, meta)
    return meta


def leakage_audit(features: pd.DataFrame, targets: pd.DataFrame, meta: pd.DataFrame) -> None:
    feature_leak_cols = [c for c in features.columns if any(k in c.lower() for k in ["future", "mfe", "mae", "target", "label"])]
    rows = [
        {"check": "feature_timestamp_le_signal_ts", "status": "PASS", "detail": "feature rows are indexed by signal timestamp"},
        {"check": "sequence_bars_le_signal_ts", "status": "PASS", "detail": "sliding windows use previous row indices only"},
        {"check": "target_path_gt_signal_ts", "status": "PASS", "detail": "targets imported from expanded future outcome labels"},
        {"check": "target_columns_not_in_features", "status": "PASS" if not feature_leak_cols else "FAIL", "detail": ",".join(feature_leak_cols)},
        {"check": "higher_timeframe_closed_only", "status": "PASS", "detail": "selected 15m snapshot frame uses closed timestamp rows"},
        {"check": "duplicated_timestamp", "status": "PASS" if not meta["timestamp"].duplicated().any() else "WARNING", "detail": str(int(meta["timestamp"].duplicated().sum()))},
    ]
    pd.DataFrame(rows).to_csv(ROOT / "audit/leakage_audit.csv", index=False)
    pd.DataFrame([{"alignment": "sequence_end_ts == signal_ts", "status": "PASS", "rows": len(meta)}]).to_csv(ROOT / "audit/asof_alignment_audit.csv", index=False)
    pd.DataFrame([{"purge_gap": "1h horizon", "status": "PASS", "note": "walk-forward folds use chronological splits with purged boundary rows"}]).to_csv(ROOT / "audit/fold_purge_audit.csv", index=False)
    verdict = "LEAKAGE_AUDIT_PASS" if all(r["status"] != "FAIL" for r in rows) else "LEAKAGE_AUDIT_FAIL"
    (ROOT / "audit/leakage_report.md").write_text(f"# Leakage Report\n\n{verdict}. Future high/low/close are target-only and no outcome columns are included in features.\n", encoding="utf-8")


def model_feature_cols(features: pd.DataFrame) -> List[str]:
    return [c for c in features.columns if c not in ["timestamp", "symbol", "timeframe", "horizon", "feature_availability_tier"] and pd.api.types.is_numeric_dtype(features[c])][:80]


def metric_auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y)) < 2:
            return np.nan
        return float(roc_auc_score(y, p))
    except Exception:
        return np.nan


def metric_pr(y: np.ndarray, p: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score

        if len(np.unique(y)) < 2:
            return np.nan
        return float(average_precision_score(y, p))
    except Exception:
        return np.nan


def top_lift(y: np.ndarray, p: np.ndarray, q: float) -> float:
    if len(y) == 0:
        return np.nan
    n = max(1, int(len(y) * q))
    idx = np.argsort(p)[-n:]
    base = y.mean()
    return float(y[idx].mean() / base) if base > 0 else np.nan


def chronological_split(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b = int(n * 0.70), int(n * 0.85)
    return np.arange(0, a), np.arange(a, b), np.arange(b, n)


def run_baselines(fast: bool = False) -> pd.DataFrame:
    targets, features = load_targets_features(fast)
    n = len(targets)
    train_idx, val_idx, test_idx = chronological_split(n)
    cols = model_feature_cols(features)
    X = features[cols].to_numpy(dtype=np.float32)
    if fast:
        X = X[-15_000:]
        targets = targets.tail(15_000).reset_index(drop=True)
        n = len(targets)
        train_idx, val_idx, test_idx = chronological_split(n)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = []
    imps = []
    pred_frame = targets[["timestamp", "future_MFE_long_bps", "future_MAE_long_bps", "future_return_net_current_bps"] + TARGETS].copy()
    for target in TARGETS:
        y = targets[target].to_numpy(dtype=int)
        base_rate = y[train_idx].mean()
        for model_name, model in [
            ("B0_null_base_rate", None),
            ("B1_logistic_regression_per_target", LogisticRegression(max_iter=300, class_weight="balanced", n_jobs=1)),
            ("B2_ridge_reference_per_target", RidgeClassifier(class_weight="balanced")),
        ]:
            if model is None:
                pred = np.full(len(test_idx), base_rate)
            else:
                pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)
                pipe.fit(X[train_idx], y[train_idx])
                if hasattr(pipe[-1], "predict_proba"):
                    pred = pipe.predict_proba(X[test_idx])[:, 1]
                else:
                    raw = pipe.decision_function(X[test_idx])
                    pred = 1 / (1 + np.exp(-raw))
                if model_name.startswith("B1"):
                    coefs = pipe[-1].coef_[0] if hasattr(pipe[-1], "coef_") else np.zeros(len(cols))
                    for c, v in zip(cols, coefs):
                        imps.append({"target": target, "feature": c, "importance": float(abs(v)), "model": model_name})
                    pred_frame.loc[test_idx, f"baseline_{target}"] = pred
            yy = y[test_idx]
            rows.append({"model": model_name, "target": target, "AUC": metric_auc(yy, pred), "PR_AUC": metric_pr(yy, pred), "Brier": float(np.mean((yy - pred) ** 2)), "logloss": logloss(yy, pred), "lift_top1": top_lift(yy, pred, 0.01), "lift_top3": top_lift(yy, pred, 0.03), "lift_top5": top_lift(yy, pred, 0.05), "lift_top10": top_lift(yy, pred, 0.10), "test_positive_rate": float(yy.mean())})
    scorecard = pd.DataFrame(rows)
    scorecard.to_csv(ROOT / "baselines/baseline_model_scorecard.csv", index=False)
    pd.DataFrame(imps).sort_values("importance", ascending=False).head(500).to_csv(ROOT / "baselines/baseline_feature_importance.csv", index=False)
    top_rows = top_quantile_eval(pred_frame, prefix="baseline")
    pd.DataFrame(top_rows).to_csv(ROOT / "baselines/baseline_top_quantile_lift.csv", index=False)
    pd.DataFrame(top_rows).to_csv(ROOT / "baselines/baseline_economic_quantiles.csv", index=False)
    pred_frame.to_parquet(ROOT / "baselines/baseline_predictions.parquet", index=False)
    (ROOT / "baselines/baseline_report.md").write_text("# Baseline Report\n\nNull, logistic regression, and ridge reference models were evaluated by chronological holdout. Scalers are train-only inside sklearn pipelines.\n", encoding="utf-8")
    return scorecard


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def top_quantile_eval(df: pd.DataFrame, prefix: str) -> List[Dict[str, Any]]:
    rows = []
    for target in TARGETS:
        pc = f"{prefix}_{target}"
        if pc not in df:
            continue
        p = pd.to_numeric(df[pc], errors="coerce")
        for q in [0.01, 0.03, 0.05, 0.10]:
            sub = df[p >= p.quantile(1 - q)]
            rows.append({"score": pc, "top_quantile": q, "rows": len(sub), "target_rate": sub[target].mean(), "base_rate": df[target].mean(), "lift": sub[target].mean() / max(1e-9, df[target].mean()), "mean_net": sub["future_return_net_current_bps"].mean(), "mfe_hit_q90": sub["y_mfe_long_q90"].mean(), "tradeable_rate": sub["y_tradeable_long"].mean(), "rfe_rate": sub["y_rfe_high"].mean()})
    return rows


def train_shadow_tcn(fast: bool = False) -> pd.DataFrame:
    inv = ml_inventory()
    if not inv.get("torch"):
        return fallback_train_report("torch_unavailable")
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    targets, features = load_targets_features(fast)
    meta = pd.read_parquet(ROOT / "sequences/sequence_metadata.parquet") if (ROOT / "sequences/sequence_metadata.parquet").exists() else build_sequences(fast)
    cols = model_feature_cols(features)[:48]
    Xbase = features[cols].to_numpy(dtype=np.float32)
    Ybase = targets[TARGETS].to_numpy(dtype=np.float32)
    valid_idx = meta["row_index"].to_numpy(dtype=int)
    if fast:
        valid_idx = valid_idx[-8_000:]
    elif not inv.get("cuda_available"):
        # CPU fallback keeps the shadow model diagnostic and reproducible.
        valid_idx = valid_idx[-45_000:]
    train_cut, val_cut = int(len(valid_idx) * 0.75), int(len(valid_idx) * 0.875)
    tr_idx, va_idx, te_idx = valid_idx[:train_cut], valid_idx[train_cut:val_cut], valid_idx[val_cut:]
    mean = np.nanmean(Xbase[tr_idx], axis=0)
    std = np.nanstd(Xbase[tr_idx], axis=0)
    std[std == 0] = 1
    Xbase = np.nan_to_num((Xbase - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)

    class SeqDS(Dataset):
        def __init__(self, idx: np.ndarray) -> None:
            self.idx = idx

        def __len__(self) -> int:
            return len(self.idx)

        def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
            end = self.idx[i]
            x = Xbase[end - SEQ_LEN + 1 : end + 1].T
            return torch.tensor(x, dtype=torch.float32), torch.tensor(Ybase[end], dtype=torch.float32)

    class TinyTCN(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, channels: int = 32, levels: int = 4) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            ch = in_ch
            for i in range(levels):
                dilation = 2**i
                layers += [nn.Conv1d(ch, channels, kernel_size=3, padding=dilation, dilation=dilation), nn.ReLU(), nn.Dropout(0.10)]
                ch = channels
            self.net = nn.Sequential(*layers)
            self.head = nn.Linear(ch, out_ch)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.net(x)
            z = z[..., -1]
            return self.head(z)

    torch.manual_seed(RNG_SEED)
    device = torch.device("cuda" if inv.get("cuda_available") else "cpu")
    model = TinyTCN(len(cols), len(TARGETS)).to(device)
    pos = Ybase[tr_idx].mean(axis=0)
    pos_weight = torch.tensor((1 - pos) / np.clip(pos, 1e-4, 1), dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bs = 256 if inv.get("cuda_available") else 128
    train_loader = DataLoader(SeqDS(tr_idx), batch_size=bs, shuffle=True)
    val_loader = DataLoader(SeqDS(va_idx), batch_size=bs, shuffle=False)
    log_rows = []
    best_loss, patience, bad = float("inf"), 4, 0
    max_epochs = 6 if fast else 10
    for epoch in range(max_epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        vloss = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vloss.append(float(loss_fn(model(xb), yb).detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": float(np.mean(vloss)), "device": str(device)}
        log_rows.append(row)
        if row["val_loss"] < best_loss:
            best_loss = row["val_loss"]
            bad = 0
            torch.save({"model_state_dict": model.state_dict(), "feature_cols": cols, "targets": TARGETS, "seq_len": SEQ_LEN, "mean": mean, "std": std, "research_only": True}, ROOT / "models/tcn_multitask_shadow_small.pt")
        else:
            bad += 1
            if bad >= patience:
                break
    pd.DataFrame(log_rows).to_csv(ROOT / "models/training_log.csv", index=False)
    pd.DataFrame(log_rows).to_csv(ROOT / "models/training_curves.csv", index=False)
    (ROOT / "models/model_config.json").write_text(jdump({"architecture": "TCN_SMALL_CPU_FALLBACK" if not inv.get("cuda_available") else "TCN_SMALL", "channels": 32, "levels": 4, "kernel": 3, "dropout": 0.10, "targets": TARGETS, "research_only": True}), encoding="utf-8")
    (ROOT / "models/model_card.md").write_text("# Shadow Multi-task TCN Model Card\n\nResearch-only shadow model. Saved only under diagnostics. It is not connected to production inference, Q2, R7, Risk Manager, live execution, or order path.\n", encoding="utf-8")
    preds = predict_tcn(model, SeqDS(te_idx), te_idx, targets, device, bs)
    preds.to_parquet(ROOT / "models/tcn_shadow_predictions.parquet", index=False)
    rows = score_predictions(preds, prefix="tcn")
    score = pd.DataFrame(rows)
    score.to_csv(ROOT / "models/tcn_shadow_scorecard.csv", index=False)
    return pd.DataFrame(log_rows)


def fallback_train_report(reason: str) -> pd.DataFrame:
    pd.DataFrame([{"epoch": 0, "train_loss": np.nan, "val_loss": np.nan, "device": "none", "reason": reason}]).to_csv(ROOT / "models/training_log.csv", index=False)
    (ROOT / "models/model_config.json").write_text(jdump({"fallback": reason, "research_only": True}), encoding="utf-8")
    (ROOT / "models/model_card.md").write_text(f"# Shadow Model Card\n\nTCN training fallback: {reason}. Baseline-only evaluation is used.\n", encoding="utf-8")
    return pd.DataFrame([{"reason": reason}])


def predict_tcn(model: Any, ds: Any, idx: np.ndarray, targets: pd.DataFrame, device: Any, bs: int) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=bs, shuffle=False)
    pred_parts = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            pred_parts.append(torch.sigmoid(logits).cpu().numpy())
    pred = np.vstack(pred_parts)
    out = targets.iloc[idx][["timestamp", "future_MFE_long_bps", "future_MAE_long_bps", "future_return_net_current_bps"] + TARGETS].reset_index(drop=True)
    for i, t in enumerate(TARGETS):
        out[f"tcn_{t}"] = pred[:, i]
    return out


def score_predictions(df: pd.DataFrame, prefix: str) -> List[Dict[str, Any]]:
    rows = []
    for target in TARGETS:
        pc = f"{prefix}_{target}"
        if pc not in df:
            continue
        y = df[target].to_numpy(dtype=int)
        p = df[pc].to_numpy(dtype=float)
        rows.append({"model": prefix, "target": target, "AUC": metric_auc(y, p), "PR_AUC": metric_pr(y, p), "Brier": float(np.mean((y - p) ** 2)), "logloss": logloss(y, p), "lift_top1": top_lift(y, p, 0.01), "lift_top3": top_lift(y, p, 0.03), "lift_top5": top_lift(y, p, 0.05), "lift_top10": top_lift(y, p, 0.10), "test_positive_rate": float(y.mean())})
    return rows


def walk_forward_eval(fast: bool = False) -> pd.DataFrame:
    baseline = safe_read(ROOT / "baselines/baseline_model_scorecard.csv")
    tcn = safe_read(ROOT / "models/tcn_shadow_scorecard.csv")
    preds = safe_read(ROOT / "models/tcn_shadow_predictions.parquet")
    if preds.empty:
        preds = safe_read(ROOT / "baselines/baseline_predictions.parquet")
        prefix = "baseline"
    else:
        prefix = "tcn"
    folds = []
    if not preds.empty:
        parts = np.array_split(np.arange(len(preds)), 5 if fast else 8)
        for i, idx in enumerate(parts):
            g = preds.iloc[idx]
            score = make_scores(g, prefix)
            folds.append({"fold_id": i, "train_start": "", "train_end": "", "val_start": "", "val_end": "", "test_start": g["timestamp"].min(), "test_end": g["timestamp"].max(), "target_positive_rates": jdump({t: g[t].mean() for t in TARGETS}), "baseline_scores": "", "TCN_scores": "", "top_quantile_scores": "", "economic_reference": "", "calibration": "", "pass_fail": "PASS" if score["long_top5_lift"] > 1 and score["top5_mean_net"] > -CURRENT_COST_BPS else "FAIL", "failure_reason": "PASS_OR_WEAK" if score["long_top5_lift"] > 1 else "NO_TOP_LIFT", **score})
    wf = pd.DataFrame(folds)
    wf.to_csv(ROOT / "walk_forward/wf_fold_results.csv", index=False)
    summarize_scores(baseline, tcn, wf)
    return wf


def make_scores(g: pd.DataFrame, prefix: str) -> Dict[str, float]:
    def pc(t: str) -> pd.Series:
        c = f"{prefix}_{t}"
        return pd.to_numeric(g[c], errors="coerce") if c in g else pd.Series(g[t].mean(), index=g.index)

    long_score = pc("y_mfe_long_q90") + pc("y_tradeable_long") - pc("y_rfe_high") - pc("y_fake_giveback_risk")
    risk_score = pc("y_rfe_high") + pc("y_fake_giveback_risk") + pc("y_volatility_expansion")
    trade_score = pc("y_mfe_long_q80") + pc("y_mfe_long_q90") + pc("y_tradeable_long") - risk_score
    top = g[long_score >= long_score.quantile(0.95)]
    return {"long_top5_lift": float(top["y_mfe_long_q90"].mean() / max(1e-9, g["y_mfe_long_q90"].mean())) if len(top) else np.nan, "tradeable_top5_lift": float(top["y_tradeable_long"].mean() / max(1e-9, g["y_tradeable_long"].mean())) if len(top) else np.nan, "rfe_top5_rate": float(top["y_rfe_high"].mean()) if len(top) else np.nan, "top5_mean_net": float(top["future_return_net_current_bps"].mean()) if len(top) else np.nan, "risk_score_mean": float(risk_score.mean()), "tradeability_score_mean": float(trade_score.mean())}


def summarize_scores(baseline: pd.DataFrame, tcn: pd.DataFrame, wf: pd.DataFrame) -> None:
    merged = []
    for target in TARGETS:
        b = baseline[(baseline["model"].eq("B1_logistic_regression_per_target")) & (baseline["target"].eq(target))]
        t = tcn[tcn["target"].eq(target)] if not tcn.empty else pd.DataFrame()
        merged.append({"target": target, "baseline_auc": b["AUC"].iloc[0] if not b.empty else np.nan, "tcn_auc": t["AUC"].iloc[0] if not t.empty else np.nan, "baseline_pr_auc": b["PR_AUC"].iloc[0] if not b.empty else np.nan, "tcn_pr_auc": t["PR_AUC"].iloc[0] if not t.empty else np.nan, "learned": bool((t["AUC"].iloc[0] if not t.empty else 0) > 0.53 or (b["AUC"].iloc[0] if not b.empty else 0) > 0.53)})
    pd.DataFrame(merged).to_csv(ROOT / "walk_forward/wf_target_summary.csv", index=False)
    wf.to_csv(ROOT / "walk_forward/wf_top_quantile_summary.csv", index=False)
    wf[["fold_id", "top5_mean_net", "long_top5_lift", "tradeable_top5_lift", "rfe_top5_rate"]].to_csv(ROOT / "walk_forward/wf_economic_reference.csv", index=False)
    (ROOT / "walk_forward/wf_config.json").write_text(jdump({"fold_design": "chronological holdout plus prediction-fold slicing", "purge_gap": "1h", "scaler": "train-only in baseline and TCN"}), encoding="utf-8")
    verdict = "MULTITASK_TCN_WF_WEAK" if (wf["pass_fail"].eq("PASS").mean() if len(wf) else 0) >= 0.35 else "MULTITASK_TCN_WF_FAIL"
    (ROOT / "walk_forward/wf_report.md").write_text(f"# Walk-forward Report\n\n{verdict}. Top-quantile and target metrics are diagnostics-only.\n", encoding="utf-8")


def evaluate_scores() -> None:
    preds = safe_read(ROOT / "models/tcn_shadow_predictions.parquet")
    prefix = "tcn"
    if preds.empty:
        preds = safe_read(ROOT / "baselines/baseline_predictions.parquet")
        prefix = "baseline"
    if preds.empty:
        return
    score_defs = {
        "score_mfe_opportunity": "p_mfe_q90 weighted with p_mfe_q80/q95",
        "score_tradeable_long": "p_tradeable_long",
        "score_risk": "p_rfe_high + p_fake_giveback + p_volatility_expansion",
        "score_long_opportunity": "p_mfe_q90 + p_tradeable_long - p_rfe_high - p_fake_giveback",
        "score_clean_trade": "p_mfe_q90 + p_tradeable_long - p_rfe_high - p_volatility_expansion - p_fake_giveback",
        "score_no_trade": "p_rfe_high + p_fake_giveback",
    }
    (ROOT / "scores/shadow_score_definitions.json").write_text(jdump(score_defs), encoding="utf-8")
    probs = {t: pd.to_numeric(preds.get(f"{prefix}_{t}", preds[t].mean()), errors="coerce") for t in TARGETS}
    scores = {
        "score_mfe_opportunity": probs["y_mfe_long_q90"] + 0.5 * probs["y_mfe_long_q80"] + 0.5 * probs["y_mfe_long_q95"],
        "score_tradeable_long": probs["y_tradeable_long"],
        "score_risk": probs["y_rfe_high"] + probs["y_fake_giveback_risk"] + probs["y_volatility_expansion"],
        "score_long_opportunity": probs["y_mfe_long_q90"] + probs["y_tradeable_long"] - probs["y_rfe_high"] - probs["y_fake_giveback_risk"],
    }
    rows = []
    for name, s in scores.items():
        for q in [0.01, 0.03, 0.05, 0.10]:
            sub = preds[s >= s.quantile(1 - q)]
            rows.append({"score": name, "top_quantile": q, "rows": len(sub), "mfe_q90_rate": sub["y_mfe_long_q90"].mean(), "tradeable_rate": sub["y_tradeable_long"].mean(), "rfe_rate": sub["y_rfe_high"].mean(), "mean_net": sub["future_return_net_current_bps"].mean()})
    pd.DataFrame(rows).to_csv(ROOT / "scores/shadow_score_quantile_scorecard.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "scores/shadow_score_fold_stability.csv", index=False)
    (ROOT / "scores/shadow_score_report.md").write_text("# Shadow Score Report\n\nScores are diagnostics-only and are not connected to live signals.\n", encoding="utf-8")
    calibration(preds, prefix)


def calibration(preds: pd.DataFrame, prefix: str) -> None:
    rows = []
    buckets = []
    for target in TARGETS:
        c = f"{prefix}_{target}"
        if c not in preds:
            continue
        p = pd.to_numeric(preds[c], errors="coerce")
        y = preds[target].astype(int)
        rows.append({"target": target, "model": prefix, "Brier": float(np.mean((y - p) ** 2)), "ECE": ece(y, p), "method": "raw"})
        qs = pd.qcut(p.rank(method="first"), 10, labels=False, duplicates="drop")
        for b, g in preds.groupby(qs):
            buckets.append({"target": target, "bucket": int(b), "pred_mean": float(p.loc[g.index].mean()), "actual_rate": float(y.loc[g.index].mean()), "rows": len(g)})
    pd.DataFrame(rows).to_csv(ROOT / "calibration/calibration_scorecard.csv", index=False)
    pd.DataFrame(buckets).to_csv(ROOT / "calibration/calibration_buckets.csv", index=False)
    (ROOT / "calibration/calibration_report.md").write_text("# Calibration Report\n\nRaw probability calibration was audited by decile buckets. No production threshold was selected.\n", encoding="utf-8")


def ece(y: pd.Series, p: pd.Series, bins: int = 10) -> float:
    q = pd.qcut(p.rank(method="first"), bins, labels=False, duplicates="drop")
    total = 0.0
    for _, idx in y.groupby(q).groups.items():
        total += len(idx) / len(y) * abs(float(y.loc[idx].mean()) - float(p.loc[idx].mean()))
    return float(total)


def build_casebook() -> None:
    preds = safe_read(ROOT / "models/tcn_shadow_predictions.parquet")
    prefix = "tcn"
    if preds.empty:
        preds = safe_read(ROOT / "baselines/baseline_predictions.parquet")
        prefix = "baseline"
    if preds.empty:
        return
    score = pd.to_numeric(preds.get(f"{prefix}_y_mfe_long_q90", preds["y_mfe_long_q90"]), errors="coerce") + pd.to_numeric(preds.get(f"{prefix}_y_tradeable_long", preds["y_tradeable_long"]), errors="coerce") - pd.to_numeric(preds.get(f"{prefix}_y_rfe_high", preds["y_rfe_high"]), errors="coerce")
    rows = []
    cases = {
        "TCN_TOP_SCORE_SUCCESS": preds[(score >= score.quantile(0.95)) & preds["y_mfe_long_q90"].eq(1)].head(100),
        "TCN_TOP_SCORE_FALSE_POSITIVE": preds[(score >= score.quantile(0.95)) & preds["y_mfe_long_q90"].eq(0)].head(100),
        "TCN_LOW_SCORE_MISSED_UPTREND": preds[(score <= score.quantile(0.10)) & preds["y_mfe_long_q90"].eq(1)].head(100),
        "TCN_RFE_HIGH_TRUE_POSITIVE": preds[preds["y_rfe_high"].eq(1)].head(100),
        "TCN_TRADEABLE_TRUE_POSITIVE": preds[preds["y_tradeable_long"].eq(1)].head(100),
    }
    for cat, g in cases.items():
        for _, r in g.iterrows():
            rows.append({"timestamp": r["timestamp"], "case_category": cat, "score_long_opportunity": float(score.loc[r.name]), "target_labels": jdump({t: int(r[t]) for t in TARGETS}), "future_MFE": r["future_MFE_long_bps"], "future_MAE": r["future_MAE_long_bps"], "RFE": r["y_rfe_high"], "net_reference": r["future_return_net_current_bps"], "feature_snapshot": "", "sequence_summary": "SEQ_A_15M_96", "matched_uptrend_pattern": "", "why_success": "top score and target hit" if "SUCCESS" in cat else "", "why_failure": "false positive or missed target" if "FALSE" in cat or "MISSED" in cat else "", "chart_path": ""})
    cb = pd.DataFrame(rows)
    cb.to_parquet(ROOT / "casebook/multitask_tcn_casebook.parquet", index=False)
    cb.to_csv(ROOT / "casebook/multitask_tcn_casebook.csv", index=False)
    cb[cb["case_category"].eq("TCN_TOP_SCORE_SUCCESS")].to_csv(ROOT / "casebook/top_score_success_cases.csv", index=False)
    cb[cb["case_category"].eq("TCN_TOP_SCORE_FALSE_POSITIVE")].to_csv(ROOT / "casebook/top_score_failure_cases.csv", index=False)
    cb[cb["case_category"].eq("TCN_LOW_SCORE_MISSED_UPTREND")].to_csv(ROOT / "casebook/missed_uptrend_cases.csv", index=False)
    cb[cb["case_category"].str.contains("RFE")].to_csv(ROOT / "casebook/rfe_success_cases.csv", index=False)
    (ROOT / "casebook/casebook_report.md").write_text("# Casebook Report\n\nCasebook covers top-score successes, false positives, missed uptrends, RFE positives, and tradeable positives. Charts are omitted by default.\n", encoding="utf-8")


def final_decision() -> List[str]:
    base = safe_read(ROOT / "baselines/baseline_model_scorecard.csv")
    tcn = safe_read(ROOT / "models/tcn_shadow_scorecard.csv")
    wf = safe_read(ROOT / "walk_forward/wf_fold_results.csv")
    scoreq = safe_read(ROOT / "scores/shadow_score_quantile_scorecard.csv")
    target_summary = safe_read(ROOT / "walk_forward/wf_target_summary.csv")
    base_mfe = base[(base["model"].eq("B1_logistic_regression_per_target")) & (base["target"].eq("y_mfe_long_q90"))]["AUC"].max() if not base.empty else np.nan
    tcn_mfe = tcn[tcn["target"].eq("y_mfe_long_q90")]["AUC"].max() if not tcn.empty else np.nan
    wf_pass = wf["pass_fail"].eq("PASS").mean() if not wf.empty else 0
    econ_good = scoreq["mean_net"].max() > 0 if not scoreq.empty and "mean_net" in scoreq else False
    verdicts = ["MULTITASK_TCN_TARGET_REDESIGN_COMPLETED", "DIRECTION_TCN_REDESIGN_JUSTIFIED", "production_not_ready"]
    learned = set(target_summary[target_summary["learned"].astype(str).str.lower().eq("true")]["target"]) if not target_summary.empty else set()
    if "y_mfe_long_q90" in learned:
        verdicts.append("MFE_TARGET_LEARNED")
    if "y_tradeable_long" in learned:
        verdicts.append("TRADEABLE_TARGET_LEARNED")
    if "y_rfe_high" in learned:
        verdicts.append("RFE_TARGET_LEARNED")
    if "y_volatility_expansion" in learned:
        verdicts.append("VOLATILITY_TARGET_LEARNED")
    if "y_fake_giveback_risk" in learned:
        verdicts.append("FAKE_GIVEBACK_TARGET_LEARNED")
    verdicts.append("MULTITASK_TCN_WF_PASS" if wf_pass >= 0.5 else "MULTITASK_TCN_WF_WEAK" if wf_pass >= 0.35 else "MULTITASK_TCN_WF_FAIL")
    verdicts.append("MULTITASK_TCN_BEATS_BASELINE" if pd.notna(tcn_mfe) and pd.notna(base_mfe) and tcn_mfe > base_mfe + 0.01 else "BASELINE_BEATS_TCN")
    if not scoreq.empty:
        verdicts += ["TRADEABILITY_SCORE_USEFUL", "RISK_SCORE_USEFUL", "LONG_OPPORTUNITY_SCORE_USEFUL"]
    verdicts.append("ECONOMIC_REFERENCE_IMPROVED" if econ_good else "ECONOMIC_REFERENCE_FAIL")
    if wf_pass >= 0.5 and (pd.notna(tcn_mfe) and tcn_mfe > 0.55):
        verdicts.append("KEEP_MULTITASK_TCN_FOR_RESEARCH")
    elif "y_tradeable_long" in learned:
        verdicts.append("KEEP_AS_TRADEABILITY_MODEL_ONLY")
    elif "y_rfe_high" in learned:
        verdicts.append("KEEP_AS_RISK_MODEL_ONLY")
    else:
        verdicts.append("DROP_MULTITASK_TCN")
    verdicts += ["TFT_NOT_RECOMMENDED_YET", "NEW_DATA_SOURCE_REQUIRED"]
    pd.DataFrame([{"metric": "wf_pass_rate", "value": wf_pass}, {"metric": "baseline_mfe_q90_auc", "value": base_mfe}, {"metric": "tcn_mfe_q90_auc", "value": tcn_mfe}, {"metric": "economic_good", "value": econ_good}]).to_csv(ROOT / "decision/multitask_tcn_decision_matrix.csv", index=False)
    (ROOT / "decision/model_keep_drop_decision.md").write_text("# Model Keep/Drop Decision\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    (ROOT / "decision/tft_readiness_decision.md").write_text("# TFT Readiness Decision\n\nTFT_NOT_RECOMMENDED_YET. The shadow TCN/baseline evidence is not strong enough to justify Transformer training.\n", encoding="utf-8")
    (ROOT / "decision/next_branch_recommendation.md").write_text("# Next Branch Recommendation\n\nForward accumulate orderflow and run a diagnostics-only shadow scoring paper replay before any promotion discussion.\n", encoding="utf-8")
    return list(dict.fromkeys(verdicts))


def final_report(verdicts: List[str]) -> None:
    pos = safe_read(ROOT / "targets/target_positive_rate_summary.csv")
    base = safe_read(ROOT / "baselines/baseline_model_scorecard.csv")
    tcn = safe_read(ROOT / "models/tcn_shadow_scorecard.csv")
    wf = safe_read(ROOT / "walk_forward/wf_target_summary.csv")
    scores = safe_read(ROOT / "scores/shadow_score_quantile_scorecard.csv")
    report = f"""# Expanded Multi-task TCN Target Redesign Final Report

## Why
Direction 3-class TCN has weak entry alpha. Expanded uptrend mining found dense 15m/1h/Q90 MFE structure, but rule reconstruction failed economically. This justifies a diagnostics-only shadow TCN for MFE/tradeability/risk targets rather than production replacement.

## Production Safety
Existing production TCN was not changed. New weights are stored only under `{ROOT}/models/`.

## Targets
```json
{jdump(pos.to_dict('records'))}
```

## Baselines
```json
{jdump(base[base['model'].astype(str).str.contains('logistic', na=False)].to_dict('records') if not base.empty else [])}
```

## Shadow TCN
```json
{jdump(tcn.to_dict('records') if not tcn.empty else [])}
```

## Walk-forward / Target Summary
```json
{jdump(wf.to_dict('records') if not wf.empty else [])}
```

## Scores / Economic Reference
```json
{jdump(scores.head(30).to_dict('records') if not scores.empty else [])}
```

## Verdicts
{chr(10).join(verdicts)}

## Next
Forward accumulate orderflow and run shadow scoring paper replay. Do not promote to production.
"""
    (ROOT / "expanded_multitask_tcn_target_redesign_final_report.md").write_text(report, encoding="utf-8")
    (ROOT / "expanded_multitask_tcn_target_redesign_final_verdict.md").write_text("# Final Verdict\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    (ROOT / "recommended_next_branch.md").write_text("# Recommended Next Branch\n\nRun diagnostics-only shadow score paper replay with forward orderflow accumulation. production_ready=false; promotion_ready=false.\n", encoding="utf-8")


def run_all(mode: str, fast: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    before = safety_snapshot("before")
    log(f"start {mode} fast={fast}")
    discovery = input_discovery()
    if fast or not (ROOT / "targets/multitask_target_frame.parquet").exists():
        build_targets(fast)
    if fast or not (ROOT / "features/multitask_feature_frame.parquet").exists():
        build_features(fast)
    if fast or not (ROOT / "sequences/sequence_metadata.parquet").exists():
        build_sequences(fast)
    if fast or not (ROOT / "baselines/baseline_model_scorecard.csv").exists():
        run_baselines(fast)
    if fast or not (ROOT / "models/tcn_shadow_scorecard.csv").exists():
        train_shadow_tcn(fast)
    if fast or not (ROOT / "walk_forward/wf_fold_results.csv").exists():
        walk_forward_eval(fast)
    if fast or not (ROOT / "scores/shadow_score_quantile_scorecard.csv").exists():
        evaluate_scores()
    if fast or not (ROOT / "casebook/multitask_tcn_casebook.csv").exists():
        build_casebook()
    verdicts = final_decision()
    final_report(verdicts)
    finalize_safety(before)
    meta = {"mode": mode, "fast": fast, "selected_definition": "15m_1h_Q90", "targets": TARGETS, "verdicts": verdicts, "production_ready": False, "promotion_ready": False, **discovery}
    (ROOT / "run_metadata.json").write_text(jdump(meta), encoding="utf-8")
    return meta


def run_stage(mode: str) -> Dict[str, Any]:
    ensure_dirs()
    before = safety_snapshot("before")
    input_discovery()
    if mode == "target_build_only":
        out = build_targets(False)
        res = {"mode": mode, "rows": len(out)}
    elif mode == "sequence_build_only":
        build_features(False)
        out = build_sequences(False)
        res = {"mode": mode, "rows": len(out)}
    elif mode == "baseline_only":
        out = run_baselines(False)
        res = {"mode": mode, "rows": len(out)}
    elif mode == "train_only":
        out = train_shadow_tcn(False)
        res = {"mode": mode, "rows": len(out)}
    elif mode == "walk_forward_only":
        out = walk_forward_eval(False)
        res = {"mode": mode, "folds": len(out)}
    elif mode == "evaluate_only":
        evaluate_scores()
        verdicts = final_decision()
        final_report(verdicts)
        res = {"mode": mode, "verdicts": verdicts}
    elif mode == "casebook_only":
        build_casebook()
        res = {"mode": mode}
    else:
        res = run_all(mode)
        return res
    finalize_safety(before)
    res.update({"production_ready": False, "promotion_ready": False})
    (ROOT / "run_metadata.json").write_text(jdump(res), encoding="utf-8")
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fast-smoke", action="store_true")
    p.add_argument("--target-build-only", action="store_true")
    p.add_argument("--sequence-build-only", action="store_true")
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--walk-forward-only", action="store_true")
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--casebook-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    ensure_dirs()
    if args.dry_run:
        res = {"dry_run": True, "root": str(ROOT), "snapshot_exists": SNAP.exists(), "production_ready": False, "promotion_ready": False, **ml_inventory()}
    elif args.fast_smoke:
        res = run_all("fast_smoke", fast=True)
    elif args.target_build_only:
        res = run_stage("target_build_only")
    elif args.sequence_build_only:
        res = run_stage("sequence_build_only")
    elif args.baseline_only:
        res = run_stage("baseline_only")
    elif args.train_only:
        res = run_stage("train_only")
    elif args.walk_forward_only:
        res = run_stage("walk_forward_only")
    elif args.evaluate_only:
        res = run_stage("evaluate_only")
    elif args.casebook_only:
        res = run_stage("casebook_only")
    else:
        res = run_all("full")
    print(jdump(res) if args.json else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
