"""
Train Meta Layer V2 — trade quality / risk / scale intelligence (diagnostics only).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.diagnostics.build_meta_label_dataset import (
    ENTRY_FEATURE_COLS_V2,
    LABEL_COLS,
    LEAKAGE_COLS,
    OUT_DIR,
    load_v2_dataset,
)

CORE_FEATURE_COLS = [c for c in ENTRY_FEATURE_COLS_V2 if c not in ("q2_score", "q2_pd_score", "q2_bdi_score")]


def _prepare_x(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    x = df[feature_cols].copy()
    for col in x.columns:
        if x[col].dtype == bool:
            x[col] = x[col].astype(int)
    return x.fillna(x.median(numeric_only=True))


def _time_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2):
    n = len(df)
    i1, i2 = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def _build_classifiers() -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42, n_jobs=-1,
        )
    except ImportError:
        pass
    try:
        import lightgbm as lgb
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, class_weight="balanced", random_state=42, verbose=-1,
        )
    except ImportError:
        pass
    return models


def _calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    err = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        err += abs(y_true[mask].mean() - y_prob[mask].mean()) * mask.mean()
    return float(err)


def _high_conf_failure_rate(df: pd.DataFrame, prob_col: np.ndarray, threshold: float = 0.7) -> float:
    if "calibration_label" not in df.columns:
        return float("nan")
    hi = prob_col >= threshold
    if hi.sum() == 0:
        return 0.0
    return float(df.loc[hi, "calibration_label"].mean())


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "calibration_curve_error": _calibration_error(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _feature_importance(model_name: str, model: Any, feature_cols: List[str]) -> List[Dict[str, Any]]:
    imp: Optional[np.ndarray] = None
    base = model
    if model_name == "logistic_regression" and hasattr(base, "named_steps"):
        imp = np.abs(base.named_steps["clf"].coef_[0])
    elif hasattr(base, "feature_importances_"):
        imp = np.array(base.feature_importances_, dtype=float)
    elif hasattr(base, "coef_") and len(getattr(base, "coef_", [])):
        imp = np.abs(base.coef_[0])
    if imp is None:
        return []
    return [{"feature": f, "importance": float(v)} for f, v in sorted(zip(feature_cols, imp), key=lambda x: -x[1])]


def _rolling_oos(df: pd.DataFrame, model_factory, feature_cols: List[str], label_col: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n = len(df)
    chunk, step = max(30, n // 4), max(8, n // 12)
    start = 0
    while start + chunk + 5 <= n and len(rows) < 20:
        sub = df.iloc[start : start + chunk + step].copy()
        split = int(len(sub) * 0.7)
        if split < 15 or len(sub) - split < 8:
            break
        x = _prepare_x(sub, feature_cols)
        y = sub[label_col].astype(int)
        if y.nunique() < 2:
            start += step
            continue
        m = model_factory()
        m.fit(x.iloc[:split], y.iloc[:split])
        prob = m.predict_proba(x.iloc[split:])[:, 1]
        met = _classification_metrics(y.iloc[split:].to_numpy(), prob)
        rows.append({"window_start": start, "test_rows": len(sub) - split, **met})
        start += step
    return rows


def _interpret_importance(top: List[Dict[str, Any]]) -> Dict[str, Any]:
    names = {x["feature"]: x["importance"] for x in top}
    total = sum(names.values()) or 1.0
    return {
        "entropy_dominant": names.get("entropy", 0) / total >= 0.12,
        "trend_state_important": any(names.get(k, 0) / total >= 0.05 for k in ("trend_up", "trend_down", "trend_sideways")),
        "high_vol_important": names.get("vol_bucket_high", 0) / total >= 0.05,
        "hybrid_a_rediscovered": names.get("hybrid_a_danger_score", names.get("hybrid_a_danger", 0)) / total >= 0.05,
        "q2_pd_rediscovered": names.get("q2_pd_penalty_score", 0) / total >= 0.05,
        "false_high_feature": names.get("false_high_signature_flag", 0) / total >= 0.05,
        "top_features": [x["feature"] for x in top[:10]],
    }


def train_models() -> Tuple[Dict[str, Any], Path, Path]:
    df, ds_path = load_v2_dataset()
    df = df.sort_values(["df_idx", "entry_ts"], na_position="last").reset_index(drop=True)

    train_df, val_df, test_df = _time_split(df)
    feature_sets = {"core": CORE_FEATURE_COLS, "full": ENTRY_FEATURE_COLS_V2}
    classifiers = _build_classifiers()

    results: Dict[str, Any] = {
        "meta_v2_version": "2.0",
        "dataset_path": str(ds_path),
        "dataset_rows": int(len(df)),
        "label_distribution": {
            "calibration_label": df["calibration_label"].value_counts().to_dict(),
            "drawdown_risk_label": df["drawdown_risk_label"].value_counts().to_dict(),
            "trade_quality_label": df["trade_quality_label"].value_counts().to_dict(),
            "scale_target_label": df["scale_target_label"].value_counts().to_dict(),
        },
        "leakage_check": {"passed": not any(c in LEAKAGE_COLS for c in CORE_FEATURE_COLS), "forbidden": list(LEAKAGE_COLS)[:12]},
        "models": {},
    }

    best_key, best_auc, best_bundle = "", -1.0, None

    for fs_name, fcols in feature_sets.items():
        x_tr = _prepare_x(train_df, fcols)
        x_va = _prepare_x(val_df, fcols)
        x_te = _prepare_x(test_df, fcols)
        y_cal_tr = train_df["drawdown_risk_label"].astype(int)
        y_cal_va = val_df["drawdown_risk_label"].astype(int)
        y_cal_te = test_df["drawdown_risk_label"].astype(int)

        for mname, template in classifiers.items():
            key = f"{mname}_{fs_name}"
            base = clone(template)
            base.fit(x_tr, y_cal_tr)

            val_raw = base.predict_proba(x_va)[:, 1]
            test_raw = base.predict_proba(x_te)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(val_raw, y_cal_va)
            platt = LogisticRegression(max_iter=500)
            platt.fit(val_raw.reshape(-1, 1), y_cal_va)

            val_prob = iso.predict(val_raw)
            test_prob = iso.predict(test_raw)
            val_m = _classification_metrics(y_cal_va.to_numpy(), val_prob)
            test_m = _classification_metrics(y_cal_te.to_numpy(), test_prob)
            imp = _feature_importance(mname, base, fcols)

            results["models"][key] = {
                "objective": "drawdown_risk_label",
                "val_metrics": val_m,
                "test_metrics": test_m,
                "calibration_fail_auc": float(roc_auc_score(
                    test_df["calibration_label"].astype(int), test_prob
                )) if test_df["calibration_label"].nunique() > 1 else float("nan"),
                "feature_importance_top10": imp[:10],
                "interpretation": _interpret_importance(imp[:10]),
                "high_confidence_failure_rate": _high_conf_failure_rate(test_df, test_prob),
            }
            if val_m["auc"] > best_auc or (np.isnan(best_auc) and not np.isnan(val_m["auc"])):
                best_auc = val_m["auc"]
                best_key = key
                best_bundle = {
                    "model": base, "isotonic": iso, "platt": platt, "model_name": mname,
                    "feature_cols": fcols, "val_metrics": val_m, "test_metrics": test_m,
                    "feature_importance": imp,
                }

    assert best_bundle is not None

    # Scale regression objective
    scale_reg = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=200, random_state=42)
    scale_reg.fit(_prepare_x(train_df, best_bundle["feature_cols"]), train_df["scale_target_label"])
    scale_pred = scale_reg.predict(_prepare_x(test_df, best_bundle["feature_cols"]))
    scale_mae = float(mean_absolute_error(test_df["scale_target_label"], scale_pred))
    scale_corr = float(np.corrcoef(test_df["scale_target_label"], scale_pred)[0, 1]) if len(test_df) > 2 else 0.0

    # Quality multi-class
    qclf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, random_state=42)
    qclf.fit(_prepare_x(train_df, best_bundle["feature_cols"]), train_df["trade_quality_label"].astype(int))
    q_acc = float(accuracy_score(test_df["trade_quality_label"], qclf.predict(_prepare_x(test_df, best_bundle["feature_cols"]))))

    # Drawdown risk
    dclf = clone(classifiers.get("hist_gradient_boosting", HistGradientBoostingClassifier(max_depth=3, max_iter=100)))
    dclf.fit(_prepare_x(train_df, best_bundle["feature_cols"]), train_df["drawdown_risk_label"].astype(int))
    dd_prob = dclf.predict_proba(_prepare_x(test_df, best_bundle["feature_cols"]))[:, 1]
    dd_auc = float(roc_auc_score(test_df["drawdown_risk_label"], dd_prob)) if test_df["drawdown_risk_label"].nunique() > 1 else float("nan")

    rolling = _rolling_oos(
        df,
        lambda: clone(classifiers[best_bundle["model_name"]]),
        best_bundle["feature_cols"],
        "drawdown_risk_label",
    )
    rolling_aucs = [r["auc"] for r in rolling if not np.isnan(r.get("auc", np.nan))]

    results["objectives"] = {
        "scale_regressor_mae": scale_mae,
        "scale_quality_correlation": scale_corr,
        "trade_quality_accuracy": q_acc,
        "drawdown_risk_auc": dd_auc,
    }
    results["rolling_oos"] = rolling
    results["model_diagnostics"] = {
        "overfit_risk": best_bundle["val_metrics"]["auc"] - best_bundle["test_metrics"]["auc"] > 0.10,
        "val_test_auc_gap": float(best_bundle["val_metrics"]["auc"] - best_bundle["test_metrics"]["auc"]),
        "rolling_oos_auc_mean": float(np.mean(rolling_aucs)) if rolling_aucs else None,
        "rolling_oos_auc_std": float(np.std(rolling_aucs)) if rolling_aucs else None,
        "sample_size_warning": len(df) < 200,
        "reliable_for_monitor_only": len(df) >= 150 and best_bundle["test_metrics"]["auc"] >= 0.58,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = OUT_DIR / f"meta_layer_v2_model_{ts}.joblib"
    report_path = OUT_DIR / f"meta_layer_training_report_{ts}.md"

    artifact = {
        "meta_v2_version": "2.0",
        "best_model_key": best_key,
        "calibration_model": best_bundle["model"],
        "isotonic_calibrator": best_bundle["isotonic"],
        "platt_calibrator": best_bundle.get("platt"),
        "scale_regressor": scale_reg,
        "quality_classifier": qclf,
        "drawdown_classifier": dclf,
        "feature_cols": best_bundle["feature_cols"],
        "test_metrics": best_bundle["test_metrics"],
        "feature_importance_top10": best_bundle["feature_importance"][:10],
        "interpretation": results["models"][best_key]["interpretation"],
        "model_diagnostics": results["model_diagnostics"],
        "objectives": results["objectives"],
        "dataset_path": str(ds_path),
        "trained_at": ts,
    }
    joblib.dump(artifact, artifact_path)

    pointer = {
        "artifact_path": str(artifact_path),
        "report_path": str(report_path),
        "best_model_key": best_key,
        "test_auc": best_bundle["test_metrics"]["auc"],
        "meta_v2_version": "2.0",
        "deployment_verdict": _deployment_verdict(results, best_bundle),
    }
    (OUT_DIR / "meta_layer_model_latest.json").write_text(json.dumps(pointer, indent=2), encoding="utf-8")

    _write_reports(ts, results, best_key, best_bundle, report_path)
    results["artifact_path"] = str(artifact_path)
    results["best_model_key"] = best_key
    results["deployment_verdict"] = pointer["deployment_verdict"]
    return results, artifact_path, report_path


def _deployment_verdict(results: Dict[str, Any], bundle: Dict[str, Any]) -> str:
    diag = results["model_diagnostics"]
    test_auc = bundle["test_metrics"]["auc"]
    if diag["sample_size_warning"] or test_auc < 0.52:
        return "reject"
    if diag.get("reliable_for_monitor_only"):
        return "monitor_only"
    if test_auc >= 0.65 and not diag["overfit_risk"]:
        return "candidate"
    return "reject"


def _write_reports(ts: str, results: Dict[str, Any], best_key: str, bundle: Dict[str, Any], report_path: Path) -> None:
    tm = bundle["test_metrics"]
    lines = [
        "# Meta Layer V2 Training Report",
        "",
        f"- rows: {results['dataset_rows']}",
        f"- best model: **{best_key}**",
        f"- deployment verdict: **{results.get('deployment_verdict', _deployment_verdict(results, bundle))}**",
        "",
        "## New Label Distribution",
        json.dumps(results["label_distribution"], indent=2),
        "",
        "## OOS Calibration Metrics",
        f"- AUC: {tm['auc']:.4f}",
        f"- calibration_curve_error: {tm.get('calibration_curve_error', 0):.4f}",
        f"- brier_score: {tm.get('brier_score', 0):.4f}",
        "",
        "## Multi-Objective Results",
        json.dumps(results["objectives"], indent=2),
        "",
        "## Feature Importance Top 10",
    ]
    for row in bundle["feature_importance"][:10]:
        lines.append(f"- {row['feature']}: {row['importance']:.6f}")
    lines += ["", "## Model Diagnostics"]
    for k, v in results["model_diagnostics"].items():
        lines.append(f"- {k}: {v}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    label_report = OUT_DIR / f"meta_label_distribution_{ts}.md"
    label_report.write_text("# Label Distribution Report\n\n" + json.dumps(results["label_distribution"], indent=2) + "\n", encoding="utf-8")

    fi_report = OUT_DIR / f"meta_feature_importance_{ts}.md"
    fi_report.write_text(
        "# Feature Importance Report\n\n" + "\n".join(f"- {r['feature']}: {r['importance']:.6f}" for r in bundle["feature_importance"][:15]) + "\n",
        encoding="utf-8",
    )

    cal_report = OUT_DIR / f"meta_calibration_analysis_{ts}.md"
    cal_report.write_text(
        "# Calibration Analysis\n\n"
        f"- test_auc: {tm['auc']}\n"
        f"- calibration_curve_error: {tm.get('calibration_curve_error')}\n"
        f"- brier: {tm.get('brier_score')}\n"
        f"- interpretation: {json.dumps(results['models'][best_key]['interpretation'])}\n",
        encoding="utf-8",
    )

    oos_report = OUT_DIR / f"meta_rolling_oos_{ts}.md"
    oos_report.write_text("# Rolling OOS Report\n\n" + json.dumps(results["rolling_oos"], indent=2, default=str) + "\n", encoding="utf-8")

    leak_report = OUT_DIR / f"meta_leakage_diagnosis_{ts}.md"
    leak_report.write_text("# Leakage Diagnosis\n\n" + json.dumps(results["leakage_check"], indent=2) + "\n", encoding="utf-8")


def main() -> None:
    results, artifact_path, report_path = train_models()
    print(f"best_model: {results['best_model_key']}")
    print(f"test_auc: {results['models'][results['best_model_key']]['test_metrics']['auc']:.4f}")
    print(f"deployment_verdict: {results['deployment_verdict']}")
    print(f"artifact: {artifact_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
