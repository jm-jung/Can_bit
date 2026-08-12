"""One-time frozen model fit from lock — matured labels only; never daily retrain."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from canbit_equity.regime.prospective.config import DIAG_ROOT, PROSP_ROOT, ensure_dirs
from canbit_equity.regime.prospective.file_lock import atomic_write_json
from canbit_equity.regime.prospective.lock import audit_lock, sha256_file


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _training_index_hash(idxs: List[int]) -> str:
    return _sha_bytes(",".join(str(i) for i in idxs).encode())


def build_or_load_frozen_model() -> Dict[str, Any]:
    ensure_dirs()
    lock_audit = audit_lock()
    if lock_audit["verdict"] != "QQQ_PROSPECTIVE_LOCK_PASS":
        return {"verdict": lock_audit["verdict"], "lock_audit": lock_audit}

    snap = json.loads((PROSP_ROOT / "manifests/locked_candidate_snapshot.json").read_text())
    model_path = PROSP_ROOT / "models/qqq_regime_locked_model.joblib"
    prep_path = PROSP_ROOT / "models/qqq_regime_locked_preprocessor.joblib"
    man_path = PROSP_ROOT / "models/qqq_regime_locked_model_manifest.json"

    features: List[str] = list(snap["exact_feature_order"])
    thr = float(snap["fixed_prospective_threshold"])
    cfg = snap.get("model_config") or {}
    seed = int(cfg.get("seed", 42))

    # Load regime feature table (includes price + regime + labels)
    feat_path = PROSP_ROOT.parents[0] / "features/qqq_regime_features.parquet"
    if not feat_path.exists():
        return {"verdict": "QQQ_PROSPECTIVE_MODEL_UNRESOLVED", "reason": "missing_regime_features"}

    df = pd.read_parquet(feat_path).reset_index(drop=True)
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.normalize()
    feature_snapshot_end = str(df["session_date"].max().date())

    # Supervised fit: ONLY rows with matured labels
    for c in features + ["label_long_20d"]:
        if c not in df.columns:
            return {"verdict": "QQQ_PROSPECTIVE_MODEL_UNRESOLVED", "reason": f"missing_col:{c}"}

    labeled = df[df["label_long_20d"].notna()].copy()
    # also require features finite for fit rows
    labeled = labeled.dropna(subset=features).reset_index(drop=True)
    if len(labeled) == 0:
        return {"verdict": "QQQ_PROSPECTIVE_MODEL_UNRESOLVED", "reason": "no_labeled_rows"}

    label_matured_fit_end = str(pd.Timestamp(labeled["session_date"].iloc[-1]).date())
    unlabeled_in_fit = 0  # by construction
    train_idx = labeled.index.tolist()  # after reset — use original positions from df
    # better: global indices from original df
    train_mask = df["label_long_20d"].notna() & df[features].notna().all(axis=1)
    train_df = df.loc[train_mask].copy()
    global_idxs = train_df.index.astype(int).tolist()
    label_matured_fit_end = str(pd.Timestamp(train_df["session_date"].iloc[-1]).date())
    training_rows = int(len(train_df))
    training_index_hash = _training_index_hash(global_idxs)

    if man_path.exists() and model_path.exists() and prep_path.exists():
        man = json.loads(man_path.read_text())
        same = (
            man.get("feature_order_hash") == snap["feature_order_hash"]
            and man.get("fixed_threshold") == thr
            and man.get("training_index_hash") == training_index_hash
            and man.get("candidate_id") == snap["candidate_id"]
        )
        if same:
            out = {
                "verdict": "QQQ_PROSPECTIVE_MODEL_PASS",
                "model_source": "EXISTING_ARTIFACT",
                "model_hash": man.get("model_hash"),
                "preprocessor_hash": man.get("preprocessor_hash"),
                "model_config_hash": man.get("model_config_hash"),
                "feature_snapshot_end": man.get("feature_snapshot_end"),
                "label_matured_fit_end": man.get("label_matured_fit_end"),
                "actual_model_fit_end": man.get("actual_model_fit_end"),
                "training_rows": man.get("training_rows"),
                "training_index_hash": man.get("training_index_hash"),
                "unlabeled_rows_used_for_fit": 0,
                "model_retrained_daily": False,
                "fixed_threshold": thr,
                "model_path": str(model_path),
                "prep_path": str(prep_path),
            }
            atomic_write_json(DIAG_ROOT / "model_audit/model_audit.json", out)
            (DIAG_ROOT / "model_audit/model_audit.md").write_text(
                f"# Model Audit\n\n`{out['verdict']}`\nfit_end={out['actual_model_fit_end']} rows={out['training_rows']}\n"
            )
            return out

    # One-time deterministic refit
    X = train_df[features].astype(float).values
    y = train_df["label_long_20d"].astype(int).values
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=int(cfg.get("max_iter", 2000)),
                    solver=str(cfg.get("solver", "lbfgs")),
                    random_state=seed,
                    class_weight=cfg.get("class_weight", "balanced"),
                ),
            ),
        ]
    )
    pipe.fit(X, y)

    joblib.dump(pipe, model_path)
    # preprocessor = imputer+scaler only for hash clarity
    prep = Pipeline(pipe.steps[:2])
    joblib.dump(prep, prep_path)

    coef = pipe.named_steps["clf"].coef_.ravel()
    intercept = float(pipe.named_steps["clf"].intercept_[0])
    model_hash = sha256_file(model_path)
    prep_hash = sha256_file(prep_path)
    model_config_hash = _sha_bytes(json.dumps(cfg, sort_keys=True).encode())
    coef_hash = _sha_bytes((",".join(f"{c:.10f}" for c in coef) + f"|{intercept:.10f}").encode())

    man = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "QQQ_PROSPECTIVE_MODEL_REBUILT_FROM_LOCK",
        "candidate_id": snap["candidate_id"],
        "exact_feature_order": features,
        "feature_order_hash": snap["feature_order_hash"],
        "feature_hash": snap.get("feature_hash"),
        "fixed_threshold": thr,
        "threshold_policy": snap.get("threshold_policy"),
        "model_config": cfg,
        "model_config_hash": model_config_hash,
        "model_hash": model_hash,
        "preprocessor_hash": prep_hash,
        "coefficient_hash": coef_hash,
        "intercept": intercept,
        "n_coefficients": int(len(coef)),
        "feature_snapshot_end": feature_snapshot_end,
        "label_matured_fit_end": label_matured_fit_end,
        "actual_model_fit_end": label_matured_fit_end,
        "lock_final_refit_end_meaning": "FEATURE_SNAPSHOT_END_NOT_SUPERVISED_FIT_END",
        "training_rows": training_rows,
        "training_index_hash": training_index_hash,
        "unlabeled_rows_used_for_fit": unlabeled_in_fit,
        "model_retrained_daily": False,
        "random_seed": seed,
        "sklearn_note": "Pipeline(SimpleImputer median + StandardScaler + LogisticRegression)",
        "model_path": str(model_path),
        "preprocessor_path": str(prep_path),
        "production_ready": False,
        "promotion_ready": False,
    }
    atomic_write_json(man_path, man)

    out = {
        "verdict": "QQQ_PROSPECTIVE_MODEL_REBUILT_FROM_LOCK",
        "model_source": "ONE_TIME_REFIT_FROM_LOCK",
        "model_hash": model_hash,
        "preprocessor_hash": prep_hash,
        "model_config_hash": model_config_hash,
        "coefficient_hash": coef_hash,
        "feature_snapshot_end": feature_snapshot_end,
        "label_matured_fit_end": label_matured_fit_end,
        "actual_model_fit_end": label_matured_fit_end,
        "training_rows": training_rows,
        "training_index_hash": training_index_hash,
        "unlabeled_rows_used_for_fit": 0,
        "model_retrained_daily": False,
        "fixed_threshold": thr,
        "model_path": str(model_path),
        "prep_path": str(prep_path),
    }
    atomic_write_json(DIAG_ROOT / "model_audit/model_audit.json", out)
    (DIAG_ROOT / "model_audit/model_audit.md").write_text(
        f"# Model Audit\n\n`{out['verdict']}`\n"
        f"feature_snapshot_end={feature_snapshot_end}\n"
        f"label_matured_fit_end={label_matured_fit_end}\n"
        f"training_rows={training_rows}\nunlabeled_in_fit=0\n"
        f"fixed_threshold={thr}\n"
    )
    return out


def load_frozen_pipeline() -> Tuple[Pipeline, Dict[str, Any]]:
    man = json.loads((PROSP_ROOT / "models/qqq_regime_locked_model_manifest.json").read_text())
    pipe = joblib.load(PROSP_ROOT / "models/qqq_regime_locked_model.joblib")
    return pipe, man
