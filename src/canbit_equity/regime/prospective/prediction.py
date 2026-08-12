"""Strict/late prospective prediction generation — frozen model only."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401 — Any used by get_latest_valid_*
from zoneinfo import ZoneInfo

import pandas as pd

from canbit_equity.calendar import get_calendar
from canbit_equity.regime.prospective.config import PROSP_ROOT
from canbit_equity.regime.prospective.model import load_frozen_pipeline
from canbit_equity.strategies import signal_dual_trend

KST = ZoneInfo("Asia/Seoul")


def _normalize(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.normalize()


def existing_prediction_sessions(strict: pd.DataFrame, late: pd.DataFrame) -> set:
    out = set()
    for df in (strict, late):
        if len(df) and "signal_session" in df.columns:
            out |= set(_normalize(df["signal_session"]).tolist())
    return out


def get_latest_valid_prediction_for_session(
    strict: pd.DataFrame,
    session: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Return latest immutable STRICT_PROSPECTIVE ledger row (optionally for one session)."""
    if strict is None or len(strict) == 0 or "signal_session" not in strict.columns:
        return None
    df = strict.copy()
    if "provenance_tier" in df.columns:
        df = df[df["provenance_tier"].astype(str) == "STRICT_PROSPECTIVE"]
    if session is not None:
        sd = pd.Timestamp(pd.Timestamp(session).date())
        df = df[_normalize(df["signal_session"]) == sd]
    if len(df) == 0:
        return None
    sort_cols = [c for c in ("signal_session", "generated_at_utc") if c in df.columns]
    df = df.sort_values(sort_cols)
    row = df.iloc[-1]
    proba = row.get("probability_long")
    thr = row.get("fixed_threshold")
    try:
        proba_f = float(proba)
        thr_f = float(thr)
    except (TypeError, ValueError):
        return None
    if not (proba_f == proba_f and thr_f == thr_f):  # NaN check
        return None
    sig = row.get("target_signal")
    if sig not in ("LONG", "FLAT"):
        try:
            sig = "LONG" if float(row.get("target_position")) == 1.0 else "FLAT"
        except (TypeError, ValueError):
            return None
    if sig not in ("LONG", "FLAT"):
        return None
    dual = row.get("dual_signal")
    if dual not in ("LONG", "FLAT"):
        try:
            dp = row.get("dual_target_position")
            dual = ("LONG" if float(dp) == 1.0 else "FLAT") if dp is not None and str(dp) != "nan" else None
        except (TypeError, ValueError):
            dual = None
    return {
        "signal_session": str(pd.Timestamp(row["signal_session"]).date()),
        "signal": sig,
        "probability": proba_f,
        "threshold": thr_f,
        "dual": dual,
        "provenance": "STRICT_PROSPECTIVE",
        "model_hash": row.get("model_hash"),
        "generated_at_utc": row.get("generated_at_utc"),
        "feature_completeness": row.get("feature_completeness") or "COMPLETE",
        "next_session": (
            str(pd.Timestamp(row["next_session"]).date()) if row.get("next_session") is not None else None
        ),
    }


def get_latest_valid_strict_prediction(strict: pd.DataFrame) -> Optional[Dict[str, Any]]:
    return get_latest_valid_prediction_for_session(strict, session=None)


def classify_provenance(now_utc: datetime, next_open_utc: datetime) -> str:
    if now_utc < next_open_utc:
        return "STRICT_PROSPECTIVE"
    return "LATE_PROVIDER_REPLAY"


def predict_session(
    signal_session: pd.Timestamp,
    live: pd.DataFrame,
    snap: Dict[str, Any],
    now_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    now_utc = now_utc or datetime.now(timezone.utc)
    now_kst = now_utc.astimezone(KST)
    cal = get_calendar("XNYS")
    sd = pd.Timestamp(pd.Timestamp(signal_session).date())

    nxt = cal.sessions_in_range(sd + pd.Timedelta(days=1), sd + pd.Timedelta(days=20))
    if len(nxt) == 0:
        return {"status": "MISSING_UNRECOVERED", "reason": "no_next_session", "signal_session": str(sd.date())}
    next_sess = pd.Timestamp(pd.Timestamp(nxt[0]).date())
    next_open = cal.session_open(nxt[0]).tz_convert("UTC").to_pydatetime()
    if next_open.tzinfo is None:
        next_open = next_open.replace(tzinfo=timezone.utc)
    before_open = now_utc < next_open
    provenance = classify_provenance(now_utc, next_open)

    row = live[live["session_date"] == sd]
    if len(row) == 0:
        return {
            "status": "MISSING_UNRECOVERED",
            "reason": "SOURCE_NOT_AVAILABLE",
            "signal_session": str(sd.date()),
            "provenance_tier": "MISSING_UNRECOVERED",
        }

    features: List[str] = list(snap["exact_feature_order"])
    missing = [c for c in features if c not in row.columns]
    if missing:
        return {
            "status": "MISSING_UNRECOVERED",
            "reason": "FEATURE_CALCULATION_FAILURE",
            "missing_features": missing,
            "signal_session": str(sd.date()),
        }
    if row[features].isna().any(axis=1).iloc[0]:
        return {
            "status": "MISSING_UNRECOVERED",
            "reason": "SOURCE_NOT_AVAILABLE",
            "signal_session": str(sd.date()),
            "feature_completeness": "INCOMPLETE",
        }
    if not np_finite(row[features].iloc[0]):
        return {
            "status": "MISSING_UNRECOVERED",
            "reason": "FEATURE_CALCULATION_FAILURE",
            "signal_session": str(sd.date()),
        }

    pipe, man = load_frozen_pipeline()
    X = row[features].astype(float).values
    proba = float(pipe.predict_proba(X)[0, 1])
    thr = float(snap["fixed_prospective_threshold"])
    target = 1.0 if proba >= thr else 0.0
    dual_pos = float(signal_dual_trend(row).iloc[0]) if "sma_200" in row.columns and "sma_50" in row.columns else None

    rec = {
        "status": "OK",
        "signal_session": sd,
        "generated_at_utc": now_utc.isoformat(),
        "generated_at_kst": now_kst.isoformat(),
        "next_session": next_sess,
        "next_session_open_utc": next_open.isoformat(),
        "provenance_tier": provenance,
        "candidate_id": snap["candidate_id"],
        "model_hash": man.get("model_hash"),
        "preprocessor_hash": man.get("preprocessor_hash"),
        "feature_order_hash": snap["feature_order_hash"],
        "feature_hash": snap.get("feature_hash"),
        "probability_long": proba,
        "fixed_threshold": thr,
        "target_position": target,
        "target_signal": "LONG" if target == 1 else "FLAT",
        "dual_target_position": dual_pos,
        "dual_signal": ("LONG" if dual_pos == 1 else "FLAT") if dual_pos is not None else None,
        "generated_before_next_open": bool(before_open),
        "production_ready": False,
        "promotion_ready": False,
        "execution_enabled": False,
        "order_calls": 0,
        "private_calls": 0,
        "data_quality": "PASS",
        "alignment": "PASS",
        "feature_completeness": "COMPLETE",
    }
    # store exact feature values as JSON-friendly dict
    for i, name in enumerate(features):
        rec[f"feat__{name}"] = float(X[0, i])
    return rec


def np_finite(series: pd.Series) -> bool:
    import numpy as np

    return bool(np.isfinite(series.astype(float).values).all())


def append_prediction_idempotent(
    rec: Dict[str, Any],
    strict: pd.DataFrame,
    late: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Append or no-op; raise on conflict."""
    sd = pd.Timestamp(pd.Timestamp(rec["signal_session"]).date())
    tier = rec["provenance_tier"]
    target = strict if tier == "STRICT_PROSPECTIVE" else late
    path_key = "strict" if tier == "STRICT_PROSPECTIVE" else "late"

    if len(target) and "signal_session" in target.columns:
        mask = _normalize(target["signal_session"]) == sd
        if mask.any():
            old = target.loc[mask].iloc[0]
            same = (
                float(old.get("probability_long", -1)) == float(rec["probability_long"])
                and float(old.get("fixed_threshold", -1)) == float(rec["fixed_threshold"])
                and str(old.get("model_hash")) == str(rec.get("model_hash"))
                and float(old.get("target_position", -1)) == float(rec["target_position"])
            )
            if same:
                return strict, late, "NOOP"
            raise RuntimeError("QQQ_PROSPECTIVE_PREDICTION_CONFLICT")

    row = {k: v for k, v in rec.items() if not k.startswith("feat__") and k != "status"}
    new_df = pd.concat([target, pd.DataFrame([row])], ignore_index=True)
    if path_key == "strict":
        strict = new_df
        strict.to_parquet(PROSP_ROOT / "predictions/strict_predictions.parquet", index=False)
    else:
        late = new_df
        late.to_parquet(PROSP_ROOT / "predictions/late_predictions.parquet", index=False)
    return strict, late, "APPENDED"
