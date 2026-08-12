"""Daily prospective observation update — no retrain, no lock overwrite."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import pandas as pd

from canbit_equity.calendar import get_calendar, latest_completed_session
from canbit_equity.regime.prospective.config import PRIMARY_HORIZON, PROSP_ROOT, ensure_dirs
from canbit_equity.regime.prospective.data_update import load_live_features, update_sources_and_features
from canbit_equity.regime.prospective.file_lock import process_lock
from canbit_equity.regime.prospective.health import current_action, operational_status, readiness_from_counts
from canbit_equity.regime.prospective.lock import audit_lock
from canbit_equity.regime.prospective.metrics import extract_shadow_summary, maybe_write_review_packages, outcome_counts
from canbit_equity.regime.prospective.model import build_or_load_frozen_model
from canbit_equity.regime.prospective.outcomes import mature_outcomes
from canbit_equity.regime.prospective.prediction import (
    append_prediction_idempotent,
    existing_prediction_sessions,
    get_latest_valid_prediction_for_session,
    get_latest_valid_strict_prediction,
    predict_session,
)
from canbit_equity.regime.prospective.reporting import write_reports
from canbit_equity.regime.prospective.shadow import update_shadow_from_predictions

KST = ZoneInfo("Asia/Seoul")


def _empty_parquet(path, columns: List[str]) -> None:
    if not path.exists():
        pd.DataFrame(columns=columns).to_parquet(path, index=False)


def _ensure_ledgers() -> None:
    _empty_parquet(
        PROSP_ROOT / "predictions/strict_predictions.parquet",
        [
            "signal_session",
            "generated_at_utc",
            "provenance_tier",
            "candidate_id",
            "probability_long",
            "fixed_threshold",
            "target_position",
            "dual_target_position",
            "next_session",
            "model_hash",
            "feature_order_hash",
        ],
    )
    _empty_parquet(PROSP_ROOT / "predictions/late_predictions.parquet", ["signal_session", "provenance_tier"])
    _empty_parquet(PROSP_ROOT / "executions/shadow_executions.parquet", ["effective_session", "strategy_id", "net_return"])
    _empty_parquet(PROSP_ROOT / "outcomes/matured_outcomes.parquet", ["signal_session", "horizon"])
    for name in ("candidate_daily_equity", "dual_daily_equity", "buy_hold_daily_equity"):
        _empty_parquet(PROSP_ROOT / f"shadow/{name}.parquet", ["effective_session", "equity", "net_return"])


def run_daily_update(force_discord: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    _ensure_ledgers()
    lock_path = PROSP_ROOT / "locks/process.lock"

    with process_lock(lock_path):
        now_utc = datetime.now(timezone.utc)
        now_kst = datetime.now(KST)

        la = audit_lock()
        if la["verdict"] != "QQQ_PROSPECTIVE_LOCK_PASS":
            out = {
                "operational_status": "CRITICAL",
                "research_readiness": "WAITING_FOR_FIRST_UNSEEN_SESSION",
                "lock_audit": la["verdict"],
                "production_ready": False,
                "promotion_ready": False,
                "execution_enabled": False,
                "private_calls": 0,
                "order_calls": 0,
                "current_action": "FIX_PROSPECTIVE_LOCK",
            }
            write_reports(out, now_kst.strftime("%Y-%m-%d"), la, {"verdict": "SKIPPED"})
            return out

        ma = build_or_load_frozen_model()
        if ma["verdict"] not in ("QQQ_PROSPECTIVE_MODEL_PASS", "QQQ_PROSPECTIVE_MODEL_REBUILT_FROM_LOCK"):
            out = {
                "operational_status": "CRITICAL",
                "research_readiness": "WAITING_FOR_FIRST_UNSEEN_SESSION",
                "model_audit": ma["verdict"],
                "production_ready": False,
                "promotion_ready": False,
                "execution_enabled": False,
                "private_calls": 0,
                "order_calls": 0,
                "current_action": "FIX_FROZEN_MODEL_ARTIFACT",
            }
            write_reports(out, now_kst.strftime("%Y-%m-%d"), la, ma)
            return out

        snap = json.loads((PROSP_ROOT / "manifests/locked_candidate_snapshot.json").read_text())
        first_unseen = pd.Timestamp(snap["first_unseen_session"])
        cal = get_calendar("XNYS")
        latest = latest_completed_session(cal)

        data_info = update_sources_and_features()
        live = load_live_features()

        strict = pd.read_parquet(PROSP_ROOT / "predictions/strict_predictions.parquet")
        late = pd.read_parquet(PROSP_ROOT / "predictions/late_predictions.parquet")
        outcomes = pd.read_parquet(PROSP_ROOT / "outcomes/matured_outcomes.parquet")

        new_prediction = False
        prediction_session = None
        provenance = None
        probability = None
        locked_signal = None
        dual_signal = None
        next_effective = None
        generated_before_next_open = None
        warning = False
        conflict = False
        pending: List[pd.Timestamp] = []
        pending_signal_session = None
        first_unseen_complete = latest >= first_unseen

        if not first_unseen_complete:
            readiness = "WAITING_FOR_FIRST_UNSEEN_SESSION"
            new_session_found = False
        else:
            new_session_found = True
            already = existing_prediction_sessions(strict, late)
            for s in cal.sessions_in_range(first_unseen, latest):
                sd = pd.Timestamp(pd.Timestamp(s).date())
                if sd not in already:
                    pending.append(sd)

            if pending:
                signal_session = pending[0]
                pending_signal_session = str(signal_session.date())
                # never emit pre-lock
                if signal_session < first_unseen:
                    raise RuntimeError("pre-lock session leaked into pending")
                try:
                    rec = predict_session(signal_session, live, snap, now_utc=now_utc)
                except Exception:
                    warning = True
                    rec = {"status": "MISSING_UNRECOVERED", "reason": "FEATURE_CALCULATION_FAILURE"}

                if rec.get("status") == "OK":
                    try:
                        strict, late, action = append_prediction_idempotent(rec, strict, late)
                        if action == "APPENDED":
                            new_prediction = True
                            prediction_session = str(pd.Timestamp(rec["signal_session"]).date())
                            provenance = rec["provenance_tier"]
                            probability = rec["probability_long"]
                            locked_signal = rec["target_signal"]
                            dual_signal = rec.get("dual_signal")
                            next_effective = str(pd.Timestamp(rec["next_session"]).date())
                            generated_before_next_open = rec["generated_before_next_open"]
                    except RuntimeError as exc:
                        if "CONFLICT" in str(exc):
                            conflict = True
                        else:
                            raise
                else:
                    warning = True
                    # Do not null-out an existing valid prediction for another session.
                    # pending_signal_session remains set for refresh/status only.

        shadow_res = update_shadow_from_predictions(live, strict, late)
        try:
            out_res = mature_outcomes(live, strict, late)
        except Exception as exc:
            # Never leave predictions uncommitted in state due to maturity helper faults.
            warning = True
            out_res = {"status": "MATURITY_ERROR", "error": type(exc).__name__, "new": 0}
        outcomes = pd.read_parquet(PROSP_ROOT / "outcomes/matured_outcomes.parquet")
        oc = outcome_counts(outcomes)
        shadow_sum = extract_shadow_summary(shadow_res)

        strict_n = int(len(strict))
        late_n = int(len(late))
        eligible = 0
        if first_unseen_complete:
            eligible = len(list(cal.sessions_in_range(first_unseen, latest)))
        missing = max(eligible - strict_n - late_n, 0)
        if eligible == 0:
            coverage = None
            coverage_reason = "NO_ELIGIBLE_SESSIONS_YET"
        else:
            coverage = float(strict_n) / float(eligible)
            coverage_reason = None

        data_ok = data_info.get("data_quality_verdict") in (
            "QQQ_REGIME_DATA_PASS",
            "QQQ_REGIME_DATA_PASS_WITH_WARNINGS",
            None,
        ) and not data_info.get("qqq_error")
        if data_info.get("data_quality_verdict") == "QQQ_REGIME_DATA_PASS_WITH_WARNINGS":
            warning = True
        revision_this_run = bool(data_info.get("source_revision_detected"))
        if revision_this_run:
            warning = True
        align_ok = int(data_info.get("future_join_count") or 0) == 0
        rev_log = PROSP_ROOT / "cache/source_revisions.jsonl"
        total_rev = 0
        latest_rev_at = None
        if rev_log.exists():
            lines = [ln for ln in rev_log.read_text().splitlines() if ln.strip()]
            total_rev = len(lines)
            if lines:
                try:
                    latest_rev_at = json.loads(lines[-1]).get("detected_at_utc")
                except Exception:
                    latest_rev_at = None

        op = operational_status(
            lock_ok=True,
            model_ok=True,
            data_ok=bool(data_ok),
            align_ok=align_ok,
            conflict=conflict,
            warning=warning,
        )
        cov_for_ready = 0.0 if coverage is None else float(coverage)
        readiness = readiness_from_counts(strict_n, oc["matured_20d"], cov_for_ready, first_unseen_complete)
        if not first_unseen_complete:
            readiness = "WAITING_FOR_FIRST_UNSEEN_SESSION"
            op = "NORMAL" if op != "CRITICAL" else op

        # Ledger is source of truth for prediction display (never null-out on retry).
        latest_valid = get_latest_valid_strict_prediction(strict)
        pending_has_pred = False
        if pending_signal_session:
            pending_has_pred = get_latest_valid_prediction_for_session(strict, pending_signal_session) is not None
        if not new_prediction and latest_valid is not None:
            prediction_session = latest_valid["signal_session"]
            provenance = latest_valid["provenance"]
            probability = latest_valid["probability"]
            locked_signal = latest_valid["signal"]
            dual_signal = latest_valid["dual"]
            next_effective = latest_valid.get("next_session") or next_effective
            generated_before_next_open = True

        if not first_unseen_complete:
            current_refresh_status = "N/A_WAITING"
        elif new_prediction:
            current_refresh_status = "COMPLETE"
        elif pending_signal_session and not pending_has_pred:
            current_refresh_status = "PENDING_OR_MISSING"
        else:
            current_refresh_status = "IDEMPOTENT_REFRESH_NO_NEW_PREDICTION"

        # FEATURES in status/discord for prediction row = COMPLETE when ledger has valid prediction
        feature_completeness = (
            "COMPLETE"
            if latest_valid is not None or new_prediction
            else (
                "N/A_WAITING"
                if not first_unseen_complete
                else current_refresh_status
            )
        )

        prev_state_path = PROSP_ROOT / "state/prospective_state.json"
        prev_matured = 0
        if prev_state_path.exists():
            try:
                prev_matured = int(json.loads(prev_state_path.read_text()).get("matured_20d") or 0)
            except Exception:
                prev_matured = 0
        matured_new = max(int(oc.get("matured_20d") or 0) - prev_matured, 0)

        from canbit_equity.regime.prospective.discord_notify import decide_notification

        preview_state = {
            "new_prediction": new_prediction,
            "prediction_available": latest_valid is not None or new_prediction,
            "locked_signal": locked_signal,
            "probability": probability,
            "dual_signal": dual_signal,
            "prediction_provenance": provenance,
            "prediction_session": prediction_session,
            "latest_valid_session": latest_valid["signal_session"] if latest_valid else None,
            "latest_valid_signal": latest_valid["signal"] if latest_valid else None,
            "latest_valid_probability": latest_valid["probability"] if latest_valid else None,
            "latest_valid_dual": latest_valid["dual"] if latest_valid else None,
            "latest_valid_provenance": latest_valid["provenance"] if latest_valid else None,
            "current_refresh_status": current_refresh_status,
            "feature_completeness": feature_completeness,
            "operational_status": op,
            "pending_signal_session": pending_signal_session,
            "pending_session_has_prediction": pending_has_pred,
            "matured_20d": oc.get("matured_20d"),
            "matured_20d_new": matured_new,
            "latest_completed_qqq_session": str(latest.date()),
            "model_hash": ma.get("model_hash"),
            "threshold": snap["fixed_prospective_threshold"],
        }
        notif_action, _, _ = decide_notification(preview_state, now_kst=now_kst)

        state = {
            "generated_at_utc": now_utc.isoformat(),
            "generated_at_kst": now_kst.isoformat(),
            "operational_status": op,
            "research_readiness": readiness,
            "lock_audit": la["verdict"],
            "lock_hash": la["lock_hash"],
            "locked_candidate": snap["candidate_id"],
            "locked_feature_group": snap["feature_group"],
            "locked_feature_count": len(snap["exact_feature_order"]),
            "locked_feature_order_hash": snap["feature_order_hash"],
            "fixed_threshold": snap["fixed_prospective_threshold"],
            "threshold_policy": snap["threshold_policy"],
            "first_unseen_session": snap["first_unseen_session"],
            "first_strict_observed_session": (
                str(pd.to_datetime(strict["signal_session"]).min().date()) if strict_n else None
            ),
            "minimum_sessions": snap["minimum_prospective_sessions"],
            "target_sessions": snap["target_prospective_sessions"],
            "model_audit": ma["verdict"],
            "model_source": ma.get("model_source"),
            "model_hash": ma.get("model_hash"),
            "preprocessor_hash": ma.get("preprocessor_hash"),
            "model_config_hash": ma.get("model_config_hash"),
            "feature_snapshot_end": ma.get("feature_snapshot_end"),
            "label_matured_fit_end": ma.get("label_matured_fit_end"),
            "actual_model_fit_end": ma.get("actual_model_fit_end"),
            "training_rows": ma.get("training_rows"),
            "training_index_hash": ma.get("training_index_hash"),
            "unlabeled_rows_used_for_fit": ma.get("unlabeled_rows_used_for_fit", 0),
            "model_retrained_daily": False,
            "latest_completed_qqq_session": str(latest.date()),
            "new_session_found": bool(new_session_found),
            "new_prediction": new_prediction,
            "prediction_available": latest_valid is not None or new_prediction,
            "prediction_session": prediction_session,
            "prediction_provenance": provenance,
            "generated_before_next_open": generated_before_next_open,
            "probability": probability,
            "threshold": snap["fixed_prospective_threshold"],
            "locked_signal": locked_signal,
            "dual_signal": dual_signal,
            "next_effective_session": next_effective,
            "latest_valid_session": latest_valid["signal_session"] if latest_valid else None,
            "latest_valid_signal": latest_valid["signal"] if latest_valid else None,
            "latest_valid_probability": latest_valid["probability"] if latest_valid else None,
            "latest_valid_dual": latest_valid["dual"] if latest_valid else None,
            "latest_valid_provenance": latest_valid["provenance"] if latest_valid else None,
            "pending_signal_session": pending_signal_session,
            "pending_session_has_prediction": pending_has_pred,
            "current_refresh_status": current_refresh_status,
            "notification_action": notif_action,
            "data_quality": data_info.get("data_quality_verdict") or ("PASS" if data_ok else "FAIL"),
            "alignment": "PASS" if align_ok else "FAIL",
            "feature_completeness": feature_completeness,
            "future_joins": int(data_info.get("future_join_count") or 0),
            "backward_fills": 0,
            "source_revision": revision_this_run,
            "revision_detected_this_run": revision_this_run,
            "total_revision_events": total_rev,
            "latest_revision_detected_at": latest_rev_at,
            "unresolved_revision_events": 0,
            "strict_sessions": strict_n,
            "late_replays": late_n,
            "missing_sessions": int(missing),
            "strict_coverage": coverage,
            "strict_coverage_reason": coverage_reason,
            "maturity_status": out_res.get("status"),
            "matured_20d_new": matured_new,
            **oc,
            **shadow_sum,
            "discord": "PENDING" if new_prediction or matured_new or op == "CRITICAL" else "NOT_SENT_NO_EVENT",
            "production_ready": False,
            "promotion_ready": False,
            "execution_enabled": False,
            "private_calls": 0,
            "order_calls": 0,
            "live_feature_end": data_info.get("live_feature_end"),
            "historical_feature_hash_preserved": data_info.get("historical_feature_hash_preserved"),
            "current_action": current_action(readiness, op),
            "note": (
                "WAITING: first_unseen_session 2026-07-31 not yet a completed XNYS session; "
                "no pre-lock or incomplete-session predictions created."
                if not first_unseen_complete
                else None
            ),
        }
        maybe_write_review_packages(strict_n, oc["matured_20d"], cov_for_ready, state)
        write_reports(state, now_kst.strftime("%Y-%m-%d"), la, ma)
        from canbit_equity.regime.prospective.config import DIAG_ROOT

        return json.loads((DIAG_ROOT / "reports/qqq_regime_prospective_compact.json").read_text())
