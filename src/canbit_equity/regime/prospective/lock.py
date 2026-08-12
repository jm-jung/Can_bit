"""Immutable lock audit + snapshot (never overwrite original lock)."""
from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from canbit_equity.regime.prospective.config import DIAG_ROOT, ORIG_LOCK, PROSP_ROOT, ensure_dirs
from canbit_equity.regime.prospective.file_lock import atomic_write_json


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_thr_dist(raw: Any) -> Dict[str, int]:
    if isinstance(raw, dict):
        return {str(k): int(v) for k, v in raw.items()}
    if isinstance(raw, str):
        try:
            return {str(k): int(v) for k, v in json.loads(raw).items()}
        except Exception:
            return {str(k): int(v) for k, v in ast.literal_eval(raw).items()}
    return {}


def resolve_fixed_threshold(lock: Dict[str, Any]) -> Tuple[float, str, Dict[str, Any]]:
    """Deterministic prospective threshold from locked development fold distribution mode."""
    dist = _parse_thr_dist(lock.get("selected_development_threshold_distribution"))
    if not dist:
        raise RuntimeError("QQQ_PROSPECTIVE_LOCK_THRESHOLD_UNRESOLVED")
    # mode: highest count; tie → lowest threshold among modes (deterministic)
    max_c = max(dist.values())
    modes = sorted([float(k) for k, v in dist.items() if int(v) == max_c])
    thr = modes[0]
    policy = "mode_of_locked_development_fold_distribution"
    meta = {
        "policy": policy,
        "distribution": dist,
        "mode_count": max_c,
        "tied_modes": modes,
        "selected_fixed_threshold": thr,
        "note": "Not an arbitrary pick; derived from immutable lock fold threshold distribution.",
    }
    return thr, policy, meta


def audit_lock() -> Dict[str, Any]:
    ensure_dirs()
    if not ORIG_LOCK.exists():
        out = {"verdict": "QQQ_PROSPECTIVE_LOCK_INVALID", "reason": "missing_lock"}
        atomic_write_json(DIAG_ROOT / "lock_audit/lock_audit.json", out)
        return out

    lock_hash = sha256_file(ORIG_LOCK)
    lock = json.loads(ORIG_LOCK.read_text())
    features: List[str] = list(lock.get("exact_feature_names") or [])
    feature_order_hash = hashlib.sha256("|".join(features).encode()).hexdigest()

    try:
        thr, policy, thr_meta = resolve_fixed_threshold(lock)
        thr_ok = True
    except RuntimeError as exc:
        thr, policy, thr_meta = None, None, {"error": str(exc)}
        thr_ok = False

    checks = {
        "candidate_id": lock.get("candidate_id") == "LOGISTIC_PRICE_PLUS_ALL_REGIME",
        "feature_group": lock.get("feature_group") == "PRICE_PLUS_ALL_REGIME",
        "first_unseen_session": lock.get("first_unseen_session") == "2026-07-31",
        "production_ready_false": lock.get("production_ready") is False,
        "promotion_ready_false": lock.get("promotion_ready") is False,
        "exact_features_present": len(features) > 0,
        "feature_count_28": len(features) == 28,
        "model_config_present": bool(lock.get("model_config")),
        "threshold_candidates_present": bool(lock.get("threshold_candidates")),
        "threshold_resolved": thr_ok,
    }
    if not thr_ok:
        verdict = "QQQ_PROSPECTIVE_LOCK_THRESHOLD_UNRESOLVED"
    elif not all(checks.values()):
        verdict = "QQQ_PROSPECTIVE_LOCK_INVALID"
    else:
        verdict = "QQQ_PROSPECTIVE_LOCK_PASS"

    snapshot = {
        "original_lock_path": str(ORIG_LOCK),
        "original_lock_sha256": lock_hash,
        "lock_version": lock.get("lock_version"),
        "created_at_utc": lock.get("created_at_utc"),
        "candidate_id": lock.get("candidate_id"),
        "feature_group": lock.get("feature_group"),
        "exact_feature_order": features,
        "feature_hash": lock.get("feature_hash"),
        "feature_order_hash": feature_order_hash,
        "external_manifest_hash": lock.get("external_manifest_hash"),
        "config_hash": lock.get("config_hash"),
        "model_config": lock.get("model_config"),
        "threshold_candidates": lock.get("threshold_candidates"),
        "fixed_prospective_threshold": thr,
        "threshold_policy": policy,
        "threshold_resolution": thr_meta,
        "label_horizon": lock.get("label_horizon"),
        "signal_timing": lock.get("signal_timing"),
        "execution_timing": lock.get("execution_timing"),
        "first_unseen_session": lock.get("first_unseen_session"),
        "final_refit_end": lock.get("final_refit_end_session"),
        "minimum_prospective_sessions": lock.get("minimum_prospective_sessions"),
        "target_prospective_sessions": lock.get("target_prospective_sessions"),
        "old_holdout_status": lock.get("old_holdout_status"),
        "production_ready": False,
        "promotion_ready": False,
        "snapshotted_at_utc": datetime.now(timezone.utc).isoformat(),
        "lock_not_overwritten": True,
    }
    snap_path = PROSP_ROOT / "manifests/locked_candidate_snapshot.json"
    atomic_write_json(snap_path, snapshot)

    out = {
        "verdict": verdict,
        "checks": checks,
        "lock_hash": lock_hash,
        "fixed_threshold": thr,
        "threshold_policy": policy,
        "snapshot_path": str(snap_path),
        "candidate_id": lock.get("candidate_id"),
        "feature_count": len(features),
        "feature_order_hash": feature_order_hash,
        "first_unseen_session": lock.get("first_unseen_session"),
        "minimum_sessions": lock.get("minimum_prospective_sessions"),
        "target_sessions": lock.get("target_prospective_sessions"),
    }
    atomic_write_json(DIAG_ROOT / "lock_audit/lock_audit.json", out)
    (DIAG_ROOT / "lock_audit/lock_audit.md").write_text(
        f"# Lock Audit\n\n`{verdict}`\n\nfixed_threshold={thr} policy={policy}\n"
        f"candidate={lock.get('candidate_id')} features={len(features)}\n"
        f"original lock NOT modified. hash={lock_hash}\n"
    )
    return out
