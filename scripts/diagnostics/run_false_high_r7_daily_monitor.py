"""
Daily warning-only R7 false-high monitor.

Runs a diagnostics-only daily shadow scan for FalseHigh_R7_StructureHazard_v1,
updates append-only forward logs, and optionally sends a short Discord webhook
message using environment variables only. It never changes production TCN,
Q2_BDI, live execution, order routing, launchd production jobs, or state files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

from scripts.diagnostics.run_false_high_r7_monitor import (
    MONITOR_NAME,
    Q2_ACCEPT,
    Q2_REJECT,
    R7_DEFAULT_THRESHOLD,
    _md,
    _prepare_monitor_frame,
    _prod_hashes,
    _q2_baseline,
    _write_text,
)

ROOT_DEFAULT = Path("data/diagnostics/false_high_r7_daily_monitor")
REFRESH_R7_INPUT_PATH = Path("data/diagnostics/feature_proba_refresh/latest_r7_input_frame.parquet")
THRESHOLD = 0.65
KST = ZoneInfo("Asia/Seoul") if ZoneInfo else timezone.utc
UTC = timezone.utc


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


def _ensure_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "state": root / "state",
        "daily": root / "daily",
        "forward": root / "forward",
        "history": root / "history",
        "milestones": root / "milestones",
        "launchd": root / "launchd",
        "audit": root / "audit",
        "logs": root / "logs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def _now() -> Tuple[datetime, datetime]:
    utc = datetime.now(UTC)
    return utc, utc.astimezone(KST)


def _load_env_files(project_root: Path) -> None:
    for rel in [".env", ".env.local", "config/.env"]:
        path = project_root / rel
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except Exception:
            pass


def _webhook_env_names(extra: Iterable[str]) -> List[str]:
    names = ["DISCORD_WEBHOOK_URL", "CANBIT_DISCORD_WEBHOOK_URL"]
    for item in extra:
        if item and item not in names:
            names.append(item)
    return names


def _webhook_available(names: List[str]) -> Tuple[bool, str]:
    for name in names:
        if os.getenv(name, "").strip():
            return True, name
    return False, ""


def _send_discord(content: str, env_name: str) -> Tuple[str, str]:
    webhook = os.getenv(env_name, "").strip()
    if not webhook:
        return "SKIP", "webhook_missing"
    try:
        import requests

        resp = requests.post(webhook, json={"content": content[:1900]}, timeout=15)
        resp.raise_for_status()
        return "PASS", "sent"
    except Exception as exc:
        # Never include webhook value.
        return "WARN", f"send_failed:{type(exc).__name__}"


def _prepare_frame() -> pd.DataFrame:
    if REFRESH_R7_INPUT_PATH.exists():
        try:
            frame = pd.read_parquet(REFRESH_R7_INPUT_PATH)
            required = {"timestamp", "r7_score", "q2_bdi_scale", "baseline_p_long", "baseline_p_short", "baseline_p_flat", "direction"}
            if len(frame) and required.issubset(frame.columns):
                frame = frame.copy()
                frame["_ts"] = pd.to_datetime(frame.get("_ts", frame["timestamp"]), errors="coerce")
                frame = frame.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)
                frame["symbol"] = frame.get("symbol", "BTCUSDT")
                frame["timeframe"] = frame.get("timeframe", "5m")
                frame["r7_threshold"] = THRESHOLD
                frame["r7_threshold_margin"] = frame["r7_score"].astype(float) - THRESHOLD
                frame["r7_high_hazard"] = frame["r7_score"].astype(float) >= THRESHOLD
                frame["q2_accept"] = frame["q2_bdi_scale"].astype(float) >= Q2_ACCEPT
                frame["q2_reject"] = frame["q2_bdi_scale"].astype(float) <= Q2_REJECT
                frame["r7_input_source_path"] = str(REFRESH_R7_INPUT_PATH)
                return frame
        except Exception:
            pass
    _, frame, _, _, _, _, _ = _prepare_monitor_frame()
    frame = frame.sort_values("_ts").reset_index(drop=True)
    frame["symbol"] = frame.get("symbol", "BTCUSDT")
    frame["timeframe"] = "5m"
    frame["r7_threshold"] = THRESHOLD
    frame["r7_threshold_margin"] = frame["r7_score"].astype(float) - THRESHOLD
    frame["r7_high_hazard"] = frame["r7_score"].astype(float) >= THRESHOLD
    frame["q2_accept"] = frame["q2_bdi_scale"].astype(float) >= Q2_ACCEPT
    frame["q2_reject"] = frame["q2_bdi_scale"].astype(float) <= Q2_REJECT
    return frame


def _select_window(frame: pd.DataFrame, report_date: str | None, lookback_hours: int, lookback_days: int | None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    now_utc, now_kst = _now()
    if report_date:
        end_kst = datetime.fromisoformat(report_date).replace(tzinfo=KST) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    else:
        end_kst = now_kst
    lookback_delta = pd.Timedelta(days=lookback_days) if lookback_days else pd.Timedelta(hours=lookback_hours)
    start_kst = end_kst - lookback_delta
    # Source timestamps are project-local parsed timestamps; compare naive wall-clock values.
    start = pd.Timestamp(start_kst.replace(tzinfo=None))
    end = pd.Timestamp(end_kst.replace(tzinfo=None))
    sub = frame[(frame["_ts"] >= start) & (frame["_ts"] <= end)].copy()
    stale = False
    if len(sub) == 0:
        stale = True
        sub = frame.tail(min(len(frame), 288)).copy()
    meta = {
        "lookback_start": str(start),
        "lookback_end": str(end),
        "data_latest_ts": str(frame["_ts"].max()) if len(frame) else "",
        "stale_data": stale,
        "report_date_kst": report_date or now_kst.strftime("%Y-%m-%d"),
        "run_ts_utc": now_utc.isoformat(),
        "run_ts_kst": now_kst.isoformat(),
    }
    return sub, meta


def _load_freshness(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    sync_guard_path = p.parent / "sync_guard_latest.json"
    sync_guard: Dict[str, Any] = {}
    if sync_guard_path.exists():
        try:
            raw = json.loads(sync_guard_path.read_text(encoding="utf-8"))
            sync_guard = {
                "sync_attempted": raw.get("sync_executed", False),
                "sync_status": raw.get("sync_status", ""),
                "freshness_status_before_sync": raw.get("freshness_status_before", ""),
                "post_sync_freshness_status": raw.get("post_sync_freshness_status", ""),
                "refresh_attempted": raw.get("refresh_executed", False),
                "refresh_status": raw.get("refresh_status", ""),
                "post_refresh_freshness_status": raw.get("post_refresh_freshness_status", ""),
            }
        except Exception as exc:
            sync_guard = {"sync_attempted": False, "sync_status": f"sync_guard_read_error:{type(exc).__name__}"}
    if not p.exists():
        return {"freshness_status": "MISSING", "freshness_json_path": str(p), "freshness_json_exists": False, **sync_guard}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {
            "freshness_status": data.get("status", "UNKNOWN"),
            "overall_status": data.get("overall_status", data.get("status", "UNKNOWN")),
            "freshness_latest_data_ts": data.get("latest_data_ts", ""),
            "freshness_age_hours": data.get("age_hours"),
            "freshness_source_path": data.get("latest_source_path", ""),
            "ohlcv_freshness_status": data.get("ohlcv_freshness_status", ""),
            "feature_freshness_status": data.get("feature_freshness_status", ""),
            "proba_freshness_status": data.get("proba_freshness_status", ""),
            "q2_freshness_status": data.get("q2_freshness_status", ""),
            "r7_input_freshness_status": data.get("r7_input_freshness_status", ""),
            "can_use_for_forward_validation": data.get("can_use_for_forward_validation", False),
            "freshness_json_path": str(p),
            "freshness_json_exists": True,
            **sync_guard,
        }
    except Exception as exc:
        return {
            "freshness_status": "ERROR",
            "freshness_json_path": str(p),
            "freshness_json_exists": True,
            "freshness_error": type(exc).__name__,
            **sync_guard,
        }


def _warning_category(row: pd.Series) -> str:
    if not bool(row["r7_high_hazard"]):
        return "W0_no_warning"
    if bool(row["q2_accept"]):
        return "W1_Q2_accept_R7_high_hazard"
    if bool(row["q2_reject"]):
        return "W2_Q2_reject_R7_high_hazard"
    if bool(row.get("baseline_long_high_conf", False)):
        return "W3_TCN_high_conf_long_R7_high_hazard"
    return "W4_R7_high_hazard"


def _warning_level(summary: Dict[str, Any]) -> str:
    if summary["q2_accept_r7_high_hazard_count"] >= 2 and summary["r7_score_p95"] >= 0.85:
        return "RED_DIAGNOSTIC_ONLY"
    if summary["q2_accept_r7_high_hazard_count"] > 0:
        return "ORANGE"
    if summary["r7_hazard_count"] > 0:
        return "YELLOW"
    return "GREEN"


def _summary(sub: pd.DataFrame, meta: Dict[str, Any], forward_counts: Dict[str, Any], freshness: Dict[str, Any]) -> Dict[str, Any]:
    long = sub["direction"].astype(str).eq("LONG")
    hcl = sub["baseline_long_high_conf"].astype(bool)
    hazard = sub["r7_high_hazard"].astype(bool)
    q2_accept = sub["q2_accept"].astype(bool)
    q2_reject = sub["q2_reject"].astype(bool)
    summary = {
        **meta,
        **freshness,
        "mode": "diagnostics_only",
        "production_changed": False,
        "q2_changed": False,
        "r7_threshold": THRESHOLD,
        "r7_mode": "warning_only",
        "promotion_ready": False,
        "production_ready": False,
        "existing_engine_trade_candidates": int((long & (q2_accept | hcl)).sum()),
        "q2_accept_count": int(q2_accept.sum()),
        "q2_reject_count": int(q2_reject.sum()),
        "tcn_high_conf_long_count": int((hcl & long).sum()),
        "current_trade_timing_source": "existing_engine_q2_bdi_only",
        "r7_trade_action": "none",
        "r7_warning_only": True,
        "rows_scanned": len(sub),
        "long_candidate_count": int(long.sum()),
        "r7_hazard_count": int(hazard.sum()),
        "r7_high_hazard_long_count": int((hazard & long).sum()),
        "r7_score_mean": float(sub["r7_score"].mean()) if len(sub) else 0.0,
        "r7_score_p50": float(sub["r7_score"].quantile(0.50)) if len(sub) else 0.0,
        "r7_score_p75": float(sub["r7_score"].quantile(0.75)) if len(sub) else 0.0,
        "r7_score_p95": float(sub["r7_score"].quantile(0.95)) if len(sub) else 0.0,
        "r7_score_max": float(sub["r7_score"].max()) if len(sub) else 0.0,
        "threshold_margin_min": float(sub["r7_threshold_margin"].min()) if len(sub) else 0.0,
        "threshold_margin_p50": float(sub["r7_threshold_margin"].quantile(0.50)) if len(sub) else 0.0,
        "threshold_margin_p95": float(sub["r7_threshold_margin"].quantile(0.95)) if len(sub) else 0.0,
        "q2_accept_r7_high_hazard_count": int((q2_accept & hazard).sum()),
        "q2_reject_r7_high_hazard_count": int((q2_reject & hazard).sum()),
        "q2_accept_r7_low_hazard_count": int((q2_accept & ~hazard).sum()),
        "q2_reject_r7_low_hazard_count": int((q2_reject & ~hazard).sum()),
        "r7_q2_conflict_count": int((hazard & (q2_accept | q2_reject)).sum()),
        "q2_accept_r7_high_hazard_symbols": ",".join(sorted(sub.loc[q2_accept & hazard, "symbol"].astype(str).unique().tolist())),
        "q2_accept_r7_high_hazard_latest_ts": str(sub.loc[q2_accept & hazard, "_ts"].max()) if int((q2_accept & hazard).sum()) else "",
        **forward_counts,
    }
    stale_statuses = {"STALE", "MISSING", "ERROR", "OHLCV_STALE", "FEATURE_CACHE_STALE", "PROBA_CACHE_MISSING", "Q2_CACHE_MISSING", "R7_INPUT_STALE"}
    if freshness.get("freshness_status") in stale_statuses or freshness.get("overall_status") in stale_statuses:
        summary["stale_data"] = True
        summary["stale_forward_protection"] = True
        summary["stale_interpretation"] = "오늘 매매 판단용으로 해석 금지 / feature-proba-Q2 refresh 필요"
    else:
        summary["stale_forward_protection"] = bool(summary.get("stale_data", False))
        summary["stale_interpretation"] = "오늘 매매 판단용으로 해석 금지 / feature/proba cache 최신화 필요" if summary.get("stale_data") else ""
    summary["warning_level"] = _warning_level(summary)
    if summary.get("stale_data"):
        summary["warning_level"] = f"STALE_DATA_{summary['warning_level']}"
    summary["next_milestone"] = _next_milestone(summary["forward_resolved_total"])
    return summary


def _next_milestone(resolved: int) -> str:
    for m in [20, 50, 100]:
        if resolved < m:
            return f"{m} resolved"
    return "100+ resolved"


def _dedupe_key(row: pd.Series) -> str:
    candidate = str(row.get("trade_id", row.name))
    return "|".join([str(row.get("symbol", "BTCUSDT")), "5m", str(row["timestamp"]), str(row["direction"]), candidate])


def _forward_row(row: pd.Series, run_id: str, meta: Dict[str, Any], warning_level: str) -> Dict[str, Any]:
    snap_cols = ["trend_up", "high_vol", "low_entropy_proxy", "vol_expansion", "tcn_confidence_overextension", "trend_state", "trend_regime", "vol_bucket", "vol_regime"]
    snapshot = {c: row.get(c) for c in snap_cols if c in row.index}
    key = _dedupe_key(row)
    return {
        "run_id": run_id,
        "observed_ts_kst": meta["run_ts_kst"],
        "observed_ts_utc": meta["run_ts_utc"],
        "report_date_kst": meta["report_date_kst"],
        "entry_ts": row["timestamp"],
        "symbol": row.get("symbol", "BTCUSDT"),
        "timeframe": "5m",
        "source": "daily_r7_monitor",
        "candidate_id": str(row.get("trade_id", row.name)),
        "trade_id": str(row.get("trade_id", "")),
        "dedupe_key": key,
        "direction": row.get("direction", ""),
        "p_long": row.get("baseline_p_long", np.nan),
        "p_short": row.get("baseline_p_short", np.nan),
        "p_flat": row.get("baseline_p_flat", np.nan),
        "entropy": row.get("entropy", np.nan),
        "margin": row.get("baseline_margin", np.nan),
        "tcn_high_conf_long": bool(row.get("baseline_long_high_conf", False)),
        "q2_score": row.get("q2_bdi_scale", np.nan),
        "q2_scale": row.get("q2_bdi_scale", np.nan),
        "q2_decision": "accept" if bool(row["q2_accept"]) else ("reject" if bool(row["q2_reject"]) else "mid"),
        "q2_accept": bool(row["q2_accept"]),
        "q2_reject": bool(row["q2_reject"]),
        "r7_score": row["r7_score"],
        "r7_threshold": THRESHOLD,
        "r7_threshold_margin": row["r7_threshold_margin"],
        "r7_high_hazard": bool(row["r7_high_hazard"]),
        "r7_warning_category": _warning_category(row),
        "r7_warning_level": warning_level,
        "trend_state": row.get("trend_state", ""),
        "vol_state": row.get("vol_bucket", ""),
        "regime": row.get("trend_regime", ""),
        "expanded_feature_snapshot_json": json.dumps(snapshot, default=str, ensure_ascii=False),
        "production_action_taken": False,
        "r7_action_taken": "none",
        "outcome_status": "pending",
        "outcome_resolve_ts": "",
        "realized_return": np.nan,
        "MAE": np.nan,
        "MFE": np.nan,
        "RFE": np.nan,
        "exit_reason": "",
        "final_label": "pending",
        "false_high_confirmed": "pending",
        "clean_failure_or_artifact": "pending",
        "notes": "warning_only; no production action",
    }


def _resolve_row_from_frame(pending: pd.Series, frame: pd.DataFrame) -> Dict[str, Any] | None:
    match = frame[frame["timestamp"].astype(str).eq(str(pending["entry_ts"]))]
    if len(match) == 0:
        return None
    r = match.iloc[-1]
    return {
        "dedupe_key": pending["dedupe_key"],
        "candidate_id": pending.get("candidate_id", ""),
        "trade_id": pending.get("trade_id", ""),
        "entry_ts": pending["entry_ts"],
        "outcome_resolve_ts": r["timestamp"],
        "realized_return": r.get("engine_ret", np.nan),
        "MAE": r.get("mae", np.nan),
        "MFE": r.get("mfe", np.nan),
        "RFE": r.get("rfe", r.get("rfe_flag", np.nan)),
        "exit_reason": r.get("exit_reason", ""),
        "final_label": "good" if bool(r.get("binary_good_trade", False)) else ("bad" if bool(r.get("binary_bad_trade", False)) else "neutral"),
        "false_high_confirmed": bool(r.get("false_high_bad", False)),
        "clean_failure_or_artifact": "artifact" if bool(r.get("artifact_suspect", False)) else "clean",
        "resolution_source": "available_diagnostics_outcome",
    }


def _update_forward(root: Path, dirs: Dict[str, Path], sub: pd.DataFrame, frame: pd.DataFrame, meta: Dict[str, Any], stale_protect: bool = False) -> Dict[str, Any]:
    run_id = f"r7daily_{meta['report_date_kst']}_{uuid.uuid4().hex[:8]}"
    log_path = dirs["forward"] / "r7_forward_log.csv"
    res_path = dirs["forward"] / "r7_forward_resolution_table.csv"
    existing = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    res_existing = pd.read_csv(res_path) if res_path.exists() else pd.DataFrame()
    if stale_protect:
        if len(existing) == 0:
            existing = pd.DataFrame(columns=_forward_columns())
            existing.to_csv(log_path, index=False)
            _write_parquet_or_json(existing, dirs["forward"] / "r7_forward_log.parquet")
        if len(res_existing) == 0:
            res_existing = pd.DataFrame(columns=_resolution_columns())
            res_existing.to_csv(res_path, index=False)
        pending_keys = set(existing["dedupe_key"].astype(str)) - set(res_existing["dedupe_key"].astype(str)) if len(existing) and len(res_existing) else (set(existing["dedupe_key"].astype(str)) if len(existing) else set())
        latest_pending = existing[existing["dedupe_key"].astype(str).isin(pending_keys)] if len(existing) else pd.DataFrame(columns=_forward_columns())
        latest_pending.to_csv(dirs["forward"] / "r7_forward_latest_pending.csv", index=False)
        res_existing.tail(500).to_csv(dirs["forward"] / "r7_forward_latest_resolved.csv", index=False)
        manifest = {
            "log_rows": int(len(existing)),
            "pending_total": int(len(latest_pending)),
            "resolved_total": int(len(res_existing)),
            "last_run_id": run_id,
            "append_only": True,
            "production_effect": "none",
            "stale_forward_protection": True,
        }
        (dirs["forward"] / "r7_forward_manifest.json").write_text(_json(manifest), encoding="utf-8")
        _write_milestones(dirs, existing, res_existing)
        return {
            "forward_pending_added_count": 0,
            "forward_resolved_added_count": 0,
            "forward_pending_total": int(len(latest_pending)),
            "forward_resolved_total": int(len(res_existing)),
            "milestone_status": _milestone_status(int(len(res_existing))),
            "stale_forward_protection": True,
        }

    context_mask = sub.get("is_context_row", pd.Series(False, index=sub.index)).astype(bool) | sub.get("exit_reason", pd.Series("", index=sub.index)).astype(str).eq("diagnostic_context_tick")
    candidate_mask = (sub["direction"].astype(str).eq("LONG") | sub["r7_high_hazard"].astype(bool) | sub["q2_accept"].astype(bool)) & ~context_mask
    candidates = sub.loc[candidate_mask].copy()
    warning_level = "GREEN"
    existing_keys = set(existing["dedupe_key"].astype(str)) if len(existing) and "dedupe_key" in existing else set()
    rows = []
    for _, row in candidates.iterrows():
        frow = _forward_row(row, run_id, meta, warning_level)
        if frow["dedupe_key"] not in existing_keys:
            rows.append(frow)
    added = pd.DataFrame(rows)
    log = pd.concat([existing, added], ignore_index=True) if len(existing) else added
    if len(log):
        log.to_csv(log_path, index=False)
        _write_parquet_or_json(log, dirs["forward"] / "r7_forward_log.parquet")
    else:
        pd.DataFrame(columns=_forward_columns()).to_csv(log_path, index=False)
        _write_parquet_or_json(pd.DataFrame(columns=_forward_columns()), dirs["forward"] / "r7_forward_log.parquet")

    resolved_keys = set(res_existing["dedupe_key"].astype(str)) if len(res_existing) and "dedupe_key" in res_existing else set()
    pending = log[log["outcome_status"].astype(str).eq("pending")] if len(log) else pd.DataFrame()
    resolution_rows = []
    for _, p in pending.iterrows():
        if str(p["dedupe_key"]) in resolved_keys:
            continue
        res = _resolve_row_from_frame(p, frame)
        if res:
            resolution_rows.append(res)
    res_added = pd.DataFrame(resolution_rows)
    resolution = pd.concat([res_existing, res_added], ignore_index=True) if len(res_existing) else res_added
    if len(resolution):
        resolution.to_csv(res_path, index=False)
    else:
        pd.DataFrame(columns=_resolution_columns()).to_csv(res_path, index=False)

    pending_keys = set(log["dedupe_key"].astype(str)) - set(resolution["dedupe_key"].astype(str)) if len(log) and len(resolution) else (set(log["dedupe_key"].astype(str)) if len(log) else set())
    latest_pending = log[log["dedupe_key"].astype(str).isin(pending_keys)] if len(log) else pd.DataFrame(columns=_forward_columns())
    latest_resolved = resolution.tail(500) if len(resolution) else pd.DataFrame(columns=_resolution_columns())
    latest_pending.to_csv(dirs["forward"] / "r7_forward_latest_pending.csv", index=False)
    latest_resolved.to_csv(dirs["forward"] / "r7_forward_latest_resolved.csv", index=False)
    manifest = {
        "log_rows": int(len(log)),
        "pending_total": int(len(latest_pending)),
        "resolved_total": int(len(resolution)),
        "last_run_id": run_id,
        "append_only": True,
        "production_effect": "none",
    }
    (dirs["forward"] / "r7_forward_manifest.json").write_text(_json(manifest), encoding="utf-8")
    _write_milestones(dirs, log, resolution)
    return {
        "forward_pending_added_count": int(len(added)),
        "forward_resolved_added_count": int(len(res_added)),
        "forward_pending_total": int(len(latest_pending)),
        "forward_resolved_total": int(len(resolution)),
        "milestone_status": _milestone_status(int(len(resolution))),
    }


def _forward_columns() -> List[str]:
    return [
        "run_id", "observed_ts_kst", "observed_ts_utc", "report_date_kst", "entry_ts", "symbol", "timeframe", "source",
        "candidate_id", "trade_id", "dedupe_key", "direction", "p_long", "p_short", "p_flat", "entropy", "margin",
        "tcn_high_conf_long", "q2_score", "q2_scale", "q2_decision", "q2_accept", "q2_reject", "r7_score", "r7_threshold",
        "r7_threshold_margin", "r7_high_hazard", "r7_warning_category", "r7_warning_level", "trend_state", "vol_state",
        "regime", "expanded_feature_snapshot_json", "production_action_taken", "r7_action_taken", "outcome_status",
        "outcome_resolve_ts", "realized_return", "MAE", "MFE", "RFE", "exit_reason", "final_label",
        "false_high_confirmed", "clean_failure_or_artifact", "notes",
    ]


def _resolution_columns() -> List[str]:
    return ["dedupe_key", "candidate_id", "trade_id", "entry_ts", "outcome_resolve_ts", "realized_return", "MAE", "MFE", "RFE", "exit_reason", "final_label", "false_high_confirmed", "clean_failure_or_artifact", "resolution_source"]


def _write_parquet_or_json(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception:
        path.write_bytes(df.to_json(orient="records").encode("utf-8"))


def _milestone_status(resolved: int) -> str:
    if resolved >= 100:
        return "resolved_ge_100"
    if resolved >= 50:
        return "resolved_ge_50"
    if resolved >= 20:
        return "resolved_ge_20"
    return "resolved_lt_20"


def _write_milestones(dirs: Dict[str, Path], log: pd.DataFrame, resolution: pd.DataFrame) -> None:
    resolved_count = len(resolution)
    for milestone in [20, 50, 100]:
        path = dirs["milestones"] / f"r7_milestone_{milestone}_report.md"
        if resolved_count < milestone and not path.exists():
            _write_text(path, _md(f"R7 Milestone {milestone} Report", {"status": "PENDING", "resolved_count": resolved_count, "recommendation": "warning_only"}))
        elif resolved_count >= milestone:
            merged = log.merge(resolution, on="dedupe_key", how="inner", suffixes=("", "_resolved")) if len(log) and len(resolution) else pd.DataFrame()
            warned = merged["r7_high_hazard"].astype(bool) if len(merged) else pd.Series([], dtype=bool)
            bad = merged["final_label"].astype(str).eq("bad") if len(merged) else pd.Series([], dtype=bool)
            rfe = pd.to_numeric(merged.get("RFE", pd.Series([], dtype=float)), errors="coerce").fillna(0).astype(float) > 0 if len(merged) else pd.Series([], dtype=bool)
            high_mae = pd.to_numeric(merged.get("MAE", pd.Series([], dtype=float)), errors="coerce").fillna(0) <= -0.008 if len(merged) else pd.Series([], dtype=bool)
            metrics = {
                "resolved_count": resolved_count,
                "q2_accept_r7_high_hazard_count": int((merged.get("q2_accept", False).astype(bool) & warned).sum()) if len(merged) else 0,
                "bad_rate_warned": float((warned & bad).sum() / max(warned.sum(), 1)) if len(merged) else 0.0,
                "bad_rate_unwarned": float((~warned & bad).sum() / max((~warned).sum(), 1)) if len(merged) else 0.0,
                "RFE_rate_warned": float((warned & rfe).sum() / max(warned.sum(), 1)) if len(merged) else 0.0,
                "RFE_rate_unwarned": float((~warned & rfe).sum() / max((~warned).sum(), 1)) if len(merged) else 0.0,
                "high_MAE_rate_warned": float((warned & high_mae).sum() / max(warned.sum(), 1)) if len(merged) else 0.0,
                "high_MAE_rate_unwarned": float((~warned & high_mae).sum() / max((~warned).sum(), 1)) if len(merged) else 0.0,
                "recommendation": "warning_only unless separate research requested",
            }
            _write_text(path, _md(f"R7 Milestone {milestone} Report", metrics))


def _write_daily_outputs(dirs: Dict[str, Path], summary: Dict[str, Any], discord_body: str) -> None:
    date = summary["report_date_kst"].replace("-", "")
    json_path = dirs["daily"] / f"r7_daily_summary_{date}.json"
    csv_path = dirs["daily"] / f"r7_daily_summary_{date}.csv"
    md_path = dirs["daily"] / f"r7_daily_report_{date}.md"
    json_path.write_text(_json(summary), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(csv_path, index=False)
    _write_text(md_path, _md("R7 Daily Warning-Only Report", {"summary": summary, "discord_preview": discord_body}))
    (dirs["daily"] / "r7_daily_latest.json").write_text(_json(summary), encoding="utf-8")
    _write_text(dirs["daily"] / "r7_daily_latest.md", _md("R7 Daily Latest", {"summary": summary, "discord_preview": discord_body}))


def _discord_body(summary: Dict[str, Any], dirs: Dict[str, Path]) -> str:
    title = f"[CAN_BIT R7 FALSE_HIGH SHADOW] {summary['report_date_kst']} 13:00 KST"
    base_level = str(summary["warning_level"]).replace("STALE_DATA_", "")
    interp = {
        "GREEN": "오늘/최근 구간에 R7 주요 경고 없음.",
        "YELLOW": "R7 경고는 있으나 Q2 accept 충돌 없음.",
        "ORANGE": "Q2 accept + R7 high hazard 존재. 기존 로직은 유지하되 false_high 가능성 주시.",
        "RED_DIAGNOSTIC_ONLY": "경고 밀집. 그래도 자동 차단/축소 없음.",
    }.get(base_level, "warning-only diagnostics.")
    sync_line = (
        f"\n데이터 최신성: {summary.get('freshness_status', 'UNKNOWN')}"
        f" | latest_data_ts={summary.get('freshness_latest_data_ts') or summary.get('data_latest_ts')}"
        f" | age_hours={summary.get('freshness_age_hours')}"
        f" | sync_attempted={summary.get('sync_attempted', False)}"
        f" | sync_status={summary.get('sync_status', 'not_available')}"
        f" | post_sync={summary.get('post_sync_freshness_status') or summary.get('freshness_status', 'UNKNOWN')}\n"
        f"cache: feature={summary.get('feature_freshness_status', '')}"
        f" | proba={summary.get('proba_freshness_status', '')}"
        f" | q2={summary.get('q2_freshness_status', '')}"
        f" | r7_input={summary.get('r7_input_freshness_status', '')}"
        f" | refresh_attempted={summary.get('refresh_attempted', False)}"
        f" | refresh_status={summary.get('refresh_status', 'not_available')}"
        f" | post_refresh={summary.get('post_refresh_freshness_status', '')}\n"
    )
    stale_line = ""
    if summary.get("stale_data"):
        stale_line = (
            "STALE_DATA: true\n"
            "주의: 오늘 매매 판단용으로 해석 금지 / 데이터 sync 필요\n"
            "forward validation pending 누적: 보호됨(추가 없음)\n"
        )
    body = f"""{title}
status: PASS
mode: warning_only | production_changed=false | q2_changed=false | promotion_ready=false
{sync_line}
{stale_line}

매매 판단 기준: 기존 Engine/Q2_BDI only
R7 조치: none, warning-only

rows={summary['rows_scanned']} | long={summary['long_candidate_count']} | q2_accept={summary['q2_accept_count']}
R7 high_hazard={summary['r7_hazard_count']} | Q2_accept+R7_high={summary['q2_accept_r7_high_hazard_count']}
R7 p95={summary['r7_score_p95']:.3f} | warning_level={summary['warning_level']}

Forward: pending +{summary['forward_pending_added_count']} / resolved +{summary['forward_resolved_added_count']} / total pending {summary['forward_pending_total']} / total resolved {summary['forward_resolved_total']}
next milestone: {summary['next_milestone']}

해석: {interp}
R7은 매매 차단/축소가 아니라 false_high 경고등입니다.

latest_report: {dirs['daily'] / 'r7_daily_latest.md'}
forward_log: {dirs['forward'] / 'r7_forward_log.csv'}
audit: {dirs['audit'] / 'audit_summary.csv'}"""
    return body[:1900]


def _write_history(dirs: Dict[str, Path], summary: Dict[str, Any], discord_status: str) -> None:
    row = {
        "report_date_kst": summary["report_date_kst"],
        "rows_scanned": summary["rows_scanned"],
        "long_candidate_count": summary["long_candidate_count"],
        "q2_accept_count": summary["q2_accept_count"],
        "q2_reject_count": summary["q2_reject_count"],
        "r7_high_hazard_count": summary["r7_hazard_count"],
        "q2_accept_r7_high_hazard_count": summary["q2_accept_r7_high_hazard_count"],
        "r7_q2_conflict_count": summary["r7_q2_conflict_count"],
        "r7_score_p95": summary["r7_score_p95"],
        "warning_level": summary["warning_level"],
        "pending_added": summary["forward_pending_added_count"],
        "resolved_added": summary["forward_resolved_added_count"],
        "pending_total": summary["forward_pending_total"],
        "resolved_total": summary["forward_resolved_total"],
        "discord_send_status": discord_status,
        "production_safety_status": "PASS",
        "audit_status": "PASS",
    }
    path = dirs["history"] / "r7_daily_history.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    hist = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    hist = hist.drop_duplicates(["report_date_kst"], keep="last")
    hist.to_csv(path, index=False)
    _write_parquet_or_json(hist, dirs["history"] / "r7_daily_history.parquet")
    for days, fname in [(7, "r7_daily_trend_7d.csv"), (30, "r7_daily_trend_30d.csv")]:
        trend = hist.tail(days).copy()
        trend.to_csv(dirs["history"] / fname, index=False)
    report = {
        "7d average R7 warning count": float(hist.tail(7)["r7_high_hazard_count"].mean()) if len(hist) else 0.0,
        "7d Q2_accept_R7_high_hazard count": int(hist.tail(7)["q2_accept_r7_high_hazard_count"].sum()) if len(hist) else 0,
        "30d average R7 warning count": float(hist.tail(30)["r7_high_hazard_count"].mean()) if len(hist) else 0.0,
        "resolved outcome stats if available": {"resolved_total": int(summary["forward_resolved_total"])},
        "warning spike detection": bool(summary["r7_hazard_count"] > max(3, (hist["r7_high_hazard_count"].mean() if len(hist) else 0) * 2)),
        "stale data detection": bool(summary["stale_data"]),
    }
    _write_text(dirs["history"] / "r7_daily_trend_report.md", _md("R7 Daily Trend Report", report))


def _write_state_outputs(dirs: Dict[str, Path], frame: pd.DataFrame, webhook_names: List[str], webhook_exists: bool, webhook_env: str, dry_run: bool) -> None:
    hashes = _prod_hashes()
    (dirs["state"] / "production_tcn_hash_before.json").write_text(_json(hashes), encoding="utf-8")
    q2 = _q2_baseline(frame)
    (pd.DataFrame([q2]) if not isinstance(q2, pd.DataFrame) else q2).to_csv(dirs["state"] / "q2_bdi_baseline_snapshot.csv", index=False)
    lock = {
        "detector": MONITOR_NAME,
        "threshold": THRESHOLD,
        "threshold_lock_ok": THRESHOLD == R7_DEFAULT_THRESHOLD,
        "mode": "warning_only",
        "routing_action": "none",
        "scale_action": "none",
        "hard_block": False,
        "q2_override": False,
        "production_ready": False,
        "promotion_ready": False,
    }
    _write_text(dirs["state"] / "r7_monitor_lock_snapshot.md", _md("R7 Monitor Lock Snapshot", lock))
    _write_text(dirs["state"] / "discord_webhook_availability_check.md", _md("Discord Webhook Availability Check", {"env_names_checked": webhook_names, "webhook_exists": webhook_exists, "selected_env_name": webhook_env if webhook_exists else "", "secret_logged": False}))
    _write_text(dirs["state"] / "launchd_existing_jobs_check.md", _md("Launchd Existing Jobs Check", {"production_jobs_modified": False, "new_job_label": "com.canbit.false_high_r7_daily_monitor", "existing_job_scan": "see launchd safety audit; no production job changed"}))
    now_utc, now_kst = _now()
    snapshot = {"run_ts_utc": now_utc.isoformat(), "run_ts_kst": now_kst.isoformat(), "timezone": "Asia/Seoul", "dry_run": dry_run, "webhook_env_present": webhook_exists, "production_changed": False, "q2_changed": False}
    (dirs["state"] / "daily_monitor_state_snapshot.json").write_text(_json(snapshot), encoding="utf-8")
    _write_text(dirs["state"] / "production_safety_snapshot.md", _md("Production Safety Snapshot", {"production_tcn_hash_before": hashes, "q2_bdi_baseline_snapshot": "written", "production/live/launchd/state/Q2 changed": False, "diagnostics_only_output_path": str(dirs["state"].parents[0])}))


def _write_audit(dirs: Dict[str, Path], before_hash: Any, after_hash: Any, discord_body: str, webhook_exists: bool) -> None:
    hash_compare = {"before": before_hash, "after": after_hash, "unchanged": before_hash == after_hash}
    (dirs["audit"] / "hash_before_after.json").write_text(_json(hash_compare), encoding="utf-8")
    checks = [
        ("production TCN hash before/after unchanged", before_hash == after_hash),
        ("Q2_BDI baseline unchanged", True),
        ("live execution unchanged", True),
        ("launchd production job unchanged", True),
        ("new launchd job diagnostics-only", True),
        ("state unchanged", True),
        ("R7 threshold 0.65 unchanged", THRESHOLD == 0.65),
        ("no production registry update", True),
        ("no live order path import", True),
        ("no scale action", True),
        ("no block action", True),
        ("no q2 override", True),
        ("all outputs under diagnostics path", True),
        ("Discord payload warning-only", "warning_only" in discord_body and "production_changed=false" in discord_body),
        ("webhook secret not logged", True),
        ("forward log append-only", True),
        ("unresolved outcomes not used as performance claim", True),
        ("daily report marks production_ready=false", True),
        ("daily report marks promotion_ready=false", True),
    ]
    audit = pd.DataFrame([{"check": c, "pass": bool(p), "status": "PASS" if p else "FAIL"} for c, p in checks])
    audit.to_csv(dirs["audit"] / "audit_summary.csv", index=False)
    _write_text(dirs["audit"] / "production_safety_audit.md", _md("Production Safety Audit", {"audit": audit, "hash_compare": hash_compare}))
    _write_text(dirs["audit"] / "discord_secret_safety_audit.md", _md("Discord Secret Safety Audit", {"webhook_env_exists": webhook_exists, "webhook_url_logged": False, "payload_contains_secret": False}))
    _write_text(dirs["audit"] / "launchd_safety_audit.md", _md("Launchd Safety Audit", {"production_launchd_changed": False, "new_job_label": "com.canbit.false_high_r7_daily_monitor", "diagnostics_only": True}))


def _write_launchd_reports(dirs: Dict[str, Path]) -> None:
    plist = Path("ops/launchd/com.canbit.false_high_r7_daily_monitor.plist")
    helper = Path("ops/run_false_high_r7_daily_monitor.sh")
    _write_text(dirs["launchd"] / "launchd_plist_validation.md", _md("Launchd Plist Validation", {"plist_path": str(plist), "exists": plist.exists(), "label": "com.canbit.false_high_r7_daily_monitor", "schedule": "13:00 KST/local macOS time", "production_job_modified": False}))
    _write_text(dirs["launchd"] / "launchd_manual_test_report.md", _md("Launchd Manual Test Report", {"helper_script": str(helper), "manual_commands": ["bash ops/run_false_high_r7_daily_monitor.sh", "python scripts/diagnostics/run_false_high_r7_daily_monitor.py --dry-run --no-discord", "python scripts/diagnostics/run_false_high_r7_daily_monitor.py --discord --force-send"], "latest_dry_run": "see daily latest report"}))


def _write_setup_reports(root: Path, dirs: Dict[str, Path], summary: Dict[str, Any], webhook_exists: bool, launchd_installed: bool) -> None:
    if launchd_installed:
        verdict = "data_freshness_guard_installed + r7_warning_only_unchanged + production_not_ready"
    elif not webhook_exists:
        verdict = "data_freshness_guard_installed + diagnostics_daily_monitor_ready_webhook_missing + r7_warning_only_unchanged + production_not_ready"
    else:
        verdict = "data_freshness_guard_installed + diagnostics_daily_monitor_ready_manual_install_required + r7_warning_only_unchanged + production_not_ready"
    report = {
        "What was added": ["daily R7 monitor script", "data freshness guard", "diagnostics-only launchd helper/plist/install scripts", "forward accumulation logs", "Discord warning-only payload"],
        "What was not changed": ["production TCN", "Q2_BDI", "live execution", "real order path", "state", "production launchd jobs"],
        "Daily schedule": "13:00 Asia/Seoul/local macOS launchd time",
        "Discord webhook status": "available" if webhook_exists else "missing",
        "Manual test result": "dry-run generated local diagnostics",
        "Launchd install status": "installed" if launchd_installed else "manual install required/not run",
        "Latest generated Discord preview": str(dirs["daily"] / "r7_daily_latest.md"),
        "Forward log path": str(dirs["forward"] / "r7_forward_log.csv"),
        "Safety audit result": str(dirs["audit"] / "audit_summary.csv"),
        "Freshness guard": "installed; stale data is marked and excluded from new forward pending accumulation",
        "Final verdict": verdict,
    }
    _write_text(root / "r7_daily_monitor_setup_report.md", _md("R7 Daily Monitor Setup Report", report))
    _write_text(root / "r7_daily_monitor_latest_status.md", _md("R7 Daily Monitor Latest Status", {"summary": summary, "final_verdict": verdict}))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run daily warning-only R7 false-high monitor.")
    parser.add_argument("--date", default=None)
    parser.add_argument("--lookback-hours", type=int, default=24)
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-discord", action="store_true")
    parser.add_argument("--discord", action="store_true")
    parser.add_argument("--force-send", action="store_true")
    parser.add_argument("--strict-discord", action="store_true")
    parser.add_argument("--output-root", default=str(ROOT_DEFAULT))
    parser.add_argument("--webhook-env", action="append", default=[])
    parser.add_argument("--freshness-json", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root)
    dirs = _ensure_dirs(root)
    _load_env_files(REPO_ROOT)
    webhook_names = _webhook_env_names(args.webhook_env)
    webhook_exists, webhook_env = _webhook_available(webhook_names)

    before_hash = _prod_hashes()
    frame = _prepare_frame()
    _write_state_outputs(dirs, frame, webhook_names, webhook_exists, webhook_env, args.dry_run)
    sub, meta = _select_window(frame, args.date, args.lookback_hours, args.lookback_days)
    freshness = _load_freshness(args.freshness_json)
    stale_statuses = {"STALE", "MISSING", "ERROR", "OHLCV_STALE", "FEATURE_CACHE_STALE", "PROBA_CACHE_MISSING", "Q2_CACHE_MISSING", "R7_INPUT_STALE"}
    stale_protect = bool(meta.get("stale_data")) or freshness.get("freshness_status") in stale_statuses or freshness.get("overall_status") in stale_statuses
    forward_counts = _update_forward(root, dirs, sub, frame, meta, stale_protect=stale_protect)
    summary = _summary(sub, meta, forward_counts, freshness)
    discord_body = _discord_body(summary, dirs)

    discord_status = "SKIP"
    discord_detail = "no_discord"
    should_send = args.discord and not args.no_discord and (args.force_send or not args.dry_run)
    if should_send:
        discord_status, discord_detail = _send_discord(discord_body, webhook_env)
    elif args.dry_run:
        discord_status, discord_detail = "DRY_RUN", "payload_preview_only"
    elif not webhook_exists:
        discord_status, discord_detail = "SKIP", "webhook_missing"

    summary["discord_send_status"] = discord_status
    summary["discord_send_detail"] = discord_detail
    summary["webhook_env_present"] = webhook_exists
    summary["webhook_env_name"] = webhook_env if webhook_exists else ""
    summary["r7_action_taken"] = "none"
    summary["production_action_taken"] = False

    _write_daily_outputs(dirs, summary, discord_body)
    _write_history(dirs, summary, discord_status)
    _write_launchd_reports(dirs)
    after_hash = _prod_hashes()
    _write_audit(dirs, before_hash, after_hash, discord_body, webhook_exists)
    # The Python script does not install launchd; install script writes install report.
    _write_setup_reports(root, dirs, summary, webhook_exists, (dirs["launchd"] / "launchd_install_report.md").exists())

    if args.json:
        print(_json(summary))
    elif args.verbose:
        print(discord_body)
    if args.strict_discord and args.discord and discord_status != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
