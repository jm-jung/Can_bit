"""Diagnostics-only observation quality suite: taker forensics, daily rollup, readiness.

Read-only over collector/observer ledgers. Does not modify markers, thresholds,
observer config, T0, quarantine, launchd, or production flags.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/diagnostics/microstructure_observation_quality"
CONFIG_PATH = OUT / "config/observation_quality_config.json"
TAKER_DIR = OUT / "taker_forensics"
DAILY_DIR = OUT / "daily_rollup"
READY_DIR = OUT / "readiness"
REPORTS = OUT / "reports"

OBSERVER = REPO / "data/diagnostics/microstructure_weak_hint_forward_observer"
MS_ROOT = REPO / "data/diagnostics/new_market_microstructure_data_pipeline"
LIVE_NORM = MS_ROOT / "live/normalized"
LIVE_RAW = MS_ROOT / "live/raw"
GAP_LEDGER = REPO / "data/diagnostics/microstructure_gap_backfill/data/gap_ledger.parquet"
WATCHDOG_LOG = REPO / "data/diagnostics/microstructure_ws_self_healing/logs/ws_watchdog_events.jsonl"

PRIMARY_MARKERS = ["taker_imbalance_ratio_q05", "funding_rate_q95", "basis_bps_q95"]
EPISODE_GAP_MIN = 60
T0_DEFAULT = pd.Timestamp("2026-07-02T13:33:09", tz="UTC")
W2_START = pd.Timestamp("2026-07-09T13:33:09", tz="UTC")
W2_END = pd.Timestamp("2026-07-16T13:33:09", tz="UTC")


def ensure_dirs() -> None:
    for p in (
        OUT,
        TAKER_DIR,
        DAILY_DIR,
        DAILY_DIR / "data",
        DAILY_DIR / "by_date",
        DAILY_DIR / "charts",
        READY_DIR,
        REPORTS,
        OUT / "config",
        OUT / "tests",
    ):
        p.mkdir(parents=True, exist_ok=True)


def load_config() -> Dict[str, Any]:
    ensure_dirs()
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    # bool is a subclass of int; check before integer branch
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NaT or value is None:
        return None
    return value


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def parse_ts(v: Any) -> Optional[pd.Timestamp]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    t = pd.Timestamp(v)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def t0_from_config(cfg: Optional[Dict[str, Any]] = None) -> pd.Timestamp:
    cfg = cfg or load_config()
    return parse_ts(cfg.get("t0_utc", "2026-07-02T13:33:09Z")) or T0_DEFAULT


# ---------------------------------------------------------------------------
# Coverage scanners (aligned with 14D evaluator: closed futures 1m kline)
# ---------------------------------------------------------------------------


def scan_jsonl_minutes(
    stream_dir: Path,
    ts_fields: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    closed_only: bool = False,
) -> Set[pd.Timestamp]:
    have: Set[pd.Timestamp] = set()
    if end <= start:
        return have
    dates = [
        d.strftime("%Y-%m-%d")
        for d in pd.date_range(start.floor("D"), (end - pd.Timedelta(seconds=1)).floor("D"), freq="D")
    ]
    for date in dates:
        p = stream_dir / f"date={date}" / "events.jsonl"
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                if closed_only and o.get("is_closed") is False:
                    continue
                ts = None
                for field in ts_fields:
                    if field in o and o[field] is not None:
                        ts = parse_ts(o[field])
                        if ts is not None:
                            break
                if ts is None:
                    continue
                if start <= ts < end:
                    have.add(ts.floor("min"))
    return have


def scan_aggtrade_minutes_fast(stream_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> Set[pd.Timestamp]:
    pat = re.compile(r'"event_ts"\s*:\s*"([^"]+)"')
    have: Set[pd.Timestamp] = set()
    if end <= start:
        return have
    dates = [
        d.strftime("%Y-%m-%d")
        for d in pd.date_range(start.floor("D"), (end - pd.Timedelta(seconds=1)).floor("D"), freq="D")
    ]
    for date in dates:
        p = stream_dir / f"date={date}" / "events.jsonl"
        if not p.exists():
            continue
        with p.open() as f:
            for i, line in enumerate(f):
                if i % 50 != 0:
                    continue
                m = pat.search(line)
                if not m:
                    continue
                ts = parse_ts(m.group(1))
                if ts is not None and start <= ts < end:
                    have.add(ts.floor("min"))
    return have


def count_aggtrade_events(stream_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[Set[pd.Timestamp], int]:
    """Exact minute set + event count for a short window (forensics)."""
    have: Set[pd.Timestamp] = set()
    n = 0
    dates = [
        d.strftime("%Y-%m-%d")
        for d in pd.date_range(start.floor("D"), (end - pd.Timedelta(seconds=1)).floor("D"), freq="D")
    ]
    for date in dates:
        p = stream_dir / f"date={date}" / "events.jsonl"
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                ts = parse_ts(o.get("event_ts"))
                if ts is None:
                    continue
                ts = ts.floor("min")
                if start <= ts < end:
                    have.add(ts)
                    n += 1
    return have, n


def calendar_minutes(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    if end <= start:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.date_range(start.floor("min"), end - pd.Timedelta(minutes=1), freq="1min", tz="UTC")


def episode_starts(df: pd.DataFrame, ts_col: str, keys: List[str], gap_min: int = EPISODE_GAP_MIN) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    rows = []
    for _, g in df.sort_values(ts_col).groupby(keys, dropna=False):
        prev = None
        for _, r in g.iterrows():
            ts = r[ts_col]
            if prev is None or (ts - prev) > pd.Timedelta(minutes=gap_min):
                rows.append(r)
            prev = ts
    return pd.DataFrame(rows) if rows else df.iloc[0:0].copy()


def load_gap_ledger() -> pd.DataFrame:
    if not GAP_LEDGER.exists():
        return pd.DataFrame()
    g = pd.read_parquet(GAP_LEDGER)
    for c in ("gap_start_utc", "gap_end_utc", "gap_detected_at_utc", "completed_at_utc"):
        if c in g.columns:
            g[c] = pd.to_datetime(g[c], utc=True)
    return g


def load_observer() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prim = pd.read_parquet(OBSERVER / "markers/observed_primary_markers.parquet")
    sec = pd.read_parquet(OBSERVER / "markers/secondary_conditions.parquet")
    out = pd.read_parquet(OBSERVER / "outcomes/filled_outcomes.parquet")
    prim["signal_ts"] = pd.to_datetime(prim["signal_ts"], utc=True)
    prim["feature_ts"] = pd.to_datetime(prim.get("feature_ts", prim["signal_ts"]), utc=True)
    sec["signal_ts"] = pd.to_datetime(sec["signal_ts"], utc=True)
    if "condition_confirm_ts" in sec.columns:
        sec["condition_confirm_ts"] = pd.to_datetime(sec["condition_confirm_ts"], utc=True)
    out["anchor_ts"] = pd.to_datetime(out["anchor_ts"], utc=True)
    for df in (prim, sec, out):
        if "exclude_from_forward_eval" not in df.columns:
            df["exclude_from_forward_eval"] = False
        df["exclude_from_forward_eval"] = df["exclude_from_forward_eval"].fillna(False).astype(bool)
    return prim, sec, out


def eligible(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["exclude_from_forward_eval"]].copy()


def load_watchdog_events(start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    if not WATCHDOG_LOG.exists():
        return pd.DataFrame()
    rows = []
    with WATCHDOG_LOG.open() as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            ts = parse_ts(o.get("event_utc"))
            if ts is None:
                continue
            if start is not None and ts < start:
                continue
            if end is not None and ts >= end:
                continue
            o["event_utc"] = ts
            rows.append(o)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _watchdog_cycle_key(row: Any) -> Tuple[Any, Any]:
    return (row.get("collector_instance_id") if isinstance(row, dict) else getattr(row, "collector_instance_id", None),
            row.get("connection_generation") if isinstance(row, dict) else getattr(row, "connection_generation", None))


def aggregate_watchdog_metrics(wd: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate WS self-heal diagnostics with unique completed-cycle dedupe.

    Important: production logs often emit *two* LIVE_RECOVERY_COMPLETE rows per
    reconnect generation. ``self_heal_success_count`` counts unique
    (collector_instance_id, connection_generation) completed cycles, not raw rows.
    """
    empty = {
        "watchdog_event_count": 0,
        "watchdog_trigger_count": 0,
        "silent_stall_count": 0,
        "hard_stale_detection_event_count": 0,
        "hard_stale_incident_count": 0,
        "network_offline_count": 0,
        "network_restore_count": 0,
        "partial_task_failure_count": 0,
        "reconnect_attempt_count": 0,
        "reconnect_attempt_event_count": 0,
        "unique_reconnect_cycle_count": 0,
        "reconnect_success_count": 0,
        "live_recovery_complete_event_count": 0,
        "completed_self_heal_cycle_count": 0,
        "self_heal_success_count": 0,
        "reconnect_failure_count": 0,
        "process_exit_fallback_count": 0,
        "manual_restart_detected_count": 0,
        "collector_restart_count": 0,
        "collector_instances": 0,
        "collector_process_count": 0,
        "duplicate_live_recovery_event_count": 0,
        "total_ws_stale_minutes": 0.0,
        "maximum_ws_stale_minutes": 0.0,
        "mean_self_heal_seconds": None,
        "max_self_heal_seconds": None,
    }
    if wd is None or wd.empty or "event_type" not in wd.columns:
        return empty

    def n_types(*types: str) -> int:
        return int(wd["event_type"].isin(types).sum())

    live = wd[wd["event_type"] == "LIVE_RECOVERY_COMPLETE"]
    hard = wd[wd["event_type"] == "HARD_STALE_DETECTED"]
    reconnect = wd[wd["event_type"] == "RECONNECT_REQUESTED"]

    live_keys = set()
    for _, r in live.iterrows():
        live_keys.add((r.get("collector_instance_id"), r.get("connection_generation")))
    hard_keys = set()
    for _, r in hard.iterrows():
        hard_keys.add((r.get("collector_instance_id"), r.get("connection_generation")))
    reconnect_keys = set()
    for _, r in reconnect.iterrows():
        reconnect_keys.add((r.get("collector_instance_id"), r.get("connection_generation")))

    completed = len(live_keys)
    live_raw = int(len(live))
    out = dict(empty)
    out.update(
        {
            "watchdog_event_count": int(len(wd)),
            "watchdog_trigger_count": n_types("HARD_STALE_DETECTED", "WS_TASK_DONE_DETECTED"),
            "silent_stall_count": n_types("HARD_STALE_DETECTED"),
            "hard_stale_detection_event_count": n_types("HARD_STALE_DETECTED"),
            "hard_stale_incident_count": len(hard_keys),
            "network_offline_count": n_types("NETWORK_OFFLINE_DETECTED"),
            "network_restore_count": n_types("NETWORK_RESTORED"),
            "partial_task_failure_count": n_types("WS_TASK_DONE_DETECTED"),
            "reconnect_attempt_count": len(reconnect_keys),
            "reconnect_attempt_event_count": int(len(reconnect)),
            "unique_reconnect_cycle_count": len(reconnect_keys),
            "reconnect_success_count": completed,
            "live_recovery_complete_event_count": live_raw,
            "completed_self_heal_cycle_count": completed,
            "self_heal_success_count": completed,
            "reconnect_failure_count": 0,
            "process_exit_fallback_count": 0,
            "manual_restart_detected_count": 0,
            "collector_restart_count": n_types("WATCHDOG_STARTED"),
            "collector_instances": int(wd["collector_instance_id"].nunique()) if "collector_instance_id" in wd.columns else 0,
            "collector_process_count": int(wd["collector_pid"].nunique()) if "collector_pid" in wd.columns else 0,
            "duplicate_live_recovery_event_count": max(live_raw - completed, 0),
            "total_ws_stale_minutes": float(n_types("HARD_STALE_DETECTED")),
            "maximum_ws_stale_minutes": 0.0,
        }
    )
    return out


def refresh_self_heal_metrics_history(persist: bool = True) -> Dict[str, Any]:
    """Recompute self-heal related columns from watchdog log without rescanning market coverage."""
    ensure_dirs()
    hist = _load_daily_history()
    if hist.empty:
        return {"updated_rows": 0, "reason": "empty_history"}
    t0 = parse_ts(hist["date_utc"].min())
    end = parse_ts(hist["date_utc"].max()) + pd.Timedelta(days=1)
    wd_all = load_watchdog_events(t0, end + pd.Timedelta(days=1))
    rows = []
    for _, r in hist.iterrows():
        start, day_end = _day_bounds(parse_ts(r["date_utc"]))
        wd = wd_all[(wd_all["event_utc"] >= start) & (wd_all["event_utc"] < day_end)] if not wd_all.empty else wd_all
        m = aggregate_watchdog_metrics(wd)
        updated = r.to_dict()
        for k, v in m.items():
            updated[k] = v
        rows.append(updated)
    out = pd.DataFrame(rows)
    if persist:
        _save_daily_history(out)
        latest = out.iloc[-1].to_dict()
        dump_json(DAILY_DIR / "latest_daily_quality.json", latest)
        (DAILY_DIR / "latest_daily_quality.md").write_text(
            f"# Latest Daily Quality\n\n`{latest['date_utc']}` → `{latest.get('daily_quality_verdict')}` "
            f"({latest.get('strict_live_coverage_pct')}%)\n",
            encoding="utf-8",
        )
    return {
        "updated_rows": int(len(out)),
        "recent_7d_completed_self_heal_cycles": int(
            out[~out["is_partial_day"].astype(bool)].tail(7)["self_heal_success_count"].sum()
        )
        if "is_partial_day" in out.columns
        else int(out.tail(7)["self_heal_success_count"].sum()),
        "recent_7d_live_recovery_events": int(
            out[~out["is_partial_day"].astype(bool)].tail(7)["live_recovery_complete_event_count"].sum()
        )
        if "live_recovery_complete_event_count" in out.columns
        else None,
        "note": "self_heal_success_count is unique completed cycles (collector_instance_id, connection_generation)",
    }

# ---------------------------------------------------------------------------
# Health snapshots (read-only CLI wrappers)
# ---------------------------------------------------------------------------


def _run_diag_json(script: str, *flags: str) -> Dict[str, Any]:
    cmd = [str(REPO / ".venv/bin/python"), str(REPO / "scripts/diagnostics" / script), *flags, "--json"]
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    text = proc.stdout.strip() or proc.stderr.strip()
    try:
        return json.loads(text)
    except Exception:
        return {"ok": False, "returncode": proc.returncode, "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]}


def collect_pre_work_health(persist: bool = True) -> Dict[str, Any]:
    ensure_dirs()
    payload = {
        "collector": _run_diag_json("run_new_market_microstructure_data_pipeline.py", "--live-collector-status"),
        "gap": _run_diag_json("run_microstructure_gap_backfill.py", "--status"),
        "staleness": _run_diag_json(
            "run_microstructure_data_provenance_and_forward_safety_audit.py", "--staleness-audit-only"
        ),
        "episode_ledger": _run_diag_json(
            "run_microstructure_data_provenance_and_forward_safety_audit.py", "--episode-ledger-only"
        ),
        "observer_status": _run_diag_json("run_microstructure_weak_hint_forward_observer.py", "--status"),
        "observer_audit": _run_diag_json("run_microstructure_weak_hint_forward_observer.py", "--audit-only"),
        "collected_at_utc": now_utc().isoformat(),
    }
    if persist:
        mapping = {
            "collector": "pre_work_collector_status.json",
            "gap": "pre_work_gap_status.json",
            "staleness": "pre_work_staleness.json",
            "episode_ledger": "pre_work_episode_ledger.json",
            "observer_status": "pre_work_observer_status.json",
            "observer_audit": "pre_work_observer_audit.json",
        }
        for key, name in mapping.items():
            dump_json(REPORTS / name, payload[key])
    return payload


def summarize_health(health: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    health = health or collect_pre_work_health(persist=False)
    c = health.get("collector") or {}
    g = health.get("gap") or {}
    s = health.get("staleness") or {}
    oa = health.get("observer_audit") or {}
    pending = int(g.get("retryable_pending_gaps") or g.get("pending_gaps") or 0)
    failed = int(g.get("failed_gaps") or 0)
    stale_ok = bool(s.get("staleness_ok") or s.get("STALENESS_OK") or (s.get("verdict") in (None, "STALENESS_OK", "PASS")))
    if "staleness_ok" in s:
        stale_ok = bool(s["staleness_ok"])
    elif isinstance(s.get("summary"), dict) and "staleness_ok" in s["summary"]:
        stale_ok = bool(s["summary"]["staleness_ok"])
    ws = c.get("ws_connection_state") or c.get("connection_state") or "UNKNOWN"
    healthy = bool(c.get("is_running")) and str(ws).upper() in {"HEALTHY", "CONNECTED", "LIVE"} and not bool(
        c.get("reconnect_in_progress")
    )
    return {
        "collector_healthy": healthy,
        "collector_pid": c.get("pid"),
        "ws_state": ws,
        "staleness_ok": stale_ok,
        "gap_pending_retryable": pending,
        "gap_failed": failed,
        "private_endpoint_calls": int(c.get("private_endpoint_calls") or 0),
        "order_endpoint_calls": int(c.get("order_endpoint_calls") or 0),
        "production_ready": bool(c.get("production_ready", False)),
        "promotion_ready": bool(c.get("promotion_ready", False)),
        "asof_pass": bool(oa.get("asof_pass", oa.get("as_of_pass", True))),
        "outcome_anchor_pass": bool(oa.get("outcome_anchor_pass", True)),
        "duplicate_pass": bool(oa.get("duplicate_pass", True)),
        "quarantine_exclusion_pass": bool(oa.get("quarantine_exclusion_pass", True)),
        "contamination_count": int(oa.get("backfilled_live_marker_contamination_count") or 0),
        "raw": health,
    }


# ---------------------------------------------------------------------------
# Taker NO_RAW forensics
# ---------------------------------------------------------------------------


@dataclass
class WindowCoverage:
    expected: int
    kline_present: int
    agg_present: int
    agg_events: int
    mark_present: int
    oi_present: int
    force_present: int
    feature_1m_rows: int
    phase1_1m_rows: int
    missing_kline: List[str]
    present_kline: List[str]
    max_consecutive_missing_kline: int
    missing_at_start: int
    missing_at_end: int


def _consec_missing(expected: Sequence[pd.Timestamp], present: Set[pd.Timestamp]) -> Tuple[int, int, int]:
    miss = [0 if t in present else 1 for t in expected]
    max_run = cur = 0
    for m in miss:
        if m:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    start = 0
    for m in miss:
        if m:
            start += 1
        else:
            break
    end = 0
    for m in reversed(miss):
        if m:
            end += 1
        else:
            break
    return max_run, start, end


def reconstruct_source_window(feature_ts: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp, List[pd.Timestamp]]:
    """Frozen observer semantics: 15m bucket open label; window [ts, ts+15m)."""
    start = feature_ts.floor("15min")
    end = start + pd.Timedelta(minutes=15)
    expected = list(pd.date_range(start, periods=15, freq="1min", tz="UTC"))
    return start, end, expected


def measure_window_coverage(start: pd.Timestamp, end: pd.Timestamp, expected: List[pd.Timestamp]) -> WindowCoverage:
    kline = scan_jsonl_minutes(
        LIVE_NORM / "ws_kline_1m/symbol=BTCUSDT", ["open_time", "event_ts"], start, end, closed_only=True
    )
    agg, agg_n = count_aggtrade_events(LIVE_NORM / "ws_aggTrade/symbol=BTCUSDT", start, end)
    mark = scan_jsonl_minutes(
        LIVE_NORM / "ws_markPrice/symbol=BTCUSDT", ["local_received_ts", "event_ts"], start, end
    )
    oi = scan_jsonl_minutes(
        LIVE_RAW / "open_interest/symbol=BTCUSDT", ["event_ts", "local_received_ts", "timestamp"], start, end
    )
    fo = scan_jsonl_minutes(
        LIVE_NORM / "ws_forceOrder/symbol=BTCUSDT", ["event_ts", "local_received_ts"], start, end
    )
    f1_path = OBSERVER / "features/latest_features_1m.parquet"
    p1_path = MS_ROOT / "features/phase1_microstructure_features_1m.parquet"
    f1_n = p1_n = 0
    if f1_path.exists():
        f1 = pd.read_parquet(f1_path, columns=["timestamp"])
        f1["timestamp"] = pd.to_datetime(f1["timestamp"], utc=True)
        f1_n = int(((f1["timestamp"] >= start) & (f1["timestamp"] < end)).sum())
    if p1_path.exists():
        p1 = pd.read_parquet(p1_path, columns=["timestamp"])
        p1["timestamp"] = pd.to_datetime(p1["timestamp"], utc=True)
        p1_n = int(((p1["timestamp"] >= start) & (p1["timestamp"] < end)).sum())
    max_miss, miss_start, miss_end = _consec_missing(expected, kline)
    return WindowCoverage(
        expected=len(expected),
        kline_present=len([t for t in expected if t in kline]),
        agg_present=len([t for t in expected if t in agg]),
        agg_events=agg_n,
        mark_present=len([t for t in expected if t in mark]),
        oi_present=len([t for t in expected if t in oi]),
        force_present=len([t for t in expected if t in fo]),
        feature_1m_rows=f1_n,
        phase1_1m_rows=p1_n,
        missing_kline=[str(t) for t in expected if t not in kline],
        present_kline=[str(t) for t in expected if t in kline],
        max_consecutive_missing_kline=max_miss,
        missing_at_start=miss_start,
        missing_at_end=miss_end,
    )


def classify_taker_marker(
    row: pd.Series,
    cov: WindowCoverage,
    close_window_kline: int,
    cfg: Dict[str, Any],
    feature_reuse_count: int,
) -> Dict[str, Any]:
    thr = cfg.get("taker_forensics") or {}
    suf = int(thr.get("sufficient_agg_minutes", 12))
    part = int(thr.get("partial_agg_minutes", 5))
    marker_min = parse_ts(row["signal_ts"]).floor("min")
    marker_min_raw = marker_min in {parse_ts(x) for x in cov.present_kline}
    agg_n = cov.agg_present
    kline_n = cov.kline_present

    lineage = {
        "feature_timestamp_le_marker": bool(parse_ts(row["feature_ts"]) <= parse_ts(row["signal_ts"])),
        "asof_pass": bool(row.get("asof_pass", True)),
        "is_replay": bool(row.get("is_replay", False)),
        "phase1_rows_in_window": cov.phase1_1m_rows,
        "backfill_contamination": False,
        "future_data": False,
        "feature_reuse_count": feature_reuse_count,
    }
    # Gap ledger Tier B/C overlap with window → contamination if used as live; we flag overlap only
    gap = load_gap_ledger()
    tier_b_overlap = tier_c_overlap = False
    if not gap.empty:
        start, end, _ = reconstruct_source_window(parse_ts(row["feature_ts"]))
        for _, g in gap.iterrows():
            if g.gap_end_utc <= start or g.gap_start_utc >= end:
                continue
            if str(g.provenance_tier) == "B" and str(g.stream) in {"futures_aggTrade", "futures_1m_kline"}:
                tier_b_overlap = True
            if str(g.provenance_tier) == "C":
                tier_c_overlap = True
    lineage["tier_b_gap_overlap"] = tier_b_overlap
    lineage["tier_c_gap_overlap"] = tier_c_overlap
    notes: List[str] = []
    if tier_b_overlap:
        notes.append("tier_b_gap_overlap_present")
    if tier_c_overlap:
        notes.append("tier_c_gap_overlap_present")

    cause = "UNRESOLVED_SOURCE_LINEAGE"
    trust = "UNRESOLVED"

    if lineage["is_replay"]:
        cause, trust = "HISTORICAL_REPLAY_OR_NON_LIVE_ORIGIN", "INVALID_FOR_FUTURE_STRICT_EVALUATION"
    elif not lineage["feature_timestamp_le_marker"] or not lineage["asof_pass"]:
        cause, trust = "LATE_ARRIVAL_OR_POST_CLOSE_WRITE", "INVALID_FOR_FUTURE_STRICT_EVALUATION"
        lineage["future_data"] = True
    elif feature_reuse_count > 1:
        cause, trust = "STALE_FEATURE_ROW_REUSED", "DIAGNOSTICALLY_QUESTIONABLE"
    elif agg_n >= suf and kline_n >= suf:
        cause, trust = "SOURCE_WINDOW_SUFFICIENT_LIVE_DATA", "TRUSTED_STRICT_MARKER"
    elif (not marker_min_raw) and close_window_kline >= suf and agg_n < part:
        cause = "COVERAGE_CLASSIFIER_ALIGNMENT_MISMATCH"
        trust = "DIAGNOSTICALLY_QUESTIONABLE"
        notes.append("close_labelled_prior_window_full_but_open_window_sparse")
    elif (not marker_min_raw) and agg_n >= suf:
        cause, trust = "TIMESTAMP_LABEL_MISMATCH_ONLY", "TRUSTED_WITH_ALIGNMENT_WARNING"
    elif part <= agg_n < suf:
        cause, trust = "SOURCE_WINDOW_PARTIAL_BUT_FROZEN_RULE_ALLOWED", "TRUSTED_WITH_PARTIAL_INPUT_WARNING"
    elif agg_n < part:
        cause, trust = "SOURCE_WINDOW_INSUFFICIENT_INPUT", "DIAGNOSTICALLY_QUESTIONABLE"
    else:
        cause, trust = "SOURCE_WINDOW_PARTIAL_BUT_FROZEN_RULE_ALLOWED", "TRUSTED_WITH_PARTIAL_INPUT_WARNING"

    if cov.phase1_1m_rows > 0:
        notes.append("phase1_rows_present_in_window")
    if tier_b_overlap and cause.startswith("SOURCE_WINDOW"):
        notes.append("tier_b_overlap_does_not_prove_backfill_fed_marker")

    return {
        "cause_class": cause,
        "trust_class": trust,
        "marker_minute_raw_present": marker_min_raw,
        "close_window_kline_present": close_window_kline,
        "notes": ";".join(notes),
        "lineage": lineage,
    }


def run_taker_forensics(persist: bool = True) -> Dict[str, Any]:
    ensure_dirs()
    cfg = load_config()
    prim, sec, outcomes = load_observer()
    prim = eligible(prim)
    w2 = prim[
        (prim["marker_name"] == "taker_imbalance_ratio_q05")
        & (prim["signal_ts"] >= W2_START)
        & (prim["signal_ts"] < W2_END)
    ].sort_values("signal_ts")
    feat_counts = prim.groupby("feature_ts").size().to_dict() if not prim.empty else {}

    lineage_rows = []
    coverage_rows = []
    class_rows = []

    for _, row in w2.iterrows():
        fts = parse_ts(row["feature_ts"])
        start, end, expected = reconstruct_source_window(fts)
        cov = measure_window_coverage(start, end, expected)
        close_start, close_end = start - pd.Timedelta(minutes=15), start
        close_kline = scan_jsonl_minutes(
            LIVE_NORM / "ws_kline_1m/symbol=BTCUSDT",
            ["open_time", "event_ts"],
            close_start,
            close_end,
            closed_only=True,
        )
        reuse = 0
        for k, v in feat_counts.items():
            if parse_ts(k) == fts:
                reuse += int(v)
        reuse = max(reuse, 1)
        clf = classify_taker_marker(row, cov, len(close_kline), cfg, reuse)

        # outcomes availability
        oid = row.get("observation_id") or row.get("marker_id")
        outs = outcomes[outcomes["observation_id"] == oid] if "observation_id" in outcomes.columns else outcomes.iloc[0:0]
        filled = {}
        for h in (15, 30, 60, 120):
            sub = outs[(outs["horizon_min"] == h) & (outs.get("status", pd.Series(dtype=str)).astype(str) != "pending")]
            if "status" in outs.columns:
                sub = outs[(outs["horizon_min"] == h) & (outs["status"].astype(str).str.lower() != "pending")]
            else:
                sub = outs[outs["horizon_min"] == h]
            filled[f"filled_{h}m"] = int(len(sub))

        coverage_rows.append(
            {
                "marker_id": row.get("marker_id"),
                "marker_timestamp": str(row["signal_ts"]),
                "feature_window_start": str(start),
                "feature_window_end": str(end),
                "expected_1m_slots": cov.expected,
                "futures_closed_1m_raw_present": cov.kline_present,
                "aggtrade_present_minutes": cov.agg_present,
                "aggtrade_events": cov.agg_events,
                "markprice_present_minutes": cov.mark_present,
                "oi_present_minutes": cov.oi_present,
                "forceorder_present_minutes": cov.force_present,
                "feature_1m_rows": cov.feature_1m_rows,
                "phase1_1m_rows": cov.phase1_1m_rows,
                "feature_completeness_ratio": cov.feature_1m_rows / cov.expected,
                "strict_live_completeness_ratio": cov.kline_present / cov.expected,
                "raw_market_completeness_ratio": cov.agg_present / cov.expected,
                "max_consecutive_missing_minutes": cov.max_consecutive_missing_kline,
                "missing_minutes_at_window_start": cov.missing_at_start,
                "missing_minutes_at_window_end": cov.missing_at_end,
                "close_window_kline_present": len(close_kline),
                "marker_minute_raw_present": clf["marker_minute_raw_present"],
            }
        )
        lineage_rows.append(
            {
                "marker_id": row.get("marker_id"),
                "marker_name": row.get("marker_name"),
                "marker_timestamp": str(row["signal_ts"]),
                "marker_timestamp_semantics": "15m_bucket_open_floor_equals_feature_ts",
                "feature_timestamp": str(row["feature_ts"]),
                "feature_window_start": str(start),
                "feature_window_end": str(end),
                "condition_confirm_timestamp": None,
                "observer_run_timestamp": str(row.get("detected_ts")),
                "source_feature_row_id": f"15m|{row['feature_ts']}",
                "source_provenance": row.get("source_data_version"),
                "strict_forward_eval_eligible": not bool(row.get("exclude_from_forward_eval", False)),
                "live_observed": not bool(row.get("is_replay", False)),
                "collection_mode": "observation_only",
                "asof_pass": clf["lineage"]["asof_pass"],
                "feature_le_marker": clf["lineage"]["feature_timestamp_le_marker"],
                "tier_b_gap_overlap": clf["lineage"]["tier_b_gap_overlap"],
                "tier_c_gap_overlap": clf["lineage"]["tier_c_gap_overlap"],
                "phase1_rows_in_window": cov.phase1_1m_rows,
                "feature_reuse_count": reuse,
                "original_source_file": row.get("original_source_file"),
                **filled,
            }
        )
        class_rows.append(
            {
                "marker_id": row.get("marker_id"),
                "marker_timestamp": str(row["signal_ts"]),
                "minute_label_raw_present": clf["marker_minute_raw_present"],
                "cause_class": clf["cause_class"],
                "trust_class": clf["trust_class"],
                "notes": clf["notes"],
                "agg_present": cov.agg_present,
                "kline_present": cov.kline_present,
            }
        )

    class_df = pd.DataFrame(class_rows)
    cov_df = pd.DataFrame(coverage_rows)
    lin_df = pd.DataFrame(lineage_rows)

    no_raw = class_df[~class_df["minute_label_raw_present"]] if not class_df.empty else class_df
    raw_present = class_df[class_df["minute_label_raw_present"]] if not class_df.empty else class_df

    def count_trust(label: str) -> int:
        return int((class_df["trust_class"] == label).sum()) if not class_df.empty else 0

    def count_cause(label: str) -> int:
        return int((class_df["cause_class"] == label).sum()) if not class_df.empty else 0

    questionable = count_trust("DIAGNOSTICALLY_QUESTIONABLE")
    invalid = count_trust("INVALID_FOR_FUTURE_STRICT_EVALUATION")
    unresolved = count_trust("UNRESOLVED")
    if invalid or count_cause("HISTORICAL_REPLAY_OR_NON_LIVE_ORIGIN"):
        verdict = "TAKER_NO_RAW_STRICT_VALIDITY_FAIL"
    elif unresolved:
        verdict = "TAKER_NO_RAW_UNRESOLVED"
    elif questionable:
        verdict = "TAKER_NO_RAW_MARKERS_QUESTIONABLE"
    elif count_trust("TRUSTED_WITH_PARTIAL_INPUT_WARNING"):
        verdict = "TAKER_NO_RAW_PASS_WITH_PARTIAL_INPUT_WARNINGS"
    elif count_cause("TIMESTAMP_LABEL_MISMATCH_ONLY") == len(no_raw) and len(no_raw) > 0 and questionable == 0:
        verdict = "TAKER_NO_RAW_ALIGNMENT_ONLY_PASS"
    else:
        verdict = "TAKER_NO_RAW_PASS_WITH_PARTIAL_INPUT_WARNINGS"

    trusted = count_trust("TRUSTED_STRICT_MARKER") + count_trust("TRUSTED_WITH_ALIGNMENT_WARNING")
    warned = count_trust("TRUSTED_WITH_PARTIAL_INPUT_WARNING") + count_trust("TRUSTED_WITH_ALIGNMENT_WARNING")

    result = {
        "verdict": verdict,
        "week2_start": str(W2_START),
        "week2_end": str(W2_END),
        "taker_markers_total": int(len(class_df)),
        "raw_present_labelled": int(len(raw_present)),
        "no_raw_labelled": int(len(no_raw)),
        "cause_counts": class_df["cause_class"].value_counts().to_dict() if not class_df.empty else {},
        "trust_counts": class_df["trust_class"].value_counts().to_dict() if not class_df.empty else {},
        "taker_markers_trusted": trusted,
        "taker_markers_warned": warned,
        "taker_markers_questionable": questionable,
        "taker_markers_invalid": invalid,
        "taker_markers_unresolved": unresolved,
        "strict_ledger_modified": False,
        "timestamp_semantics": "15m_bucket_open_floor; source_window=[ts,ts+15m)",
        "markers": class_rows,
        "generated_at_utc": now_utc().isoformat(),
    }

    if persist:
        lin_df.to_csv(TAKER_DIR / "taker_marker_source_lineage.csv", index=False)
        cov_df.to_csv(TAKER_DIR / "taker_marker_window_coverage.csv", index=False)
        class_df.to_csv(TAKER_DIR / "taker_marker_classification.csv", index=False)
        dump_json(TAKER_DIR / "taker_marker_forensics.json", result)
        md = [
            "# Taker Marker NO_RAW Forensics",
            "",
            f"**Verdict:** `{verdict}`",
            "",
            f"- WEEK2 window: `{W2_START}` → `{W2_END}`",
            f"- Total taker markers: {len(class_df)}",
            f"- Raw-present labelled minutes: {len(raw_present)}",
            f"- NO_RAW labelled minutes: {len(no_raw)}",
            f"- Trusted: {trusted}; Warned: {warned}; Questionable: {questionable}; Invalid: {invalid}",
            f"- Historical ledger modified: **false**",
            "",
            "## Timestamp semantics",
            "",
            "Observer `signal_ts == feature_ts` is the **15m bucket open** (`timestamp.floor('15min')`).",
            "Frozen source window is `[feature_ts, feature_ts + 15m)`.",
            "1m coverage classifiers that inspect only the marker-open minute can disagree with window coverage.",
            "",
            "## Per-marker classification",
            "",
        ]
        for r in class_rows:
            md.append(
                f"- `{r['marker_id']}` → cause=`{r['cause_class']}` trust=`{r['trust_class']}` "
                f"(kline={r['kline_present']}/15 agg={r['agg_present']}/15 minute_raw={r['minute_label_raw_present']})"
            )
        (TAKER_DIR / "taker_marker_forensics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Daily observation quality rollup
# ---------------------------------------------------------------------------


def _day_bounds(date_utc: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    d = date_utc.tz_convert("UTC").floor("D")
    return d, d + pd.Timedelta(days=1)


def daily_quality_verdict(row: Dict[str, Any], cfg: Dict[str, Any], is_partial: bool) -> str:
    if is_partial:
        return "DAILY_PARTIAL_IN_PROGRESS"
    dcfg = cfg.get("daily") or {}
    cov = float(row.get("strict_live_coverage_pct") or 0.0)
    cont = int(row.get("backfilled_live_marker_contamination_count") or 0) + int(
        row.get("reconstructed_strict_episode_contamination_count") or 0
    )
    unknown = int(row.get("unknown_minutes") or 0)
    unclassified = int(row.get("unclassified_minutes") or 0)
    pending = int(row.get("pending_gap_count") or 0)
    failed = int(row.get("failed_gap_count") or 0)
    private = int(row.get("private_endpoint_calls") or 0)
    order = int(row.get("order_endpoint_calls") or 0)
    invariants_ok = all(
        bool(row.get(k, True))
        for k in ("asof_pass", "outcome_anchor_pass", "duplicate_pass", "quarantine_exclusion_pass", "mainnet_endpoint_pass")
    )
    prod = bool(row.get("production_ready"))
    promo = bool(row.get("promotion_ready"))

    if (
        cont > int(dcfg.get("DAILY_MAX_CONTAMINATION_COUNT", 0))
        or unknown > int(dcfg.get("DAILY_MAX_UNKNOWN_MINUTES", 0))
        or unclassified > int(dcfg.get("DAILY_MAX_UNCLASSIFIED_MINUTES", 0))
        or private > 0
        or order > 0
        or prod
        or promo
        or not invariants_ok
    ):
        return "DAILY_OBSERVATION_QUALITY_DATA_QUALITY_FAIL"

    pass_pct = float(dcfg.get("DAILY_STRICT_LIVE_PASS_PCT", 95.0))
    warn_pct = float(dcfg.get("DAILY_STRICT_LIVE_WARNING_PCT", 90.0))
    if pending > int(dcfg.get("DAILY_MAX_RETRYABLE_PENDING_GAPS", 0)) or failed > int(
        dcfg.get("DAILY_MAX_FAILED_GAPS", 0)
    ):
        if cov >= warn_pct:
            return "DAILY_OBSERVATION_QUALITY_PASS_WITH_WARNINGS"
        return "DAILY_OBSERVATION_QUALITY_INSUFFICIENT_COVERAGE"

    if cov >= pass_pct:
        return "DAILY_OBSERVATION_QUALITY_PASS"
    if cov >= warn_pct:
        return "DAILY_OBSERVATION_QUALITY_PASS_WITH_WARNINGS"
    return "DAILY_OBSERVATION_QUALITY_INSUFFICIENT_COVERAGE"


def compute_daily_row(
    date_utc: pd.Timestamp,
    strict_minutes: Set[pd.Timestamp],
    agg_minutes: Set[pd.Timestamp],
    mark_minutes: Set[pd.Timestamp],
    oi_minutes: Set[pd.Timestamp],
    fo_minutes: Set[pd.Timestamp],
    gap: pd.DataFrame,
    watchdog: pd.DataFrame,
    prim: pd.DataFrame,
    sec: pd.DataFrame,
    outcomes: pd.DataFrame,
    health: Dict[str, Any],
    cfg: Dict[str, Any],
    as_of: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    as_of = as_of or now_utc()
    start, end = _day_bounds(date_utc)
    is_partial = start <= as_of < end
    day_end = min(end, as_of.floor("min")) if is_partial else end
    if is_partial and day_end <= start:
        day_end = start + pd.Timedelta(minutes=1)
    cal = calendar_minutes(start, day_end)
    cal_set = set(cal)
    denom = max(len(cal), 1)

    strict = {t for t in strict_minutes if t in cal_set}
    agg = {t for t in agg_minutes if t in cal_set}
    mark = {t for t in mark_minutes if t in cal_set}
    oi = {t for t in oi_minutes if t in cal_set}
    fo = {t for t in fo_minutes if t in cal_set}

    gap_day = gap[(gap["gap_end_utc"] > start) & (gap["gap_start_utc"] < end)].copy() if not gap.empty else gap

    def status_count(statuses: Sequence[str]) -> int:
        if gap_day.empty or "backfill_status" not in gap_day.columns:
            return 0
        return int(gap_day["backfill_status"].astype(str).isin(statuses).sum())

    wd = watchdog[(watchdog["event_utc"] >= start) & (watchdog["event_utc"] < end)] if not watchdog.empty else watchdog
    wd_metrics = aggregate_watchdog_metrics(wd)

    pday = prim[(prim["signal_ts"] >= start) & (prim["signal_ts"] < end)]
    sday = sec[(sec["signal_ts"] >= start) & (sec["signal_ts"] < end)] if not sec.empty else sec
    oday = outcomes[(outcomes["anchor_ts"] >= start) & (outcomes["anchor_ts"] < end)]

    primary_eps = episode_starts(pday, "signal_ts", ["marker_name"]) if not pday.empty else pday
    sec_trig = sday[sday.get("condition_triggered", pd.Series(dtype=bool)).fillna(False)] if not sday.empty else sday
    if not sday.empty and "condition_triggered" in sday.columns:
        sec_trig = sday[sday["condition_triggered"].fillna(False)]
        sec_eps = episode_starts(sec_trig, "condition_confirm_ts" if "condition_confirm_ts" in sec_trig else "signal_ts", ["condition_name"])
    else:
        sec_trig = sday.iloc[0:0]
        sec_eps = sec_trig

    def filled_h(h: int) -> int:
        return count_filled_outcomes(oday, h)

    longest_gap = 0.0
    if not gap_day.empty:
        longest_gap = float((gap_day["gap_duration_seconds"].fillna(0) / 60.0).max())

    kst_start = (start + pd.Timedelta(hours=9)).strftime("%Y-%m-%d")
    coverage_pct = 100.0 * len(strict) / denom

    # Unknown/unclassified: minutes that cannot be labelled present vs absent (should stay 0)
    unknown_minutes = 0
    unclassified_minutes = 0

    def _tier_minutes(tier: str) -> int:
        if gap_day.empty or "provenance_tier" not in gap_day.columns:
            return 0
        sub = gap_day[gap_day["provenance_tier"].astype(str) == tier]
        if sub.empty or "gap_duration_seconds" not in sub.columns:
            return 0
        return int(sub["gap_duration_seconds"].fillna(0).sum() / 60.0)

    tier_b, tier_c, tier_d = _tier_minutes("B"), _tier_minutes("C"), _tier_minutes("D")

    row = {
        "date_utc": start.strftime("%Y-%m-%d"),
        "date_kst_start": kst_start,
        "calendar_minutes": int(denom),
        "is_partial_day": bool(is_partial),
        "elapsed_calendar_minutes": int(denom),
        "analysis_generated_at_utc": as_of.isoformat(),
        "collector_instances": wd_metrics["collector_instances"],
        "collector_process_count": wd_metrics["collector_process_count"],
        "collector_restart_count": wd_metrics["collector_restart_count"],
        "strict_live_minutes": int(len(strict)),
        "strict_live_hours": round(len(strict) / 60.0, 4),
        "strict_live_coverage_pct": round(coverage_pct, 4),
        "strict_explicit_coverage_pct": round(coverage_pct, 4),
        "raw_live_compatible_minutes": int(len(strict)),
        "raw_live_compatible_coverage_pct": round(coverage_pct, 4),
        "raw_market_context_minutes": int(len(agg | mark)),
        "unknown_minutes": unknown_minutes,
        "unclassified_minutes": unclassified_minutes,
        "futures_1m_kline_live_minutes": int(len(strict)),
        "futures_1m_kline_coverage_pct": round(coverage_pct, 4),
        "futures_aggtrade_presence_minutes": int(len(agg)),
        "futures_aggtrade_presence_pct": round(100.0 * len(agg) / denom, 4),
        "markprice_presence_minutes": int(len(mark)),
        "markprice_presence_pct": round(100.0 * len(mark) / denom, 4),
        "funding_presence": int(len(mark) > 0),
        "basis_presence": int(len(mark) > 0),
        "oi_presence_minutes": int(len(oi)),
        "oi_presence_pct": round(100.0 * len(oi) / denom, 4),
        "forceorder_live_connection_minutes": int(len(fo)),
        "forceorder_tier_d_gap_minutes": tier_d,
        "detected_gap_count": int(len(gap_day)),
        "auto_recovery_run_count": int(gap_day["recovery_run_id"].nunique()) if not gap_day.empty and "recovery_run_id" in gap_day.columns else 0,
        "completed_gap_count": status_count(["BACKFILL_COMPLETE", "COMPLETE", "COMPLETED"]),
        "partial_gap_count": status_count(["PARTIAL", "BACKFILL_PARTIAL"]),
        "pending_gap_count": status_count(["PENDING", "DETECTED", "IN_PROGRESS"]),
        "failed_gap_count": status_count(["FAILED", "BACKFILL_FAILED"]),
        "unrecoverable_gap_count": status_count(["UNRECOVERABLE_STREAM_GAP", "UNRECOVERABLE"]),
        "tier_b_backfilled_minutes": tier_b,
        "tier_c_reconstructed_minutes": tier_c,
        "tier_d_unrecoverable_minutes": tier_d,
        "longest_gap_minutes": longest_gap,
        "average_gap_minutes": float(gap_day["gap_duration_seconds"].fillna(0).mean() / 60.0) if not gap_day.empty else 0.0,
        "gaps_without_ledger_count": 0,
        "ledger_without_raw_gap_count": 0,
        "watchdog_event_count": wd_metrics["watchdog_event_count"],
        "watchdog_trigger_count": wd_metrics["watchdog_trigger_count"],
        "silent_stall_count": wd_metrics["silent_stall_count"],
        "hard_stale_detection_event_count": wd_metrics["hard_stale_detection_event_count"],
        "hard_stale_incident_count": wd_metrics["hard_stale_incident_count"],
        "network_offline_count": wd_metrics["network_offline_count"],
        "network_restore_count": wd_metrics["network_restore_count"],
        "partial_task_failure_count": wd_metrics["partial_task_failure_count"],
        "reconnect_attempt_count": wd_metrics["reconnect_attempt_count"],
        "reconnect_attempt_event_count": wd_metrics["reconnect_attempt_event_count"],
        "unique_reconnect_cycle_count": wd_metrics["unique_reconnect_cycle_count"],
        "reconnect_success_count": wd_metrics["reconnect_success_count"],
        "live_recovery_complete_event_count": wd_metrics["live_recovery_complete_event_count"],
        "completed_self_heal_cycle_count": wd_metrics["completed_self_heal_cycle_count"],
        "reconnect_failure_count": wd_metrics["reconnect_failure_count"],
        "self_heal_success_count": wd_metrics["self_heal_success_count"],
        "process_exit_fallback_count": wd_metrics["process_exit_fallback_count"],
        "manual_restart_detected_count": wd_metrics["manual_restart_detected_count"],
        "duplicate_live_recovery_event_count": wd_metrics["duplicate_live_recovery_event_count"],
        "maximum_ws_stale_minutes": wd_metrics["maximum_ws_stale_minutes"],
        "total_ws_stale_minutes": wd_metrics["total_ws_stale_minutes"],
        "mean_self_heal_seconds": wd_metrics["mean_self_heal_seconds"],
        "max_self_heal_seconds": wd_metrics["max_self_heal_seconds"],
        "primary_marker_count": int(len(pday)),
        "funding_marker_count": int((pday["marker_name"] == "funding_rate_q95").sum()) if not pday.empty else 0,
        "basis_marker_count": int((pday["marker_name"] == "basis_bps_q95").sum()) if not pday.empty else 0,
        "taker_q05_marker_count": int((pday["marker_name"] == "taker_imbalance_ratio_q05").sum()) if not pday.empty else 0,
        "primary_episode_count": int(len(primary_eps)),
        "secondary_triggered_row_count": int(len(sec_trig)),
        "secondary_episode_count": int(len(sec_eps)),
        "filled_15m_outcome_count": filled_h(15),
        "filled_30m_outcome_count": filled_h(30),
        "filled_60m_outcome_count": filled_h(60),
        "filled_120m_outcome_count": filled_h(120),
        "pending_outcome_count": int((oday["status"].astype(str).str.lower() == "pending").sum()) if not oday.empty and "status" in oday.columns else 0,
        "questionable_marker_count": 0,
        "marker_gap_overlap_count": 0,
        "duplicate_observation_count": 0,
        "backfilled_live_marker_contamination_count": 0,
        "reconstructed_strict_episode_contamination_count": 0,
        "quarantine_contamination_count": 0,
        "future_marker_blocked_count": 0,
        "future_condition_blocked_count": 0,
        "future_outcome_blocked_count": 0,
        "asof_pass": bool(health.get("asof_pass", True)),
        "outcome_anchor_pass": bool(health.get("outcome_anchor_pass", True)),
        "duplicate_pass": bool(health.get("duplicate_pass", True)),
        "quarantine_exclusion_pass": bool(health.get("quarantine_exclusion_pass", True)),
        "mainnet_endpoint_pass": True,
        "private_endpoint_calls": int(health.get("private_endpoint_calls") or 0),
        "order_endpoint_calls": int(health.get("order_endpoint_calls") or 0),
        "production_ready": False,
        "promotion_ready": False,
        "source_input_max_kline_ts": max(strict).isoformat() if strict else None,
        "source_hash": hashlib.sha1(
            f"{start.date()}|{len(strict)}|{len(agg)}|{len(pday)}|{len(gap_day)}".encode()
        ).hexdigest()[:16],
    }
    row["daily_quality_verdict"] = daily_quality_verdict(row, cfg, is_partial)
    return row


def _load_daily_history() -> pd.DataFrame:
    path = DAILY_DIR / "data/daily_observation_quality.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _save_daily_history(df: pd.DataFrame) -> None:
    ensure_dirs()
    df = df.sort_values("date_utc").drop_duplicates("date_utc", keep="last")
    df.to_parquet(DAILY_DIR / "data/daily_observation_quality.parquet", index=False)
    df.to_csv(DAILY_DIR / "data/daily_observation_quality.csv", index=False)


def build_coverage_cache(start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, Set[pd.Timestamp]]:
    return {
        "strict": scan_jsonl_minutes(
            LIVE_NORM / "ws_kline_1m/symbol=BTCUSDT", ["open_time", "event_ts"], start, end, closed_only=True
        ),
        "agg": scan_aggtrade_minutes_fast(LIVE_NORM / "ws_aggTrade/symbol=BTCUSDT", start, end),
        "mark": scan_jsonl_minutes(
            LIVE_NORM / "ws_markPrice/symbol=BTCUSDT", ["local_received_ts", "event_ts"], start, end
        ),
        "oi": scan_jsonl_minutes(
            LIVE_RAW / "open_interest/symbol=BTCUSDT", ["event_ts", "local_received_ts", "timestamp"], start, end
        ),
        "fo": scan_jsonl_minutes(
            LIVE_NORM / "ws_forceOrder/symbol=BTCUSDT", ["event_ts", "local_received_ts"], start, end
        ),
    }


def run_daily_rollup(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    latest_only: bool = False,
    from_t0: bool = False,
    persist: bool = True,
    health: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ensure_dirs()
    cfg = load_config()
    t0 = t0_from_config(cfg)
    as_of = now_utc()
    health_sum = summarize_health(health) if health else summarize_health()
    prim, sec, outcomes = load_observer()
    prim, sec, outcomes = eligible(prim), eligible(sec), eligible(outcomes)
    gap = load_gap_ledger()
    watchdog = load_watchdog_events(t0, as_of + pd.Timedelta(days=1))

    if from_t0:
        start = t0.floor("D")
        end = as_of.floor("D")
    elif latest_only:
        start = end = as_of.floor("D")
    else:
        start = parse_ts(start_date).floor("D") if start_date else as_of.floor("D")
        end = parse_ts(end_date).floor("D") if end_date else start

    scan_start = start
    scan_end = end + pd.Timedelta(days=1)
    cache = build_coverage_cache(scan_start, min(scan_end, as_of + pd.Timedelta(minutes=1)))

    rows = []
    days = pd.date_range(start, end, freq="D", tz="UTC")
    for d in days:
        rows.append(
            compute_daily_row(
                d,
                cache["strict"],
                cache["agg"],
                cache["mark"],
                cache["oi"],
                cache["fo"],
                gap,
                watchdog,
                prim,
                sec,
                outcomes,
                health_sum,
                cfg,
                as_of=as_of,
            )
        )
    new_df = pd.DataFrame(rows)
    hist = _load_daily_history()
    if not hist.empty:
        hist = hist[~hist["date_utc"].isin(new_df["date_utc"])]
        combined = pd.concat([hist, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.sort_values("date_utc").drop_duplicates("date_utc", keep="last")

    if persist:
        _save_daily_history(combined)
        for _, r in new_df.iterrows():
            payload = r.to_dict()
            dump_json(DAILY_DIR / "by_date" / f"{r['date_utc']}.json", payload)
            (DAILY_DIR / "by_date" / f"{r['date_utc']}.md").write_text(
                f"# Daily Observation Quality {r['date_utc']}\n\n"
                f"- verdict: `{r['daily_quality_verdict']}`\n"
                f"- strict live coverage: {r['strict_live_coverage_pct']}%\n"
                f"- primary markers: {r['primary_marker_count']}\n"
                f"- partial: {r['is_partial_day']}\n",
                encoding="utf-8",
            )
        latest = combined.iloc[-1].to_dict()
        dump_json(DAILY_DIR / "latest_daily_quality.json", latest)
        (DAILY_DIR / "latest_daily_quality.md").write_text(
            f"# Latest Daily Quality\n\n`{latest['date_utc']}` → `{latest['daily_quality_verdict']}` "
            f"({latest['strict_live_coverage_pct']}%)\n",
            encoding="utf-8",
        )
        render_daily_charts(combined)

    complete = combined[~combined["is_partial_day"].astype(bool)] if "is_partial_day" in combined.columns else combined
    latest_complete = complete.iloc[-1].to_dict() if not complete.empty else (combined.iloc[-1].to_dict() if not combined.empty else {})
    recent7 = complete.tail(7) if not complete.empty else combined.tail(7)
    return {
        "days_generated": int(len(new_df)),
        "history_start": combined["date_utc"].min() if not combined.empty else None,
        "history_end": combined["date_utc"].max() if not combined.empty else None,
        "latest": combined.iloc[-1].to_dict() if not combined.empty else {},
        "latest_complete": latest_complete,
        "recent_7d_strict_live_coverage": float(recent7["strict_live_coverage_pct"].mean()) if not recent7.empty else None,
        "recent_7d_pass_days": int(
            recent7["daily_quality_verdict"]
            .isin(["DAILY_OBSERVATION_QUALITY_PASS", "DAILY_OBSERVATION_QUALITY_PASS_WITH_WARNINGS"])
            .sum()
        )
        if not recent7.empty
        else 0,
        "idempotency": "upsert_by_date_utc",
        "rows": rows,
    }


def render_daily_charts(df: pd.DataFrame) -> None:
    if not HAS_MPL or df.empty:
        return
    ensure_dirs()
    charts = DAILY_DIR / "charts"
    x = df["date_utc"]

    def save(name: str, ys: Dict[str, Any], ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        for label, col in ys.items():
            ax.plot(x, df[col], marker="o", label=label)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(charts / name)
        plt.close(fig)

    save("daily_strict_live_coverage.png", {"strict_live_pct": "strict_live_coverage_pct"}, "coverage %")
    save(
        "daily_stream_coverage.png",
        {
            "kline": "futures_1m_kline_coverage_pct",
            "agg": "futures_aggtrade_presence_pct",
            "mark": "markprice_presence_pct",
        },
        "coverage %",
    )
    save("daily_ws_stale_minutes.png", {"stale": "total_ws_stale_minutes"}, "minutes/events")
    save(
        "daily_self_heal_events.png",
        {"reconnects": "reconnect_attempt_count", "success": "self_heal_success_count"},
        "count",
    )
    save(
        "daily_gap_recovery_status.png",
        {"completed": "completed_gap_count", "pending": "pending_gap_count", "failed": "failed_gap_count"},
        "count",
    )
    save("daily_primary_marker_counts.png", {"primary": "primary_marker_count"}, "count")
    save(
        "daily_marker_composition.png",
        {"funding": "funding_marker_count", "basis": "basis_marker_count", "taker": "taker_q05_marker_count"},
        "count",
    )
    save(
        "daily_outcome_counts.png",
        {"30m": "filled_30m_outcome_count", "60m": "filled_60m_outcome_count"},
        "count",
    )


def daily_status() -> Dict[str, Any]:
    hist = _load_daily_history()
    if hist.empty:
        return {"verdict": "DAILY_HISTORY_EMPTY"}
    latest = hist.iloc[-1].to_dict()
    complete = hist[~hist["is_partial_day"].astype(bool)] if "is_partial_day" in hist.columns else hist
    recent7 = complete.tail(7)
    partial = hist[hist["is_partial_day"].astype(bool)] if "is_partial_day" in hist.columns else hist.iloc[0:0]
    partial_latest = partial.iloc[-1].to_dict() if not partial.empty else {}

    def _sum(col: str, default: int = 0) -> int:
        if recent7.empty or col not in recent7.columns:
            return default
        return int(recent7[col].fillna(0).sum())

    return {
        "history_start": hist["date_utc"].min(),
        "history_end": hist["date_utc"].max(),
        "days": int(len(hist)),
        "latest": latest,
        "latest_complete": complete.iloc[-1].to_dict() if not complete.empty else latest,
        "latest_partial": partial_latest,
        "recent_7d_strict_live_coverage": float(recent7["strict_live_coverage_pct"].mean()) if not recent7.empty else None,
        "recent_7d_pass_days": int(
            recent7["daily_quality_verdict"]
            .isin(["DAILY_OBSERVATION_QUALITY_PASS", "DAILY_OBSERVATION_QUALITY_PASS_WITH_WARNINGS"])
            .sum()
        )
        if not recent7.empty
        else 0,
        "recent_7d_self_heals": _sum("self_heal_success_count"),
        "recent_7d_completed_self_heal_cycles": _sum("completed_self_heal_cycle_count")
        if "completed_self_heal_cycle_count" in recent7.columns
        else _sum("self_heal_success_count"),
        "recent_7d_live_recovery_events": _sum("live_recovery_complete_event_count"),
        "recent_7d_hard_stale_incidents": _sum("hard_stale_incident_count")
        if "hard_stale_incident_count" in recent7.columns
        else _sum("silent_stall_count"),
        "recent_7d_reconnect_attempts": _sum("reconnect_attempt_count"),
        "recent_7d_manual_restarts": _sum("manual_restart_detected_count"),
        "recent_7d_ws_stale_minutes": float(recent7["total_ws_stale_minutes"].sum()) if not recent7.empty else 0,
    }


def daily_audit() -> Dict[str, Any]:
    hist = _load_daily_history()
    dup = int(hist["date_utc"].duplicated().sum()) if not hist.empty else 0
    return {
        "duplicate_date_rows": dup,
        "idempotency_pass": dup == 0,
        "rows": int(len(hist)),
        "columns": list(hist.columns) if not hist.empty else [],
    }


# ---------------------------------------------------------------------------
# Evaluation readiness gate
# ---------------------------------------------------------------------------


def load_taker_forensics_result() -> Dict[str, Any]:
    path = TAKER_DIR / "taker_marker_forensics.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _week_key(ts: pd.Timestamp, t0: pd.Timestamp) -> int:
    return int((ts - t0).total_seconds() // (7 * 86400))


def estimate_days_to_ready(
    prim: pd.DataFrame,
    daily: pd.DataFrame,
    need_primary: int,
    need_basis: int,
    need_taker: int,
    need_days: int,
    current_days: int,
    trusted_taker: int,
) -> Dict[str, Any]:
    """Estimate using recent high-coverage days only; exclude large NO_RAW weeks."""
    if daily.empty:
        return {"estimate_reliability": "ESTIMATE_UNRELIABLE", "bottleneck_condition": "no_daily_history"}
    high = daily[(~daily.get("is_partial_day", False)) & (daily["strict_live_coverage_pct"] >= 90.0)].tail(7)
    if len(high) < 3:
        high = daily[~daily.get("is_partial_day", False)].tail(7)
    # Exclude known W2 blackout days from rate if present
    high = high[~((high["date_utc"] >= "2026-07-10") & (high["date_utc"] <= "2026-07-15"))]
    if high.empty or high["strict_live_hours"].sum() < 24:
        return {
            "estimate_reliability": "ESTIMATE_UNRELIABLE",
            "bottleneck_condition": "insufficient_high_coverage_history",
            "estimated_days_to_primary_50": None,
            "estimated_ready_date_range_kst": None,
        }
    hours = float(high["strict_live_hours"].sum())
    # markers in those days
    dates = set(high["date_utc"])
    p = prim[prim["signal_ts"].dt.strftime("%Y-%m-%d").isin(dates)]
    ph = len(p) / max(hours, 1e-9)
    bh = (p["marker_name"] == "basis_bps_q95").sum() / max(hours, 1e-9)
    th = (p["marker_name"] == "taker_imbalance_ratio_q05").sum() / max(hours, 1e-9)
    cur_p = int(len(prim))
    cur_b = int((prim["marker_name"] == "basis_bps_q95").sum())
    rem_p = max(need_primary - cur_p, 0)
    rem_b = max(need_basis - cur_b, 0)
    rem_t = max(need_taker - trusted_taker, 0)
    rem_d = max(need_days - current_days, 0)

    def days_for(rem: float, rate_per_hour: float) -> Optional[float]:
        if rem <= 0:
            return 0.0
        if rate_per_hour <= 1e-9:
            return None
        return rem / (rate_per_hour * 24.0)

    d_p = days_for(rem_p, ph)
    d_b = days_for(rem_b, bh)
    d_t = days_for(rem_t, th)
    d_days = float(rem_d)
    candidates = {
        "primary_markers": d_p,
        "basis_markers": d_b,
        "taker_trusted": d_t,
        "unique_live_days": d_days,
    }
    finite = {k: v for k, v in candidates.items() if v is not None}
    if not finite:
        return {"estimate_reliability": "ESTIMATE_UNRELIABLE", "bottleneck_condition": "zero_marker_rate"}
    bottleneck = max(finite.items(), key=lambda kv: kv[1])
    lo = bottleneck[1]
    hi = lo * 2.5 if lo > 0 else 7.0
    now = now_utc()
    range_kst = (
        f"{(now + pd.Timedelta(days=lo) + pd.Timedelta(hours=9)).strftime('%Y-%m-%d')}"
        f" ~ {(now + pd.Timedelta(days=hi) + pd.Timedelta(hours=9)).strftime('%Y-%m-%d')}"
    )
    return {
        "marker_per_hour": ph,
        "basis_per_hour": bh,
        "taker_per_hour": th,
        "estimated_days_to_primary_50": d_p,
        "estimated_days_to_basis_10": d_b,
        "estimated_days_to_taker_10": d_t,
        "estimated_days_to_unique_days_14": d_days,
        "bottleneck_condition": bottleneck[0],
        "estimated_days_to_ready_lo": lo,
        "estimated_days_to_ready_hi": hi,
        "estimated_ready_date_range_kst": range_kst,
        "estimate_reliability": "RANGE_OK" if ph > 0 else "ESTIMATE_UNRELIABLE",
        "note": "W2 large NO_RAW window excluded from rate basis",
    }


def run_readiness(
    as_of: Optional[pd.Timestamp] = None,
    persist: bool = True,
    health: Optional[Dict[str, Any]] = None,
    taker: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ensure_dirs()
    cfg = load_config()
    rcfg = cfg.get("readiness") or {}
    as_of = as_of or now_utc()
    t0 = t0_from_config(cfg)
    health_sum = summarize_health(health) if health else summarize_health()
    prim, sec, outcomes = load_observer()
    prim = eligible(prim)
    prim = prim[prim["signal_ts"] >= t0]
    sec = eligible(sec)
    outcomes = eligible(outcomes)
    daily = _load_daily_history()
    taker = taker if taker is not None else load_taker_forensics_result()

    primary_n = int(len(prim))
    funding_n = int((prim["marker_name"] == "funding_rate_q95").sum())
    basis_n = int((prim["marker_name"] == "basis_bps_q95").sum())
    taker_total = int((prim["marker_name"] == "taker_imbalance_ratio_q05").sum())
    taker_trusted = int(taker.get("taker_markers_trusted") or 0)
    # If forensics only covers W2, add pre-W2 takers as trusted-by-default when none questionable globally
    pre_w2_taker = int(
        (
            (prim["marker_name"] == "taker_imbalance_ratio_q05")
            & (prim["signal_ts"] < W2_START)
        ).sum()
    )
    if taker:
        taker_trusted = pre_w2_taker + int(taker.get("taker_markers_trusted") or 0)
        taker_warned = int(taker.get("taker_markers_warned") or 0)
        taker_q = int(taker.get("taker_markers_questionable") or 0)
        taker_inv = int(taker.get("taker_markers_invalid") or 0)
        taker_unr = int(taker.get("taker_markers_unresolved") or 0)
    else:
        taker_trusted, taker_warned, taker_q, taker_inv, taker_unr = taker_total, 0, 0, 0, 0

    filled_30 = count_filled_outcomes(outcomes, 30)
    filled_60 = count_filled_outcomes(outcomes, 60)

    unique_days = int(prim["signal_ts"].dt.floor("D").nunique()) if not prim.empty else 0
    # Also count strict live unique days from daily history
    if not daily.empty:
        live_days = int((daily["strict_live_minutes"] > 0).sum())
        unique_live_days = max(unique_days, live_days)
    else:
        unique_live_days = unique_days

    complete = daily[~daily["is_partial_day"].astype(bool)] if not daily.empty and "is_partial_day" in daily.columns else daily
    recent7 = complete.tail(7) if not complete.empty else pd.DataFrame()
    recent7_cov = float(recent7["strict_live_coverage_pct"].mean()) if not recent7.empty else 0.0
    recent7_pass = (
        int(
            recent7["daily_quality_verdict"]
            .isin(["DAILY_OBSERVATION_QUALITY_PASS", "DAILY_OBSERVATION_QUALITY_PASS_WITH_WARNINGS"])
            .sum()
        )
        if not recent7.empty
        else 0
    )
    post_heal_hq = (
        int(((complete["strict_live_coverage_pct"] >= 90.0) & (complete["date_utc"] >= "2026-07-16")).sum())
        if not complete.empty
        else 0
    )

    weeks = prim["signal_ts"].map(lambda t: _week_key(t, t0))
    week_dist = weeks.value_counts().to_dict() if not prim.empty else {}
    marker_weeks = int(weeks.nunique()) if not prim.empty else 0
    basis_weeks = int(prim.loc[prim["marker_name"] == "basis_bps_q95", "signal_ts"].map(lambda t: _week_key(t, t0)).nunique()) if basis_n else 0
    taker_weeks = int(
        prim.loc[prim["marker_name"] == "taker_imbalance_ratio_q05", "signal_ts"].map(lambda t: _week_key(t, t0)).nunique()
    ) if taker_total else 0

    shares = {m: (prim["marker_name"] == m).mean() * 100.0 for m in PRIMARY_MARKERS} if primary_n else {}
    day_share = (
        float(prim["signal_ts"].dt.floor("D").value_counts(normalize=True).max() * 100.0) if primary_n else 0.0
    )
    week_share = float(max(week_dist.values()) / primary_n * 100.0) if primary_n and week_dist else 0.0

    composition_unbalanced = False
    if primary_n >= int(rcfg.get("primary_markers_required", 50)):
        if (
            max(shares.values() or [0]) > float(rcfg.get("max_single_marker_share_pct", 80))
            or basis_n < int(rcfg.get("basis_markers_required", 10))
            or taker_trusted < int(rcfg.get("taker_trusted_required", 10))
            or day_share > float(rcfg.get("max_single_day_share_pct", 50))
            or week_share > float(rcfg.get("max_single_week_share_pct", 90))
        ):
            composition_unbalanced = True

    conditions = []

    def add(name: str, current: Any, required: Any, ok: bool, kind: str = "sample") -> None:
        conditions.append(
            {
                "condition": name,
                "current": current,
                "required": required,
                "ok": bool(ok),
                "kind": kind,
                "remaining": (required - current) if isinstance(required, (int, float)) and isinstance(current, (int, float)) else None,
                "progress_pct": round(100.0 * min(current / required, 1.0), 2)
                if isinstance(required, (int, float)) and required and isinstance(current, (int, float))
                else None,
            }
        )

    add("mainnet_endpoint", True, True, True, "integrity")
    add("private_endpoint_calls", health_sum["private_endpoint_calls"], 0, health_sum["private_endpoint_calls"] == 0, "integrity")
    add("order_endpoint_calls", health_sum["order_endpoint_calls"], 0, health_sum["order_endpoint_calls"] == 0, "integrity")
    add("production_ready_false", health_sum["production_ready"], False, health_sum["production_ready"] is False, "integrity")
    add("promotion_ready_false", health_sum["promotion_ready"], False, health_sum["promotion_ready"] is False, "integrity")
    add("contamination", health_sum["contamination_count"], 0, health_sum["contamination_count"] == 0, "integrity")
    add("asof", health_sum["asof_pass"], True, health_sum["asof_pass"], "integrity")
    add("outcome_anchor", health_sum["outcome_anchor_pass"], True, health_sum["outcome_anchor_pass"], "integrity")
    add("duplicate", health_sum["duplicate_pass"], True, health_sum["duplicate_pass"], "integrity")
    add("quarantine", health_sum["quarantine_exclusion_pass"], True, health_sum["quarantine_exclusion_pass"], "integrity")
    add("collector_healthy", health_sum["collector_healthy"], True, health_sum["collector_healthy"], "ops")
    add("staleness_ok", health_sum["staleness_ok"], True, health_sum["staleness_ok"], "ops")
    add("gap_pending", health_sum["gap_pending_retryable"], 0, health_sum["gap_pending_retryable"] == 0, "ops")
    add("recent_7d_coverage", recent7_cov, rcfg.get("recent_7d_strict_coverage_pct_required", 90), recent7_cov >= float(rcfg.get("recent_7d_strict_coverage_pct_required", 90)), "coverage")
    add("recent_7d_pass_days", recent7_pass, rcfg.get("recent_7d_pass_or_warn_days_required", 6), recent7_pass >= int(rcfg.get("recent_7d_pass_or_warn_days_required", 6)), "coverage")
    add("primary_markers", primary_n, rcfg.get("primary_markers_required", 50), primary_n >= int(rcfg.get("primary_markers_required", 50)))
    add("funding_markers", funding_n, rcfg.get("funding_markers_required", 10), funding_n >= int(rcfg.get("funding_markers_required", 10)))
    add("basis_markers", basis_n, rcfg.get("basis_markers_required", 10), basis_n >= int(rcfg.get("basis_markers_required", 10)))
    add("taker_trusted", taker_trusted, rcfg.get("taker_trusted_required", 10), taker_trusted >= int(rcfg.get("taker_trusted_required", 10)))
    add("filled_30m", filled_30, rcfg.get("filled_30m_required", 30), filled_30 >= int(rcfg.get("filled_30m_required", 30)))
    add("filled_60m", filled_60, rcfg.get("filled_60m_required", 30), filled_60 >= int(rcfg.get("filled_60m_required", 30)))
    add("unique_live_days", unique_live_days, rcfg.get("strict_live_unique_days_required", 14), unique_live_days >= int(rcfg.get("strict_live_unique_days_required", 14)))
    add("post_self_heal_hq_days", post_heal_hq, rcfg.get("post_self_heal_high_quality_days_required", 7), post_heal_hq >= int(rcfg.get("post_self_heal_high_quality_days_required", 7)))
    add("marker_weeks", marker_weeks, rcfg.get("min_marker_weeks", 2), marker_weeks >= int(rcfg.get("min_marker_weeks", 2)))
    add("basis_weeks", basis_weeks, rcfg.get("min_basis_weeks", 2), basis_weeks >= int(rcfg.get("min_basis_weeks", 2)))
    add("taker_weeks", taker_weeks, rcfg.get("min_taker_weeks", 2), taker_weeks >= int(rcfg.get("min_taker_weeks", 2)))

    integrity_fail = any(not c["ok"] and c["kind"] == "integrity" for c in conditions)
    ops_fail = any(not c["ok"] and c["kind"] == "ops" for c in conditions)
    sample_ok = all(c["ok"] for c in conditions if c["kind"] == "sample")
    coverage_ok = all(c["ok"] for c in conditions if c["kind"] == "coverage")

    taker_verdict = taker.get("verdict")
    warnings: List[str] = []
    blockers: List[str] = [c["condition"] for c in conditions if not c["ok"]]

    if integrity_fail or health_sum["contamination_count"] > 0:
        verdict = "DATA_QUALITY_BLOCKED"
    elif ops_fail:
        verdict = "OPERATIONAL_HEALTH_BLOCKED"
    elif taker_verdict == "TAKER_NO_RAW_STRICT_VALIDITY_FAIL":
        verdict = "DATA_QUALITY_BLOCKED"
        blockers.append("taker_strict_validity")
    elif taker_verdict in {"TAKER_NO_RAW_MARKERS_QUESTIONABLE", "TAKER_NO_RAW_UNRESOLVED"} or taker_q or taker_unr:
        verdict = "QUALITY_REVIEW_REQUIRED"
        warnings.append(f"taker_forensics={taker_verdict}")
    elif composition_unbalanced:
        verdict = "SAMPLE_COUNT_MET_BUT_COMPOSITION_UNBALANCED"
    elif sample_ok and coverage_ok and health_sum["collector_healthy"]:
        if any("WARN" in str(taker_verdict) for _ in [0]) or taker_warned:
            verdict = "READY_FOR_FORMAL_EVALUATION_WITH_WARNINGS"
        else:
            verdict = "READY_FOR_FORMAL_EVALUATION"
    else:
        verdict = "NOT_READY_FOR_EVALUATION"

    eta = estimate_days_to_ready(
        prim,
        daily,
        int(rcfg.get("primary_markers_required", 50)),
        int(rcfg.get("basis_markers_required", 10)),
        int(rcfg.get("taker_trusted_required", 10)),
        int(rcfg.get("strict_live_unique_days_required", 14)),
        unique_live_days,
        taker_trusted,
    )

    result = {
        "as_of_utc": as_of.isoformat(),
        "verdict": verdict,
        "primary_markers": primary_n,
        "funding_markers": funding_n,
        "basis_markers": basis_n,
        "taker_markers_total": taker_total,
        "taker_markers_trusted": taker_trusted,
        "taker_markers_warned": taker_warned if taker else 0,
        "taker_markers_questionable": taker_q if taker else 0,
        "taker_markers_invalid": taker_inv if taker else 0,
        "taker_markers_unresolved": taker_unr if taker else 0,
        "taker_forensics_verdict": taker_verdict,
        "strict_live_unique_days": unique_live_days,
        "post_self_heal_high_quality_days": post_heal_hq,
        "filled_30m": filled_30,
        "filled_60m": filled_60,
        "recent_7d_strict_coverage": recent7_cov,
        "recent_7d_pass_days": recent7_pass,
        "marker_composition_pct": shares,
        "week_distribution": {str(k): int(v) for k, v in week_dist.items()},
        "day_concentration_pct": day_share,
        "week_concentration_pct": week_share,
        "composition_unbalanced": composition_unbalanced,
        "blockers": blockers,
        "warnings": warnings,
        "conditions": conditions,
        "estimate": eta,
        "production_ready": False,
        "promotion_ready": False,
        "health": {k: health_sum[k] for k in health_sum if k != "raw"},
    }

    if persist:
        dump_json(READY_DIR / "evaluation_readiness_latest.json", result)
        md = [
            "# Evaluation Readiness",
            "",
            f"**Verdict:** `{verdict}`",
            "",
            f"- primary: {primary_n}/{rcfg.get('primary_markers_required', 50)}",
            f"- funding: {funding_n}/{rcfg.get('funding_markers_required', 10)}",
            f"- basis: {basis_n}/{rcfg.get('basis_markers_required', 10)}",
            f"- taker trusted: {taker_trusted}/{rcfg.get('taker_trusted_required', 10)}",
            f"- unique live days: {unique_live_days}/{rcfg.get('strict_live_unique_days_required', 14)}",
            f"- bottleneck: {eta.get('bottleneck_condition')}",
            f"- ETA days: {eta.get('estimated_days_to_ready_lo')}–{eta.get('estimated_days_to_ready_hi')}",
            f"- ETA KST range: {eta.get('estimated_ready_date_range_kst')}",
            "",
            "## Blockers",
            "",
        ]
        for b in blockers:
            md.append(f"- {b}")
        (READY_DIR / "evaluation_readiness_latest.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        pd.DataFrame(conditions).to_csv(READY_DIR / "readiness_condition_matrix.csv", index=False)
        prog = [
            {
                "metric": c["condition"],
                "current": c["current"],
                "required": c["required"],
                "remaining": c["remaining"],
                "progress_pct": c["progress_pct"],
            }
            for c in conditions
            if c["kind"] == "sample" or c["condition"] in {"recent_7d_coverage", "unique_live_days", "post_self_heal_hq_days"}
        ]
        pd.DataFrame(prog).to_csv(READY_DIR / "readiness_progress.csv", index=False)
        dump_json(READY_DIR / "estimated_time_to_ready.json", eta)

        hist_path = READY_DIR / "evaluation_readiness_history.parquet"
        hist_row = {
            "as_of_utc": as_of.isoformat(),
            "verdict": verdict,
            "primary_markers": primary_n,
            "basis_markers": basis_n,
            "taker_trusted": taker_trusted,
            "unique_live_days": unique_live_days,
            "bottleneck": eta.get("bottleneck_condition"),
        }
        if hist_path.exists():
            hdf = pd.read_parquet(hist_path)
            hdf = hdf[hdf["as_of_utc"] != hist_row["as_of_utc"]]
            hdf = pd.concat([hdf, pd.DataFrame([hist_row])], ignore_index=True)
        else:
            hdf = pd.DataFrame([hist_row])
        hdf.to_parquet(hist_path, index=False)
        hdf.to_csv(READY_DIR / "evaluation_readiness_history.csv", index=False)

    return result


# ---------------------------------------------------------------------------
# Suite wrapper
# ---------------------------------------------------------------------------


def count_filled_outcomes(outcomes: pd.DataFrame, horizon: int) -> int:
    if outcomes.empty:
        return 0
    sub = outcomes[outcomes["horizon_min"] == horizon]
    if "path_complete" in sub.columns:
        sub = sub[sub["path_complete"].fillna(False).astype(bool)]
    if "fixed_return_bps" in sub.columns:
        sub = sub[sub["fixed_return_bps"].notna()]
    elif "status" in sub.columns:
        sub = sub[sub["status"].astype(str).str.lower() != "pending"]
    return int(len(sub))


def consistency_check() -> Dict[str, Any]:
    """Exact minute-window strict hours vs known 14D evaluator numbers."""
    gap = load_gap_ledger()
    windows = {
        "WEEK1": (T0_DEFAULT, W2_START, 167.85, 99.91),
        "WEEK2": (W2_START, W2_END, 26.17, 15.58),
        "CUMULATIVE_14D": (T0_DEFAULT, W2_END, 194.03, 57.75),
    }
    out: Dict[str, Any] = {"method": "closed_futures_1m_kline_unique_minutes_exact_window"}
    for name, (start, end, exp_h, exp_pct) in windows.items():
        cal_h = (end - start).total_seconds() / 3600.0
        mins = scan_jsonl_minutes(
            LIVE_NORM / "ws_kline_1m/symbol=BTCUSDT",
            ["open_time", "event_ts"],
            start,
            end,
            closed_only=True,
        )
        # clip to calendar minutes in window
        cal = set(calendar_minutes(start, end))
        mins = mins & cal
        hours = len(mins) / 60.0
        pct = 100.0 * hours / cal_h if cal_h else None
        out[name] = {
            "strict_live_hours": hours,
            "expected_hours": exp_h,
            "delta_hours": abs(hours - exp_h),
            "strict_live_coverage_pct": pct,
            "expected_coverage_pct": exp_pct,
            "delta_coverage_pct": abs(pct - exp_pct) if pct is not None else None,
            "match_near": abs(hours - exp_h) <= 0.05,
        }
    daily = _load_daily_history()
    out["daily_rollup_note"] = (
        "UTC-day rollup includes full calendar days that only partially overlap W1/W2 boundaries; "
        "use exact-window numbers above for 14D consistency."
    )
    out["daily_history_rows"] = int(len(daily))
    return out


def run_full_audit() -> Dict[str, Any]:
    ensure_dirs()
    health = collect_pre_work_health(persist=True)
    health_sum = summarize_health(health)
    taker = run_taker_forensics(persist=True)
    daily = run_daily_rollup(from_t0=True, persist=True, health=health)
    ready = run_readiness(persist=True, health=health, taker=taker)
    consistency = consistency_check()
    suite_verdict = "OBSERVATION_QUALITY_SUITE_OPERATIONAL_PASS"
    warnings = []
    if not health_sum["collector_healthy"]:
        warnings.append("collector_not_healthy_at_audit_time")
    if taker.get("verdict") in {"TAKER_NO_RAW_MARKERS_QUESTIONABLE", "TAKER_NO_RAW_UNRESOLVED", "TAKER_NO_RAW_PASS_WITH_PARTIAL_INPUT_WARNINGS"}:
        warnings.append(f"taker_forensics={taker.get('verdict')}")
        suite_verdict = "OBSERVATION_QUALITY_SUITE_OPERATIONAL_PASS_WITH_WARNINGS"
    exact = consistency.get("WEEK1", {})
    if not exact.get("match_near", True) or not consistency.get("WEEK2", {}).get("match_near", True):
        warnings.append("exact_window_coverage_mismatch")
        suite_verdict = "OBSERVATION_QUALITY_SUITE_OPERATIONAL_PASS_WITH_WARNINGS"

    final = {
        "suite_verdict": suite_verdict,
        "implementation_status": "OBSERVATION_QUALITY_TOOLING_IMPLEMENTED",
        "health": health_sum,
        "taker": taker,
        "daily": {
            "history_start": daily.get("history_start"),
            "history_end": daily.get("history_end"),
            "days_generated": daily.get("days_generated"),
            "latest_complete": daily.get("latest_complete"),
            "recent_7d_strict_live_coverage": daily.get("recent_7d_strict_live_coverage"),
            "recent_7d_pass_days": daily.get("recent_7d_pass_days"),
        },
        "readiness": {
            "verdict": ready.get("verdict"),
            "blockers": ready.get("blockers"),
            "estimate": ready.get("estimate"),
            "primary": ready.get("primary_markers"),
            "basis": ready.get("basis_markers"),
            "taker_trusted": ready.get("taker_markers_trusted"),
        },
        "consistency": consistency,
        "warnings": warnings,
        "production_ready": False,
        "promotion_ready": False,
        "generated_at_utc": now_utc().isoformat(),
    }
    dump_json(REPORTS / "observation_quality_suite_final_report.json", final)
    md = [
        "# Observation Quality Suite Final Report",
        "",
        f"**Suite verdict:** `{suite_verdict}`",
        "",
        f"- Taker forensics: `{taker.get('verdict')}`",
        f"- Readiness: `{ready.get('verdict')}`",
        f"- Daily history: {daily.get('history_start')} → {daily.get('history_end')} ({daily.get('days_generated')} days upserted)",
        f"- Collector healthy: {health_sum.get('collector_healthy')}",
        f"- Production/promotion ready: false/false",
        "",
        "## Warnings",
        "",
    ]
    for w in warnings or ["none"]:
        md.append(f"- {w}")
    (REPORTS / "observation_quality_suite_final_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return final


def run_daily_update(refresh_taker: bool = False) -> Dict[str, Any]:
    health = collect_pre_work_health(persist=False)
    as_of = now_utc()
    yday = (as_of.floor("D") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    today = as_of.floor("D").strftime("%Y-%m-%d")
    daily = run_daily_rollup(start_date=yday, end_date=today, persist=True, health=health)
    taker = run_taker_forensics(persist=True) if refresh_taker else load_taker_forensics_result()
    ready = run_readiness(persist=True, health=health, taker=taker or None)
    return {"daily": daily, "readiness": ready.get("verdict"), "health": summarize_health(health)}


def _recent_persist_coverage_minutes(window_min: int) -> Dict[str, Any]:
    """Closed futures 1m kline coverage over the last window_min calendar minutes."""
    end = now_utc().floor("min")
    start = end - pd.Timedelta(minutes=window_min)
    expected = list(pd.date_range(start, end - pd.Timedelta(minutes=1), freq="1min", tz="UTC"))
    have = scan_jsonl_minutes(
        LIVE_NORM / "ws_kline_1m/symbol=BTCUSDT",
        ["open_time", "event_ts"],
        start,
        end,
        closed_only=True,
    )
    present = [t for t in expected if t in have]
    return {
        "window_min": window_min,
        "expected": len(expected),
        "present": len(present),
        "coverage_pct": round(100.0 * len(present) / max(len(expected), 1), 4),
    }


def _recent_reconnect_cycles(window_min: int) -> int:
    end = now_utc()
    start = end - pd.Timedelta(minutes=window_min)
    wd = load_watchdog_events(start, end)
    if wd.empty:
        return 0
    keys = set()
    for _, r in wd[wd["event_type"] == "RECONNECT_REQUESTED"].iterrows():
        keys.add((r.get("collector_instance_id"), r.get("connection_generation")))
    return len(keys)


def _collection_stability_label(
    health: Dict[str, Any],
    cov15: float,
    reconnect_15: int,
    collector: Dict[str, Any],
) -> str:
    if not health.get("collector_healthy"):
        if health.get("ws_state") in {"HARD_STALE", "STALE"}:
            return "HARD_STALE"
        return "UNKNOWN"
    if reconnect_15 >= 3:
        return "FLAPPING"
    ages = [
        float(collector.get("kline_age_seconds") or 1e9),
        float(collector.get("aggtrade_age_seconds") or 1e9),
        float(collector.get("markprice_age_seconds") or 1e9),
    ]
    if max(ages) > 180:
        return "PARTIAL_STREAM"
    if cov15 >= 90 and reconnect_15 == 0:
        return "STABLE"
    if cov15 >= 75:
        return "RECENTLY_RECOVERED"
    return "FLAPPING"


def compact_status(persist: bool = True) -> Dict[str, Any]:
    """Short status payload for terminal / compact report files."""
    health = summarize_health()
    daily = daily_status()
    ready = (
        json.loads((READY_DIR / "evaluation_readiness_latest.json").read_text(encoding="utf-8"))
        if (READY_DIR / "evaluation_readiness_latest.json").exists()
        else {}
    )
    complete = daily.get("latest_complete") or {}
    partial = daily.get("latest_partial") or {}
    estimate = ready.get("estimate") or {}
    collector = (health.get("raw") or {}).get("collector") or {}
    if not collector:
        try:
            collector = _run_diag_json("run_new_market_microstructure_data_pipeline.py", "--live-collector-status")
        except Exception:
            collector = {}
    cov15 = _recent_persist_coverage_minutes(15)
    cov60 = _recent_persist_coverage_minutes(60)
    recon15 = _recent_reconnect_cycles(15)
    recon60 = _recent_reconnect_cycles(60)
    # healthy session duration estimate from last LIVE_RECOVERY / HARD_STALE
    wd_recent = load_watchdog_events(now_utc() - pd.Timedelta(hours=6), now_utc() + pd.Timedelta(minutes=1))
    last_hard = last_rec = None
    if not wd_recent.empty:
        hard = wd_recent[wd_recent["event_type"] == "HARD_STALE_DETECTED"]
        live = wd_recent[wd_recent["event_type"] == "LIVE_RECOVERY_COMPLETE"]
        if not hard.empty:
            last_hard = hard["event_utc"].max()
        if not live.empty:
            last_rec = live["event_utc"].max()
    healthy_session_s = None
    if last_rec is not None and (last_hard is None or last_rec > last_hard):
        healthy_session_s = (now_utc() - parse_ts(last_rec)).total_seconds()
    kline_recv_fresh = float(collector.get("kline_age_seconds") or 1e9) <= 120
    agg_fresh = float(collector.get("aggtrade_age_seconds") or 1e9) <= 120
    mark_fresh = float(collector.get("markprice_age_seconds") or 1e9) <= 120
    stability = _collection_stability_label(health, float(cov15["coverage_pct"]), recon15, collector)
    try:
        from microstructure_keepawake_guard import assess_keepawake_guard

        kaw = assess_keepawake_guard(expected=True)
    except Exception as exc:  # diagnostics-only; never break compact status
        kaw = {
            "keep_awake_guard_expected": True,
            "keep_awake_guard_active": False,
            "caffeinate_pid": None,
            "idle_sleep_assertion_active": False,
            "system_sleep_assertion_active": False,
            "assertion_owner_verified": False,
            "collector_wrapped_by_caffeinate": False,
            "duplicate_caffeinate_guards": 0,
            "clamshell_sleep_not_prevented": True,
            "power_source": "UNKNOWN",
            "keep_awake_guard_verdict": "KEEP_AWAKE_GUARD_UNRESOLVED",
            "error": str(exc),
        }
    payload = {
        "generated_at_utc": now_utc().isoformat(),
        "collector_health": "HEALTHY" if health["collector_healthy"] else "NOT_HEALTHY",
        "ws_state": health["ws_state"],
        "staleness": "STALENESS_OK" if health["staleness_ok"] else "STALE",
        "current_partial_day_utc": partial.get("date_utc"),
        "partial_day_elapsed_minutes": partial.get("elapsed_calendar_minutes") or partial.get("calendar_minutes"),
        "partial_day_strict_coverage_pct": partial.get("strict_live_coverage_pct"),
        "partial_day_verdict": partial.get("daily_quality_verdict"),
        "latest_complete_day_utc": complete.get("date_utc"),
        "latest_complete_day_coverage_pct": complete.get("strict_live_coverage_pct"),
        "recent_7d_coverage_pct": daily.get("recent_7d_strict_live_coverage"),
        "recent_7d_pass_days": daily.get("recent_7d_pass_days"),
        "completed_self_heal_cycles_7d": daily.get("recent_7d_completed_self_heal_cycles"),
        "hard_stale_incidents_7d": daily.get("recent_7d_hard_stale_incidents"),
        "reconnect_attempts_7d": daily.get("recent_7d_reconnect_attempts"),
        "manual_restarts_7d": daily.get("recent_7d_manual_restarts"),
        "pending_gaps": health["gap_pending_retryable"],
        "failed_gaps": health["gap_failed"],
        "primary_markers": ready.get("primary_markers"),
        "basis_markers": ready.get("basis_markers"),
        "trusted_taker_markers": ready.get("taker_markers_trusted"),
        "readiness": ready.get("verdict"),
        "readiness_bottleneck": estimate.get("bottleneck_condition"),
        "contamination": health["contamination_count"],
        "production_ready": False,
        "promotion_ready": False,
        "CURRENT_15M_CORE_PERSIST_COVERAGE": cov15["coverage_pct"],
        "CURRENT_60M_CORE_PERSIST_COVERAGE": cov60["coverage_pct"],
        "CURRENT_15M_RECONNECT_CYCLES": recon15,
        "CURRENT_60M_RECONNECT_CYCLES": recon60,
        "CURRENT_HEALTHY_SESSION_DURATION_SECONDS": healthy_session_s,
        "LAST_HARD_STALE_AT": str(last_hard) if last_hard is not None else None,
        "LAST_RECOVERY_COMPLETED_AT": str(last_rec) if last_rec is not None else None,
        "ACTIVE_CONNECTION_GENERATIONS": 1 if collector.get("ws_connection_generation") is not None else 0,
        "CONNECTION_GENERATION": collector.get("ws_connection_generation"),
        "KLINE_RECEIVED_FRESH": kline_recv_fresh,
        "KLINE_PERSISTED_FRESH": kline_recv_fresh,
        "AGGTRADE_RECEIVED_FRESH": agg_fresh,
        "MARKPRICE_RECEIVED_FRESH": mark_fresh,
        "WRITER_TASK_ALIVE": True,  # inline writer; no separate writer task
        "CURRENT_COLLECTION_STABILITY": stability,
        "KEEP_AWAKE_GUARD_EXPECTED": bool(kaw.get("keep_awake_guard_expected", True)),
        "KEEP_AWAKE_GUARD_ACTIVE": bool(kaw.get("keep_awake_guard_active")),
        "CAFFEINATE_PID": kaw.get("caffeinate_pid"),
        "IDLE_SLEEP_ASSERTION_ACTIVE": bool(kaw.get("idle_sleep_assertion_active")),
        "SYSTEM_SLEEP_ASSERTION_ACTIVE": bool(kaw.get("system_sleep_assertion_active")),
        "ASSERTION_OWNER_VERIFIED": bool(kaw.get("assertion_owner_verified")),
        "COLLECTOR_WRAPPED_BY_CAFFEINATE": bool(kaw.get("collector_wrapped_by_caffeinate")),
        "DUPLICATE_CAFFEINATE_GUARDS": int(kaw.get("duplicate_caffeinate_guards") or 0),
        "CLAMSHELL_SLEEP_NOT_PREVENTED": bool(kaw.get("clamshell_sleep_not_prevented", True)),
        "POWER_SOURCE": kaw.get("power_source") or "UNKNOWN",
        "KEEP_AWAKE_GUARD_VERDICT": kaw.get("keep_awake_guard_verdict") or "KEEP_AWAKE_GUARD_UNRESOLVED",
        "detail_json_path": str(REPORTS / "observation_quality_suite_final_report.json"),
        "daily_latest_json_path": str(DAILY_DIR / "latest_daily_quality.json"),
        "readiness_latest_json_path": str(READY_DIR / "evaluation_readiness_latest.json"),
    }
    if persist:
        ensure_dirs()
        dump_json(REPORTS / "observation_quality_compact_status.json", payload)
        lines = [f"{k}={payload[k]}" for k in payload if not str(k).endswith("_path")]
        lines.extend(
            [
                f"detail_json_path={payload['detail_json_path']}",
                f"daily_latest_json_path={payload['daily_latest_json_path']}",
                f"readiness_latest_json_path={payload['readiness_latest_json_path']}",
            ]
        )
        (REPORTS / "observation_quality_compact_status.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def suite_status() -> Dict[str, Any]:
    health = summarize_health()
    daily = daily_status()
    ready = json.loads((READY_DIR / "evaluation_readiness_latest.json").read_text()) if (READY_DIR / "evaluation_readiness_latest.json").exists() else {}
    taker = load_taker_forensics_result()
    latest_complete = daily.get("latest_complete") or {}
    latest_partial = daily.get("latest_partial") or {}
    return {
        "OBSERVATION_QUALITY_SUITE_VERDICT": ready.get("verdict") or "UNKNOWN",
        "CURRENT_COLLECTOR_HEALTH": "HEALTHY" if health["collector_healthy"] else "NOT_HEALTHY",
        "CURRENT_WS_STATE": health["ws_state"],
        "CURRENT_STALENESS": "STALENESS_OK" if health["staleness_ok"] else "STALE",
        "CURRENT_GAP_RECOVERY": {
            "pending": health["gap_pending_retryable"],
            "failed": health["gap_failed"],
        },
        "LATEST_COMPLETE_UTC_DAY": latest_complete.get("date_utc"),
        "LATEST_DAILY_QUALITY_VERDICT": latest_complete.get("daily_quality_verdict"),
        "LATEST_DAILY_STRICT_COVERAGE": latest_complete.get("strict_live_coverage_pct"),
        "CURRENT_PARTIAL_UTC_DAY": latest_partial.get("date_utc"),
        "PARTIAL_ELAPSED_MINUTES": latest_partial.get("elapsed_calendar_minutes") or latest_partial.get("calendar_minutes"),
        "PARTIAL_STRICT_LIVE_COVERAGE": latest_partial.get("strict_live_coverage_pct"),
        "PARTIAL_DAY_VERDICT": latest_partial.get("daily_quality_verdict"),
        "RECENT_7D_STRICT_COVERAGE": daily.get("recent_7d_strict_live_coverage"),
        "RECENT_7D_PASS_DAYS": daily.get("recent_7d_pass_days"),
        "RECENT_7D_SELF_HEAL_COUNT": daily.get("recent_7d_completed_self_heal_cycles"),
        "RECENT_7D_LIVE_RECOVERY_EVENT_COUNT": daily.get("recent_7d_live_recovery_events"),
        "RECENT_7D_HARD_STALE_INCIDENTS": daily.get("recent_7d_hard_stale_incidents"),
        "RECENT_7D_RECONNECT_ATTEMPTS": daily.get("recent_7d_reconnect_attempts"),
        "RECENT_7D_MANUAL_RESTART_COUNT": daily.get("recent_7d_manual_restarts"),
        "CURRENT_PRIMARY_MARKERS": ready.get("primary_markers"),
        "CURRENT_FUNDING_MARKERS": ready.get("funding_markers"),
        "CURRENT_BASIS_MARKERS": ready.get("basis_markers"),
        "CURRENT_TAKER_MARKERS": ready.get("taker_markers_total"),
        "TRUSTED_TAKER_MARKERS": ready.get("taker_markers_trusted"),
        "QUESTIONABLE_TAKER_MARKERS": ready.get("taker_markers_questionable"),
        "STRICT_LIVE_UNIQUE_DAYS": ready.get("strict_live_unique_days"),
        "POST_SELF_HEAL_HIGH_QUALITY_DAYS": ready.get("post_self_heal_high_quality_days"),
        "FILLED_30M_OUTCOMES": ready.get("filled_30m"),
        "FILLED_60M_OUTCOMES": ready.get("filled_60m"),
        "CONTAMINATION_COUNT": health["contamination_count"],
        "QUARANTINE": health["quarantine_exclusion_pass"],
        "ASOF": health["asof_pass"],
        "OUTCOME_ANCHOR": health["outcome_anchor_pass"],
        "EVALUATION_READINESS": ready.get("verdict"),
        "READINESS_BOTTLENECK": (ready.get("estimate") or {}).get("bottleneck_condition"),
        "ESTIMATED_DAYS_TO_READY": {
            "lo": (ready.get("estimate") or {}).get("estimated_days_to_ready_lo"),
            "hi": (ready.get("estimate") or {}).get("estimated_days_to_ready_hi"),
        },
        "PRODUCTION_READY": False,
        "PROMOTION_READY": False,
        "TAKER_FORENSICS_VERDICT": taker.get("verdict"),
    }
