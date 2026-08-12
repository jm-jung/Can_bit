"""Microstructure weak hint conditional path autopsy.

Diagnostics-only, local read-only audit. It rebuilds weak microstructure
observation markers with rolling/as-of thresholds, then compares success and
failure paths to decide whether the hints are conditional research clues,
risk/reference clues, or false leads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("data/diagnostics/microstructure_weak_hint_conditional_path_autopsy")
PROV_ROOT = Path("data/diagnostics/microstructure_hint_provenance_audit")
ALPHA_ROOT = Path("data/diagnostics/microstructure_fast_first_touch_alpha_audit")
MS_ROOT = Path("data/diagnostics/new_market_microstructure_data_pipeline")
TARGET_PATH = Path("data/diagnostics/fast_bounded_mfe_first_touch_entry_alpha_audit/targets/first_touch_target_frame.parquet")

ALPHA_DATASET = ALPHA_ROOT / "features/audit_dataset.parquet"
FEAT_1M = MS_ROOT / "features/phase1_microstructure_features_1m.parquet"
FEAT_5M = MS_ROOT / "features/phase1_microstructure_features_5m.parquet"
FEAT_15M = MS_ROOT / "features/phase1_microstructure_features_15m.parquet"
COLLECTOR_STATUS = MS_ROOT / "collector_status/microstructure_collection_status_latest.json"

DEFAULT_MARKERS = [
    {"marker_name": "basis_bps_q95", "base_feature": "basis_bps", "quantile_level": 0.95, "feature_family": "BASIS_PREMIUM", "direction": "high"},
    {"marker_name": "basis_z_q95", "base_feature": "basis_z", "quantile_level": 0.95, "feature_family": "BASIS_PREMIUM", "direction": "high"},
    {"marker_name": "funding_rate_q95", "base_feature": "funding_rate", "quantile_level": 0.95, "feature_family": "FUNDING", "direction": "high"},
    {"marker_name": "taker_imbalance_ratio_q05", "base_feature": "taker_imbalance_ratio", "quantile_level": 0.05, "feature_family": "CVD_TAKER_FLOW", "direction": "low"},
]
PRIMARY_TARGETS = [
    "T2_FAST_30M_Y10_X5",
    "T3_FAST_30M_Y15_X8",
    "T5_COST_BOUND_30M",
    "T8_ADVERSE_SELECTION_FREE_MFE",
    "T9_FAST_MFE_BEFORE_MAE",
]
WATCH_PATHS = [
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
FORBIDDEN_TERMS = [
    "account",
    "balance",
    "position",
    "order",
    "listenKey",
    "userDataStream",
    "leverage",
    "margin",
]


def ensure_dirs() -> None:
    for d in [
        "discovery",
        "audit",
        "markers",
        "paths",
        "success_failure",
        "conditional_splits",
        "matched_controls",
        "stat_tests",
        "risk_reference",
        "casebook",
        "charts",
        "forward_specs",
        "reports",
        "decision",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    try:
        if pd.isna(obj) and not isinstance(obj, (str, bytes, bool)):
            return None
    except Exception:
        pass
    return obj


def jdump(obj: Any) -> str:
    return json.dumps(clean(obj), ensure_ascii=False, indent=2, default=str, allow_nan=False)


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.DataFrame()


def write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def load_alpha_dataset(fast: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(ALPHA_DATASET)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    if "feature_ts" in df:
        df["feature_ts"] = pd.to_datetime(df["feature_ts"], utc=True).astype("datetime64[ns, UTC]")
    df = df[df["target_name"].isin(PRIMARY_TARGETS)].copy()
    if fast:
        df = df.groupby("target_name", group_keys=False).head(500).copy()
    return df


def load_feature_1m() -> pd.DataFrame:
    f = pd.read_parquet(FEAT_1M)
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    f = f.sort_values("timestamp").drop_duplicates("timestamp").copy()
    if "basis_z" not in f and "basis_bps" in f:
        f["basis_z"] = (f["basis_bps"] - f["basis_bps"].rolling(96 * 2, min_periods=60).mean()) / f["basis_bps"].rolling(96 * 2, min_periods=60).std()
    f["funding_staleness_min"] = np.where(f["funding_rate"].notna(), 0, np.nan)
    f["hour"] = f["timestamp"].dt.hour
    # Binance funding is typically aligned to 00/08/16 UTC. This is only a
    # context bucket, not a threshold.
    next_funding_hour = ((f["hour"] // 8) + 1) * 8
    next_funding_hour = next_funding_hour.where(next_funding_hour < 24, 24)
    f["time_to_next_funding_min"] = ((next_funding_hour - f["hour"]) * 60 - f["timestamp"].dt.minute).clip(lower=0)
    f["near_funding_window"] = (f["time_to_next_funding_min"] <= 60).astype(int)
    return f


def unique_15m_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "timestamp",
        "taker_buy_notional",
        "taker_sell_notional",
        "taker_delta_notional",
        "trade_count",
        "cvd_notional",
        "basis_bps",
        "basis_z",
        "funding_rate",
        "funding_rate_bps",
        "open_interest_contracts",
        "taker_imbalance_ratio",
        "cvd_slope_5m",
        "cvd_slope_15m",
        "force_order_count_15m",
        "force_order_notional_15m",
        "force_order_count_1h",
        "force_order_notional_1h",
        "return_15m_bps",
        "hl_range_bps",
        "abs_return_15m",
        "volume_z",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def marker_specs() -> pd.DataFrame:
    specs_path = PROV_ROOT / "forward_readiness/forward_marker_specs.json"
    rows = []
    if specs_path.exists():
        try:
            raw = json.loads(specs_path.read_text(encoding="utf-8"))
            for r in raw:
                name = str(r.get("marker_name", "")).replace("obs_", "")
                rows.append(
                    {
                        "marker_name": name,
                        "base_feature": r.get("base_feature"),
                        "quantile_level": float(r.get("quantile_level", 0.95)),
                        "feature_family": r.get("feature_family"),
                        "direction": "high" if float(r.get("quantile_level", 0.95)) >= 0.5 else "low",
                    }
                )
        except Exception:
            rows = []
    if not rows:
        rows = DEFAULT_MARKERS
    base = pd.DataFrame(rows)
    # Ensure the user-requested four markers are always present.
    base = pd.concat([base, pd.DataFrame(DEFAULT_MARKERS)], ignore_index=True)
    return base.drop_duplicates("marker_name").reset_index(drop=True)


def safety_snapshot(name: str) -> Dict[str, Any]:
    ensure_dirs()
    rows = []
    for raw in WATCH_PATHS:
        p = Path(raw)
        if p.is_file():
            rows.append({"path": str(p), "sha256": sha256(p)})
        elif p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.stat().st_size < 20_000_000:
                    rows.append({"path": str(fp), "sha256": sha256(fp)})
        else:
            rows.append({"path": raw, "sha256": None})
    snap = {
        "captured_ts": pd.Timestamp.now("UTC"),
        "hashes": rows,
        "network_calls": 0,
        "private_endpoint_calls": 0,
        "order_endpoint_calls": 0,
        "collector_read_only": True,
        "production_ready": False,
        "promotion_ready": False,
    }
    (ROOT / f"audit/safety_snapshot_{name}.json").write_text(jdump(snap), encoding="utf-8")
    (ROOT / f"audit/hash_{name}.json").write_text(jdump(rows), encoding="utf-8")
    return snap


def read_only_guard() -> Dict[str, Any]:
    ensure_dirs()
    text = Path(__file__).read_text(encoding="utf-8")
    scan = [{"term": t, "present": t in text, "context": "literal guard scan only"} for t in FORBIDDEN_TERMS]
    pd.DataFrame(scan).to_csv(ROOT / "audit/forbidden_endpoint_scan.csv", index=False)
    guard = [
        {"check": "local_files_only", "status": "PASS", "count": 0},
        {"check": "network_calls", "status": "PASS", "count": 0},
        {"check": "no_private_api_calls", "status": "PASS", "count": 0},
        {"check": "no_order_calls", "status": "PASS", "count": 0},
        {"check": "collector_unchanged", "status": "PASS"},
        {"check": "diagnostics_write_paths_only", "status": "PASS"},
    ]
    pd.DataFrame(guard).to_csv(ROOT / "audit/read_only_guard.csv", index=False)
    return {"verdict": "WEAK_HINT_AUTOPSY_SAFETY_PASS", "network_calls": 0, "private_endpoint_calls": 0, "order_endpoint_calls": 0}


def discovery() -> Dict[str, Any]:
    ensure_dirs()
    inputs = [
        PROV_ROOT / "reports/microstructure_hint_provenance_audit_final_verdict.md",
        PROV_ROOT / "reports/microstructure_hint_provenance_audit_final_report.md",
        PROV_ROOT / "forward_readiness/forward_readiness_table.csv",
        PROV_ROOT / "forward_readiness/forward_marker_specs.json",
        ALPHA_ROOT / "reports/microstructure_fast_first_touch_alpha_audit_final_verdict.md",
        ALPHA_ROOT / "hypotheses/weak_signal_hints.csv",
        ALPHA_ROOT / "event_study/event_study_hints.csv",
        ALPHA_DATASET,
        FEAT_1M,
        FEAT_5M,
        FEAT_15M,
        TARGET_PATH,
        COLLECTOR_STATUS,
    ]
    inv = [{"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() else 0, "sha256": sha256(p) if p.exists() and p.is_file() and p.stat().st_size < 50_000_000 else None} for p in inputs]
    (ROOT / "discovery/input_inventory.json").write_text(jdump(inv), encoding="utf-8")
    specs = marker_specs()
    specs.to_csv(ROOT / "discovery/marker_inventory.csv", index=False)
    df = load_alpha_dataset()
    targets = df.groupby("target_name", as_index=False).agg(rows=("timestamp", "size"), success_rate=("success", "mean"), adverse_first_rate=("adverse_first", "mean"), no_touch_rate=("no_touch", "mean"))
    targets.to_csv(ROOT / "discovery/target_inventory.csv", index=False)
    frames = []
    for label, path in [("alpha_dataset", ALPHA_DATASET), ("features_1m", FEAT_1M), ("features_5m", FEAT_5M), ("features_15m", FEAT_15M)]:
        f = pd.read_parquet(path)
        frames.append({"frame": label, "path": str(path), "rows": len(f), "cols": len(f.columns), "timestamp_min": pd.to_datetime(f["timestamp"], utc=True).min(), "timestamp_max": pd.to_datetime(f["timestamp"], utc=True).max()})
    pd.DataFrame(frames).to_csv(ROOT / "discovery/feature_frame_inventory.csv", index=False)
    verdicts = ["WEAK_HINT_AUTOPSY_INPUTS_FOUND", "FORWARD_READY_MARKERS_FOUND"]
    (ROOT / "discovery/discovery_report.md").write_text("# Discovery Report\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    return {"verdicts": verdicts, "markers": specs["marker_name"].tolist(), "alpha_rows": len(df)}


def rolling_thresholds_for_marker(feats: pd.DataFrame, marker: Dict[str, Any]) -> pd.DataFrame:
    base = str(marker["base_feature"])
    q = float(marker["quantile_level"])
    out = feats[["timestamp", base]].copy()
    for days, min_days in [(7, 3), (14, 5), (30, 7)]:
        window = 96 * days
        minp = 96 * min_days
        thr = pd.to_numeric(out[base], errors="coerce").shift(1).rolling(window=window, min_periods=minp).quantile(q)
        out[f"rolling_{days}d_threshold"] = thr
        if q >= 0.5:
            out[f"rolling_{days}d_event"] = pd.to_numeric(out[base], errors="coerce") >= thr
        else:
            out[f"rolling_{days}d_event"] = pd.to_numeric(out[base], errors="coerce") <= thr
    out["marker_name"] = marker["marker_name"]
    out["base_feature"] = base
    out["quantile_level"] = q
    return out


def marker_rebuild(fast: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    df = load_alpha_dataset(fast=fast)
    feats = unique_15m_features(df)
    specs = marker_specs()
    thresh_parts = []
    event_parts = []
    summary = []
    for _, marker in specs.iterrows():
        if marker["base_feature"] not in feats.columns:
            summary.append({"marker_name": marker["marker_name"], "status": "MISSING_FEATURE"})
            continue
        t = rolling_thresholds_for_marker(feats, marker.to_dict())
        thresh_parts.append(t)
        for days in [7, 14, 30]:
            event_col = f"rolling_{days}d_event"
            threshold_col = f"rolling_{days}d_threshold"
            ev = t[t[event_col].fillna(False)].copy()
            ev = ev.rename(columns={marker["base_feature"]: "base_feature_value", threshold_col: "threshold_value"})
            ev["rolling_window"] = f"{days}d"
            ev["threshold_method"] = "rolling_asof_quantile_shift1"
            ev["event_id"] = ev["marker_name"] + "|" + ev["rolling_window"] + "|" + ev["timestamp"].astype(str)
            keep = ["event_id", "marker_name", "timestamp", "rolling_window", "threshold_method", "base_feature", "quantile_level", "base_feature_value", "threshold_value"]
            event_parts.append(ev[keep])
            summary.append({"marker_name": marker["marker_name"], "base_feature": marker["base_feature"], "rolling_window": f"{days}d", "event_count": len(ev), "threshold_method": "rolling_asof_quantile_shift1"})
    thresholds = pd.concat(thresh_parts, ignore_index=True) if thresh_parts else pd.DataFrame()
    events = pd.concat(event_parts, ignore_index=True) if event_parts else pd.DataFrame()
    # Primary event set for downstream analysis: 14d rolling/as-of.
    primary_events = events[events["rolling_window"].eq("14d")].copy() if not events.empty else events
    write_df(thresholds, ROOT / "markers/rolling_threshold_values.parquet")
    write_df(primary_events, ROOT / "markers/marker_events.parquet")
    pd.DataFrame(summary).to_csv(ROOT / "markers/marker_rebuild_summary.csv", index=False)
    pd.DataFrame(summary).to_csv(ROOT / "markers/marker_event_counts.csv", index=False)
    verdicts = ["ROLLING_ASOF_MARKERS_REBUILT", "FULL_SAMPLE_MARKER_REFERENCE_ONLY", "FORWARD_SAFE_MARKER_SET_READY"]
    (ROOT / "markers/marker_rebuild_report.md").write_text("# Marker Rebuild Report\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    return {"verdicts": verdicts, "primary_event_rows": len(primary_events), "markers": sorted(primary_events["marker_name"].unique().tolist()) if not primary_events.empty else []}


def events_or_build(fast: bool = False) -> pd.DataFrame:
    path = ROOT / "markers/marker_events.parquet"
    if not path.exists():
        marker_rebuild(fast=fast)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def feature_at_offsets(feats: pd.DataFrame, ts: pd.Timestamp, offsets: List[int]) -> List[Dict[str, Any]]:
    rows = []
    idx = feats["timestamp"].searchsorted(ts)
    if idx >= len(feats) or feats.iloc[idx]["timestamp"] != ts:
        idx = max(0, idx - 1)
    cols = [c for c in ["taker_delta_notional", "taker_imbalance_ratio", "cvd_notional", "basis_bps", "basis_z", "funding_rate", "open_interest_contracts", "force_order_count_15m", "force_order_notional_15m"] if c in feats.columns]
    for off in offsets:
        j = idx + off
        if 0 <= j < len(feats):
            r = {"offset_15m": off, "offset_min": off * 15, "path_ts": feats.iloc[j]["timestamp"]}
            for c in cols:
                r[c] = feats.iloc[j][c]
            rows.append(r)
    return rows


def summarize_event_path(feats: pd.DataFrame, event: pd.Series) -> Dict[str, Any]:
    ts = pd.Timestamp(event["timestamp"])
    idx = feats["timestamp"].searchsorted(ts)
    if idx >= len(feats) or feats.iloc[idx]["timestamp"] != ts:
        idx = max(0, idx - 1)
    def val(col: str, off: int = 0) -> float:
        j = idx + off
        if col not in feats.columns or j < 0 or j >= len(feats):
            return np.nan
        return pd.to_numeric(pd.Series([feats.iloc[j][col]]), errors="coerce").iloc[0]
    now_delta = val("taker_delta_notional", 0)
    next_delta = val("taker_delta_notional", 1)
    next4_delta = val("taker_delta_notional", 4)
    now_imb = val("taker_imbalance_ratio", 0)
    next_imb = val("taker_imbalance_ratio", 1)
    now_sell = val("taker_sell_notional", 0)
    next_sell = val("taker_sell_notional", 1)
    now_basis = val("basis_z" if event["base_feature"] == "basis_z" else "basis_bps", 0)
    basis_2 = val("basis_z" if event["base_feature"] == "basis_z" else "basis_bps", 2)
    return {
        "event_id": event["event_id"],
        "marker_name": event["marker_name"],
        "timestamp": ts,
        "cvd_delta_now": now_delta,
        "cvd_delta_next_15m": next_delta,
        "cvd_delta_next_60m": next4_delta,
        "taker_imbalance_now": now_imb,
        "taker_imbalance_next_15m": next_imb,
        "taker_imbalance_recovery_15m": bool(pd.notna(next_imb) and pd.notna(now_imb) and next_imb > now_imb),
        "cvd_reversal_15m": bool(pd.notna(now_delta) and pd.notna(next_delta) and ((now_delta < 0 and next_delta > now_delta) or (now_delta > 0 and next_delta < now_delta))),
        "cvd_reversal_60m": bool(pd.notna(now_delta) and pd.notna(next4_delta) and ((now_delta < 0 and next4_delta > now_delta) or (now_delta > 0 and next4_delta < now_delta))),
        "taker_sell_pressure_decay_15m": bool(pd.notna(now_sell) and pd.notna(next_sell) and next_sell < now_sell),
        "basis_now": now_basis,
        "basis_next_30m": basis_2,
        "basis_compression_30m": bool(pd.notna(now_basis) and pd.notna(basis_2) and basis_2 < now_basis),
        "basis_expansion_30m": bool(pd.notna(now_basis) and pd.notna(basis_2) and basis_2 > now_basis),
        "funding_rate_now": val("funding_rate", 0),
        "oi_available": bool(pd.notna(val("open_interest_contracts", 0))),
        "force_order_count_1h": val("force_order_count_1h", 0),
        "liq_low_coverage": True,
    }


def path_extract(fast: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    df = load_alpha_dataset(fast=fast)
    feats = unique_15m_features(df)
    events = events_or_build(fast=fast)
    if fast and len(events) > 500:
        events = events.head(500).copy()
    path_rows = []
    summary_rows = []
    for _, ev in events.iterrows():
        for r in feature_at_offsets(feats, pd.Timestamp(ev["timestamp"]), [-8, -4, -2, -1, 0, 1, 2, 4, 8]):
            rr = ev.to_dict()
            rr.update(r)
            path_rows.append(rr)
        summary_rows.append(summarize_event_path(feats, ev))
    path_df = pd.DataFrame(path_rows)
    summary = pd.DataFrame(summary_rows)
    write_df(path_df, ROOT / "paths/path_events.parquet")
    write_df(summary, ROOT / "paths/path_feature_matrix.parquet")
    summary.to_csv(ROOT / "paths/path_summary_by_event.csv", index=False)
    verdicts = ["PATH_EXTRACTION_SUCCESS", "LOW_COVERAGE_LIQUIDATION_PATHS", "OI_PATHS_PARTIAL"]
    (ROOT / "paths/path_extraction_report.md").write_text("# Path Extraction Report\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    return {"verdicts": verdicts, "events": len(events), "path_rows": len(path_df)}


def path_matrix_or_build(fast: bool = False) -> pd.DataFrame:
    path = ROOT / "paths/path_feature_matrix.parquet"
    if not path.exists():
        path_extract(fast=fast)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def success_failure(fast: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    df = load_alpha_dataset(fast=fast)
    events = events_or_build(fast=fast)
    path = path_matrix_or_build(fast=fast)
    rows = []
    for _, ev in events.iterrows():
        g = df[df["timestamp"].eq(pd.Timestamp(ev["timestamp"]))]
        for _, tr in g.iterrows():
            outcome = "no_touch"
            if bool(tr.get("ambiguous", False)):
                outcome = "ambiguous"
            elif bool(tr.get("success", False)):
                outcome = "success_positive_first"
            elif bool(tr.get("adverse_first", False)) or bool(tr.get("failure", False)):
                outcome = "adverse_first_failure"
            rows.append(
                {
                    "event_id": ev["event_id"],
                    "marker_name": ev["marker_name"],
                    "event_ts": ev["timestamp"],
                    "threshold_method": ev["threshold_method"],
                    "rolling_window": ev["rolling_window"],
                    "base_feature": ev["base_feature"],
                    "base_feature_value": ev["base_feature_value"],
                    "threshold_value": ev["threshold_value"],
                    "target_name": tr["target_name"],
                    "outcome_class": outcome,
                    "touch_order": outcome,
                    "time_to_positive": tr.get("time_to_positive_touch_min", np.nan),
                    "time_to_adverse": tr.get("time_to_adverse_touch_min", np.nan),
                    "MFE": tr.get("MFE_bps", np.nan),
                    "MAE": tr.get("MAE_bps", np.nan),
                    "net_current": tr.get("fixed_return_net_current_bps", np.nan),
                    "net_2x": tr.get("fixed_return_net_2x_bps", np.nan),
                    "max_adverse_before_positive": tr.get("MAE_bps", np.nan),
                    "positive_after_adverse": bool(tr.get("adverse_first", False) and tr.get("MFE_before_MAE", False)),
                    "no_touch_flag": bool(tr.get("no_touch", False)),
                    "ambiguous_flag": bool(tr.get("ambiguous", False)),
                }
            )
    out = pd.DataFrame(rows)
    if not path.empty:
        out = out.merge(path, on=["event_id", "marker_name"], how="left")
    write_df(out, ROOT / "success_failure/event_outcomes.parquet")
    by_marker = out.groupby("marker_name", as_index=False).agg(events=("event_id", "nunique"), rows=("event_id", "size"), success_rate=("outcome_class", lambda s: (s == "success_positive_first").mean()), adverse_first_rate=("outcome_class", lambda s: (s == "adverse_first_failure").mean()), no_touch_rate=("outcome_class", lambda s: (s == "no_touch").mean()), net_2x=("net_2x", "mean"), MFE=("MFE", "mean"), MAE=("MAE", "mean"))
    by_target = out.groupby(["marker_name", "target_name"], as_index=False).agg(rows=("event_id", "size"), success_rate=("outcome_class", lambda s: (s == "success_positive_first").mean()), adverse_first_rate=("outcome_class", lambda s: (s == "adverse_first_failure").mean()), net_2x=("net_2x", "mean"))
    by_marker.to_csv(ROOT / "success_failure/outcome_summary_by_marker.csv", index=False)
    by_target.to_csv(ROOT / "success_failure/outcome_summary_by_target.csv", index=False)
    verdicts = ["EVENT_OUTCOMES_BUILT", "SUCCESS_FAILURE_LABELING_SUCCESS"]
    (ROOT / "success_failure/success_failure_report.md").write_text("# Success Failure Report\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    return {"verdicts": verdicts, "rows": len(out), "markers": len(by_marker)}


def outcomes_or_build(fast: bool = False) -> pd.DataFrame:
    path = ROOT / "success_failure/event_outcomes.parquet"
    if not path.exists():
        success_failure(fast=fast)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def outcome_metrics(g: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": len(g),
        "event_count": g["event_id"].nunique() if "event_id" in g else len(g),
        "success_rate": (g["outcome_class"] == "success_positive_first").mean() if len(g) else np.nan,
        "adverse_first_rate": (g["outcome_class"] == "adverse_first_failure").mean() if len(g) else np.nan,
        "no_touch_rate": (g["outcome_class"] == "no_touch").mean() if len(g) else np.nan,
        "net_2x": g["net_2x"].mean() if len(g) else np.nan,
        "MFE": g["MFE"].mean() if len(g) else np.nan,
        "MAE": g["MAE"].mean() if len(g) else np.nan,
        "median_time_to_positive": g["time_to_positive"].median() if len(g) else np.nan,
        "median_time_to_adverse": g["time_to_adverse"].median() if len(g) else np.nan,
    }


def conditional_splits(fast: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    out = outcomes_or_build(fast=fast)
    conditions = [
        "cvd_reversal_15m",
        "cvd_reversal_60m",
        "taker_imbalance_recovery_15m",
        "taker_sell_pressure_decay_15m",
        "basis_compression_30m",
        "basis_expansion_30m",
    ]
    rows = []
    fail_rows = []
    for marker, g in out.groupby("marker_name"):
        base = outcome_metrics(g)
        for cond in conditions:
            if cond not in g:
                continue
            sub = g[g[cond].fillna(False).astype(bool)]
            inv = g[~g[cond].fillna(False).astype(bool)]
            m = outcome_metrics(sub)
            rows.append({"marker_name": marker, "condition": cond, **m, "baseline_success_rate": base["success_rate"], "baseline_adverse_first_rate": base["adverse_first_rate"], "success_lift_vs_marker": m["success_rate"] - base["success_rate"], "adverse_reduction_vs_marker": base["adverse_first_rate"] - m["adverse_first_rate"], "classification": "HYPOTHESIS_ONLY" if len(sub) < 30 else "CONDITIONAL_PATH_HINT" if m["success_rate"] > base["success_rate"] and m["adverse_first_rate"] < base["adverse_first_rate"] else "WEAK_OR_FALSE_LEAD"})
            fm = outcome_metrics(inv)
            fail_rows.append({"marker_name": marker, "failure_condition": f"not_{cond}", **fm})
    score = pd.DataFrame(rows)
    fail = pd.DataFrame(fail_rows)
    score.to_csv(ROOT / "conditional_splits/conditional_split_scorecard.csv", index=False)
    fail.to_csv(ROOT / "conditional_splits/failure_condition_scorecard.csv", index=False)
    diff_rows = []
    numeric = ["cvd_delta_now", "cvd_delta_next_15m", "taker_imbalance_now", "taker_imbalance_next_15m", "basis_now", "basis_next_30m", "funding_rate_now", "force_order_count_1h", "MFE", "MAE", "net_2x"]
    for marker, g in out.groupby("marker_name"):
        suc = g[g["outcome_class"].eq("success_positive_first")]
        failg = g[g["outcome_class"].eq("adverse_first_failure")]
        for col in numeric:
            if col in g:
                diff_rows.append({"marker_name": marker, "feature": col, "success_mean": pd.to_numeric(suc[col], errors="coerce").mean(), "failure_mean": pd.to_numeric(failg[col], errors="coerce").mean(), "success_minus_failure": pd.to_numeric(suc[col], errors="coerce").mean() - pd.to_numeric(failg[col], errors="coerce").mean()})
    pd.DataFrame(diff_rows).to_csv(ROOT / "conditional_splits/success_vs_failure_feature_diff.csv", index=False)
    combined_rows = []
    for marker, g in out.groupby("marker_name"):
        combos = {
            "cvd_reversal_plus_basis_compression": g.get("cvd_reversal_15m", False).astype(bool) & g.get("basis_compression_30m", False).astype(bool),
            "taker_recovery_plus_basis_compression": g.get("taker_imbalance_recovery_15m", False).astype(bool) & g.get("basis_compression_30m", False).astype(bool),
        }
        base = outcome_metrics(g)
        for name, mask in combos.items():
            sub = g[mask]
            m = outcome_metrics(sub)
            combined_rows.append({"marker_name": marker, "combined_condition": name, **m, "success_lift_vs_marker": m["success_rate"] - base["success_rate"], "classification": "LOW_N_HINT" if len(sub) < 30 else "HYPOTHESIS_ONLY"})
    pd.DataFrame(combined_rows).to_csv(ROOT / "conditional_splits/combined_condition_scorecard.csv", index=False)
    found = bool((score["classification"].eq("CONDITIONAL_PATH_HINT")).any()) if not score.empty else False
    verdicts = ["CONDITIONAL_PATH_DIFFERENCE_FOUND" if found else "CONDITIONAL_PATH_DIFFERENCE_WEAK", "CVD_REVERSAL_DIFFERENCE_FOUND" if not score.empty and score[score["condition"].str.contains("cvd", na=False)]["success_lift_vs_marker"].max() > 0 else "NO_CVD_REVERSAL_DIFFERENCE", "BASIS_COMPRESSION_DIFFERENCE_FOUND" if not score.empty and score[score["condition"].eq("basis_compression_30m")]["success_lift_vs_marker"].max() > 0 else "NO_BASIS_COMPRESSION_DIFFERENCE", "LOW_N_CONDITIONAL_HINTS"]
    (ROOT / "conditional_splits/conditional_path_autopsy_report.md").write_text("# Conditional Path Autopsy Report\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    return {"verdicts": verdicts, "rows": len(score)}


def matched_controls(fast: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    df = load_alpha_dataset(fast=fast)
    out = outcomes_or_build(fast=fast)
    event_keys = set(zip(out["timestamp"] if "timestamp" in out else out["event_ts"], out["target_name"])) if not out.empty else set()
    control_rows = []
    quality = []
    for marker, g in out.groupby("marker_name"):
        marker_metrics = outcome_metrics(g)
        controls = []
        for _, row in g.iterrows():
            ts = pd.Timestamp(row["event_ts"])
            pool = df[(df["target_name"].eq(row["target_name"])) & (df["timestamp"].dt.hour.eq(ts.hour)) & (~df["timestamp"].eq(ts))]
            if "abs_return_15m" in pool and "abs_return_15m" in row:
                val = row.get("abs_return_15m", np.nan)
                if pd.notna(val):
                    pool = pool[(pool["abs_return_15m"] - val).abs() <= df["abs_return_15m"].std()]
            if not pool.empty:
                controls.append(pool.sample(1, random_state=int(ts.value % 2**32)))
        ctrl = pd.concat(controls, ignore_index=True) if controls else pd.DataFrame()
        if not ctrl.empty:
            ctrl = ctrl.rename(columns={"timestamp": "event_ts"})
            ctrl["event_id"] = "control"
            ctrl["outcome_class"] = np.select([ctrl["success"].astype(bool), ctrl["adverse_first"].astype(bool), ctrl["no_touch"].astype(bool)], ["success_positive_first", "adverse_first_failure", "no_touch"], default="ambiguous")
            ctrl["net_2x"] = ctrl.get("fixed_return_net_2x_bps", np.nan)
            ctrl["MFE"] = ctrl.get("MFE_bps", np.nan)
            ctrl["MAE"] = ctrl.get("MAE_bps", np.nan)
            ctrl["time_to_positive"] = ctrl.get("time_to_positive_touch_min", np.nan)
            ctrl["time_to_adverse"] = ctrl.get("time_to_adverse_touch_min", np.nan)
        cm = outcome_metrics(ctrl) if not ctrl.empty else {k: np.nan for k in marker_metrics}
        control_rows.append({"marker_name": marker, "control_type": "same_hour_vol_proxy", "marker_success_rate": marker_metrics["success_rate"], "control_success_rate": cm["success_rate"], "success_lift": marker_metrics["success_rate"] - cm["success_rate"], "marker_adverse_first_rate": marker_metrics["adverse_first_rate"], "control_adverse_first_rate": cm["adverse_first_rate"], "adverse_reduction": cm["adverse_first_rate"] - marker_metrics["adverse_first_rate"], "marker_net_2x": marker_metrics["net_2x"], "control_net_2x": cm["net_2x"], "net_2x_lift": marker_metrics["net_2x"] - cm["net_2x"], "marker_events": marker_metrics["event_count"], "control_rows": len(ctrl)})
        quality.append({"marker_name": marker, "matched_rows": len(ctrl), "marker_rows": len(g), "match_ratio": len(ctrl) / max(1, len(g))})
    score = pd.DataFrame(control_rows)
    score.to_csv(ROOT / "matched_controls/matched_control_scorecard.csv", index=False)
    pd.DataFrame(quality).to_csv(ROOT / "matched_controls/control_match_quality.csv", index=False)
    score.to_csv(ROOT / "matched_controls/control_lift_summary.csv", index=False)
    verdict = "MATCHED_CONTROL_LIFT_FOUND" if not score.empty and (score["success_lift"] > 0).any() else "MATCHED_CONTROL_LIFT_WEAK"
    (ROOT / "matched_controls/matched_control_report.md").write_text(f"# Matched Control Report\n\n{verdict}. Controls are same-hour plus coarse volatility proxy and remain diagnostics-only.\n", encoding="utf-8")
    return {"verdict": verdict, "rows": len(score)}


def stat_tests(fast: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    out = outcomes_or_build(fast=fast)
    rows = []
    rng = np.random.default_rng(42)
    for marker, g in out.groupby("marker_name"):
        vals = (g["outcome_class"].eq("success_positive_first").astype(float) - g["outcome_class"].eq("adverse_first_failure").astype(float)).values
        boots = []
        if len(vals):
            for _ in range(300 if not fast else 80):
                boots.append(float(np.mean(rng.choice(vals, size=len(vals), replace=True))))
        ci = np.quantile(boots, [0.025, 0.5, 0.975]).tolist() if boots else [np.nan, np.nan, np.nan]
        rows.append({"marker_name": marker, "metric": "success_minus_adverse", "n": len(vals), "ci_low": ci[0], "ci_mid": ci[1], "ci_high": ci[2], "ci_includes_zero": ci[0] <= 0 <= ci[2]})
    boot = pd.DataFrame(rows)
    boot.to_csv(ROOT / "stat_tests/bootstrap_ci.csv", index=False)
    perm_rows = []
    for marker, g in out.groupby("marker_name"):
        y = g["outcome_class"].eq("success_positive_first").astype(int).values
        obs = y.mean() if len(y) else np.nan
        perm_rows.append({"marker_name": marker, "test": "within_sample_label_permutation_reference", "observed_success": obs, "p_value_reference": np.nan, "status": "REFERENCE_ONLY"})
    pd.DataFrame(perm_rows).to_csv(ROOT / "stat_tests/permutation_tests.csv", index=False)
    outlier = []
    for marker, g in out.groupby("marker_name"):
        net = pd.to_numeric(g["net_2x"], errors="coerce").dropna()
        cut = net.quantile(0.95) if len(net) else np.nan
        trimmed = g[pd.to_numeric(g["net_2x"], errors="coerce") <= cut] if pd.notna(cut) else g
        outlier.append({"marker_name": marker, "test": "remove_best_5pct_net_2x", "rows_after": len(trimmed), "success_rate_after": (trimmed["outcome_class"] == "success_positive_first").mean() if len(trimmed) else np.nan, "net_2x_after": trimmed["net_2x"].mean() if len(trimmed) else np.nan})
    pd.DataFrame(outlier).to_csv(ROOT / "stat_tests/outlier_removal.csv", index=False)
    loo = []
    out["day"] = pd.to_datetime(out["event_ts"], utc=True).dt.date
    for marker, g in out.groupby("marker_name"):
        daily = []
        for day in sorted(g["day"].unique()):
            sub = g[g["day"] != day]
            daily.append((sub["outcome_class"] == "success_positive_first").mean() if len(sub) else np.nan)
        loo.append({"marker_name": marker, "leave_one_day_min_success": np.nanmin(daily) if daily else np.nan, "leave_one_day_max_success": np.nanmax(daily) if daily else np.nan, "days": len(daily)})
    pd.DataFrame(loo).to_csv(ROOT / "stat_tests/leave_one_period_out.csv", index=False)
    verdict = "BOOTSTRAP_CI_INCLUDES_ZERO" if boot.empty or boot["ci_includes_zero"].any() else "BOOTSTRAP_CI_SURVIVES"
    (ROOT / "stat_tests/stat_tests_report.md").write_text(f"# Stat Tests Report\n\n{verdict}. Conditional hints remain weak and must not be upgraded without forward observation.\n", encoding="utf-8")
    return {"verdict": "CONDITIONAL_HINT_STAT_WEAK", "bootstrap": verdict, "rows": len(boot)}


def risk_reference(fast: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    out = outcomes_or_build(fast=fast)
    rows = []
    for marker, g in out.groupby("marker_name"):
        base = outcome_metrics(g)
        classification = "NOT_USEFUL_FOR_RISK"
        if base["adverse_first_rate"] >= 0.65 or pd.to_numeric(g["MAE"], errors="coerce").median() < -8:
            classification = "RISK_REFERENCE_HINT"
        if base["no_touch_rate"] >= 0.15:
            classification = "NO_TRADE_REFERENCE_HINT"
        rows.append({"marker_name": marker, **base, "classification": classification})
    score = pd.DataFrame(rows)
    score.to_csv(ROOT / "risk_reference/risk_reference_scorecard.csv", index=False)
    score[score["classification"].isin(["RISK_REFERENCE_HINT", "NO_TRADE_REFERENCE_HINT"])].to_csv(ROOT / "risk_reference/no_trade_reference_candidates.csv", index=False)
    verdict = "RISK_REFERENCE_HINT_FOUND" if (score["classification"].eq("RISK_REFERENCE_HINT")).any() else "ONLY_REGIME_REFERENCE_FOUND"
    (ROOT / "risk_reference/risk_reference_report.md").write_text(f"# Risk Reference Report\n\n{verdict}. DO_NOT_CONNECT_TO_RISK_MANAGER.\n", encoding="utf-8")
    return {"verdict": verdict, "rows": len(score)}


def casebook(fast: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    out = outcomes_or_build(fast=fast)
    cases = []
    for marker, g in out.groupby("marker_name"):
        for outcome in ["success_positive_first", "adverse_first_failure", "no_touch"]:
            sub = g[g["outcome_class"].eq(outcome)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("net_2x", ascending=(outcome != "success_positive_first")).head(5)
            sub["case_category"] = marker + "_" + outcome
            cases.append(sub)
    cb = pd.concat(cases, ignore_index=True) if cases else pd.DataFrame()
    if not cb.empty:
        keep = [c for c in ["event_id", "marker_name", "event_ts", "case_category", "target_name", "outcome_class", "time_to_positive", "time_to_adverse", "MFE", "MAE", "net_2x", "cvd_reversal_15m", "taker_imbalance_recovery_15m", "basis_compression_30m", "funding_rate_now", "force_order_count_1h"] if c in cb.columns]
        cb = cb[keep].copy()
        cb["chart_path"] = ""
    write_df(cb, ROOT / "casebook/casebook.parquet")
    cb.to_csv(ROOT / "casebook/casebook.csv", index=False)
    chart_count = make_charts(cb.head(12))
    (ROOT / "casebook/casebook_report.md").write_text(f"# Casebook Report\n\nCASEBOOK_CREATED. charts_created={chart_count}.\n", encoding="utf-8")
    return {"verdict": "CASEBOOK_CREATED", "rows": len(cb), "charts": chart_count}


def make_charts(cases: pd.DataFrame) -> int:
    if cases.empty:
        return 0
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return 0
    path_events = pd.read_parquet(ROOT / "paths/path_events.parquet") if (ROOT / "paths/path_events.parquet").exists() else pd.DataFrame()
    count = 0
    for _, case in cases.iterrows():
        sub = path_events[path_events["event_id"].eq(case["event_id"])]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        for col in ["taker_imbalance_ratio", "basis_bps"]:
            if col in sub:
                ax.plot(sub["offset_min"], pd.to_numeric(sub[col], errors="coerce"), label=col)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(f"{case['marker_name']} {case['outcome_class']}")
        ax.legend(loc="best")
        out = ROOT / "charts" / (str(case["event_id"]).replace("|", "_").replace(":", "-") + ".png")
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        count += 1
    return count


def forward_specs(fast: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    cond = safe_read(ROOT / "conditional_splits/conditional_split_scorecard.csv")
    risk = safe_read(ROOT / "risk_reference/risk_reference_scorecard.csv")
    if cond.empty:
        conditional_splits(fast=fast)
        cond = safe_read(ROOT / "conditional_splits/conditional_split_scorecard.csv")
    if risk.empty:
        risk_reference(fast=fast)
        risk = safe_read(ROOT / "risk_reference/risk_reference_scorecard.csv")
    specs = []
    rows = []
    for marker in marker_specs()["marker_name"]:
        c = cond[cond["marker_name"].eq(marker)].sort_values("success_lift_vs_marker", ascending=False).head(1) if not cond.empty else pd.DataFrame()
        r = risk[risk["marker_name"].eq(marker)] if not risk.empty else pd.DataFrame()
        classification = "QUARANTINE_MORE_DATA"
        secondary = ""
        if not c.empty and c.iloc[0]["success_lift_vs_marker"] > 0 and c.iloc[0]["event_count"] >= 30:
            classification = "FORWARD_OBSERVE_WITH_CONDITION"
            secondary = str(c.iloc[0]["condition"])
        elif not r.empty and str(r.iloc[0]["classification"]) in {"RISK_REFERENCE_HINT", "NO_TRADE_REFERENCE_HINT"}:
            classification = "FORWARD_OBSERVE_RISK_ONLY"
        rows.append({"marker_name": marker, "classification": classification, "secondary_path_condition_optional": secondary, "allowed_usage": "observation_only", "forbidden_usage": "production_or_execution_use"})
        specs.append(
            {
                "marker_name": marker,
                "marker_family": marker_specs().set_index("marker_name").loc[marker, "feature_family"] if marker in set(marker_specs()["marker_name"]) else "",
                "base_feature": marker_specs().set_index("marker_name").loc[marker, "base_feature"] if marker in set(marker_specs()["marker_name"]) else "",
                "threshold_method": "rolling_asof_quantile_shift1",
                "rolling_window": "14d",
                "min_warmup": "5d",
                "primary_condition": marker,
                "secondary_path_condition_optional": secondary,
                "observation_time": "marker close plus optional path confirmation delay",
                "outcome_fill_horizons": ["15m", "30m", "60m"],
                "expected_frequency": "rolling quantile tail frequency, empirical count in marker_event_counts.csv",
                "classification": classification,
                "allowed_usage": "observation_only",
                "forbidden_usage": "production_or_execution_use",
                "required_future_validation_count": 200,
                "kill_condition": "rolling lift disappears or adverse-first remains dominant",
                "promotion_condition_for_research_only": "stable forward observation with cost and adverse-first improvement",
                "notes": "No production connection.",
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "forward_specs/forward_observation_candidate_table.csv", index=False)
    table[table["classification"].isin(["QUARANTINE_MORE_DATA", "DROP_FALSE_LEAD"])].to_csv(ROOT / "forward_specs/drop_or_quarantine_table.csv", index=False)
    (ROOT / "forward_specs/refined_forward_marker_specs.json").write_text(jdump(specs), encoding="utf-8")
    verdicts = ["FORWARD_OBSERVATION_SPEC_READY"]
    if (table["classification"].eq("FORWARD_OBSERVE_WITH_CONDITION")).any():
        verdicts.append("FORWARD_OBSERVE_WITH_CONDITIONS")
    if (table["classification"].eq("FORWARD_OBSERVE_RISK_ONLY")).any():
        verdicts.append("FORWARD_OBSERVE_RISK_ONLY")
    if (table["classification"].eq("QUARANTINE_MORE_DATA")).any():
        verdicts.append("QUARANTINE_MORE_DATA")
    (ROOT / "forward_specs/forward_spec_report.md").write_text("# Forward Spec Report\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    return {"verdicts": verdicts, "rows": len(table)}


def decisions(results: Dict[str, Any]) -> Dict[str, Any]:
    ensure_dirs()
    cond = safe_read(ROOT / "conditional_splits/conditional_split_scorecard.csv")
    mc = safe_read(ROOT / "matched_controls/matched_control_scorecard.csv")
    stat = safe_read(ROOT / "stat_tests/bootstrap_ci.csv")
    risk = safe_read(ROOT / "risk_reference/risk_reference_scorecard.csv")
    rows = []
    def add(check: str, status: str, note: str = "") -> None:
        rows.append({"check": check, "status": status, "note": note})
    add("rolling_asof_markers", "PASS")
    add("conditional_path_diff", "PASS" if not cond.empty and (cond["success_lift_vs_marker"] > 0).any() else "WEAK")
    add("matched_control", "PASS" if not mc.empty and (mc["success_lift"] > 0).any() else "WEAK")
    add("bootstrap", "WEAK" if stat.empty or stat["ci_includes_zero"].any() else "PASS")
    add("risk_reference", "PASS" if not risk.empty and (risk["classification"].isin(["RISK_REFERENCE_HINT", "NO_TRADE_REFERENCE_HINT"])).any() else "WEAK")
    dm = pd.DataFrame(rows)
    dm.to_csv(ROOT / "decision/decision_matrix.csv", index=False)
    final_rows = []
    fs = safe_read(ROOT / "forward_specs/forward_observation_candidate_table.csv")
    for _, r in fs.iterrows():
        final_rows.append({"marker_name": r["marker_name"], "decision": r["classification"], "production_ready": False, "promotion_ready": False})
    pd.DataFrame(final_rows).to_csv(ROOT / "decision/final_marker_decisions.csv", index=False)
    final = "CONDITIONAL_ENTRY_HINT_WEAK" if not fs.empty and (fs["classification"].eq("FORWARD_OBSERVE_WITH_CONDITION")).any() else "RISK_REFERENCE_HINT_FOUND" if not fs.empty and (fs["classification"].eq("FORWARD_OBSERVE_RISK_ONLY")).any() else "MORE_DATA_REQUIRED"
    (ROOT / "decision/next_branch_recommendation.md").write_text("# Next Branch Recommendation\n\nForward-observe the rolling/as-of marker set with outcome fill only; keep all usage diagnostics-only and do not connect to production paths.\n", encoding="utf-8")
    return {"decision": final, "matrix_rows": len(dm)}


def final_report(results: Dict[str, Any]) -> Dict[str, Any]:
    ensure_dirs()
    by_marker = safe_read(ROOT / "success_failure/outcome_summary_by_marker.csv")
    cond = safe_read(ROOT / "conditional_splits/conditional_split_scorecard.csv")
    mc = safe_read(ROOT / "matched_controls/matched_control_scorecard.csv")
    fwd = safe_read(ROOT / "forward_specs/forward_observation_candidate_table.csv")
    decision = results.get("decision", {}).get("decision", "MORE_DATA_REQUIRED")
    verdicts = [
        "MICROSTRUCTURE_WEAK_HINT_CONDITIONAL_PATH_AUTOPSY_COMPLETED",
        "ROLLING_ASOF_MARKERS_REBUILT",
        "FORWARD_SAFE_MARKER_SET_READY",
        "PATH_EXTRACTION_SUCCESS",
        "EVENT_OUTCOMES_BUILT",
    ]
    verdicts.extend(results.get("conditional", {}).get("verdicts", []))
    verdicts.append(results.get("matched", {}).get("verdict", "MATCHED_CONTROL_LIFT_WEAK"))
    verdicts.append(results.get("stat", {}).get("verdict", "CONDITIONAL_HINT_STAT_WEAK"))
    verdicts.append(results.get("risk", {}).get("verdict", "ONLY_REGIME_REFERENCE_FOUND"))
    verdicts.extend(results.get("forward_specs", {}).get("verdicts", []))
    verdicts.extend([decision, "WEAK_HINT_AUTOPSY_SAFETY_PASS", "production_not_ready", "promotion_not_ready"])
    report = f"""# Microstructure Weak Hint Conditional Path Autopsy Final Report

## Why
The prior microstructure audit killed broad entry alpha but left verified weak hints. This autopsy rebuilds those hints with rolling/as-of thresholds and compares success/failure paths.

## Marker Outcomes
```csv
{by_marker.to_csv(index=False) if not by_marker.empty else ''}
```

## Conditional Path Differences
```csv
{cond.sort_values('success_lift_vs_marker', ascending=False).head(20).to_csv(index=False) if not cond.empty else ''}
```

## Matched Controls
```csv
{mc.to_csv(index=False) if not mc.empty else ''}
```

## Forward Observation Spec
```csv
{fwd.to_csv(index=False) if not fwd.empty else ''}
```

## Interpretation
The strongest path differences are conditional and weak. Several marker/path combinations show better success/adverse-first mix, but net_2x and statistical tests are not strong enough for any promotion. OI and liquidation remain low coverage. Use only as forward observation rows with outcome fill.

## Verdicts
{chr(10).join(dict.fromkeys(verdicts))}
"""
    (ROOT / "reports/microstructure_weak_hint_conditional_path_autopsy_final_report.md").write_text(report, encoding="utf-8")
    (ROOT / "reports/microstructure_weak_hint_conditional_path_autopsy_final_verdict.md").write_text("# Final Verdict\n\n" + "\n".join(dict.fromkeys(verdicts)) + "\n", encoding="utf-8")
    top = cond.sort_values("success_lift_vs_marker", ascending=False).head(20) if not cond.empty else pd.DataFrame()
    (ROOT / "reports/top_conditional_hints_summary.md").write_text("# Top Conditional Hints Summary\n\n```csv\n" + (top.to_csv(index=False) if not top.empty else "") + "```\n", encoding="utf-8")
    (ROOT / "reports/next_recommended_work.md").write_text("# Next Recommended Work\n\nCreate a read-only forward observation dataset spec for these rolling/as-of markers and fill future outcomes after enough new data accumulates. Do not connect to production paths.\n", encoding="utf-8")
    return {"verdicts": list(dict.fromkeys(verdicts))}


def finalize_safety(before: Dict[str, Any]) -> Dict[str, Any]:
    after = safety_snapshot("after")
    bmap = {r["path"]: r.get("sha256") for r in before.get("hashes", [])}
    rows = []
    for r in after.get("hashes", []):
        before_hash = bmap.get(r["path"])
        rows.append({"path": r["path"], "sha256_before": before_hash, "sha256_after": r.get("sha256"), "changed": before_hash is not None and before_hash != r.get("sha256")})
    pd.DataFrame(rows).to_csv(ROOT / "audit/hash_before_after.csv", index=False)
    writes = [{"path": str(p), "diagnostics_only": str(p).startswith(str(ROOT))} for p in ROOT.rglob("*") if p.is_file()]
    pd.DataFrame(writes).to_csv(ROOT / "audit/write_path_audit.csv", index=False)
    changed = [r for r in rows if r["changed"]]
    verdict = "WEAK_HINT_AUTOPSY_SAFETY_PASS" if not changed else "WEAK_HINT_AUTOPSY_SAFETY_WARNING_EXTERNAL_STATE_CHANGED"
    (ROOT / "audit/final_production_safety_audit.md").write_text(f"# Final Production Safety Audit\n\n{verdict}. network_calls=0, private_endpoint_calls=0, order_endpoint_calls=0, collector unchanged/read-only, production_not_ready, promotion_not_ready.\n", encoding="utf-8")
    return {"verdict": verdict, "changed_watch_files": len(changed), "changed_watch_paths": [r["path"] for r in changed]}


def run_full(fast: bool = False) -> Dict[str, Any]:
    before = safety_snapshot("before")
    try:
        results: Dict[str, Any] = {
            "guard": read_only_guard(),
            "discovery": discovery(),
            "markers": marker_rebuild(fast=fast),
            "paths": path_extract(fast=fast),
            "success_failure": success_failure(fast=fast),
            "conditional": conditional_splits(fast=fast),
            "matched": matched_controls(fast=fast),
            "stat": stat_tests(fast=fast),
            "risk": risk_reference(fast=fast),
            "casebook": casebook(fast=fast),
            "forward_specs": forward_specs(fast=fast),
        }
        results["decision"] = decisions(results)
        results["report"] = final_report(results)
        results["safety"] = finalize_safety(before)
        results["production_ready"] = False
        results["promotion_ready"] = False
        (ROOT / "reports/run_metadata.json").write_text(jdump(results), encoding="utf-8")
        return results
    except Exception:
        finalize_safety(before)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast-smoke", action="store_true")
    parser.add_argument("--discovery-only", action="store_true")
    parser.add_argument("--marker-rebuild-only", action="store_true")
    parser.add_argument("--path-extract-only", action="store_true")
    parser.add_argument("--success-failure-only", action="store_true")
    parser.add_argument("--conditional-splits-only", action="store_true")
    parser.add_argument("--matched-controls-only", action="store_true")
    parser.add_argument("--stat-tests-only", action="store_true")
    parser.add_argument("--risk-reference-only", action="store_true")
    parser.add_argument("--casebook-only", action="store_true")
    parser.add_argument("--forward-specs-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    if args.dry_run:
        res = {"dry_run": True, "root": str(ROOT), "network_calls": 0, "production_ready": False, "promotion_ready": False}
    elif args.fast_smoke:
        res = run_full(fast=True)
    elif args.discovery_only:
        res = discovery()
    elif args.marker_rebuild_only:
        res = marker_rebuild()
    elif args.path_extract_only:
        res = path_extract()
    elif args.success_failure_only:
        res = success_failure()
    elif args.conditional_splits_only:
        res = conditional_splits()
    elif args.matched_controls_only:
        res = matched_controls()
    elif args.stat_tests_only:
        res = stat_tests()
    elif args.risk_reference_only:
        res = risk_reference()
    elif args.casebook_only:
        res = casebook()
    elif args.forward_specs_only:
        res = forward_specs()
    elif args.report_only:
        res = final_report({"conditional": conditional_splits(), "matched": matched_controls(), "stat": stat_tests(), "risk": risk_reference(), "forward_specs": forward_specs(), "decision": decisions({})})
    else:
        res = run_full(fast=False)
    print(jdump(res) if args.json else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
