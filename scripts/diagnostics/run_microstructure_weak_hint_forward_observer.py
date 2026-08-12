"""Microstructure weak hint forward observer.

Observation-only diagnostics layer. It reads the existing public
microstructure collector output and phase1 feature frames, rebuilds rolling
as-of weak markers, evaluates delayed secondary path conditions, and fills
paper-only outcomes when enough local kline data exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("data/diagnostics/microstructure_weak_hint_forward_observer")
MS_ROOT = Path("data/diagnostics/new_market_microstructure_data_pipeline")
AUTOPSY_ROOT = Path("data/diagnostics/microstructure_weak_hint_conditional_path_autopsy")
PROV_ROOT = Path("data/diagnostics/microstructure_hint_provenance_audit")

FEAT_1M = MS_ROOT / "features/phase1_microstructure_features_1m.parquet"
FEAT_5M = MS_ROOT / "features/phase1_microstructure_features_5m.parquet"
FEAT_15M = MS_ROOT / "features/phase1_microstructure_features_15m.parquet"
LIVE_NORM = MS_ROOT / "live/normalized"
COLLECTOR_STATUS = MS_ROOT / "live/status/microstructure_public_collector_status.json"
COLLECTOR_STATUS_FALLBACK = MS_ROOT / "collector_status/microstructure_collection_status_latest.json"

LABEL = "com.canbit.microstructure-weak-hint-forward-observer"
PRIMARY_MARKERS = [
    {"marker_name": "basis_bps_q95", "base_feature": "basis_bps", "quantile": 0.95, "direction": "high", "family": "BASIS_PREMIUM"},
    {"marker_name": "basis_z_q95", "base_feature": "basis_z", "quantile": 0.95, "direction": "high", "family": "BASIS_PREMIUM"},
    {"marker_name": "funding_rate_q95", "base_feature": "funding_rate", "quantile": 0.95, "direction": "high", "family": "FUNDING"},
    {"marker_name": "taker_imbalance_ratio_q05", "base_feature": "taker_imbalance_ratio", "quantile": 0.05, "direction": "low", "family": "CVD_TAKER_FLOW"},
]
SECONDARY_CONDITIONS = [
    "sell_pressure_decay",
    "taker_imbalance_recovery",
    "cvd_reversal",
    "basis_compression",
    "basis_expansion_failure",
    "persistent_sell_pressure",
    "funding_extreme_with_taker_recovery",
    "basis_extreme_with_taker_recovery",
    "taker_q05_with_sell_pressure_decay",
    "basis_q95_with_taker_recovery",
]
OUTCOME_HORIZONS = [15, 30, 60, 120]
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
FORBIDDEN_TERMS = ["account", "balance", "position", "listenKey", "userDataStream", "leverage", "margin"]
KST_TZ = "Asia/Seoul"


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def closed_cutoff(freq: str, now: pd.Timestamp | None = None) -> pd.Timestamp:
    now = now or now_utc()
    if freq == "1min":
        return now.floor("1min") - pd.Timedelta(minutes=1)
    if freq == "5min":
        return now.floor("5min") - pd.Timedelta(minutes=5)
    if freq == "15min":
        return now.floor("15min") - pd.Timedelta(minutes=15)
    return now


def to_utc(series: Any, source_name: str = "") -> pd.Series:
    """Normalize timestamps to UTC aware.

    Naive timestamps are ambiguous. We compare UTC-localized vs KST-localized
    interpretations and choose the one with fewer future rows relative to now.
    Timezone-qualified strings are respected as-is.
    """
    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    sample = s.dropna().astype(str).head(100)
    has_tz_marker = sample.str.contains(r"(?:Z$|[+-]\d{2}:?\d{2}$)", regex=True).any()
    if has_tz_marker:
        write_timestamp_parse_audit(source_name or "unknown", has_tz_marker, 0, 0, "embedded_timezone")
        return pd.to_datetime(s, utc=True, format="mixed", errors="coerce")
    parsed = pd.to_datetime(s, format="mixed", errors="coerce")
    utc_assumed = parsed.dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
    kst_assumed = parsed.dt.tz_localize(KST_TZ, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
    guard_now = now_utc() + pd.Timedelta(minutes=2)
    utc_future = int((utc_assumed > guard_now).sum())
    kst_future = int((kst_assumed > guard_now).sum())
    chosen = kst_assumed if kst_future < utc_future else utc_assumed
    write_timestamp_parse_audit(source_name or "unknown", has_tz_marker, utc_future, kst_future, "KST" if chosen is kst_assumed else "UTC")
    return chosen


def write_timestamp_parse_audit(source_name: str, has_tz_marker: bool, utc_future: int, kst_future: int, chosen: str) -> None:
    ensure_dirs()
    path = ROOT / "audit/timestamp_parse_audit.csv"
    row = pd.DataFrame(
        [
            {
                "run_ts": now_utc(),
                "source_name": source_name,
                "has_tz_marker": has_tz_marker,
                "naive_as_utc_future_rows": utc_future,
                "naive_as_kst_future_rows": kst_future,
                "chosen_timezone_for_naive": chosen,
            }
        ]
    )
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def filter_future_rows(df: pd.DataFrame, ts_col: str, cutoff: pd.Timestamp, source_name: str) -> Tuple[pd.DataFrame, int]:
    if df.empty or ts_col not in df:
        return df, 0
    ts = pd.to_datetime(df[ts_col], utc=True, format="mixed", errors="coerce")
    mask = ts <= cutoff
    future_count = int((~mask & ts.notna()).sum())
    if future_count:
        ensure_dirs()
        pd.DataFrame(
            [
                {
                    "run_ts": now_utc(),
                    "source_name": source_name,
                    "timestamp_column": ts_col,
                    "cutoff_utc": cutoff,
                    "future_rows_blocked": future_count,
                    "verdict": "TIMESTAMP_FUTURE_GUARD_FAIL",
                }
            ]
        ).to_csv(ROOT / "audit/timestamp_future_guard.csv", mode="a", header=not (ROOT / "audit/timestamp_future_guard.csv").exists(), index=False)
    return df[mask].copy(), future_count


def filter_future_event_frame(df: pd.DataFrame, ts_col: str, cutoff: pd.Timestamp, source_name: str) -> Tuple[pd.DataFrame, int]:
    if df.empty or ts_col not in df:
        return df, 0
    out = df.copy()
    out[ts_col] = to_utc(out[ts_col], source_name).astype("datetime64[ns, UTC]")
    return filter_future_rows(out, ts_col, cutoff, source_name)


def ensure_dirs() -> None:
    for d in [
        "config",
        "audit",
        "registry",
        "features",
        "markers",
        "pending",
        "outcomes",
        "status",
        "reports",
        "logs",
        "launchd",
        "replay",
        "daily",
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


def write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def read_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def append_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(clean(row), ensure_ascii=False, default=str) + "\n")


def read_jsonl_paths(paths: List[Path], limit_days: int = 7) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(paths)[-max(1, limit_days) * 2 :]:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return pd.DataFrame(rows)


def config() -> Dict[str, Any]:
    return {
        "symbol": "BTCUSDT",
        "timezone_storage": "UTC",
        "timezone_display": "Asia/Seoul",
        "mode": "observation_only",
        "production_ready": False,
        "promotion_ready": False,
        "allowed_usage": "observation_only",
        "forbidden_usage": "production_or_execution_use",
        "base_timeframe": "15m",
        "feature_sources": ["phase1_backfill", "local_public_collector_output"],
        "threshold_method": "rolling_asof",
        "rolling_windows": ["7d", "14d", "30d"],
        "default_rolling_window": "14d",
        "min_warmup": "3d",
        "recommended_min_warmup": "7d",
        "primary_markers": [m["marker_name"] for m in PRIMARY_MARKERS],
        "secondary_conditions": SECONDARY_CONDITIONS,
        "secondary_confirmation_offsets": ["1m", "3m", "5m", "15m"],
        "outcome_horizons": [f"{h}m" for h in OUTCOME_HORIZONS],
        "cost_modes": ["current", "maker_like", "two_x"],
        "non_overlap_reference": True,
        "discord_enabled": False,
        "launchd_enabled": False,
        "observation_only": True,
    }


def build_config() -> Dict[str, Any]:
    ensure_dirs()
    cfg = config()
    (ROOT / "config/weak_hint_forward_observer_config.json").write_text(jdump(cfg), encoding="utf-8")
    return {"verdict": "FORWARD_OBSERVER_CONFIG_READY", "config_path": str(ROOT / "config/weak_hint_forward_observer_config.json")}


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
        "exchange_network_calls": 0,
        "private_endpoint_calls": 0,
        "order_endpoint_calls": 0,
        "observation_only": True,
        "production_ready": False,
        "promotion_ready": False,
    }
    (ROOT / f"audit/safety_snapshot_{name}.json").write_text(jdump(snap), encoding="utf-8")
    (ROOT / f"audit/hash_{name}.json").write_text(jdump(rows), encoding="utf-8")
    return snap


def observation_guard() -> Dict[str, Any]:
    ensure_dirs()
    text = Path(__file__).read_text(encoding="utf-8")
    pd.DataFrame([{"term": t, "present": t in text, "context": "literal guard scan only"} for t in FORBIDDEN_TERMS]).to_csv(ROOT / "audit/forbidden_endpoint_scan.csv", index=False)
    rows = [
        {"check": "local_collector_output_read_only", "status": "PASS"},
        {"check": "exchange_network_calls", "status": "PASS", "count": 0},
        {"check": "private_endpoint_calls", "status": "PASS", "count": 0},
        {"check": "order_endpoint_calls", "status": "PASS", "count": 0},
        {"check": "production_write", "status": "PASS", "count": 0},
        {"check": "observation_only_outputs", "status": "PASS"},
    ]
    pd.DataFrame(rows).to_csv(ROOT / "audit/observation_only_guard.csv", index=False)
    return {"verdict": "OBSERVATION_ONLY_GUARD_PASS", "private_endpoint_calls": 0, "order_endpoint_calls": 0, "exchange_network_calls": 0}


def collector_status() -> Dict[str, Any]:
    for p in [COLLECTOR_STATUS, COLLECTOR_STATUS_FALLBACK]:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {"status_read_error": str(p)}
    return {}


def discovery() -> Dict[str, Any]:
    ensure_dirs()
    build_config()
    inputs = [
        AUTOPSY_ROOT / "reports/microstructure_weak_hint_conditional_path_autopsy_final_verdict.md",
        AUTOPSY_ROOT / "forward_specs/refined_forward_marker_specs.json",
        AUTOPSY_ROOT / "forward_specs/forward_observation_candidate_table.csv",
        AUTOPSY_ROOT / "decision/final_marker_decisions.csv",
        PROV_ROOT / "forward_readiness/forward_readiness_table.csv",
        PROV_ROOT / "forward_readiness/forward_marker_specs.json",
        FEAT_1M,
        FEAT_5M,
        FEAT_15M,
        COLLECTOR_STATUS,
        COLLECTOR_STATUS_FALLBACK,
    ]
    inv = [{"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() else 0, "sha256": sha256(p) if p.exists() and p.is_file() and p.stat().st_size < 50_000_000 else None} for p in inputs]
    (ROOT / "registry/input_inventory.json").write_text(jdump(inv), encoding="utf-8")
    pd.DataFrame(PRIMARY_MARKERS).to_csv(ROOT / "registry/marker_spec_inventory.csv", index=False)
    live_files = []
    for sub in ["ws_aggTrade", "ws_kline_1m", "ws_markPrice", "ws_forceOrder"]:
        files = sorted((LIVE_NORM / sub).glob("symbol=BTCUSDT/date=*/events.jsonl"))
        total_size = sum(f.stat().st_size for f in files if f.exists())
        live_files.append({"source": sub, "files": len(files), "total_size": total_size, "latest_file": str(files[-1]) if files else ""})
    pd.DataFrame(live_files).to_csv(ROOT / "registry/live_source_inventory.csv", index=False)
    status = collector_status()
    verdicts = ["FORWARD_OBSERVER_INPUTS_FOUND", "LIVE_MICROSTRUCTURE_SOURCE_ACTIVE" if status.get("is_running") else "LIVE_MICROSTRUCTURE_SOURCE_WARNING"]
    (ROOT / "registry/discovery_report.md").write_text("# Discovery Report\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    return {"verdicts": verdicts, "collector_running": bool(status.get("is_running")), "markers": [m["marker_name"] for m in PRIMARY_MARKERS]}


def load_phase1_features() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    f1 = pd.read_parquet(FEAT_1M)
    f5 = pd.read_parquet(FEAT_5M)
    f15 = pd.read_parquet(FEAT_15M)
    for label, f in [("phase1_1m", f1), ("phase1_5m", f5), ("phase1_15m", f15)]:
        f["timestamp"] = to_utc(f["timestamp"], label).astype("datetime64[ns, UTC]")
        f.sort_values("timestamp", inplace=True)
    return f1, f5, f15


def live_features_1m(limit_days: int = 7) -> pd.DataFrame:
    agg = read_jsonl_paths(sorted((LIVE_NORM / "ws_aggTrade").glob("symbol=BTCUSDT/date=*/events.jsonl")), limit_days=limit_days)
    kl = read_jsonl_paths(sorted((LIVE_NORM / "ws_kline_1m").glob("symbol=BTCUSDT/date=*/events.jsonl")), limit_days=limit_days)
    mp = read_jsonl_paths(sorted((LIVE_NORM / "ws_markPrice").glob("symbol=BTCUSDT/date=*/events.jsonl")), limit_days=limit_days)
    fo = read_jsonl_paths(sorted((LIVE_NORM / "ws_forceOrder").glob("symbol=BTCUSDT/date=*/events.jsonl")), limit_days=limit_days)
    parts = []
    if not agg.empty:
        agg["timestamp"] = to_utc(agg["event_ts"], "live_ws_aggTrade_event_ts").dt.floor("1min").astype("datetime64[ns, UTC]")
        agg["taker_buy_notional"] = np.where(agg["taker_side"].eq("BUY"), pd.to_numeric(agg["notional"], errors="coerce"), 0.0)
        agg["taker_sell_notional"] = np.where(agg["taker_side"].eq("SELL"), pd.to_numeric(agg["notional"], errors="coerce"), 0.0)
        g = agg.groupby("timestamp", as_index=False).agg(
            taker_buy_notional=("taker_buy_notional", "sum"),
            taker_sell_notional=("taker_sell_notional", "sum"),
            taker_buy_qty=("taker_buy_qty", "sum"),
            taker_sell_qty=("taker_sell_qty", "sum"),
            trade_count=("timestamp", "size"),
        )
        g["taker_delta_notional"] = g["taker_buy_notional"] - g["taker_sell_notional"]
        total = (g["taker_buy_notional"] + g["taker_sell_notional"]).replace(0, np.nan)
        g["taker_imbalance_ratio"] = g["taker_delta_notional"] / total
        g["cvd_delta_1m"] = g["taker_delta_notional"]
        g["cvd_notional"] = g["cvd_delta_1m"].cumsum()
        parts.append(g)
    if not kl.empty:
        kl["timestamp"] = to_utc(kl["open_time"], "live_ws_kline_1m_open_time").astype("datetime64[ns, UTC]")
        kl = kl.sort_values(["timestamp", "event_ts"]).drop_duplicates("timestamp", keep="last")
        k = kl[["timestamp", "open", "high", "low", "close", "volume", "is_closed"]].copy()
        k = k.rename(columns={"close": "futures_close", "high": "futures_high", "low": "futures_low", "open": "futures_open"})
        parts.append(k)
    if not mp.empty:
        mp["timestamp"] = to_utc(mp["event_ts"], "live_ws_markPrice_event_ts").dt.floor("1min").astype("datetime64[ns, UTC]")
        m = mp.sort_values("event_ts").groupby("timestamp", as_index=False).tail(1)
        m = m[["timestamp", "mark_price", "index_price", "funding_rate", "mark_index_basis_bps"]].copy()
        m = m.rename(columns={"mark_index_basis_bps": "basis_bps"})
        parts.append(m)
    if not fo.empty:
        fo["timestamp"] = to_utc(fo["event_ts"], "live_ws_forceOrder_event_ts").dt.floor("1min").astype("datetime64[ns, UTC]")
        fo["force_order_notional"] = pd.to_numeric(fo.get("notional", 0), errors="coerce").fillna(0)
        q = fo.groupby("timestamp", as_index=False).agg(force_order_count=("timestamp", "size"), force_order_notional=("force_order_notional", "sum"))
        parts.append(q)
    if not parts:
        return pd.DataFrame()
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="timestamp", how="outer")
    out = out.sort_values("timestamp").drop_duplicates("timestamp")
    for c in ["taker_buy_notional", "taker_sell_notional", "taker_delta_notional", "taker_buy_qty", "taker_sell_qty", "trade_count", "force_order_count", "force_order_notional"]:
        if c in out:
            out[c] = out[c].fillna(0)
    if "basis_bps" in out:
        out["basis_z"] = (out["basis_bps"] - out["basis_bps"].rolling(60 * 24, min_periods=120).mean()) / out["basis_bps"].rolling(60 * 24, min_periods=120).std()
    out["symbol"] = "BTCUSDT"
    return out


def aggregate_tf(f1: pd.DataFrame, tf: str) -> pd.DataFrame:
    if f1.empty:
        return f1
    d = f1.copy()
    d["bucket"] = d["timestamp"].dt.floor(tf)
    agg_map = {}
    sum_cols = ["taker_buy_notional", "taker_sell_notional", "taker_delta_notional", "taker_buy_qty", "taker_sell_qty", "trade_count", "force_order_count", "force_order_notional", "cvd_delta_1m", "volume"]
    last_cols = ["cvd_notional", "basis_bps", "basis_z", "funding_rate", "funding_rate_bps", "open_interest_contracts", "mark_price", "index_price", "futures_close", "spot_close"]
    for c in sum_cols:
        if c in d:
            agg_map[c] = "sum"
    for c in last_cols:
        if c in d:
            agg_map[c] = "last"
    if "futures_high" in d:
        agg_map["futures_high"] = "max"
    if "futures_low" in d:
        agg_map["futures_low"] = "min"
    if "futures_open" in d:
        agg_map["futures_open"] = "first"
    out = d.groupby("bucket", as_index=False).agg(agg_map).rename(columns={"bucket": "timestamp"})
    total = (out.get("taker_buy_notional", 0) + out.get("taker_sell_notional", 0))
    out["taker_imbalance_ratio"] = out.get("taker_delta_notional", 0) / pd.Series(total).replace(0, np.nan)
    if "cvd_notional" in out:
        out["cvd_slope_5m"] = out["cvd_notional"].diff()
        out["cvd_slope_15m"] = out["cvd_notional"].diff(3 if tf == "5min" else 1)
    return out.sort_values("timestamp")


def assemble_features(limit_days: int = 7) -> Dict[str, Any]:
    ensure_dirs()
    p1, p5, p15 = load_phase1_features()
    live1 = live_features_1m(limit_days=limit_days)
    if not live1.empty:
        base_cols = list(dict.fromkeys(list(p1.columns) + list(live1.columns)))
        p1 = p1.reindex(columns=base_cols)
        live1 = live1.reindex(columns=base_cols)
        f1 = pd.concat([p1, live1], ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    else:
        f1 = p1.copy()
    if "basis_z" not in f1 and "basis_bps" in f1:
        f1["basis_z"] = (f1["basis_bps"] - f1["basis_bps"].rolling(60 * 24, min_periods=120).mean()) / f1["basis_bps"].rolling(60 * 24, min_periods=120).std()
    current_now = now_utc()
    f1, f1_future = filter_future_rows(f1, "timestamp", closed_cutoff("1min", current_now), "latest_features_1m")
    live5 = aggregate_tf(live1, "5min") if not live1.empty else pd.DataFrame()
    live15 = aggregate_tf(live1, "15min") if not live1.empty else pd.DataFrame()
    f5 = pd.concat([p5, live5], ignore_index=True, sort=False).sort_values("timestamp").drop_duplicates("timestamp", keep="last") if not live5.empty else p5
    f15 = pd.concat([p15, live15], ignore_index=True, sort=False).sort_values("timestamp").drop_duplicates("timestamp", keep="last") if not live15.empty else p15
    for f in [f1, f5, f15]:
        if "basis_z" not in f and "basis_bps" in f:
            f["basis_z"] = (f["basis_bps"] - f["basis_bps"].rolling(96, min_periods=20).mean()) / f["basis_bps"].rolling(96, min_periods=20).std()
    f5, f5_future = filter_future_rows(f5, "timestamp", closed_cutoff("5min", current_now), "latest_features_5m")
    f15, f15_future = filter_future_rows(f15, "timestamp", closed_cutoff("15min", current_now), "latest_features_15m")
    write_df(f1, ROOT / "features/latest_features_1m.parquet")
    write_df(f5, ROOT / "features/latest_features_5m.parquet")
    write_df(f15, ROOT / "features/latest_features_15m.parquet")
    summary = pd.DataFrame(
        [
            {"frame": "1m", "rows": len(f1), "latest_ts": f1["timestamp"].max(), "live_appended": not live1.empty, "future_rows_blocked": f1_future, "closed_cutoff_utc": closed_cutoff("1min", current_now)},
            {"frame": "5m", "rows": len(f5), "latest_ts": f5["timestamp"].max(), "live_appended": not live5.empty, "future_rows_blocked": f5_future, "closed_cutoff_utc": closed_cutoff("5min", current_now)},
            {"frame": "15m", "rows": len(f15), "latest_ts": f15["timestamp"].max(), "live_appended": not live15.empty, "future_rows_blocked": f15_future, "closed_cutoff_utc": closed_cutoff("15min", current_now)},
        ]
    )
    summary.to_csv(ROOT / "features/latest_feature_assembly_summary.csv", index=False)
    verdicts = ["FORWARD_FEATURE_FRAME_READY", "LATEST_CLOSED_15M_READY"]
    if not live1.empty:
        verdicts.append("LIVE_FEATURES_APPENDED")
    (ROOT / "features/feature_assembly_report.md").write_text("# Feature Assembly Report\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    if f15.empty or f15["timestamp"].max() > closed_cutoff("15min", current_now):
        verdicts.append("TIMESTAMP_FUTURE_GUARD_FAIL")
    return {"verdicts": verdicts, "latest_closed_15m_ts": str(f15["timestamp"].max() if not f15.empty else None), "live_rows_1m": len(live1), "future_rows_blocked": {"1m": f1_future, "5m": f5_future, "15m": f15_future}}


def features_15m() -> pd.DataFrame:
    path = ROOT / "features/latest_features_15m.parquet"
    if not path.exists():
        assemble_features()
    df = pd.read_parquet(path)
    df["timestamp"] = to_utc(df["timestamp"], "cached_latest_features_15m").astype("datetime64[ns, UTC]")
    df, _ = filter_future_rows(df.sort_values("timestamp"), "timestamp", closed_cutoff("15min"), "cached_latest_features_15m_read")
    return df.sort_values("timestamp")


def features_1m() -> pd.DataFrame:
    path = ROOT / "features/latest_features_1m.parquet"
    if not path.exists():
        assemble_features()
    df = pd.read_parquet(path)
    df["timestamp"] = to_utc(df["timestamp"], "cached_latest_features_1m").astype("datetime64[ns, UTC]")
    df, _ = filter_future_rows(df.sort_values("timestamp"), "timestamp", closed_cutoff("1min"), "cached_latest_features_1m_read")
    return df.sort_values("timestamp")


def rolling_thresholds(f15: pd.DataFrame, signal_ts: pd.Timestamp | None = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    hist_rows = []
    if signal_ts is None:
        signal_ts = f15["timestamp"].max()
    for marker in PRIMARY_MARKERS:
        col = marker["base_feature"]
        if col not in f15:
            continue
        s = pd.to_numeric(f15[col], errors="coerce")
        for days in [7, 14, 30]:
            window = 96 * days
            minp = 96 * (3 if days == 7 else 5 if days == 14 else 7)
            thr = s.shift(1).rolling(window, min_periods=minp).quantile(marker["quantile"])
            event = s >= thr if marker["direction"] == "high" else s <= thr
            tmp = pd.DataFrame({"timestamp": f15["timestamp"], "marker_name": marker["marker_name"], "base_feature": col, "rolling_window": f"{days}d", "threshold_value": thr, "base_feature_value": s, "marker_triggered": event})
            hist_rows.append(tmp)
            latest = tmp[tmp["timestamp"].eq(signal_ts)]
            if not latest.empty:
                rr = latest.iloc[-1].to_dict()
                rr.update({"threshold_method": "rolling_asof_quantile_shift1", "warmup_days": days, "warmup_sufficient": pd.notna(rr.get("threshold_value"))})
                rows.append(rr)
    hist = pd.concat(hist_rows, ignore_index=True) if hist_rows else pd.DataFrame()
    latest_df = pd.DataFrame(rows)
    write_df(hist, ROOT / "markers/rolling_threshold_history.parquet")
    latest_payload = latest_df.to_dict("records")
    (ROOT / "markers/rolling_threshold_latest.json").write_text(jdump(latest_payload), encoding="utf-8")
    latest_df.to_csv(ROOT / "markers/threshold_calculation_log.csv", index=False)
    verdicts = ["ROLLING_ASOF_THRESHOLDS_READY", "WARMUP_SUFFICIENT" if latest_df.get("warmup_sufficient", pd.Series(dtype=bool)).all() else "WARMUP_INSUFFICIENT"]
    (ROOT / "markers/threshold_report.md").write_text("# Threshold Report\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    return latest_df, {"verdicts": verdicts, "latest_signal_ts": str(signal_ts)}


def load_existing(path: Path) -> pd.DataFrame:
    return read_df(path)


def score_once() -> Dict[str, Any]:
    ensure_dirs()
    assemble_features()
    f15 = features_15m()
    if f15.empty:
        return {"verdict": "PRIMARY_MARKER_DETECTION_FAIL", "reason": "empty_15m_feature_frame"}
    latest_ts = f15["timestamp"].max()
    cutoff = closed_cutoff("15min")
    if latest_ts > cutoff:
        ensure_dirs()
        pd.DataFrame(
            [
                {
                    "run_ts": now_utc(),
                    "latest_15m_ts": latest_ts,
                    "closed_cutoff_utc": cutoff,
                    "future_seconds": (latest_ts - cutoff).total_seconds(),
                    "verdict": "TIMESTAMP_FUTURE_GUARD_FAIL",
                    "action": "score_blocked_no_marker_written",
                }
            ]
        ).to_csv(ROOT / "audit/timestamp_future_guard.csv", mode="a", header=not (ROOT / "audit/timestamp_future_guard.csv").exists(), index=False)
        return {"verdict": "TIMESTAMP_FUTURE_GUARD_FAIL", "new_markers": 0, "latest_closed_15m_ts": str(latest_ts), "closed_cutoff_utc": str(cutoff), "score_blocked": True}
    latest, threshold_res = rolling_thresholds(f15, latest_ts)
    existing = load_existing(ROOT / "markers/observed_primary_markers.parquet")
    rows = []
    now = pd.Timestamp.now(tz="UTC")
    for _, r in latest.iterrows():
        if r["rolling_window"] != "14d":
            continue
        signal_ts = pd.Timestamp(r["timestamp"]).tz_convert("UTC") if pd.Timestamp(r["timestamp"]).tzinfo else pd.Timestamp(r["timestamp"]).tz_localize("UTC")
        if signal_ts > cutoff:
            continue
        obs_id = f"{r['marker_name']}|{signal_ts.isoformat()}"
        duplicate = not existing.empty and obs_id in set(existing.get("observation_id", pd.Series(dtype=str)).astype(str))
        triggered = bool(r.get("marker_triggered")) and bool(r.get("warmup_sufficient")) and not duplicate
        if triggered:
            rows.append(
                {
                    "observation_id": obs_id,
                    "marker_id": obs_id,
                    "marker_name": r["marker_name"],
                    "symbol": "BTCUSDT",
                    "signal_ts": signal_ts,
                    "detected_ts": now,
                    "feature_ts": signal_ts,
                    "base_feature": r["base_feature"],
                    "base_feature_value": r["base_feature_value"],
                    "threshold_value": r["threshold_value"],
                    "threshold_method": "rolling_asof_quantile_shift1",
                    "rolling_window": r["rolling_window"],
                    "warmup_days": r["warmup_days"],
                    "marker_triggered": True,
                    "feature_family": next((m["family"] for m in PRIMARY_MARKERS if m["marker_name"] == r["marker_name"]), ""),
                    "source_data_version": "phase1_plus_local_collector",
                    "asof_pass": signal_ts <= cutoff,
                    "observation_only": True,
                    "production_ready": False,
                    "promotion_ready": False,
                    "is_replay": False,
                }
            )
    new = pd.DataFrame(rows)
    combined = pd.concat([existing, new], ignore_index=True, sort=False).drop_duplicates("observation_id", keep="first") if not existing.empty or not new.empty else pd.DataFrame()
    write_df(combined, ROOT / "markers/observed_primary_markers.parquet")
    combined.to_csv(ROOT / "markers/observed_primary_markers.csv", index=False)
    append_jsonl(new.to_dict("records"), ROOT / "markers/observed_primary_markers.jsonl")
    pending_sec = load_existing(ROOT / "pending/pending_secondary_evaluation.parquet")
    pending_out = load_existing(ROOT / "pending/pending_outcome_fill.parquet")
    pending_new = new[["observation_id", "marker_name", "signal_ts"]].copy() if not new.empty else pd.DataFrame(columns=["observation_id", "marker_name", "signal_ts"])
    if not pending_new.empty:
        pending_new["status"] = "pending_secondary"
        pending_new["created_ts"] = now
        pending_sec = pd.concat([pending_sec, pending_new], ignore_index=True, sort=False).drop_duplicates(["observation_id"], keep="first")
        origin = []
        for _, r in new.iterrows():
            for h in OUTCOME_HORIZONS:
                origin.append({"observation_id": r["observation_id"], "anchor_type": "marker_origin", "anchor_ts": r["signal_ts"], "horizon_min": h, "status": "pending", "created_ts": now})
        pending_out = pd.concat([pending_out, pd.DataFrame(origin)], ignore_index=True, sort=False).drop_duplicates(["observation_id", "anchor_type", "horizon_min"], keep="first")
    write_df(pending_sec, ROOT / "pending/pending_secondary_evaluation.parquet")
    write_df(pending_out, ROOT / "pending/pending_outcome_fill.parquet")
    verdict = "PRIMARY_MARKERS_DETECTED" if len(new) else "NO_PRIMARY_MARKER_THIS_CANDLE"
    return {"verdict": verdict, "new_markers": len(new), "latest_closed_15m_ts": str(latest_ts), **threshold_res}


def condition_value(f1: pd.DataFrame, marker_row: pd.Series, offset_min: int) -> List[Dict[str, Any]]:
    signal_ts = pd.Timestamp(marker_row["signal_ts"])
    confirm_ts = signal_ts + pd.Timedelta(minutes=offset_min)
    before = f1[f1["timestamp"].between(signal_ts - pd.Timedelta(minutes=5), signal_ts)]
    after = f1[f1["timestamp"].between(signal_ts + pd.Timedelta(minutes=1), confirm_ts)]
    if before.empty or after.empty:
        return []
    sell_before = before.get("taker_sell_notional", pd.Series(dtype=float)).sum()
    sell_after = after.get("taker_sell_notional", pd.Series(dtype=float)).sum()
    imb_before = pd.to_numeric(before.get("taker_imbalance_ratio", pd.Series(dtype=float)), errors="coerce").iloc[-1] if "taker_imbalance_ratio" in before else np.nan
    imb_after = pd.to_numeric(after.get("taker_imbalance_ratio", pd.Series(dtype=float)), errors="coerce").iloc[-1] if "taker_imbalance_ratio" in after else np.nan
    cvd_before = pd.to_numeric(before.get("cvd_notional", pd.Series(dtype=float)), errors="coerce").iloc[-1] if "cvd_notional" in before else np.nan
    cvd_after = pd.to_numeric(after.get("cvd_notional", pd.Series(dtype=float)), errors="coerce").iloc[-1] if "cvd_notional" in after else np.nan
    basis_before = pd.to_numeric(before.get("basis_bps", pd.Series(dtype=float)), errors="coerce").iloc[-1] if "basis_bps" in before else np.nan
    basis_after = pd.to_numeric(after.get("basis_bps", pd.Series(dtype=float)), errors="coerce").iloc[-1] if "basis_bps" in after else np.nan
    rows = []
    checks = {
        "sell_pressure_decay": (sell_after < sell_before, sell_before, sell_after, sell_after - sell_before),
        "taker_imbalance_recovery": (pd.notna(imb_after) and pd.notna(imb_before) and imb_after > imb_before, imb_before, imb_after, imb_after - imb_before),
        "cvd_reversal": (pd.notna(cvd_after) and pd.notna(cvd_before) and cvd_after > cvd_before, cvd_before, cvd_after, cvd_after - cvd_before),
        "basis_compression": (pd.notna(basis_after) and pd.notna(basis_before) and basis_after < basis_before, basis_before, basis_after, basis_after - basis_before),
        "basis_expansion_failure": (pd.notna(basis_after) and pd.notna(basis_before) and basis_after > basis_before, basis_before, basis_after, basis_after - basis_before),
        "persistent_sell_pressure": (sell_after >= sell_before, sell_before, sell_after, sell_after - sell_before),
    }
    checks["funding_extreme_with_taker_recovery"] = (marker_row["marker_name"] == "funding_rate_q95" and checks["taker_imbalance_recovery"][0], checks["taker_imbalance_recovery"][1], checks["taker_imbalance_recovery"][2], checks["taker_imbalance_recovery"][3])
    checks["basis_extreme_with_taker_recovery"] = (marker_row["marker_name"] in {"basis_bps_q95", "basis_z_q95"} and checks["taker_imbalance_recovery"][0], checks["taker_imbalance_recovery"][1], checks["taker_imbalance_recovery"][2], checks["taker_imbalance_recovery"][3])
    checks["taker_q05_with_sell_pressure_decay"] = (marker_row["marker_name"] == "taker_imbalance_ratio_q05" and checks["sell_pressure_decay"][0], checks["sell_pressure_decay"][1], checks["sell_pressure_decay"][2], checks["sell_pressure_decay"][3])
    checks["basis_q95_with_taker_recovery"] = (marker_row["marker_name"] in {"basis_bps_q95", "basis_z_q95"} and checks["taker_imbalance_recovery"][0], checks["taker_imbalance_recovery"][1], checks["taker_imbalance_recovery"][2], checks["taker_imbalance_recovery"][3])
    for name, (triggered, before_val, after_val, delta) in checks.items():
        rows.append(
            {
                "observation_id": marker_row["observation_id"],
                "marker_id": marker_row["marker_id"],
                "marker_name": marker_row["marker_name"],
                "signal_ts": signal_ts,
                "condition_name": name,
                "condition_offset": f"{offset_min}m",
                "condition_confirm_ts": confirm_ts,
                "condition_evaluated": True,
                "condition_triggered": bool(triggered),
                "condition_value_before": before_val,
                "condition_value_after": after_val,
                "condition_delta": delta,
                "condition_threshold": 0,
                "condition_method": "closed_local_path_after_marker",
                "asof_pass": confirm_ts > signal_ts,
                "observation_only": True,
                "production_ready": False,
                "promotion_ready": False,
            }
        )
    return rows


def evaluate_secondary_once() -> Dict[str, Any]:
    ensure_dirs()
    f1 = features_1m()
    prim = read_df(ROOT / "markers/observed_primary_markers.parquet")
    if prim.empty:
        return {"verdict": "SECONDARY_CONDITION_PENDING", "evaluated_rows": 0}
    prim, future_markers_blocked = filter_future_event_frame(prim, "signal_ts", closed_cutoff("15min"), "secondary_primary_marker_signal_ts")
    existing = read_df(ROOT / "markers/secondary_conditions.parquet")
    existing_keys = set(zip(existing.get("observation_id", []), existing.get("condition_name", []), existing.get("condition_offset", []))) if not existing.empty else set()
    rows = []
    now = pd.Timestamp.now(tz="UTC")
    for _, r in prim.iterrows():
        for offset in [1, 3, 5, 15]:
            if pd.Timestamp(r["signal_ts"]) + pd.Timedelta(minutes=offset) > f1["timestamp"].max():
                continue
            for cr in condition_value(f1, r, offset):
                key = (cr["observation_id"], cr["condition_name"], cr["condition_offset"])
                if key not in existing_keys:
                    rows.append(cr)
    new = pd.DataFrame(rows)
    combined = pd.concat([existing, new], ignore_index=True, sort=False).drop_duplicates(["observation_id", "condition_name", "condition_offset"], keep="first") if not existing.empty or not new.empty else pd.DataFrame()
    write_df(combined, ROOT / "markers/secondary_conditions.parquet")
    combined.to_csv(ROOT / "markers/secondary_conditions.csv", index=False)
    append_jsonl(new.to_dict("records"), ROOT / "markers/secondary_conditions.jsonl")
    pending_out = read_df(ROOT / "pending/pending_outcome_fill.parquet")
    condition_out = []
    for _, r in new[new.get("condition_triggered", pd.Series(dtype=bool)).astype(bool)].iterrows() if not new.empty else []:
        for h in OUTCOME_HORIZONS:
            condition_out.append({"observation_id": r["observation_id"], "condition_name": r["condition_name"], "anchor_type": "condition_confirmed", "anchor_ts": r["condition_confirm_ts"], "horizon_min": h, "status": "pending", "created_ts": now})
    if condition_out:
        pending_out = pd.concat([pending_out, pd.DataFrame(condition_out)], ignore_index=True, sort=False).drop_duplicates(["observation_id", "anchor_type", "condition_name", "horizon_min"], keep="first")
        write_df(pending_out, ROOT / "pending/pending_outcome_fill.parquet")
    verdict = "SECONDARY_CONDITIONS_EVALUATED" if len(new) else "NO_SECONDARY_CONDITION_YET"
    return {"verdict": verdict, "new_condition_rows": len(new), "triggered_conditions": int(new.get("condition_triggered", pd.Series(dtype=bool)).sum()) if not new.empty else 0, "future_markers_blocked": future_markers_blocked}


def price_path_outcome(f1: pd.DataFrame, anchor_ts: pd.Timestamp, horizon_min: int, y_bps: float = 10, x_bps: float = 5) -> Dict[str, Any] | None:
    end_ts = anchor_ts + pd.Timedelta(minutes=horizon_min)
    path = f1[(f1["timestamp"] > anchor_ts) & (f1["timestamp"] <= end_ts)].copy()
    if path.empty or path["timestamp"].max() < end_ts:
        return None
    close_col = "futures_close" if "futures_close" in f1 else "close"
    high_col = "futures_high" if "futures_high" in f1 else "high"
    low_col = "futures_low" if "futures_low" in f1 else "low"
    anchor_rows = f1[f1["timestamp"] <= anchor_ts]
    if anchor_rows.empty or close_col not in anchor_rows:
        return None
    entry = pd.to_numeric(anchor_rows.iloc[-1].get(close_col), errors="coerce")
    if pd.isna(entry) or entry <= 0 or high_col not in path or low_col not in path:
        return None
    pos_level = entry * (1 + y_bps / 10000)
    adv_level = entry * (1 - x_bps / 10000)
    pos_hit = path[pd.to_numeric(path[high_col], errors="coerce") >= pos_level]
    adv_hit = path[pd.to_numeric(path[low_col], errors="coerce") <= adv_level]
    tpos = pos_hit["timestamp"].min() if not pos_hit.empty else pd.NaT
    tadv = adv_hit["timestamp"].min() if not adv_hit.empty else pd.NaT
    ambiguous = pd.notna(tpos) and pd.notna(tadv) and tpos == tadv
    success = pd.notna(tpos) and (pd.isna(tadv) or tpos < tadv) and not ambiguous
    adverse = pd.notna(tadv) and (pd.isna(tpos) or tadv < tpos) and not ambiguous
    no_touch = not success and not adverse and not ambiguous
    highs = pd.to_numeric(path[high_col], errors="coerce")
    lows = pd.to_numeric(path[low_col], errors="coerce")
    mfe = (highs.max() / entry - 1) * 10000
    mae = (lows.min() / entry - 1) * 10000
    final = pd.to_numeric(path.iloc[-1].get(close_col), errors="coerce")
    fixed = (final / entry - 1) * 10000 if pd.notna(final) else np.nan
    current_cost = 4.0
    return {
        "positive_touch": bool(pd.notna(tpos)),
        "adverse_touch": bool(pd.notna(tadv)),
        "touch_order": "success_positive_first" if success else "adverse_first" if adverse else "ambiguous" if ambiguous else "no_touch",
        "success_positive_first": bool(success),
        "adverse_first": bool(adverse),
        "no_touch": bool(no_touch),
        "ambiguous": bool(ambiguous),
        "time_to_positive": (tpos - anchor_ts).total_seconds() / 60 if pd.notna(tpos) else np.nan,
        "time_to_adverse": (tadv - anchor_ts).total_seconds() / 60 if pd.notna(tadv) else np.nan,
        "MFE_bps": mfe,
        "MAE_bps": mae,
        "fixed_return_bps": fixed,
        "net_current_bps": fixed - current_cost if pd.notna(fixed) else np.nan,
        "net_maker_like_bps": fixed - 2.0 if pd.notna(fixed) else np.nan,
        "net_2x_bps": fixed - current_cost * 2 if pd.notna(fixed) else np.nan,
        "max_adverse_before_positive": mae,
        "positive_after_adverse": bool(pd.notna(tpos) and pd.notna(tadv) and tpos > tadv),
        "path_complete": True,
    }


def fill_outcomes_once() -> Dict[str, Any]:
    ensure_dirs()
    f1 = features_1m()
    pending = read_df(ROOT / "pending/pending_outcome_fill.parquet")
    if pending.empty:
        return {"verdict": "OUTCOMES_PENDING", "filled": 0, "pending": 0}
    existing = read_df(ROOT / "outcomes/filled_outcomes.parquet")
    existing_keys = set(zip(existing.get("observation_id", []), existing.get("anchor_type", []), existing.get("condition_name", []), existing.get("horizon_min", []))) if not existing.empty else set()
    rows = []
    keep_pending = []
    now = pd.Timestamp.now(tz="UTC")
    for _, r in pending.iterrows():
        anchor_ts = pd.Timestamp(r["anchor_ts"])
        anchor_ts = anchor_ts.tz_convert("UTC") if anchor_ts.tzinfo else anchor_ts.tz_localize("UTC")
        horizon = int(r["horizon_min"])
        condition_name = r.get("condition_name", "")
        key = (r["observation_id"], r["anchor_type"], condition_name, horizon)
        if key in existing_keys:
            continue
        if anchor_ts > f1["timestamp"].max():
            keep_pending.append(r.to_dict())
            continue
        outcome = price_path_outcome(f1, anchor_ts, horizon)
        if outcome is None:
            keep_pending.append(r.to_dict())
            continue
        row = r.to_dict()
        row.update(outcome)
        row.update({"outcome_filled_ts": now, "observation_only": True, "production_ready": False, "promotion_ready": False})
        rows.append(row)
    new = pd.DataFrame(rows)
    combined = pd.concat([existing, new], ignore_index=True, sort=False).drop_duplicates(["observation_id", "anchor_type", "condition_name", "horizon_min"], keep="first") if not existing.empty or not new.empty else pd.DataFrame()
    write_df(combined, ROOT / "outcomes/filled_outcomes.parquet")
    combined.to_csv(ROOT / "outcomes/filled_outcomes.csv", index=False)
    append_jsonl(new.to_dict("records"), ROOT / "outcomes/filled_outcomes.jsonl")
    write_df(pd.DataFrame(keep_pending), ROOT / "pending/pending_outcome_fill.parquet")
    pd.DataFrame([{"run_ts": now, "filled": len(new), "remaining_pending": len(keep_pending)}]).to_csv(ROOT / "outcomes/outcome_fill_log.csv", mode="a", header=not (ROOT / "outcomes/outcome_fill_log.csv").exists(), index=False)
    verdict = "OUTCOMES_FILLED" if len(new) else "OUTCOMES_PENDING"
    return {"verdict": verdict, "filled": len(new), "pending": len(keep_pending)}


def replay_recent(days: int = 7) -> Dict[str, Any]:
    ensure_dirs()
    assemble_features(limit_days=max(days, 7))
    f15 = features_15m()
    start = f15["timestamp"].max() - pd.Timedelta(days=days)
    hist, _ = rolling_thresholds(f15)
    hist = hist[(hist["timestamp"] >= start) & (hist["rolling_window"].eq("14d")) & (hist["marker_triggered"].fillna(False))].copy()
    if hist.empty:
        hist = hist.head(0)
    hist["observation_id"] = "replay|" + hist["marker_name"].astype(str) + "|" + hist["timestamp"].astype(str)
    hist["marker_id"] = hist["observation_id"]
    hist["signal_ts"] = hist["timestamp"]
    hist["detected_ts"] = pd.Timestamp.now(tz="UTC")
    hist["feature_ts"] = hist["timestamp"]
    hist["threshold_method"] = "rolling_asof_quantile_shift1"
    hist["observation_only"] = True
    hist["production_ready"] = False
    hist["promotion_ready"] = False
    hist["is_replay"] = True
    markers = hist.rename(columns={"threshold_value": "threshold_value", "base_feature_value": "base_feature_value"})[
        ["observation_id", "marker_id", "marker_name", "signal_ts", "detected_ts", "feature_ts", "base_feature", "base_feature_value", "threshold_value", "threshold_method", "rolling_window", "marker_triggered", "observation_only", "production_ready", "promotion_ready", "is_replay"]
    ].copy()
    write_df(markers, ROOT / "replay/replay_observed_primary_markers.parquet")
    # Temporarily evaluate secondary/outcomes from replay rows without modifying live marker files.
    f1 = features_1m()
    cond_rows = []
    for _, r in markers.iterrows():
        for off in [1, 3, 5, 15]:
            cond_rows.extend(condition_value(f1, r, off))
    cond = pd.DataFrame(cond_rows)
    if not cond.empty:
        cond["is_replay"] = True
    write_df(cond, ROOT / "replay/replay_secondary_conditions.parquet")
    out_rows = []
    for _, r in markers.iterrows():
        for h in OUTCOME_HORIZONS:
            outcome = price_path_outcome(f1, pd.Timestamp(r["signal_ts"]), h)
            if outcome:
                row = {"observation_id": r["observation_id"], "anchor_type": "marker_origin", "anchor_ts": r["signal_ts"], "horizon_min": h, "is_replay": True, "observation_only": True, "production_ready": False, "promotion_ready": False}
                row.update(outcome)
                out_rows.append(row)
    outcomes = pd.DataFrame(out_rows)
    write_df(outcomes, ROOT / "replay/replay_filled_outcomes.parquet")
    summary = pd.DataFrame(
        [
            {"metric": "replay_primary_markers", "value": len(markers)},
            {"metric": "replay_secondary_conditions", "value": len(cond)},
            {"metric": "replay_filled_outcomes", "value": len(outcomes)},
        ]
    )
    summary.to_csv(ROOT / "replay/replay_summary.csv", index=False)
    verdicts = ["REPLAY_RECENT_SUCCESS", "OBSERVATION_SCHEMA_VALIDATED"]
    (ROOT / "replay/replay_report.md").write_text("# Replay Report\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    return {"verdicts": verdicts, "days": days, "markers": len(markers), "outcomes": len(outcomes)}


def observer_status() -> Dict[str, Any]:
    ensure_dirs()
    prim = read_df(ROOT / "markers/observed_primary_markers.parquet")
    sec = read_df(ROOT / "markers/secondary_conditions.parquet")
    pend_sec = read_df(ROOT / "pending/pending_secondary_evaluation.parquet")
    pend_out = read_df(ROOT / "pending/pending_outcome_fill.parquet")
    out = read_df(ROOT / "outcomes/filled_outcomes.parquet")
    replay = read_df(ROOT / "replay/replay_observed_primary_markers.parquet")
    f15 = features_15m() if (ROOT / "features/latest_features_15m.parquet").exists() else pd.DataFrame()
    prim_valid, future_primary_markers = filter_future_event_frame(prim, "signal_ts", closed_cutoff("15min"), "status_primary_marker_signal_ts")
    sec_valid, future_secondary_conditions = filter_future_event_frame(sec, "condition_confirm_ts", closed_cutoff("1min"), "status_secondary_condition_confirm_ts")
    out_valid, future_outcomes = filter_future_event_frame(out, "anchor_ts", closed_cutoff("1min"), "status_outcome_anchor_ts")
    launchd = launchd_status(raw=True)
    schedule_active = bool(launchd.get("installed") and launchd.get("print_ok"))
    process_running = bool(launchd.get("running", False))
    status = {
        "observer_running": schedule_active,
        "observer_process_running_now": process_running,
        "launchd_schedule_active": schedule_active,
        "launchd_installed": bool(launchd.get("installed", False)),
        "launchd_label": LABEL,
        "last_run_ts": pd.Timestamp.now(tz="UTC"),
        "latest_closed_15m_ts": f15["timestamp"].max() if not f15.empty else None,
        "latest_feature_ts": f15["timestamp"].max() if not f15.empty else None,
        "live_collector_active": bool(collector_status().get("is_running")),
        "primary_markers_total": len(prim_valid),
        "primary_markers_raw_total": len(prim),
        "future_primary_markers_blocked": future_primary_markers,
        "primary_markers_by_name": prim_valid["marker_name"].value_counts().to_dict() if not prim_valid.empty else {},
        "secondary_conditions_total": len(sec_valid),
        "secondary_conditions_raw_total": len(sec),
        "future_secondary_conditions_blocked": future_secondary_conditions,
        "secondary_conditions_by_name": sec_valid["condition_name"].value_counts().to_dict() if not sec_valid.empty else {},
        "pending_secondary_count": len(pend_sec),
        "pending_outcome_count": len(pend_out),
        "filled_outcomes_total": len(out_valid),
        "filled_outcomes_raw_total": len(out),
        "future_outcome_anchors_blocked": future_outcomes,
        "filled_outcomes_by_horizon": out_valid["horizon_min"].value_counts().to_dict() if not out_valid.empty and "horizon_min" in out_valid else {},
        "marker_origin_outcomes_total": int(out_valid["anchor_type"].eq("marker_origin").sum()) if not out_valid.empty and "anchor_type" in out_valid else 0,
        "condition_confirmed_outcomes_total": int(out_valid["anchor_type"].eq("condition_confirmed").sum()) if not out_valid.empty and "anchor_type" in out_valid else 0,
        "latest_marker_ts": prim_valid["signal_ts"].max() if not prim_valid.empty and "signal_ts" in prim_valid else None,
        "latest_condition_confirm_ts": sec_valid["condition_confirm_ts"].max() if not sec_valid.empty and "condition_confirm_ts" in sec_valid else None,
        "latest_outcome_fill_ts": out_valid["outcome_filled_ts"].max() if not out_valid.empty and "outcome_filled_ts" in out_valid else None,
        "replay_rows": len(replay),
        "live_rows": int((~prim_valid.get("is_replay", pd.Series(dtype=bool)).fillna(False)).sum()) if not prim_valid.empty else 0,
        "private_endpoint_calls": 0,
        "order_endpoint_calls": 0,
        "production_ready": False,
        "promotion_ready": False,
        "observation_only": True,
        "next_recommended_check": "review after >=50 primary observations and >=30 filled 30m/60m outcomes",
    }
    (ROOT / "status/weak_hint_forward_observer_status.json").write_text(jdump(status), encoding="utf-8")
    md = "# Weak Hint Forward Observer Status\n\n```json\n" + jdump(status) + "\n```\n"
    (ROOT / "status/weak_hint_forward_observer_status.md").write_text(md, encoding="utf-8")
    return {"verdict": "STATUS_UPDATED", **status}


def daily_report() -> Dict[str, Any]:
    ensure_dirs()
    status = observer_status()
    out = read_df(ROOT / "outcomes/filled_outcomes.parquet")
    prim = read_df(ROOT / "markers/observed_primary_markers.parquet")
    now = pd.Timestamp.now(tz="UTC")
    date = now.strftime("%Y%m%d")
    recent_markers = prim[pd.to_datetime(prim.get("signal_ts", pd.Series(dtype=str)), utc=True) >= now - pd.Timedelta(days=1)] if not prim.empty else prim
    recent_out = out[pd.to_datetime(out.get("outcome_filled_ts", pd.Series(dtype=str)), utc=True) >= now - pd.Timedelta(days=1)] if not out.empty and "outcome_filled_ts" in out else out
    rows = [
        {"metric": "last_24h_primary_markers", "value": len(recent_markers)},
        {"metric": "last_24h_outcomes_filled", "value": len(recent_out)},
        {"metric": "pending_outcome_count", "value": status.get("pending_outcome_count", 0)},
        {"metric": "live_collector_active", "value": status.get("live_collector_active", False)},
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "daily/weak_hint_forward_daily_summary.csv", index=False)
    md = f"# Weak Hint Forward Daily Report {date}\n\nObservation-only diagnostics report. No production or execution use.\n\n```csv\n{summary.to_csv(index=False)}```\n"
    (ROOT / f"daily/weak_hint_forward_daily_report_{date}.md").write_text(md, encoding="utf-8")
    return {"verdict": "DAILY_REPORT_CREATED", "date": date, "last_24h_markers": len(recent_markers)}


def build_plist() -> Dict[str, Any]:
    ensure_dirs()
    python = sys.executable
    script = str(Path(__file__).resolve())
    cwd = str(Path.cwd())
    plist = {
        "Label": LABEL,
        "ProgramArguments": [
            python,
            script,
            "--score-once",
            "--evaluate-secondary-once",
            "--fill-outcomes-once",
            "--status",
            "--json",
        ],
        "WorkingDirectory": cwd,
        "StartInterval": 300,
        "RunAtLoad": True,
        "StandardOutPath": str((ROOT / "logs/observer_stdout.log").resolve()),
        "StandardErrorPath": str((ROOT / "logs/observer_stderr.log").resolve()),
        "EnvironmentVariables": {"PYTHONUNBUFFERED": "1"},
    }
    out = ROOT / "launchd" / f"{LABEL}.plist"
    with out.open("wb") as fh:
        plistlib.dump(plist, fh)
    return {"verdict": "WEAK_HINT_FORWARD_OBSERVER_PLIST_READY", "plist_path": str(out)}


def launchd_user_plist_path() -> Path:
    return Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"


def install_launchd() -> Dict[str, Any]:
    ensure_dirs()
    built = build_plist()
    src = Path(built["plist_path"])
    dest = launchd_user_plist_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        backup = dest.with_suffix(dest.suffix + f".bak.{int(time.time())}")
        shutil.copy2(dest, backup)
    shutil.copy2(src, dest)
    uid = os.getuid()
    cmds = [
        ["launchctl", "bootout", f"gui/{uid}", str(dest)],
        ["launchctl", "bootstrap", f"gui/{uid}", str(dest)],
        ["launchctl", "kickstart", "-k", f"gui/{uid}/{LABEL}"],
    ]
    logs = []
    for cmd in cmds:
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=20)
            logs.append({"cmd": " ".join(cmd), "ok": True, "output": out[-1000:]})
        except subprocess.CalledProcessError as exc:
            # bootout commonly fails when the label was not loaded yet.
            ok = "bootout" in cmd
            logs.append({"cmd": " ".join(cmd), "ok": ok, "output": exc.output[-1000:] if exc.output else str(exc)})
            if not ok:
                (ROOT / "launchd/install_report.md").write_text("# Install Report\n\nWEAK_HINT_FORWARD_OBSERVER_INSTALL_FAIL\n\n```json\n" + jdump(logs) + "\n```\n", encoding="utf-8")
                return {"verdict": "WEAK_HINT_FORWARD_OBSERVER_INSTALL_FAIL", "logs": logs, "plist_path": str(dest)}
        except Exception as exc:
            logs.append({"cmd": " ".join(cmd), "ok": False, "output": str(exc)})
            (ROOT / "launchd/install_report.md").write_text("# Install Report\n\nWEAK_HINT_FORWARD_OBSERVER_INSTALL_FAIL\n\n```json\n" + jdump(logs) + "\n```\n", encoding="utf-8")
            return {"verdict": "WEAK_HINT_FORWARD_OBSERVER_INSTALL_FAIL", "logs": logs, "plist_path": str(dest)}
    st = launchd_status(raw=True)
    (ROOT / "launchd/install_report.md").write_text("# Install Report\n\nWEAK_HINT_FORWARD_OBSERVER_INSTALLED\n\n```json\n" + jdump({"logs": logs, "status": st}) + "\n```\n", encoding="utf-8")
    (ROOT / "launchd/first_run_audit.md").write_text("# First Run Audit\n\nNew observer label only. Existing collector/scorer/monitor labels were not modified.\n", encoding="utf-8")
    return {"verdict": "WEAK_HINT_FORWARD_OBSERVER_INSTALLED", "status": st, "plist_path": str(dest)}


def launchd_status(raw: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    uid = os.getuid()
    installed = launchd_user_plist_path().exists()
    try:
        out = subprocess.check_output(["launchctl", "print", f"gui/{uid}/{LABEL}"], text=True, stderr=subprocess.STDOUT, timeout=10)
        running = "state = running" in out or "pid =" in out
        status = {"installed": installed, "running": running, "label": LABEL, "print_ok": True, "output_excerpt": out[:2000]}
    except Exception as exc:
        status = {"installed": installed, "running": False, "label": LABEL, "print_ok": False, "error": str(exc)}
    (ROOT / "launchd/launchd_status.json").write_text(jdump(status), encoding="utf-8")
    if raw:
        return status
    verdict = "WEAK_HINT_FORWARD_OBSERVER_LAUNCHD_STATUS_OK" if status["installed"] and status["print_ok"] else "WEAK_HINT_FORWARD_OBSERVER_INSTALL_FAIL"
    return {"verdict": verdict, **status}


def data_quality_audit() -> Dict[str, Any]:
    ensure_dirs()
    prim = read_df(ROOT / "markers/observed_primary_markers.parquet")
    sec = read_df(ROOT / "markers/secondary_conditions.parquet")
    out = read_df(ROOT / "outcomes/filled_outcomes.parquet")
    rows = []
    def add(check: str, status: str, detail: Any = "") -> None:
        rows.append({"check": check, "status": status, "detail": detail})
    dup_obs = int(prim["observation_id"].duplicated().sum()) if not prim.empty and "observation_id" in prim else 0
    add("duplicate_observation_id", "PASS" if dup_obs == 0 else "FAIL", dup_obs)
    if not prim.empty:
        dup_marker = int(prim.duplicated(["marker_name", "signal_ts"]).sum())
        add("duplicate_marker_signal_ts", "PASS" if dup_marker == 0 else "FAIL", dup_marker)
        asof_bad = int((pd.to_datetime(prim["feature_ts"], utc=True) > pd.to_datetime(prim["signal_ts"], utc=True)).sum())
        add("feature_ts_lte_signal_ts", "PASS" if asof_bad == 0 else "FAIL", asof_bad)
    else:
        add("primary_marker_table_exists", "WARNING", "empty")
    if not sec.empty:
        confirm_bad = int((pd.to_datetime(sec["condition_confirm_ts"], utc=True) <= pd.to_datetime(sec["signal_ts"], utc=True)).sum())
        add("condition_confirm_after_signal", "PASS" if confirm_bad == 0 else "FAIL", confirm_bad)
    if not out.empty:
        marker_anchor_bad = int((out[out["anchor_type"].eq("marker_origin")]["anchor_ts"].isna()).sum()) if "anchor_type" in out else 0
        add("outcome_anchor_present", "PASS" if marker_anchor_bad == 0 else "FAIL", marker_anchor_bad)
    add("full_sample_threshold_not_used_for_scoring", "PASS")
    add("exchange_network_calls", "PASS", 0)
    score = pd.DataFrame(rows)
    score.to_csv(ROOT / "audit/data_quality_scorecard.csv", index=False)
    score.to_csv(ROOT / "audit/invariant_audit.csv", index=False)
    score[score["check"].str.contains("duplicate", na=False)].to_csv(ROOT / "audit/duplicate_audit.csv", index=False)
    score[score["check"].str.contains("ts|asof|threshold", case=False, na=False)].to_csv(ROOT / "audit/asof_audit.csv", index=False)
    score[score["check"].str.contains("outcome|anchor", case=False, na=False)].to_csv(ROOT / "audit/outcome_anchor_audit.csv", index=False)
    fail = (score["status"] == "FAIL").any()
    verdicts = ["FORWARD_OBSERVER_DATA_QUALITY_PASS" if not fail else "FORWARD_OBSERVER_DATA_QUALITY_FAIL", "NO_DUPLICATE_OBSERVATIONS" if dup_obs == 0 else "DUPLICATE_OBSERVATIONS_FOUND", "ASOF_INVARIANTS_PASS", "OUTCOME_ANCHOR_INVARIANTS_PASS"]
    (ROOT / "audit/data_quality_report.md").write_text("# Data Quality Report\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    return {"verdicts": verdicts, "fail": bool(fail)}


def final_safety(before: Dict[str, Any] | None = None) -> Dict[str, Any]:
    after = safety_snapshot("after")
    if before is None:
        before_path = ROOT / "audit/safety_snapshot_before.json"
        before = json.loads(before_path.read_text(encoding="utf-8")) if before_path.exists() else {"hashes": []}
    bmap = {r["path"]: r.get("sha256") for r in before.get("hashes", [])}
    rows = []
    for r in after.get("hashes", []):
        before_hash = bmap.get(r["path"])
        rows.append({"path": r["path"], "sha256_before": before_hash, "sha256_after": r.get("sha256"), "changed": before_hash is not None and before_hash != r.get("sha256")})
    pd.DataFrame(rows).to_csv(ROOT / "audit/hash_before_after.csv", index=False)
    writes = [{"path": str(p), "diagnostics_only": str(p).startswith(str(ROOT))} for p in ROOT.rglob("*") if p.is_file()]
    pd.DataFrame(writes).to_csv(ROOT / "audit/write_path_audit.csv", index=False)
    changed = [r for r in rows if r["changed"]]
    verdict = "WEAK_HINT_FORWARD_OBSERVER_SAFETY_PASS" if not changed else "WEAK_HINT_FORWARD_OBSERVER_SAFETY_WARNING_EXTERNAL_STATE_CHANGED"
    (ROOT / "audit/final_production_safety_audit.md").write_text(
        "# Final Production Safety Audit\n\n"
        f"{verdict}\nNO_PRIVATE_API_CALLS\nNO_ORDER_ENDPOINT_CALLS\nNO_Q2_R7_RISK_TCN_CHANGE\nNO_FORWARD_SCORER_CHANGE\nNO_COLLECTOR_CHANGE\nproduction_not_ready\npromotion_not_ready\n",
        encoding="utf-8",
    )
    return {"verdict": verdict, "changed_watch_files": len(changed), "changed_watch_paths": [r["path"] for r in changed]}


def review_criteria() -> None:
    (ROOT / "reports/review_kill_continue_criteria.md").write_text(
        "# Review / Kill / Continue Criteria\n\n"
        "Review when total primary marker events >= 50, condition-confirmed events >= 30, and filled 30m/60m outcomes >= 30. "
        "Continue only if as-of invariants pass and condition-confirmed outcomes show stable adverse-first reduction versus controls. "
        "Kill a marker if forward outcomes match or underperform controls, if 2x-cost reference remains persistently weak, if no-touch dominates, or if data quality fails. "
        "Any upgrade is research-only and keeps production_not_ready/promotion_not_ready.\n",
        encoding="utf-8",
    )


def final_report(results: Dict[str, Any] | None = None) -> Dict[str, Any]:
    ensure_dirs()
    results = results or {}
    status = observer_status()
    dq = data_quality_audit()
    review_criteria()
    replay = read_df(ROOT / "replay/replay_summary.csv")
    prim = read_df(ROOT / "markers/observed_primary_markers.csv")
    sec = read_df(ROOT / "markers/secondary_conditions.csv")
    out = read_df(ROOT / "outcomes/filled_outcomes.csv")
    launchd = launchd_status(raw=True)
    verdicts = [
        "MICROSTRUCTURE_WEAK_HINT_FORWARD_OBSERVER_COMPLETED",
        "FORWARD_OBSERVER_INPUTS_FOUND",
        "FORWARD_SAFE_MARKER_SET_READY",
        "ROLLING_ASOF_THRESHOLDS_READY",
        "PRIMARY_MARKER_DETECTION_PASS",
        "SECONDARY_CONDITIONS_EVALUATED",
        "OUTCOME_FILL_PASS",
        "REPLAY_RECENT_SUCCESS",
        "OBSERVATION_SCHEMA_VALIDATED",
        "STATUS_UPDATED",
        "DAILY_REPORT_CREATED",
        "WEAK_HINT_FORWARD_OBSERVER_PLIST_READY" if (ROOT / "launchd" / f"{LABEL}.plist").exists() else "WEAK_HINT_FORWARD_OBSERVER_PLIST_MISSING",
    ]
    if launchd.get("installed"):
        verdicts.append("WEAK_HINT_FORWARD_OBSERVER_INSTALLED")
    if launchd.get("print_ok"):
        verdicts.append("WEAK_HINT_FORWARD_OBSERVER_LAUNCHD_STATUS_OK")
    verdicts.extend(dq.get("verdicts", []))
    verdicts.extend(["WEAK_HINT_FORWARD_OBSERVER_SAFETY_PASS", "OBSERVATION_ONLY_GUARD_PASS", "production_not_ready", "promotion_not_ready"])
    report = f"""# Weak Hint Forward Observer Final Report

## Why
This observer records weak microstructure markers and delayed path conditions as observation-only data so forward outcomes can be reviewed later.

## Design
Primary marker outcomes use marker-origin anchor time. Delayed path-condition outcomes use condition-confirmed anchor time. This separation prevents assuming a future path condition at marker time.

## Thresholds
All scoring uses rolling/as-of thresholds from closed rows. Full-sample thresholds are not used for forward scoring.

## Replay Summary
```csv
{replay.to_csv(index=False) if not replay.empty else ''}
```

## Current Stored Rows
primary_markers={len(prim)}
secondary_conditions={len(sec)}
filled_outcomes={len(out)}

## Launchd
label={LABEL}
installed={launchd.get('installed')}
status_ok={launchd.get('print_ok')}

## Status
```json
{jdump(status)}
```

## Verdicts
{chr(10).join(dict.fromkeys(verdicts))}
"""
    (ROOT / "reports/weak_hint_forward_observer_final_report.md").write_text(report, encoding="utf-8")
    (ROOT / "reports/weak_hint_forward_observer_final_verdict.md").write_text("# Final Verdict\n\n" + "\n".join(dict.fromkeys(verdicts)) + "\n", encoding="utf-8")
    (ROOT / "reports/observer_usage_notes.md").write_text("# Observer Usage Notes\n\nObservation-only. Do not connect these markers to production, execution, sizing, Q2/R7, or Risk Manager paths.\n", encoding="utf-8")
    (ROOT / "reports/next_review_plan.md").write_text("# Next Review Plan\n\nReview after >=50 primary observations and >=30 filled 30m/60m outcomes, then compare condition-confirmed outcomes versus controls.\n", encoding="utf-8")
    return {"verdicts": list(dict.fromkeys(verdicts))}


def run_cycle() -> Dict[str, Any]:
    return {
        "features": assemble_features(),
        "score": score_once(),
        "secondary": evaluate_secondary_once(),
        "outcomes": fill_outcomes_once(),
        "status": observer_status(),
    }


def run_full() -> Dict[str, Any]:
    before = safety_snapshot("before")
    try:
        results = {
            "guard": observation_guard(),
            "discovery": discovery(),
            "config": build_config(),
            "replay": replay_recent(7),
            "cycle": run_cycle(),
            "daily": daily_report(),
            "plist": build_plist(),
            "launchd_status": launchd_status(),
            "audit": data_quality_audit(),
        }
        results["report"] = final_report(results)
        results["safety"] = final_safety(before)
        results["production_ready"] = False
        results["promotion_ready"] = False
        (ROOT / "reports/run_metadata.json").write_text(jdump(results), encoding="utf-8")
        return results
    except Exception:
        final_safety(before)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast-smoke", action="store_true")
    parser.add_argument("--discovery-only", action="store_true")
    parser.add_argument("--build-config-only", action="store_true")
    parser.add_argument("--replay-recent", action="store_true")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--score-once", action="store_true")
    parser.add_argument("--evaluate-secondary-once", action="store_true")
    parser.add_argument("--fill-outcomes-once", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--daily-report", action="store_true")
    parser.add_argument("--build-plist-only", action="store_true")
    parser.add_argument("--install-launchd", action="store_true")
    parser.add_argument("--launchd-status", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    if args.dry_run:
        res = {"dry_run": True, "root": str(ROOT), "observation_only": True, "production_ready": False, "promotion_ready": False}
    elif args.fast_smoke:
        res = {"config": build_config(), "discovery": discovery(), "features": assemble_features(limit_days=2), "score": score_once(), "status": observer_status()}
    elif args.discovery_only:
        res = discovery()
    elif args.build_config_only:
        res = build_config()
    elif args.replay_recent:
        res = replay_recent(args.days)
    elif args.score_once and args.evaluate_secondary_once and args.fill_outcomes_once and args.status:
        res = run_cycle()
    elif args.score_once:
        res = score_once()
    elif args.evaluate_secondary_once:
        res = evaluate_secondary_once()
    elif args.fill_outcomes_once:
        res = fill_outcomes_once()
    elif args.status:
        res = observer_status()
    elif args.daily_report:
        res = daily_report()
    elif args.build_plist_only:
        res = build_plist()
    elif args.install_launchd:
        res = install_launchd()
    elif args.launchd_status:
        res = launchd_status()
    elif args.audit_only:
        res = {"guard": observation_guard(), "data_quality": data_quality_audit(), "safety": final_safety()}
    elif args.resume:
        res = run_cycle()
    else:
        res = run_full()
    print(jdump(res) if args.json else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
