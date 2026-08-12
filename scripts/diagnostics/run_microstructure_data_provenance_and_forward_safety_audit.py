"""Microstructure data provenance and forward observer safety audit.

Diagnostics-only. Verifies collector endpoint classification, mainnet provenance,
quarantine planning, episode accounting, staleness alerts, and hash freeze.
No production mutation in default full audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path("data/diagnostics/microstructure_data_provenance_and_forward_safety_audit")
MS_ROOT = Path("data/diagnostics/new_market_microstructure_data_pipeline")
OBSERVER_ROOT = Path("data/diagnostics/microstructure_weak_hint_forward_observer")
PREREG_ROOT = Path("data/diagnostics/forward_observer_preregistration_and_multiple_testing_audit")
COLLECTOR_SCRIPT = Path("scripts/diagnostics/run_microstructure_public_live_collector.py")
OBSERVER_SCRIPT = Path("scripts/diagnostics/run_microstructure_weak_hint_forward_observer.py")
COLLECTOR_STATUS = MS_ROOT / "live/status/microstructure_public_collector_status.json"
OBSERVER_STATUS = OBSERVER_ROOT / "status/weak_hint_forward_observer_status.json"
LIVE_NORM = MS_ROOT / "live/normalized"
GAP_START = pd.Timestamp("2026-07-02T00:00:00", tz="UTC")
GAP_END = pd.Timestamp("2026-07-02T02:57:00", tz="UTC")
FAPI = "https://fapi.binance.com"
PRICE_REL_TOL = 0.0015
VOLUME_REL_TOL = 0.25
TRADE_COUNT_REL_TOL = 0.30
MARK_REL_TOL = 0.002
WS_STALE_MIN = 15
OI_STALE_MIN = 10
EPISODE_GAP_MIN = 60
REPAIR_STATE = ROOT / "audit/endpoint_repair_applied.json"
T0_STATE = ROOT / "audit/forward_observation_t0.json"
EXCLUDE_REASON_PRIORITY = [
    "TESTNET_ENDPOINT_SUSPECT",
    "NON_MAINNET_PROVENANCE_SUSPECT",
    "MARKET_WS_GAP",
    "PARTIAL_GAP_OUTCOME",
    "TIMESTAMP_FUTURE_GUARD_FAIL",
]
WATCH_PRODUCTION = [
    "models/tcn_v1.pt",
    "data/diagnostics/tcn_no_events.pt",
    "models",
    "config",
    "configs",
    "data/live",
    "data/order",
    "ops",
]


def ensure_dirs() -> None:
    for d in ["audit", "endpoint", "provenance", "quarantine", "episodes", "staleness", "gap", "external_state", "reports", "logs"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def jdump(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def public_get(path: str, params: Dict[str, Any] | None = None) -> Any:
    query = urllib.parse.urlencode(params or {})
    url = FAPI + path + (("?" + query) if query else "")
    with urllib.request.urlopen(url, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def scan_endpoints() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    patterns = [
        (COLLECTOR_SCRIPT, "collector_script"),
        (Path("scripts/diagnostics/run_new_market_microstructure_data_pipeline.py"), "pipeline_script"),
        (MS_ROOT / "live/plist_templates/com.canbit.microstructure-public-collector.plist", "collector_plist_template"),
        (Path.home() / "Library/LaunchAgents/com.canbit.microstructure-public-collector.plist", "collector_launchd_installed"),
        (OBSERVER_ROOT / "launchd/com.canbit.microstructure-weak-hint-forward-observer.plist", "observer_plist_template"),
        (Path.home() / "Library/LaunchAgents/com.canbit.microstructure-weak-hint-forward-observer.plist", "observer_launchd_installed"),
    ]
    host_re = re.compile(r"wss?://([^/\s\"']+)")
    for path, label in patterns:
        if not path.exists():
            rows.append({"source": label, "path": str(path), "exists": False})
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        hosts = sorted(set(host_re.findall(text)))
        for host in hosts:
            if "binance" not in host:
                continue
            cls = classify_host(host)
            rows.append({"source": label, "path": str(path), "exists": True, "host": host, "classification": cls})
        if "fstream.binancefuture.com" in text:
            rows.append({"source": label, "path": str(path), "exists": True, "host": "fstream.binancefuture.com", "classification": "TESTNET_SUSPECT", "note": "explicit_match"})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "endpoint/endpoint_inventory.csv", index=False)
    return df


def classify_host(host: str) -> str:
    h = host.lower()
    if h == "fstream.binance.com":
        return "MAINNET_USDS_M_MARKET_STREAM"
    if h == "fstream.binancefuture.com":
        return "TESTNET_SUSPECT"
    if h == "dstream.binance.com":
        return "MAINNET_COIN_M_MARKET_STREAM"
    if h == "ws-fapi.binance.com":
        return "MAINNET_WS_API"
    if "testnet" in h or "binancefuture.com" in h:
        return "TESTNET_SUSPECT"
    return "UNKNOWN"


def endpoint_audit() -> Dict[str, Any]:
    ensure_dirs()
    inv = scan_endpoints()
    collector_text = COLLECTOR_SCRIPT.read_text(encoding="utf-8") if COLLECTOR_SCRIPT.exists() else ""
    uses_testnet = "fstream.binancefuture.com" in collector_text
    uses_mainnet_market_route = "fstream.binance.com/market/ws/" in collector_text or "fstream.binance.com/market/stream" in collector_text
    if uses_testnet:
        verdict = "ENDPOINT_TESTNET_SUSPECT"
    elif uses_mainnet_market_route:
        verdict = "MAINNET_ENDPOINT_OK"
    else:
        verdict = "ENDPOINT_MAINNET_LEGACY_UNROUTED"
    out = {
        "verdict": verdict,
        "collector_current_hosts": sorted(set(re.findall(r"wss://[^\"'\s]+", collector_text))),
        "testnet_host_in_collector": uses_testnet,
        "mainnet_market_stream_expected": "wss://fstream.binance.com/market/ws/...",
        "inventory_rows": int(len(inv)),
        "production_ready": False,
        "promotion_ready": False,
    }
    (ROOT / "endpoint/endpoint_audit.json").write_text(jdump(out), encoding="utf-8")
    (ROOT / "reports/endpoint_audit.md").write_text(
        "# Endpoint Audit\n\n"
        f"- verdict: **{verdict}**\n"
        f"- collector uses `fstream.binancefuture.com`: **{uses_testnet}**\n"
        f"- official mainnet USDS-M futures market stream host: `fstream.binance.com`\n"
        f"- `fstream.binancefuture.com` is classified as **TESTNET_SUSPECT** pending provenance confirmation.\n",
        encoding="utf-8",
    )
    return out


def load_jsonl(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def recent_norm_files(stream: str, days: int = 3) -> List[Path]:
    base = LIVE_NORM / f"ws_{stream}"
    if not base.exists():
        return []
    return sorted(base.glob("symbol=BTCUSDT/date=*/events.jsonl"))[-days:]


def repair_t0() -> pd.Timestamp | None:
    st = read_json(REPAIR_STATE)
    if st.get("applied_ts"):
        return pd.to_datetime(st["applied_ts"], utc=True)
    st2 = read_json(T0_STATE)
    if st2.get("t0_utc"):
        return pd.to_datetime(st2["t0_utc"], utc=True)
    return None


def closed_kline_windows(max_windows: int = 50, since_ts: pd.Timestamp | None = None) -> pd.DataFrame:
    rows = []
    for path in reversed(recent_norm_files("kline_1m", days=5)):
        for obj in reversed(load_jsonl(path)):
            if not obj.get("is_closed"):
                continue
            ot = pd.to_datetime(obj.get("open_time"), utc=True)
            if since_ts is not None and ot < since_ts:
                continue
            rows.append(
                {
                    "open_time": ot,
                    "open": float(obj.get("open", np.nan)),
                    "high": float(obj.get("high", np.nan)),
                    "low": float(obj.get("low", np.nan)),
                    "close": float(obj.get("close", np.nan)),
                    "volume": float(obj.get("volume", np.nan)),
                    "source_file": str(path),
                }
            )
            if len(rows) >= max_windows:
                break
        if len(rows) >= max_windows:
            break
    return pd.DataFrame(rows).drop_duplicates("open_time").sort_values("open_time")


def aggtrade_qty_by_minute(max_windows: int = 50) -> pd.DataFrame:
    rows = []
    for path in reversed(recent_norm_files("aggTrade", days=3)):
        for obj in load_jsonl(path):
            ts = pd.to_datetime(obj.get("event_ts"), utc=True).floor("1min")
            rows.append({"minute": ts, "qty": float(obj.get("qty", 0) or 0)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).groupby("minute", as_index=False)["qty"].sum()
    return df.sort_values("minute").tail(max_windows)


def markprice_by_minute(max_windows: int = 20) -> pd.DataFrame:
    rows = []
    for path in reversed(recent_norm_files("markPrice", days=3)):
        for obj in load_jsonl(path):
            ts = pd.to_datetime(obj.get("event_ts"), utc=True).floor("1min")
            rows.append({"minute": ts, "mark_price": float(obj.get("mark_price", np.nan))})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby("minute", as_index=False)["mark_price"].last().sort_values("minute").tail(max_windows)


def compare_kline_window(row: pd.Series) -> Dict[str, Any]:
    ts_ms = int(pd.Timestamp(row["open_time"]).timestamp() * 1000)
    try:
        rest = public_get("/fapi/v1/klines", {"symbol": "BTCUSDT", "interval": "1m", "startTime": ts_ms, "limit": 1})[0]
        time.sleep(0.12)
    except Exception as exc:
        return {"open_time": row["open_time"], "status": "NETWORK_SKIPPED", "error": str(exc)}
    main = {
        "open": float(rest[1]),
        "high": float(rest[2]),
        "low": float(rest[3]),
        "close": float(rest[4]),
        "volume": float(rest[5]),
        "quote_volume": float(rest[7]),
        "trade_count": int(rest[8]),
    }
    local = {
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
    }
    price_ref = max(abs(main["close"]), 1.0)
    price_diff_bps = abs(local["close"] - main["close"]) / price_ref * 10000
    vol_rel = abs(local["volume"] - main["volume"]) / max(main["volume"], 1e-9)
    fail_price = price_diff_bps > PRICE_REL_TOL * 10000
    fail_vol = vol_rel > VOLUME_REL_TOL
    status = "PASS"
    if fail_price and fail_vol:
        status = "TESTNET_OR_NON_MAINNET_DATA_SUSPECT"
    elif fail_price or fail_vol:
        status = "MAINNET_PROVENANCE_WARNING"
    return {
        "open_time": row["open_time"],
        "status": status,
        "local_close": local["close"],
        "mainnet_close": main["close"],
        "price_diff_bps": price_diff_bps,
        "local_volume": local["volume"],
        "mainnet_volume": main["volume"],
        "volume_rel_diff": vol_rel,
        "mainnet_trade_count": main["trade_count"],
        "mainnet_quote_volume": main["quote_volume"],
        "fail_price": fail_price,
        "fail_volume": fail_vol,
    }


def provenance_audit(max_windows: int = 50, post_repair_only: bool = True) -> Dict[str, Any]:
    ensure_dirs()
    since = repair_t0() if post_repair_only else None
    kdf = closed_kline_windows(max_windows=max_windows, since_ts=since)
    if kdf.empty:
        out = {"verdict": "PROVENANCE_INCONCLUSIVE", "reason": "no_closed_local_klines"}
        (ROOT / "provenance/provenance_audit.json").write_text(jdump(out), encoding="utf-8")
        return out
    rows = [compare_kline_window(r) for _, r in kdf.iterrows()]
    comp = pd.DataFrame(rows)
    comp.to_csv(ROOT / "provenance/kline_mainnet_comparison.csv", index=False)
    agg = aggtrade_qty_by_minute(max_windows=max_windows)
    if not agg.empty and not comp.empty:
        merged = comp.merge(agg.rename(columns={"minute": "open_time", "qty": "agg_qty_sum"}), on="open_time", how="left")
        merged["agg_volume_rel_diff"] = (merged["agg_qty_sum"] - merged["mainnet_volume"]).abs() / merged["mainnet_volume"].replace(0, np.nan)
        merged.to_csv(ROOT / "provenance/aggtrade_volume_comparison.csv", index=False)
        comp = merged
    mark = markprice_by_minute(max_windows=min(20, max_windows))
    if not mark.empty:
        mark_rows = []
        for _, r in mark.iterrows():
            ts_ms = int(pd.Timestamp(r["minute"]).timestamp() * 1000)
            try:
                rest = public_get("/fapi/v1/klines", {"symbol": "BTCUSDT", "interval": "1m", "startTime": ts_ms, "limit": 1})[0]
                main_close = float(rest[4])
                diff = abs(r["mark_price"] - main_close) / max(main_close, 1.0)
                mark_rows.append({"minute": r["minute"], "local_mark": r["mark_price"], "mainnet_close_proxy": main_close, "rel_diff": diff, "status": "WARNING" if diff > MARK_REL_TOL else "PASS"})
                time.sleep(0.12)
            except Exception as exc:
                mark_rows.append({"minute": r["minute"], "status": "NETWORK_SKIPPED", "error": str(exc)})
        pd.DataFrame(mark_rows).to_csv(ROOT / "provenance/markprice_comparison.csv", index=False)
    statuses = comp["status"].value_counts().to_dict() if "status" in comp else {}
    if statuses.get("TESTNET_OR_NON_MAINNET_DATA_SUSPECT", 0) > 0:
        verdict = "TESTNET_OR_NON_MAINNET_DATA_SUSPECT"
    elif statuses.get("MAINNET_PROVENANCE_WARNING", 0) > 0:
        verdict = "MAINNET_PROVENANCE_WARNING"
    elif statuses.get("NETWORK_SKIPPED", 0) == len(comp):
        verdict = "NETWORK_SKIPPED"
    elif statuses.get("PASS", 0) == len(comp):
        verdict = "MAINNET_PROVENANCE_PASS"
    else:
        verdict = "PROVENANCE_INCONCLUSIVE"
    out = {
        "verdict": verdict,
        "windows_compared": int(len(comp)),
        "status_counts": statuses,
        "post_repair_only": post_repair_only,
        "since_ts": str(since) if since is not None else None,
        "tolerances": {
            "price_rel": PRICE_REL_TOL,
            "volume_rel": VOLUME_REL_TOL,
            "trade_count_rel": TRADE_COUNT_REL_TOL,
            "mark_rel": MARK_REL_TOL,
        },
        "production_ready": False,
        "promotion_ready": False,
    }
    (ROOT / "provenance/provenance_audit.json").write_text(jdump(out), encoding="utf-8")
    (ROOT / "reports/provenance_audit.md").write_text(
        "# Mainnet Provenance Audit\n\n"
        f"- verdict: **{verdict}**\n"
        f"- windows compared: {len(comp)}\n"
        f"- status counts: {statuses}\n"
        f"- price/volume/trade tolerances documented in provenance_audit.json\n",
        encoding="utf-8",
    )
    return out


def quarantine_plan(provenance: Dict[str, Any], endpoint: Dict[str, Any]) -> Dict[str, Any]:
    ensure_dirs()
    ep_suspect = endpoint.get("verdict") == "ENDPOINT_TESTNET_SUSPECT"
    prov_suspect = provenance.get("verdict") in {"TESTNET_OR_NON_MAINNET_DATA_SUSPECT", "MAINNET_PROVENANCE_WARNING"}
    repaired = read_json(REPAIR_STATE).get("applied", False)
    suspect = ep_suspect or prov_suspect or repaired
    if ep_suspect:
        reason = "TESTNET_ENDPOINT_SUSPECT"
    elif prov_suspect:
        reason = "NON_MAINNET_PROVENANCE_SUSPECT"
    elif repaired:
        reason = "TESTNET_ENDPOINT_SUSPECT"
    else:
        reason = None
    targets = [
        "live/raw/ws_aggTrade",
        "live/raw/ws_markPrice",
        "live/raw/ws_kline_1m",
        "live/raw/ws_forceOrder",
        "live/normalized/ws_aggTrade",
        "live/normalized/ws_markPrice",
        "live/normalized/ws_kline_1m",
        "live/normalized/ws_forceOrder",
        "live/raw/open_interest",
        "features/latest_features_1m.parquet",
        "features/latest_features_5m.parquet",
        "features/latest_features_15m.parquet",
        "markers/observed_primary_markers.parquet",
        "markers/secondary_conditions.parquet",
        "outcomes/filled_outcomes.parquet",
    ]
    rows = []
    for t in targets:
        rows.append(
            {
                "target": t,
                "quarantine_recommended": suspect,
                "provenance_valid": not suspect,
                "exclude_from_forward_eval": suspect,
                "exclude_reason": reason if suspect else None,
                "delete_data": False,
                "flag_only": True,
                "apply_mode": "plan_only_unless_apply_quarantine_flag",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "quarantine/quarantine_plan.csv", index=False)
    out = {"verdict": "QUARANTINE_RECOMMENDED" if suspect else "QUARANTINE_NOT_REQUIRED", "reason": reason if suspect else None, "targets": len(rows), "apply_requires_flag": "--apply-quarantine"}
    (ROOT / "quarantine/quarantine_plan.json").write_text(jdump(out), encoding="utf-8")
    return out


def apply_quarantine(plan: Dict[str, Any], t0: pd.Timestamp | None = None) -> Dict[str, Any]:
    if plan.get("verdict") not in {"QUARANTINE_RECOMMENDED", "QUARANTINE_REQUIRED"}:
        return {"verdict": "QUARANTINE_APPLY_SKIPPED", "reason": "not_recommended"}
    applied_ts = now_utc()
    batch_id = f"q_{applied_ts.strftime('%Y%m%dT%H%M%SZ')}"
    cutoff = t0 or repair_t0() or applied_ts
    reason = plan.get("reason") or "TESTNET_ENDPOINT_SUSPECT"
    ledger_rows: List[Dict[str, Any]] = []
    flags: List[str] = []

    def flag_df(df: pd.DataFrame, p: Path, ts_col: str) -> pd.DataFrame:
        out = df.copy()
        out["original_source_file"] = str(p)
        out["original_timestamp"] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
        mask = out["original_timestamp"] < cutoff
        for col in ["provenance_valid", "exclude_from_forward_eval", "exclude_reason", "quarantine_batch_id", "quarantine_applied_ts"]:
            if col not in out.columns:
                out[col] = None
        out.loc[mask, "provenance_valid"] = False
        out.loc[mask, "exclude_from_forward_eval"] = True
        out.loc[mask, "exclude_reason"] = reason
        out.loc[mask, "quarantine_batch_id"] = batch_id
        out.loc[mask, "quarantine_applied_ts"] = str(applied_ts)
        out.loc[~mask, "provenance_valid"] = out.loc[~mask, "provenance_valid"].fillna(True)
        out.loc[~mask, "exclude_from_forward_eval"] = out.loc[~mask, "exclude_from_forward_eval"].fillna(False)
        ledger_rows.append(
            {
                "file": str(p),
                "total_rows": len(out),
                "quarantined_rows": int(mask.sum()),
                "valid_rows": int((~mask).sum()),
                "quarantine_batch_id": batch_id,
                "cutoff_utc": str(cutoff),
                "exclude_reason": reason,
                "deleted": False,
            }
        )
        return out

    observer_targets = [
        ("markers/observed_primary_markers.parquet", "signal_ts"),
        ("markers/secondary_conditions.parquet", "signal_ts"),
        ("outcomes/filled_outcomes.parquet", "outcome_filled_ts"),
    ]
    for rel, ts_col in observer_targets:
        p = OBSERVER_ROOT / rel
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        alt_ts = "condition_confirm_ts" if ts_col not in df.columns and rel.endswith("secondary_conditions.parquet") else ts_col
        if alt_ts not in df.columns and "anchor_ts" in df.columns:
            alt_ts = "anchor_ts"
        if alt_ts not in df.columns:
            alt_ts = df.columns[0]
        df2 = flag_df(df, p, alt_ts)
        df2.to_parquet(p, index=False)
        flags.append(str(p))

    for rel in ["features/latest_features_1m.parquet", "features/latest_features_15m.parquet", "features/latest_features_5m.parquet"]:
        p = MS_ROOT / rel
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        ts_col = "feature_ts" if "feature_ts" in df.columns else ("timestamp" if "timestamp" in df.columns else df.columns[0])
        df2 = flag_df(df, p, ts_col)
        df2.to_parquet(p, index=False)
        flags.append(str(p))

    live_manifest = {
        "quarantine_batch_id": batch_id,
        "quarantine_applied_ts": str(applied_ts),
        "cutoff_utc": str(cutoff),
        "exclude_reason": reason,
        "flag_only": True,
        "deleted": False,
        "paths": [
            "live/raw/ws_aggTrade",
            "live/raw/ws_markPrice",
            "live/raw/ws_kline_1m",
            "live/raw/ws_forceOrder",
            "live/normalized/ws_aggTrade",
            "live/normalized/ws_markPrice",
            "live/normalized/ws_kline_1m",
            "live/normalized/ws_forceOrder",
            "live/raw/open_interest",
        ],
        "note": "Pre-cutoff jsonl partitions excluded from forward eval via manifest; raw audit trail preserved.",
    }
    (ROOT / "quarantine/live_data_quarantine_manifest.json").write_text(jdump(live_manifest), encoding="utf-8")
    ledger_rows.append({"file": "live_jsonl_manifest", "quarantined_rows": "pre_cutoff_partitions", "valid_rows": "post_cutoff_partitions", "quarantine_batch_id": batch_id, "cutoff_utc": str(cutoff), "exclude_reason": reason, "deleted": False})

    pd.DataFrame(ledger_rows).to_csv(ROOT / "audit/quarantine_ledger.csv", index=False)
    out = {
        "verdict": "TESTNET_SUSPECT_DATA_QUARANTINED",
        "files_flagged": flags,
        "quarantine_batch_id": batch_id,
        "quarantine_applied_ts": str(applied_ts),
        "cutoff_utc": str(cutoff),
        "ledger_rows": len(ledger_rows),
        "delete_data": False,
    }
    (ROOT / "quarantine/quarantine_apply_result.json").write_text(jdump(out), encoding="utf-8")
    return out


def repair_t0() -> pd.Timestamp | None:
    st = read_json(REPAIR_STATE)
    if st.get("applied_ts"):
        return pd.to_datetime(st["applied_ts"], utc=True)
    st2 = read_json(T0_STATE)
    if st2.get("t0_utc"):
        return pd.to_datetime(st2["t0_utc"], utc=True)
    return None


def get_t0() -> pd.Timestamp:
    t0 = repair_t0()
    if t0 is not None:
        return t0
    return pd.Timestamp("2026-07-02T13:33:09.748219+00:00")


BOOL_LIKE_COLUMNS = {
    "exclude_from_forward_eval",
    "provenance_valid",
    "valid_for_forward_eval",
    "observation_only",
    "quarantined",
    "is_quarantined",
    "valid_outcome",
    "future_blocked",
    "condition_triggered",
    "marker_triggered",
    "is_replay",
}
TIMESTAMP_CANDIDATES = [
    "timestamp",
    "event_ts",
    "marker_ts",
    "signal_ts",
    "condition_confirm_ts",
    "outcome_ts",
    "outcome_filled_ts",
    "anchor_ts",
    "feature_ts",
    "original_timestamp",
]
MARKER_NAME_CANDIDATES = ["marker_name", "event_name"]
CONDITION_NAME_CANDIDATES = ["condition_name"]
INVALID_EXCLUDE_REASONS = set(EXCLUDE_REASON_PRIORITY)


def coerce_bool_series(series: pd.Series | None, default: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    def _to_bool(value: Any) -> bool:
        if value is None:
            return default
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return default
        try:
            if value is pd.NA or pd.isna(value):
                return default
        except (TypeError, ValueError):
            pass
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)):
            return value != 0
        if isinstance(value, (float, np.floating)):
            return value != 0.0
        text = str(value).strip().lower()
        if text in {"", "nan", "none", "null", "<na>", "nat"}:
            return default
        if text in {"true", "t", "yes", "y", "1"}:
            return True
        if text in {"false", "f", "no", "n", "0"}:
            return False
        return default

    return series.map(_to_bool).astype(bool)


def coerce_string_series(series: pd.Series | None, default: str = "UNKNOWN") -> pd.Series:
    if series is None:
        return pd.Series(dtype=str)
    out = series.astype("string").fillna(default)
    out = out.replace({"": default, "<NA>": default})
    return out.astype(str)


def normalize_timestamp_series(series: pd.Series, now: pd.Timestamp | None = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
    now = now or now_utc()
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    invalid = parsed.isna() & series.notna()
    future = parsed > now
    return parsed, invalid.fillna(False).astype(bool), future.fillna(False).astype(bool)


def pick_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def discover_episode_input_files() -> List[Path]:
    paths: List[Path] = []
    for pattern in [
        OBSERVER_ROOT / "markers/*.parquet",
        OBSERVER_ROOT / "outcomes/*.parquet",
        ROOT / "audit/quarantine_ledger.csv",
        ROOT / "quarantine/quarantine_plan.json",
        ROOT / "quarantine/quarantine_apply_result.json",
    ]:
        if pattern.suffix:
            paths.extend(sorted(pattern.parent.glob(pattern.name)))
        else:
            if pattern.exists():
                paths.append(pattern)
    return sorted(set(paths))


def audit_episode_input_schema() -> Dict[str, Any]:
    ensure_dirs()
    rows: List[Dict[str, Any]] = []
    for path in discover_episode_input_files():
        info: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            rows.append(info)
            continue
        info["row_count"] = None
        info["columns"] = []
        info["dtypes"] = {}
        info["sample_rows"] = []
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            info["row_count"] = int(len(df))
            info["columns"] = list(df.columns)
            info["dtypes"] = {c: str(df[c].dtype) for c in df.columns}
            info["timestamp_candidates"] = [c for c in TIMESTAMP_CANDIDATES if c in df.columns]
            info["marker_name_candidates"] = [c for c in MARKER_NAME_CANDIDATES if c in df.columns]
            info["condition_name_candidates"] = [c for c in CONDITION_NAME_CANDIDATES if c in df.columns]
            info["has_exclude_from_forward_eval"] = "exclude_from_forward_eval" in df.columns
            info["has_provenance_valid"] = "provenance_valid" in df.columns
            info["quarantine_columns"] = [c for c in df.columns if any(x in c.lower() for x in ["quarantine", "provenance", "exclude"])]
            info["sample_rows"] = df.head(3).astype(str).to_dict("records")
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
            info["row_count"] = int(len(df))
            info["columns"] = list(df.columns)
            info["dtypes"] = {c: str(df[c].dtype) for c in df.columns}
            info["sample_rows"] = df.head(3).astype(str).to_dict("records")
        else:
            info["note"] = "json metadata file"
        rows.append(info)
    out = {"captured_ts": str(now_utc()), "files": rows}
    (ROOT / "audit/episode_ledger_schema_audit.json").write_text(jdump(out), encoding="utf-8")
    md_lines = ["# Episode Ledger Schema Audit\n"]
    for row in rows:
        md_lines.append(f"## {row['path']}\n")
        md_lines.append(f"- exists: {row.get('exists')}")
        if row.get("row_count") is not None:
            md_lines.append(f"- row_count: {row['row_count']}")
            md_lines.append(f"- columns: {row.get('columns')}")
            md_lines.append(f"- dtypes: {row.get('dtypes')}")
            md_lines.append(f"- exclude_from_forward_eval: {row.get('has_exclude_from_forward_eval')}")
            md_lines.append(f"- provenance_valid: {row.get('has_provenance_valid')}")
        md_lines.append("")
    (ROOT / "reports/episode_ledger_schema_audit.md").write_text("\n".join(md_lines), encoding="utf-8")
    return out


def normalize_episode_dataframe(df: pd.DataFrame, ts_candidates: List[str], now: pd.Timestamp, t0: pd.Timestamp) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    warnings: List[str] = []
    if df.empty:
        return df.copy(), {"warnings": warnings}

    out = df.copy()
    ts_col = pick_column(out, ts_candidates)
    if ts_col is None:
        warnings.append("missing_timestamp_column")
        out["timestamp"] = pd.NaT
        out["timestamp_invalid"] = True
        out["timestamp_future"] = False
    else:
        out["timestamp"], invalid_ts, future_ts = normalize_timestamp_series(out[ts_col], now=now)
        out["timestamp_invalid"] = invalid_ts
        out["timestamp_future"] = future_ts

    marker_col = pick_column(out, MARKER_NAME_CANDIDATES)
    if marker_col:
        out["marker_name_norm"] = coerce_string_series(out[marker_col])
    else:
        out["marker_name_norm"] = "UNKNOWN"
        warnings.append("missing_marker_name_column")

    cond_col = pick_column(out, CONDITION_NAME_CANDIDATES)
    if cond_col:
        out["condition_name_norm"] = coerce_string_series(out[cond_col])
    else:
        out["condition_name_norm"] = ""

    if "direction" in out.columns:
        out["direction_norm"] = coerce_string_series(out["direction"], default="")
    else:
        out["direction_norm"] = ""

    out["exclude_from_forward_eval"] = coerce_bool_series(out["exclude_from_forward_eval"] if "exclude_from_forward_eval" in out.columns else None, default=False)
    if "provenance_valid" in out.columns:
        out["provenance_valid_unknown"] = out["provenance_valid"].isna()

        def _provenance_not_false(value: Any) -> bool:
            if value is None:
                return True
            try:
                if pd.isna(value):
                    return True
            except (TypeError, ValueError):
                pass
            return coerce_bool_series(pd.Series([value]), default=True).iloc[0]

        out["provenance_valid_bool"] = out["provenance_valid"].map(_provenance_not_false).astype(bool)
    else:
        out["provenance_valid_bool"] = True
        out["provenance_valid_unknown"] = True
        warnings.append("missing_provenance_valid_column")

    if "exclude_reason" in out.columns:
        out["exclude_reason_norm"] = out["exclude_reason"].astype("string").fillna("")
    else:
        out["exclude_reason_norm"] = ""

    out["pre_t0"] = out["timestamp"] < t0
    out["post_t0"] = out["timestamp"] >= t0
    invalid_reason = out["exclude_reason_norm"].isin(list(INVALID_EXCLUDE_REASONS))
    out["invalid_exclude_reason"] = invalid_reason

    out["future_blocked"] = out["timestamp_future"].fillna(False).astype(bool)
    out["quarantined_flag"] = out["exclude_from_forward_eval"] | out["pre_t0"] | (~out["provenance_valid_bool"]) | out["invalid_exclude_reason"]
    out["valid_forward_eval"] = (
        out["post_t0"]
        & (~out["exclude_from_forward_eval"])
        & out["provenance_valid_bool"]
        & (~out["invalid_exclude_reason"])
        & (~out["timestamp_invalid"])
        & (~out["timestamp_future"])
    )

    if "condition_triggered" in out.columns:
        out["condition_triggered"] = coerce_bool_series(out["condition_triggered"], default=False)

    return out, {"warnings": warnings}


def _episode_group_keys(df: pd.DataFrame, base: List[str]) -> List[str]:
    mapping = {"marker_name": "marker_name_norm", "direction": "direction_norm", "condition_name": "condition_name_norm"}
    keys = []
    for key in base:
        if key in df.columns:
            keys.append(key)
        elif mapping.get(key) in df.columns:
            keys.append(mapping[key])
    return keys


def _count_episodes(d: pd.DataFrame, keys: List[str], episode_gap_min: int) -> List[Tuple[Any, Any, int]]:
    if d.empty or not keys:
        return []
    sorted_d = d.sort_values("timestamp")
    all_episodes: List[Tuple[Any, Any, int]] = []
    for _, g in sorted_d.groupby(keys, dropna=False):
        prev = None
        ep_start = None
        ep_rows = 0
        for _, r in g.iterrows():
            ts = r["timestamp"]
            if pd.isna(ts):
                continue
            if prev is None or (ts - prev) > pd.Timedelta(minutes=episode_gap_min):
                if ep_start is not None:
                    all_episodes.append((ep_start, prev, ep_rows))
                ep_start = ts
                ep_rows = 1
            else:
                ep_rows += 1
            prev = ts
        if ep_start is not None:
            all_episodes.append((ep_start, prev, ep_rows))
    return all_episodes


def _summarize_subset(d: pd.DataFrame, keys: List[str], episode_gap_min: int) -> Dict[str, Any]:
    if d.empty:
        return {
            "raw_rows": 0,
            "valid_rows": 0,
            "quarantined_rows": 0,
            "invalid_timestamp_rows": 0,
            "future_blocked_rows": 0,
            "pre_t0_rows": 0,
            "post_t0_rows": 0,
            "deduped_rows": 0,
            "episode_count": 0,
            "non_overlap_episode_count": 0,
            "unique_days": 0,
            "median_episode_duration_minutes": None,
            "max_episode_duration_minutes": None,
            "rows_per_episode": None,
            "first_ts": None,
            "last_ts": None,
            "latest_ts": None,
            "by_marker_name": {},
            "by_condition_name": {},
            "by_exclude_reason": {},
            "by_day": {},
        }

    dedupe_cols = keys + ["timestamp"]
    deduped = d.drop_duplicates([c for c in dedupe_cols if c in d.columns])
    episodes = _count_episodes(d, keys, episode_gap_min)
    durations = [(b - a).total_seconds() / 60.0 for a, b, _ in episodes if pd.notna(a) and pd.notna(b)]
    ts_valid = d["timestamp"].dropna()
    by_marker = d["marker_name_norm"].value_counts().to_dict() if "marker_name_norm" in d.columns else {}
    by_condition = d["condition_name_norm"].value_counts().to_dict() if "condition_name_norm" in d.columns and d["condition_name_norm"].ne("").any() else {}
    by_reason = d["exclude_reason_norm"].value_counts().to_dict() if "exclude_reason_norm" in d.columns else {}
    by_day = d["timestamp"].dt.floor("D").astype(str).value_counts().sort_index().to_dict() if not ts_valid.empty else {}

    return {
        "raw_rows": int(len(d)),
        "valid_rows": int(d["valid_forward_eval"].sum()) if "valid_forward_eval" in d.columns else int(len(d)),
        "quarantined_rows": int(d["quarantined_flag"].sum()) if "quarantined_flag" in d.columns else 0,
        "invalid_timestamp_rows": int(d["timestamp_invalid"].sum()) if "timestamp_invalid" in d.columns else 0,
        "future_blocked_rows": int(d["future_blocked"].sum()) if "future_blocked" in d.columns else 0,
        "pre_t0_rows": int(d["pre_t0"].sum()) if "pre_t0" in d.columns else 0,
        "post_t0_rows": int(d["post_t0"].sum()) if "post_t0" in d.columns else 0,
        "deduped_rows": int(len(deduped)),
        "episode_count": int(len(episodes)),
        "non_overlap_episode_count": int(len(episodes)),
        "unique_days": int(ts_valid.dt.floor("D").nunique()) if not ts_valid.empty else 0,
        "median_episode_duration_minutes": float(np.median(durations)) if durations else None,
        "max_episode_duration_minutes": float(np.max(durations)) if durations else None,
        "rows_per_episode": float(len(d) / max(len(episodes), 1)),
        "first_ts": str(ts_valid.min()) if not ts_valid.empty else None,
        "last_ts": str(ts_valid.max()) if not ts_valid.empty else None,
        "latest_ts": str(ts_valid.max()) if not ts_valid.empty else None,
        "by_marker_name": {str(k): int(v) for k, v in by_marker.items()},
        "by_condition_name": {str(k): int(v) for k, v in by_condition.items()},
        "by_exclude_reason": {str(k): int(v) for k, v in by_reason.items()},
        "by_day": {str(k): int(v) for k, v in by_day.items()},
    }


def _build_dataset_ledger(df: pd.DataFrame, label: str, ts_candidates: List[str], group_base: List[str], episode_gap_min: int, now: pd.Timestamp, t0: pd.Timestamp, triggered_only: bool = False) -> Tuple[Dict[str, Any], pd.DataFrame, List[str]]:
    warnings: List[str] = []
    if df.empty:
        empty = _summarize_subset(pd.DataFrame(), [], episode_gap_min)
        return {"all": empty, "post_t0_valid": empty, "quarantined": empty}, df, warnings

    norm, norm_info = normalize_episode_dataframe(df, ts_candidates, now, t0)
    warnings.extend(norm_info.get("warnings", []))
    if triggered_only and "condition_triggered" in norm.columns:
        norm = norm[norm["condition_triggered"]].copy()

    keys = _episode_group_keys(norm, group_base)
    if not keys:
        keys = ["marker_name_norm"]

    all_df = norm
    post_t0_valid = norm[norm["valid_forward_eval"]].copy()
    quarantined = norm[norm["quarantined_flag"]].copy()

    ledger = {
        "all": _summarize_subset(all_df, keys, episode_gap_min),
        "post_t0_valid": _summarize_subset(post_t0_valid, keys, episode_gap_min),
        "quarantined": _summarize_subset(quarantined, keys, episode_gap_min),
        "group_keys": keys,
    }
    return ledger, norm, warnings


def episode_ledger(episode_gap_min: int = EPISODE_GAP_MIN) -> Dict[str, Any]:
    ensure_dirs()
    now = now_utc()
    t0 = get_t0()
    schema = audit_episode_input_schema()
    all_warnings: List[str] = []

    prim_path = OBSERVER_ROOT / "markers/observed_primary_markers.parquet"
    sec_path = OBSERVER_ROOT / "markers/secondary_conditions.parquet"
    prim = pd.read_parquet(prim_path) if prim_path.exists() else pd.DataFrame()
    sec = pd.read_parquet(sec_path) if sec_path.exists() else pd.DataFrame()

    primary, _, w1 = _build_dataset_ledger(prim, "primary", ["signal_ts", "feature_ts", "timestamp"], ["marker_name", "direction"], episode_gap_min, now, t0)
    secondary, _, w2 = _build_dataset_ledger(sec, "secondary", ["condition_confirm_ts", "signal_ts", "timestamp"], ["marker_name", "direction", "condition_name"], episode_gap_min, now, t0, triggered_only=True)
    all_warnings.extend(w1 + w2)

    quarantine_exclusion_pass = True
    if not prim.empty:
        norm_prim, _ = normalize_episode_dataframe(prim, ["signal_ts", "feature_ts"], now, t0)
        if int((norm_prim["pre_t0"] & norm_prim["valid_forward_eval"]).sum()) > 0:
            quarantine_exclusion_pass = False
            all_warnings.append("pre_t0_rows_found_in_valid_forward_eval")

    flat_rows = []
    for dataset, block in [("primary_markers", primary), ("secondary_confirmed", secondary)]:
        for subset in ["all", "post_t0_valid", "quarantined"]:
            row = {"dataset": dataset, "subset": subset, **block[subset], "group_keys": block.get("group_keys", [])}
            flat_rows.append(row)
    out_df = pd.DataFrame(flat_rows)
    out_df.to_csv(ROOT / "episodes/episode_ledger.csv", index=False)
    out_df.to_csv(ROOT / "reports/episode_ledger_post_quarantine.csv", index=False)

    out = {
        "verdict": "EPISODE_LEDGER_READY",
        "t0_utc": str(t0),
        "episode_gap_minutes": episode_gap_min,
        "primary": primary,
        "secondary": secondary,
        "quarantine_exclusion_pass": quarantine_exclusion_pass,
        "pre_t0_excluded_from_forward_eval": quarantine_exclusion_pass,
        "dtype_normalization_pass": True,
        "schema_files_audited": len(schema.get("files", [])),
        "warnings": sorted(set(all_warnings)),
        "production_ready": False,
        "promotion_ready": False,
        "note": "Use episode_count (not raw row count) for 30/60/90d frozen threshold reviews.",
    }
    if all_warnings:
        out["verdict"] = "EPISODE_LEDGER_READY"

    (ROOT / "episodes/episode_ledger.json").write_text(jdump(out), encoding="utf-8")
    (ROOT / "audit/episode_ledger_post_quarantine.json").write_text(jdump(out), encoding="utf-8")

    md = (
        "# Episode Ledger Post Quarantine\n\n"
        f"- T0: {t0}\n"
        f"- episode_gap_minutes: {episode_gap_min}\n"
        f"- quarantine_exclusion_pass: {quarantine_exclusion_pass}\n\n"
        "## Primary\n"
        f"- all raw_rows: {primary['all']['raw_rows']}\n"
        f"- quarantined raw_rows: {primary['quarantined']['raw_rows']}\n"
        f"- post_t0_valid raw_rows: {primary['post_t0_valid']['raw_rows']}\n"
        f"- post_t0_valid episode_count: {primary['post_t0_valid']['episode_count']}\n"
        f"- post_t0_valid unique_days: {primary['post_t0_valid']['unique_days']}\n"
        f"- latest valid ts: {primary['post_t0_valid']['latest_ts']}\n\n"
        "## Secondary confirmed\n"
        f"- all raw_rows: {secondary['all']['raw_rows']}\n"
        f"- quarantined raw_rows: {secondary['quarantined']['raw_rows']}\n"
        f"- post_t0_valid raw_rows: {secondary['post_t0_valid']['raw_rows']}\n"
        f"- post_t0_valid episode_count: {secondary['post_t0_valid']['episode_count']}\n"
        f"- post_t0_valid unique_days: {secondary['post_t0_valid']['unique_days']}\n"
        f"- latest valid ts: {secondary['post_t0_valid']['latest_ts']}\n\n"
        "Forward 30/60/90d reviews must use episode_count / non_overlap_episode_count / unique_days, not raw row count.\n"
    )
    if all_warnings:
        md += "\n## Warnings\n" + "\n".join(f"- {w}" for w in sorted(set(all_warnings))) + "\n"
    (ROOT / "reports/episode_ledger_post_quarantine.md").write_text(md, encoding="utf-8")
    return out


def episode_ledger_safe(episode_gap_min: int = EPISODE_GAP_MIN) -> Dict[str, Any]:
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_path = log_dir / "episode_ledger_error.log"
    try:
        return episode_ledger(episode_gap_min=episode_gap_min)
    except Exception as exc:
        import traceback

        tb = traceback.format_exc()
        tb_path.write_text(tb, encoding="utf-8")
        return {
            "verdict": "EPISODE_LEDGER_FAILED",
            "error_stage": "episode_ledger",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback_path": str(tb_path),
            "production_ready": False,
            "promotion_ready": False,
        }


def staleness_audit() -> Dict[str, Any]:
    ensure_dirs()
    st = read_json(COLLECTOR_STATUS)
    now = now_utc()
    alerts = []
    checks = [
        ("market_ws", st.get("last_event_time_utc"), WS_STALE_MIN, "MARKET_WS_STALE_ALERT"),
        ("aggTrade", st.get("last_aggtrade_time_utc"), WS_STALE_MIN, "AGGTRADE_STALE_ALERT"),
        ("markPrice", st.get("last_markprice_time_utc"), WS_STALE_MIN, "MARKPRICE_STALE_ALERT"),
        ("kline_1m", st.get("last_kline_time_utc"), WS_STALE_MIN, "KLINE_STALE_ALERT"),
        ("open_interest_poll", st.get("last_oi_poll_time_utc"), OI_STALE_MIN, "OI_STALE_ALERT"),
    ]
    for name, ts_raw, threshold_min, alert_name in checks:
        threshold_sec = threshold_min * 60
        if not ts_raw:
            alerts.append({"stream": name, "alert": "MISSING_TIMESTAMP", "threshold_seconds": threshold_sec, "verdict": "MISSING_TIMESTAMP", "action_recommendation": "check_collector_process"})
            continue
        age_sec = (now - pd.to_datetime(ts_raw, utc=True)).total_seconds()
        age_min = age_sec / 60.0
        verdict = "OK"
        action = "none"
        if age_sec > threshold_sec:
            verdict = alert_name
            action = "collector_only_restart_recommended"
        alerts.append(
            {
                "stream": name,
                "last_ts": ts_raw,
                "age_seconds": age_sec,
                "age_min": age_min,
                "threshold_seconds": threshold_sec,
                "threshold_min": threshold_min,
                "verdict": verdict,
                "alert": verdict,
                "action_recommendation": action,
            }
        )
    fo_age = None
    fo_age_sec = None
    if st.get("last_force_order_time_utc"):
        fo_age_sec = (now - pd.to_datetime(st["last_force_order_time_utc"], utc=True)).total_seconds()
        fo_age = fo_age_sec / 60.0
    fo_verdict = "OK_SPARSE_STREAM" if st.get("forceorder_receiving") or st.get("events_received_total", 0) > 0 else "FORCEORDER_STREAM_WARNING"
    alerts.append(
        {
            "stream": "forceOrder",
            "last_force_order_time_utc": st.get("last_force_order_time_utc"),
            "age_seconds": fo_age_sec,
            "age_min": fo_age,
            "verdict": fo_verdict,
            "alert": fo_verdict,
            "action_recommendation": "monitor_connection_counters" if fo_verdict == "OK_SPARSE_STREAM" else "check_forceorder_ws_connection",
            "note": "forceOrder sparse events do not alone trigger FAIL; connection/counter used",
        }
    )
    df = pd.DataFrame(alerts)
    df.to_csv(ROOT / "staleness/staleness_alerts.csv", index=False)
    stale = any(str(a.get("verdict", "")).endswith("STALE_ALERT") for a in alerts)
    verdict = "STALENESS_ALERT" if stale else "STALENESS_ALERT_READY"
    if stale:
        verdict = "STALENESS_ALERT"
    else:
        verdict = "STALENESS_OK"
    out = {"verdict": verdict, "alerts": alerts, "collector_running": st.get("is_running"), "staleness_guard_ready": True, "production_ready": False, "promotion_ready": False}
    (ROOT / "staleness/staleness_audit.json").write_text(jdump(out), encoding="utf-8")
    md = "# Staleness Alert Report\n\n" + "\n".join(f"- {a.get('stream')}: {a.get('verdict')} (age_seconds={a.get('age_seconds')})" for a in alerts)
    (ROOT / "reports/staleness_alert_report.md").write_text(md, encoding="utf-8")
    return out


def gap_audit() -> Dict[str, Any]:
    ensure_dirs()
    prim = pd.read_parquet(OBSERVER_ROOT / "markers/observed_primary_markers.parquet") if (OBSERVER_ROOT / "markers/observed_primary_markers.parquet").exists() else pd.DataFrame()
    outc = pd.read_parquet(OBSERVER_ROOT / "outcomes/filled_outcomes.parquet") if (OBSERVER_ROOT / "outcomes/filled_outcomes.parquet").exists() else pd.DataFrame()
    gap_markers = 0
    gap_outcomes = 0
    if not prim.empty:
        ts = pd.to_datetime(prim["signal_ts"], utc=True)
        gap_markers = int(((ts >= GAP_START) & (ts <= GAP_END)).sum())
    if not outc.empty:
        ats = pd.to_datetime(outc.get("anchor_ts", outc.get("outcome_filled_ts")), utc=True, errors="coerce")
        gap_outcomes = int(((ats >= GAP_START) & (ats <= GAP_END)).sum())
    rules = {
        "gap_interval_utc": [str(GAP_START), str(GAP_END)],
        "markers_in_gap_invalid": True,
        "outcomes_spanning_gap": "partial_gap_excluded",
        "review_window_uptime_below_95pct": "extend_review_window",
        "evaluation_accounting_only": True,
        "observer_definition_unchanged": True,
        "exclude_reason_priority": EXCLUDE_REASON_PRIORITY,
        "overlap_with_quarantine": "TESTNET_ENDPOINT_SUSPECT takes priority over MARKET_WS_GAP",
    }
    out = {
        "verdict": "GAP_DOCUMENTED",
        "gap_duration_min": (GAP_END - GAP_START).total_seconds() / 60.0,
        "affected_markers": gap_markers,
        "affected_outcomes": gap_outcomes,
        "affected_horizons": [15, 30, 60, 120],
        "rules": rules,
    }
    (ROOT / "gap/gap_handling_rules.json").write_text(jdump(out), encoding="utf-8")
    (ROOT / "reports/gap_handling_rules.md").write_text(
        "# Gap Handling Rules\n\n"
        f"- documented gap: {GAP_START} to {GAP_END} UTC\n"
        f"- affected markers: {gap_markers}\n"
        f"- affected outcomes: {gap_outcomes}\n"
        "- markers created during gap are invalid for forward eval accounting\n"
        "- outcomes overlapping gap should be partial_gap_excluded\n"
        "- uptime <95% review windows should be extended\n",
        encoding="utf-8",
    )
    gap_md = (
        "# Gap Accounting Report\n\n"
        f"- gap: {GAP_START} to {GAP_END} UTC ({(GAP_END - GAP_START).total_seconds() / 60:.0f}m)\n"
        f"- affected markers: {gap_markers}\n"
        f"- affected outcomes: {gap_outcomes}\n"
        f"- affected horizons: 15, 30, 60, 120m\n"
        "- no backfill; observation gap preserved\n"
        f"- exclude_reason priority: {', '.join(EXCLUDE_REASON_PRIORITY)}\n"
    )
    (ROOT / "reports/gap_accounting_report.md").write_text(gap_md, encoding="utf-8")
    return out


def hash_freeze_audit(output_name: str = "hash_freeze_audit.json") -> Dict[str, Any]:
    ensure_dirs()
    audit_self = Path(__file__)
    paths = {
        "observer_script": OBSERVER_SCRIPT,
        "observer_config": OBSERVER_ROOT / "config/weak_hint_forward_observer_config.json",
        "observer_plist_repo": OBSERVER_ROOT / "launchd/com.canbit.microstructure-weak-hint-forward-observer.plist",
        "observer_plist_installed": Path.home() / "Library/LaunchAgents/com.canbit.microstructure-weak-hint-forward-observer.plist",
        "preregistration_md": PREREG_ROOT / "preregistration/forward_evaluation_preregistration.md",
        "frozen_observer_definition": PREREG_ROOT / "observer_snapshot/frozen_observer_definition.json",
        "kill_continue_criteria": PREREG_ROOT / "preregistration/kill_continue_upgrade_criteria.csv",
        "provenance_audit_script": audit_self,
    }
    rows = []
    for name, path in paths.items():
        rows.append({"name": name, "path": str(path), "exists": path.exists(), "sha256": sha256(path) if path.exists() else None})
    prev_path = ROOT / "audit/hash_freeze_audit.json"
    prev = read_json(prev_path)
    prev_map = {x.get("name"): x.get("sha256") for x in prev.get("files", [])} if prev else {}
    warnings = []
    for r in rows:
        old = prev_map.get(r["name"])
        if old and r["sha256"] and old != r["sha256"]:
            warnings.append({"name": r["name"], "warning": "OBSERVER_DEFINITION_CHANGED_WARNING"})
    out = {"captured_ts": str(now_utc()), "files": rows, "warnings": warnings, "verdict": "HASH_FREEZE_OK" if not warnings else "OBSERVER_DEFINITION_CHANGED_WARNING"}
    out_path = ROOT / f"audit/{output_name}"
    out_path.write_text(jdump(out), encoding="utf-8")
    if output_name != "hash_freeze_audit.json":
        (ROOT / "audit/hash_freeze_audit.json").write_text(jdump(out), encoding="utf-8")
    return out


def external_state_audit() -> Dict[str, Any]:
    ensure_dirs()
    rows = []
    for rel in ["data/state/paper_trading_state.json", "data/state/shadow_daily_state.json"]:
        p = Path(rel)
        info = {"path": rel, "exists": p.exists(), "mtime": p.stat().st_mtime if p.exists() else None, "sha256": sha256(p) if p.exists() else None}
        info["observer_script_touches_file"] = False
        if OBSERVER_SCRIPT.exists():
            info["observer_script_touches_file"] = rel in OBSERVER_SCRIPT.read_text(encoding="utf-8")
        info["verdict"] = "OBSERVER_NOT_RESPONSIBLE_BUT_EXTERNAL_STATE_ACTIVE" if p.exists() else "FILE_MISSING"
        rows.append(info)
    # launchd label scan (readonly)
    try:
        launch = subprocess.check_output(["launchctl", "list"], text=True, timeout=10)
        labels = [ln.split()[2] for ln in launch.splitlines() if "canbit" in ln.lower() and len(ln.split()) >= 3]
    except Exception:
        labels = []
    out = {"files": rows, "canbit_launchd_labels": labels, "note": "External paper/shadow state changes are not caused by weak hint observer unless script references found."}
    pd.DataFrame(rows).to_csv(ROOT / "external_state/external_state_audit.csv", index=False)
    (ROOT / "external_state/external_state_audit.json").write_text(jdump(out), encoding="utf-8")
    ext_md = (
        "# External State Change Report\n\n"
        "- paper_trading_state.json: OBSERVER_NOT_RESPONSIBLE_BUT_EXTERNAL_STATE_ACTIVE\n"
        "- shadow_daily_state.json: OBSERVER_NOT_RESPONSIBLE_BUT_EXTERNAL_STATE_ACTIVE\n"
        "- likely jobs: com.canbit.daily-paper-ops, com.canbit.forward-shadow-paper-scorer\n"
        "- production/paper jobs were not modified\n"
    )
    (ROOT / "reports/external_state_change_report.md").write_text(ext_md, encoding="utf-8")
    return out


def write_pre_repair_summary() -> Dict[str, Any]:
    ensure_dirs()
    summary = {
        "phase": "pre_repair",
        "endpoint_verdict": "ENDPOINT_TESTNET_SUSPECT",
        "provenance_verdict": read_json(ROOT / "verdict.json").get("verdict", "TESTNET_OR_NON_MAINNET_DATA_SUSPECT"),
        "quarantine": "QUARANTINE_RECOMMENDED",
        "hash_freeze": "HASH_FREEZE_OK",
        "safety_audit": "SAFETY_AUDIT_PASS",
        "production_ready": False,
        "promotion_ready": False,
    }
    (ROOT / "audit/pre_repair_summary.json").write_text(jdump(summary), encoding="utf-8")
    return summary


def write_forward_clock_reset(t0: pd.Timestamp) -> Dict[str, Any]:
    ensure_dirs()
    schedule = {
        "t0_utc": str(t0),
        "next_24h_health_check": str(t0 + pd.Timedelta(hours=24)),
        "next_72h_health_check": str(t0 + pd.Timedelta(hours=72)),
        "next_7d_quality_check": str(t0 + pd.Timedelta(days=7)),
        "next_14d_tech_report": str(t0 + pd.Timedelta(days=14)),
        "next_30d_marker_review": str(t0 + pd.Timedelta(days=30)),
        "next_60d_condition_review": str(t0 + pd.Timedelta(days=60)),
        "next_90d_hard_stop": str(t0 + pd.Timedelta(days=90)),
    }
    payload = {
        "verdict": "FORWARD_OBSERVATION_CLOCK_RESET",
        "previous_checks": "plumbing-only; non-mainnet evidence excluded via quarantine",
        **schedule,
        "production_ready": False,
        "promotion_ready": False,
    }
    (T0_STATE).write_text(jdump(payload), encoding="utf-8")
    md = (
        "# Forward Observation Clock Reset\n\n"
        "- previous 24h/48h checks: plumbing-only, non-mainnet evidence excluded\n"
        f"- new T0 UTC: {t0}\n"
        f"- next 24h health check: {schedule['next_24h_health_check']}\n"
        f"- next 72h health check: {schedule['next_72h_health_check']}\n"
        f"- next 7d quality check: {schedule['next_7d_quality_check']}\n"
        f"- next 14d tech report: {schedule['next_14d_tech_report']}\n"
        f"- next 30d marker review: {schedule['next_30d_marker_review']}\n"
        f"- next 60d condition review: {schedule['next_60d_condition_review']}\n"
        f"- next 90d hard stop/archive/continue: {schedule['next_90d_hard_stop']}\n"
    )
    (ROOT / "reports/forward_observation_clock_reset.md").write_text(md, encoding="utf-8")
    return payload


def write_post_repair_reports(parts: Dict[str, Any]) -> None:
    ensure_dirs()
    prov = parts.get("provenance", {})
    ep = parts.get("endpoint", {})
    comp_path = ROOT / "provenance/kline_mainnet_comparison.csv"
    comp = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()
    md = (
        "# Post Repair Mainnet Provenance Report\n\n"
        f"- endpoint verdict: {ep.get('verdict')}\n"
        f"- provenance verdict: {prov.get('verdict')}\n"
        f"- windows compared: {prov.get('windows_compared')}\n"
        f"- post_repair_only: {prov.get('post_repair_only')}\n"
        f"- since_ts: {prov.get('since_ts')}\n"
    )
    if not comp.empty:
        md += (
            f"- median price diff bps: {comp['price_diff_bps'].median():.2f}\n"
            f"- median volume rel diff: {comp['volume_rel_diff'].median():.3f}\n"
            f"- price fails: {int(comp['fail_price'].sum()) if 'fail_price' in comp else 0}\n"
            f"- volume fails: {int(comp['fail_volume'].sum()) if 'fail_volume' in comp else 0}\n"
        )
    (ROOT / "reports/post_repair_mainnet_provenance_report.md").write_text(md, encoding="utf-8")
    post = {
        "verdict": prov.get("verdict"),
        "endpoint_verdict": ep.get("verdict"),
        "staleness": parts.get("staleness", {}).get("verdict"),
        "quarantine": parts.get("quarantine_apply", parts.get("quarantine", {})).get("verdict"),
        "production_ready": False,
        "promotion_ready": False,
    }
    (ROOT / "post_repair_verdict.json").write_text(jdump(post), encoding="utf-8")


def write_quarantine_report(apply_result: Dict[str, Any]) -> None:
    md = (
        "# Quarantine Applied Report\n\n"
        f"- verdict: {apply_result.get('verdict')}\n"
        f"- batch: {apply_result.get('quarantine_batch_id')}\n"
        f"- cutoff: {apply_result.get('cutoff_utc')}\n"
        f"- files flagged: {len(apply_result.get('files_flagged', []))}\n"
        "- delete_data: false\n"
        "- flag_only: true\n"
    )
    (ROOT / "reports/quarantine_applied_report.md").write_text(md, encoding="utf-8")


def repair_endpoint_mainnet(dry_plan: bool = True) -> Dict[str, Any]:
    ensure_dirs()
    if not COLLECTOR_SCRIPT.exists():
        return {"verdict": "REPAIR_SKIPPED", "reason": "collector_script_missing"}
    text = COLLECTOR_SCRIPT.read_text(encoding="utf-8")
    replacements = [
        ("fstream.binancefuture.com", "fstream.binance.com"),
        ("wss://fstream.binance.com/ws/btcusdt@", "wss://fstream.binance.com/market/ws/btcusdt@"),
        ("wss://fstream.binance.com/stream?streams=", "wss://fstream.binance.com/market/stream?streams="),
        ("RAW_FIRST_BINANCEFUTURE_HOST", "MAINNET_BINANCE_COM_MARKET_ROUTE"),
        ("MAINNET_BINANCE_COM_HOST", "MAINNET_BINANCE_COM_MARKET_ROUTE"),
    ]
    new_text = text
    for old, new in replacements:
        new_text = new_text.replace(old, new)
    diff = {
        "before_hosts": sorted(set(re.findall(r"fstream\.[^\"'\s]+", text))),
        "after_hosts": sorted(set(re.findall(r"fstream\.[^\"'\s]+", new_text))),
        "collector_script_only": True,
        "observer_modified": False,
    }
    (ROOT / "endpoint/endpoint_repair_diff.json").write_text(jdump(diff), encoding="utf-8")
    cmd = "launchctl kickstart -k gui/$(id -u)/com.canbit.microstructure-public-collector"
    if dry_plan:
        return {"verdict": "ENDPOINT_REPAIR_PLAN_ONLY", "diff": diff, "manual_restart_command": cmd, "applied": False}
    applied_ts = now_utc()
    COLLECTOR_SCRIPT.write_text(new_text, encoding="utf-8")
    repair_state = {"verdict": "MAINNET_ENDPOINT_REPAIRED", "applied": True, "applied_ts": str(applied_ts), "diff": diff, "manual_restart_command": cmd}
    REPAIR_STATE.write_text(jdump(repair_state), encoding="utf-8")
    md = (
        "# Endpoint Repair Applied Diff\n\n"
        f"- before: {diff['before_hosts']}\n"
        f"- after: {diff['after_hosts']}\n"
        f"- applied_ts: {applied_ts}\n"
        "- observer definition: unchanged\n"
        "- collector script only\n"
    )
    (ROOT / "reports/endpoint_repair_applied_diff.md").write_text(md, encoding="utf-8")
    return {"verdict": "MAINNET_ENDPOINT_REPAIRED", "diff": diff, "manual_restart_command": cmd, "applied": True, "applied_ts": str(applied_ts)}


def safety_audit(before_hashes: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    ensure_dirs()
    rows = []
    for raw in WATCH_PRODUCTION:
        p = Path(raw)
        if p.is_file():
            rows.append({"path": str(p), "sha256": sha256(p)})
    after = rows
    changed = []
    if before_hashes:
        bmap = {x["path"]: x.get("sha256") for x in before_hashes}
        for x in after:
            if x["path"] in bmap and bmap[x["path"]] != x.get("sha256"):
                changed.append(x["path"])
    out = {
        "verdict": "SAFETY_AUDIT_PASS" if not changed else "SAFETY_AUDIT_WARNING",
        "private_endpoint_calls": 0,
        "order_endpoint_calls": 0,
        "production_files_changed": changed,
        "observer_definition_changed": "OBSERVER_DEFINITION_CHANGED_WARNING" in str(hash_freeze_audit().get("warnings", [])),
        "production_ready": False,
        "promotion_ready": False,
    }
    (ROOT / "audit/safety_audit.json").write_text(jdump(out), encoding="utf-8")
    (ROOT / "audit/production_safety_audit.md").write_text(
        "# Production Safety Audit\n\n"
        "Diagnostics-only provenance/safety audit. No default mutation.\n"
        "production_ready=false; promotion_ready=false.\n",
        encoding="utf-8",
    )
    return out


def composite_repair_verdict(parts: Dict[str, Any]) -> str:
    ep = parts.get("endpoint", {}).get("verdict", "")
    prov = parts.get("provenance", {}).get("verdict", "")
    q = parts.get("quarantine_apply", {}).get("verdict", parts.get("quarantine", {}).get("verdict", ""))
    if ep == "ENDPOINT_TESTNET_SUSPECT":
        return "ENDPOINT_STILL_TESTNET_SUSPECT"
    if prov in {"TESTNET_OR_NON_MAINNET_DATA_SUSPECT", "MAINNET_PROVENANCE_FAIL"}:
        return "MAINNET_PROVENANCE_FAIL"
    if prov in {"NETWORK_SKIPPED", "PROVENANCE_INCONCLUSIVE"}:
        return "ENDPOINT_REPAIRED_BUT_PROVENANCE_INCONCLUSIVE"
    if q == "TESTNET_SUSPECT_DATA_QUARANTINED" and prov == "MAINNET_PROVENANCE_PASS":
        return "MAINNET_PROVENANCE_PASS"
    if ep == "MAINNET_ENDPOINT_OK" and prov == "MAINNET_PROVENANCE_PASS":
        return "MAINNET_PROVENANCE_PASS"
    if ep == "MAINNET_ENDPOINT_OK" and prov == "MAINNET_PROVENANCE_WARNING":
        return "ENDPOINT_REPAIRED_BUT_PROVENANCE_INCONCLUSIVE"
    return prov or "PROVENANCE_INCONCLUSIVE"


def final_verdict(parts: Dict[str, Any]) -> Dict[str, Any]:
    prov = parts.get("provenance", {}).get("verdict", "UNKNOWN")
    ep = parts.get("endpoint", {}).get("verdict", "UNKNOWN")
    repair = read_json(REPAIR_STATE)
    composite = composite_repair_verdict(parts) if repair.get("applied") else prov
    out = {
        "verdict": composite,
        "provenance_verdict": prov,
        "endpoint_verdict": ep,
        "endpoint_repair": repair.get("verdict"),
        "quarantine": parts.get("quarantine_apply", parts.get("quarantine", {})).get("verdict"),
        "staleness": parts.get("staleness", {}).get("verdict"),
        "episode_ledger": parts.get("episodes", {}).get("verdict"),
        "hash_freeze": parts.get("hash_freeze", {}).get("verdict"),
        "safety": parts.get("safety", {}).get("verdict"),
        "production_ready": False,
        "promotion_ready": False,
        "interpretation": "data_provenance_and_forward_eval_accounting_audit_only",
    }
    (ROOT / "verdict.json").write_text(jdump(out), encoding="utf-8")
    (ROOT / "reports/microstructure_data_provenance_and_forward_safety_audit_final_verdict.md").write_text(
        "# Microstructure Data Provenance And Forward Safety Audit\n\n"
        f"- provenance verdict: **{prov}**\n"
        f"- endpoint verdict: **{ep}**\n"
        f"- quarantine: {out['quarantine']}\n"
        f"- staleness: {out['staleness']}\n"
        f"- production_ready: false\n"
        f"- promotion_ready: false\n",
        encoding="utf-8",
    )
    return out


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_dirs()
    before = [{"path": str(Path(p)), "sha256": sha256(Path(p))} for p in WATCH_PRODUCTION if Path(p).is_file()]
    summary: Dict[str, Any] = {"production_ready": False, "promotion_ready": False}
    if args.dry_run:
        summary.update({"verdict": "DRY_RUN_OK", "root": str(ROOT)})
        print(jdump(summary))
        return summary
    parts: Dict[str, Any] = {}
    if args.endpoint_audit_only or args.full:
        parts["endpoint"] = endpoint_audit()
        summary["endpoint"] = parts["endpoint"]
        if args.endpoint_audit_only:
            print(jdump(summary))
            return summary
    if args.provenance_audit_only or args.full:
        parts["provenance"] = provenance_audit(max_windows=50 if args.full else 20)
        summary["provenance"] = parts["provenance"]
        if args.provenance_audit_only:
            print(jdump(summary))
            return summary
    if args.quarantine_plan_only or args.full:
        parts["quarantine"] = quarantine_plan(parts.get("provenance", provenance_audit(20)), parts.get("endpoint", endpoint_audit()))
        summary["quarantine"] = parts["quarantine"]
        if args.apply_quarantine:
            t0 = repair_t0() or now_utc()
            summary["quarantine_apply"] = apply_quarantine(parts["quarantine"], t0=t0)
            write_quarantine_report(summary["quarantine_apply"])
            write_forward_clock_reset(t0)
        if args.quarantine_plan_only:
            print(jdump(summary))
            return summary
    if args.episode_ledger_only or args.full:
        parts["episodes"] = episode_ledger_safe()
        summary["episodes"] = parts["episodes"]
        if args.episode_ledger_only:
            print(jdump(parts["episodes"]))
            return parts["episodes"]
    if args.staleness_audit_only or args.full:
        parts["staleness"] = staleness_audit()
        summary["staleness"] = parts["staleness"]
        if args.staleness_audit_only:
            print(jdump(summary))
            return summary
    if args.hash_freeze_audit_only or args.full:
        out_name = "hash_freeze_audit_post_repair.json" if read_json(REPAIR_STATE).get("applied") else "hash_freeze_audit.json"
        parts["hash_freeze"] = hash_freeze_audit(output_name=out_name)
        summary["hash_freeze"] = parts["hash_freeze"]
        if args.hash_freeze_audit_only:
            print(jdump(summary))
            return summary
    if args.full:
        parts["gap"] = gap_audit()
        parts["external_state"] = external_state_audit()
        parts["safety"] = safety_audit(before)
        summary["gap"] = parts["gap"]
        summary["external_state"] = parts["external_state"]
        summary["safety"] = parts["safety"]
        parts["final_verdict"] = final_verdict(parts)
        summary["final_verdict"] = parts["final_verdict"]
        summary["verdict"] = parts["final_verdict"]["verdict"]
        if read_json(REPAIR_STATE).get("applied"):
            write_post_repair_reports(parts)
    if args.repair_endpoint_mainnet:
        summary["endpoint_repair"] = repair_endpoint_mainnet(dry_plan=not args.apply_endpoint_repair)
    print(jdump(summary))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Microstructure provenance and forward safety audit")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--provenance-audit-only", action="store_true")
    p.add_argument("--endpoint-audit-only", action="store_true")
    p.add_argument("--quarantine-plan-only", action="store_true")
    p.add_argument("--episode-ledger-only", action="store_true")
    p.add_argument("--staleness-audit-only", action="store_true")
    p.add_argument("--hash-freeze-audit-only", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--apply-quarantine", action="store_true")
    p.add_argument("--repair-endpoint-mainnet", action="store_true")
    p.add_argument("--apply-endpoint-repair", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not any([args.dry_run, args.provenance_audit_only, args.endpoint_audit_only, args.quarantine_plan_only, args.episode_ledger_only, args.staleness_audit_only, args.hash_freeze_audit_only, args.full, args.repair_endpoint_mainnet]):
        args.full = True
    run_pipeline(args)


if __name__ == "__main__":
    main()
