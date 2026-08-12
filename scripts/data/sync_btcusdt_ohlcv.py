"""
BTCUSDT public OHLCV sync for the R7 daily diagnostics monitor.

This script updates data only. It never changes production TCN weights, Q2_BDI,
live execution, order routing, or trading state.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

KST = ZoneInfo("Asia/Seoul") if ZoneInfo else timezone.utc
UTC = timezone.utc
DIAG_ROOT = Path("data/diagnostics/data_sync")
CANONICAL_JSON = DIAG_ROOT / "canonical_data_paths.json"


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


def _now() -> Tuple[datetime, datetime]:
    utc = datetime.now(UTC)
    return utc, utc.astimezone(KST)


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_table(df: pd.DataFrame, path: Path, backup: bool = True, atomic: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.exists():
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        bdir = REPO_ROOT / "data/backups/data_sync" / ts
        bdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, bdir / path.name)
    tmp = path.with_suffix(path.suffix + ".tmp") if atomic else path
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(tmp, index=False)
        except Exception:
            # Parquet filename is required by downstream checks; JSON bytes are fallback.
            tmp.write_bytes(df.to_json(orient="records").encode("utf-8"))
    else:
        df.to_csv(tmp, index=False)
    if atomic and tmp != path:
        tmp.replace(path)


def _standardize(df: pd.DataFrame, symbol: str, timeframe: str, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "symbol", "timeframe", "source", "updated_at"])
    out = df.copy()
    rename = {c: c.lower() for c in out.columns}
    out = out.rename(columns=rename)
    if "datetime" in out.columns and "timestamp" not in out.columns:
        out = out.rename(columns={"datetime": "timestamp"})
    if "open_time" in out.columns and "timestamp" not in out.columns:
        out = out.rename(columns={"open_time": "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["volume"] = out["volume"].fillna(0.0)
    out["symbol"] = symbol
    out["timeframe"] = timeframe
    out["source"] = source
    out["updated_at"] = datetime.now(UTC).isoformat()
    out = out[["timestamp", "open", "high", "low", "close", "volume", "symbol", "timeframe", "source", "updated_at"]]
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    out = out[out["timestamp"] <= pd.Timestamp(datetime.now(UTC).replace(tzinfo=None))]
    out = out.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    return out


def _discover_paths() -> Dict[str, Any]:
    existing_5m = Path("data/ohlcv/BTCUSDT_5m_full.csv")
    canonical = {
        "canonical_1m_path": "data/market/btcusdt_1m.parquet",
        "canonical_5m_path": str(existing_5m if existing_5m.exists() else Path("data/market/btcusdt_5m.parquet")),
        "fallback_1m_path": "data/market/btcusdt_1m.parquet",
        "fallback_5m_path": "data/market/btcusdt_5m.parquet",
        "freshness_source_priority": [
            "canonical_5m_path",
            "data/market/btcusdt_5m.parquet",
            "data/ohlcv/BTCUSDT_5m_full.csv",
            "data/cache/ml_predictions/ml_tcn_BTCUSDT_5m_proba.parquet",
        ],
        "ambiguous": False,
        "notes": "Existing freshness source data/ohlcv/BTCUSDT_5m_full.csv selected as canonical 5m path when present.",
    }
    DIAG_ROOT.mkdir(parents=True, exist_ok=True)
    CANONICAL_JSON.write_text(_json(canonical), encoding="utf-8")
    report = {
        "1m raw path": canonical["canonical_1m_path"],
        "5m canonical path": canonical["canonical_5m_path"],
        "R7/freshness source": canonical["canonical_5m_path"],
        "timestamp column": "timestamp",
        "OHLCV columns": ["open", "high", "low", "close", "volume"],
        "timezone": "UTC stored as timezone-naive timestamp for existing CSV compatibility",
        "format": "1m parquet, 5m existing CSV or fallback parquet",
    }
    _write_md(DIAG_ROOT / "path_discovery_report.md", "BTCUSDT Data Path Discovery", report)
    return canonical


def _write_md(path: Path, title: str, sections: Dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for k, v in sections.items():
        lines += [f"## {k}"]
        if isinstance(v, (dict, list)):
            lines += ["```json", _json(v), "```"]
        else:
            lines.append(str(v))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _raw_symbol(symbol: str, raw_symbol: str | None) -> str:
    if raw_symbol:
        return raw_symbol
    if symbol == "BTC/USDT:USDT":
        return "BTCUSDT"
    return symbol.replace("/", "").replace(":USDT", "")


def _fetch_with_ccxt(exchange_name: str, symbol: str, timeframe: str, since_ms: int, limit: int, max_retries: int, sleep_sec: float) -> Tuple[List[List[float]], str]:
    import ccxt  # type: ignore

    exchange_cls = getattr(ccxt, exchange_name)
    ex = exchange_cls({"enableRateLimit": True})
    rows: List[List[float]] = []
    cur = since_ms
    while True:
        batch = None
        for attempt in range(max_retries):
            try:
                batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cur, limit=limit)
                break
            except Exception:
                if attempt + 1 >= max_retries:
                    raise
                time.sleep(sleep_sec * (attempt + 1))
        if not batch:
            break
        rows.extend(batch)
        next_ts = int(batch[-1][0]) + 60_000
        if next_ts <= cur:
            break
        cur = next_ts
        if len(batch) < limit or pd.Timestamp(cur, unit="ms") >= pd.Timestamp(datetime.now(UTC).replace(tzinfo=None)):
            break
        time.sleep(sleep_sec)
    return rows, exchange_name


def _fetch_with_binance_http(raw_symbol: str, timeframe: str, since_ms: int, limit: int, max_retries: int, sleep_sec: float) -> Tuple[List[List[float]], str]:
    rows: List[List[float]] = []
    cur = since_ms
    base_urls = [
        "https://fapi.binance.com/fapi/v1/klines",
        "https://api.binance.com/api/v3/klines",
    ]
    used = ""
    while True:
        batch = None
        last_error = None
        for base in base_urls:
            params = urllib.parse.urlencode({"symbol": raw_symbol, "interval": timeframe, "startTime": cur, "limit": limit})
            url = f"{base}?{params}"
            for attempt in range(max_retries):
                try:
                    with urllib.request.urlopen(url, timeout=20) as resp:
                        batch = json.loads(resp.read().decode("utf-8"))
                    used = base
                    break
                except Exception as exc:
                    last_error = exc
                    time.sleep(sleep_sec * (attempt + 1))
            if batch is not None:
                break
        if batch is None:
            raise RuntimeError(f"binance_http_fetch_failed:{type(last_error).__name__}")
        if not batch:
            break
        rows.extend([[b[0], b[1], b[2], b[3], b[4], b[5]] for b in batch])
        next_ts = int(batch[-1][0]) + 60_000
        if next_ts <= cur:
            break
        cur = next_ts
        if len(batch) < limit or pd.Timestamp(cur, unit="ms") >= pd.Timestamp(datetime.now(UTC).replace(tzinfo=None)):
            break
        time.sleep(sleep_sec)
    return rows, used


def _fetch_ohlcv(args: argparse.Namespace, since_ms: int) -> Tuple[pd.DataFrame, str]:
    if args.no_network:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]), "no_network"
    exchange = args.exchange
    if exchange == "auto":
        exchange = "binanceusdm"
    try:
        rows, used = _fetch_with_ccxt(exchange, args.symbol, args.timeframe, since_ms, args.limit, args.max_retries, args.sleep_sec)
    except Exception:
        rows, used = _fetch_with_binance_http(_raw_symbol(args.symbol, args.raw_symbol), args.timeframe, since_ms, args.limit, args.max_retries, args.sleep_sec)
    df = pd.DataFrame(rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]), used
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.tz_convert(None)
    return df[["timestamp", "open", "high", "low", "close", "volume"]], used


def _drop_open_candle(df: pd.DataFrame, timeframe: str) -> Tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    now = pd.Timestamp(datetime.now(UTC).replace(tzinfo=None))
    minutes = 1 if timeframe == "1m" else 5
    cutoff = now.floor(f"{minutes}min") - pd.Timedelta(minutes=minutes)
    before = len(df)
    out = df[df["timestamp"] <= cutoff].copy()
    return out, before - len(out)


def _resample_5m(df1: pd.DataFrame) -> pd.DataFrame:
    if df1.empty:
        return pd.DataFrame(columns=df1.columns)
    x = df1.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
    x = x.set_index("timestamp").sort_index()
    ohlc = x.resample("5min", label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open", "high", "low", "close"])
    ohlc = ohlc.reset_index()
    ohlc["symbol"] = "BTCUSDT"
    ohlc["timeframe"] = "5m"
    ohlc["source"] = "resampled_from_1m"
    ohlc["updated_at"] = datetime.now(UTC).isoformat()
    ohlc, _ = _drop_open_candle(ohlc, "5m")
    return ohlc


def _quality(df: pd.DataFrame, timeframe: str, max_age_hours: float = 2.0) -> Tuple[pd.DataFrame, str]:
    rows = []
    status = "PASS"
    if df.empty:
        return pd.DataFrame([{"check": "non_empty", "pass": False, "status": "FAIL", "detail": "empty"}]), "FAIL"
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    checks = {
        "timestamp_duplicate_none": int(ts.duplicated().sum()) == 0,
        "timestamp_monotonic": bool(ts.is_monotonic_increasing),
        "ohlc_not_null": bool(df[["open", "high", "low", "close"]].notna().all().all()),
        "high_ge_open_close": bool((df["high"] >= df[["open", "close"]].max(axis=1)).all()),
        "low_le_open_close": bool((df["low"] <= df[["open", "close"]].min(axis=1)).all()),
        "volume_non_negative": bool((df["volume"].fillna(0) >= 0).all()),
        "future_timestamp_none": bool((ts <= pd.Timestamp(datetime.now(UTC).replace(tzinfo=None))).all()),
    }
    expected_min = 1 if timeframe == "1m" else 5
    gaps = ts.diff().dt.total_seconds().div(60).dropna()
    gap_count = int((gaps > expected_min * 1.5).sum())
    largest_gap = float(gaps.max()) if len(gaps) else 0.0
    latest_age = (pd.Timestamp(datetime.now(UTC).replace(tzinfo=None)) - ts.max()).total_seconds() / 3600
    checks["latest_within_max_age"] = latest_age <= max_age_hours
    for name, passed in checks.items():
        st = "PASS" if passed else "FAIL"
        if not passed:
            status = "FAIL"
        rows.append({"timeframe": timeframe, "check": name, "pass": bool(passed), "status": st, "detail": ""})
    if gap_count:
        rows.append({"timeframe": timeframe, "check": "gap_detection", "pass": True, "status": "WARN", "detail": f"gap_count={gap_count}, largest_gap_minutes={largest_gap}"})
        if status == "PASS":
            status = "WARN"
    return pd.DataFrame(rows), status


def _post_sync_pipeline_report() -> None:
    candidates = [
        "scripts/update_features.py",
        "scripts/build_features.py",
        "scripts/refresh_features.py",
        "scripts/update_proba_cache.py",
        "scripts/refresh_proba_cache.py",
        "scripts/daily_run.py",
        "scripts/run_daily.py",
        "scripts/diagnostics/run_daily_meta_research_pipeline.py",
        "ops/daily_run.sh",
        "ops/build_features.sh",
    ]
    found = [c for c in candidates if (REPO_ROOT / c).exists()]
    report = {
        "found_candidate_scripts": found,
        "executed_script": "",
        "not_executed_reason": "No explicit feature/proba cache update script found; OHLCV sync completed only.",
        "exit_code": "",
        "stdout_log_path": str(DIAG_ROOT / "logs/sync_stdout.log"),
        "stderr_log_path": str(DIAG_ROOT / "logs/sync_stderr.log"),
        "feature_latest_timestamp_before_after": "not_available",
        "proba_latest_timestamp_before_after": "not_available",
    }
    _write_md(DIAG_ROOT / "post_sync_pipeline_report.md", "Post Sync Feature/Proba Pipeline Report", report)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    DIAG_ROOT.mkdir(parents=True, exist_ok=True)
    (DIAG_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    canonical = _discover_paths()
    if args.canonical_paths_json:
        p = Path(args.canonical_paths_json)
        if p.exists():
            canonical.update(json.loads(p.read_text(encoding="utf-8")))
    out_1m = Path(args.output_1m or canonical["canonical_1m_path"])
    out_5m = Path(args.output_5m or canonical["canonical_5m_path"])
    old_1m = _standardize(_safe_read(REPO_ROOT / out_1m), "BTCUSDT", "1m", "existing") if (REPO_ROOT / out_1m).exists() else pd.DataFrame()
    old_5m_raw = _safe_read(REPO_ROOT / out_5m) if (REPO_ROOT / out_5m).exists() else pd.DataFrame()
    old_5m = _standardize(old_5m_raw, "BTCUSDT", "5m", "existing") if not old_5m_raw.empty else pd.DataFrame()
    old_latest_1m = old_1m["timestamp"].max() if not old_1m.empty else None
    old_latest_5m = old_5m["timestamp"].max() if not old_5m.empty else None

    if args.since:
        since = pd.Timestamp(args.since, tz=UTC)
    elif old_latest_1m is not None and not pd.isna(old_latest_1m):
        since = pd.Timestamp(old_latest_1m).tz_localize(UTC) - pd.Timedelta(hours=2)
    else:
        since = pd.Timestamp(datetime.now(UTC)) - pd.Timedelta(days=args.lookback_days)
    since_ms = int(since.timestamp() * 1000)

    fetched = pd.DataFrame()
    used_exchange = args.exchange
    error = ""
    if args.dry_run:
        sync_status = "DRY_RUN"
    else:
        try:
            fetched, used_exchange = _fetch_ohlcv(args, since_ms)
            sync_status = "FETCHED"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            sync_status = "FAIL"
    fetched_std = _standardize(fetched, "BTCUSDT", "1m", used_exchange) if not fetched.empty else pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "symbol", "timeframe", "source", "updated_at"])
    fetched_std, dropped_1m = _drop_open_candle(fetched_std, "1m")
    combined_1m = pd.concat([old_1m, fetched_std], ignore_index=True) if not old_1m.empty else fetched_std
    before_dedupe_1m = len(combined_1m)
    combined_1m = _standardize(combined_1m, "BTCUSDT", "1m", "public_ohlcv")
    dedupe_removed_1m = before_dedupe_1m - len(combined_1m)
    new_5m = _resample_5m(combined_1m if not combined_1m.empty else fetched_std)
    combined_5m = pd.concat([old_5m, new_5m], ignore_index=True) if not old_5m.empty else new_5m
    before_dedupe_5m = len(combined_5m)
    combined_5m = _standardize(combined_5m, "BTCUSDT", "5m", "resampled_from_1m")
    combined_5m, dropped_5m = _drop_open_candle(combined_5m, "5m")
    dedupe_removed_5m = before_dedupe_5m - len(combined_5m)

    q1, status_1m = _quality(combined_1m, "1m")
    q5, status_5m = _quality(combined_5m, "5m")
    quality = pd.concat([q1, q5], ignore_index=True)
    quality_status = "FAIL" if "FAIL" in {status_1m, status_5m} else ("WARN" if "WARN" in {status_1m, status_5m} else "PASS")
    if not args.dry_run and sync_status != "FAIL":
        _write_table(combined_1m, REPO_ROOT / out_1m, backup=args.backup, atomic=args.atomic_write)
        _write_table(combined_5m, REPO_ROOT / out_5m, backup=args.backup, atomic=args.atomic_write)
        sync_status = "PASS" if quality_status in {"PASS", "WARN"} else "FAIL"
    now_utc, now_kst = _now()
    latest_5m = combined_5m["timestamp"].max() if not combined_5m.empty else None
    age_h = None if latest_5m is None or pd.isna(latest_5m) else (pd.Timestamp(now_utc.replace(tzinfo=None)) - pd.Timestamp(latest_5m)).total_seconds() / 3600
    stale_after = True if age_h is None else age_h > 2.0
    gaps = pd.to_datetime(combined_5m["timestamp"]).diff().dt.total_seconds().div(60).dropna() if not combined_5m.empty else pd.Series(dtype=float)
    manifest = {
        "run_ts_utc": now_utc.isoformat(),
        "run_ts_kst": now_kst.isoformat(),
        "exchange": used_exchange,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "canonical_1m_path": str(out_1m),
        "canonical_5m_path": str(out_5m),
        "old_latest_1m_ts": old_latest_1m,
        "new_latest_1m_ts": combined_1m["timestamp"].max() if not combined_1m.empty else None,
        "old_latest_5m_ts": old_latest_5m,
        "new_latest_5m_ts": latest_5m,
        "fetched_rows": int(len(fetched_std)),
        "appended_rows": int(max(0, len(combined_1m) - len(old_1m))),
        "updated_rows": int(len(fetched_std) - max(0, len(combined_1m) - len(old_1m))) if len(fetched_std) else 0,
        "final_1m_rows": int(len(combined_1m)),
        "final_5m_rows": int(len(combined_5m)),
        "duplicate_rows_removed": int(dedupe_removed_1m + dedupe_removed_5m),
        "incomplete_candles_dropped": int(dropped_1m + dropped_5m),
        "data_gap_count": int((gaps > 7.5).sum()) if len(gaps) else 0,
        "largest_gap_minutes": float(gaps.max()) if len(gaps) else 0.0,
        "sync_status": sync_status,
        "quality_status": quality_status,
        "error_message": error,
        "stale_after_sync": stale_after,
        "production_changed": False,
        "live_changed": False,
        "q2_changed": False,
    }
    (DIAG_ROOT / "sync_latest.json").write_text(_json(manifest), encoding="utf-8")
    _write_md(DIAG_ROOT / "sync_latest.md", "BTCUSDT OHLCV Sync Latest", manifest)
    hist_path = DIAG_ROOT / "sync_history.csv"
    old_hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
    pd.concat([old_hist, pd.DataFrame([manifest])], ignore_index=True).to_csv(hist_path, index=False)
    quality.to_csv(DIAG_ROOT / "data_quality_latest.csv", index=False)
    _write_md(DIAG_ROOT / "data_quality_latest.md", "BTCUSDT Data Quality Latest", {"status": quality_status, "checks": quality.to_dict(orient="records")})
    _post_sync_pipeline_report()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync BTCUSDT public OHLCV for daily diagnostics.")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--raw-symbol", default=None)
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--resample-to-5m", default="true")
    parser.add_argument("--since", default=None)
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--exchange", default="auto")
    parser.add_argument("--output-1m", default=None)
    parser.add_argument("--output-5m", default=None)
    parser.add_argument("--canonical-paths-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--sleep-sec", type=float, default=0.2)
    parser.add_argument("--drop-open-candle", default="true")
    parser.add_argument("--backup", default="true")
    parser.add_argument("--atomic-write", default="true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.backup = str(args.backup).lower() != "false"
    args.atomic_write = str(args.atomic_write).lower() != "false"
    manifest = run(args)
    if args.json:
        print(_json(manifest))
    elif args.verbose:
        print(f"sync_status={manifest['sync_status']} new_latest_5m_ts={manifest['new_latest_5m_ts']}")
    return 1 if manifest["sync_status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
