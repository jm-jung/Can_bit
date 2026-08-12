"""Data provider abstraction + Yahoo Finance unofficial adapter."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from .calendar import expected_sessions, get_calendar, latest_completed_session, session_bounds_utc
from .config import QQQConfig, ensure_dirs, paths
from .schemas import NORMALIZED_COLUMNS


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_meta(repo: Path) -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=5).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True, timeout=5).strip() != ""
        return {"git_commit_if_available": commit, "git_dirty": dirty}
    except Exception:
        return {"git_commit_if_available": None, "git_dirty": None}


def download_qqq_daily(cfg: QQQConfig = QQQConfig()) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ensure_dirs(cfg)
    p = paths(cfg)
    cal = get_calendar(cfg.exchange)
    end_session = latest_completed_session(cal)
    end_str = end_session.strftime("%Y-%m-%d")
    start_str = cfg.start_date
    downloaded_at = datetime.now(timezone.utc).isoformat()

    ticker = yf.Ticker(cfg.symbol)
    # auto_adjust=False to keep raw OHLC + Adj Close separately
    hist = ticker.history(start=start_str, end=(pd.Timestamp(end_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), auto_adjust=False, actions=True)
    if hist is None or hist.empty:
        raise RuntimeError("QQQ_DATA_PROVIDER_BLOCKED: empty history from yfinance")

    hist = hist.copy()
    hist.index = pd.to_datetime(hist.index)
    if hist.index.tz is not None:
        hist["session_date"] = hist.index.tz_convert(cfg.exchange_timezone).normalize().tz_localize(None)
    else:
        hist["session_date"] = hist.index.normalize()

    # Keep only completed sessions <= end_session
    hist = hist[hist["session_date"] <= end_session].copy()

    rows = []
    cal_first = pd.Timestamp(cal.first_session).normalize()
    for _, r in hist.iterrows():
        sd = pd.Timestamp(r["session_date"]).normalize()
        # exchange_calendars may start later than QQQ listing; accept pre-calendar weekdays from provider
        if sd >= cal_first:
            if not cal.is_session(sd):
                continue
        else:
            if sd.dayofweek >= 5:
                continue
        open_raw = float(r["Open"])
        high_raw = float(r["High"])
        low_raw = float(r["Low"])
        close_raw = float(r["Close"])
        adj_close = float(r["Adj Close"])
        if close_raw <= 0 or not np.isfinite(close_raw) or not np.isfinite(adj_close):
            continue
        factor = adj_close / close_raw
        if sd >= cal_first:
            open_utc, close_utc = session_bounds_utc(sd, cal)
        else:
            ny = ZoneInfo("America/New_York")
            open_utc = pd.Timestamp(datetime(sd.year, sd.month, sd.day, 9, 30, tzinfo=ny)).tz_convert("UTC")
            close_utc = pd.Timestamp(datetime(sd.year, sd.month, sd.day, 16, 0, tzinfo=ny)).tz_convert("UTC")
        rows.append(
            {
                "symbol": cfg.symbol,
                "session_date": sd,
                "open_raw": open_raw,
                "high_raw": high_raw,
                "low_raw": low_raw,
                "close_raw": close_raw,
                "adj_close": adj_close,
                "adjustment_factor": factor,
                "open_adj": open_raw * factor,
                "high_adj": high_raw * factor,
                "low_adj": low_raw * factor,
                "close_adj": adj_close,
                "volume": float(r.get("Volume") or 0.0),
                "dividends": float(r.get("Dividends") or 0.0),
                "stock_splits": float(r.get("Stock Splits") or 0.0),
                "provider": cfg.provider,
                "provider_provenance": cfg.provider_provenance,
                "session_open_utc": open_utc.isoformat(),
                "session_close_utc": close_utc.isoformat(),
                "source_timestamp": sd.strftime("%Y-%m-%d"),
                "downloaded_at_utc": downloaded_at,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("QQQ_DATA_PROVIDER_BLOCKED: no valid sessions after calendar filter")
    df = df.drop_duplicates(subset=["symbol", "session_date"]).sort_values("session_date").reset_index(drop=True)
    df = df[NORMALIZED_COLUMNS]

    # Save raw snapshot
    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = p["raw"] / f"qqq_1d_{ts_tag}.parquet"
    hist.reset_index().to_parquet(raw_path, index=False)

    norm_path = p["normalized_file"]
    revision_info = {"revised_rows": 0, "revised_dates": []}
    if norm_path.exists():
        old = pd.read_parquet(norm_path)
        old = old.sort_values("session_date")
        merged = pd.concat([old, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["symbol", "session_date"], keep="last").sort_values("session_date").reset_index(drop=True)
        # revision detection on overlapping dates
        common = set(pd.to_datetime(old["session_date"]).dt.normalize()) & set(pd.to_datetime(df["session_date"]).dt.normalize())
        if common:
            o = old.set_index(pd.to_datetime(old["session_date"]).dt.normalize())
            n = df.set_index(pd.to_datetime(df["session_date"]).dt.normalize())
            for d in sorted(common):
                if abs(float(o.loc[d, "close_adj"]) - float(n.loc[d, "close_adj"])) > 1e-8:
                    revision_info["revised_rows"] += 1
                    revision_info["revised_dates"].append(str(pd.Timestamp(d).date()))
        df = merged[NORMALIZED_COLUMNS]

    # Cache-hit if identical hash to previous normalized
    tmp = p["cache"] / "qqq_1d_normalized_tmp.parquet"
    df.to_parquet(tmp, index=False)
    new_hash = _sha256_file(tmp)
    cache_hit = False
    if norm_path.exists() and p["manifest_latest"].exists():
        prev = json.loads(p["manifest_latest"].read_text())
        if prev.get("normalized_sha256") == new_hash:
            cache_hit = True
            tmp.unlink(missing_ok=True)
        else:
            tmp.replace(norm_path)
    else:
        tmp.replace(norm_path)

    raw_sha = _sha256_file(raw_path)
    norm_sha = _sha256_file(norm_path) if norm_path.exists() else new_hash

    audit_start = max(pd.Timestamp(df["session_date"].min()), pd.Timestamp(cal.first_session))
    audit_end = min(pd.Timestamp(df["session_date"].max()), pd.Timestamp(cal.last_session))
    exp = expected_sessions(audit_start, audit_end, cal)
    have = set(pd.to_datetime(df["session_date"]).dt.normalize())
    expected = set(pd.to_datetime(exp).normalize())
    missing = sorted(expected - have)
    in_cal = {x for x in have if pd.Timestamp(cal.first_session) <= x <= pd.Timestamp(cal.last_session)}
    unexpected = sorted(in_cal - expected)

    manifest = {
        "symbol": cfg.symbol,
        "provider": cfg.provider,
        "provider_provenance": cfg.provider_provenance,
        "interval": cfg.interval,
        "requested_start": start_str,
        "requested_end": end_str,
        "actual_start": str(pd.Timestamp(df["session_date"].min()).date()),
        "actual_end": str(pd.Timestamp(df["session_date"].max()).date()),
        "row_count": int(len(df)),
        "downloaded_at_utc": downloaded_at,
        "timezone": cfg.exchange_timezone,
        "regular_session_only": cfg.regular_session_only,
        "price_adjustment_mode": cfg.price_adjustment_mode,
        "raw_file_path": str(raw_path),
        "normalized_file_path": str(norm_path),
        "raw_sha256": raw_sha,
        "normalized_sha256": norm_sha,
        "duplicate_count": 0,
        "missing_expected_sessions": [str(pd.Timestamp(x).date()) for x in missing[:50]],
        "missing_expected_sessions_count": len(missing),
        "unexpected_sessions": [str(pd.Timestamp(x).date()) for x in unexpected[:50]],
        "unexpected_sessions_count": len(unexpected),
        "incomplete_sessions_removed": True,
        "schema_version": cfg.schema_version,
        "pipeline_version": cfg.pipeline_version,
        "cache_hit": cache_hit,
        "provider_revision": revision_info,
        **_git_meta(p["repo"]),
        "note": "Adjusted OHLC derived via adj_close/close_raw factor; dividends not double-counted on top of adj_close.",
    }
    p["manifest_latest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return df, manifest
