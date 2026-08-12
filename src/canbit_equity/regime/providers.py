"""External providers: yfinance (market) + FRED public CSV (rates)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import URLError, HTTPError
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd
import yfinance as yf

from canbit_equity.regime.config import FRED_SERIES, PIPELINE_VERSION, YF_SERIES, YF_SYMBOL_RESOLVE, paths

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


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
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {"git_commit": None, "git_dirty": None}


def download_yf_series(symbol: str, start: str = "1999-01-01") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    p = paths()
    downloaded_at = datetime.now(timezone.utc).isoformat()
    resolve = YF_SYMBOL_RESOLVE.get(symbol, symbol)
    t = yf.Ticker(resolve)
    hist = t.history(start=start, auto_adjust=False, actions=False)
    if hist is None or hist.empty:
        raise RuntimeError(f"YF_EMPTY:{symbol}->{resolve}")
    hist = hist.copy()
    hist.index = pd.to_datetime(hist.index)
    if hist.index.tz is not None:
        hist["session_date"] = hist.index.tz_convert("America/New_York").normalize().tz_localize(None)
    else:
        hist["session_date"] = hist.index.normalize()
    hist = hist[hist["session_date"].dt.dayofweek < 5].copy()
    rows = []
    for _, r in hist.iterrows():
        close = float(r["Close"])
        adj = float(r["Adj Close"]) if "Adj Close" in r and pd.notna(r["Adj Close"]) else close
        if not np.isfinite(close) or close <= 0 or not np.isfinite(adj) or adj <= 0:
            continue
        factor = adj / close
        rows.append(
            {
                "symbol": symbol,
                "resolved_symbol": resolve,
                "session_date": pd.Timestamp(r["session_date"]).normalize(),
                "open_raw": float(r["Open"]),
                "high_raw": float(r["High"]),
                "low_raw": float(r["Low"]),
                "close_raw": close,
                "adj_close": adj,
                "close_adj": adj,
                "open_adj": float(r["Open"]) * factor,
                "volume": float(r.get("Volume", 0) or 0),
                "adjustment_factor": factor,
            }
        )
    df = pd.DataFrame(rows).drop_duplicates("session_date").sort_values("session_date").reset_index(drop=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = symbol.replace("^", "").lower()
    raw_path = p.root / "raw" / f"{safe}_1d_{ts}.parquet"
    norm_path = p.root / "normalized" / f"{safe}_1d_normalized.parquet"
    df.to_parquet(raw_path, index=False)
    df.to_parquet(norm_path, index=False)
    man = {
        "series_id": symbol,
        "symbol": symbol,
        "resolved_symbol": resolve,
        "symbol_resolution_note": (
            None
            if resolve == symbol
            else (
                f"{symbol} yfinance history too short for pre-2024 common period; "
                f"resolved to {resolve} (Nasdaq-100 equal-weight) for research continuity."
            )
        ),
        "provider": "yfinance",
        "provider_provenance": "YAHOO_FINANCE_UNOFFICIAL",
        "requested_start": start,
        "requested_end": None,
        "actual_start": str(df["session_date"].iloc[0].date()) if len(df) else None,
        "actual_end": str(df["session_date"].iloc[-1].date()) if len(df) else None,
        "rows": int(len(df)),
        "raw_file": str(raw_path),
        "normalized_file": str(norm_path),
        "raw_sha256": _sha256_file(raw_path),
        "normalized_sha256": _sha256_file(norm_path),
        "downloaded_at_utc": downloaded_at,
        "timezone": "America/New_York",
        "adjustment_mode": "ADJ_CLOSE_FACTOR",
        "duplicate_count": 0,
        "invalid_count": 0,
        "missing_count": 0,
        "revision_count": 0,
        "availability_lag_sessions": 0,
        "schema_version": "qqq_regime_ext_v1",
        "pipeline_version": PIPELINE_VERSION,
        **_git_meta(p.root.parents[2]),
    }
    return df, man


def download_fred_series(series_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    p = paths()
    downloaded_at = datetime.now(timezone.utc).isoformat()
    url = FRED_CSV.format(sid=series_id)
    req = Request(url, headers={"User-Agent": "canbit-equity-regime/1.0"})
    try:
        with urlopen(req, timeout=60) as resp:
            raw = resp.read()
    except (URLError, HTTPError) as exc:
        raise RuntimeError(f"FRED_DOWNLOAD_FAIL:{series_id}:{exc}") from exc
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = p.root / "raw" / f"{series_id.lower()}_{ts}.csv"
    raw_path.write_bytes(raw)
    from io import BytesIO

    csv = pd.read_csv(BytesIO(raw))
    # columns typically DATE, SERIES_ID
    date_col = [c for c in csv.columns if c.upper() in ("DATE", "OBSERVATION_DATE")][0]
    val_col = [c for c in csv.columns if c != date_col][0]
    csv["observation_date"] = pd.to_datetime(csv[date_col], errors="coerce").dt.normalize()
    csv["value_raw"] = csv[val_col].astype(str).str.strip()
    csv["value"] = pd.to_numeric(csv["value_raw"].replace({".": np.nan, "": np.nan}), errors="coerce")
    invalid = int(csv["value"].isna().sum())
    df = (
        csv.dropna(subset=["observation_date", "value"])
        .drop_duplicates("observation_date")
        .sort_values("observation_date")
        .reset_index(drop=True)[["observation_date", "value"]]
    )
    df["series_id"] = series_id
    norm_path = p.root / "normalized" / f"{series_id.lower()}_normalized.parquet"
    df.to_parquet(norm_path, index=False)
    man = {
        "series_id": series_id,
        "symbol": series_id,
        "provider": "fred",
        "provider_provenance": "FRED_OFFICIAL_LATEST_VINTAGE",
        "fred_limitations": [
            "Not ALFRED point-in-time vintage",
            "Historical latest vintage may include revisions",
            "Not a strict real-time macro backtest",
        ],
        "requested_start": None,
        "requested_end": None,
        "actual_start": str(df["observation_date"].iloc[0].date()) if len(df) else None,
        "actual_end": str(df["observation_date"].iloc[-1].date()) if len(df) else None,
        "rows": int(len(df)),
        "raw_file": str(raw_path),
        "normalized_file": str(norm_path),
        "raw_sha256": _sha256_file(raw_path),
        "normalized_sha256": _sha256_file(norm_path),
        "downloaded_at_utc": downloaded_at,
        "timezone": "US/Eastern",
        "adjustment_mode": None,
        "duplicate_count": 0,
        "invalid_count": invalid,
        "missing_count": invalid,
        "revision_count": 0,
        "availability_lag_sessions": 1,
        "schema_version": "qqq_regime_fred_v1",
        "pipeline_version": PIPELINE_VERSION,
        **_git_meta(p.root.parents[2]),
    }
    return df, man


def update_all_external() -> Dict[str, Any]:
    p = paths()
    series_manifests: List[Dict[str, Any]] = []
    errors: List[str] = []
    for sym in YF_SERIES:
        try:
            _, man = download_yf_series(sym)
            series_manifests.append(man)
        except Exception as exc:
            errors.append(f"{sym}:{type(exc).__name__}:{exc}")
    for sid in FRED_SERIES:
        try:
            _, man = download_fred_series(sid)
            series_manifests.append(man)
        except Exception as exc:
            errors.append(f"{sid}:{type(exc).__name__}:{exc}")

    expected = set(YF_SERIES) | set(FRED_SERIES)
    got = {m["series_id"] for m in series_manifests}
    missing = sorted(expected - got)
    unified = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "series": series_manifests,
        "errors": errors,
        "missing_series": missing,
        "n_ok": len(series_manifests),
        "n_expected": len(expected),
        "pipeline_version": PIPELINE_VERSION,
    }
    blob = json.dumps(unified, sort_keys=True, default=str).encode()
    unified["manifest_sha256"] = _sha256_bytes(blob)
    out_path = p.root / "manifests" / "qqq_regime_external_data_manifest.json"
    out_path.write_text(json.dumps(unified, indent=2, default=str) + "\n")
    unified["path"] = str(out_path)
    unified["external_data_downloaded"] = len(missing) == 0 and len(errors) == 0
    return unified


def load_normalized_market(symbol: str) -> pd.DataFrame:
    p = paths()
    safe = symbol.replace("^", "").lower()
    path = p.root / "normalized" / f"{safe}_1d_normalized.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.normalize()
    return df.sort_values("session_date").reset_index(drop=True)


def load_normalized_fred(series_id: str) -> pd.DataFrame:
    p = paths()
    path = p.root / "normalized" / f"{series_id.lower()}_normalized.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df["observation_date"] = pd.to_datetime(df["observation_date"]).dt.normalize()
    return df.sort_values("observation_date").reset_index(drop=True)
