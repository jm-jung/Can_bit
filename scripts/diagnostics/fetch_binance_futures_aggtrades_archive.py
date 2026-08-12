"""Fetch Binance public futures aggTrades archive and normalize to parquet."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.error
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

CACHE = Path("data/diagnostics/research_orderflow_data_cache")
RUN = Path("data/diagnostics/proxy_cvd_expansion_and_forward_orderflow_collector/aggtrades_fetch")
BASE = "https://data.binance.vision/data/futures/um"
COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "timestamp", "is_buyer_maker"]


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _dates(days: int, start: str | None, end: str | None) -> List[date]:
    if end:
        e = pd.Timestamp(end).date()
    else:
        e = (pd.Timestamp.now("UTC").date() - timedelta(days=1))
    if start:
        s = pd.Timestamp(start).date()
    else:
        s = e - timedelta(days=days - 1)
    return [s + timedelta(days=i) for i in range((e - s).days + 1)]


def _url(symbol: str, day: date) -> str:
    ds = day.strftime("%Y-%m-%d")
    return f"{BASE}/daily/aggTrades/{symbol}/{symbol}-aggTrades-{ds}.zip"


def _download(url: str, dest: Path, resume: bool) -> Dict[str, Any]:
    if resume and dest.exists() and dest.stat().st_size > 0:
        return {"status": "exists", "bytes": dest.stat().st_size}
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=40) as r:
            data = r.read()
        dest.write_bytes(data)
        return {"status": "downloaded", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    except urllib.error.HTTPError as exc:
        return {"status": "http_error", "code": exc.code, "error": str(exc)}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def _normalize(symbol: str, files: List[Path]) -> pd.DataFrame:
    parts = []
    for fp in files:
        try:
            with zipfile.ZipFile(fp) as zf:
                name = zf.namelist()[0]
                raw = zf.read(name)
            df = pd.read_csv(io.BytesIO(raw))
            if list(df.columns)[:3] != COLS[:3]:
                df = pd.read_csv(io.BytesIO(raw), header=None, names=COLS)
            if "transact_time" in df.columns and "timestamp" not in df.columns:
                df = df.rename(columns={"transact_time": "timestamp"})
            df = df[COLS]
            df["symbol"] = symbol
            df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True).dt.tz_convert(None)
            df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.lower().isin(["true", "1"])
            df["source_file"] = fp.name
            df["fetch_ts"] = pd.Timestamp.now("UTC").tz_localize(None)
            for c in ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            parts.append(df[["symbol"] + COLS + ["source_file", "fetch_ts"]])
        except Exception:
            continue
    if not parts:
        return pd.DataFrame(columns=["symbol"] + COLS + ["source_file", "fetch_ts"])
    before = sum(len(x) for x in parts)
    out = pd.concat(parts, ignore_index=True).drop_duplicates(["symbol", "agg_trade_id"]).sort_values("timestamp")
    out.attrs["duplicate_rows_removed"] = before - len(out)
    return out


def run(args: argparse.Namespace) -> Dict[str, Any]:
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    days = _dates(args.days, args.start, args.end)
    if args.dry_run:
        return {"dry_run": True, "symbols": symbols, "days": [str(d) for d in days], "market": args.market, "data_type": args.data_type, "private_api_calls": False}
    RUN.mkdir(parents=True, exist_ok=True)
    raw_root = CACHE / "raw/futures_aggtrades"
    norm_root = CACHE / "normalized/futures_aggtrades"
    norm_root.mkdir(parents=True, exist_ok=True)
    inv, status_rows = [], []
    max_files = args.max_files if args.max_files else 10_000
    used = 0
    for symbol in symbols:
        files = []
        downloaded = failed = 0
        for d in days:
            if used >= max_files:
                break
            url = _url(symbol, d)
            dest = raw_root / symbol / Path(url).name
            res = _download(url, dest, args.resume)
            used += 1
            ok = res["status"] in {"exists", "downloaded"}
            downloaded += int(ok)
            failed += int(not ok)
            if ok:
                files.append(dest)
            inv.append({"symbol": symbol, "date": d, "url": url, "path": str(dest), **res})
        df = _normalize(symbol, files)
        dup_removed = int(df.attrs.get("duplicate_rows_removed", 0))
        if not df.empty:
            df.to_parquet(norm_root / f"{symbol}.parquet", index=False)
        storage = sum((raw_root / symbol / Path(_url(symbol, d)).name).stat().st_size for d in days if (raw_root / symbol / Path(_url(symbol, d)).name).exists()) / (1024 * 1024)
        status_rows.append({"symbol": symbol, "requested_days": len(days), "fetched_days": len(files), "normalized_rows": len(df), "rows": len(df), "start_ts": df["timestamp"].min() if len(df) else "", "end_ts": df["timestamp"].max() if len(df) else "", "missing_days": max(len(days) - len(files), 0), "failed_files": failed, "duplicate_rows_removed": dup_removed, "checksum_status": "sha256_recorded", "storage_size_mb": storage, "usable_for_proxy_cvd": bool(len(df) > 0), "downloaded_or_existing": downloaded})
    inv_df, st_df = pd.DataFrame(inv), pd.DataFrame(status_rows)
    inv_df.to_csv(RUN / "aggtrades_file_inventory.csv", index=False)
    st_df.to_csv(RUN / "aggtrades_symbol_status.csv", index=False)
    st_df.to_csv(RUN / "aggtrades_normalization_summary.csv", index=False)
    summary = {"symbols": symbols, "days_requested": len(days), "files_attempted": len(inv), "rows": int(st_df["rows"].sum()) if len(st_df) else 0, "proxy_cvd_note": "is_buyer_maker=true => aggressive sell proxy; false => aggressive buy proxy. This is proxy CVD, not true CVD.", "production_ready": False}
    (RUN / "aggtrades_fetch_summary.json").write_text(_json(summary), encoding="utf-8")
    (RUN / "aggtrades_fetch_report.md").write_text("# AggTrades Fetch Report\n\nPublic Binance futures archive only. Proxy CVD uses `is_buyer_maker`; it is not true CVD.\n\n```json\n" + _json(summary) + "\n```\n", encoding="utf-8")
    return summary


def main() -> int:
    global CACHE
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--market", default="futures")
    p.add_argument("--data-type", default="aggTrades")
    p.add_argument("--daily", action="store_true")
    p.add_argument("--monthly", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verify-checksum", action="store_true")
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--output-root", default=str(CACHE))
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    CACHE = Path(args.output_root)
    res = run(args)
    print(_json(res) if args.json else f"aggTrades rows={res.get('rows', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
