"""
Public V4-lite orderflow data fetcher.

Diagnostics-only public endpoints. No API keys, no private endpoints, no order /
account / balance / position endpoints. Writes only under diagnostics cache.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT"]
DEFAULT_FAMILIES = ["funding", "oi", "taker", "mark", "spot", "premium"]
ROOT = Path("data/diagnostics/research_orderflow_data_cache")
REPORT_ROOT = Path("data/diagnostics/public_v4_lite_btc_centric_retry/fetcher")
FAPI = "https://fapi.binance.com"
SAPI = "https://api.binance.com"


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def _parse_symbols(s: str | None) -> List[str]:
    return [x.strip().upper() for x in (s or ",".join(DEFAULT_SYMBOLS)).split(",") if x.strip()]


def _parse_families(s: str | None) -> List[str]:
    aliases = {"open_interest": "oi", "futures_ohlcv": "futures", "spot_ohlcv": "spot"}
    out = []
    for x in (s or ",".join(DEFAULT_FAMILIES)).split(","):
        k = aliases.get(x.strip().lower(), x.strip().lower())
        if k:
            out.append(k)
    return out


def _date_range(args: argparse.Namespace) -> Tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC").floor("5min")
    start = pd.Timestamp(args.start, tz="UTC") if args.start else end - pd.Timedelta(days=int(args.days))
    return start, end


def _get_json(base: str, path: str, params: Dict[str, Any], retries: int = 3) -> Any:
    url = base + path + "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    last = ""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last = str(exc)
            time.sleep(min(2.0 * (attempt + 1), 5.0))
    raise RuntimeError(last)


def _klines_to_df(rows: List[Any], symbol: str, family: str) -> pd.DataFrame:
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trade_count", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols[: len(rows[0])]) if rows else pd.DataFrame(columns=cols)
    if df.empty:
        return df
    df["symbol"] = symbol
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    df["asof_available_ts"] = pd.to_datetime(df["close_time"], unit="ms", utc=True).dt.tz_convert(None) + pd.Timedelta(milliseconds=1)
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["source"] = "binance_public"
    df["data_family"] = family
    return df


def _dict_rows_to_df(rows: List[Dict[str, Any]], symbol: str, family: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "symbol" not in df:
        df["symbol"] = symbol
    ts_col = next((c for c in ["timestamp", "time", "fundingTime", "T"] if c in df.columns), None)
    if ts_col:
        raw = pd.to_numeric(df[ts_col], errors="coerce")
        if raw.dropna().median() > 10_000_000_000:
            df["timestamp"] = pd.to_datetime(raw, unit="ms", utc=True).dt.tz_convert(None)
        else:
            df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True).dt.tz_convert(None)
    if "timestamp" not in df:
        df["timestamp"] = pd.Timestamp.now(tz="UTC").tz_convert(None)
    df["asof_available_ts"] = df.get("asof_available_ts", df["timestamp"] + pd.Timedelta(minutes=5))
    df["source"] = "binance_public"
    df["data_family"] = family
    for c in df.columns:
        if c not in {"symbol", "timestamp", "asof_available_ts", "source", "data_family"}:
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().sum() > 0:
                df[c] = converted
    return df


def _fetch_paged(symbol: str, family: str, start: pd.Timestamp, end: pd.Timestamp, max_pages: int, sample_only: bool) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    logs: List[Dict[str, Any]] = []
    rows: List[Any] = []
    cur = start
    pages = 0
    max_pages = 1 if sample_only else max_pages
    while cur < end and pages < max_pages:
        pages += 1
        try:
            if family == "funding":
                data = _get_json(FAPI, "/fapi/v1/fundingRate", {"symbol": symbol, "startTime": _ms(cur), "endTime": _ms(end), "limit": 1000})
                rows.extend(data)
                if not data:
                    break
                cur = pd.to_datetime(max(int(x["fundingTime"]) for x in data), unit="ms", utc=True) + pd.Timedelta(milliseconds=1)
            elif family == "oi":
                window_end = min(cur + pd.Timedelta(days=29), end)
                data = _get_json(FAPI, "/futures/data/openInterestHist", {"symbol": symbol, "period": "5m", "startTime": _ms(cur), "endTime": _ms(window_end), "limit": 500})
                rows.extend(data)
                if not data:
                    cur = window_end + pd.Timedelta(milliseconds=1)
                    if cur >= end:
                        break
                    continue
                newest = pd.to_datetime(max(int(x["timestamp"]) for x in data), unit="ms", utc=True) + pd.Timedelta(milliseconds=1)
                if newest <= cur:
                    cur = window_end + pd.Timedelta(milliseconds=1)
                else:
                    cur = newest
                if cur > window_end and window_end < end:
                    cur = min(cur, window_end + pd.Timedelta(milliseconds=1))
            elif family == "taker":
                window_end = min(cur + pd.Timedelta(days=29), end)
                data = _get_json(FAPI, "/futures/data/takerlongshortRatio", {"symbol": symbol, "period": "5m", "startTime": _ms(cur), "endTime": _ms(window_end), "limit": 500})
                rows.extend(data)
                if not data:
                    cur = window_end + pd.Timedelta(milliseconds=1)
                    if cur >= end:
                        break
                    continue
                newest = pd.to_datetime(max(int(x["timestamp"]) for x in data), unit="ms", utc=True) + pd.Timedelta(milliseconds=1)
                if newest <= cur:
                    cur = window_end + pd.Timedelta(milliseconds=1)
                else:
                    cur = newest
                if cur > window_end and window_end < end:
                    cur = min(cur, window_end + pd.Timedelta(milliseconds=1))
            elif family in {"futures", "spot", "mark", "premium"}:
                if family == "futures":
                    base, path = FAPI, "/fapi/v1/klines"
                elif family == "spot":
                    base, path = SAPI, "/api/v3/klines"
                elif family == "mark":
                    base, path = FAPI, "/fapi/v1/markPriceKlines"
                else:
                    base, path = FAPI, "/fapi/v1/premiumIndexKlines"
                data = _get_json(base, path, {"symbol": symbol, "interval": "5m", "startTime": _ms(cur), "endTime": _ms(end), "limit": 1000})
                rows.extend(data)
                if not data:
                    break
                cur = pd.to_datetime(max(int(x[0]) for x in data), unit="ms", utc=True) + pd.Timedelta(minutes=5)
            elif family == "aggtrades":
                data = _get_json(FAPI, "/fapi/v1/aggTrades", {"symbol": symbol, "startTime": _ms(cur), "endTime": _ms(min(cur + pd.Timedelta(hours=6), end)), "limit": 1000})
                rows.extend(data)
                if not data:
                    cur += pd.Timedelta(hours=6)
                else:
                    cur = pd.to_datetime(max(int(x["T"]) for x in data), unit="ms", utc=True) + pd.Timedelta(milliseconds=1)
            else:
                break
            logs.append({"symbol": symbol, "family": family, "page": pages, "rows_total": len(rows), "status": "success"})
            time.sleep(0.08)
        except Exception as exc:
            logs.append({"symbol": symbol, "family": family, "page": pages, "rows_total": len(rows), "status": "failed", "error": str(exc)[:300]})
            break
        if sample_only:
            break
    if family in {"futures", "spot", "mark", "premium"}:
        df = _klines_to_df(rows, symbol, family)
    else:
        df = _dict_rows_to_df(rows, symbol, family)
    return df, logs


def _quality(df: pd.DataFrame, symbol: str, family: str) -> Dict[str, Any]:
    if df.empty:
        return {"symbol": symbol, "data_family": family, "rows": 0, "start": "", "end": "", "missing_ratio": 1.0, "gap_ratio": 1.0, "duplicate_ratio": 0.0}
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    gaps = ts.sort_values().diff().dt.total_seconds().fillna(300)
    return {
        "symbol": symbol,
        "data_family": family,
        "rows": len(df),
        "start": ts.min(),
        "end": ts.max(),
        "missing_ratio": float(df.isna().mean().mean()),
        "gap_ratio": float((gaps > 1800).mean()) if family not in {"funding", "aggtrades"} else 0.0,
        "duplicate_ratio": float(df.duplicated(subset=["symbol", "timestamp"]).mean()),
    }


def _write_outputs(registry_rows: List[Dict[str, Any]], quality_rows: List[Dict[str, Any]], logs: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(registry_rows).to_csv(ROOT / "cache_registry.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(ROOT / "data_quality_summary.csv", index=False)
    with (ROOT / "fetch_log.jsonl").open("a", encoding="utf-8") as f:
        for row in logs:
            f.write(json.dumps(row, default=str) + "\n")
    (REPORT_ROOT / "fetcher_run_summary.json").write_text(_json(summary), encoding="utf-8")
    pd.DataFrame(registry_rows).to_csv(REPORT_ROOT / "fetcher_data_family_status.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(REPORT_ROOT / "fetcher_symbol_status.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(REPORT_ROOT / "fetcher_quality_summary.csv", index=False)
    (REPORT_ROOT / "fetcher_private_api_safety_audit.md").write_text("# Fetcher Private API Safety Audit\n\nprivate_api_calls=false\n\norder_account_balance_position_calls=false\n\napi_key_required=false\n", encoding="utf-8")
    (REPORT_ROOT / "fetcher_report.md").write_text("# Public V4-lite Fetcher Report\n\n```json\n" + _json(summary) + "\n```\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    symbols = _parse_symbols(args.symbols)
    families = _parse_families(args.data_family)
    start, end = _date_range(args)
    if args.dry_run:
        return {"dry_run": True, "symbols": symbols, "families": families, "start": start, "end": end, "output_root": str(ROOT), "private_api_calls": False, "order_endpoint_calls": False}
    registry_rows: List[Dict[str, Any]] = []
    quality_rows: List[Dict[str, Any]] = []
    all_logs: List[Dict[str, Any]] = []
    fetch_ts = pd.Timestamp.now(tz="UTC").tz_convert(None)
    for symbol in symbols:
        for family in families:
            df, logs = _fetch_paged(symbol, family, start, end, int(args.max_pages), bool(args.sample_only))
            all_logs.extend(logs)
            if not df.empty:
                df["fetch_ts"] = fetch_ts
                df["publication_delay_seconds"] = (pd.to_datetime(df["asof_available_ts"]) - pd.to_datetime(df["timestamp"])).dt.total_seconds()
                df["quality_missing"] = df.isna().any(axis=1)
                df["quality_duplicate"] = df.duplicated(subset=["symbol", "timestamp"])
                df["quality_gap"] = False
                df["quality_outlier"] = False
                df["is_forward_only"] = family in {"premium"} and args.sample_only
                df["is_proxy"] = family in {"aggtrades"}
                df["notes"] = "proxy source only" if family == "aggtrades" else ""
                out_dir = ROOT / "normalized" / family
                raw_dir = ROOT / "raw" / family / symbol
                feat_dir = ROOT / "features" / family
                out_dir.mkdir(parents=True, exist_ok=True)
                raw_dir.mkdir(parents=True, exist_ok=True)
                feat_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{symbol}.parquet"
                if args.resume and out_path.exists() and not args.force_refresh:
                    old = pd.read_parquet(out_path)
                    df = pd.concat([old, df], ignore_index=True).drop_duplicates(subset=["symbol", "timestamp"], keep="last")
                df.sort_values("timestamp").to_parquet(out_path, index=False)
                df.sort_values("timestamp").to_parquet(feat_dir / f"{symbol}.parquet", index=False)
                (raw_dir / f"{fetch_ts.strftime('%Y%m%d%H%M%S')}.json").write_text(df.head(200).to_json(orient="records", date_format="iso"), encoding="utf-8")
            q = _quality(df, symbol, family)
            quality_rows.append(q)
            registry_rows.append({"symbol": symbol, "data_family": family, "path": str(ROOT / "normalized" / family / f"{symbol}.parquet"), "rows": q["rows"], "start": q["start"], "end": q["end"], "fetch_status": "success" if q["rows"] > 0 else "empty_or_failed", "sample_only": bool(args.sample_only)})
    summary = {"symbols": symbols, "families": families, "start": start, "end": end, "sample_only": bool(args.sample_only), "registry_rows": len(registry_rows), "private_api_calls": False, "order_endpoint_calls": False, "production_ready": False, "promotion_ready": False}
    _write_outputs(registry_rows, quality_rows, all_logs, summary)
    return summary


def main() -> int:
    global ROOT
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--symbols")
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--data-family")
    p.add_argument("--sample-only", action="store_true")
    p.add_argument("--max-pages", type=int, default=10_000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--no-private", action="store_true", default=True)
    p.add_argument("--rate-limit-safe", action="store_true", default=True)
    p.add_argument("--output-root", default=str(ROOT))
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    ROOT = Path(args.output_root)
    result = run(args)
    print(_json(result) if args.json else f"fetch_public_v4_lite rows={result.get('registry_rows', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
