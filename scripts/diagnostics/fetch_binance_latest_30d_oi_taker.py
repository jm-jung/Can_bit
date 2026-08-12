"""Fetch latest Binance public OI/taker data for V4-lite recent30 validation."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT = Path("data/diagnostics/research_orderflow_data_cache")
OUT = Path("data/diagnostics/btc_centric_v4_lite_30d_oi_taker_proxy_cvd/latest30_fetch")
FAPI = "https://fapi.binance.com"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT"]


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def _get(path: str, params: Dict[str, Any]) -> Any:
    url = FAPI + path + "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    with urllib.request.urlopen(url, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_one(symbol: str, family: str, period: str, days: int, limit: int, chunk_hours: int) -> Tuple[pd.DataFrame, List[Dict[str, Any]], str]:
    end = pd.Timestamp.now(tz="UTC").floor("5min")
    floor = end - pd.Timedelta(days=days)
    rows: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []
    fallback_used = "backward_endTime"
    page = 0
    path = "/futures/data/openInterestHist" if family == "oi" else "/futures/data/takerlongshortRatio"
    while end > floor and page < 600:
        page += 1
        params = {"symbol": symbol, "period": period, "limit": limit, "endTime": _ms(end)}
        try:
            data = _get(path, params)
        except Exception as exc:
            logs.append({"symbol": symbol, "family": family, "page": page, "status": "failed_endTime", "error": str(exc)[:240]})
            try:
                data = _get(path, {"symbol": symbol, "period": period, "limit": limit})
                fallback_used = "latest_no_time"
            except Exception as exc2:
                logs.append({"symbol": symbol, "family": family, "page": page, "status": "failed_latest", "error": str(exc2)[:240]})
                break
        if not data:
            logs.append({"symbol": symbol, "family": family, "page": page, "status": "empty"})
            break
        rows.extend(data)
        ts_vals = [int(x.get("timestamp", x.get("time", 0))) for x in data if x.get("timestamp", x.get("time", 0))]
        if not ts_vals:
            break
        oldest = pd.to_datetime(min(ts_vals), unit="ms", utc=True)
        newest = pd.to_datetime(max(ts_vals), unit="ms", utc=True)
        logs.append({"symbol": symbol, "family": family, "page": page, "status": "success", "rows_total": len(rows), "oldest": oldest, "newest": newest})
        next_end = oldest - pd.Timedelta(milliseconds=1)
        if next_end >= end:
            break
        end = next_end
        time.sleep(0.08)
        if fallback_used == "latest_no_time":
            break
    df = pd.DataFrame(rows)
    if df.empty:
        return df, logs, fallback_used
    if "symbol" not in df:
        df["symbol"] = symbol
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True).dt.tz_convert(None)
    df = df[df["timestamp"] >= (pd.Timestamp.now(tz="UTC").tz_convert(None) - pd.Timedelta(days=days + 1))]
    df["asof_available_ts"] = df["timestamp"] + pd.Timedelta(minutes=5)
    df["source"] = "binance_public_latest30"
    df["data_family"] = "open_interest_history" if family == "oi" else "taker_buy_sell_volume"
    for c in df.columns:
        if c not in {"symbol", "timestamp", "asof_available_ts", "source", "data_family"}:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() > 0:
                df[c] = conv
    df = df.drop_duplicates(["symbol", "timestamp"]).sort_values("timestamp")
    return df, logs, fallback_used


def _quality(df: pd.DataFrame, symbol: str, family: str, period: str, fallback: str, err_count: int) -> Dict[str, Any]:
    expected = int(30 * 24 * 60 / 5)
    if df.empty:
        return {"symbol": symbol, "data_family": family, "period": period, "rows": 0, "start_ts": "", "end_ts": "", "coverage_days": 0, "expected_rows_30d_5m": expected, "coverage_ratio": 0, "gap_count": 0, "duplicate_count": 0, "endpoint_error_count": err_count, "fallback_used": fallback, "latest_30d_only_confirmed": True, "usable_for_recent30d_validation": False}
    ts = df["timestamp"].sort_values()
    gaps = int((ts.diff().dt.total_seconds().fillna(300) > 900).sum())
    days = float((ts.max() - ts.min()).total_seconds() / 86400)
    return {"symbol": symbol, "data_family": family, "period": period, "rows": len(df), "start_ts": ts.min(), "end_ts": ts.max(), "coverage_days": days, "expected_rows_30d_5m": expected, "coverage_ratio": min(len(df) / expected, 1.0), "gap_count": gaps, "duplicate_count": int(df.duplicated(["symbol", "timestamp"]).sum()), "endpoint_error_count": err_count, "fallback_used": fallback, "latest_30d_only_confirmed": True, "usable_for_recent30d_validation": days >= 1 and len(df) > 100}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    families = [f.strip().lower() for f in args.data_family.split(",") if f.strip()]
    if args.dry_run:
        return {"dry_run": True, "symbols": symbols, "families": families, "latest_days_only": args.days, "private_api_calls": False, "order_endpoint_calls": False}
    OUT.mkdir(parents=True, exist_ok=True)
    registry_rows, quality_rows, all_logs = [], [], []
    for symbol in symbols:
        for fam in families:
            df, logs, fallback = _fetch_one(symbol, fam, args.period, args.days, args.limit, args.chunk_hours)
            all_logs.extend(logs)
            err_count = sum(1 for r in logs if str(r.get("status", "")).startswith("failed"))
            norm_family = "open_interest_history" if fam == "oi" else "taker_buy_sell_volume"
            out_dir = ROOT / "normalized" / norm_family
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{symbol}.parquet"
            if not df.empty:
                if args.resume and out_path.exists() and not args.force_refresh:
                    old = pd.read_parquet(out_path)
                    df = pd.concat([old, df], ignore_index=True).drop_duplicates(["symbol", "timestamp"], keep="last").sort_values("timestamp")
                df.to_parquet(out_path, index=False)
            q = _quality(df, symbol, norm_family, args.period, fallback, err_count)
            quality_rows.append(q)
            registry_rows.append({"symbol": symbol, "data_family": norm_family, "path": str(out_path), "rows": q["rows"], "start": q["start_ts"], "end": q["end_ts"], "fetch_status": "success" if q["rows"] else "empty_or_failed", "latest30_only": True})
    ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(registry_rows).to_csv(ROOT / "cache_registry_latest30_oi_taker.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(ROOT / "data_quality_latest30_oi_taker.csv", index=False)
    with (ROOT / "fetch_log.jsonl").open("a", encoding="utf-8") as f:
        for row in all_logs:
            f.write(json.dumps(row, default=str) + "\n")
    summary = {"symbols": symbols, "families": families, "period": args.period, "days": args.days, "rows": int(sum(r["rows"] for r in quality_rows)), "private_api_calls": False, "order_endpoint_calls": False, "production_ready": False, "promotion_ready": False}
    (OUT / "latest30_fetch_summary.json").write_text(_json(summary), encoding="utf-8")
    qdf = pd.DataFrame(quality_rows)
    qdf.to_csv(OUT / "latest30_symbol_status.csv", index=False)
    qdf.groupby("data_family").agg(symbols=("symbol", "nunique"), avg_coverage_ratio=("coverage_ratio", "mean"), usable_symbols=("usable_for_recent30d_validation", "sum")).reset_index().to_csv(OUT / "latest30_data_family_status.csv", index=False)
    qdf.to_csv(OUT / "latest30_coverage_report.csv", index=False)
    (OUT / "latest30_fetch_report.md").write_text("# Latest 30d OI/Taker Fetch Report\n\nThis is latest 30 days only, not arbitrary historical 30d.\n\n```json\n" + _json(summary) + "\n```\n", encoding="utf-8")
    return summary


def main() -> int:
    global ROOT
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--symbols", default=",".join(SYMBOLS))
    p.add_argument("--period", default="5m")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--data-family", default="oi,taker")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--chunk-hours", type=int, default=36)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--rate-limit-safe", action="store_true", default=True)
    p.add_argument("--no-private", action="store_true", default=True)
    p.add_argument("--output-root", default=str(ROOT))
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    ROOT = Path(args.output_root)
    res = run(args)
    print(_json(res) if args.json else f"latest30 rows={res.get('rows', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
