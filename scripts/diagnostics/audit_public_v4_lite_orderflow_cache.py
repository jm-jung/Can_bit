"""Audit public V4-lite orderflow cache readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

CACHE = Path("data/diagnostics/research_orderflow_data_cache")
OUT = Path("data/diagnostics/public_v4_lite_btc_centric_retry/cache_audit")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT"]
FAMILIES = ["funding", "oi", "taker", "mark", "spot"]


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _read(symbol: str, family: str) -> pd.DataFrame:
    p = CACHE / "normalized" / family / f"{symbol}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "asof_available_ts" in df:
        df["asof_available_ts"] = pd.to_datetime(df["asof_available_ts"], errors="coerce")
    return df


def _quality(symbol: str, family: str, days_required: int) -> Dict[str, Any]:
    df = _read(symbol, family)
    if df.empty:
        return {"symbol": symbol, "data_family": family, "rows": 0, "start": "", "end": "", "days": 0, "coverage_pass": False, "missing_ratio": 1.0, "gap_ratio": 1.0, "duplicate_ratio": 0.0, "asof_pass": False}
    ts = df["timestamp"].dropna().sort_values()
    days = float((ts.max() - ts.min()).total_seconds() / 86400) if len(ts) else 0.0
    gaps = ts.diff().dt.total_seconds().fillna(300)
    gap_threshold = 12 * 3600 if family == "funding" else 1800
    return {
        "symbol": symbol,
        "data_family": family,
        "rows": len(df),
        "start": ts.min(),
        "end": ts.max(),
        "days": days,
        "coverage_pass": days >= days_required * 0.9 and len(df) > 0,
        "missing_ratio": float(df.isna().mean().mean()),
        "gap_ratio": float((gaps > gap_threshold).mean()),
        "duplicate_ratio": float(df.duplicated(subset=["symbol", "timestamp"]).mean()) if "symbol" in df else float(df.duplicated(subset=["timestamp"]).mean()),
        "asof_pass": "asof_available_ts" in df and bool((df["asof_available_ts"] >= df["timestamp"]).fillna(False).mean() > 0.99),
    }


def run(days_required: int = 180) -> Dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    registry_rows = []
    for p in (CACHE / "normalized").glob("*/*.parquet"):
        try:
            df_meta = pd.read_parquet(p, columns=["symbol", "timestamp"])
            registry_rows.append({"symbol": str(df_meta["symbol"].dropna().iloc[0]) if len(df_meta) and "symbol" in df_meta else p.stem, "data_family": p.parent.name, "path": str(p), "rows": len(df_meta), "start": pd.to_datetime(df_meta["timestamp"], errors="coerce").min() if len(df_meta) else "", "end": pd.to_datetime(df_meta["timestamp"], errors="coerce").max() if len(df_meta) else "", "fetch_status": "success" if len(df_meta) else "empty"})
        except Exception:
            registry_rows.append({"symbol": p.stem, "data_family": p.parent.name, "path": str(p), "rows": 0, "start": "", "end": "", "fetch_status": "read_failed"})
    if registry_rows:
        pd.DataFrame(registry_rows).to_csv(CACHE / "cache_registry.csv", index=False)
    rows: List[Dict[str, Any]] = []
    for symbol in SYMBOLS:
        for family in FAMILIES:
            rows.append(_quality(symbol, family, days_required))
    q = pd.DataFrame(rows)
    q.to_csv(OUT / "cache_quality_summary.csv", index=False)
    q.to_csv(OUT / "symbol_data_quality.csv", index=False)
    fam = q.groupby("data_family").agg(symbols_with_data=("rows", lambda s: int((s > 0).sum())), coverage_pass_symbols=("coverage_pass", "sum"), avg_days=("days", "mean"), avg_missing=("missing_ratio", "mean")).reset_index()
    fam.to_csv(OUT / "data_family_quality.csv", index=False)
    btc_pass = bool(q[(q["symbol"].eq("BTCUSDT")) & (q["data_family"].isin(FAMILIES))]["coverage_pass"].all())
    context_pass = int(q[(~q["symbol"].eq("BTCUSDT")) & (q["data_family"].isin(["oi", "taker", "funding"]))].groupby("symbol")["coverage_pass"].all().sum())
    min_ready = btc_pass and context_pass >= 7
    readiness = pd.DataFrame([{"criterion": "minimum_v4_lite", "pass": min_ready, "status": "V4_LITE_DATA_READY" if min_ready else "V4_LITE_DATA_PARTIAL_READY" if context_pass >= 3 else "V4_LITE_DATA_INSUFFICIENT", "btc_pass": btc_pass, "context_symbols_pass": context_pass}])
    readiness.to_csv(OUT / "minimum_v4_lite_readiness.csv", index=False)
    basis_rows = []
    for symbol in SYMBOLS:
        mark = _read(symbol, "mark")
        spot = _read(symbol, "spot")
        possible = not mark.empty and not spot.empty
        basis_rows.append({"symbol": symbol, "basis_proxy_possible": possible, "mark_rows": len(mark), "spot_rows": len(spot)})
    pd.DataFrame(basis_rows).to_csv(OUT / "basis_computability.csv", index=False)
    cvd_rows = []
    for symbol in SYMBOLS:
        agg = _read(symbol, "aggtrades")
        cvd_rows.append({"symbol": symbol, "proxy_cvd_possible": not agg.empty, "aggtrade_rows": len(agg), "status": "AGGTRADES_PROXY_CVD_READY" if len(agg) > 10000 else "AGGTRADES_PROXY_CVD_SAMPLE_ONLY" if len(agg) else "AGGTRADES_PROXY_CVD_MISSING"})
    pd.DataFrame(cvd_rows).to_csv(OUT / "proxy_cvd_feasibility.csv", index=False)
    (OUT / "cache_quality_report.md").write_text("# Cache Quality Report\n\n```json\n" + _json({"readiness": readiness.to_dict("records"), "families": fam.to_dict("records")}) + "\n```\n", encoding="utf-8")
    return {"minimum_status": readiness.iloc[0]["status"], "btc_pass": btc_pass, "context_symbols_pass": context_pass, "private_api_calls": False, "order_endpoint_calls": False}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true")
    p.add_argument("--days-required", type=int, default=180)
    args = p.parse_args()
    result = run(args.days_required)
    print(_json(result) if args.json else result["minimum_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
