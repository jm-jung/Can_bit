"""Build proxy CVD features from Binance futures aggTrades."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

INPUT = Path("data/diagnostics/research_orderflow_data_cache/normalized/futures_aggtrades")
OUTPUT = Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd")
RUN = Path("data/diagnostics/proxy_cvd_expansion_and_forward_orderflow_collector/proxy_cvd_build")


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _z(s: pd.Series, n: int = 96) -> pd.Series:
    return (s - s.rolling(n, min_periods=max(10, n // 10)).mean()) / s.rolling(n, min_periods=max(10, n // 10)).std()


def _build(symbol: str, tf: str, trades: pd.DataFrame) -> pd.DataFrame:
    freq = tf.replace("m", "min") if tf.endswith("m") else tf
    df = trades.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "price", "quantity"]).sort_values("timestamp")
    df["quote"] = df["price"] * df["quantity"]
    df["aggressive_buy_qty_proxy"] = np.where(~df["is_buyer_maker"].astype(bool), df["quantity"], 0.0)
    df["aggressive_sell_qty_proxy"] = np.where(df["is_buyer_maker"].astype(bool), df["quantity"], 0.0)
    df["aggressive_buy_quote_proxy"] = np.where(~df["is_buyer_maker"].astype(bool), df["quote"], 0.0)
    df["aggressive_sell_quote_proxy"] = np.where(df["is_buyer_maker"].astype(bool), df["quote"], 0.0)
    q95 = df["quantity"].quantile(0.95) if len(df) else np.nan
    df["large_trade"] = df["quantity"] >= q95
    df["large_trade_signed_qty"] = np.where(~df["is_buyer_maker"].astype(bool), df["quantity"], -df["quantity"])
    g = df.set_index("timestamp").resample(freq)
    out = g.agg(
        aggressive_buy_qty_proxy=("aggressive_buy_qty_proxy", "sum"),
        aggressive_sell_qty_proxy=("aggressive_sell_qty_proxy", "sum"),
        aggressive_buy_quote_proxy=("aggressive_buy_quote_proxy", "sum"),
        aggressive_sell_quote_proxy=("aggressive_sell_quote_proxy", "sum"),
        trade_count=("quantity", "size"),
        close_proxy_price=("price", "last"),
        large_trade_count=("large_trade", "sum"),
        large_trade_delta=("large_trade_signed_qty", "sum"),
    ).reset_index()
    out["symbol"] = symbol
    out["timeframe"] = tf
    out["delta_qty_proxy"] = out["aggressive_buy_qty_proxy"] - out["aggressive_sell_qty_proxy"]
    out["delta_quote_proxy"] = out["aggressive_buy_quote_proxy"] - out["aggressive_sell_quote_proxy"]
    out["cvd_qty_proxy"] = out["delta_qty_proxy"].cumsum()
    out["cvd_quote_proxy"] = out["delta_quote_proxy"].cumsum()
    out["cvd_slope"] = out["cvd_qty_proxy"].diff()
    out["cvd_zscore"] = _z(out["delta_qty_proxy"], 96)
    out["cvd_rolling_delta_5"] = out["delta_qty_proxy"].rolling(5, min_periods=1).sum()
    out["cvd_rolling_delta_12"] = out["delta_qty_proxy"].rolling(12, min_periods=1).sum()
    out["cvd_rolling_delta_36"] = out["delta_qty_proxy"].rolling(36, min_periods=1).sum()
    out["price_return"] = out["close_proxy_price"].pct_change()
    out["cvd_divergence_vs_price"] = np.sign(out["cvd_slope"].fillna(0)) - np.sign(out["price_return"].fillna(0))
    out["cvd_reclaim"] = ((out["cvd_zscore"] > 1.0) & (out["price_return"] > 0)).astype(int)
    out["cvd_breakdown"] = ((out["cvd_zscore"] < -1.0) & (out["price_return"] < 0)).astype(int)
    out["cvd_absorption_proxy"] = ((out["cvd_zscore"].abs() > 1.5) & (out["price_return"].abs() < out["price_return"].rolling(96, min_periods=10).std())).astype(int)
    out["cvd_exhaustion_proxy"] = ((out["cvd_zscore"].abs() > 2.0) & (out["price_return"].shift(-1).abs() < out["price_return"].abs())).astype(int)
    out["large_trade_aggression_score"] = _z(out["large_trade_delta"], 96)
    total_qty = out["aggressive_buy_qty_proxy"] + out["aggressive_sell_qty_proxy"]
    out["buy_aggression_ratio"] = out["aggressive_buy_qty_proxy"] / total_qty.replace(0, np.nan)
    out["sell_aggression_ratio"] = out["aggressive_sell_qty_proxy"] / total_qty.replace(0, np.nan)
    out["cvd_data_quality_score"] = 1.0 - out[["close_proxy_price", "delta_qty_proxy", "trade_count"]].isna().mean(axis=1)
    out["asof_available_ts"] = out["timestamp"] + pd.Timedelta(freq)
    out["cvd_type"] = "proxy_from_aggtrades_is_buyer_maker"
    return out


def run(args: argparse.Namespace) -> Dict[str, Any]:
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    tfs = [t.strip() for t in args.timeframes.split(",") if t.strip()]
    if args.dry_run:
        return {"dry_run": True, "symbols": symbols, "timeframes": tfs, "proxy_cvd": True, "true_cvd": False}
    RUN.mkdir(parents=True, exist_ok=True)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    rows, schema = [], {}
    for symbol in symbols:
        p = Path(args.input_root) / f"{symbol}.parquet"
        if not p.exists():
            rows.append({"symbol": symbol, "status": "missing_aggtrades", "rows": 0})
            continue
        trades = pd.read_parquet(p)
        for tf in tfs:
            out = _build(symbol, tf, trades)
            out_path = Path(args.output_root) / f"{symbol}_{tf}.parquet"
            out.to_parquet(out_path, index=False)
            rows.append({"symbol": symbol, "timeframe": tf, "status": "success", "rows": len(out), "start_ts": out["timestamp"].min(), "end_ts": out["timestamp"].max(), "path": str(out_path)})
            schema = {c: str(out[c].dtype) for c in out.columns}
    q = pd.DataFrame(rows)
    q.to_csv(RUN / "proxy_cvd_quality_summary.csv", index=False)
    (RUN / "proxy_cvd_feature_schema.json").write_text(_json(schema), encoding="utf-8")
    summary = {"symbols": symbols, "timeframes": tfs, "rows": int(q.get("rows", pd.Series(dtype=int)).sum()) if len(q) else 0, "proxy_note": "is_buyer_maker=true maps to aggressive sell proxy; false maps to aggressive buy proxy. This is not true CVD.", "production_ready": False}
    (RUN / "proxy_cvd_build_summary.json").write_text(_json(summary), encoding="utf-8")
    (RUN / "proxy_cvd_build_report.md").write_text("# Proxy CVD Build Report\n\nThis is proxy CVD from futures aggTrades, not true CVD.\n\n```json\n" + _json(summary) + "\n```\n", encoding="utf-8")
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    p.add_argument("--timeframes", default="1m,5m,15m,1h")
    p.add_argument("--input-root", default=str(INPUT))
    p.add_argument("--output-root", default=str(OUTPUT))
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    res = run(args)
    print(_json(res) if args.json else f"proxy_cvd rows={res.get('rows', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
