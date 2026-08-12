"""Diagnostics-only forward orderflow collector V4."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path("data/diagnostics/forward_orderflow_collector_v4")
FAPI = "https://fapi.binance.com"
SAPI = "https://api.binance.com"


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _get(base: str, path: str, params: Dict[str, Any]) -> tuple[int, Any, float, str]:
    endpoint = base + path
    url = endpoint + "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    start = time.time()
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return resp.status, payload, (time.time() - start) * 1000, endpoint
    except Exception as exc:
        return 0, {"error": str(exc)[:240]}, (time.time() - start) * 1000, endpoint


def _dirs(run_ts: str) -> Dict[str, Path]:
    d = {
        "state": ROOT / "state",
        "run": ROOT / "runs" / run_ts,
        "cache": ROOT / "cache",
        "health": ROOT / "health",
        "logs": ROOT / "logs",
        "design": Path("data/diagnostics/proxy_cvd_expansion_and_forward_orderflow_collector/forward_collector"),
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def _append_parquet(path: Path, df: pd.DataFrame, subset: List[str]) -> None:
    if df.empty:
        return
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(subset, keep="last")
    df.to_parquet(path, index=False)


def _snapshot(run_ts: str, symbol: str, family: str, source: str, payload: Any, status: int, latency: float, endpoint: str) -> Dict[str, Any]:
    now = pd.Timestamp.now("UTC").tz_localize(None)
    numeric = {}
    if isinstance(payload, dict):
        for k, v in payload.items():
            try:
                numeric[k] = float(v)
            except Exception:
                pass
    return {"run_ts": run_ts, "snapshot_ts": now, "symbol": symbol, "data_family": family, "source": source, "value_json": json.dumps(payload, default=str), "numeric_fields": json.dumps(numeric), "latency_ms": latency, "http_status": status, "endpoint": endpoint, "asof_available_ts": now, "quality_missing": payload in ({}, []) or status != 200, "quality_stale": False, "quality_error": status != 200, "is_forward_only": True, "is_private_endpoint": False, "production_action": "none"}


def collect_once(args: argparse.Namespace) -> Dict[str, Any]:
    if args.production_action != "none":
        raise SystemExit("production-action must be none")
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    run_ts = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")
    d = _dirs(run_ts)
    snapshots, orderbooks, trades, liqs, logs = [], [], [], [], []
    for symbol in symbols:
        calls = [
            ("oi_current", FAPI, "/fapi/v1/openInterest", {"symbol": symbol}),
            ("premium_mark", FAPI, "/fapi/v1/premiumIndex", {"symbol": symbol}),
            ("funding_recent", FAPI, "/fapi/v1/fundingRate", {"symbol": symbol, "limit": 1}),
            ("oi_latest_history", FAPI, "/futures/data/openInterestHist", {"symbol": symbol, "period": "5m", "limit": 12}),
            ("taker_latest_history", FAPI, "/futures/data/takerlongshortRatio", {"symbol": symbol, "period": "5m", "limit": 12}),
            ("spot_price", SAPI, "/api/v3/ticker/price", {"symbol": symbol}),
        ]
        for family, base, path, params in calls:
            status, payload, latency, endpoint = _get(base, path, params)
            snapshots.append(_snapshot(run_ts, symbol, family, "binance_public", payload, status, latency, endpoint))
            logs.append({"run_ts": run_ts, "symbol": symbol, "family": family, "status": status, "endpoint": endpoint})
            time.sleep(0.03)
        if args.include_orderbook:
            status, payload, latency, endpoint = _get(FAPI, "/fapi/v1/depth", {"symbol": symbol, "limit": args.orderbook_depth_limit})
            snapshots.append(_snapshot(run_ts, symbol, "orderbook_depth", "binance_public_forward_snapshot", payload, status, latency, endpoint))
            if status == 200 and isinstance(payload, dict):
                bids = [(float(p), float(q)) for p, q in payload.get("bids", [])]
                asks = [(float(p), float(q)) for p, q in payload.get("asks", [])]
                def depth(side: list[tuple[float, float]], n: int) -> float:
                    return sum(p * q for p, q in side[:n])
                best_bid = bids[0][0] if bids else None
                best_ask = asks[0][0] if asks else None
                mid = (best_bid + best_ask) / 2 if best_bid and best_ask else None
                row = {"run_ts": run_ts, "snapshot_ts": pd.Timestamp.now("UTC").tz_localize(None), "symbol": symbol, "best_bid": best_bid, "best_ask": best_ask, "spread": (best_ask - best_bid) if best_bid and best_ask else None, "mid": mid, "bid_depth_top_5": depth(bids, 5), "ask_depth_top_5": depth(asks, 5), "bid_depth_top_20": depth(bids, 20), "ask_depth_top_20": depth(asks, 20), "bid_depth_top_100": depth(bids, 100), "ask_depth_top_100": depth(asks, 100), "is_forward_only": True, "production_action": "none"}
                for n in [5, 20, 100]:
                    bd, ad = row[f"bid_depth_top_{n}"], row[f"ask_depth_top_{n}"]
                    row[f"depth_imbalance_{n}"] = (bd - ad) / (bd + ad) if (bd + ad) else None
                row["thin_book_score"] = 1 / max(row["bid_depth_top_20"] + row["ask_depth_top_20"], 1)
                orderbooks.append(row)
        if args.include_recent_aggtrades:
            status, payload, latency, endpoint = _get(FAPI, "/fapi/v1/aggTrades", {"symbol": symbol, "limit": 500})
            snapshots.append(_snapshot(run_ts, symbol, "recent_aggtrades", "binance_public", payload, status, latency, endpoint))
            if status == 200 and isinstance(payload, list):
                for x in payload:
                    qty = float(x.get("q", 0))
                    price = float(x.get("p", 0))
                    buyer_maker = bool(x.get("m"))
                    trades.append({"run_ts": run_ts, "symbol": symbol, "trade_ts": pd.to_datetime(int(x.get("T", 0)), unit="ms"), "agg_trade_id": x.get("a"), "price": price, "quantity": qty, "is_buyer_maker": buyer_maker, "aggressive_side_proxy": "sell" if buyer_maker else "buy", "quote_qty": price * qty, "production_action": "none"})
        if args.include_liquidation_if_available:
            status, payload, latency, endpoint = _get(FAPI, "/fapi/v1/forceOrders", {"symbol": symbol, "limit": 50})
            snapshots.append(_snapshot(run_ts, symbol, "liquidation_force_orders_if_available", "binance_public_if_available", payload, status, latency, endpoint))
            if status == 200 and isinstance(payload, list):
                for x in payload:
                    liqs.append({"run_ts": run_ts, "event_ts": pd.to_datetime(int(x.get("time", 0)), unit="ms"), "symbol": symbol, "side": x.get("side"), "price": float(x.get("price", 0)), "qty": float(x.get("origQty", 0)), "notional": float(x.get("price", 0)) * float(x.get("origQty", 0)), "source": "binance_public_forceOrders", "is_forward_only": True, "production_action": "none"})
    snap_df = pd.DataFrame(snapshots)
    ob_df = pd.DataFrame(orderbooks)
    tr_df = pd.DataFrame(trades)
    liq_df = pd.DataFrame(liqs)
    snap_df.to_parquet(d["run"] / "snapshots.parquet", index=False)
    ob_df.to_parquet(d["run"] / "orderbook_snapshots.parquet", index=False)
    tr_df.to_parquet(d["run"] / "recent_aggtrades.parquet", index=False)
    liq_df.to_parquet(d["run"] / "liquidation_events.parquet", index=False)
    _append_parquet(d["cache"] / "forward_orderflow_snapshots.parquet", snap_df, ["run_ts", "symbol", "data_family"])
    _append_parquet(d["cache"] / "forward_orderbook_snapshots.parquet", ob_df, ["run_ts", "symbol"])
    _append_parquet(d["cache"] / "forward_recent_aggtrades.parquet", tr_df, ["symbol", "agg_trade_id"])
    _append_parquet(d["cache"] / "forward_liquidation_events.parquet", liq_df, ["run_ts", "symbol", "event_ts", "side", "price"])
    family_cov = snap_df.groupby("data_family").agg(rows=("symbol", "size"), ok=("quality_error", lambda x: int((~x).sum()))).reset_index()
    sym_cov = snap_df.groupby("symbol").agg(rows=("data_family", "size"), errors=("quality_error", "sum")).reset_index()
    health = pd.DataFrame([{"run_ts": run_ts, "symbols": len(symbols), "snapshot_rows": len(snap_df), "orderbook_rows": len(ob_df), "recent_aggtrades_rows": len(tr_df), "liquidation_rows": len(liq_df), "error_count": int(snap_df["quality_error"].sum()) if len(snap_df) else 0, "production_action": "none", "private_endpoint_calls": 0, "order_endpoint_calls": 0, "account_balance_position_calls": 0}])
    family_cov.to_csv(d["health"] / "data_family_coverage.csv", index=False)
    sym_cov.to_csv(d["health"] / "symbol_coverage.csv", index=False)
    health.to_csv(d["health"] / "collector_health.csv", index=False)
    with (d["logs"] / "collector_log.jsonl").open("a", encoding="utf-8") as f:
        for row in logs:
            f.write(json.dumps(row, default=str) + "\n")
    state = {"last_run_ts": run_ts, "last_run_path": str(d["run"]), "production_action": "none", "private_endpoint_calls": 0, "order_endpoint_calls": 0, "account_balance_position_calls": 0}
    (d["state"] / "collector_state.json").write_text(_json(state), encoding="utf-8")
    (d["run"] / "run_summary.json").write_text(_json({**state, **health.iloc[0].to_dict()}), encoding="utf-8")
    return {"status": "FORWARD_ORDERFLOW_COLLECTOR_ONE_SHOT_SUCCESS", **health.iloc[0].to_dict(), "state_path": str(d["state"] / "collector_state.json"), "production_ready": False, "promotion_ready": False}


def run(args: argparse.Namespace) -> dict:
    if args.dry_run:
        return {"dry_run": True, "symbols": [s.strip().upper() for s in args.symbols.split(",") if s.strip()], "production_action": args.production_action, "private_endpoint_calls": 0, "order_endpoint_calls": 0, "account_balance_position_calls": 0, "output_root": str(args.output_root), "production_ready": False}
    global ROOT
    ROOT = Path(args.output_root)
    if args.once:
        return collect_once(args)
    return {"status": "NOOP_USE_ONCE_OR_DRY_RUN", "production_action": args.production_action, "production_ready": False}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--once", action="store_true")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT")
    p.add_argument("--include-orderbook", action="store_true")
    p.add_argument("--orderbook-depth-limit", type=int, default=100)
    p.add_argument("--include-liquidation-if-available", action="store_true")
    p.add_argument("--include-recent-aggtrades", action="store_true")
    p.add_argument("--production-action", default="none")
    p.add_argument("--output-root", default=str(ROOT))
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    res = run(args)
    print(_json(res) if args.json else res["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
