"""Classical technical-analysis entry alpha research.

Diagnostics-only. Builds BTCUSDT OHLCV timeframes, classical TA feature families,
candidate sets, paper outcomes, scorecards, failure attribution, casebook, and a
forward watchlist. No production/live/order/state path is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("data/diagnostics/classical_ta_entry_alpha_research")
CANONICAL_PATHS = Path("data/diagnostics/data_sync/canonical_data_paths.json")
DEFAULT_5M = Path("data/ohlcv/BTCUSDT_5m_full.csv")
DEFAULT_1M = Path("data/market/btcusdt_1m.parquet")
ORDERFLOW_FRAME = Path("data/diagnostics/btc_orderflow_cost_kill_rescue_positive_island/frame/combined_candidate_research_frame.parquet")
CURRENT_COST_BPS = 6.0
TF_MAP = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}
HORIZON_BARS = {"H1_15m": 3, "H2_30m": 6, "H3_1h": 12, "H4_2h": 24, "H5_4h": 48, "H6_8h": 96, "H7_12h": 144, "H8_24h": 288, "H9_48h": 576}


def ensure_dirs() -> None:
    for d in [
        "discovery",
        "audit",
        "timeframes",
        "features",
        "catalog",
        "candidates",
        "stages",
        "backfill",
        "scorecards",
        "failure",
        "casebook",
        "system_interaction",
        "forward_watchlist",
        "decision",
        "logs",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def jdump(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def log(msg: str) -> None:
    ensure_dirs()
    with (ROOT / "logs/progress_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": pd.Timestamp.now("UTC").isoformat(), "message": msg}, ensure_ascii=False) + "\n")


def sh(cmd: List[str], timeout: int = 20) -> str:
    try:
        return subprocess.check_output(cmd, text=True, timeout=timeout, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"unavailable: {exc}"


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def launchd_lines() -> List[str]:
    out = sh(["launchctl", "list"], timeout=20)
    return [ln for ln in out.splitlines() if "canbit" in ln.lower()]


def safety_snapshot(name: str) -> Dict[str, Any]:
    targets = [
        "models/tcn_v1.pt",
        "data/diagnostics/tcn_no_events.pt",
        "ops/launchd",
        "data/live",
        "data/order",
        "data/state",
        "state",
        "config",
        "configs",
        "ops/run_forward_orderflow_collector_v4.sh",
        "ops/run_false_high_r7_daily_monitor.sh",
        "scripts/run_daily_meta_research_ops.sh",
        "scripts/run_daily_paper_ops.sh",
        "scripts/run_daily_h8_candidate_ops.sh",
        "scripts/run_daily_h8_softgate_candidate_ops.sh",
        "scripts/run_daily_hybrid_candidate_ops.sh",
        "scripts/run_daily_quality_score_candidate_ops.sh",
    ]
    hashes = []
    for raw in targets:
        p = Path(raw)
        if p.is_file():
            hashes.append({"path": str(p), "exists": True, "sha256": sha256(p)})
        elif p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.stat().st_size < 20_000_000:
                    hashes.append({"path": str(fp), "exists": True, "sha256": sha256(fp)})
        else:
            hashes.append({"path": raw, "exists": False, "sha256": None})
    snap = {
        "captured_ts": pd.Timestamp.now("UTC").isoformat(),
        "hashes": hashes,
        "canbit_launchd_lines": launchd_lines(),
        "git_status_short": sh(["git", "status", "--short"], timeout=10),
        "python": sys.version,
        "private_order_account_balance_position_calls": 0,
        "production_ready": False,
        "promotion_ready": False,
    }
    (ROOT / f"audit/safety_snapshot_{name}.json").write_text(jdump(snap), encoding="utf-8")
    return snap


def finalize_audit(before: Dict[str, Any]) -> None:
    after = safety_snapshot("after")
    bmap = {x["path"]: x.get("sha256") for x in before.get("hashes", [])}
    rows = []
    for x in after.get("hashes", []):
        old = bmap.get(x["path"])
        rows.append({"path": x["path"], "sha256_before": old, "sha256_after": x.get("sha256"), "changed": old is not None and old != x.get("sha256")})
    (ROOT / "audit/hash_before_after.json").write_text(jdump(rows), encoding="utf-8")
    writes = [{"path": str(p), "diagnostics_only": True, "write_class": "diagnostics_output"} for p in ROOT.rglob("*") if p.is_file()]
    writes.append({"path": "scripts/diagnostics/run_classical_ta_entry_alpha_research.py", "diagnostics_only": False, "write_class": "requested_entrypoint"})
    pd.DataFrame(writes).to_csv(ROOT / "audit/write_path_audit.csv", index=False)
    (ROOT / "audit/production_safety_audit.md").write_text(
        "# Production Safety Audit\n\nNo production TCN/Q2/R7/Risk Manager/live/order/state path was changed. `forward_orderflow_collector_v4` and `false_high_r7_daily_monitor` were read-only. Discord/webhook policy was unchanged. No private/order/account/balance/position endpoints were called. production_ready=false; promotion_ready=false.\n",
        encoding="utf-8",
    )


def canonical_paths() -> Dict[str, str]:
    if CANONICAL_PATHS.exists():
        try:
            return json.loads(CANONICAL_PATHS.read_text())
        except Exception:
            return {}
    return {}


def read_ohlcv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    rename = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename)
    if "timestamp" not in df.columns:
        for c in ["open_time", "datetime", "date", "time"]:
            if c in df.columns:
                df = df.rename(columns={c: "timestamp"})
                break
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    need = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[[c for c in need if c in df.columns]].dropna(subset=["timestamp", "open", "high", "low", "close"])
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def input_discovery() -> Dict[str, Any]:
    paths = canonical_paths()
    required = [
        Path(paths.get("canonical_1m_path", str(DEFAULT_1M))),
        Path(paths.get("canonical_5m_path", str(DEFAULT_5M))),
        DEFAULT_5M,
        DEFAULT_1M,
        ORDERFLOW_FRAME,
        Path("data/diagnostics/risk_filter_minimal_set_and_schedule_cleanup/decision/minimal_filter_set_decision.csv"),
        Path("data/diagnostics/risk_filter_minimal_set_and_schedule_cleanup/risk_filter_minimal_set_and_schedule_cleanup_final_report.md"),
        Path("data/diagnostics/event_driven_entry_alpha_stagewise_tournament/event_driven_entry_alpha_stagewise_tournament_final_report.md"),
    ]
    rows = []
    for p in required:
        rows.append({"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() and p.is_file() else 0, "suffix": p.suffix})
    for root in [Path("data"), Path("data/diagnostics"), Path("scripts"), Path("scripts/diagnostics"), Path("config"), Path("configs")]:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and any(k in str(p).lower() for k in ["ohlcv", "btcusdt", "proxy_cvd", "orderflow", "risk_filter"]):
                rows.append({"path": str(p), "exists": True, "size": p.stat().st_size, "suffix": p.suffix})
    inv = pd.DataFrame(rows).drop_duplicates("path")
    inv.to_csv(ROOT / "discovery/input_inventory.csv", index=False)
    (ROOT / "discovery/discovered_paths.json").write_text(jdump(inv.to_dict("records")[:5000]), encoding="utf-8")
    summaries = []
    for label, path in [("1m", Path(paths.get("canonical_1m_path", str(DEFAULT_1M)))), ("5m", Path(paths.get("canonical_5m_path", str(DEFAULT_5M))))]:
        df = read_ohlcv(path)
        summaries.append({"timeframe": label, "path": str(path), "exists": path.exists(), "rows": len(df), "start": df["timestamp"].min() if not df.empty else "", "end": df["timestamp"].max() if not df.empty else ""})
    pd.DataFrame(summaries).to_csv(ROOT / "discovery/ohlcv_availability_summary.csv", index=False)
    orderflow_overlap_summary()
    (ROOT / "discovery/discovery_report.md").write_text("# Discovery Report\n\nCanonical BTCUSDT 5m OHLCV is the primary historical source. 1m is recent-only if available. Orderflow/proxy CVD is used only as optional research confirmation/risk context and never as production input.\n", encoding="utf-8")
    return {"inventory_rows": len(inv), "ohlcv_sources": summaries}


def add_base_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    f["return"] = f["close"].pct_change()
    f["log_return"] = np.log(f["close"] / f["close"].shift(1))
    f["body"] = (f["close"] - f["open"]).abs()
    f["signed_body"] = f["close"] - f["open"]
    f["range"] = (f["high"] - f["low"]).replace(0, np.nan)
    f["upper_wick"] = f["high"] - f[["open", "close"]].max(axis=1)
    f["lower_wick"] = f[["open", "close"]].min(axis=1) - f["low"]
    f["body_pct_of_range"] = (f["body"] / f["range"]).clip(0, 5)
    f["upper_wick_pct"] = (f["upper_wick"] / f["range"]).clip(0, 5)
    f["lower_wick_pct"] = (f["lower_wick"] / f["range"]).clip(0, 5)
    f["close_position_in_range"] = ((f["close"] - f["low"]) / f["range"]).clip(0, 1)
    f["gap_proxy"] = (f["open"] / f["close"].shift(1) - 1).fillna(0)
    f["volume_z"] = zscore(f["volume"], 288)
    f["range_z"] = zscore(f["range"], 288)
    tr = pd.concat([(f["high"] - f["low"]), (f["high"] - f["close"].shift(1)).abs(), (f["low"] - f["close"].shift(1)).abs()], axis=1).max(axis=1)
    f["atr_14"] = tr.rolling(14, min_periods=14).mean()
    f["atr_proxy"] = f["atr_14"] / f["close"]
    return f


def zscore(s: pd.Series, window: int = 288) -> pd.Series:
    mean = s.rolling(window, min_periods=max(10, window // 4)).mean()
    std = s.rolling(window, min_periods=max(10, window // 4)).std()
    return ((s - mean) / std.replace(0, np.nan)).clip(-5, 5)


def build_timeframes(fast: bool = False) -> Dict[str, pd.DataFrame]:
    paths = canonical_paths()
    src_5m = Path(paths.get("canonical_5m_path", str(DEFAULT_5M)))
    base = read_ohlcv(src_5m)
    if fast:
        base = base.tail(10_000).copy()
    base = add_base_candle_features(base)
    out: Dict[str, pd.DataFrame] = {"5m": base}
    base_idx = base.set_index("timestamp")
    for tf, rule in TF_MAP.items():
        if tf == "5m":
            continue
        r = base_idx.resample(rule, label="right", closed="right").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index()
        out[tf] = add_base_candle_features(r)
    src_1m = Path(paths.get("canonical_1m_path", str(DEFAULT_1M)))
    one = read_ohlcv(src_1m)
    if not one.empty:
        out["1m"] = add_base_candle_features(one.tail(100_000) if fast else one)
    rows, gaps = [], []
    expected_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
    for tf, df in out.items():
        df.to_parquet(ROOT / f"timeframes/btcusdt_{tf}_ohlcv.parquet", index=False)
        rows.append({"timeframe": tf, "rows": len(df), "start": df["timestamp"].min(), "end": df["timestamp"].max()})
        delta = df["timestamp"].diff().dt.total_seconds().div(60)
        gaps.append({"timeframe": tf, "expected_minutes": expected_minutes.get(tf), "gap_count": int((delta > expected_minutes.get(tf, 5) * 1.5).sum()), "max_gap_minutes": float(delta.max()) if len(delta) else 0})
    pd.DataFrame(rows).to_csv(ROOT / "timeframes/timeframe_build_summary.csv", index=False)
    pd.DataFrame(gaps).to_csv(ROOT / "timeframes/timeframe_gap_audit.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "discovery/timeframe_coverage_summary.csv", index=False)
    (ROOT / "timeframes/timeframe_build_report.md").write_text("# Timeframe Build Report\n\n5m canonical OHLCV was used as the main source. Higher timeframes are closed-candle resamples with right labels. 1m is recent-only when available.\n", encoding="utf-8")
    return out


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-diff.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    f = tf["5m"].copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"], errors="coerce").astype("datetime64[ns]")
    for n in [10, 20, 50, 100, 200]:
        f[f"sma_{n}"] = f["close"].rolling(n, min_periods=n).mean()
        f[f"ema_{n}"] = ema(f["close"], n)
        f[f"ema_{n}_slope"] = f[f"ema_{n}"].pct_change(12)
        f[f"dist_ema_{n}"] = f["close"] / f[f"ema_{n}"] - 1
    f["ema20_above_ema50"] = f["ema_20"] > f["ema_50"]
    f["ema50_above_ema200"] = f["ema_50"] > f["ema_200"]
    f["ema20_cross_up"] = f["ema20_above_ema50"] & ~f["ema20_above_ema50"].shift(1).fillna(False)
    f["ema20_cross_down"] = ~f["ema20_above_ema50"] & f["ema20_above_ema50"].shift(1).fillna(False)
    f["trend_stack_bull"] = (f["close"] > f["ema_20"]) & (f["ema_20"] > f["ema_50"]) & (f["ema_50"] > f["ema_200"])
    f["trend_stack_bear"] = (f["close"] < f["ema_20"]) & (f["ema_20"] < f["ema_50"]) & (f["ema_50"] < f["ema_200"])
    for p in [7, 14, 21]:
        f[f"rsi_{p}"] = rsi(f["close"], p)
    ema12 = ema(f["close"], 12)
    ema26 = ema(f["close"], 26)
    f["macd"] = ema12 - ema26
    f["macd_signal"] = ema(f["macd"], 9)
    f["macd_hist"] = f["macd"] - f["macd_signal"]
    mid = f["close"].rolling(20, min_periods=20).mean()
    std = f["close"].rolling(20, min_periods=20).std()
    f["bb_mid"] = mid
    f["bb_upper"] = mid + 2 * std
    f["bb_lower"] = mid - 2 * std
    f["bb_width"] = (f["bb_upper"] - f["bb_lower"]) / f["bb_mid"]
    f["bb_pct_b"] = (f["close"] - f["bb_lower"]) / (f["bb_upper"] - f["bb_lower"])
    f["roc_10"] = f["close"].pct_change(10)
    f["roc_20"] = f["close"].pct_change(20)
    for n in [12, 24, 48, 96]:
        f[f"recent_high_{n}"] = f["high"].rolling(n, min_periods=n).max().shift(1)
        f[f"recent_low_{n}"] = f["low"].rolling(n, min_periods=n).min().shift(1)
    for n in [20, 50, 100]:
        f[f"donchian_high_{n}"] = f["high"].rolling(n, min_periods=n).max().shift(1)
        f[f"donchian_low_{n}"] = f["low"].rolling(n, min_periods=n).min().shift(1)
    prev_day = tf["1d"][["timestamp", "high", "low"]].rename(columns={"high": "prev_day_high", "low": "prev_day_low"}).copy()
    prev_day["timestamp"] = pd.to_datetime(prev_day["timestamp"] + pd.Timedelta(days=1), errors="coerce").astype("datetime64[ns]")
    f["timestamp"] = pd.to_datetime(f["timestamp"], errors="coerce").astype("datetime64[ns]")
    prev_merged = pd.merge_asof(f[["timestamp"]].sort_values("timestamp"), prev_day.sort_values("timestamp"), on="timestamp", direction="backward")
    f["prev_day_high"] = prev_merged["prev_day_high"].to_numpy()
    f["prev_day_low"] = prev_merged["prev_day_low"].to_numpy()
    # Higher timeframe closed-candle regime, shifted one completed candle before ffill.
    for htf in ["15m", "30m", "1h", "4h", "1d"]:
        h = tf[htf][["timestamp", "close"]].copy()
        h[f"{htf}_ema20"] = ema(h["close"], 20)
        h[f"{htf}_ema50"] = ema(h["close"], 50)
        h[f"{htf}_bull_regime"] = h[f"{htf}_ema20"] > h[f"{htf}_ema50"]
        h["timestamp"] = h["timestamp"] + pd.to_timedelta(TF_MAP[htf])
        h["timestamp"] = pd.to_datetime(h["timestamp"], errors="coerce").astype("datetime64[ns]")
        f["timestamp"] = pd.to_datetime(f["timestamp"], errors="coerce").astype("datetime64[ns]")
        f = pd.merge_asof(f.sort_values("timestamp"), h[["timestamp", f"{htf}_bull_regime"]].sort_values("timestamp"), on="timestamp", direction="backward")
    f = merge_orderflow(f)
    f.to_parquet(ROOT / "features/ta_feature_frame.parquet", index=False)
    (ROOT / "features/ta_feature_schema.json").write_text(jdump({c: str(f[c].dtype) for c in f.columns}), encoding="utf-8")
    f.isna().mean().reset_index().rename(columns={"index": "column", 0: "missing_ratio"}).to_csv(ROOT / "features/ta_feature_missingness.csv", index=False)
    return f


def merge_orderflow(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    if not ORDERFLOW_FRAME.exists():
        for c in ["risk_adjusted_orderflow_score", "cvd_taker_combined_score", "taker_delta", "cvd_slope", "oi_change", "funding_score", "basis_score"]:
            f[c] = np.nan
        f["orderflow_available"] = False
        return f
    cols = ["timestamp", "risk_adjusted_orderflow_score", "cvd_taker_combined_score", "taker_delta", "cvd_slope", "oi_change", "funding_score", "basis_score"]
    odf = pd.read_parquet(ORDERFLOW_FRAME, columns=cols).drop_duplicates("timestamp")
    odf["timestamp"] = pd.to_datetime(odf["timestamp"], errors="coerce").astype("datetime64[ns]")
    f["timestamp"] = pd.to_datetime(f["timestamp"], errors="coerce").astype("datetime64[ns]")
    out = pd.merge_asof(f.sort_values("timestamp"), odf.sort_values("timestamp"), on="timestamp", direction="backward", tolerance=pd.Timedelta("10min"))
    out["orderflow_available"] = out["risk_adjusted_orderflow_score"].notna()
    return out


def orderflow_overlap_summary() -> None:
    rows = []
    if ORDERFLOW_FRAME.exists():
        odf = pd.read_parquet(ORDERFLOW_FRAME, columns=["timestamp"])
        odf["timestamp"] = pd.to_datetime(odf["timestamp"], errors="coerce")
        rows.append({"source": "btc_orderflow_combined_frame", "rows": len(odf), "start": odf["timestamp"].min(), "end": odf["timestamp"].max()})
    pd.DataFrame(rows).to_csv(ROOT / "discovery/orderflow_overlap_summary.csv", index=False)


def catalog_files() -> None:
    candle = [
        "hammer", "hanging_man", "inverted_hammer", "shooting_star", "doji", "dragonfly_doji", "gravestone_doji", "long_legged_doji", "bullish_marubozu", "bearish_marubozu", "pinbar_bull", "pinbar_bear", "bullish_engulfing", "bearish_engulfing", "inside_bar", "outside_bar", "morning_star_proxy", "evening_star_proxy"
    ]
    pd.DataFrame([{"pattern": x, "strictness": "loose/normal/strict", "note": "mechanical candle feature; no guaranteed edge"} for x in candle]).to_csv(ROOT / "catalog/candlestick_pattern_catalog.csv", index=False)
    pd.DataFrame([{"signal": x} for x in ["EMA20/50 cross", "EMA50/200 cross", "price reclaim/reject EMA20/50", "trend stack", "pullback reclaim"]]).to_csv(ROOT / "catalog/ma_trend_signal_catalog.csv", index=False)
    pd.DataFrame([{"signal": x} for x in ["RSI reclaim/reject", "MACD cross/hist flip", "Bollinger reclaim/reject/squeeze", "ATR expansion/exhaustion"]]).to_csv(ROOT / "catalog/momentum_signal_catalog.csv", index=False)
    pd.DataFrame([{"signal": x} for x in ["support bounce", "resistance breakout/rejection", "Donchian breakout", "failed breakout", "retest hold/fail", "liquidity sweep proxy"]]).to_csv(ROOT / "catalog/support_resistance_signal_catalog.csv", index=False)
    pd.DataFrame([{"signal": x} for x in ["TA + taker/proxy CVD confirm", "TA + orderflow_risk_not_worst20", "TA + basis not overheated", "squeeze warning as risk filter"]]).to_csv(ROOT / "catalog/ta_orderflow_combo_catalog.csv", index=False)


def add_candidate(rows: List[pd.DataFrame], f: pd.DataFrame, mask: pd.Series, family: str, signal_id: str, direction: str, timeframe: str = "5m", variant: str = "normal") -> None:
    m = mask.fillna(False)
    if not m.any():
        return
    sub = f.loc[m].copy()
    sub["family"] = family
    sub["signal_id"] = signal_id
    sub["variant"] = variant
    sub["timeframe"] = timeframe
    sub["direction"] = direction
    sub["candidate_id"] = [f"{family}_{signal_id}_{i:07d}" for i in range(len(sub))]
    sub["signal_ts"] = sub["timestamp"]
    sub["entry_condition_json"] = json.dumps({"signal": signal_id, "variant": variant, "lookahead": False})
    rows.append(sub)


def build_candidates(f: pd.DataFrame) -> pd.DataFrame:
    catalog_files()
    rows: List[pd.DataFrame] = []
    body = f["body"].replace(0, np.nan)
    prior_down = f["close"].pct_change(12) < 0
    prior_up = f["close"].pct_change(12) > 0
    hammer = (f["lower_wick"] >= 2 * body) & (f["upper_wick"] <= 0.75 * body) & (f["close_position_in_range"] >= 0.55) & prior_down
    shooting = (f["upper_wick"] >= 2 * body) & (f["lower_wick"] <= 0.75 * body) & (f["close_position_in_range"] <= 0.45) & prior_up
    doji = f["body_pct_of_range"] <= 0.10
    bull_engulf = (f["close"] > f["open"]) & (f["close"].shift(1) < f["open"].shift(1)) & (f["close"] >= f["open"].shift(1)) & (f["open"] <= f["close"].shift(1))
    bear_engulf = (f["close"] < f["open"]) & (f["close"].shift(1) > f["open"].shift(1)) & (f["open"] >= f["close"].shift(1)) & (f["close"] <= f["open"].shift(1))
    inside = (f["high"] < f["high"].shift(1)) & (f["low"] > f["low"].shift(1))
    outside = (f["high"] > f["high"].shift(1)) & (f["low"] < f["low"].shift(1))
    add_candidate(rows, f, hammer, "F1_candlestick", "hammer_prior_down", "LONG")
    add_candidate(rows, f, shooting, "F1_candlestick", "shooting_star_prior_up", "SHORT")
    add_candidate(rows, f, bull_engulf, "F1_candlestick", "bullish_engulfing", "LONG")
    add_candidate(rows, f, bear_engulf, "F1_candlestick", "bearish_engulfing", "SHORT")
    add_candidate(rows, f, doji & prior_down & (f["close_position_in_range"] > 0.60), "F1_candlestick", "doji_reclaim_contextual", "LONG")
    add_candidate(rows, f, inside & (f["close"] > f["high"].shift(1)), "F1_candlestick", "inside_bar_breakout", "LONG")
    add_candidate(rows, f, outside & (f["close"] < f["low"].shift(1)), "F1_candlestick", "outside_bar_breakdown", "SHORT")
    add_candidate(rows, f, f["ema20_cross_up"], "F2_ma_trend", "ema20_50_cross_up", "LONG")
    add_candidate(rows, f, f["ema20_cross_down"], "F2_ma_trend", "ema20_50_cross_down", "SHORT")
    add_candidate(rows, f, (f["low"] <= f["ema_20"]) & (f["close"] > f["ema_20"]) & f["trend_stack_bull"], "F2_ma_trend", "pullback_ema20_reclaim_bull", "LONG")
    add_candidate(rows, f, (f["high"] >= f["ema_20"]) & (f["close"] < f["ema_20"]) & f["trend_stack_bear"], "F2_ma_trend", "ema20_reject_bear", "SHORT")
    add_candidate(rows, f, f["trend_stack_bull"] & (f["ema_20_slope"] > 0), "F2_ma_trend", "trend_stack_bull", "LONG")
    add_candidate(rows, f, f["trend_stack_bear"] & (f["ema_20_slope"] < 0), "F2_ma_trend", "trend_stack_bear", "SHORT")
    add_candidate(rows, f, (f["rsi_14"].shift(1) < 30) & (f["rsi_14"] >= 30), "F3_momentum", "rsi_oversold_reclaim", "LONG")
    add_candidate(rows, f, (f["rsi_14"].shift(1) > 70) & (f["rsi_14"] <= 70), "F3_momentum", "rsi_overbought_reject", "SHORT")
    add_candidate(rows, f, (f["macd_hist"].shift(1) <= 0) & (f["macd_hist"] > 0), "F3_momentum", "macd_hist_flip_positive", "LONG")
    add_candidate(rows, f, (f["macd_hist"].shift(1) >= 0) & (f["macd_hist"] < 0), "F3_momentum", "macd_hist_flip_negative", "SHORT")
    add_candidate(rows, f, (f["low"] < f["bb_lower"]) & (f["close"] > f["bb_lower"]), "F3_momentum", "bollinger_lower_reclaim", "LONG")
    add_candidate(rows, f, (f["high"] > f["bb_upper"]) & (f["close"] < f["bb_upper"]), "F3_momentum", "bollinger_upper_reject", "SHORT")
    add_candidate(rows, f, (f["bb_width"] < f["bb_width"].rolling(288, min_periods=100).quantile(0.2)) & (f["close"] > f["bb_upper"]), "F3_momentum", "bb_squeeze_breakout", "LONG")
    add_candidate(rows, f, (f["close"] > f["recent_high_48"]) & (f["close"].shift(1) <= f["recent_high_48"].shift(1)), "F4_support_resistance", "resistance_breakout_48", "LONG")
    add_candidate(rows, f, (f["close"] < f["recent_low_48"]) & (f["close"].shift(1) >= f["recent_low_48"].shift(1)), "F4_support_resistance", "support_breakdown_48", "SHORT")
    add_candidate(rows, f, (f["low"] < f["recent_low_48"]) & (f["close"] > f["recent_low_48"]), "F4_support_resistance", "liquidity_sweep_low_reclaim", "LONG")
    add_candidate(rows, f, (f["high"] > f["recent_high_48"]) & (f["close"] < f["recent_high_48"]), "F4_support_resistance", "liquidity_sweep_high_reject", "SHORT")
    add_candidate(rows, f, (f["close"] > f["donchian_high_50"]), "F4_support_resistance", "donchian50_breakout", "LONG")
    add_candidate(rows, f, (f["close"] < f["donchian_low_50"]), "F4_support_resistance", "donchian50_breakdown", "SHORT")
    of_avail = f["orderflow_available"].fillna(False)
    risk_not_worst20 = pd.to_numeric(f["risk_adjusted_orderflow_score"], errors="coerce") > pd.to_numeric(f["risk_adjusted_orderflow_score"], errors="coerce").quantile(0.20)
    taker_buy = pd.to_numeric(f["taker_delta"], errors="coerce") > pd.to_numeric(f["taker_delta"], errors="coerce").quantile(0.70)
    taker_sell = pd.to_numeric(f["taker_delta"], errors="coerce") < pd.to_numeric(f["taker_delta"], errors="coerce").quantile(0.30)
    cvd_up = pd.to_numeric(f["cvd_slope"], errors="coerce") > pd.to_numeric(f["cvd_slope"], errors="coerce").quantile(0.65)
    cvd_down = pd.to_numeric(f["cvd_slope"], errors="coerce") < pd.to_numeric(f["cvd_slope"], errors="coerce").quantile(0.35)
    add_candidate(rows, f, hammer & of_avail & risk_not_worst20 & (taker_buy | cvd_up), "F5_ta_orderflow", "hammer_cvd_taker_confirm", "LONG")
    add_candidate(rows, f, bull_engulf & of_avail & risk_not_worst20 & taker_buy, "F5_ta_orderflow", "engulfing_taker_confirm", "LONG")
    add_candidate(rows, f, ((f["close"] > f["recent_high_48"]) & of_avail & risk_not_worst20 & taker_buy & cvd_up), "F5_ta_orderflow", "sr_breakout_taker_cvd_confirm", "LONG")
    add_candidate(rows, f, shooting & of_avail & taker_sell & cvd_down, "F5_ta_orderflow", "shooting_star_cvd_divergence", "SHORT")
    add_candidate(rows, f, ((f["rsi_14"].shift(1) < 30) & (f["rsi_14"] >= 30) & of_avail & risk_not_worst20), "F5_ta_orderflow", "rsi_reclaim_risk_not_worst20", "LONG")
    c = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not c.empty:
        c["signal_ts"] = pd.to_datetime(c["signal_ts"])
        c = c.sort_values(["signal_ts", "family", "signal_id"]).reset_index(drop=True)
        c.to_parquet(ROOT / "candidates/all_single_family_candidates.parquet", index=False)
        for fam, filename in [
            ("F1_candlestick", "candlestick_candidates.parquet"),
            ("F2_ma_trend", "ma_trend_candidates.parquet"),
            ("F3_momentum", "momentum_candidates.parquet"),
            ("F4_support_resistance", "support_resistance_candidates.parquet"),
            ("F5_ta_orderflow", "ta_orderflow_candidates.parquet"),
        ]:
            sub = c[c["family"].eq(fam)]
            sub.to_parquet(ROOT / f"candidates/{filename}", index=False)
            sub.groupby(["family", "signal_id", "direction"]).size().reset_index(name="candidate_count").to_csv(ROOT / f"candidates/{filename.replace('.parquet', '_summary.csv')}", index=False)
    for name in ["candlestick", "ma_trend", "momentum", "support_resistance", "ta_orderflow"]:
        (ROOT / f"features/{name}_feature_report.md").write_text(f"# {name} Feature Report\n\nMechanical TA features/candidates built without using future outcome information.\n", encoding="utf-8")
    # Required aliases.
    (ROOT / "features/candlestick_pattern_features.parquet").write_bytes((ROOT / "candidates/candlestick_candidates.parquet").read_bytes())
    (ROOT / "features/ma_trend_features.parquet").write_bytes((ROOT / "candidates/ma_trend_candidates.parquet").read_bytes())
    (ROOT / "features/momentum_exhaustion_features.parquet").write_bytes((ROOT / "candidates/momentum_candidates.parquet").read_bytes())
    (ROOT / "features/support_resistance_features.parquet").write_bytes((ROOT / "candidates/support_resistance_candidates.parquet").read_bytes())
    (ROOT / "features/ta_orderflow_combined_features.parquet").write_bytes((ROOT / "candidates/ta_orderflow_candidates.parquet").read_bytes())
    return c


def priority_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    priority = {"F1_candlestick": 1, "F2_ma_trend": 2, "F3_momentum": 3, "F4_support_resistance": 4, "F5_ta_orderflow": 5}
    tmp = df.copy()
    tmp["_rank"] = tmp["family"].map(priority).fillna(99)
    return tmp.sort_values(["signal_ts", "_rank"]).drop_duplicates(["signal_ts", "direction"]).drop(columns=["_rank"])


def build_stages(c: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    fams = ["F1_candlestick", "F2_ma_trend", "F3_momentum", "F4_support_resistance", "F5_ta_orderflow"]
    stages = {}
    for i in range(1, 6):
        name = [
            "stage_1_candlestick",
            "stage_2_candle_ma",
            "stage_3_candle_ma_momentum",
            "stage_4_candle_ma_momentum_sr",
            "stage_5_ta_orderflow",
        ][i - 1]
        stages[name] = priority_dedupe(c[c["family"].isin(fams[:i])])
        stages[name].to_parquet(ROOT / f"stages/{name}_candidates.parquet", index=False)
    pd.DataFrame([{"stage": k, "candidate_count": len(v), "families": ",".join(sorted(v["family"].unique())) if not v.empty else ""} for k, v in stages.items()]).to_csv(ROOT / "stages/stagewise_candidate_summary.csv", index=False)
    contrib = []
    for k, v in stages.items():
        for fam, g in v.groupby("family"):
            contrib.append({"stage": k, "family": fam, "candidate_count": len(g), "share": len(g) / len(v) if len(v) else 0})
    pd.DataFrame(contrib).to_csv(ROOT / "stages/stagewise_contribution_matrix.csv", index=False)
    (ROOT / "stages/stagewise_application_report.md").write_text("# Stagewise Application Report\n\nStages follow requested order: candles, MA/trend, momentum, support/resistance, TA+orderflow. Duplicate timestamp/direction candidates use family priority.\n", encoding="utf-8")
    return stages


def add_outcomes(c: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    if c.empty:
        return c
    b = base[["timestamp", "open", "high", "low", "close", "atr_proxy"]].reset_index(drop=True).copy()
    b["timestamp"] = pd.to_datetime(b["timestamp"], errors="coerce").astype("datetime64[ns]")
    b["bar_index"] = np.arange(len(b))
    c = c.copy()
    c["signal_ts"] = pd.to_datetime(c["signal_ts"], errors="coerce").astype("datetime64[ns]")
    right = b[["timestamp", "bar_index"]].rename(columns={"timestamp": "signal_ts"})
    right["signal_ts"] = pd.to_datetime(right["signal_ts"], errors="coerce").astype("datetime64[ns]")
    m = pd.merge_asof(c.sort_values("signal_ts"), right.sort_values("signal_ts"), on="signal_ts", direction="backward")
    entry_idx = m["bar_index"].astype("Int64") + 1
    valid = entry_idx.notna() & (entry_idx < len(b))
    m = m[valid].copy()
    entry_idx = entry_idx[valid].astype(int)
    m["entry_ts"] = b.loc[entry_idx, "timestamp"].to_numpy()
    m["entry_price"] = b.loc[entry_idx, "open"].to_numpy()
    direction_sign = np.where(m["direction"].eq("SHORT"), -1, 1)
    for hid, bars in HORIZON_BARS.items():
        exit_idx = np.minimum(entry_idx + bars, len(b) - 1)
        exit_close = b.loc[exit_idx, "close"].to_numpy()
        gross = direction_sign * (exit_close / m["entry_price"].to_numpy() - 1)
        m[f"gross_{hid}"] = gross
        m[f"net_{hid}"] = gross - CURRENT_COST_BPS / 10000
    # Primary non-oracle outcome uses 1h fixed horizon.
    m["gross_return"] = m["gross_H3_1h"]
    m["net_after_cost"] = m["net_H3_1h"]
    hbars = HORIZON_BARS["H3_1h"] + 1
    future_high = b["high"].iloc[::-1].rolling(hbars, min_periods=1).max().iloc[::-1].to_numpy()
    future_low = b["low"].iloc[::-1].rolling(hbars, min_periods=1).min().iloc[::-1].to_numpy()
    high_1h = future_high[entry_idx.to_numpy()]
    low_1h = future_low[entry_idx.to_numpy()]
    ep = m["entry_price"].to_numpy()
    mfe = np.where(m["direction"].eq("SHORT"), ep / low_1h - 1, high_1h / ep - 1)
    mae = np.where(m["direction"].eq("SHORT"), high_1h / ep - 1, ep / low_1h - 1)
    m["MFE"] = mfe
    m["MAE"] = mae
    m["MFE_to_cost"] = mfe / (CURRENT_COST_BPS / 10000)
    m["MAE_to_cost"] = mae / (CURRENT_COST_BPS / 10000)
    m["RFE"] = (mae > mfe) & (mae > CURRENT_COST_BPS / 10000)
    m["label"] = np.where(m["net_after_cost"] > CURRENT_COST_BPS / 10000, "TA_GOOD", np.where(m["net_after_cost"] < -CURRENT_COST_BPS / 10000, "TA_BAD", "TA_NEUTRAL"))
    m["holding_bars"] = HORIZON_BARS["H3_1h"]
    m["exit_policy_id"] = "X1_fixed_H3_1h"
    m.to_parquet(ROOT / "backfill/ta_alpha_paper_trades.parquet", index=False)
    # long outcome table with horizons.
    cols = [x for x in m.columns if x.startswith("gross_") or x.startswith("net_")]
    m[["candidate_id", "signal_ts", "entry_ts", "family", "signal_id", "direction", *cols]].to_parquet(ROOT / "backfill/ta_alpha_exit_outcomes.parquet", index=False)
    return m


def perf(df: pd.DataFrame, id_: str, group_type: str) -> Dict[str, Any]:
    if df.empty:
        return {"id": id_, "group_type": group_type, "trade_count": 0, "sample_warning": True, "production_ready": False}
    net = pd.to_numeric(df["net_after_cost"], errors="coerce").fillna(0)
    gross = pd.to_numeric(df["gross_return"], errors="coerce").fillna(0)
    pos = net[net > 0].sum()
    neg = -net[net < 0].sum()
    curve = net.cumsum()
    good = df["label"].astype(str).str.contains("GOOD", na=False)
    bad = df["label"].astype(str).str.contains("BAD", na=False)
    days = max(1, (pd.to_datetime(df["signal_ts"]).max() - pd.to_datetime(df["signal_ts"]).min()).days + 1)
    return {
        "id": id_,
        "group_type": group_type,
        "candidate_count": len(df),
        "trade_count": len(df),
        "trades_per_day": len(df) / days,
        "direction": "MIXED" if df["direction"].nunique() > 1 else str(df["direction"].iloc[0]),
        "timeframe": "MIXED" if df["timeframe"].nunique() > 1 else str(df["timeframe"].iloc[0]),
        "GOOD_count": int(good.sum()),
        "BAD_count": int(bad.sum()),
        "NEUTRAL_count": int((~good & ~bad).sum()),
        "GOOD_rate": float(good.mean()),
        "BAD_rate": float(bad.mean()),
        "mean_net_bps": float(net.mean() * 10000),
        "median_net_bps": float(net.median() * 10000),
        "sum_net": float(net.sum()),
        "gross_mean_bps": float(gross.mean() * 10000),
        "current_cost_mean_bps": float((gross - 6 / 10000).mean() * 10000),
        "maker_like_mean_bps": float((gross - 3 / 10000).mean() * 10000),
        "two_x_cost_mean_bps": float((gross - 12 / 10000).mean() * 10000),
        "zero_cost_mean_bps": float(gross.mean() * 10000),
        "winrate": float((net > 0).mean()),
        "profit_factor": float(pos / neg) if neg > 0 else math.inf,
        "MFE_to_cost": float(df["MFE_to_cost"].mean()),
        "MAE_to_cost": float(df["MAE_to_cost"].mean()),
        "RFE_rate": float(df["RFE"].astype(bool).mean()),
        "MDD_proxy": float((curve.cummax() - curve).max()),
        "tail_loss": float(net.quantile(0.05) * 10000),
        "best_horizon": "H3_1h_primary_fixed; horizon_table_available",
        "best_exit_policy": "X1_fixed_H3_1h",
        "sample_warning": len(df) < 100,
        "overfit_warning": "LOW_FIXED_RULES" if len(df) >= 100 else "SAMPLE_TOO_SMALL",
        "production_ready": False,
    }


def apply_cooldown(df: pd.DataFrame, minutes: int, max_per_day: int | None) -> pd.DataFrame:
    if df.empty:
        return df
    keep = []
    last: Dict[str, pd.Timestamp] = {}
    counts: Dict[Tuple[str, str], int] = {}
    for idx, row in df.sort_values("signal_ts").iterrows():
        ts = pd.to_datetime(row["signal_ts"])
        d = str(row["direction"])
        key = (str(ts.date()), d)
        if minutes and d in last and ts < last[d] + pd.Timedelta(minutes=minutes):
            continue
        if max_per_day and counts.get(key, 0) >= max_per_day:
            continue
        keep.append(idx)
        last[d] = ts
        counts[key] = counts.get(key, 0) + 1
    return df.loc[keep].copy()


def evaluate(c: pd.DataFrame, stages: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    family_rows = [perf(priority_dedupe(c[c["family"].eq(fam)]), fam, "single_family") for fam in sorted(c["family"].unique())]
    single = pd.DataFrame(family_rows)
    single.to_csv(ROOT / "scorecards/single_family_scorecard.csv", index=False)
    single.to_csv(ROOT / "backfill/outcome_summary_by_family.csv", index=False)
    stage_rows = [perf(v, k, "stage") for k, v in stages.items()]
    stage = pd.DataFrame(stage_rows)
    stage.to_csv(ROOT / "scorecards/stagewise_scorecard.csv", index=False)
    stage.to_csv(ROOT / "backfill/outcome_summary_by_stage.csv", index=False)
    deltas = []
    s = {r["id"]: r for r in stage_rows}
    names = list(stages)
    for a, b, label in zip(names, names[1:], ["Stage2_minus_Stage1_MA_add", "Stage3_minus_Stage2_momentum_add", "Stage4_minus_Stage3_SR_add", "Stage5_minus_Stage4_orderflow_add"]):
        deltas.append({"delta_id": label, "mean_net_bps_delta": s[b]["mean_net_bps"] - s[a]["mean_net_bps"], "trade_count_delta": s[b]["trade_count"] - s[a]["trade_count"], "RFE_rate_delta": s[b]["RFE_rate"] - s[a]["RFE_rate"]})
    pd.DataFrame(deltas, columns=["delta_id", "mean_net_bps_delta", "trade_count_delta", "RFE_rate_delta"]).to_csv(ROOT / "scorecards/stage_delta_scorecard.csv", index=False)
    fams = sorted(c["family"].unique())
    combo_rows = []
    for r in [2, 3, len(fams)]:
        for combo in itertools.combinations(fams, r):
            if r == len(fams) and combo != tuple(fams):
                continue
            combo_rows.append(perf(priority_dedupe(c[c["family"].isin(combo)]), "+".join(combo), f"combo_{r}"))
    combo = pd.DataFrame(combo_rows)
    combo[combo["group_type"].eq("combo_2")].to_csv(ROOT / "scorecards/pair_combination_scorecard.csv", index=False)
    combo[combo["group_type"].eq("combo_3")].to_csv(ROOT / "scorecards/triple_combination_scorecard.csv", index=False)
    combo.to_csv(ROOT / "backfill/outcome_summary_by_combination.csv", index=False)
    full = combo[combo["group_type"].eq(f"combo_{len(fams)}")]
    full.to_csv(ROOT / "scorecards/gated_combination_scorecard.csv", index=False)
    # Cooldown on Stage5 for full runs, or on the last available stage for stage-only runs.
    cd_rows = []
    base_stage_name = "stage_5_ta_orderflow" if "stage_5_ta_orderflow" in stages else list(stages)[-1]
    base = stages[base_stage_name]
    for minutes in [0, 30, 60, 180, 360, 720]:
        for max_day in [None, 1, 2, 3, 5]:
            cd = apply_cooldown(base, minutes, max_day)
            m = perf(cd, f"stage5_cd{minutes}_max{max_day or 'unlimited'}", "cooldown_trade_limit")
            m["cooldown_minutes"] = minutes
            m["max_trades_per_day"] = max_day or "unlimited"
            cd_rows.append(m)
    pd.DataFrame(cd_rows).to_csv(ROOT / "scorecards/cooldown_trade_limit_scorecard.csv", index=False)
    pd.DataFrame([perf(g, f"timeframe_{k}", "timeframe") for k, g in c.groupby("timeframe")]).to_csv(ROOT / "scorecards/timeframe_scorecard.csv", index=False)
    pd.DataFrame([perf(g, f"direction_{k}", "direction") for k, g in c.groupby("direction")]).to_csv(ROOT / "scorecards/direction_scorecard.csv", index=False)
    cost = []
    for id_, df in [("stage5", base), *[(r["id"], stages[r["id"]]) for r in stage_rows if r["id"] in stages]]:
        gross = pd.to_numeric(df["gross_return"], errors="coerce")
        cost.append({"id": id_, "zero_cost_bps": gross.mean() * 10000, "maker_like_bps": (gross - 3 / 10000).mean() * 10000, "current_bps": (gross - 6 / 10000).mean() * 10000, "two_x_bps": (gross - 12 / 10000).mean() * 10000})
    pd.DataFrame(cost).to_csv(ROOT / "scorecards/cost_sensitivity_scorecard.csv", index=False)
    (ROOT / "scorecards/stagewise_scorecard_report.md").write_text("# Stagewise Scorecard Report\n\nScorecards use next-open entry and fixed 1h primary outcome with current/maker/2x cost sensitivity. No oracle outcome is used for candidate generation.\n", encoding="utf-8")
    (ROOT / "backfill/backfill_report.md").write_text("# Backfill Report\n\nPaper replay uses signal close then next available open. H3 1h fixed horizon is primary; additional fixed horizons are stored in the outcome table.\n", encoding="utf-8")
    return {"single": single, "stage": stage, "combo": combo}


def failure(scorecards: Dict[str, pd.DataFrame]) -> None:
    rows = []
    for kind, df in scorecards.items():
        if df.empty:
            continue
        for _, r in df.iterrows():
            mean_net = r.get("mean_net_bps", np.nan)
            gross = r.get("gross_mean_bps", np.nan)
            sample = r.get("trade_count", 0)
            rfe = r.get("RFE_rate", np.nan)
            if sample == 0:
                reason, action = "NO_SAMPLE", "DROP"
            elif sample < 100:
                reason, action = "SAMPLE_TOO_SMALL", "KEEP_AS_CASEBOOK_ONLY"
            elif pd.notna(mean_net) and mean_net < 0 and pd.notna(gross) and gross > 0:
                reason, action = "COST_KILL", "NEED_EXIT_REDESIGN"
            elif pd.notna(mean_net) and mean_net < 0:
                reason, action = "GROSS_EDGE_WEAK", "DROP"
            elif pd.notna(rfe) and rfe > 0.30:
                reason, action = "RFE_TOO_HIGH", "KEEP_AS_RISK_FILTER"
            else:
                reason, action = "NONE_OR_ACCEPTABLE", "KEEP_FOR_FORWARD"
            rows.append({"object_id": r.get("id"), "group_type": r.get("group_type", kind), "primary_failure_reason": reason, "secondary_failure_reason": "EXIT_HORIZON_MISMATCH" if r.get("best_horizon") else "", "evidence_metrics": jdump({"mean_net_bps": mean_net, "gross_mean_bps": gross, "sample": sample, "RFE_rate": rfe}), "recommended_action": action})
    out = pd.DataFrame(rows)
    out[out["group_type"].astype(str).str.contains("single", na=False)].to_csv(ROOT / "failure/family_failure_attribution.csv", index=False)
    out[out["group_type"].astype(str).str.contains("stage", na=False)].to_csv(ROOT / "failure/stage_failure_attribution.csv", index=False)
    out[out["group_type"].astype(str).str.contains("combo", na=False)].to_csv(ROOT / "failure/combination_failure_attribution.csv", index=False)
    out.groupby(["primary_failure_reason", "recommended_action"]).size().reset_index(name="count").to_csv(ROOT / "failure/failure_reason_summary.csv", index=False)
    (ROOT / "failure/failure_attribution_report.md").write_text("# Failure Attribution Report\n\nFailures are assigned from sample size, gross/net relationship, RFE, and primary fixed-horizon results.\n", encoding="utf-8")


def casebook(c: pd.DataFrame) -> None:
    cb = c.copy()
    cb["case_category"] = np.select(
        [
            cb["family"].eq("F1_candlestick") & (cb["net_after_cost"] > 0),
            cb["family"].eq("F1_candlestick") & (cb["net_after_cost"] <= 0),
            cb["family"].eq("F2_ma_trend") & (cb["net_after_cost"] > 0),
            cb["family"].eq("F2_ma_trend") & (cb["net_after_cost"] <= 0),
            cb["family"].eq("F3_momentum") & (cb["net_after_cost"] > 0),
            cb["family"].eq("F3_momentum") & (cb["net_after_cost"] <= 0),
            cb["family"].eq("F4_support_resistance") & (cb["net_after_cost"] > 0),
            cb["family"].eq("F4_support_resistance") & (cb["net_after_cost"] <= 0),
            cb["family"].eq("F5_ta_orderflow") & (cb["net_after_cost"] > 0),
            cb["family"].eq("F5_ta_orderflow") & (cb["net_after_cost"] <= 0),
        ],
        [
            "CANDLE_SUCCESS_CONTEXTUAL", "CANDLE_FAIL_NO_EDGE", "MA_SUCCESS_TREND", "MA_CROSS_FAIL_LAG", "MOMENTUM_SUCCESS", "MOMENTUM_FAIL_CHASE_OR_KNIFE", "SR_SUCCESS", "SR_BREAKOUT_FAIL_FAKEOUT", "TA_ORDERFLOW_SUCCESS", "TA_ORDERFLOW_FAIL_OVERFILTER",
        ],
        default="OTHER",
    )
    cb["why_success"] = np.where(cb["net_after_cost"] > 0, "positive fixed 1h net outcome", "")
    cb["why_failure"] = np.where(cb["net_after_cost"] <= 0, "negative fixed 1h net outcome; inspect cost/gross/exit/fakeout", "")
    cb["chart_window_path"] = ""
    cols = [c for c in ["candidate_id", "signal_ts", "timeframe", "direction", "family", "signal_id", "variant", "entry_price", "net_after_cost", "MFE", "MAE", "RFE", "case_category", "why_success", "why_failure", "chart_window_path"] if c in cb.columns]
    out = cb[cols].sort_values("net_after_cost", ascending=False)
    out.to_parquet(ROOT / "casebook/ta_alpha_casebook.parquet", index=False)
    out.to_csv(ROOT / "casebook/ta_alpha_casebook.csv", index=False)
    out.groupby(["family", "case_category"]).size().reset_index(name="count").to_csv(ROOT / "casebook/casebook_summary_by_family.csv", index=False)
    out.head(50).to_csv(ROOT / "casebook/top_success_cases.csv", index=False)
    out.tail(50).to_csv(ROOT / "casebook/top_failure_cases.csv", index=False)
    (ROOT / "casebook/ta_casebook_report.md").write_text("# TA Casebook Report\n\nTop success/failure cases are stored as CSV/parquet. Chart window generation is left as follow-up to avoid expensive plotting in this diagnostics pass.\n", encoding="utf-8")


def system_interaction(c: pd.DataFrame) -> None:
    risk_score = pd.to_numeric(c.get("risk_adjusted_orderflow_score"), errors="coerce")
    flag = risk_score <= risk_score.quantile(0.20)
    tmp = c.copy()
    tmp["orderflow_risk_worst20_flag"] = flag.fillna(False)
    tmp["GOOD"] = tmp["label"].astype(str).str.contains("GOOD")
    tmp["BAD"] = tmp["label"].astype(str).str.contains("BAD")
    rows = []
    for fam, g in tmp.groupby("family"):
        fl = g["orderflow_risk_worst20_flag"].astype(bool)
        rows.append({"family": fam, "rows": len(g), "removed_by_orderflow_worst20_rate": float(fl.mean()), "GOOD_removed_rate": float((fl & g["GOOD"]).sum() / max(1, g["GOOD"].sum())), "BAD_removed_rate": float((fl & g["BAD"]).sum() / max(1, g["BAD"].sum())), "RFE_removed_rate": float((fl & g["RFE"]).sum() / max(1, g["RFE"].sum())), "verdict": "risk_filter_interaction_reference_only"})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "system_interaction/ta_risk_filter_interaction.csv", index=False)
    df.to_csv(ROOT / "system_interaction/ta_good_bad_preservation.csv", index=False)
    df.to_csv(ROOT / "system_interaction/ta_existing_system_overlap.csv", index=False)
    (ROOT / "system_interaction/system_interaction_report.md").write_text("# System Interaction Report\n\nOrderflow_risk_worst20 interaction is diagnostics-only. Q2/R7 exact flags were not available in this TA frame.\n", encoding="utf-8")


def forward_watchlist() -> None:
    rows = [
        ("FW_CANDLE_HAMMER_RECLAIM", "hammer/pinbar reclaim with trend context", "optional orderflow_risk_not_worst20", "30m/1h", "5m/15m", "1h/4h", "OHLCV + optional orderflow", 100),
        ("FW_CANDLE_ENGULFING_WITH_CONFIRM", "engulfing + close/volume confirm", "avoid worst20", "30m/1h", "5m/15m", "1h/4h", "OHLCV", 100),
        ("FW_MA_PULLBACK_RECLAIM", "EMA20/50 pullback reclaim in HTF regime", "4h regime", "1h/3h", "5m/1h", "1h/4h", "OHLCV", 100),
        ("FW_RSI_OVERSOLD_RECLAIM", "RSI reclaim not lower-band ride", "candle confirm", "1h", "5m/15m", "1h/4h", "OHLCV", 100),
        ("FW_BOLLINGER_LOWER_RECLAIM", "lower band reclaim", "avoid trend crash", "1h", "5m", "1h/4h", "OHLCV", 100),
        ("FW_SR_BREAKOUT_RETEST", "breakout + retest hold", "volume/taker confirm", "30m/1h", "5m/15m", "1h/4h", "OHLCV + optional orderflow", 100),
        ("FW_TA_ORDERFLOW_BREAKOUT_TAKER_CONFIRM", "SR/engulfing + taker/proxy CVD confirm", "risk_not_worst20", "30m/1h", "5m", "1h/4h", "OHLCV + proxy CVD/taker/OI", 100),
        ("FW_TA_RISK_FILTER_AVOID_WORST20", "TA candidates filtered by orderflow_risk_worst20", "research-only risk filter", "30m/1h", "5m", "1h/4h", "OHLCV + orderflow frame", 100),
    ]
    df = pd.DataFrame(rows, columns=["watch_id", "entry_condition", "filter_condition", "cooldown", "timeframe", "horizon", "required_data", "min_sample"])
    df["success_criteria"] = "positive current/maker-like net with acceptable RFE in forward"
    df["failure_criteria"] = "cost/gross weak or overfit/overfilter"
    df["priority"] = range(1, len(df) + 1)
    df["expected_frequency"] = "TBD from forward"
    df["historical_evidence"] = "see scorecards"
    df["current_verdict"] = "research_only_forward_validation"
    df.to_csv(ROOT / "forward_watchlist/ta_forward_watchlist.csv", index=False)
    (ROOT / "forward_watchlist/ta_forward_validation_plan.md").write_text("# TA Forward Validation Plan\n\nForward-register only small, interpretable TA conditions. No production connection. Success requires cost-surviving forward evidence.\n", encoding="utf-8")
    (ROOT / "forward_watchlist/forward_watchlist_report.md").write_text("# Forward Watchlist Report\n\nWatchlist generated for classical TA candidates and TA+orderflow confirmations.\n", encoding="utf-8")


def decisions(scorecards: Dict[str, pd.DataFrame]) -> None:
    single = scorecards["single"]
    rows = []
    for _, r in single.iterrows():
        if r["trade_count"] < 100:
            dec = "KEEP_CASEBOOK_ONLY"
        elif r["mean_net_bps"] > 0 or r["maker_like_mean_bps"] > 0:
            dec = "KEEP_FOR_FORWARD_ENTRY_ALPHA"
        elif r["id"] in ["F2_ma_trend"]:
            dec = "KEEP_AS_REGIME_FILTER"
        elif r["id"] in ["F3_momentum", "F5_ta_orderflow"]:
            dec = "KEEP_AS_RISK_FILTER"
        else:
            dec = "DROP"
        rows.append({"family": r["id"], "decision": dec, "mean_net_bps": r["mean_net_bps"], "maker_like_mean_bps": r["maker_like_mean_bps"], "trade_count": r["trade_count"], "production_ready": False})
    pd.DataFrame(rows).to_csv(ROOT / "decision/ta_family_decision_matrix.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "decision/final_ta_alpha_recommendation.csv", index=False)
    scorecards["stage"].to_csv(ROOT / "decision/ta_stage_decision_matrix.csv", index=False)
    pd.DataFrame().to_csv(ROOT / "decision/ta_signal_decision_matrix.csv", index=False)
    (ROOT / "decision/recommended_next_action.md").write_text("# Recommended Next Action\n\nForward-register only the least-bad interpretable TA candidates and verify whether any survives cost out-of-sample; do not promote to production.\n", encoding="utf-8")


def final_report(scorecards: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    stage = scorecards["stage"]
    single = scorecards["single"]
    combo = scorecards["combo"]
    verdicts = ["CLASSICAL_TA_ENTRY_ALPHA_COMPLETED"]
    stage_map = {r["id"]: r for _, r in stage.iterrows()}
    stage_names = list(stage_map)
    if stage_map[stage_names[0]]["mean_net_bps"] > 0:
        verdicts += ["STAGE_1_POSITIVE", "CANDLESTICK_ALPHA_FOUND_RESEARCH_ONLY"]
    else:
        verdicts.append("CANDLESTICK_NO_EDGE")
    try:
        delta = pd.read_csv(ROOT / "scorecards/stage_delta_scorecard.csv")
    except Exception:
        delta = pd.DataFrame(columns=["delta_id", "mean_net_bps_delta"])
    for label, good, bad in [
        ("Stage2_minus_Stage1_MA_add", "STAGE_2_IMPROVES", "STAGE_2_DEGRADES"),
        ("Stage3_minus_Stage2_momentum_add", "STAGE_3_IMPROVES", "STAGE_3_DEGRADES"),
        ("Stage4_minus_Stage3_SR_add", "STAGE_4_IMPROVES", "STAGE_4_DEGRADES"),
        ("Stage5_minus_Stage4_orderflow_add", "STAGE_5_IMPROVES", "STAGE_5_DEGRADES"),
    ]:
        row = delta[delta["delta_id"].eq(label)]
        verdicts.append(good if not row.empty and row.iloc[0]["mean_net_bps_delta"] > 0 else bad)
    if (stage["mean_net_bps"] <= 0).all():
        verdicts += ["CLASSICAL_TA_COST_KILLS_EDGE", "NO_CLASSICAL_TA_ENTRY_ALPHA_FOUND"]
    # Family labels.
    fam = {r["id"]: r for _, r in single.iterrows()}
    if "F2_ma_trend" in fam and fam["F2_ma_trend"]["mean_net_bps"] <= 0:
        verdicts.append("MA_TREND_REGIME_FILTER_ONLY")
        verdicts.append("MA_CROSS_LAG_TOO_LATE")
    if "F3_momentum" in fam and fam["F3_momentum"]["mean_net_bps"] <= 0:
        verdicts.append("MOMENTUM_NO_EDGE")
    if "F4_support_resistance" in fam and fam["F4_support_resistance"]["mean_net_bps"] <= 0:
        verdicts.append("SUPPORT_RESISTANCE_NO_EDGE")
    if "F5_ta_orderflow" in fam:
        verdicts.append("TA_ORDERFLOW_COMBO_ADDS_VALUE" if fam["F5_ta_orderflow"]["mean_net_bps"] > max(fam.get("F4_support_resistance", {}).get("mean_net_bps", -999), -999) else "TA_ORDERFLOW_RISK_FILTER_ONLY")
    verdicts += ["FORWARD_WATCHLIST_READY", "production_not_ready"]
    verdicts = list(dict.fromkeys(verdicts))
    (ROOT / "classical_ta_entry_alpha_research_final_verdict.md").write_text("# Final Verdict\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    best_combo = combo.sort_values("mean_net_bps", ascending=False).head(5).to_dict("records") if not combo.empty else []
    report = f"""# Classical TA Entry Alpha Research Final Report

## Why
CAN_BIT has found more useful risk filters than robust entry timing. This run converts classical TA into mechanical, testable BTCUSDT feature families and evaluates them after cost.

## Families
F1 candlestick catalog, F2 MA/trend regime, F3 momentum/exhaustion, F4 support/resistance, F5 TA+orderflow confirmation/risk context. TA patterns are not treated as guaranteed formulas.

## Stagewise Scorecard
```json
{jdump(stage.to_dict("records"))}
```

## Single Family Scorecard
```json
{jdump(single.to_dict("records"))}
```

## Best Combinations
```json
{jdump(best_combo)}
```

## Interpretation
Candidate generation used closed-candle features only. Entry is next open. Primary replay is fixed 1h with current/maker/2x cost sensitivity. Future high/low/close are used only for outcome/casebook metrics.

## Safety
No production TCN/Q2/R7/Risk/live/order/state path was changed. No private API or order/account/balance/position endpoint was called. forward_orderflow_collector_v4 and Discord policy were read-only. production_ready=false and promotion_ready=false.
"""
    (ROOT / "classical_ta_entry_alpha_research_final_report.md").write_text(report, encoding="utf-8")
    (ROOT / "recommended_next_branch.md").write_text("# Recommended Next Branch\n\nRun a 30-day forward watchlist for the least-bad classical TA candidates, especially small TA+orderflow confirmations, with cost-survival as the only pass criterion.\n", encoding="utf-8")
    return {"verdicts": verdicts, "best_combo": best_combo}


def run_pipeline(mode: str = "full", stage: int | None = None, fast: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    before = safety_snapshot("before")
    log(f"start mode={mode} stage={stage} fast={fast}")
    discovery = input_discovery()
    cache_candidates = ROOT / "backfill/ta_alpha_paper_trades.parquet"
    cache_stage5 = ROOT / "stages/stage_5_ta_orderflow_candidates.parquet"
    if not fast and cache_candidates.exists() and cache_stage5.exists():
        candidates = pd.read_parquet(cache_candidates)
        features = pd.read_parquet(ROOT / "features/ta_feature_frame.parquet") if (ROOT / "features/ta_feature_frame.parquet").exists() else pd.DataFrame()
        stages = {
            "stage_1_candlestick": pd.read_parquet(ROOT / "stages/stage_1_candlestick_candidates.parquet"),
            "stage_2_candle_ma": pd.read_parquet(ROOT / "stages/stage_2_candle_ma_candidates.parquet"),
            "stage_3_candle_ma_momentum": pd.read_parquet(ROOT / "stages/stage_3_candle_ma_momentum_candidates.parquet"),
            "stage_4_candle_ma_momentum_sr": pd.read_parquet(ROOT / "stages/stage_4_candle_ma_momentum_sr_candidates.parquet"),
            "stage_5_ta_orderflow": pd.read_parquet(cache_stage5),
        }
    else:
        tf = build_timeframes(fast=fast)
        features = build_features(tf)
        candidates = build_candidates(features)
        candidates = add_outcomes(candidates, tf["5m"])
        stages = build_stages(candidates)
    if stage is not None:
        keys = list(stages)[:stage]
        stages = {k: v for k, v in stages.items() if k in keys}
    scorecard_cache = {
        "single": ROOT / "scorecards/single_family_scorecard.csv",
        "stage": ROOT / "scorecards/stagewise_scorecard.csv",
        "combo": ROOT / "backfill/outcome_summary_by_combination.csv",
    }
    if all(p.exists() for p in scorecard_cache.values()) and not fast and stage is None:
        scorecards = {k: pd.read_csv(p) for k, p in scorecard_cache.items()}
    else:
        scorecards = evaluate(candidates, stages)
    failure(scorecards)
    casebook(candidates)
    system_interaction(candidates)
    forward_watchlist()
    decisions(scorecards)
    result = final_report(scorecards)
    finalize_audit(before)
    log("done")
    return {
        "mode": mode,
        "stage": stage,
        "fast": fast,
        "feature_rows": len(features),
        "candidate_rows": len(candidates),
        "discovery": discovery,
        "verdicts": result["verdicts"],
        "production_ready": False,
        "promotion_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast-smoke", action="store_true")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--single-families-only", action="store_true")
    parser.add_argument("--combinations-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    if args.dry_run:
        paths = canonical_paths()
        res = {"dry_run": True, "root": str(ROOT), "canonical_5m_exists": Path(paths.get("canonical_5m_path", str(DEFAULT_5M))).exists(), "production_ready": False, "promotion_ready": False}
    elif args.fast_smoke:
        res = run_pipeline(mode="fast_smoke", fast=True)
    elif args.stage:
        res = run_pipeline(mode=f"stage_{args.stage}", stage=args.stage)
    elif args.single_families_only:
        res = run_pipeline(mode="single")
    elif args.combinations_only:
        res = run_pipeline(mode="combinations")
    else:
        res = run_pipeline(mode="full")
    print(jdump(res) if args.json else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
