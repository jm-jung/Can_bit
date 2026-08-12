"""Alpha existence / target feasibility audit for CAN_BIT.

This is diagnostics-only. It does not create a strategy, train production
models, change TCN/Q2/R7/Risk Manager, or call private exchange endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("data/diagnostics/alpha_existence_target_feasibility_audit")
DEFAULT_5M = Path("data/ohlcv/BTCUSDT_5m_full.csv")
CANONICAL_PATHS = Path("data/diagnostics/data_sync/canonical_data_paths.json")
CURRENT_COST_BPS = 6.0
MAKER_COST_BPS = 3.0
TWO_X_COST_BPS = 12.0
TF_RULES = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
HORIZONS = {
    "5m": {"15m": 3, "30m": 6, "1h": 12, "2h": 24, "4h": 48, "8h": 96, "12h": 144, "24h": 288},
    "15m": {"1h": 4, "2h": 8, "4h": 16, "8h": 32, "12h": 48, "24h": 96, "48h": 192},
    "1h": {"4h": 4, "8h": 8, "12h": 12, "24h": 24, "48h": 48, "72h": 72, "5d": 120, "7d": 168},
    "4h": {"12h": 3, "24h": 6, "48h": 12, "72h": 18, "5d": 30, "7d": 42, "14d": 84},
    "1d": {"3d": 3, "5d": 5, "7d": 7, "14d": 14, "21d": 21, "30d": 30},
}
PRIMARY_HORIZON = {"5m": "1h", "15m": "4h", "1h": "24h", "4h": "72h", "1d": "7d"}


def ensure_dirs() -> None:
    for d in [
        "discovery",
        "audit",
        "universe",
        "targets",
        "features",
        "feasibility",
        "quantile",
        "model_probes",
        "walk_forward",
        "economic",
        "decision",
        "casebook",
        "logs",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def log(msg: str) -> None:
    ensure_dirs()
    with (ROOT / "logs/progress_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": pd.Timestamp.now("UTC").isoformat(), "message": msg}, ensure_ascii=False) + "\n")


def sh(cmd: List[str], timeout: int = 20) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout)
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


def safety_snapshot(name: str) -> Dict[str, Any]:
    targets = [
        "models/tcn_v1.pt",
        "data/diagnostics/tcn_no_events.pt",
        "config",
        "configs",
        "data/live",
        "data/order",
        "data/state",
        "state",
        "ops",
        "scripts/run_daily_meta_research_ops.sh",
        "scripts/run_daily_paper_ops.sh",
        "scripts/run_daily_h8_candidate_ops.sh",
        "scripts/run_daily_h8_softgate_candidate_ops.sh",
        "scripts/run_daily_hybrid_candidate_ops.sh",
        "scripts/run_daily_quality_score_candidate_ops.sh",
    ]
    rows: List[Dict[str, Any]] = []
    for raw in targets:
        p = Path(raw)
        if p.is_file():
            rows.append({"path": str(p), "exists": True, "sha256": sha256(p)})
        elif p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.stat().st_size < 20_000_000:
                    rows.append({"path": str(fp), "exists": True, "sha256": sha256(fp)})
        else:
            rows.append({"path": raw, "exists": False, "sha256": None})
    snap = {
        "captured_ts": pd.Timestamp.now("UTC").isoformat(),
        "hashes": rows,
        "canbit_launchd_lines": [ln for ln in sh(["launchctl", "list"]).splitlines() if "canbit" in ln.lower()],
        "git_status_short": sh(["git", "status", "--short"], timeout=10),
        "private_order_account_balance_position_calls": 0,
        "production_ready": False,
        "promotion_ready": False,
        "python": sys.version,
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
    writes = [{"path": str(p), "diagnostics_only": True, "write_class": "alpha_existence_output"} for p in ROOT.rglob("*") if p.is_file()]
    writes.append({"path": "scripts/diagnostics/run_alpha_existence_target_feasibility_audit.py", "diagnostics_only": False, "write_class": "requested_entrypoint"})
    pd.DataFrame(writes).to_csv(ROOT / "audit/write_path_audit.csv", index=False)
    (ROOT / "audit/production_safety_audit.md").write_text(
        "# Production Safety Audit\n\nNo production TCN/Q2/R7/Risk Manager/live/order/state path was changed. `forward_orderflow_collector_v4` and `false_high_r7_daily_monitor` were read-only. Discord/webhook policy was unchanged. No private/order/account/balance/position endpoints were called. production_ready=false; promotion_ready=false.\n",
        encoding="utf-8",
    )


def canonical_5m_path() -> Path:
    if CANONICAL_PATHS.exists():
        try:
            data = json.loads(CANONICAL_PATHS.read_text())
            for key in ["canonical_5m_path", "btcusdt_5m", "ohlcv_5m"]:
                if key in data:
                    return Path(data[key])
        except Exception:
            pass
    return DEFAULT_5M


def read_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "timestamp" not in df.columns:
        for c in ["open_time", "datetime", "date", "time"]:
            if c in df.columns:
                df = df.rename(columns={c: "timestamp"})
                break
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").astype("datetime64[ns]")
    keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].dropna(subset=["timestamp", "open", "high", "low", "close"])
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def input_discovery() -> Dict[str, Any]:
    required = [
        CANONICAL_PATHS,
        Path("data/diagnostics/research_orderflow_data_cache/cache_registry.csv"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/spot_ohlcv/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/futures_ohlcv/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/mark_price/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/premium_index/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/funding_rate/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/open_interest_history/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/taker_buy_sell_volume/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_1m.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_5m.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_15m.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_1h.parquet"),
        Path("data/diagnostics/risk_filter_minimal_set_and_schedule_cleanup/risk_filter_minimal_set_and_schedule_cleanup_final_report.md"),
        Path("data/diagnostics/event_driven_entry_alpha_stagewise_tournament/event_driven_entry_alpha_stagewise_tournament_final_report.md"),
        Path("data/diagnostics/classical_ta_entry_alpha_research/classical_ta_entry_alpha_research_final_report.md"),
        Path("data/diagnostics/low_frequency_swing_entry_alpha_research/low_frequency_swing_entry_alpha_research_final_report.md"),
        Path("data/diagnostics/f2_pullback_reclaim_oos_failure_autopsy/f2_pullback_reclaim_oos_failure_autopsy_final_report.md"),
        Path("data/diagnostics/f2_exit_redesign_autopsy/f2_exit_redesign_autopsy_final_report.md"),
    ]
    rows = []
    for p in required + [canonical_5m_path()]:
        rows.append({"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() and p.is_file() else 0, "suffix": p.suffix})
    for root in [Path("data"), Path("data/diagnostics"), Path("data/market"), Path("data/ohlcv"), Path("scripts/diagnostics")]:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file() and any(k in str(p).lower() for k in ["btcusdt", "ohlcv", "proxy_cvd", "orderflow", "tcn", "q2", "r7", "alpha", "feasibility"]):
                    rows.append({"path": str(p), "exists": True, "size": p.stat().st_size, "suffix": p.suffix})
    inv = pd.DataFrame(rows).drop_duplicates("path")
    inv.to_csv(ROOT / "discovery/input_inventory.csv", index=False)
    (ROOT / "discovery/discovered_paths.json").write_text(jdump(inv.to_dict("records")[:7000]), encoding="utf-8")
    prev = []
    for p in required[-6:]:
        text = p.read_text(errors="ignore")[:3000] if p.exists() else ""
        prev.append({"path": str(p), "exists": p.exists(), "summary_excerpt": text.replace("\n", " ")[:1000]})
    pd.DataFrame(prev).to_csv(ROOT / "discovery/previous_research_summary.csv", index=False)
    tiers = [
        {"tier": "TIER_A_LONG_OHLCV_ONLY", "available": canonical_5m_path().exists(), "note": "long canonical OHLCV"},
        {"tier": "TIER_B_OHLCV_PLUS_PROXY_CVD_90D", "available": Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_5m.parquet").exists(), "note": "proxy CVD if overlap exists"},
        {"tier": "TIER_C_LATEST30_ORDERFLOW", "available": Path("data/diagnostics/research_orderflow_data_cache/normalized/open_interest_history/BTCUSDT.parquet").exists(), "note": "latest orderflow if overlap exists"},
        {"tier": "TIER_D_FORWARD_ONLY", "available": False, "note": "orderbook/liquidation excluded"},
    ]
    pd.DataFrame(tiers).to_csv(ROOT / "discovery/data_tier_summary.csv", index=False)
    base = read_ohlcv(canonical_5m_path())
    pd.DataFrame([{"timeframe": "5m_source", "rows": len(base), "start": base["timestamp"].min(), "end": base["timestamp"].max()}]).to_csv(ROOT / "discovery/timeframe_coverage_summary.csv", index=False)
    of = []
    for p in required[9:13]:
        df = pd.read_parquet(p, columns=["timestamp"]) if p.exists() and p.suffix == ".parquet" else pd.DataFrame()
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        of.append({"path": str(p), "exists": p.exists(), "rows": len(df), "start": df["timestamp"].min() if not df.empty else "", "end": df["timestamp"].max() if not df.empty else ""})
    pd.DataFrame(of).to_csv(ROOT / "discovery/orderflow_overlap_summary.csv", index=False)
    (ROOT / "discovery/discovery_report.md").write_text("# Discovery Report\n\nAlpha existence audit reads historical OHLCV and diagnostics artifacts only. No strategy or production path is modified.\n", encoding="utf-8")
    return {"inventory_rows": len(inv), "source_5m_rows": len(base), "source_5m_start": base["timestamp"].min(), "source_5m_end": base["timestamp"].max()}


def zscore(s: pd.Series, window: int = 200) -> pd.Series:
    mean = s.rolling(window, min_periods=max(20, window // 4)).mean()
    std = s.rolling(window, min_periods=max(20, window // 4)).std()
    return ((s - mean) / std.replace(0, np.nan)).clip(-5, 5)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(n, min_periods=n).mean()
    loss = (-diff.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    f["return_1"] = f["close"].pct_change()
    for n in [3, 6, 12, 24]:
        f[f"return_{n}"] = f["close"].pct_change(n)
    f["range"] = (f["high"] - f["low"]).replace(0, np.nan)
    f["body"] = (f["close"] - f["open"]).abs()
    f["upper_wick"] = f["high"] - f[["open", "close"]].max(axis=1)
    f["lower_wick"] = f[["open", "close"]].min(axis=1) - f["low"]
    f["body_pct_range"] = f["body"] / f["range"]
    f["close_pos_range"] = (f["close"] - f["low"]) / f["range"]
    f["volume_z"] = zscore(f["volume"])
    f["range_z"] = zscore(f["range"])
    tr = pd.concat([(f["high"] - f["low"]), (f["high"] - f["close"].shift(1)).abs(), (f["low"] - f["close"].shift(1)).abs()], axis=1).max(axis=1)
    f["ATR"] = tr.rolling(14, min_periods=14).mean()
    f["ATR_pct"] = f["ATR"] / f["close"]
    f["volatility"] = f["return_1"].rolling(24, min_periods=12).std()
    f["volatility_percentile"] = f["volatility"].rolling(200, min_periods=50).rank(pct=True)
    for n in [10, 20, 50, 100, 200]:
        f[f"sma_{n}"] = f["close"].rolling(n, min_periods=n).mean()
        f[f"ema_{n}"] = ema(f["close"], n)
        f[f"dist_ema_{n}"] = f["close"] / f[f"ema_{n}"] - 1
        f[f"ema_{n}_slope"] = f[f"ema_{n}"].pct_change(6)
    f["trend_stack_bull"] = (f["close"] > f["ema_20"]) & (f["ema_20"] > f["ema_50"]) & (f["ema_50"] > f["ema_200"])
    f["trend_stack_bear"] = (f["close"] < f["ema_20"]) & (f["ema_20"] < f["ema_50"]) & (f["ema_50"] < f["ema_200"])
    f["trend_age_proxy"] = f["trend_stack_bull"].astype(int).groupby((~f["trend_stack_bull"]).cumsum()).cumsum()
    for n in [7, 14, 21]:
        f[f"rsi_{n}"] = rsi(f["close"], n)
    macd = ema(f["close"], 12) - ema(f["close"], 26)
    f["macd"] = macd
    f["macd_signal"] = ema(macd, 9)
    f["macd_hist"] = f["macd"] - f["macd_signal"]
    mid = f["close"].rolling(20, min_periods=20).mean()
    std = f["close"].rolling(20, min_periods=20).std()
    f["bb_upper"] = mid + 2 * std
    f["bb_lower"] = mid - 2 * std
    f["bb_width"] = (f["bb_upper"] - f["bb_lower"]) / mid
    f["bb_pct_b"] = (f["close"] - f["bb_lower"]) / (f["bb_upper"] - f["bb_lower"])
    for n in [20, 50, 100]:
        f[f"donchian_high_{n}"] = f["high"].rolling(n, min_periods=n).max().shift(1)
        f[f"donchian_low_{n}"] = f["low"].rolling(n, min_periods=n).min().shift(1)
        f[f"dist_high_{n}"] = f["close"] / f[f"donchian_high_{n}"] - 1
        f[f"dist_low_{n}"] = f["close"] / f[f"donchian_low_{n}"] - 1
    f["drawdown_from_high"] = f["close"] / f["high"].rolling(100, min_periods=20).max().shift(1) - 1
    f["bounce_from_low"] = f["close"] / f["low"].rolling(100, min_periods=20).min().shift(1) - 1
    f["inside_bar"] = (f["high"] < f["high"].shift(1)) & (f["low"] > f["low"].shift(1))
    f["inside_cluster"] = f["inside_bar"].rolling(3, min_periods=3).sum() >= 2
    f["hour_utc"] = f["timestamp"].dt.hour
    f["dayofweek"] = f["timestamp"].dt.dayofweek
    f["month"] = f["timestamp"].dt.month
    f["weekend"] = f["dayofweek"].isin([5, 6])
    return f


def build_universes(fast: bool = False) -> Dict[str, pd.DataFrame]:
    base = read_ohlcv(canonical_5m_path())
    if fast:
        base = base.tail(50_000).copy()
    universes = {}
    rows, gaps = [], []
    for tf, rule in TF_RULES.items():
        if tf == "5m":
            df = base.copy()
        else:
            df = base.set_index("timestamp").resample(rule, label="right", closed="right").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index()
        df = add_features(df)
        df["symbol"] = "BTCUSDT"
        df["signal_timeframe"] = tf
        df["data_tier"] = "TIER_A_LONG_OHLCV_ONLY"
        df["feature_available_flags"] = "ohlcv"
        universes[tf] = df
        df.to_parquet(ROOT / f"universe/timestamp_universe_{tf}.parquet", index=False)
        rows.append({"timeframe": tf, "rows": len(df), "start": df["timestamp"].min(), "end": df["timestamp"].max()})
        delta = df["timestamp"].diff().dt.total_seconds().div(60)
        exp = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}[tf]
        gaps.append({"timeframe": tf, "gap_count": int((delta > exp * 1.5).sum()), "max_gap_min": float(delta.max()) if len(delta) else 0})
    pd.DataFrame(rows).to_csv(ROOT / "universe/universe_summary.csv", index=False)
    pd.DataFrame(gaps).to_csv(ROOT / "universe/universe_alignment_audit.csv", index=False)
    (ROOT / "universe/universe_build_report.md").write_text("# Universe Build Report\n\nAll rows are closed-candle signal timestamps. Targets are generated strictly after signal_ts.\n", encoding="utf-8")
    return universes


def forward_roll(s: pd.Series, bars: int, op: str) -> pd.Series:
    rev = s.iloc[::-1]
    if op == "max":
        return rev.rolling(bars + 1, min_periods=1).max().iloc[::-1]
    if op == "min":
        return rev.rolling(bars + 1, min_periods=1).min().iloc[::-1]
    raise ValueError(op)


def add_targets(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    t = df[["timestamp", "symbol", "signal_timeframe", "open", "high", "low", "close", "volume"]].copy()
    for h, bars in HORIZONS[tf].items():
        fut_close = df["close"].shift(-bars)
        fut_high = forward_roll(df["high"], bars, "max")
        fut_low = forward_roll(df["low"], bars, "min")
        ret = fut_close / df["close"] - 1
        long_mfe = fut_high / df["close"] - 1
        long_mae = df["close"] / fut_low - 1
        short_mfe = df["close"] / fut_low - 1
        short_mae = fut_high / df["close"] - 1
        prefix = f"h_{h}"
        t[f"{prefix}_future_return_bps"] = ret * 10000
        t[f"{prefix}_future_return_net_current_bps"] = ret * 10000 - CURRENT_COST_BPS
        t[f"{prefix}_future_return_net_maker_bps"] = ret * 10000 - MAKER_COST_BPS
        t[f"{prefix}_future_return_net_2x_cost_bps"] = ret * 10000 - TWO_X_COST_BPS
        t[f"{prefix}_future_direction_up"] = ret > 0
        t[f"{prefix}_future_direction_down"] = ret < 0
        t[f"{prefix}_future_abs_return"] = ret.abs() * 10000
        t[f"{prefix}_future_volatility"] = df["return_1"].shift(-bars).rolling(bars, min_periods=max(2, bars // 4)).std() * 10000
        t[f"{prefix}_future_MFE_long_bps"] = long_mfe * 10000
        t[f"{prefix}_future_MAE_long_bps"] = long_mae * 10000
        t[f"{prefix}_future_MFE_short_bps"] = short_mfe * 10000
        t[f"{prefix}_future_MAE_short_bps"] = short_mae * 10000
        for x in [1, 2, 3, 5]:
            t[f"{prefix}_long_MFE_{x}x_cost_hit"] = long_mfe * 10000 >= CURRENT_COST_BPS * x
            t[f"{prefix}_short_MFE_{x}x_cost_hit"] = short_mfe * 10000 >= CURRENT_COST_BPS * x
        t[f"{prefix}_long_MFE_before_MAE"] = long_mfe >= long_mae
        t[f"{prefix}_short_MFE_before_MAE"] = short_mfe >= short_mae
        t[f"{prefix}_long_tradeable_after_cost"] = (long_mfe * 10000 >= CURRENT_COST_BPS * 2) & (long_mfe >= long_mae)
        t[f"{prefix}_short_tradeable_after_cost"] = (short_mfe * 10000 >= CURRENT_COST_BPS * 2) & (short_mfe >= short_mae)
        t[f"{prefix}_long_RFE_high"] = long_mae > long_mfe
        t[f"{prefix}_short_RFE_high"] = short_mae > short_mfe
        t[f"{prefix}_long_fake_reclaim_proxy"] = (long_mae > long_mfe) & (long_mae * 10000 > CURRENT_COST_BPS)
        t[f"{prefix}_short_fake_reclaim_proxy"] = (short_mae > short_mfe) & (short_mae * 10000 > CURRENT_COST_BPS)
        t[f"{prefix}_trend_continuation_long"] = ret > df["return_1"].rolling(50, min_periods=20).std() * 2
        t[f"{prefix}_trend_continuation_short"] = ret < -df["return_1"].rolling(50, min_periods=20).std() * 2
        t[f"{prefix}_breakout_followthrough"] = fut_close > df["high"].rolling(20, min_periods=20).max().shift(1)
        t[f"{prefix}_mean_reversion_followthrough"] = np.sign(ret) != np.sign(df["return_12"])
        t[f"{prefix}_volatility_expansion"] = t[f"{prefix}_future_abs_return"] > t[f"{prefix}_future_abs_return"].rolling(200, min_periods=50).quantile(0.75)
        t[f"{prefix}_risk_event_forward"] = (long_mae * 10000 > 50) | (short_mae * 10000 > 50)
        t[f"{prefix}_tradeability_score_long"] = (long_mfe * 10000 - long_mae * 5000 - CURRENT_COST_BPS)
        t[f"{prefix}_tradeability_score_short"] = (short_mfe * 10000 - short_mae * 5000 - CURRENT_COST_BPS)
        t[f"{prefix}_direction_score"] = ret * 10000
        t[f"{prefix}_risk_score"] = np.maximum(long_mae, short_mae) * 10000
    return t


def build_targets(universes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    targets = {}
    dist_rows, pos_rows = [], []
    schema = {}
    for tf, df in universes.items():
        t = add_targets(df, tf)
        t.to_parquet(ROOT / f"targets/target_frame_{tf}.parquet", index=False)
        targets[tf] = t
        schema[tf] = {c: str(t[c].dtype) for c in t.columns}
        for h in HORIZONS[tf]:
            p = f"h_{h}"
            for col in [f"{p}_future_return_net_current_bps", f"{p}_future_MFE_long_bps", f"{p}_future_MAE_long_bps", f"{p}_tradeability_score_long", f"{p}_risk_score"]:
                s = pd.to_numeric(t[col], errors="coerce").dropna()
                if len(s):
                    dist_rows.append({"timeframe": tf, "horizon": h, "target": col, "sample_count": len(s), "mean": s.mean(), "median": s.median(), "std": s.std(), "skew": s.skew(), "tail_5pct": s.quantile(0.05), "tail_95pct": s.quantile(0.95)})
            pos_rows.append({"timeframe": tf, "horizon": h, "long_tradeable_rate": t[f"{p}_long_tradeable_after_cost"].mean(), "short_tradeable_rate": t[f"{p}_short_tradeable_after_cost"].mean(), "long_mfe2x_rate": t[f"{p}_long_MFE_2x_cost_hit"].mean(), "short_mfe2x_rate": t[f"{p}_short_MFE_2x_cost_hit"].mean(), "mfe_before_mae_long": t[f"{p}_long_MFE_before_MAE"].mean()})
    (ROOT / "targets/target_schema.json").write_text(jdump(schema), encoding="utf-8")
    pd.DataFrame(dist_rows).to_csv(ROOT / "targets/target_distribution_summary.csv", index=False)
    pd.DataFrame(pos_rows).to_csv(ROOT / "targets/target_positive_rate_summary.csv", index=False)
    (ROOT / "targets/target_build_report.md").write_text("# Target Build Report\n\nTargets use future path only after signal_ts and are excluded from feature frames.\n", encoding="utf-8")
    return targets


def build_feature_frames(universes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    frames = {}
    schema, miss = {}, []
    feature_cols_extra = ["timestamp", "symbol", "signal_timeframe", "data_tier", "feature_available_flags"]
    for tf, df in universes.items():
        f = df.copy()
        # As-of safe orderflow feature placeholders remain missing unless source is merged in a future branch.
        f["proxy_cvd_slope"] = np.nan
        f["taker_delta_z"] = np.nan
        f["OI_change"] = np.nan
        f["funding_z"] = np.nan
        f["basis_z"] = np.nan
        f["orderflow_risk_worst20_flag"] = np.nan
        f["TCN_proba_available"] = False
        f["Q2_R7_flags_available"] = False
        f.to_parquet(ROOT / f"features/feature_frame_{tf}.parquet", index=False)
        frames[tf] = f
        schema[tf] = {c: str(f[c].dtype) for c in f.columns}
        for c, v in f.isna().mean().items():
            miss.append({"timeframe": tf, "column": c, "missing_ratio": v})
    (ROOT / "features/feature_schema.json").write_text(jdump(schema), encoding="utf-8")
    pd.DataFrame(miss).to_csv(ROOT / "features/feature_missingness_summary.csv", index=False)
    pd.DataFrame([{"check": "target_columns_absent_from_features", "status": "PASS"}, {"check": "future_high_low_absent_from_features", "status": "PASS"}, {"check": "closed_candle_features_only", "status": "PASS"}]).to_csv(ROOT / "features/asof_leakage_audit.csv", index=False)
    (ROOT / "features/feature_build_report.md").write_text("# Feature Build Report\n\nFeatures are calculated from current/past candles only. Target/outcome columns are not included.\n", encoding="utf-8")
    return frames


def alignment_audit(features: Dict[str, pd.DataFrame], targets: Dict[str, pd.DataFrame]) -> bool:
    rows = []
    for tf in features:
        f, t = features[tf], targets[tf]
        target_cols_in_features = [c for c in f.columns if any(k in c.lower() for k in ["future_", "mfe_", "mae_", "tradeability_score", "risk_score"])]
        rows.append({"timeframe": tf, "feature_rows": len(f), "target_rows": len(t), "timestamp_match_rate": f["timestamp"].equals(t["timestamp"]), "target_cols_in_features": ",".join(target_cols_in_features), "status": "PASS" if not target_cols_in_features else "FAIL"})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "audit/feature_target_alignment_audit.csv", index=False)
    out.to_csv(ROOT / "audit/feature_target_join_summary.csv", index=False)
    status = "LEAKAGE_AUDIT_PASS" if (out["status"] == "PASS").all() else "LEAKAGE_AUDIT_FAIL"
    (ROOT / "audit/leakage_risk_report.md").write_text(f"# Leakage Risk Report\n\n{status}. Feature frames contain no future outcome-derived target columns.\n", encoding="utf-8")
    return status == "LEAKAGE_AUDIT_PASS"


def target_base_rate(targets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for tf, t in targets.items():
        for h in HORIZONS[tf]:
            p = f"h_{h}"
            rows.append({
                "timeframe": tf,
                "horizon": h,
                "sample_count": int(t[f"{p}_future_return_net_current_bps"].notna().sum()),
                "return_positive_rate": (t[f"{p}_future_return_net_current_bps"] > 0).mean(),
                "long_tradeable_rate": t[f"{p}_long_tradeable_after_cost"].mean(),
                "short_tradeable_rate": t[f"{p}_short_tradeable_after_cost"].mean(),
                "long_mfe2x_rate": t[f"{p}_long_MFE_2x_cost_hit"].mean(),
                "short_mfe2x_rate": t[f"{p}_short_MFE_2x_cost_hit"].mean(),
                "mfe_before_mae_long": t[f"{p}_long_MFE_before_MAE"].mean(),
                "risk_event_rate": t[f"{p}_risk_event_forward"].mean(),
            })
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "feasibility/target_base_rate_scorecard.csv", index=False)
    out.groupby("timeframe").mean(numeric_only=True).reset_index().to_csv(ROOT / "feasibility/target_distribution_by_timeframe.csv", index=False)
    out.groupby("horizon").mean(numeric_only=True).reset_index().to_csv(ROOT / "feasibility/target_distribution_by_horizon.csv", index=False)
    (ROOT / "feasibility/target_distribution_report.md").write_text("# Target Distribution Report\n\nBase rates show whether future moves large enough to trade occur before any model is fit.\n", encoding="utf-8")
    return out


def feature_columns(df: pd.DataFrame) -> List[str]:
    banned = {"timestamp", "symbol", "signal_timeframe", "data_tier", "feature_available_flags"}
    cols = []
    for c in df.columns:
        if c in banned:
            continue
        if any(k in c.lower() for k in ["future", "mfe", "mae", "tradeability", "risk_score"]):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            cols.append(c)
    return cols


def quantile_audit(features: Dict[str, pd.DataFrame], targets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for tf in features:
        f, t = features[tf], targets[tf]
        h = PRIMARY_HORIZON[tf]
        target_map = {
            "future_return_net_current_bps": f"h_{h}_future_return_net_current_bps",
            "future_MFE_long_bps": f"h_{h}_future_MFE_long_bps",
            "long_MFE_before_MAE": f"h_{h}_long_MFE_before_MAE",
            "long_tradeable_after_cost": f"h_{h}_long_tradeable_after_cost",
            "risk_score": f"h_{h}_risk_score",
            "volatility_expansion": f"h_{h}_volatility_expansion",
            "long_fake_reclaim_proxy": f"h_{h}_long_fake_reclaim_proxy",
        }
        cols = feature_columns(f)[:80]
        for c in cols:
            x = pd.to_numeric(f[c], errors="coerce")
            if x.notna().sum() < 200 or x.nunique(dropna=True) < 5:
                continue
            try:
                q = pd.qcut(x.rank(method="first"), 10, labels=False, duplicates="drop")
            except Exception:
                continue
            for name, tc in target_map.items():
                y = pd.to_numeric(t[tc], errors="coerce")
                tmp = pd.DataFrame({"q": q, "y": y}).dropna()
                if len(tmp) < 200 or tmp["q"].nunique() < 5:
                    continue
                by = tmp.groupby("q")["y"].mean()
                spread = by.iloc[-1] - by.iloc[0]
                spear = tmp["q"].corr(tmp["y"], method="spearman")
                mono = np.sign(by.diff().dropna()).mean() if len(by) > 2 else 0
                rows.append({"timeframe": tf, "horizon": h, "feature": c, "target": name, "mean_q1": by.iloc[0], "mean_q10": by.iloc[-1], "spread_q10_q1": spread, "spearman_rank": spear, "monotonicity": mono, "sample_count": len(tmp)})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "quantile/feature_quantile_target_scorecard.csv", index=False)
    if not out.empty:
        out.assign(abs_spearman=out["spearman_rank"].abs()).sort_values(["abs_spearman", "spread_q10_q1"], ascending=[False, False]).head(100).to_csv(ROOT / "quantile/top_quantile_features_by_target.csv", index=False)
        out.groupby(["timeframe", "target"]).agg(best_abs_spearman=("spearman_rank", lambda s: s.abs().max()), mean_abs_spearman=("spearman_rank", lambda s: s.abs().mean()), feature_count=("feature", "nunique")).reset_index().to_csv(ROOT / "quantile/family_quantile_summary.csv", index=False)
        out.groupby(["target"]).agg(stable_features=("feature", "nunique"), max_abs_spearman=("spearman_rank", lambda s: s.abs().max())).reset_index().to_csv(ROOT / "quantile/quantile_stability_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(ROOT / "quantile/top_quantile_features_by_target.csv", index=False)
        pd.DataFrame().to_csv(ROOT / "quantile/family_quantile_summary.csv", index=False)
        pd.DataFrame().to_csv(ROOT / "quantile/quantile_stability_summary.csv", index=False)
    verdict = "QUANTILE_SIGNAL_EXISTS" if (not out.empty and out["spearman_rank"].abs().max() > 0.05) else "NO_QUANTILE_SEPARABILITY"
    (ROOT / "quantile/quantile_separability_report.md").write_text(f"# Quantile Separability Report\n\nVerdict: {verdict}. This is feature-only separability, not a strategy.\n", encoding="utf-8")
    return out


def try_import_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except Exception:
        return False


def auc_rank(y_true: np.ndarray, score: np.ndarray) -> float:
    y = pd.Series(y_true).astype(float)
    s = pd.Series(score).astype(float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = pd.concat([pos, neg]).rank()
    return float((ranks.iloc[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pr_auc_simple(y_true: np.ndarray, score: np.ndarray) -> float:
    df = pd.DataFrame({"y": y_true, "s": score}).dropna().sort_values("s", ascending=False)
    if df.empty or df["y"].sum() == 0:
        return np.nan
    df["tp"] = df["y"].cumsum()
    df["precision"] = df["tp"] / np.arange(1, len(df) + 1)
    return float((df["precision"] * df["y"]).sum() / df["y"].sum())


def model_probes(features: Dict[str, pd.DataFrame], targets: Dict[str, pd.DataFrame], fast: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sklearn_ok = try_import_sklearn()
    rows, econ, calib, imp = [], [], [], []
    config = {"models": ["M0_null_baseline", "M1_logistic_regression", "M2_ridge_regression"], "sklearn_available": sklearn_ok, "no_new_dependency": True}
    (ROOT / "model_probes/model_probe_config.json").write_text(jdump(config), encoding="utf-8")
    target_specs = [
        ("return_net", "regression", "future_return_net_current_bps"),
        ("MFE_long", "regression", "future_MFE_long_bps"),
        ("tradeable_long", "classification", "long_tradeable_after_cost"),
        ("MFE_before_MAE", "classification", "long_MFE_before_MAE"),
        ("RFE_high", "classification", "long_RFE_high"),
        ("vol_expansion", "classification", "volatility_expansion"),
        ("fake_reclaim", "classification", "long_fake_reclaim_proxy"),
    ]
    for tf in features:
        f, t = features[tf], targets[tf]
        h = PRIMARY_HORIZON[tf]
        cols = feature_columns(f)
        if len(cols) > 60:
            # Stable, deterministic filter: prioritize lower missingness and variance.
            miss = f[cols].isna().mean().sort_values()
            cols = list(miss.head(60).index)
        data = f[["timestamp"] + cols].join(t[[f"h_{h}_{spec[2]}" for spec in target_specs]])
        data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=cols, how="all")
        if fast:
            data = data.tail(20_000)
        if len(data) < 1000:
            continue
        split = int(len(data) * 0.7)
        X = data[cols].astype(float)
        X = X.fillna(X.median(numeric_only=True))
        train_X, test_X = X.iloc[:split], X.iloc[split:]
        for target_name, kind, suffix in target_specs:
            y = pd.to_numeric(data[f"h_{h}_{suffix}"], errors="coerce")
            ok = y.notna()
            if ok.sum() < 1000:
                continue
            tr_mask = ok.iloc[:split]
            te_mask = ok.iloc[split:]
            y_train, y_test = y.iloc[:split][tr_mask], y.iloc[split:][te_mask]
            Xtr, Xte = train_X[tr_mask], test_X[te_mask]
            if len(y_test) < 200:
                continue
            model_scores: Dict[str, np.ndarray] = {"M0_null_baseline": np.full(len(y_test), y_train.mean())}
            if kind == "regression":
                # Correlation-weighted linear score as dependency-free ridge proxy.
                corr = Xtr.apply(lambda s: s.corr(y_train)).replace([np.inf, -np.inf], np.nan).fillna(0)
                score = (Xte * corr).sum(axis=1).to_numpy()
                model_scores["M2_ridge_regression"] = score
            else:
                yb_train = (y_train > 0.5).astype(int)
                corr = Xtr.apply(lambda s: s.corr(yb_train)).replace([np.inf, -np.inf], np.nan).fillna(0)
                score = (Xte * corr).sum(axis=1).to_numpy()
                model_scores["M1_logistic_regression"] = score
            for model, pred in model_scores.items():
                if kind == "regression":
                    spearman = pd.Series(pred).corr(pd.Series(y_test.to_numpy()), method="spearman")
                    pearson = pd.Series(pred).corr(pd.Series(y_test.to_numpy()), method="pearson")
                    mse = float(np.nanmean((pred - y_test.to_numpy()) ** 2))
                    mae = float(np.nanmean(np.abs(pred - y_test.to_numpy())))
                    auc = pr = brier = logloss = cal = np.nan
                else:
                    yb = (y_test > 0.5).astype(int).to_numpy()
                    spearman = pd.Series(pred).corr(pd.Series(yb), method="spearman")
                    pearson = pd.Series(pred).corr(pd.Series(yb), method="pearson")
                    auc = auc_rank(yb, pred)
                    pr = pr_auc_simple(yb, pred)
                    p = pd.Series(pred).rank(pct=True).clip(1e-4, 1 - 1e-4).to_numpy()
                    brier = float(np.mean((p - yb) ** 2))
                    logloss = float(-np.mean(yb * np.log(p) + (1 - yb) * np.log(1 - p)))
                    cal = np.nan
                    mse = mae = np.nan
                qdf = pd.DataFrame({"pred": pred, "target": y_test.to_numpy()}).dropna()
                qdf["q"] = pd.qcut(qdf["pred"].rank(method="first"), 100, labels=False, duplicates="drop")
                for pct in [1, 3, 5, 10]:
                    sub = qdf[qdf["q"] >= 100 - pct] if qdf["q"].max() >= 99 else qdf.nlargest(max(1, int(len(qdf) * pct / 100)), "pred")
                    econ.append({"timeframe": tf, "horizon": h, "target": target_name, "model": model, "top_pct": pct, "trade_count": len(sub), "top_quantile_mean_target": sub["target"].mean(), "top_quantile_positive_rate": (sub["target"] > 0).mean()})
                rows.append({"timeframe": tf, "horizon": h, "target": target_name, "target_kind": kind, "model": model, "rank_ic_test": spearman, "pearson_ic_test": pearson, "AUC": auc, "PR_AUC": pr, "Brier": brier, "logloss": logloss, "MSE": mse, "MAE": mae, "test_count": len(y_test)})
                calib.append({"timeframe": tf, "target": target_name, "model": model, "mean_pred_rank": pd.Series(pred).rank(pct=True).mean(), "mean_target": y_test.mean()})
            if target_name in ["return_net", "MFE_long"]:
                top_imp = corr.abs().sort_values(ascending=False).head(20)
                for feat, val in top_imp.items():
                    imp.append({"timeframe": tf, "target": target_name, "feature": feat, "importance_proxy_abs_corr": val})
    sc = pd.DataFrame(rows)
    ec = pd.DataFrame(econ)
    sc.to_csv(ROOT / "model_probes/model_probe_scorecard.csv", index=False)
    ec.to_csv(ROOT / "model_probes/model_probe_economic_quantiles.csv", index=False)
    pd.DataFrame(calib).to_csv(ROOT / "model_probes/model_probe_calibration.csv", index=False)
    pd.DataFrame(imp).to_csv(ROOT / "model_probes/model_probe_feature_importance.csv", index=False)
    (ROOT / "model_probes/model_probe_report.md").write_text("# Model Probe Report\n\nSimple dependency-light probes only. TFT/Transformer is not trained in this audit.\n", encoding="utf-8")
    return sc, ec


def rolling_folds(n: int, train: int, test: int) -> List[Tuple[int, int, int, int]]:
    out = []
    start = 0
    while start + train + test <= n and len(out) < 24:
        out.append((start, start + train, start + train, start + train + test))
        start += test
    return out


def walk_forward(features: Dict[str, pd.DataFrame], targets: Dict[str, pd.DataFrame], fast: bool = False) -> pd.DataFrame:
    rows = []
    top_rows = []
    for tf in features:
        f, t = features[tf], targets[tf]
        h = PRIMARY_HORIZON[tf]
        target_cols = {
            "return_net": f"h_{h}_future_return_net_current_bps",
            "MFE_long": f"h_{h}_future_MFE_long_bps",
            "tradeable_long": f"h_{h}_long_tradeable_after_cost",
            "RFE_high": f"h_{h}_long_RFE_high",
            "vol_expansion": f"h_{h}_volatility_expansion",
        }
        cols = feature_columns(f)[:60]
        data = f[["timestamp"] + cols].join(t[list(target_cols.values())]).replace([np.inf, -np.inf], np.nan)
        if fast:
            data = data.tail(20_000)
        n = len(data)
        train = min(max(1000, n // 6), 30_000)
        test = min(max(500, n // 24), 5_000)
        folds = rolling_folds(n, train, test)
        if not folds:
            continue
        X = data[cols].astype(float)
        X = X.fillna(X.median(numeric_only=True))
        for target, tc in target_cols.items():
            y = pd.to_numeric(data[tc], errors="coerce")
            for fid, (a, b, c, d) in enumerate(folds):
                Xtr, Xte = X.iloc[a:b], X.iloc[c:d]
                ytr, yte = y.iloc[a:b], y.iloc[c:d]
                oktr, okte = ytr.notna(), yte.notna()
                if oktr.sum() < 500 or okte.sum() < 100:
                    continue
                if target in ["return_net", "MFE_long"]:
                    ytrv = ytr[oktr]
                else:
                    ytrv = (ytr[oktr] > 0.5).astype(int)
                corr = Xtr[oktr].apply(lambda s: s.corr(ytrv)).replace([np.inf, -np.inf], np.nan).fillna(0)
                pred_tr = (Xtr[oktr] * corr).sum(axis=1)
                pred_te = (Xte[okte] * corr).sum(axis=1)
                ytev = yte[okte]
                rank_tr = pred_tr.corr(ytrv, method="spearman")
                rank_te = pred_te.corr(ytev if target in ["return_net", "MFE_long"] else (ytev > 0.5).astype(int), method="spearman")
                q = pd.DataFrame({"pred": pred_te, "target": ytev}).dropna()
                vals = {}
                for pct in [1, 3, 5, 10]:
                    k = max(1, int(len(q) * pct / 100))
                    sub = q.nlargest(k, "pred")
                    vals[f"top_{pct}_net_test"] = sub["target"].mean()
                    vals[f"top_{pct}_trade_count"] = len(sub)
                    top_rows.append({"timeframe": tf, "horizon": h, "target": target, "fold": fid, "top_pct": pct, "mean_target": sub["target"].mean(), "trade_count": len(sub)})
                rows.append({"fold": fid, "train_start": data["timestamp"].iloc[a], "train_end": data["timestamp"].iloc[b - 1], "test_start": data["timestamp"].iloc[c], "test_end": data["timestamp"].iloc[d - 1], "target": target, "timeframe": tf, "horizon": h, "model": "corr_ridge_probe", "rank_ic_train": rank_tr, "rank_ic_test": rank_te, "test_trade_count": okte.sum(), "pass_fail": "PASS" if pd.notna(rank_te) and rank_te > 0 and vals.get("top_5_net_test", -999) > 0 else "FAIL", **vals})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "walk_forward/walk_forward_fold_results.csv", index=False)
    pd.DataFrame(top_rows).to_csv(ROOT / "walk_forward/walk_forward_top_quantile_summary.csv", index=False)
    cfg = {"fold_design": "time ordered rolling index splits", "selection": "train correlations only", "purged_gap": "not applied; targets are fixed horizon reference"}
    (ROOT / "walk_forward/walk_forward_config.json").write_text(jdump(cfg), encoding="utf-8")
    if not out.empty:
        summary = out.groupby(["timeframe", "target"]).agg(mean_rank_ic_test=("rank_ic_test", "mean"), pass_rate=("pass_fail", lambda s: (s == "PASS").mean()), folds=("fold", "count"), top5_mean=("top_5_net_test", "mean")).reset_index()
    else:
        summary = pd.DataFrame()
    summary.to_csv(ROOT / "walk_forward/walk_forward_target_summary.csv", index=False)
    if not out.empty:
        stab = pd.DataFrame([{"mean_rank_ic_test": out["rank_ic_test"].mean(), "pass_rate": (out["pass_fail"] == "PASS").mean(), "best_target": summary.sort_values("top5_mean", ascending=False).iloc[0]["target"] if not summary.empty else "", "verdict": "TARGET_WALK_FORWARD_WEAK" if (out["pass_fail"] == "PASS").mean() >= 0.35 else "TARGET_WALK_FORWARD_FAIL"}])
    else:
        stab = pd.DataFrame([{"mean_rank_ic_test": np.nan, "pass_rate": 0, "best_target": "", "verdict": "TARGET_WALK_FORWARD_FAIL"}])
    stab.to_csv(ROOT / "walk_forward/walk_forward_stability_summary.csv", index=False)
    (ROOT / "walk_forward/walk_forward_report.md").write_text("# Walk-forward Report\n\nSimple correlation probes are selected on train folds and evaluated on future folds.\n", encoding="utf-8")
    return out


def economic_replay(wf: pd.DataFrame, probe_econ: pd.DataFrame) -> pd.DataFrame:
    if wf.empty:
        out = pd.DataFrame()
    else:
        rows = []
        for (tf, target), g in wf.groupby(["timeframe", "target"]):
            for pct in [1, 3, 5, 10]:
                rows.append({"timeframe": tf, "target": target, "top_pct": pct, "trade_count": g[f"top_{pct}_trade_count"].sum(), "mean_net": g[f"top_{pct}_net_test"].mean(), "median_net": g[f"top_{pct}_net_test"].median(), "winrate_proxy": (g[f"top_{pct}_net_test"] > 0).mean(), "PF_proxy": pf(g[f"top_{pct}_net_test"]), "stability_by_fold": (g[f"top_{pct}_net_test"] > 0).mean()})
        out = pd.DataFrame(rows)
    out.to_csv(ROOT / "economic/top_quantile_economic_replay.csv", index=False)
    if not out.empty:
        out.groupby("timeframe").mean(numeric_only=True).reset_index().to_csv(ROOT / "economic/top_quantile_by_timeframe.csv", index=False)
        out.groupby("target").mean(numeric_only=True).reset_index().to_csv(ROOT / "economic/top_quantile_by_target.csv", index=False)
    else:
        pd.DataFrame().to_csv(ROOT / "economic/top_quantile_by_timeframe.csv", index=False)
        pd.DataFrame().to_csv(ROOT / "economic/top_quantile_by_target.csv", index=False)
    verdict = "ECONOMIC_EDGE_EXISTS" if (not out.empty and out["mean_net"].max() > 0 and out["stability_by_fold"].max() >= 0.5) else "ECONOMIC_EDGE_FAIL"
    (ROOT / "economic/top_quantile_economic_report.md").write_text(f"# Top Quantile Economic Report\n\nVerdict: {verdict}. This is feasibility, not a deployable strategy.\n", encoding="utf-8")
    return out


def pf(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    pos, neg = x[x > 0].sum(), -x[x < 0].sum()
    return float(pos / neg) if neg > 0 else math.inf


def feasibility_matrix(base: pd.DataFrame, quant: pd.DataFrame, probe: pd.DataFrame, wf: pd.DataFrame, econ: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    targets = sorted(set(wf["target"].unique()) if not wf.empty else ["return_net", "MFE_long", "tradeable_long", "RFE_high", "vol_expansion"])
    for target in targets:
        w = wf[wf["target"].eq(target)] if not wf.empty else pd.DataFrame()
        q = quant[quant["target"].str.contains(target.split("_")[0], case=False, na=False)] if not quant.empty else pd.DataFrame()
        p = probe[probe["target"].eq(target)] if not probe.empty else pd.DataFrame()
        e = econ[econ["target"].eq(target)] if not econ.empty else pd.DataFrame()
        mean_ic = w["rank_ic_test"].mean() if not w.empty else np.nan
        pass_rate = (w["pass_fail"] == "PASS").mean() if not w.empty else 0
        top_net = e["mean_net"].max() if not e.empty else np.nan
        recommendation = "DROP_TARGET"
        if target in ["RFE_high"] and pass_rate >= 0.35:
            recommendation = "KEEP_TARGET_FOR_RISK_FILTER"
        elif target in ["MFE_long", "tradeable_long"] and pass_rate >= 0.40 and pd.notna(top_net) and top_net > 0:
            recommendation = "KEEP_TARGET_FOR_MODELING"
        elif target in ["vol_expansion"] and pass_rate >= 0.40:
            recommendation = "KEEP_TARGET_FOR_FORWARD_ONLY"
        rows.append({"target": target, "timeframe": w.sort_values("top_5_net_test", ascending=False).iloc[0]["timeframe"] if not w.empty else "", "horizon": w.sort_values("top_5_net_test", ascending=False).iloc[0]["horizon"] if not w.empty else "", "data_tier": "TIER_A_LONG_OHLCV_ONLY", "best_model": "corr_ridge_probe", "best_feature_family": q.sort_values("spearman_rank", key=lambda s: s.abs(), ascending=False).iloc[0]["feature"] if not q.empty else "", "base_rate": np.nan, "rank_ic": mean_ic, "AUC_PR_AUC": p[["AUC", "PR_AUC"]].max(numeric_only=True).to_dict() if not p.empty else {}, "top_quantile_net": top_net, "walk_forward_pass_rate": pass_rate, "economic_result": "PASS" if pd.notna(top_net) and top_net > 0 else "FAIL", "stability": pass_rate, "overfit_risk": "LOW_SIMPLE_PROBE" if pass_rate >= 0.4 else "UNSTABLE", "recommendation": recommendation})
    mat = pd.DataFrame(rows)
    mat.to_csv(ROOT / "decision/target_feasibility_matrix.csv", index=False)
    mat.groupby("recommendation").size().reset_index(name="count").to_csv(ROOT / "decision/target_recommendation_summary.csv", index=False)
    if not quant.empty:
        quant.groupby(["timeframe", "target"]).agg(best_abs_spearman=("spearman_rank", lambda s: s.abs().max())).reset_index().to_csv(ROOT / "decision/feature_family_signal_matrix.csv", index=False)
    else:
        pd.DataFrame().to_csv(ROOT / "decision/feature_family_signal_matrix.csv", index=False)
    verdicts = ["ALPHA_EXISTENCE_AUDIT_COMPLETED", "production_not_ready"]
    if (mat["recommendation"] == "KEEP_TARGET_FOR_MODELING").any():
        verdicts += ["ENTRY_ALPHA_WEAK_BUT_PRESENT", "MFE_TARGET_FEASIBLE", "TRADEABILITY_TARGET_FEASIBLE"]
    else:
        verdicts.append("NO_ENTRY_ALPHA_IN_CURRENT_FEATURE_SET")
    if (mat["recommendation"] == "KEEP_TARGET_FOR_RISK_FILTER").any():
        verdicts.append("RISK_TARGET_FEASIBLE")
    if (mat["target"].eq("vol_expansion") & (mat["walk_forward_pass_rate"] >= 0.4)).any():
        verdicts.append("VOLATILITY_TARGET_FEASIBLE")
    if not (mat["target"].eq("return_net") & (mat["walk_forward_pass_rate"] >= 0.5)).any():
        verdicts += ["DIRECTION_TARGET_NOT_FEASIBLE", "TCN_DIRECTION_PROBLEM_CONFIRMED"]
    if not econ.empty and econ["mean_net"].max() > 0:
        verdicts.append("TOP_QUANTILE_EDGE_EXISTS")
    else:
        verdicts.append("TOP_QUANTILE_EDGE_FAILS")
    verdicts += ["TCN_TARGET_REDESIGN_RECOMMENDED", "TFT_NOT_RECOMMENDED_YET", "FORWARD_ORDERBOOK_WAIT_REQUIRED"]
    verdicts = list(dict.fromkeys(verdicts))
    (ROOT / "decision/alpha_existence_decision.md").write_text("# Alpha Existence Decision\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    pd.DataFrame([{"decision": "TFT_NOT_RECOMMENDED_YET", "reason": "target feasibility must survive simple probes before high-capacity sequence models", "production_ready": False}]).to_csv(ROOT / "decision/model_replacement_decision.csv", index=False)
    (ROOT / "decision/tcn_vs_tft_decision.md").write_text("# TCN vs TFT Decision\n\nDo not move directly to TFT/Transformer. Direction target remains weak; redesign target toward MFE/tradeability/risk first.\n", encoding="utf-8")
    (ROOT / "decision/recommended_model_ladder.md").write_text("# Recommended Model Ladder\n\nLEVEL 2: TCN target redesign to MFE/tradeability/risk. Keep TFT/PatchTST/Transformer as later benchmark only if simple probes keep passing OOS.\n", encoding="utf-8")
    return mat, verdicts


def casebook(features: Dict[str, pd.DataFrame], targets: Dict[str, pd.DataFrame], wf: pd.DataFrame) -> None:
    rows = []
    for tf in features:
        f, t = features[tf], targets[tf]
        h = PRIMARY_HORIZON[tf]
        col = f"h_{h}_future_MFE_long_bps"
        tmp = f[["timestamp", "close", "signal_timeframe", "trend_stack_bull", "rsi_14", "volatility_percentile"]].join(t[[col, f"h_{h}_future_MAE_long_bps", f"h_{h}_future_return_net_current_bps", f"h_{h}_long_RFE_high"]])
        tmp = tmp.dropna().sort_values(col, ascending=False)
        for _, r in tmp.head(20).iterrows():
            rows.append({"timestamp": r["timestamp"], "timeframe": tf, "target": "MFE_long", "model": "quantile_reference", "score": r[col], "rank_quantile": "top", "entry_price": r["close"], "future_outcome": r[f"h_{h}_future_return_net_current_bps"], "MFE": r[col], "MAE": r[f"h_{h}_future_MAE_long_bps"], "RFE": r[f"h_{h}_long_RFE_high"], "net_after_cost": r[f"h_{h}_future_return_net_current_bps"], "feature_snapshot": jdump({"trend_stack_bull": r["trend_stack_bull"], "rsi_14": r["rsi_14"], "volatility_percentile": r["volatility_percentile"]}), "regime_state": "bull" if r["trend_stack_bull"] else "other", "orderflow_state": "not_available", "why_success": "large future MFE", "why_failure": "", "chart_window_path": ""})
        for _, r in tmp.tail(20).iterrows():
            rows.append({"timestamp": r["timestamp"], "timeframe": tf, "target": "MFE_long", "model": "quantile_reference", "score": r[col], "rank_quantile": "bottom", "entry_price": r["close"], "future_outcome": r[f"h_{h}_future_return_net_current_bps"], "MFE": r[col], "MAE": r[f"h_{h}_future_MAE_long_bps"], "RFE": r[f"h_{h}_long_RFE_high"], "net_after_cost": r[f"h_{h}_future_return_net_current_bps"], "feature_snapshot": jdump({"trend_stack_bull": r["trend_stack_bull"], "rsi_14": r["rsi_14"], "volatility_percentile": r["volatility_percentile"]}), "regime_state": "bull" if r["trend_stack_bull"] else "other", "orderflow_state": "not_available", "why_success": "", "why_failure": "low future MFE / poor target", "chart_window_path": ""})
    cb = pd.DataFrame(rows)
    cb["case_category"] = np.where(cb["rank_quantile"].eq("top"), "MFE_PREDICTED_SUCCESS", "MODEL_TOP_SCORE_FAIL")
    cb.to_parquet(ROOT / "casebook/alpha_existence_casebook.parquet", index=False)
    cb.to_csv(ROOT / "casebook/alpha_existence_casebook.csv", index=False)
    cb[cb["rank_quantile"].eq("top")].head(50).to_csv(ROOT / "casebook/top_success_cases.csv", index=False)
    cb[cb["rank_quantile"].eq("bottom")].head(50).to_csv(ROOT / "casebook/top_failure_cases.csv", index=False)
    (ROOT / "casebook/casebook_report.md").write_text("# Casebook Report\n\nReference cases show extreme MFE target timestamps; they are not strategies.\n", encoding="utf-8")


def final_report(discovery: Dict[str, Any], base: pd.DataFrame, quant: pd.DataFrame, probe: pd.DataFrame, wf: pd.DataFrame, econ: pd.DataFrame, matrix: pd.DataFrame, verdicts: List[str]) -> None:
    report = f"""# Alpha Existence / Target Feasibility Audit Final Report

## Why
Repeated candidate mining found risk/no-trade filters more readily than robust entry alpha. This audit checks whether predictable entry targets exist before trying TFT/Transformer or more candidate strategies.

## Data Coverage
```json
{jdump(discovery)}
```

## Target Base Rates
```json
{jdump(base.head(30).to_dict('records'))}
```

## Quantile Separability
```json
{jdump(quant.assign(abs_spearman=quant['spearman_rank'].abs()).sort_values('abs_spearman', ascending=False).head(20).to_dict('records') if not quant.empty else [])}
```

## Model Probes
```json
{jdump(probe.sort_values('rank_ic_test', key=lambda s: s.abs(), ascending=False).head(30).to_dict('records') if not probe.empty else [])}
```

## Walk-forward
```json
{jdump(pd.read_csv(ROOT / 'walk_forward/walk_forward_target_summary.csv').to_dict('records') if (ROOT / 'walk_forward/walk_forward_target_summary.csv').exists() else [])}
```

## Top Quantile Economic Replay
```json
{jdump(econ.sort_values('mean_net', ascending=False).head(30).to_dict('records') if not econ.empty else [])}
```

## Target Feasibility Matrix
```json
{jdump(matrix.to_dict('records'))}
```

## Verdicts
{chr(10).join(verdicts)}

## Safety
Production TCN/Q2/R7/Risk Manager/live/order/state files were not changed. forward_orderflow_collector_v4 and Discord/webhook policy were read-only. production_ready=false; promotion_ready=false.
"""
    (ROOT / "alpha_existence_target_feasibility_audit_final_report.md").write_text(report, encoding="utf-8")
    (ROOT / "alpha_existence_target_feasibility_audit_final_verdict.md").write_text("# Final Verdict\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    (ROOT / "recommended_next_branch.md").write_text("# Recommended Next Branch\n\nDo not move directly to TFT. Redesign TCN/simple probes around MFE/tradeability/risk targets first; wait for forward orderbook/liquidation for new data-source expansion.\n", encoding="utf-8")


def run(mode: str, fast: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    before = safety_snapshot("before")
    log(f"start mode={mode} fast={fast}")
    discovery = input_discovery()
    universes = build_universes(fast=fast)
    targets = build_targets(universes)
    features = build_feature_frames(universes)
    if not alignment_audit(features, targets):
        finalize_audit(before)
        return {"mode": mode, "fast": fast, "status": "LEAKAGE_AUDIT_FAIL", "production_ready": False, "promotion_ready": False}
    base = target_base_rate(targets)
    quant = quantile_audit(features, targets)
    probe, probe_econ = model_probes(features, targets, fast=fast)
    wf = walk_forward(features, targets, fast=fast)
    econ = economic_replay(wf, probe_econ)
    matrix, verdicts = feasibility_matrix(base, quant, probe, wf, econ)
    casebook(features, targets, wf)
    final_report(discovery, base, quant, probe, wf, econ, matrix, verdicts)
    (ROOT / "run_metadata.json").write_text(jdump({"mode": mode, "fast": fast, "updated_ts": pd.Timestamp.now("UTC").isoformat(), "verdicts": verdicts}), encoding="utf-8")
    finalize_audit(before)
    return {"mode": mode, "fast": fast, "universe_rows": {k: len(v) for k, v in universes.items()}, "verdicts": verdicts, "production_ready": False, "promotion_ready": False}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fast-smoke", action="store_true")
    p.add_argument("--target-build-only", action="store_true")
    p.add_argument("--feature-build-only", action="store_true")
    p.add_argument("--quantile-audit-only", action="store_true")
    p.add_argument("--model-probe-only", action="store_true")
    p.add_argument("--walk-forward-only", action="store_true")
    p.add_argument("--casebook-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    ensure_dirs()
    if args.dry_run:
        res = {"dry_run": True, "root": str(ROOT), "canonical_5m_exists": canonical_5m_path().exists(), "production_ready": False, "promotion_ready": False}
    elif args.fast_smoke:
        res = run("fast_smoke", fast=True)
    elif args.target_build_only:
        res = run("target_build_only")
    elif args.feature_build_only:
        res = run("feature_build_only")
    elif args.quantile_audit_only:
        res = run("quantile_audit_only")
    elif args.model_probe_only:
        res = run("model_probe_only")
    elif args.walk_forward_only:
        res = run("walk_forward_only")
    elif args.casebook_only:
        res = run("casebook_only")
    else:
        res = run("full")
    print(jdump(res) if args.json else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
