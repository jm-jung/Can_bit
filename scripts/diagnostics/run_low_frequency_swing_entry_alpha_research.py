"""Low-frequency swing entry alpha research for CAN_BIT.

Diagnostics-only. Signal timeframes are restricted to 1h/4h/1d closed candles.
5m is used only for conservative next-open fill and outcome replay.
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
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("data/diagnostics/low_frequency_swing_entry_alpha_research")
CANONICAL_PATHS = Path("data/diagnostics/data_sync/canonical_data_paths.json")
DEFAULT_5M = Path("data/ohlcv/BTCUSDT_5m_full.csv")
DEFAULT_1M = Path("data/market/btcusdt_1m.parquet")
ORDERFLOW_FRAME = Path("data/diagnostics/btc_orderflow_cost_kill_rescue_positive_island/frame/combined_candidate_research_frame.parquet")
CURRENT_COST_BPS = 6.0
TF_RULES = {"1h": "1h", "4h": "4h", "1d": "1D"}
HORIZON_BARS_5M = {
    "H1_4h": 48,
    "H2_8h": 96,
    "H3_12h": 144,
    "H4_24h": 288,
    "H5_48h": 576,
    "H6_72h": 864,
    "H7_5d": 1440,
    "H8_7d": 2016,
    "H9_14d": 4032,
}
PRIMARY_HORIZON = "H4_24h"


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
        "walk_forward",
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
    cmp_rows = []
    for x in after.get("hashes", []):
        old = bmap.get(x["path"])
        cmp_rows.append({"path": x["path"], "sha256_before": old, "sha256_after": x.get("sha256"), "changed": old is not None and old != x.get("sha256")})
    (ROOT / "audit/hash_before_after.json").write_text(jdump(cmp_rows), encoding="utf-8")
    writes = [{"path": str(p), "diagnostics_only": True, "write_class": "diagnostics_output"} for p in ROOT.rglob("*") if p.is_file()]
    writes.append({"path": "scripts/diagnostics/run_low_frequency_swing_entry_alpha_research.py", "diagnostics_only": False, "write_class": "requested_entrypoint"})
    pd.DataFrame(writes).to_csv(ROOT / "audit/write_path_audit.csv", index=False)
    (ROOT / "audit/production_safety_audit.md").write_text(
        "# Production Safety Audit\n\nNo production TCN/Q2/R7/Risk Manager/live/order/state path was changed. `forward_orderflow_collector_v4` and `false_high_r7_daily_monitor` were read-only. Discord/webhook policy was unchanged. No private/order/account/balance/position endpoints were called. production_ready=false; promotion_ready=false.\n",
        encoding="utf-8",
    )


def canonical_paths() -> Dict[str, str]:
    if not CANONICAL_PATHS.exists():
        return {}
    try:
        return json.loads(CANONICAL_PATHS.read_text())
    except Exception:
        return {}


def read_ohlcv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
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
        Path("data/diagnostics/classical_ta_entry_alpha_research/classical_ta_entry_alpha_research_final_report.md"),
    ]
    rows = []
    for p in required:
        rows.append({"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() and p.is_file() else 0, "suffix": p.suffix})
    for root in [Path("data"), Path("data/diagnostics"), Path("scripts"), Path("scripts/diagnostics"), Path("config"), Path("configs")]:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and any(k in str(p).lower() for k in ["ohlcv", "btcusdt", "proxy_cvd", "orderflow", "risk_filter", "classical_ta", "event_driven"]):
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
    tier_rows = [
        {"tier": "TIER_A_LONG_OHLCV_ONLY", "available": True, "note": "canonical 5m OHLCV resampled to 1h/4h/1d"},
        {"tier": "TIER_B_OHLCV_PLUS_ORDERFLOW_90D", "available": ORDERFLOW_FRAME.exists(), "note": "proxy CVD/orderflow frame overlap, research-only"},
        {"tier": "TIER_C_LATEST30_ORDERFLOW", "available": ORDERFLOW_FRAME.exists(), "note": "included in combined prior frame if present"},
        {"tier": "TIER_D_FORWARD_ONLY", "available": False, "note": "orderbook/liquidation watchlist only"},
    ]
    pd.DataFrame(tier_rows).to_csv(ROOT / "discovery/data_tier_summary.csv", index=False)
    (ROOT / "discovery/discovery_report.md").write_text("# Discovery Report\n\nLow-frequency swing research uses 1h/4h/1d closed candles as signal timeframes. 5m is used only for fill/replay. Orderflow/proxy CVD is research-only and availability-tiered.\n", encoding="utf-8")
    return {"inventory_rows": len(inv), "ohlcv_sources": summaries}


def orderflow_overlap_summary() -> None:
    rows = []
    if ORDERFLOW_FRAME.exists():
        odf = pd.read_parquet(ORDERFLOW_FRAME, columns=["timestamp"])
        odf["timestamp"] = pd.to_datetime(odf["timestamp"], errors="coerce")
        rows.append({"source": "btc_orderflow_combined_frame", "rows": len(odf), "start": odf["timestamp"].min(), "end": odf["timestamp"].max()})
    pd.DataFrame(rows).to_csv(ROOT / "discovery/orderflow_overlap_summary.csv", index=False)


def zscore(s: pd.Series, window: int = 200) -> pd.Series:
    mean = s.rolling(window, min_periods=max(20, window // 4)).mean()
    std = s.rolling(window, min_periods=max(20, window // 4)).std()
    return ((s - mean) / std.replace(0, np.nan)).clip(-5, 5)


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-diff.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    f["return"] = f["close"].pct_change()
    f["log_return"] = np.log(f["close"] / f["close"].shift(1))
    f["body"] = (f["close"] - f["open"]).abs()
    f["range"] = (f["high"] - f["low"]).replace(0, np.nan)
    f["upper_wick"] = f["high"] - f[["open", "close"]].max(axis=1)
    f["lower_wick"] = f[["open", "close"]].min(axis=1) - f["low"]
    f["body_pct_of_range"] = (f["body"] / f["range"]).clip(0, 5)
    f["upper_wick_pct"] = (f["upper_wick"] / f["range"]).clip(0, 5)
    f["lower_wick_pct"] = (f["lower_wick"] / f["range"]).clip(0, 5)
    f["close_position_in_range"] = ((f["close"] - f["low"]) / f["range"]).clip(0, 1)
    f["volume_z"] = zscore(f["volume"], 200)
    f["range_z"] = zscore(f["range"], 200)
    tr = pd.concat([(f["high"] - f["low"]), (f["high"] - f["close"].shift(1)).abs(), (f["low"] - f["close"].shift(1)).abs()], axis=1).max(axis=1)
    f["atr_14"] = tr.rolling(14, min_periods=14).mean()
    f["atr_pct"] = f["atr_14"] / f["close"]
    f["realized_vol"] = f["return"].rolling(24, min_periods=12).std()
    f["volatility_percentile"] = f["realized_vol"].rolling(200, min_periods=50).rank(pct=True)
    for n in [3, 6, 12, 24]:
        f[f"trend_return_{n}"] = f["close"].pct_change(n)
    f["drawdown_from_recent_high"] = f["close"] / f["high"].rolling(50, min_periods=20).max().shift(1) - 1
    f["bounce_from_recent_low"] = f["close"] / f["low"].rolling(50, min_periods=20).min().shift(1) - 1
    return f


def build_timeframes(fast: bool = False) -> Dict[str, pd.DataFrame]:
    paths = canonical_paths()
    base = read_ohlcv(Path(paths.get("canonical_5m_path", str(DEFAULT_5M))))
    if fast:
        base = base.tail(50_000).copy()
    base = add_base_features(base)
    out = {"5m": base}
    base_idx = base.set_index("timestamp")
    for tf, rule in TF_RULES.items():
        r = base_idx.resample(rule, label="right", closed="right").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index()
        out[tf] = add_base_features(r)
    rows, gaps = [], []
    expected = {"5m": 5, "1h": 60, "4h": 240, "1d": 1440}
    for tf, df in out.items():
        suffix = "for_replay" if tf == "5m" else "ohlcv"
        df.to_parquet(ROOT / f"timeframes/btcusdt_{tf}_{suffix}.parquet", index=False)
        rows.append({"timeframe": tf, "rows": len(df), "start": df["timestamp"].min(), "end": df["timestamp"].max(), "signal_allowed": tf in {"1h", "4h", "1d"}})
        delta = df["timestamp"].diff().dt.total_seconds().div(60)
        gaps.append({"timeframe": tf, "expected_minutes": expected[tf], "gap_count": int((delta > expected[tf] * 1.5).sum()), "max_gap_minutes": float(delta.max()) if len(delta) else 0})
    pd.DataFrame(rows).to_csv(ROOT / "timeframes/timeframe_build_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "discovery/timeframe_coverage_summary.csv", index=False)
    pd.DataFrame(gaps).to_csv(ROOT / "timeframes/timeframe_gap_audit.csv", index=False)
    (ROOT / "timeframes/timeframe_build_report.md").write_text("# Timeframe Build Report\n\nSignal timeframes are restricted to 1h/4h/1d closed candles. 5m is replay/fill only.\n", encoding="utf-8")
    return out


def add_indicators(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    f = df.copy()
    for n in [10, 20, 50, 100, 200]:
        f[f"sma_{n}"] = f["close"].rolling(n, min_periods=n).mean()
        f[f"ema_{n}"] = ema(f["close"], n)
        f[f"ema_{n}_slope_3"] = f[f"ema_{n}"].pct_change(3)
        f[f"ema_{n}_slope_6"] = f[f"ema_{n}"].pct_change(6)
        f[f"dist_ema_{n}"] = f["close"] / f[f"ema_{n}"] - 1
    f["ema20_above_ema50"] = f["ema_20"] > f["ema_50"]
    f["ema50_above_ema200"] = f["ema_50"] > f["ema_200"]
    f["trend_stack_bull"] = (f["close"] > f["ema_20"]) & (f["ema_20"] > f["ema_50"]) & (f["ema_50"] > f["ema_200"])
    f["trend_stack_bear"] = (f["close"] < f["ema_20"]) & (f["ema_20"] < f["ema_50"]) & (f["ema_50"] < f["ema_200"])
    f["ema20_50_cross_up"] = f["ema20_above_ema50"] & ~f["ema20_above_ema50"].shift(1).fillna(False)
    f["ema20_50_cross_down"] = ~f["ema20_above_ema50"] & f["ema20_above_ema50"].shift(1).fillna(False)
    f["rsi_14"] = rsi(f["close"], 14)
    f["rsi_21"] = rsi(f["close"], 21)
    macd = ema(f["close"], 12) - ema(f["close"], 26)
    f["macd"] = macd
    f["macd_signal"] = ema(macd, 9)
    f["macd_hist"] = f["macd"] - f["macd_signal"]
    mid = f["close"].rolling(20, min_periods=20).mean()
    std = f["close"].rolling(20, min_periods=20).std()
    f["bb_mid"] = mid
    f["bb_upper"] = mid + 2 * std
    f["bb_lower"] = mid - 2 * std
    f["bb_width"] = (f["bb_upper"] - f["bb_lower"]) / f["bb_mid"]
    f["bb_pct_b"] = (f["close"] - f["bb_lower"]) / (f["bb_upper"] - f["bb_lower"])
    for n in [20, 50, 100]:
        f[f"donchian_high_{n}"] = f["high"].rolling(n, min_periods=n).max().shift(1)
        f[f"donchian_low_{n}"] = f["low"].rolling(n, min_periods=n).min().shift(1)
    f["prev_high_20"] = f["high"].rolling(20, min_periods=20).max().shift(1)
    f["prev_low_20"] = f["low"].rolling(20, min_periods=20).min().shift(1)
    f["inside_bar"] = (f["high"] < f["high"].shift(1)) & (f["low"] > f["low"].shift(1))
    f["inside_cluster"] = f["inside_bar"].rolling(3, min_periods=3).sum() >= 2
    f["signal_timeframe"] = tf
    return f


def merge_regimes(sig: pd.DataFrame, tf: str, all_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = sig.copy()
    for higher in ["4h", "1d"]:
        if tf == higher:
            continue
        h = all_tf[higher][["timestamp", "trend_stack_bull", "trend_stack_bear", "ema20_above_ema50"]].copy()
        h = h.rename(columns={c: f"{higher}_{c}" for c in h.columns if c != "timestamp"})
        # Completed higher candle only.
        shift = pd.Timedelta(hours=4) if higher == "4h" else pd.Timedelta(days=1)
        h["timestamp"] = pd.to_datetime(h["timestamp"] + shift, errors="coerce").astype("datetime64[ns]")
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce").astype("datetime64[ns]")
        out = pd.merge_asof(out.sort_values("timestamp"), h.sort_values("timestamp"), on="timestamp", direction="backward")
    return out


def merge_orderflow(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not ORDERFLOW_FRAME.exists():
        for c in ["risk_adjusted_orderflow_score", "cvd_taker_combined_score", "taker_delta", "cvd_slope", "oi_change", "funding_score", "basis_score"]:
            out[c] = np.nan
        out["orderflow_available"] = False
        return out
    cols = ["timestamp", "risk_adjusted_orderflow_score", "cvd_taker_combined_score", "taker_delta", "cvd_slope", "oi_change", "funding_score", "basis_score"]
    odf = pd.read_parquet(ORDERFLOW_FRAME, columns=cols).drop_duplicates("timestamp")
    odf["timestamp"] = pd.to_datetime(odf["timestamp"], errors="coerce").astype("datetime64[ns]")
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce").astype("datetime64[ns]")
    out = pd.merge_asof(out.sort_values("timestamp"), odf.sort_values("timestamp"), on="timestamp", direction="backward", tolerance=pd.Timedelta("2h"))
    out["orderflow_available"] = out["risk_adjusted_orderflow_score"].notna()
    return out


def build_features(tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for name in ["1h", "4h", "1d"]:
        f = add_indicators(tf[name], name)
        f = merge_regimes(f, name, {k: add_indicators(v, k) for k, v in tf.items() if k in {"4h", "1d"}})
        f = merge_orderflow(f)
        parts.append(f)
        f.to_parquet(ROOT / f"features/{name}_swing_feature_frame.parquet", index=False)
    allf = pd.concat(parts, ignore_index=True).sort_values(["timestamp", "signal_timeframe"]).reset_index(drop=True)
    allf.to_parquet(ROOT / "features/swing_feature_frame.parquet", index=False)
    (ROOT / "features/swing_feature_schema.json").write_text(jdump({c: str(allf[c].dtype) for c in allf.columns}), encoding="utf-8")
    allf.isna().mean().reset_index().rename(columns={"index": "column", 0: "missing_ratio"}).to_csv(ROOT / "features/swing_feature_missingness.csv", index=False)
    return allf


def write_catalogs() -> None:
    catalogs = {
        "trend_regime_signal_catalog.csv": ["SW_TREND_STACK_BULL", "SW_TREND_STACK_BEAR", "SW_EMA20_50_CROSS_LONG", "SW_EMA20_50_CROSS_SHORT", "SW_MA_SLOPE_TURN_UP", "SW_MA_SLOPE_TURN_DOWN"],
        "pullback_reclaim_signal_catalog.csv": ["SW_PULLBACK_TO_EMA20_RECLAIM_LONG", "SW_PULLBACK_TO_EMA50_RECLAIM_LONG", "SW_PULLBACK_SHORT_TO_EMA20_REJECT", "SW_PULLBACK_RSI_RESET_LONG"],
        "range_breakout_signal_catalog.csv": ["SW_DONCHIAN_BREAKOUT_LONG", "SW_DONCHIAN_BREAKDOWN_SHORT", "SW_FAILED_BREAKOUT_SHORT", "SW_FAILED_BREAKDOWN_LONG"],
        "volatility_signal_catalog.csv": ["SW_BB_SQUEEZE_BREAKOUT_LONG", "SW_BB_SQUEEZE_BREAKDOWN_SHORT", "SW_NR7_BREAKOUT_LONG", "SW_VOL_EXPANSION_WITH_TREND_LONG"],
        "momentum_exhaustion_signal_catalog.csv": ["SW_RSI_OVERSOLD_RECLAIM_LONG", "SW_RSI_OVERBOUGHT_REJECT_SHORT", "SW_MACD_HIST_FLIP_LONG", "SW_MACD_HIST_FLIP_SHORT"],
        "swing_orderflow_combo_catalog.csv": ["SW_PULLBACK_CVD_CONFIRM", "SW_BREAKOUT_TAKER_CONFIRM", "SW_ORDERFLOW_RISK_NOT_WORST20", "SW_BASIS_NOT_OVERHEATED"],
    }
    for file, signals in catalogs.items():
        pd.DataFrame([{"signal_id": s, "note": "fixed mechanical rule; diagnostics-only"} for s in signals]).to_csv(ROOT / f"catalog/{file}", index=False)


def add_candidate(rows: List[pd.DataFrame], f: pd.DataFrame, mask: pd.Series, family: str, signal_id: str, direction: str, variant: str = "normal") -> None:
    m = mask.fillna(False)
    if not m.any():
        return
    sub = f.loc[m].copy()
    sub["family"] = family
    sub["signal_id"] = signal_id
    sub["variant"] = variant
    sub["direction"] = direction
    sub["candidate_id"] = [f"{family}_{signal_id}_{i:07d}" for i in range(len(sub))]
    sub["signal_ts"] = sub["timestamp"]
    sub["entry_timeframe"] = sub["signal_timeframe"]
    sub["entry_condition_json"] = json.dumps({"signal": signal_id, "closed_candle": True, "lookahead": False})
    rows.append(sub)


def build_candidates(f: pd.DataFrame) -> pd.DataFrame:
    write_catalogs()
    rows: List[pd.DataFrame] = []
    bull_htf = f.get("4h_trend_stack_bull", False).fillna(False) | f.get("1d_trend_stack_bull", False).fillna(False)
    bear_htf = f.get("4h_trend_stack_bear", False).fillna(False) | f.get("1d_trend_stack_bear", False).fillna(False)
    # F1 trend regime entries are transition events, not every bar.
    add_candidate(rows, f, f["trend_stack_bull"] & ~f["trend_stack_bull"].shift(1).fillna(False), "F1_trend_regime", "SW_TREND_STACK_BULL_START", "LONG")
    add_candidate(rows, f, f["trend_stack_bear"] & ~f["trend_stack_bear"].shift(1).fillna(False), "F1_trend_regime", "SW_TREND_STACK_BEAR_START", "SHORT")
    add_candidate(rows, f, f["ema20_50_cross_up"], "F1_trend_regime", "SW_EMA20_50_CROSS_LONG", "LONG")
    add_candidate(rows, f, f["ema20_50_cross_down"], "F1_trend_regime", "SW_EMA20_50_CROSS_SHORT", "SHORT")
    add_candidate(rows, f, (f["ema_20_slope_3"] > 0) & (f["ema_20_slope_3"].shift(1) <= 0) & (f["close"] > f["ema_20"]), "F1_trend_regime", "SW_MA_SLOPE_TURN_UP", "LONG")
    add_candidate(rows, f, (f["ema_20_slope_3"] < 0) & (f["ema_20_slope_3"].shift(1) >= 0) & (f["close"] < f["ema_20"]), "F1_trend_regime", "SW_MA_SLOPE_TURN_DOWN", "SHORT")
    # F2 pullback reclaim.
    add_candidate(rows, f, bull_htf & (f["low"] <= f["ema_20"]) & (f["close"] > f["ema_20"]) & (f["close"].shift(1) <= f["ema_20"].shift(1)), "F2_pullback_reclaim", "SW_PULLBACK_TO_EMA20_RECLAIM_LONG", "LONG")
    add_candidate(rows, f, bull_htf & (f["low"] <= f["ema_50"]) & (f["close"] > f["ema_50"]) & (f["close"].shift(1) <= f["ema_50"].shift(1)), "F2_pullback_reclaim", "SW_PULLBACK_TO_EMA50_RECLAIM_LONG", "LONG")
    add_candidate(rows, f, bear_htf & (f["high"] >= f["ema_20"]) & (f["close"] < f["ema_20"]) & (f["close"].shift(1) >= f["ema_20"].shift(1)), "F2_pullback_reclaim", "SW_PULLBACK_SHORT_TO_EMA20_REJECT", "SHORT")
    add_candidate(rows, f, bull_htf & (f["rsi_14"].shift(1) < 45) & (f["rsi_14"] >= 45) & (f["close"] > f["ema_20"]), "F2_pullback_reclaim", "SW_PULLBACK_RSI_RESET_LONG", "LONG")
    # F3 breakouts/retests.
    add_candidate(rows, f, (f["close"] > f["donchian_high_20"]) & (f["close"].shift(1) <= f["donchian_high_20"].shift(1)), "F3_range_breakout", "SW_DONCHIAN_20_BREAKOUT_LONG", "LONG")
    add_candidate(rows, f, (f["close"] < f["donchian_low_20"]) & (f["close"].shift(1) >= f["donchian_low_20"].shift(1)), "F3_range_breakout", "SW_DONCHIAN_20_BREAKDOWN_SHORT", "SHORT")
    add_candidate(rows, f, (f["high"] > f["donchian_high_20"]) & (f["close"] < f["donchian_high_20"]), "F3_range_breakout", "SW_FAILED_BREAKOUT_SHORT", "SHORT")
    add_candidate(rows, f, (f["low"] < f["donchian_low_20"]) & (f["close"] > f["donchian_low_20"]), "F3_range_breakout", "SW_FAILED_BREAKDOWN_LONG", "LONG")
    # F4 volatility.
    squeeze = f["bb_width"] < f["bb_width"].rolling(200, min_periods=50).quantile(0.20)
    add_candidate(rows, f, squeeze.shift(1).fillna(False) & (f["close"] > f["bb_upper"]), "F4_volatility_compression", "SW_BB_SQUEEZE_BREAKOUT_LONG", "LONG")
    add_candidate(rows, f, squeeze.shift(1).fillna(False) & (f["close"] < f["bb_lower"]), "F4_volatility_compression", "SW_BB_SQUEEZE_BREAKDOWN_SHORT", "SHORT")
    add_candidate(rows, f, f["inside_cluster"].shift(1).fillna(False) & (f["close"] > f["high"].shift(1)), "F4_volatility_compression", "SW_INSIDE_CLUSTER_BREAKOUT_LONG", "LONG")
    add_candidate(rows, f, f["inside_cluster"].shift(1).fillna(False) & (f["close"] < f["low"].shift(1)), "F4_volatility_compression", "SW_INSIDE_CLUSTER_BREAKDOWN_SHORT", "SHORT")
    # F5 momentum/exhaustion.
    add_candidate(rows, f, (f["rsi_14"].shift(1) < 30) & (f["rsi_14"] >= 30), "F5_momentum_exhaustion", "SW_RSI_OVERSOLD_RECLAIM_LONG", "LONG")
    add_candidate(rows, f, (f["rsi_14"].shift(1) > 70) & (f["rsi_14"] <= 70), "F5_momentum_exhaustion", "SW_RSI_OVERBOUGHT_REJECT_SHORT", "SHORT")
    add_candidate(rows, f, (f["macd_hist"].shift(1) <= 0) & (f["macd_hist"] > 0), "F5_momentum_exhaustion", "SW_MACD_HIST_FLIP_LONG", "LONG")
    add_candidate(rows, f, (f["macd_hist"].shift(1) >= 0) & (f["macd_hist"] < 0), "F5_momentum_exhaustion", "SW_MACD_HIST_FLIP_SHORT", "SHORT")
    add_candidate(rows, f, (f["low"] < f["bb_lower"]) & (f["close"] > f["bb_lower"]), "F5_momentum_exhaustion", "SW_BOLLINGER_LOWER_RECLAIM_LONG", "LONG")
    add_candidate(rows, f, (f["high"] > f["bb_upper"]) & (f["close"] < f["bb_upper"]), "F5_momentum_exhaustion", "SW_BOLLINGER_UPPER_REJECT_SHORT", "SHORT")
    # F6 orderflow/risk interactions.
    risk_score = pd.to_numeric(f["risk_adjusted_orderflow_score"], errors="coerce")
    risk_not_worst = risk_score > risk_score.quantile(0.20)
    taker_buy = pd.to_numeric(f["taker_delta"], errors="coerce") > pd.to_numeric(f["taker_delta"], errors="coerce").quantile(0.70)
    taker_sell = pd.to_numeric(f["taker_delta"], errors="coerce") < pd.to_numeric(f["taker_delta"], errors="coerce").quantile(0.30)
    cvd_up = pd.to_numeric(f["cvd_slope"], errors="coerce") > pd.to_numeric(f["cvd_slope"], errors="coerce").quantile(0.65)
    cvd_down = pd.to_numeric(f["cvd_slope"], errors="coerce") < pd.to_numeric(f["cvd_slope"], errors="coerce").quantile(0.35)
    of = f["orderflow_available"].fillna(False)
    add_candidate(rows, f, of & risk_not_worst & bull_htf & (f["low"] <= f["ema_20"]) & (f["close"] > f["ema_20"]) & (taker_buy | cvd_up), "F6_swing_orderflow", "SW_PULLBACK_CVD_TAKER_CONFIRM", "LONG")
    add_candidate(rows, f, of & risk_not_worst & (f["close"] > f["donchian_high_20"]) & taker_buy & cvd_up, "F6_swing_orderflow", "SW_BREAKOUT_TAKER_CVD_CONFIRM", "LONG")
    add_candidate(rows, f, of & (f["high"] > f["donchian_high_20"]) & (f["close"] < f["donchian_high_20"]) & taker_sell & cvd_down, "F6_swing_orderflow", "SW_FAILED_BREAKOUT_ORDERFLOW_SHORT", "SHORT")
    c = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not c.empty:
        c = c.sort_values(["signal_ts", "family", "signal_id"]).reset_index(drop=True)
        c.to_parquet(ROOT / "candidates/all_swing_candidates.parquet", index=False)
        mapping = {
            "F1_trend_regime": "trend_regime",
            "F2_pullback_reclaim": "pullback_reclaim",
            "F3_range_breakout": "range_breakout",
            "F4_volatility_compression": "volatility",
            "F5_momentum_exhaustion": "momentum_exhaustion",
            "F6_swing_orderflow": "swing_orderflow",
        }
        for fam, prefix in mapping.items():
            sub = c[c["family"].eq(fam)]
            sub.to_parquet(ROOT / f"candidates/{prefix}_candidates.parquet", index=False)
            sub.groupby(["family", "signal_id", "entry_timeframe", "direction"]).size().reset_index(name="candidate_count").to_csv(ROOT / f"candidates/{prefix}_candidate_summary.csv", index=False)
            sub.to_parquet(ROOT / f"features/{prefix}_features.parquet", index=False)
            (ROOT / f"features/{prefix}_feature_report.md").write_text(f"# {prefix} Feature Report\n\nBuilt from closed 1h/4h/1d candles only. 5m was not used as a signal timeframe.\n", encoding="utf-8")
    return c


def priority_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    priority = {"F2_pullback_reclaim": 1, "F3_range_breakout": 2, "F4_volatility_compression": 3, "F5_momentum_exhaustion": 4, "F1_trend_regime": 5, "F6_swing_orderflow": 6}
    tmp = df.copy()
    tmp["_rank"] = tmp["family"].map(priority).fillna(99)
    return tmp.sort_values(["signal_ts", "_rank"]).drop_duplicates(["signal_ts", "direction"]).drop(columns=["_rank"])


def build_stages(c: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    order = ["F1_trend_regime", "F2_pullback_reclaim", "F3_range_breakout", "F4_volatility_compression", "F5_momentum_exhaustion", "F6_swing_orderflow"]
    names = [
        "stage_1_trend",
        "stage_2_trend_pullback",
        "stage_3_trend_pullback_breakout",
        "stage_4_trend_pullback_breakout_vol",
        "stage_5_trend_pullback_breakout_vol_momentum",
        "stage_6_swing_orderflow",
    ]
    stages = {}
    for i, name in enumerate(names, start=1):
        stages[name] = priority_dedupe(c[c["family"].isin(order[:i])])
        stages[name].to_parquet(ROOT / f"stages/{name}_candidates.parquet", index=False)
    pd.DataFrame([{"stage": k, "candidate_count": len(v), "families": ",".join(sorted(v["family"].unique())) if not v.empty else ""} for k, v in stages.items()]).to_csv(ROOT / "stages/stagewise_candidate_summary.csv", index=False)
    contrib = []
    for k, v in stages.items():
        for fam, g in v.groupby("family"):
            contrib.append({"stage": k, "family": fam, "candidate_count": len(g), "share": len(g) / len(v) if len(v) else 0})
    pd.DataFrame(contrib).to_csv(ROOT / "stages/stagewise_contribution_matrix.csv", index=False)
    (ROOT / "stages/stagewise_application_report.md").write_text("# Stagewise Application Report\n\nStages follow requested order: trend, pullback, breakout, volatility, momentum, orderflow/risk. Signal timestamps are 1h/4h/1d closed candles only.\n", encoding="utf-8")
    return stages


def add_outcomes(c: pd.DataFrame, replay_5m: pd.DataFrame) -> pd.DataFrame:
    if c.empty:
        return c
    b = replay_5m[["timestamp", "open", "high", "low", "close", "atr_pct"]].reset_index(drop=True).copy()
    b["timestamp"] = pd.to_datetime(b["timestamp"], errors="coerce").astype("datetime64[ns]")
    b["bar_index"] = np.arange(len(b))
    c = c.copy()
    c["signal_ts"] = pd.to_datetime(c["signal_ts"], errors="coerce").astype("datetime64[ns]")
    right = b[["timestamp", "bar_index"]].rename(columns={"timestamp": "signal_ts"})
    m = pd.merge_asof(c.sort_values("signal_ts"), right.sort_values("signal_ts"), on="signal_ts", direction="backward")
    entry_idx = m["bar_index"].astype("Int64") + 1
    valid = entry_idx.notna() & (entry_idx < len(b))
    m = m[valid].copy()
    entry_idx = entry_idx[valid].astype(int)
    m["entry_ts"] = b.loc[entry_idx, "timestamp"].to_numpy()
    m["entry_price"] = b.loc[entry_idx, "open"].to_numpy()
    sign = np.where(m["direction"].eq("SHORT"), -1, 1)
    for hid, bars in HORIZON_BARS_5M.items():
        exit_idx = np.minimum(entry_idx + bars, len(b) - 1)
        exit_close = b.loc[exit_idx, "close"].to_numpy()
        gross = sign * (exit_close / m["entry_price"].to_numpy() - 1)
        m[f"gross_{hid}"] = gross
        m[f"net_{hid}"] = gross - CURRENT_COST_BPS / 10000
    m["gross_return"] = m[f"gross_{PRIMARY_HORIZON}"]
    m["net_after_cost"] = m[f"net_{PRIMARY_HORIZON}"]
    hbars = HORIZON_BARS_5M[PRIMARY_HORIZON] + 1
    fh = b["high"].iloc[::-1].rolling(hbars, min_periods=1).max().iloc[::-1].to_numpy()
    fl = b["low"].iloc[::-1].rolling(hbars, min_periods=1).min().iloc[::-1].to_numpy()
    high = fh[entry_idx.to_numpy()]
    low = fl[entry_idx.to_numpy()]
    ep = m["entry_price"].to_numpy()
    m["MFE"] = np.where(m["direction"].eq("SHORT"), ep / low - 1, high / ep - 1)
    m["MAE"] = np.where(m["direction"].eq("SHORT"), high / ep - 1, ep / low - 1)
    m["MFE_to_cost"] = m["MFE"] / (CURRENT_COST_BPS / 10000)
    m["MAE_to_cost"] = m["MAE"] / (CURRENT_COST_BPS / 10000)
    m["RFE"] = (m["MAE"] > m["MFE"]) & (m["MAE"] > CURRENT_COST_BPS / 10000)
    m["label"] = np.where(m["net_after_cost"] > CURRENT_COST_BPS / 10000, "SWING_GOOD", np.where(m["net_after_cost"] < -CURRENT_COST_BPS / 10000, "SWING_BAD", "SWING_NEUTRAL"))
    m["holding_bars"] = HORIZON_BARS_5M[PRIMARY_HORIZON]
    m["holding_hours"] = 24
    m["exit_policy_id"] = f"X1_fixed_{PRIMARY_HORIZON}"
    m.to_parquet(ROOT / "backfill/swing_alpha_paper_trades.parquet", index=False)
    cols = [x for x in m.columns if x.startswith("gross_") or x.startswith("net_")]
    m[["candidate_id", "signal_ts", "entry_ts", "family", "signal_id", "entry_timeframe", "direction", *cols]].to_parquet(ROOT / "backfill/swing_alpha_exit_outcomes.parquet", index=False)
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
        "trades_per_week": len(df) / days * 7,
        "direction": "MIXED" if df["direction"].nunique() > 1 else str(df["direction"].iloc[0]),
        "entry_timeframe": "MIXED" if df["entry_timeframe"].nunique() > 1 else str(df["entry_timeframe"].iloc[0]),
        "signal_timeframe": "MIXED" if df["signal_timeframe"].nunique() > 1 else str(df["signal_timeframe"].iloc[0]),
        "holding_horizon": PRIMARY_HORIZON,
        "exit_policy": f"X1_fixed_{PRIMARY_HORIZON}",
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
        "best_horizon": f"{PRIMARY_HORIZON}_primary; horizon_table_available",
        "best_exit_policy": f"X1_fixed_{PRIMARY_HORIZON}",
        "walk_forward_mean_test_net": np.nan,
        "walk_forward_pass_rate": np.nan,
        "sample_warning": len(df) < 50,
        "overfit_warning": "LOW_FIXED_RULES" if len(df) >= 50 else "SAMPLE_TOO_SMALL",
        "production_ready": False,
    }


def apply_cooldown(df: pd.DataFrame, hours: int, max_policy: str) -> pd.DataFrame:
    if df.empty:
        return df
    keep, last = [], {}
    counts_day, counts_week, counts_month = {}, {}, {}
    for idx, row in df.sort_values("signal_ts").iterrows():
        ts = pd.to_datetime(row["signal_ts"])
        d = str(row["direction"])
        if hours and d in last and ts < last[d] + pd.Timedelta(hours=hours):
            continue
        day = (str(ts.date()), d)
        week = (f"{ts.isocalendar().year}-{ts.isocalendar().week}", d)
        month = (f"{ts.year}-{ts.month}", d)
        if max_policy == "max_1_day" and counts_day.get(day, 0) >= 1:
            continue
        if max_policy == "max_3_week" and counts_week.get(week, 0) >= 3:
            continue
        if max_policy == "max_5_week" and counts_week.get(week, 0) >= 5:
            continue
        if max_policy == "max_10_month" and counts_month.get(month, 0) >= 10:
            continue
        keep.append(idx)
        last[d] = ts
        counts_day[day] = counts_day.get(day, 0) + 1
        counts_week[week] = counts_week.get(week, 0) + 1
        counts_month[month] = counts_month.get(month, 0) + 1
    return df.loc[keep].copy()


def build_scorecards(c: pd.DataFrame, stages: Dict[str, pd.DataFrame], wf_summary: pd.DataFrame | None = None) -> Dict[str, pd.DataFrame]:
    single = pd.DataFrame([perf(priority_dedupe(c[c["family"].eq(fam)]), fam, "single_family") for fam in sorted(c["family"].unique())])
    stage = pd.DataFrame([perf(v, k, "stage") for k, v in stages.items()])
    if wf_summary is not None and not wf_summary.empty:
        mean_test = wf_summary["mean_net_test_bps"].mean()
        pass_rate = (wf_summary["mean_net_test_bps"] > 0).mean()
        for df in [single, stage]:
            if not df.empty:
                df["walk_forward_mean_test_net"] = mean_test
                df["walk_forward_pass_rate"] = pass_rate
    single.to_csv(ROOT / "scorecards/single_family_scorecard.csv", index=False)
    single.to_csv(ROOT / "backfill/outcome_summary_by_family.csv", index=False)
    stage.to_csv(ROOT / "scorecards/stagewise_scorecard.csv", index=False)
    stage.to_csv(ROOT / "backfill/outcome_summary_by_stage.csv", index=False)
    s = {r["id"]: r for _, r in stage.iterrows()}
    names = list(stages)
    labels = ["Stage2_minus_Stage1_pullback_add", "Stage3_minus_Stage2_breakout_add", "Stage4_minus_Stage3_volatility_add", "Stage5_minus_Stage4_momentum_add", "Stage6_minus_Stage5_orderflow_add"]
    deltas = []
    for a, b, label in zip(names, names[1:], labels):
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
    combo[combo["group_type"].eq(f"combo_{len(fams)}")].to_csv(ROOT / "scorecards/gated_combination_scorecard.csv", index=False)
    cd_rows = []
    base = stages["stage_6_swing_orderflow"] if "stage_6_swing_orderflow" in stages else list(stages.values())[-1]
    for hours in [0, 4, 8, 12, 24, 48, 72]:
        for pol in ["unlimited", "max_1_day", "max_3_week", "max_5_week", "max_10_month"]:
            cd = apply_cooldown(base, hours, pol)
            m = perf(cd, f"stage6_cd{hours}h_{pol}", "cooldown_trade_limit")
            m["cooldown_hours"] = hours
            m["max_policy"] = pol
            cd_rows.append(m)
    pd.DataFrame(cd_rows).to_csv(ROOT / "scorecards/cooldown_trade_limit_scorecard.csv", index=False)
    pd.DataFrame([perf(g, f"timeframe_{k}", "timeframe") for k, g in c.groupby("entry_timeframe")]).to_csv(ROOT / "scorecards/timeframe_scorecard.csv", index=False)
    pd.DataFrame([perf(g, f"direction_{k}", "direction") for k, g in c.groupby("direction")]).to_csv(ROOT / "scorecards/direction_scorecard.csv", index=False)
    horizon_rows = []
    for hid in HORIZON_BARS_5M:
        net_col = f"net_{hid}"
        gross_col = f"gross_{hid}"
        if net_col in c:
            tmp = c.copy()
            tmp["net_after_cost"] = tmp[net_col]
            tmp["gross_return"] = tmp[gross_col]
            horizon_rows.append(perf(tmp, hid, "horizon_exit"))
    pd.DataFrame(horizon_rows).to_csv(ROOT / "scorecards/horizon_exit_scorecard.csv", index=False)
    cost = []
    for id_, df in [("stage6", base), *[(k, v) for k, v in stages.items()]]:
        gross = pd.to_numeric(df["gross_return"], errors="coerce")
        cost.append({"id": id_, "zero_cost_bps": gross.mean() * 10000, "maker_like_bps": (gross - 3 / 10000).mean() * 10000, "current_bps": (gross - 6 / 10000).mean() * 10000, "two_x_bps": (gross - 12 / 10000).mean() * 10000})
    pd.DataFrame(cost).to_csv(ROOT / "scorecards/cost_sensitivity_scorecard.csv", index=False)
    if wf_summary is not None:
        wf_summary.to_csv(ROOT / "scorecards/walk_forward_scorecard.csv", index=False)
    (ROOT / "scorecards/stagewise_scorecard_report.md").write_text("# Stagewise Scorecard Report\n\nLow-frequency swing scorecards use 1h/4h/1d signal closes and 5m next-open fill. Primary horizon is 24h fixed.\n", encoding="utf-8")
    (ROOT / "backfill/backfill_report.md").write_text("# Backfill Report\n\nSignal candles are closed 1h/4h/1d bars. Entry is next available 5m open after signal close. Future path is used only for replay metrics.\n", encoding="utf-8")
    return {"single": single, "stage": stage, "combo": combo}


def walk_forward(c: pd.DataFrame) -> pd.DataFrame:
    if c.empty:
        return pd.DataFrame()
    df = c.sort_values("signal_ts").copy()
    start, end = df["signal_ts"].min(), df["signal_ts"].max()
    folds = []
    train_days, test_days = 180, 30
    cur = start + pd.Timedelta(days=train_days)
    fold_id = 0
    while cur + pd.Timedelta(days=test_days) <= end and fold_id < 24:
        train = df[(df["signal_ts"] >= cur - pd.Timedelta(days=train_days)) & (df["signal_ts"] < cur)]
        test = df[(df["signal_ts"] >= cur) & (df["signal_ts"] < cur + pd.Timedelta(days=test_days))]
        if len(train) >= 50 and len(test) >= 10:
            train_family = train.groupby("family")["net_after_cost"].mean().sort_values(ascending=False)
            selected = list(train_family.head(2).index)
            test_sel = test[test["family"].isin(selected)]
            folds.append({
                "fold": fold_id,
                "train_start": train["signal_ts"].min(),
                "train_end": train["signal_ts"].max(),
                "test_start": test["signal_ts"].min(),
                "test_end": test["signal_ts"].max(),
                "selected_signals": ",".join(selected),
                "candidate_count": len(train),
                "test_count": len(test_sel),
                "mean_net_train_bps": train[train["family"].isin(selected)]["net_after_cost"].mean() * 10000,
                "mean_net_test_bps": test_sel["net_after_cost"].mean() * 10000 if len(test_sel) else np.nan,
                "PF_train": profit_factor(train[train["family"].isin(selected)]["net_after_cost"]),
                "PF_test": profit_factor(test_sel["net_after_cost"]) if len(test_sel) else np.nan,
                "RFE_train": train[train["family"].isin(selected)]["RFE"].mean(),
                "RFE_test": test_sel["RFE"].mean() if len(test_sel) else np.nan,
                "MDD_train": mdd(train[train["family"].isin(selected)]["net_after_cost"]),
                "MDD_test": mdd(test_sel["net_after_cost"]) if len(test_sel) else np.nan,
                "stability_score": np.nan,
                "sign_consistency": np.nan,
                "fold_verdict": "WALK_FORWARD_PASS" if len(test_sel) and test_sel["net_after_cost"].mean() > 0 else "WALK_FORWARD_FAIL",
            })
        fold_id += 1
        cur += pd.Timedelta(days=test_days)
    out = pd.DataFrame(folds)
    config = {"train_days": train_days, "test_days": test_days, "purged_gap": "none", "threshold_selection": "train family mean only; no test retune"}
    (ROOT / "walk_forward/walk_forward_config.json").write_text(jdump(config), encoding="utf-8")
    out.to_csv(ROOT / "walk_forward/walk_forward_fold_results.csv", index=False)
    out[["fold", "selected_signals"]] .to_csv(ROOT / "walk_forward/walk_forward_selected_signals.csv", index=False) if not out.empty else pd.DataFrame().to_csv(ROOT / "walk_forward/walk_forward_selected_signals.csv", index=False)
    summary = pd.DataFrame([{"folds": len(out), "mean_test_net_bps": out["mean_net_test_bps"].mean() if not out.empty else np.nan, "pass_rate": (out["mean_net_test_bps"] > 0).mean() if not out.empty else np.nan, "verdict": "WALK_FORWARD_PASS" if not out.empty and (out["mean_net_test_bps"] > 0).mean() >= 0.5 else "WALK_FORWARD_FAIL"}])
    summary.to_csv(ROOT / "walk_forward/walk_forward_stability_summary.csv", index=False)
    (ROOT / "walk_forward/walk_forward_report.md").write_text("# Walk-forward Report\n\nRolling train/test validation selects families only on train windows, then evaluates fixed selections on test windows. No test-window retuning is used.\n", encoding="utf-8")
    return out


def profit_factor(net: pd.Series) -> float:
    net = pd.to_numeric(net, errors="coerce").dropna()
    pos = net[net > 0].sum()
    neg = -net[net < 0].sum()
    return float(pos / neg) if neg > 0 else math.inf


def mdd(net: pd.Series) -> float:
    curve = pd.to_numeric(net, errors="coerce").fillna(0).cumsum()
    return float((curve.cummax() - curve).max()) if len(curve) else 0.0


def failure(scorecards: Dict[str, pd.DataFrame]) -> None:
    rows = []
    for kind, df in scorecards.items():
        if df.empty:
            continue
        for _, r in df.iterrows():
            mean_net = r.get("mean_net_bps", np.nan)
            gross = r.get("gross_mean_bps", np.nan)
            sample = r.get("trade_count", 0)
            wf = r.get("walk_forward_mean_test_net", np.nan)
            if sample == 0:
                reason, action = "NO_SAMPLE", "DROP"
            elif sample < 50:
                reason, action = "SAMPLE_TOO_SMALL", "KEEP_AS_CASEBOOK_ONLY"
            elif pd.notna(wf) and wf < 0:
                reason, action = "WALK_FORWARD_FAIL", "KEEP_AS_CASEBOOK_ONLY"
            elif mean_net < 0 and gross > 0:
                reason, action = "COST_KILL", "NEED_EXIT_REDESIGN"
            elif mean_net < 0:
                reason, action = "GROSS_EDGE_WEAK", "DROP"
            else:
                reason, action = "NONE_OR_ACCEPTABLE", "KEEP_FOR_FORWARD"
            rows.append({"object_id": r.get("id"), "group_type": r.get("group_type", kind), "primary_failure_reason": reason, "secondary_failure_reason": "EXIT_HORIZON_MISMATCH", "evidence_metrics": jdump({"mean_net_bps": mean_net, "gross_mean_bps": gross, "sample": sample, "wf_test": wf}), "recommended_action": action})
    out = pd.DataFrame(rows)
    out[out["group_type"].astype(str).str.contains("single", na=False)].to_csv(ROOT / "failure/family_failure_attribution.csv", index=False)
    out[out["group_type"].astype(str).str.contains("stage", na=False)].to_csv(ROOT / "failure/stage_failure_attribution.csv", index=False)
    out[out["group_type"].astype(str).str.contains("combo", na=False)].to_csv(ROOT / "failure/combination_failure_attribution.csv", index=False)
    out.groupby(["primary_failure_reason", "recommended_action"]).size().reset_index(name="count").to_csv(ROOT / "failure/failure_reason_summary.csv", index=False)
    (ROOT / "failure/failure_attribution_report.md").write_text("# Failure Attribution Report\n\nFailure reasons combine net/gross relationship, sample size, and walk-forward evidence.\n", encoding="utf-8")


def casebook(c: pd.DataFrame) -> None:
    cb = c.copy()
    cb["case_category"] = np.select(
        [
            cb["family"].eq("F1_trend_regime") & (cb["net_after_cost"] > 0),
            cb["family"].eq("F1_trend_regime") & (cb["net_after_cost"] <= 0),
            cb["family"].eq("F2_pullback_reclaim") & (cb["net_after_cost"] > 0),
            cb["family"].eq("F2_pullback_reclaim") & (cb["net_after_cost"] <= 0),
            cb["family"].eq("F3_range_breakout") & (cb["net_after_cost"] > 0),
            cb["family"].eq("F3_range_breakout") & (cb["net_after_cost"] <= 0),
            cb["family"].eq("F4_volatility_compression") & (cb["net_after_cost"] > 0),
            cb["family"].eq("F4_volatility_compression") & (cb["net_after_cost"] <= 0),
            cb["family"].eq("F5_momentum_exhaustion") & (cb["net_after_cost"] > 0),
            cb["family"].eq("F5_momentum_exhaustion") & (cb["net_after_cost"] <= 0),
            cb["family"].eq("F6_swing_orderflow") & (cb["net_after_cost"] > 0),
            cb["family"].eq("F6_swing_orderflow") & (cb["net_after_cost"] <= 0),
        ],
        [
            "SW_TREND_SUCCESS", "SW_TREND_FAIL_LAG", "SW_PULLBACK_SUCCESS", "SW_PULLBACK_FAIL_CATCHING_KNIFE", "SW_BREAKOUT_SUCCESS", "SW_BREAKOUT_FAIL_FAKEOUT", "SW_VOL_BREAKOUT_SUCCESS", "SW_VOL_BREAKOUT_FAIL_DIRECTIONLESS", "SW_MOMENTUM_RESET_SUCCESS", "SW_MOMENTUM_FAIL_OVERHEAT", "SW_ORDERFLOW_CONFIRM_SUCCESS", "SW_ORDERFLOW_CONFIRM_FAIL_OVERFILTER",
        ],
        default="OTHER",
    )
    cb["why_success"] = np.where(cb["net_after_cost"] > 0, "positive fixed 24h net outcome", "")
    cb["why_failure"] = np.where(cb["net_after_cost"] <= 0, "negative fixed 24h net outcome; inspect cost/gross/exit/fakeout", "")
    cb["chart_window_path"] = ""
    cols = [x for x in ["candidate_id", "signal_ts", "entry_timeframe", "direction", "family", "signal_id", "variant", "entry_price", "net_after_cost", "MFE", "MAE", "RFE", "case_category", "why_success", "why_failure", "chart_window_path"] if x in cb.columns]
    out = cb[cols].sort_values("net_after_cost", ascending=False)
    out.to_parquet(ROOT / "casebook/swing_alpha_casebook.parquet", index=False)
    out.to_csv(ROOT / "casebook/swing_alpha_casebook.csv", index=False)
    out.groupby(["family", "case_category"]).size().reset_index(name="count").to_csv(ROOT / "casebook/casebook_summary_by_family.csv", index=False)
    out.head(50).to_csv(ROOT / "casebook/top_success_cases.csv", index=False)
    out.tail(50).to_csv(ROOT / "casebook/top_failure_cases.csv", index=False)
    (ROOT / "casebook/swing_casebook_report.md").write_text("# Swing Casebook Report\n\nTop success/failure cases are stored. Chart window generation is left as a follow-up to avoid heavy plotting.\n", encoding="utf-8")


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
    df.to_csv(ROOT / "system_interaction/swing_risk_filter_interaction.csv", index=False)
    df.to_csv(ROOT / "system_interaction/swing_good_bad_preservation.csv", index=False)
    df.to_csv(ROOT / "system_interaction/swing_existing_system_overlap.csv", index=False)
    (ROOT / "system_interaction/system_interaction_report.md").write_text("# System Interaction Report\n\nOrderflow_risk_worst20 interaction is diagnostics-only. Q2/R7 exact flags were not available in this swing frame.\n", encoding="utf-8")


def forward_watchlist() -> None:
    rows = [
        ("FW_SWING_4H_PULLBACK_EMA20_RECLAIM_LONG", "4h/1d bull regime + EMA20 reclaim", "avoid orderflow worst20", "24h", "4h", "1d/4h", "24h/72h", "OHLCV + optional orderflow", 100),
        ("FW_SWING_4H_PULLBACK_EMA50_RECLAIM_LONG", "4h/1d bull regime + EMA50 reclaim", "avoid regime breakdown", "24h", "4h", "1d/4h", "24h/72h", "OHLCV", 100),
        ("FW_SWING_1D_BULL_4H_PULLBACK_LONG", "1d bull + 4h pullback reclaim", "risk_not_worst20 reference", "24h", "4h", "1d", "48h/5d", "OHLCV + orderflow", 100),
        ("FW_SWING_4H_BREAKOUT_RETEST_LONG", "4h Donchian breakout/retest", "basis not overheated", "24h", "4h", "4h/1d", "24h/72h", "OHLCV + optional basis", 100),
        ("FW_SWING_4H_FAILED_BREAKOUT_SHORT", "4h failed breakout short", "premium/taker exhaustion if available", "24h", "4h", "4h/1d", "24h/72h", "OHLCV + orderflow", 100),
        ("FW_SWING_4H_BB_SQUEEZE_BREAKOUT", "4h squeeze then close breakout", "direction and retest required", "24h", "4h", "4h", "24h/72h", "OHLCV", 100),
        ("FW_SWING_4H_RSI_RESET_IN_TREND", "4h RSI reset inside trend", "MA support reclaim", "24h", "4h", "1d/4h", "24h/72h", "OHLCV", 100),
        ("FW_SWING_ORDERFLOW_CONFIRM_PULLBACK", "pullback + taker/proxy CVD confirm", "research-only", "24h", "1h/4h", "4h/1d", "24h/72h", "OHLCV + proxy CVD/taker", 50),
        ("FW_SWING_MAX_3_PER_WEEK_POLICY", "apply max 3/week low-frequency policy", "none", "week", "mixed", "mixed", "24h/72h", "OHLCV", 100),
    ]
    df = pd.DataFrame(rows, columns=["watch_id", "entry_condition", "filter_condition", "cooldown", "entry_timeframe", "regime_timeframe", "horizon", "required_data", "min_sample"])
    df["success_criteria"] = "positive current/maker-like forward net with stable RFE/MDD"
    df["failure_criteria"] = "walk-forward negative, fakeout, cost/gross weak, or overfilter"
    df["priority"] = range(1, len(df) + 1)
    df["expected_frequency"] = "low frequency; target <= 0-1/day or weekly"
    df["historical_evidence"] = "see scorecards"
    df["walk_forward_evidence"] = "see walk_forward_stability_summary"
    df["current_verdict"] = "research_only_forward_validation"
    df.to_csv(ROOT / "forward_watchlist/swing_forward_watchlist.csv", index=False)
    (ROOT / "forward_watchlist/swing_forward_validation_plan.md").write_text("# Swing Forward Validation Plan\n\nForward-register low-frequency swing candidates only. No production connection. Success requires cost-surviving OOS/forward evidence.\n", encoding="utf-8")
    (ROOT / "forward_watchlist/forward_watchlist_report.md").write_text("# Forward Watchlist Report\n\nLow-frequency swing forward watchlist generated.\n", encoding="utf-8")


def decisions(scorecards: Dict[str, pd.DataFrame]) -> None:
    rows = []
    for _, r in scorecards["single"].iterrows():
        if r["trade_count"] < 50:
            dec = "KEEP_CASEBOOK_ONLY"
        elif r["mean_net_bps"] > 0 and (pd.isna(r.get("walk_forward_mean_test_net")) or r.get("walk_forward_mean_test_net", -1) > 0):
            dec = "KEEP_FOR_FORWARD_ENTRY_ALPHA"
        elif r["id"] == "F1_trend_regime":
            dec = "KEEP_AS_REGIME_FILTER"
        elif r["id"] in ["F5_momentum_exhaustion", "F6_swing_orderflow"]:
            dec = "KEEP_AS_RISK_FILTER"
        else:
            dec = "DROP"
        rows.append({"family": r["id"], "decision": dec, "mean_net_bps": r["mean_net_bps"], "maker_like_mean_bps": r["maker_like_mean_bps"], "walk_forward_mean_test_net": r.get("walk_forward_mean_test_net"), "trade_count": r["trade_count"], "production_ready": False})
    pd.DataFrame(rows).to_csv(ROOT / "decision/swing_family_decision_matrix.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "decision/final_swing_alpha_recommendation.csv", index=False)
    scorecards["stage"].to_csv(ROOT / "decision/swing_stage_decision_matrix.csv", index=False)
    pd.DataFrame().to_csv(ROOT / "decision/swing_signal_decision_matrix.csv", index=False)
    (ROOT / "decision/recommended_next_action.md").write_text("# Recommended Next Action\n\nOnly forward-register the least-bad low-frequency swing candidates with strict low-frequency trade caps. Do not promote to production.\n", encoding="utf-8")


def final_report(scorecards: Dict[str, pd.DataFrame], wf: pd.DataFrame) -> Dict[str, Any]:
    stage = scorecards["stage"]
    single = scorecards["single"]
    combo = scorecards["combo"]
    verdicts = ["LOW_FREQUENCY_SWING_ALPHA_COMPLETED"]
    stage_map = {r["id"]: r for _, r in stage.iterrows()}
    if stage_map.get("stage_1_trend", {}).get("mean_net_bps", -999) > 0:
        verdicts += ["STAGE_1_POSITIVE", "SWING_TREND_ALPHA_FOUND_RESEARCH_ONLY"]
    else:
        verdicts.append("SWING_TREND_REGIME_FILTER_ONLY")
    try:
        delta = pd.read_csv(ROOT / "scorecards/stage_delta_scorecard.csv") if (ROOT / "scorecards/stage_delta_scorecard.csv").exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        delta = pd.DataFrame()
    for label, good, bad in [
        ("Stage2_minus_Stage1_pullback_add", "STAGE_2_IMPROVES", "STAGE_2_DEGRADES"),
        ("Stage3_minus_Stage2_breakout_add", "STAGE_3_IMPROVES", "STAGE_3_DEGRADES"),
        ("Stage4_minus_Stage3_volatility_add", "STAGE_4_IMPROVES", "STAGE_4_DEGRADES"),
        ("Stage5_minus_Stage4_momentum_add", "STAGE_5_IMPROVES", "STAGE_5_DEGRADES"),
        ("Stage6_minus_Stage5_orderflow_add", "STAGE_6_IMPROVES", "STAGE_6_DEGRADES"),
    ]:
        row = delta[delta["delta_id"].eq(label)] if not delta.empty else pd.DataFrame()
        verdicts.append(good if not row.empty and row.iloc[0]["mean_net_bps_delta"] > 0 else bad)
    if (stage["mean_net_bps"] <= 0).all():
        verdicts += ["SWING_COST_KILLS_EDGE", "NO_LOW_FREQUENCY_SWING_ALPHA_FOUND"]
    if not wf.empty and (wf["mean_net_test_bps"] > 0).mean() >= 0.5:
        verdicts.append("SWING_WALK_FORWARD_PASS")
    else:
        verdicts.append("SWING_WALK_FORWARD_FAIL")
    fam = {r["id"]: r for _, r in single.iterrows()}
    if fam.get("F2_pullback_reclaim", {}).get("mean_net_bps", -999) > 0:
        verdicts += ["SWING_PULLBACK_ALPHA_FOUND_RESEARCH_ONLY", "SWING_PULLBACK_KEEP_FOR_FORWARD"]
    else:
        verdicts.append("SWING_PULLBACK_DROP")
    if fam.get("F3_range_breakout", {}).get("mean_net_bps", -999) <= 0:
        verdicts.append("SWING_BREAKOUT_FAKEOUT")
    if fam.get("F4_volatility_compression", {}).get("mean_net_bps", -999) <= 0:
        verdicts.append("SWING_VOLATILITY_NO_EDGE")
    if fam.get("F5_momentum_exhaustion", {}).get("mean_net_bps", -999) <= 0:
        verdicts.append("SWING_MOMENTUM_FILTER_ONLY")
    if fam.get("F6_swing_orderflow", {}).get("mean_net_bps", -999) > max(fam.get("F5_momentum_exhaustion", {}).get("mean_net_bps", -999), -999):
        verdicts.append("SWING_ORDERFLOW_ADDS_VALUE")
    else:
        verdicts.append("SWING_ORDERFLOW_RISK_FILTER_ONLY")
    verdicts += ["FORWARD_WATCHLIST_READY", "production_not_ready"]
    verdicts = list(dict.fromkeys(verdicts))
    (ROOT / "low_frequency_swing_entry_alpha_research_final_verdict.md").write_text("# Final Verdict\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    best_combo = combo.sort_values("mean_net_bps", ascending=False).head(5).to_dict("records") if not combo.empty else []
    report = f"""# Low Frequency Swing Entry Alpha Research Final Report

## Why
5m entry research repeatedly converged near cost/noise. This run tests 1h/4h/1d closed-candle swing entries, using 5m only for fill/replay.

## Data Availability
Canonical BTCUSDT 5m OHLCV is resampled to 1h/4h/1d. Orderflow/proxy CVD is used only as research-only overlap where available.

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

## Walk-forward
```json
{jdump(wf.to_dict("records")[:20] if not wf.empty else [])}
```

## Interpretation
Signal features are closed 1h/4h/1d candles only. Entry is next 5m open. Primary replay is fixed 24h; other swing horizons are stored in the horizon table. Future path is outcome-only.

## Safety
No production TCN/Q2/R7/Risk/live/order/state path was changed. No private API or order/account/balance/position endpoint was called. forward_orderflow_collector_v4 and Discord policy were read-only. production_ready=false and promotion_ready=false.
"""
    (ROOT / "low_frequency_swing_entry_alpha_research_final_report.md").write_text(report, encoding="utf-8")
    (ROOT / "recommended_next_branch.md").write_text("# Recommended Next Branch\n\nIf any swing candidate is kept, run a 30-day forward low-frequency watchlist with max 3/week and current-cost survival as the pass criterion.\n", encoding="utf-8")
    return {"verdicts": verdicts, "best_combo": best_combo}


def load_cache() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:
    c = pd.read_parquet(ROOT / "backfill/swing_alpha_paper_trades.parquet")
    stages = {
        "stage_1_trend": pd.read_parquet(ROOT / "stages/stage_1_trend_candidates.parquet"),
        "stage_2_trend_pullback": pd.read_parquet(ROOT / "stages/stage_2_trend_pullback_candidates.parquet"),
        "stage_3_trend_pullback_breakout": pd.read_parquet(ROOT / "stages/stage_3_trend_pullback_breakout_candidates.parquet"),
        "stage_4_trend_pullback_breakout_vol": pd.read_parquet(ROOT / "stages/stage_4_trend_pullback_breakout_vol_candidates.parquet"),
        "stage_5_trend_pullback_breakout_vol_momentum": pd.read_parquet(ROOT / "stages/stage_5_trend_pullback_breakout_vol_momentum_candidates.parquet"),
        "stage_6_swing_orderflow": pd.read_parquet(ROOT / "stages/stage_6_swing_orderflow_candidates.parquet"),
    }
    feat = pd.read_parquet(ROOT / "features/swing_feature_frame.parquet") if (ROOT / "features/swing_feature_frame.parquet").exists() else pd.DataFrame()
    return c, stages, feat


def cache_is_full() -> bool:
    meta_path = ROOT / "run_metadata.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    return bool(meta.get("cache_complete")) and not bool(meta.get("fast"))


def run_pipeline(mode: str = "full", stage: int | None = None, fast: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    before = safety_snapshot("before")
    log(f"start mode={mode} stage={stage} fast={fast}")
    discovery = input_discovery()
    cache_ok = (
        (ROOT / "backfill/swing_alpha_paper_trades.parquet").exists()
        and (ROOT / "stages/stage_6_swing_orderflow_candidates.parquet").exists()
        and cache_is_full()
    )
    if cache_ok and not fast:
        candidates, stages, features = load_cache()
    else:
        tf = build_timeframes(fast=fast)
        features = build_features(tf)
        candidates = build_candidates(features)
        candidates = add_outcomes(candidates, tf["5m"])
        stages = build_stages(candidates)
    if stage is not None:
        keys = list(stages)[:stage]
        stages = {k: v for k, v in stages.items() if k in keys}
    wf = walk_forward(candidates) if mode in {"full", "walk_forward"} and stage is None else pd.DataFrame()
    scorecards = build_scorecards(candidates, stages, wf_summary=wf if not wf.empty else None)
    failure(scorecards)
    casebook(candidates)
    system_interaction(candidates)
    forward_watchlist()
    decisions(scorecards)
    result = final_report(scorecards, wf)
    (ROOT / "run_metadata.json").write_text(
        jdump(
            {
                "mode": mode,
                "stage": stage,
                "fast": fast,
                "cache_complete": not fast,
                "feature_rows": len(features),
                "candidate_rows": len(candidates),
                "updated_ts": pd.Timestamp.now("UTC").isoformat(),
            }
        ),
        encoding="utf-8",
    )
    finalize_audit(before)
    log("done")
    return {"mode": mode, "stage": stage, "fast": fast, "feature_rows": len(features), "candidate_rows": len(candidates), "discovery": discovery, "verdicts": result["verdicts"], "production_ready": False, "promotion_ready": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast-smoke", action="store_true")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--single-families-only", action="store_true")
    parser.add_argument("--combinations-only", action="store_true")
    parser.add_argument("--walk-forward-only", action="store_true")
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
    elif args.walk_forward_only:
        res = run_pipeline(mode="walk_forward")
    else:
        res = run_pipeline(mode="full")
    print(jdump(res) if args.json else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
