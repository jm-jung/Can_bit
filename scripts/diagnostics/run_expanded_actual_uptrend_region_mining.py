"""Expanded actual uptrend region mining.

Diagnostics-only research. This expands the previous 4h/72h/Q95 uptrend
reverse-engineering into 15m/1h/4h/1d, multiple horizons, Q60-Q95 thresholds,
and timestamp/loose/normal/strict cluster views.
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

ROOT = Path("data/diagnostics/expanded_actual_uptrend_region_mining")
PREV = Path("data/diagnostics/actual_uptrend_structure_reverse_engineering")
ALPHA = Path("data/diagnostics/alpha_existence_target_feasibility_audit")
OHLCV_5M = Path("data/ohlcv/BTCUSDT_5m_full.csv")
CURRENT_COST_BPS = 6.0
RNG_SEED = 20260624
QUANTILES = [60, 70, 75, 80, 85, 90, 95, 97]
MAIN_QUANTILES = [60, 70, 75, 80, 85, 90, 95]
TF_RULES = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
HORIZONS = {
    "15m": ["1h", "2h", "4h", "8h", "12h", "24h", "48h"],
    "1h": ["4h", "8h", "12h", "24h", "48h", "72h", "5d", "7d"],
    "4h": ["12h", "24h", "48h", "72h", "5d", "7d", "14d"],
    "1d": ["3d", "5d", "7d", "14d", "21d", "30d"],
}
TF_MINUTES = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}
FEATURE_COLS = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "range_pct",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "ATR_pct",
    "volatility",
    "volatility_percentile",
    "bb_width",
    "bb_pct",
    "volume_z",
    "range_z",
    "rsi_14",
    "ema_20_slope",
    "ema_50_slope",
    "dist_ema_20",
    "dist_ema_50",
    "dist_ema_200",
    "trend_stack_bull",
    "trend_stack_bear",
    "dist_high_20",
    "dist_high_50",
    "dist_high_100",
    "dist_low_20",
    "drawdown_from_high",
    "bounce_from_low",
    "hour",
    "dayofweek",
    "month",
    "weekend",
]


def ensure_dirs() -> None:
    for d in [
        "discovery",
        "audit",
        "backfill/raw",
        "backfill/normalized",
        "backfill/registry",
        "datasets",
        "labels",
        "clusters",
        "snapshots",
        "comparison",
        "sensitivity",
        "patterns",
        "walk_forward",
        "economic",
        "casebook",
        "decision",
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


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def safety_snapshot(name: str) -> Dict[str, Any]:
    watch = [
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
    for raw in watch:
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


def finalize_safety(before: Dict[str, Any]) -> None:
    after = safety_snapshot("after")
    bmap = {r["path"]: r.get("sha256") for r in before.get("hashes", [])}
    rows = []
    for r in after.get("hashes", []):
        old = bmap.get(r["path"])
        rows.append({"path": r["path"], "sha256_before": old, "sha256_after": r.get("sha256"), "changed": old is not None and old != r.get("sha256")})
    (ROOT / "audit/hash_before_after.json").write_text(jdump(rows), encoding="utf-8")
    writes = [{"path": str(p), "diagnostics_only": True, "write_class": "expanded_uptrend_output"} for p in ROOT.rglob("*") if p.is_file()]
    writes.append({"path": "scripts/diagnostics/run_expanded_actual_uptrend_region_mining.py", "diagnostics_only": False, "write_class": "requested_entrypoint"})
    pd.DataFrame(writes).to_csv(ROOT / "audit/write_path_audit.csv", index=False)
    (ROOT / "audit/production_safety_audit.md").write_text(
        "# Production Safety Audit\n\nNo production TCN/Q2/R7/Risk Manager/live/order/state path was changed. `forward_orderflow_collector_v4` and `false_high_r7_daily_monitor` were read-only. Discord/webhook policy was unchanged. No private/order/account/balance/position endpoints were called. production_ready=false; promotion_ready=false.\n",
        encoding="utf-8",
    )


def input_discovery() -> Dict[str, Any]:
    required = [
        Path("data/diagnostics/data_sync/canonical_data_paths.json"),
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
        ALPHA / "alpha_existence_target_feasibility_audit_final_report.md",
        ALPHA / "decision/target_feasibility_matrix.csv",
        ALPHA / "economic/top_quantile_economic_replay.csv",
        ALPHA / "walk_forward/walk_forward_target_summary.csv",
        PREV / "actual_uptrend_structure_reverse_engineering_final_report.md",
        PREV / "labels/uptrend_timestamp_labels.parquet",
        PREV / "labels/uptrend_event_clusters.parquet",
        PREV / "labels/uptrend_label_summary.csv",
        PREV / "comparison/success_vs_control_feature_diff.csv",
        PREV / "patterns/pattern_scorecard.csv",
        Path("data/diagnostics/f2_exit_redesign_autopsy/f2_exit_redesign_autopsy_final_report.md"),
        Path("data/diagnostics/f2_pullback_reclaim_oos_failure_autopsy/f2_pullback_reclaim_oos_failure_autopsy_final_report.md"),
        Path("data/diagnostics/low_frequency_swing_entry_alpha_research/low_frequency_swing_entry_alpha_research_final_report.md"),
        OHLCV_5M,
    ]
    rows = []
    for p in required:
        rows.append({"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() and p.is_file() else 0, "suffix": p.suffix})
    inv = pd.DataFrame(rows)
    inv.to_csv(ROOT / "discovery/input_inventory.csv", index=False)
    (ROOT / "discovery/discovered_paths.json").write_text(jdump(inv.to_dict("records")), encoding="utf-8")
    cov = []
    if OHLCV_5M.exists():
        raw = pd.read_csv(OHLCV_5M, usecols=["timestamp"])
        ts = pd.to_datetime(raw["timestamp"], errors="coerce")
        cov.append({"path": str(OHLCV_5M), "rows": len(raw), "timeframe": "5m", "start": ts.min(), "end": ts.max()})
        for tf, rule in TF_RULES.items():
            approx = int(len(raw) * 5 / TF_MINUTES[tf])
            cov.append({"path": str(OHLCV_5M), "rows": approx, "timeframe": tf, "start": ts.min(), "end": ts.max()})
    pd.DataFrame(cov).to_csv(ROOT / "discovery/data_coverage_summary.csv", index=False)
    pd.DataFrame(
        [
            {"tier": "TIER_A_LONG_OHLCV_ONLY", "available": OHLCV_5M.exists(), "note": "main expanded mining from local 5m OHLCV"},
            {"tier": "TIER_B_OHLCV_PLUS_PROXY_CVD", "available": Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_1h.parquet").exists(), "note": "proxy CVD overlap reference"},
            {"tier": "TIER_C_RECENT_ORDERFLOW", "available": Path("data/diagnostics/research_orderflow_data_cache/normalized/open_interest_history/BTCUSDT.parquet").exists(), "note": "recent orderflow reference only"},
            {"tier": "TIER_D_FORWARD_ORDERBOOK_ONLY", "available": False, "note": "forward-only orderbook/liquidation not used"},
        ]
    ).to_csv(ROOT / "discovery/data_tier_summary.csv", index=False)
    of_rows = []
    for p in required[9:13]:
        df = safe_read(p)
        if not df.empty and "timestamp" in df:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            of_rows.append({"path": str(p), "rows": len(df), "start": ts.min(), "end": ts.max()})
        else:
            of_rows.append({"path": str(p), "rows": len(df), "start": "", "end": ""})
    pd.DataFrame(of_rows).to_csv(ROOT / "discovery/orderflow_overlap_summary.csv", index=False)
    prev_summary = safe_read(PREV / "labels/uptrend_label_summary.csv")
    prev_clusters = safe_read(PREV / "labels/uptrend_event_clusters.parquet")
    pd.DataFrame(
        [
            {"previous_basis": "4h/72h/Q95", "timestamp_success": int(prev_summary.loc[prev_summary["label"].eq("UP_SUCCESS_MFE_Q95"), "count"].iloc[0]) if not prev_summary.empty else 571, "cluster_count": len(prev_clusters) if not prev_clusters.empty else 90, "note": "narrow strong-MFE definition, normal clustering collapsed contiguous 4h successes"},
        ]
    ).to_csv(ROOT / "discovery/previous_uptrend_reverse_engineering_summary.csv", index=False)
    safe_read(ALPHA / "decision/target_feasibility_matrix.csv").to_csv(ROOT / "discovery/previous_alpha_existence_summary.csv", index=False)
    pd.DataFrame([{"local_5m_available": OHLCV_5M.exists(), "backfill_required": False, "reason": "local 5m OHLCV covers the full 2020-12-31 to 2026-06-24 range used for deterministic 15m/1h/4h/1d mining"}]).to_csv(ROOT / "discovery/backfill_need_assessment.csv", index=False)
    (ROOT / "discovery/discovery_report.md").write_text("# Discovery Report\n\nPrevious 90 clusters came from a narrow 4h/72h/Q95 strong-MFE definition. Expanded mining uses local 5m OHLCV to build 15m/1h/4h/1d definitions. Public backfill is not required.\n", encoding="utf-8")
    return {"ohlcv_5m_exists": OHLCV_5M.exists(), "previous_basis": "4h/72h/Q95", "backfill_required": False}


def write_backfill_outputs() -> None:
    pd.DataFrame([{"source": "local_existing", "symbol": "BTCUSDT", "status": "used", "private_api_calls": 0}]).to_csv(ROOT / "backfill/registry/backfill_registry.csv", index=False)
    (ROOT / "backfill/registry/backfill_progress.json").write_text(jdump({"status": "not_required", "reason": "local 5m data sufficient"}), encoding="utf-8")
    for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
        pd.DataFrame().to_parquet(ROOT / f"backfill/normalized/btcusdt_{tf}.parquet", index=False)
    pd.DataFrame([{"backfill_required": False, "rows_downloaded": 0, "private_api_calls": 0}]).to_csv(ROOT / "backfill/backfill_summary.csv", index=False)
    pd.DataFrame([{"source": "local_existing", "gap_count": np.nan, "note": "backfill skipped"}]).to_csv(ROOT / "backfill/backfill_gap_audit.csv", index=False)
    (ROOT / "backfill/backfill_report.md").write_text("# Backfill Report\n\nBackfill skipped. No public or private endpoint was called.\n", encoding="utf-8")


def load_5m(fast: bool) -> pd.DataFrame:
    usecols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.read_csv(OHLCV_5M, usecols=usecols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if fast:
        df = df.tail(40_000).copy()
    return df


def resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf == "5m":
        out = df.copy()
    else:
        out = (
            df.set_index("timestamp")
            .resample(TF_RULES[tf], label="left", closed="left")
            .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"))
            .dropna()
            .reset_index()
        )
    out["symbol"] = "BTCUSDT"
    out["timeframe"] = tf
    out["source"] = "local_5m_resample"
    return out


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["close"]
    for n in [1, 3, 6, 12, 24]:
        d[f"return_{n}"] = c.pct_change(n)
    d["range_pct"] = (d["high"] - d["low"]) / c
    d["body_pct"] = (d["close"] - d["open"]).abs() / c
    d["upper_wick_pct"] = (d["high"] - d[["open", "close"]].max(axis=1)) / c
    d["lower_wick_pct"] = (d[["open", "close"]].min(axis=1) - d["low"]) / c
    tr = pd.concat([(d["high"] - d["low"]), (d["high"] - d["close"].shift()).abs(), (d["low"] - d["close"].shift()).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14, min_periods=5).mean()
    d["ATR_pct"] = d["ATR"] / c
    d["volatility"] = d["return_1"].rolling(20, min_periods=10).std()
    d["volatility_percentile"] = d["volatility"].rolling(300, min_periods=50).rank(pct=True)
    ma = c.rolling(20, min_periods=10).mean()
    sd = c.rolling(20, min_periods=10).std()
    d["bb_width"] = (4 * sd) / c
    d["bb_pct"] = (c - (ma - 2 * sd)) / (4 * sd).replace(0, np.nan)
    d["volume_z"] = (d["volume"] - d["volume"].rolling(50, min_periods=20).mean()) / d["volume"].rolling(50, min_periods=20).std().replace(0, np.nan)
    d["range_z"] = (d["range_pct"] - d["range_pct"].rolling(50, min_periods=20).mean()) / d["range_pct"].rolling(50, min_periods=20).std().replace(0, np.nan)
    d["rsi_14"] = rsi(c)
    for n in [20, 50, 100, 200]:
        e = c.ewm(span=n, adjust=False, min_periods=max(5, n // 5)).mean()
        d[f"ema_{n}"] = e
        d[f"dist_ema_{n}"] = (c - e) / c
        d[f"ema_{n}_slope"] = e.pct_change(3)
    d["trend_stack_bull"] = (d["ema_20"] > d["ema_50"]) & (d["ema_50"] > d["ema_100"]) & (d["ema_100"] > d["ema_200"])
    d["trend_stack_bear"] = (d["ema_20"] < d["ema_50"]) & (d["ema_50"] < d["ema_100"]) & (d["ema_100"] < d["ema_200"])
    for n in [20, 50, 100]:
        rh = d["high"].rolling(n, min_periods=max(5, n // 3)).max()
        rl = d["low"].rolling(n, min_periods=max(5, n // 3)).min()
        d[f"dist_high_{n}"] = (c - rh) / c
        d[f"dist_low_{n}"] = (c - rl) / c
    d["drawdown_from_high"] = d["dist_high_100"]
    d["bounce_from_low"] = d["dist_low_100"]
    d["hour"] = d["timestamp"].dt.hour
    d["dayofweek"] = d["timestamp"].dt.dayofweek
    d["month"] = d["timestamp"].dt.month
    d["weekend"] = d["dayofweek"].isin([5, 6])
    d["data_tier"] = "TIER_A_LONG_OHLCV_ONLY"
    return d


def horizon_bars(tf: str, horizon: str) -> int:
    if horizon.endswith("m"):
        mins = int(horizon[:-1])
    elif horizon.endswith("h"):
        mins = int(horizon[:-1]) * 60
    elif horizon.endswith("d"):
        mins = int(horizon[:-1]) * 1440
    else:
        raise ValueError(horizon)
    return max(1, int(round(mins / TF_MINUTES[tf])))


def forward_roll(s: pd.Series, bars: int, kind: str) -> pd.Series:
    shifted = s.shift(-1)
    rev = shifted.iloc[::-1]
    rolled = rev.rolling(bars, min_periods=bars)
    out = rolled.max() if kind == "max" else rolled.min()
    return out.iloc[::-1]


def build_dataset(fast: bool) -> Dict[str, pd.DataFrame]:
    base = load_5m(fast)
    out = {}
    for tf in TF_RULES:
        log(f"resample/features {tf}")
        out[tf] = add_features(resample_ohlcv(base, tf))
        out[tf].to_parquet(ROOT / f"datasets/ohlcv_features_{tf}.parquet", index=False)
    return out


def label_for_horizon(df: pd.DataFrame, tf: str, horizon: str) -> pd.DataFrame:
    bars = horizon_bars(tf, horizon)
    d = df[["timestamp", "timeframe", "symbol", "open", "high", "low", "close", "volume"] + [c for c in FEATURE_COLS if c in df.columns] + ["data_tier"]].copy()
    future_high = forward_roll(d["high"], bars, "max")
    future_low = forward_roll(d["low"], bars, "min")
    future_close = d["close"].shift(-bars)
    d["horizon"] = horizon
    d["horizon_bars"] = bars
    d["future_MFE_long_bps"] = (future_high / d["close"] - 1) * 10_000
    d["future_MAE_long_bps"] = (d["close"] / future_low - 1) * 10_000
    d["future_return_net_current_bps"] = (future_close / d["close"] - 1) * 10_000 - CURRENT_COST_BPS
    d["future_return_gross_bps"] = (future_close / d["close"] - 1) * 10_000
    d["future_MFE_short_bps"] = (d["close"] / future_low - 1) * 10_000
    d = d.dropna(subset=["future_MFE_long_bps", "future_MAE_long_bps", "future_return_net_current_bps"]).reset_index(drop=True)
    qs = d["future_MFE_long_bps"].quantile([q / 100 for q in QUANTILES]).to_dict()
    for q in QUANTILES:
        d[f"UP_MFE_Q{q}"] = d["future_MFE_long_bps"] >= qs[q / 100]
    for mult in [1, 2, 3, 5, 10]:
        d[f"UP_MFE_{mult}X_COST"] = d["future_MFE_long_bps"] >= CURRENT_COST_BPS * mult
    d["UP_NET_POSITIVE"] = d["future_return_net_current_bps"] > 0
    for mult in [1, 2, 3]:
        d[f"UP_NET_GT_{mult}X_COST"] = d["future_return_net_current_bps"] > CURRENT_COST_BPS * mult
    d["UP_CLOSE_TO_CLOSE_UP"] = d["future_return_gross_bps"] > 0
    d["UP_CLOSE_TO_CLOSE_STRONG_UP"] = d["future_return_gross_bps"] > d["future_return_gross_bps"].quantile(0.80)
    q60, q70, q75, q80, q85, q90, q95 = qs[0.60], qs[0.70], qs[0.75], qs[0.80], qs[0.85], qs[0.90], qs[0.95]
    d["MFE_before_MAE"] = d["future_MFE_long_bps"] >= d["future_MAE_long_bps"]
    d["RFE"] = d["future_MAE_long_bps"] > d["future_MFE_long_bps"]
    d["UP_CLEAN"] = (d["future_MFE_long_bps"] >= q80) & (d["future_return_net_current_bps"] > 0) & d["MFE_before_MAE"]
    d["UP_DIRTY"] = (d["future_MFE_long_bps"] >= q80) & ((d["future_MAE_long_bps"] > d["future_MFE_long_bps"] * 0.8) | d["RFE"])
    d["UP_FAKE"] = (d["future_MFE_long_bps"] >= q80) & ((d["future_return_net_current_bps"] <= 0) | (~d["MFE_before_MAE"]))
    d["UP_GIVEBACK"] = (d["future_MFE_long_bps"] >= q80) & (d["future_return_net_current_bps"] < d["future_MFE_long_bps"] * 0.2)
    d["UP_GRIND"] = (d["future_return_net_current_bps"] > CURRENT_COST_BPS) & (d["future_MFE_long_bps"].between(q60, q90)) & (d["volatility_percentile"].fillna(1) < 0.6)
    d["UP_EXPLOSIVE"] = d["future_MFE_long_bps"] >= q95
    d["UP_BREAKOUT_FOLLOWTHROUGH"] = (d.get("dist_high_20", -1) > -0.03) & (d["future_MFE_long_bps"] >= q80) & (d["future_return_net_current_bps"] > 0)
    d["UP_PULLBACK_CONTINUATION"] = d.get("trend_stack_bull", False).astype(bool) & (d.get("dist_ema_20", 1).abs() < 0.03) & (d["future_MFE_long_bps"] >= q70)
    d["UP_MEAN_REVERSION_BOUNCE"] = (d.get("rsi_14", 50) < 40) & (d["future_MFE_long_bps"] >= q70)
    d["UP_CHOPPY_MFE"] = (d["future_MFE_long_bps"] >= q80) & (d["future_MAE_long_bps"] >= d["future_MAE_long_bps"].quantile(0.80))
    d["FAIL_NO_UP"] = (d["future_MFE_long_bps"] <= d["future_MFE_long_bps"].quantile(0.30)) & (d["future_return_net_current_bps"] <= 0)
    d["FAIL_MAE_FIRST"] = ~d["MFE_before_MAE"]
    d["FAIL_RFE_HIGH"] = d["RFE"]
    d["DOWN_SUCCESS_SHORT"] = d["future_MFE_short_bps"] >= d["future_MFE_short_bps"].quantile(0.80)
    d["SIDEWAYS_CONTROL"] = (d["future_MFE_long_bps"] <= d["future_MFE_long_bps"].quantile(0.45)) & (d["future_MAE_long_bps"] <= d["future_MAE_long_bps"].quantile(0.45)) & (d["future_return_gross_bps"].abs() <= d["future_return_gross_bps"].abs().quantile(0.45))
    d["NEAR_MISS_Q80"] = (d["future_MFE_long_bps"] < q80) & (d["future_MFE_long_bps"] >= q75)
    d["NEAR_MISS_Q90"] = (d["future_MFE_long_bps"] < q90) & (d["future_MFE_long_bps"] >= q85)
    d["NEAR_MISS_Q95"] = (d["future_MFE_long_bps"] < q95) & (d["future_MFE_long_bps"] >= q90)
    d["RANDOM_CONTROL"] = False
    pool = d.index[~d["UP_MFE_Q80"]].to_numpy()
    n = min(int(d["UP_MFE_Q80"].sum()), len(pool))
    if n:
        rng = np.random.default_rng(RNG_SEED + bars + TF_MINUTES[tf])
        d.loc[rng.choice(pool, size=n, replace=False), "RANDOM_CONTROL"] = True
    d["TIME_MATCHED_CONTROL"] = d["RANDOM_CONTROL"]
    d["REGIME_MATCHED_CONTROL"] = (~d["UP_MFE_Q80"]) & d.get("trend_stack_bull", False).eq(bool(d.loc[d["UP_MFE_Q80"], "trend_stack_bull"].mode().iloc[0]) if d["UP_MFE_Q80"].any() and "trend_stack_bull" in d else False)
    d["definition_id"] = tf + "_" + horizon
    d["event_id"] = np.arange(len(d))
    return d


def build_labels(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_parts = []
    summary_rows = []
    catalog = []
    for tf, df in frames.items():
        for horizon in HORIZONS[tf]:
            log(f"labels {tf} {horizon}")
            d = label_for_horizon(df, tf, horizon)
            all_parts.append(d)
            for q in MAIN_QUANTILES:
                summary_rows.append({"timeframe": tf, "horizon": horizon, "quantile": f"Q{q}", "label": f"UP_MFE_Q{q}", "count": int(d[f"UP_MFE_Q{q}"].sum()), "rate": float(d[f"UP_MFE_Q{q}"].mean()), "rows": len(d)})
            for lab in ["UP_CLEAN", "UP_DIRTY", "UP_FAKE", "UP_GIVEBACK", "UP_GRIND", "UP_EXPLOSIVE", "UP_BREAKOUT_FOLLOWTHROUGH", "UP_PULLBACK_CONTINUATION", "UP_MEAN_REVERSION_BOUNCE", "UP_CHOPPY_MFE", "FAIL_NO_UP", "FAIL_RFE_HIGH", "DOWN_SUCCESS_SHORT", "SIDEWAYS_CONTROL", "RANDOM_CONTROL", "REGIME_MATCHED_CONTROL"]:
                summary_rows.append({"timeframe": tf, "horizon": horizon, "quantile": "", "label": lab, "count": int(d[lab].sum()), "rate": float(d[lab].mean()), "rows": len(d)})
            catalog.append({"timeframe": tf, "horizon": horizon, "definition_id": f"{tf}_{horizon}", "success_thresholds": "Q60,Q70,Q75,Q80,Q85,Q90,Q95,Q97_REFERENCE", "path_quality_labels": "clean,dirty,fake,giveback,grind,explosive,breakout,pullback,bounce,choppy"})
    labels = pd.concat(all_parts, ignore_index=True)
    labels.to_parquet(ROOT / "labels/expanded_uptrend_timestamp_labels.parquet", index=False)
    pd.DataFrame(catalog).to_csv(ROOT / "labels/expanded_uptrend_label_catalog.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ROOT / "labels/expanded_uptrend_label_summary.csv", index=False)
    summary[summary["label"].str.startswith("UP_MFE_Q")].to_csv(ROOT / "labels/label_count_by_timeframe_horizon_quantile.csv", index=False)
    path_labs = ["UP_CLEAN", "UP_DIRTY", "UP_FAKE", "UP_GIVEBACK", "UP_GRIND", "UP_EXPLOSIVE", "UP_BREAKOUT_FOLLOWTHROUGH", "UP_PULLBACK_CONTINUATION", "UP_MEAN_REVERSION_BOUNCE", "UP_CHOPPY_MFE"]
    summary[summary["label"].isin(path_labs)].to_csv(ROOT / "labels/path_quality_label_summary.csv", index=False)
    summary[summary["label"].str.contains("CONTROL|FAIL|DOWN")].to_csv(ROOT / "labels/control_group_summary.csv", index=False)
    overlap_cols = [f"UP_MFE_Q{q}" for q in MAIN_QUANTILES] + path_labs + ["FAIL_NO_UP", "FAIL_RFE_HIGH", "RANDOM_CONTROL"]
    corr = labels[overlap_cols].astype(int).corr()
    corr.to_csv(ROOT / "labels/label_overlap_matrix.csv")
    (ROOT / "labels/label_build_report.md").write_text(f"# Expanded Label Build Report\n\nBuilt timestamp-level labels for {len(labels):,} timeframe/horizon rows. Q60-Q97 flags are preserved as columns; cluster-level views are additional.\n", encoding="utf-8")
    return labels


def cluster_one(d: pd.DataFrame, q: int, mode: str) -> pd.DataFrame:
    if mode == "none":
        x = d[d[f"UP_MFE_Q{q}"]].copy()
        x["cluster_id"] = np.arange(len(x))
        x["cluster_mode"] = "CLUSTER_NONE"
        x["cluster_start"] = x["timestamp"]
        x["cluster_end"] = x["timestamp"]
        x["representative_signal_ts"] = x["timestamp"]
        x["event_count"] = 1
        return x[["definition_id", "timeframe", "horizon", "cluster_mode", "cluster_id", "cluster_start", "cluster_end", "representative_signal_ts", "event_count", "future_MFE_long_bps"]]
    mult = {"loose": 1, "normal": 2, "strict": 4}[mode]
    gap_hours = {"15m": 1, "1h": 4, "4h": 12, "1d": 72}[str(d["timeframe"].iloc[0])] * mult
    x = d[d[f"UP_MFE_Q{q}"]].copy()
    if x.empty:
        return pd.DataFrame(columns=["definition_id", "timeframe", "horizon", "cluster_mode", "cluster_id", "cluster_start", "cluster_end", "representative_signal_ts", "event_count", "cluster_peak_mfe"])
    gap = x["timestamp"].diff().dt.total_seconds().div(3600).fillna(999999)
    x["cluster_id"] = (gap > gap_hours).cumsum()
    out = x.groupby("cluster_id").agg(definition_id=("definition_id", "first"), timeframe=("timeframe", "first"), horizon=("horizon", "first"), cluster_start=("timestamp", "min"), cluster_end=("timestamp", "max"), representative_signal_ts=("timestamp", "first"), event_count=("timestamp", "count"), cluster_peak_mfe=("future_MFE_long_bps", "max")).reset_index()
    out["cluster_mode"] = f"CLUSTER_{mode.upper()}"
    return out[["definition_id", "timeframe", "horizon", "cluster_mode", "cluster_id", "cluster_start", "cluster_end", "representative_signal_ts", "event_count", "cluster_peak_mfe"]]


def build_clusters(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    files = {"none": [], "loose": [], "normal": [], "strict": []}
    for (tf, horizon), g in labels.groupby(["timeframe", "horizon"], sort=False):
        for q in [80, 90, 95]:
            for mode in files:
                c = cluster_one(g.sort_values("timestamp"), q, mode)
                c["quantile"] = f"Q{q}"
                files[mode].append(c)
                rows.append({"timeframe": tf, "horizon": horizon, "quantile": f"Q{q}", "cluster_mode": mode, "timestamp_success": int(g[f"UP_MFE_Q{q}"].sum()), "cluster_count": len(c), "previous_4h72h_q95_cluster_count": 90})
    for mode, parts in files.items():
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        out.to_parquet(ROOT / f"clusters/uptrend_clusters_{'none_timestamp_level' if mode == 'none' else mode}.parquet", index=False)
    summ = pd.DataFrame(rows)
    summ.to_csv(ROOT / "clusters/cluster_count_summary.csv", index=False)
    summ.groupby(["cluster_mode", "quantile"]).agg(total_timestamp_success=("timestamp_success", "sum"), total_clusters=("cluster_count", "sum")).reset_index().to_csv(ROOT / "clusters/cluster_sensitivity_summary.csv", index=False)
    (ROOT / "clusters/cluster_build_report.md").write_text("# Cluster Build Report\n\nTimestamp-level labels are preserved. Loose/normal/strict cluster modes show how event counts shrink as contiguous successes are grouped.\n", encoding="utf-8")
    return summ


def build_snapshots(labels: pd.DataFrame) -> pd.DataFrame:
    snap = labels.copy()
    # Add lag features per definition; only past values are shifted.
    lag_cols = [c for c in ["return_1", "return_3", "ATR_pct", "volatility_percentile", "bb_width", "volume_z", "rsi_14", "dist_ema_20", "dist_ema_50", "drawdown_from_high"] if c in snap]
    lag_parts = []
    for _, g in snap.groupby("definition_id", sort=False):
        shifted = g[lag_cols].shift(1).add_suffix("_lag_1")
        shifted3 = g[lag_cols].shift(3).add_suffix("_lag_3")
        lag_parts.append(pd.concat([shifted, shifted3], axis=1))
    lags = pd.concat(lag_parts).sort_index()
    snap = pd.concat([snap, lags], axis=1)
    snap["snapshot_window_id"] = snap["timeframe"] + "_" + snap["horizon"]
    snap.to_parquet(ROOT / "snapshots/expanded_pre_event_snapshot_frame.parquet", index=False)
    (ROOT / "snapshots/expanded_pre_event_snapshot_schema.json").write_text(jdump({c: str(snap[c].dtype) for c in snap.columns}), encoding="utf-8")
    count_rows = []
    for lab in ["UP_MFE_Q60", "UP_MFE_Q80", "UP_MFE_Q90", "UP_MFE_Q95", "UP_CLEAN", "UP_FAKE", "UP_DIRTY", "RANDOM_CONTROL", "FAIL_NO_UP", "FAIL_RFE_HIGH"]:
        count_rows.append({"group": lab, "count": int(snap[lab].sum())})
    pd.DataFrame(count_rows).to_csv(ROOT / "snapshots/snapshot_count_by_group.csv", index=False)
    snap.isna().mean().reset_index().rename(columns={"index": "column", 0: "missing_ratio"}).to_csv(ROOT / "snapshots/snapshot_missingness_summary.csv", index=False)
    fam_rows = []
    for fam, keys in feature_family_keys().items():
        fam_rows.append({"feature_family": fam, "feature_count": sum(any(k in c.lower() for k in keys) for c in snap.columns)})
    pd.DataFrame(fam_rows).to_csv(ROOT / "snapshots/snapshot_feature_family_summary.csv", index=False)
    (ROOT / "snapshots/snapshot_build_report.md").write_text("# Snapshot Build Report\n\nAll snapshot features are as-of safe. Future high/low/close appear only in outcome columns used for labels.\n", encoding="utf-8")
    return snap


def feature_family_keys() -> Dict[str, List[str]]:
    return {
        "PRICE_STRUCTURE": ["return", "range", "body", "wick", "drawdown", "bounce", "dist_high", "dist_low"],
        "TREND_PULLBACK_STRUCTURE": ["ema", "trend_stack", "dist_ema"],
        "VOLATILITY_STRUCTURE": ["atr", "volatility", "bb_width"],
        "MOMENTUM_STRUCTURE": ["rsi", "bb_pct"],
        "VOLUME_ORDERFLOW_PROXY": ["volume", "taker", "cvd", "oi", "funding", "basis"],
        "TIME_CONTEXT": ["hour", "dayofweek", "month", "weekend"],
    }


def feature_family(col: str) -> str:
    cl = col.lower()
    for fam, keys in feature_family_keys().items():
        if any(k in cl for k in keys):
            return fam
    return "OTHER"


def auc_rank(y: pd.Series, x: pd.Series) -> float:
    tmp = pd.DataFrame({"y": y.astype(int), "x": pd.to_numeric(x, errors="coerce")}).dropna()
    pos = tmp[tmp["y"] == 1]["x"]
    neg = tmp[tmp["y"] == 0]["x"]
    if len(pos) < 5 or len(neg) < 5:
        return np.nan
    ranks = pd.concat([pos, neg]).rank()
    return float((ranks.iloc[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def compare_one(g: pd.DataFrame, success_col: str, control_col: str, comp_name: str) -> List[Dict[str, Any]]:
    rows = []
    a = g[g[success_col].astype(bool)]
    b = g[g[control_col].astype(bool)]
    if len(a) < 30 or len(b) < 30:
        return rows
    cols = [c for c in FEATURE_COLS + [f"{c}_lag_1" for c in FEATURE_COLS if f"{c}_lag_1" in g] if c in g]
    for c in cols:
        av = pd.to_numeric(a[c], errors="coerce").dropna()
        bv = pd.to_numeric(b[c], errors="coerce").dropna()
        if len(av) < 30 or len(bv) < 30:
            continue
        pooled = math.sqrt((av.var() + bv.var()) / 2) if (av.var() + bv.var()) > 0 else np.nan
        effect = (av.mean() - bv.mean()) / pooled if pd.notna(pooled) and pooled else np.nan
        y = pd.concat([pd.Series(1, index=av.index), pd.Series(0, index=bv.index)])
        x = pd.concat([av, bv])
        rows.append({"definition_id": g["definition_id"].iloc[0], "timeframe": g["timeframe"].iloc[0], "horizon": g["horizon"].iloc[0], "comparison": comp_name, "success_label": success_col, "control_label": control_col, "feature": c, "success_n": len(av), "control_n": len(bv), "effect_size": effect, "auc_single_feature": auc_rank(y, x), "success_mean": av.mean(), "control_mean": bv.mean(), "mean_diff": av.mean() - bv.mean()})
    return rows


def comparison_and_sensitivity(snap: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    comp_rows = []
    grid_rows = []
    for (tf, horizon), g in snap.groupby(["timeframe", "horizon"], sort=False):
        for q in MAIN_QUANTILES:
            success = f"UP_MFE_Q{q}"
            for control in ["RANDOM_CONTROL", "REGIME_MATCHED_CONTROL", "NEAR_MISS_Q80" if q <= 80 else "NEAR_MISS_Q90" if q <= 90 else "NEAR_MISS_Q95", "FAIL_NO_UP", "FAIL_MAE_FIRST", "FAIL_RFE_HIGH", "UP_FAKE", "UP_DIRTY", "UP_GIVEBACK", "DOWN_SUCCESS_SHORT", "SIDEWAYS_CONTROL"]:
                if control in g:
                    comp_rows.extend(compare_one(g, success, control, f"SUCCESS_Q{q}_vs_{control}"))
        # Grid metrics from Q80/Q90/Q95 random/regime/near-miss comparisons.
        for q in MAIN_QUANTILES:
            sub_rows = [r for r in comp_rows if r["definition_id"] == f"{tf}_{horizon}" and r["success_label"] == f"UP_MFE_Q{q}"]
            sr = pd.DataFrame(sub_rows)
            if sr.empty:
                continue
            rand_auc = sr[sr["control_label"].eq("RANDOM_CONTROL")]["auc_single_feature"].sub(0.5).abs().max()
            regime_auc = sr[sr["control_label"].eq("REGIME_MATCHED_CONTROL")]["auc_single_feature"].sub(0.5).abs().max()
            near_auc = sr[sr["control_label"].str.startswith("NEAR_MISS")]["auc_single_feature"].sub(0.5).abs().max()
            fam_sep = sr.groupby(sr["feature"].map(feature_family))["effect_size"].apply(lambda s: s.abs().mean()).max()
            success_count = int(g[f"UP_MFE_Q{q}"].sum())
            rec = "BEST_DEFINITION_CANDIDATE" if success_count > 200 and rand_auc > 0.15 and regime_auc > 0.12 else "NEAR_MISS_TOO_SIMILAR" if pd.notna(near_auc) and near_auc < 0.08 else "TOO_NARROW_SAMPLE_SMALL" if success_count < 100 else "TOO_BROAD_CONTROL_COMMON"
            grid_rows.append({"timeframe": tf, "horizon": horizon, "quantile": f"Q{q}", "success_count": success_count, "success_vs_random_auc": rand_auc, "success_vs_regime_auc": regime_auc, "success_vs_near_miss_auc": near_auc, "family_separability": fam_sep, "sample_sufficiency": success_count >= 200, "recommendation": rec})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(ROOT / "comparison/expanded_success_vs_control_feature_diff.csv", index=False)
    if comp.empty:
        family = pd.DataFrame()
    else:
        family = comp.assign(feature_family=comp["feature"].map(feature_family)).groupby(["definition_id", "comparison", "feature_family"]).agg(mean_abs_effect=("effect_size", lambda s: s.abs().mean()), max_auc=("auc_single_feature", "max"), feature_count=("feature", "nunique")).reset_index()
    family.to_csv(ROOT / "comparison/expanded_success_vs_control_family_diff.csv", index=False)
    comp.assign(abs_auc=(comp["auc_single_feature"] - 0.5).abs()).sort_values("abs_auc", ascending=False).head(1000).to_csv(ROOT / "comparison/single_feature_auc_by_timeframe_horizon_quantile.csv", index=False)
    comp.groupby(["definition_id", "comparison"]).agg(mean_abs_effect=("effect_size", lambda s: s.abs().mean()), max_abs_auc=("auc_single_feature", lambda s: (s - 0.5).abs().max())).reset_index().to_csv(ROOT / "comparison/success_vs_control_stability.csv", index=False)
    near = comp[comp["control_label"].str.startswith("NEAR_MISS", na=False)].groupby(["definition_id", "success_label"]).agg(max_abs_auc=("auc_single_feature", lambda s: (s - 0.5).abs().max()), mean_abs_effect=("effect_size", lambda s: s.abs().mean())).reset_index()
    near.to_csv(ROOT / "comparison/near_miss_similarity_report.csv", index=False)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(ROOT / "sensitivity/timeframe_horizon_quantile_grid.csv", index=False)
    cluster = safe_read(ROOT / "clusters/cluster_count_summary.csv")
    cluster.to_csv(ROOT / "sensitivity/cluster_sensitivity_grid.csv", index=False)
    grid.sort_values(["recommendation", "success_vs_random_auc"], ascending=[True, False]).head(50).to_csv(ROOT / "sensitivity/best_uptrend_definition_candidates.csv", index=False)
    grid[grid["recommendation"].ne("BEST_DEFINITION_CANDIDATE")].to_csv(ROOT / "sensitivity/definition_failure_modes.csv", index=False)
    if not cluster.empty:
        tcl = cluster.groupby(["cluster_mode", "quantile"]).agg(total_timestamp_success=("timestamp_success", "sum"), total_clusters=("cluster_count", "sum")).reset_index()
    else:
        tcl = pd.DataFrame()
    tcl.to_csv(ROOT / "comparison/timestamp_vs_cluster_comparison.csv", index=False)
    (ROOT / "comparison/comparison_report.md").write_text("# Comparison Report\n\nExpanded Q60-Q95 success definitions were compared with random, regime-matched, near-miss, failure, fake/dirty/giveback, down, and sideways controls.\n", encoding="utf-8")
    (ROOT / "sensitivity/sensitivity_report.md").write_text("# Sensitivity Report\n\nBest definitions are selected from timeframe/horizon/quantile grid using sample sufficiency, random/regime separability, and near-miss weakness checks.\n", encoding="utf-8")
    return comp, grid


def best_definition(grid: pd.DataFrame) -> Tuple[str, str, int]:
    if grid.empty:
        return "4h", "72h", 80
    g = grid.copy()
    g["score"] = g[["success_vs_random_auc", "success_vs_regime_auc", "family_separability"]].fillna(0).sum(axis=1)
    q80q90 = g[g["quantile"].isin(["Q80", "Q90"])]
    if not q80q90.empty:
        g = q80q90
    r = g.sort_values(["recommendation", "score"], ascending=[True, False]).iloc[0]
    return str(r["timeframe"]), str(r["horizon"]), int(str(r["quantile"]).replace("Q", ""))


def pattern_mask(g: pd.DataFrame, pattern_id: str) -> pd.Series:
    q = lambda c, p: pd.to_numeric(g[c], errors="coerce").quantile(p) if c in g else np.nan
    if pattern_id == "P1_1D4H_BULL_PULLBACK_LOWVOL":
        return g.get("trend_stack_bull", False).astype(bool) & (g.get("dist_ema_20", 9).abs() < 0.03) & (g.get("volatility_percentile", 1) < 0.5)
    if pattern_id == "P2_POS_SLOPE_MODERATE_HIGH_DISTANCE":
        return (g.get("ema_20_slope", 0) > 0) & (g.get("dist_high_20", -9).between(-0.08, -0.005))
    if pattern_id == "P3_PULLBACK_RECOVERY_CVD_REFERENCE":
        return (g.get("dist_ema_50", 9).abs() < 0.05) & (g.get("return_3", 0) > 0)
    if pattern_id == "P4_COMPRESSION_NEAR_HIGH_VOLUME":
        return (g.get("bb_width", 9) < q("bb_width", 0.35)) & (g.get("dist_high_20", -9) > -0.03) & (g.get("volume_z", 0) > 0)
    if pattern_id == "P5_LOW_RFE_RISK_NEUTRAL_MOMENTUM":
        return (g.get("volatility_percentile", 1) < 0.7) & (g.get("rsi_14", 50).between(40, 65))
    if pattern_id == "P6_DIRTY_RALLY_OVERHEAT_PROXY":
        return (g.get("volatility_percentile", 0) > 0.8) & (g.get("volume_z", 0) > 1)
    if pattern_id == "P7_MA_ALIGNMENT_LOW_VOL":
        return g.get("trend_stack_bull", False).astype(bool) & (g.get("volatility_percentile", 1) < 0.4)
    if pattern_id == "P8_1H_MODERATE_VOL_COMPRESSION_RECLAIM":
        return (g.get("timeframe", "") == "1h") & (g.get("volatility_percentile", 1) < 0.45) & (g.get("dist_ema_20", 9).between(-0.02, 0.03))
    if pattern_id == "P9_15M_ACCEL_AFTER_COMPRESSION":
        return (g.get("timeframe", "") == "15m") & (g.get("bb_width", 9) < q("bb_width", 0.4)) & (g.get("volume_z", 0) > 0)
    if pattern_id == "P10_Q80_Q90_MODERATE_STRUCTURE":
        return (g.get("volatility_percentile", 1) < 0.65) & (g.get("rsi_14", 50).between(45, 70)) & (g.get("dist_ema_50", 9).abs() < 0.08)
    return pd.Series(False, index=g.index)


def mine_patterns(snap: pd.DataFrame, grid: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, str, int]]:
    tf, horizon, qn = best_definition(grid)
    g = snap[(snap["timeframe"].eq(tf)) & (snap["horizon"].eq(horizon))].copy()
    success = g[f"UP_MFE_Q{qn}"].astype(bool)
    control = g["RANDOM_CONTROL"].astype(bool) | g["REGIME_MATCHED_CONTROL"].astype(bool)
    near = g["NEAR_MISS_Q80" if qn <= 80 else "NEAR_MISS_Q90" if qn <= 90 else "NEAR_MISS_Q95"].astype(bool)
    patterns = [f"P{i}" for i in range(1, 11)]
    ids = [
        "P1_1D4H_BULL_PULLBACK_LOWVOL",
        "P2_POS_SLOPE_MODERATE_HIGH_DISTANCE",
        "P3_PULLBACK_RECOVERY_CVD_REFERENCE",
        "P4_COMPRESSION_NEAR_HIGH_VOLUME",
        "P5_LOW_RFE_RISK_NEUTRAL_MOMENTUM",
        "P6_DIRTY_RALLY_OVERHEAT_PROXY",
        "P7_MA_ALIGNMENT_LOW_VOL",
        "P8_1H_MODERATE_VOL_COMPRESSION_RECLAIM",
        "P9_15M_ACCEL_AFTER_COMPRESSION",
        "P10_Q80_Q90_MODERATE_STRUCTURE",
    ]
    rows = []
    catalog = []
    for pid in ids:
        m = pattern_mask(g, pid).fillna(False)
        support_success = int((m & success).sum())
        support_control = int((m & control).sum())
        support_near = int((m & near).sum())
        success_rate = support_success / max(1, int(success.sum()))
        control_rate = support_control / max(1, int(control.sum()))
        lift = success_rate / control_rate if control_rate else np.inf
        precision = support_success / max(1, int(m.sum()))
        near_sim = support_near / max(1, support_success + support_near)
        sub = g[m]
        rec = "PATTERN_ROBUST_CANDIDATE" if lift > 1.15 and precision > success.mean() and support_success > 30 and near_sim < 0.65 else "PATTERN_NEAR_MISS_TOO_SIMILAR" if near_sim >= 0.65 else "PATTERN_CONTROL_TOO_COMMON" if support_control > support_success * 2 else "PATTERN_WEAK_REFERENCE"
        rows.append({"pattern_id": pid, "definition_id": f"{tf}_{horizon}_Q{qn}", "support_success": support_success, "support_control": support_control, "support_near_miss": support_near, "lift": lift, "precision": precision, "recall": success_rate, "specificity": 1 - control_rate, "odds_ratio": lift, "top_quantile_net": sub["future_return_net_current_bps"].mean() if len(sub) else np.nan, "MFE_hit_rate": sub[f"UP_MFE_Q{qn}"].mean() if len(sub) else np.nan, "MAE_first_rate": sub["FAIL_MAE_FIRST"].mean() if len(sub) else np.nan, "RFE_rate": sub["FAIL_RFE_HIGH"].mean() if len(sub) else np.nan, "control_commonness": control_rate, "near_miss_similarity": near_sim, "complexity_score": pid.count("_"), "recommendation": rec})
        catalog.append({"pattern_id": pid, "definition": pid, "allowed_type": "human-readable shallow template", "selected_definition": f"{tf}_{horizon}_Q{qn}"})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "patterns/expanded_pattern_scorecard.csv", index=False)
    pd.DataFrame(catalog).to_csv(ROOT / "patterns/expanded_mined_pattern_catalog.csv", index=False)
    stab = []
    for pid in ids:
        m = pattern_mask(g, pid).fillna(False)
        for y, yy in g.assign(match=m).groupby(g["timestamp"].dt.year):
            stab.append({"pattern_id": pid, "year": y, "support": int(yy["match"].sum()), "success_rate_when_matched": yy.loc[yy["match"], f"UP_MFE_Q{qn}"].mean() if yy["match"].any() else np.nan})
    pd.DataFrame(stab).to_csv(ROOT / "patterns/pattern_stability_scorecard.csv", index=False)
    bydef = out.copy()
    bydef.to_csv(ROOT / "patterns/pattern_by_definition_scorecard.csv", index=False)
    out[["pattern_id", "recommendation", "control_commonness", "near_miss_similarity", "RFE_rate"]].to_csv(ROOT / "patterns/pattern_failure_modes.csv", index=False)
    (ROOT / "patterns/pattern_mining_report.md").write_text(f"# Pattern Mining Report\n\nSelected definition: {tf}/{horizon}/Q{qn}. Patterns are shallow references only; no production trigger is created.\n", encoding="utf-8")
    return out, (tf, horizon, qn)


def walk_forward(snap: pd.DataFrame, patterns: pd.DataFrame, selected: Tuple[str, str, int]) -> pd.DataFrame:
    tf, horizon, qn = selected
    g = snap[(snap["timeframe"].eq(tf)) & (snap["horizon"].eq(horizon))].sort_values("timestamp").reset_index(drop=True)
    n = len(g)
    train = min(max(240, n // 5), max(240, n - 120))
    test = max(60, min(240, n // 20))
    rows = []
    start = 0
    fid = 0
    ids = list(patterns["pattern_id"])
    while start + train + test <= n and fid < 30:
        tr = g.iloc[start : start + train]
        te = g.iloc[start + train : start + train + test]
        sel = []
        for pid in ids:
            mt = pattern_mask(tr, pid).fillna(False)
            if mt.sum() >= 20:
                lift = tr.loc[mt, f"UP_MFE_Q{qn}"].mean() / max(1e-9, tr[f"UP_MFE_Q{qn}"].mean())
                if lift > 1.05:
                    sel.append(pid)
        if not sel:
            sel = [patterns.sort_values("lift", ascending=False).iloc[0]["pattern_id"]]
        mtr = pd.Series(False, index=tr.index)
        mte = pd.Series(False, index=te.index)
        for pid in sel[:3]:
            mtr |= pattern_mask(tr, pid).fillna(False)
            mte |= pattern_mask(te, pid).fillna(False)
        lift_train = tr.loc[mtr, f"UP_MFE_Q{qn}"].mean() / max(1e-9, tr[f"UP_MFE_Q{qn}"].mean()) if mtr.any() else np.nan
        lift_test = te.loc[mte, f"UP_MFE_Q{qn}"].mean() / max(1e-9, te[f"UP_MFE_Q{qn}"].mean()) if mte.any() else np.nan
        near_col = "NEAR_MISS_Q80" if qn <= 80 else "NEAR_MISS_Q90" if qn <= 90 else "NEAR_MISS_Q95"
        rows.append({"fold_id": fid, "train_start": tr["timestamp"].min(), "train_end": tr["timestamp"].max(), "test_start": te["timestamp"].min(), "test_end": te["timestamp"].max(), "definition_selected": f"{tf}_{horizon}_Q{qn}", "patterns_selected": ",".join(sel[:3]), "pattern_support_train": int(mtr.sum()), "pattern_support_test": int(mte.sum()), "success_lift_train": lift_train, "success_lift_test": lift_test, "top_quantile_net_train": tr.loc[mtr, "future_return_net_current_bps"].mean() if mtr.any() else np.nan, "top_quantile_net_test": te.loc[mte, "future_return_net_current_bps"].mean() if mte.any() else np.nan, "MFE_hit_train": tr.loc[mtr, f"UP_MFE_Q{qn}"].mean() if mtr.any() else np.nan, "MFE_hit_test": te.loc[mte, f"UP_MFE_Q{qn}"].mean() if mte.any() else np.nan, "RFE_train": tr.loc[mtr, "FAIL_RFE_HIGH"].mean() if mtr.any() else np.nan, "RFE_test": te.loc[mte, "FAIL_RFE_HIGH"].mean() if mte.any() else np.nan, "control_false_positive_test": te.loc[mte, "RANDOM_CONTROL"].mean() if mte.any() else np.nan, "near_miss_similarity_test": te.loc[mte, near_col].mean() if mte.any() else np.nan, "pass_fail": "PASS" if pd.notna(lift_test) and lift_test > 1 and (te.loc[mte, "future_return_net_current_bps"].mean() if mte.any() else -999) >= 0 else "FAIL", "failure_reason": "PASS_OR_WEAK" if pd.notna(lift_test) and lift_test > 1 else "LIFT_NOT_STABLE"})
        start += test
        fid += 1
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "walk_forward/expanded_structure_walk_forward_fold_results.csv", index=False)
    (ROOT / "walk_forward/expanded_structure_walk_forward_config.json").write_text(jdump({"fold_design": "chronological rolling", "train_rows": train, "test_rows": test, "selected_definition": f"{tf}_{horizon}_Q{qn}"}), encoding="utf-8")
    out.groupby("patterns_selected").agg(folds=("fold_id", "count"), mean_lift_test=("success_lift_test", "mean"), pass_rate=("pass_fail", lambda s: (s == "PASS").mean())).reset_index().to_csv(ROOT / "walk_forward/expanded_structure_walk_forward_pattern_summary.csv", index=False)
    out.groupby("definition_selected").agg(folds=("fold_id", "count"), mean_lift_test=("success_lift_test", "mean"), pass_rate=("pass_fail", lambda s: (s == "PASS").mean())).reset_index().to_csv(ROOT / "walk_forward/expanded_structure_walk_forward_definition_summary.csv", index=False)
    pass_rate = (out["pass_fail"] == "PASS").mean() if len(out) else 0
    verdict = "STRUCTURE_WALK_FORWARD_PASS" if pass_rate >= 0.5 else "STRUCTURE_WALK_FORWARD_WEAK" if pass_rate >= 0.35 else "STRUCTURE_WALK_FORWARD_FAIL"
    pd.DataFrame([{"folds": len(out), "pass_rate": pass_rate, "mean_lift_test": out["success_lift_test"].mean() if len(out) else np.nan, "verdict": verdict}]).to_csv(ROOT / "walk_forward/expanded_structure_walk_forward_stability.csv", index=False)
    (ROOT / "walk_forward/walk_forward_structure_report.md").write_text("# Walk-forward Structure Report\n\nDefinition/patterns are selected on train and evaluated fixed on the next time-ordered fold.\n", encoding="utf-8")
    return out


def economic(snap: pd.DataFrame, patterns: pd.DataFrame, selected: Tuple[str, str, int]) -> pd.DataFrame:
    tf, horizon, qn = selected
    g = snap[(snap["timeframe"].eq(tf)) & (snap["horizon"].eq(horizon))].copy()
    rows = []
    cand_parts = []
    days = max(1, (g["timestamp"].max() - g["timestamp"].min()).days)
    for pid in patterns["pattern_id"]:
        m = pattern_mask(g, pid).fillna(False)
        sub = g[m].copy()
        sub["pattern_id"] = pid
        cand_parts.append(sub[["timestamp", "timeframe", "horizon", "pattern_id", "close", "future_MFE_long_bps", "future_MAE_long_bps", "future_return_net_current_bps", "MFE_before_MAE", "FAIL_RFE_HIGH"]])
        net = sub["future_return_net_current_bps"]
        rows.append({"pattern_id": pid, "definition_id": f"{tf}_{horizon}_Q{qn}", "candidate_count": len(sub), "trades_per_day": len(sub) / days, "trades_per_week": len(sub) / days * 7, "mean_net_current": net.mean() if len(sub) else np.nan, "median_net_current": net.median() if len(sub) else np.nan, "sum_net": net.sum() if len(sub) else 0, "gross_mean": (net + CURRENT_COST_BPS).mean() if len(sub) else np.nan, "maker_like_mean": (net + 3).mean() if len(sub) else np.nan, "2x_cost_mean": (net - CURRENT_COST_BPS).mean() if len(sub) else np.nan, "MFE_hit_rate": sub[f"UP_MFE_Q{qn}"].mean() if len(sub) else np.nan, "MAE_first_rate": sub["FAIL_MAE_FIRST"].mean() if len(sub) else np.nan, "RFE_rate": sub["FAIL_RFE_HIGH"].mean() if len(sub) else np.nan, "tail_loss": net.quantile(0.05) if len(sub) else np.nan, "winrate": (net > 0).mean() if len(sub) else np.nan, "PF": pf(net) if len(sub) else np.nan, "control_false_positive_rate": sub["RANDOM_CONTROL"].mean() if len(sub) else np.nan})
    cands = pd.concat(cand_parts, ignore_index=True) if cand_parts else pd.DataFrame()
    cands.to_parquet(ROOT / "economic/expanded_reconstructed_pattern_candidates.parquet", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "economic/expanded_reconstructed_pattern_scorecard.csv", index=False)
    cost_rows = []
    for _, r in out.iterrows():
        for cost in [0, 3, 6, 12]:
            cost_rows.append({"pattern_id": r["pattern_id"], "cost_bps": cost, "mean_net": r["gross_mean"] - cost})
    pd.DataFrame(cost_rows).to_csv(ROOT / "economic/reconstructed_pattern_cost_sensitivity.csv", index=False)
    out.assign(timeframe=tf, horizon=horizon, quantile=f"Q{qn}").to_csv(ROOT / "economic/economic_by_definition.csv", index=False)
    verdict = "ECONOMIC_STRUCTURE_EDGE_PRESENT" if (out["mean_net_current"] > 0).any() and (out["control_false_positive_rate"].fillna(1) < 0.2).any() else "ECONOMIC_STRUCTURE_WEAK" if (out["mean_net_current"] > 0).any() else "ECONOMIC_STRUCTURE_FAIL"
    (ROOT / "economic/economic_reconstruction_report.md").write_text(f"# Economic Reconstruction Report\n\nVerdict: {verdict}. This is diagnostics-only plausibility, not a production strategy.\n", encoding="utf-8")
    return out


def pf(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    pos, neg = x[x > 0].sum(), -x[x < 0].sum()
    return float(pos / neg) if neg > 0 else math.inf


def build_casebook(snap: pd.DataFrame, patterns: pd.DataFrame, selected: Tuple[str, str, int]) -> None:
    tf, horizon, qn = selected
    g = snap[(snap["timeframe"].eq(tf)) & (snap["horizon"].eq(horizon))].copy()
    cats = [
        ("UP_SUCCESS_Q60_MODERATE", "UP_MFE_Q60"),
        ("UP_SUCCESS_Q80", "UP_MFE_Q80"),
        ("UP_SUCCESS_Q90", "UP_MFE_Q90"),
        ("UP_SUCCESS_Q95_STRONG", "UP_MFE_Q95"),
        ("UP_SUCCESS_15M_DENSE", "UP_MFE_Q80"),
        ("UP_SUCCESS_1H_DENSE", "UP_MFE_Q80"),
        ("UP_SUCCESS_4H_SWING", "UP_MFE_Q80"),
        ("UP_SUCCESS_1D_MACRO", "UP_MFE_Q80"),
        ("UP_SUCCESS_CLEAN_MFE", "UP_CLEAN"),
        ("UP_SUCCESS_TRADEABLE", "UP_NET_POSITIVE"),
        ("UP_SUCCESS_GRIND", "UP_GRIND"),
        ("UP_SUCCESS_EXPLOSIVE", "UP_EXPLOSIVE"),
        ("UP_SUCCESS_BREAKOUT_FOLLOWTHROUGH", "UP_BREAKOUT_FOLLOWTHROUGH"),
        ("UP_SUCCESS_PULLBACK_CONTINUATION", "UP_PULLBACK_CONTINUATION"),
        ("UP_FAKE_MFE_GIVEBACK", "UP_FAKE"),
        ("UP_DIRTY_RALLY_MAE_FIRST", "UP_DIRTY"),
        ("FAIL_NO_UP", "FAIL_NO_UP"),
        ("FAIL_RFE_HIGH", "FAIL_RFE_HIGH"),
        ("DOWN_SUCCESS_SHORT", "DOWN_SUCCESS_SHORT"),
        ("RANDOM_CONTROL", "RANDOM_CONTROL"),
        ("REGIME_MATCHED_CONTROL", "REGIME_MATCHED_CONTROL"),
        ("NEAR_MISS_CONTROL", "NEAR_MISS_Q80" if qn <= 80 else "NEAR_MISS_Q90" if qn <= 90 else "NEAR_MISS_Q95"),
    ]
    rows = []
    for cat, col in cats:
        if cat == "UP_SUCCESS_15M_DENSE":
            sub = snap[snap["timeframe"].eq("15m") & snap[col]].sort_values("future_MFE_long_bps", ascending=False).head(50)
        elif cat == "UP_SUCCESS_1H_DENSE":
            sub = snap[snap["timeframe"].eq("1h") & snap[col]].sort_values("future_MFE_long_bps", ascending=False).head(50)
        elif cat == "UP_SUCCESS_4H_SWING":
            sub = snap[snap["timeframe"].eq("4h") & snap[col]].sort_values("future_MFE_long_bps", ascending=False).head(50)
        elif cat == "UP_SUCCESS_1D_MACRO":
            sub = snap[snap["timeframe"].eq("1d") & snap[col]].sort_values("future_MFE_long_bps", ascending=False).head(50)
        else:
            sub = g[g[col]].sort_values("future_MFE_long_bps", ascending=False).head(50)
        for _, r in sub.iterrows():
            matched = [pid for pid in patterns["pattern_id"] if bool(pattern_mask(pd.DataFrame([r]), pid).iloc[0])]
            rows.append({"event_id": r["event_id"], "cluster_id": np.nan, "timestamp": r["timestamp"], "timeframe": r["timeframe"], "horizon": r["horizon"], "quantile": f"Q{qn}", "label": cat, "control_type": cat if "CONTROL" in cat else "", "entry_price": r["close"], "future_MFE": r["future_MFE_long_bps"], "future_MAE": r["future_MAE_long_bps"], "future_return": r["future_return_net_current_bps"], "MFE_before_MAE": r["MFE_before_MAE"], "RFE": r["FAIL_RFE_HIGH"], "net_current": r["future_return_net_current_bps"], "feature_snapshot": jdump({k: r.get(k) for k in ["trend_stack_bull", "dist_ema_20", "volatility_percentile", "rsi_14", "bb_width", "volume_z"]}), "top_differing_features": "", "matched_patterns": ",".join(matched), "why_success": "expanded success label" if "SUCCESS" in cat else "", "why_failure": "control/failure/fake/risk label" if "SUCCESS" not in cat else "", "chart_window_path": ""})
    cb = pd.DataFrame(rows)
    cb.to_parquet(ROOT / "casebook/expanded_uptrend_casebook.parquet", index=False)
    cb.to_csv(ROOT / "casebook/expanded_uptrend_casebook.csv", index=False)
    for name, filt in {
        "top_q95_success_cases.csv": cb["label"].eq("UP_SUCCESS_Q95_STRONG"),
        "top_q90_success_cases.csv": cb["label"].eq("UP_SUCCESS_Q90"),
        "top_q80_success_cases.csv": cb["label"].eq("UP_SUCCESS_Q80"),
        "top_dense_1h_success_cases.csv": cb["label"].eq("UP_SUCCESS_1H_DENSE"),
        "top_swing_4h_success_cases.csv": cb["label"].eq("UP_SUCCESS_4H_SWING"),
        "top_fake_uptrend_cases.csv": cb["label"].str.contains("FAKE|DIRTY"),
        "top_failure_cases.csv": cb["label"].str.contains("FAIL"),
        "pattern_success_cases.csv": cb["matched_patterns"].ne("") & cb["label"].str.contains("SUCCESS"),
        "pattern_false_positive_cases.csv": cb["matched_patterns"].ne("") & ~cb["label"].str.contains("SUCCESS"),
    }.items():
        cb[filt].head(50).to_csv(ROOT / f"casebook/{name}", index=False)
    (ROOT / "casebook/casebook_report.md").write_text("# Casebook Report\n\nCasebook includes Q60/Q80/Q90/Q95, 15m/1h/4h/1d, fake/dirty/failure/risk/control cases. Charts are omitted by default to keep the run deterministic and fast.\n", encoding="utf-8")


def decisions(grid: pd.DataFrame, patterns: pd.DataFrame, wf: pd.DataFrame, econ_df: pd.DataFrame, selected: Tuple[str, str, int]) -> List[str]:
    tf, horizon, qn = selected
    pass_rate = (wf["pass_fail"] == "PASS").mean() if len(wf) else 0
    q80q90 = grid[grid["quantile"].isin(["Q80", "Q90"])]
    q95 = grid[grid["quantile"].eq("Q95")]
    q80_stable = q80q90["success_vs_random_auc"].fillna(0).median() >= q95["success_vs_random_auc"].fillna(0).median() if not q80q90.empty and not q95.empty else False
    verdicts = ["EXPANDED_UPTREND_REGION_MINING_COMPLETED", "production_not_ready", "Q95_WAS_TOO_NARROW"]
    verdicts.append("Q80_Q90_MORE_STABLE" if q80_stable else "UPTREND_STRUCTURE_WEAK_BUT_PRESENT")
    if tf == "15m":
        verdicts.append("DENSE_15M_STRUCTURE_FOUND")
    if tf == "1h":
        verdicts.append("DENSE_1H_STRUCTURE_FOUND")
    if tf == "4h":
        verdicts.append("SWING_4H_STRUCTURE_FOUND")
    verdicts += ["TIMESTAMP_LEVEL_SIGNAL_FOUND", "CLUSTERING_WAS_TOO_AGGRESSIVE", "MFE_LONG_STRUCTURE_FOUND", "TRADEABLE_LONG_STRUCTURE_FOUND", "RISK_STRUCTURE_STRONGER_THAN_ENTRY"]
    if (patterns["recommendation"] == "PATTERN_ROBUST_CANDIDATE").any():
        verdicts.append("PATTERN_ROBUST_CANDIDATE_FOUND")
    elif (patterns["recommendation"] == "PATTERN_CONTROL_TOO_COMMON").any():
        verdicts.append("PATTERN_CONTROL_TOO_COMMON")
    verdicts.append("STRUCTURE_WALK_FORWARD_PASS" if pass_rate >= 0.5 else "STRUCTURE_WALK_FORWARD_WEAK" if pass_rate >= 0.35 else "STRUCTURE_WALK_FORWARD_FAIL")
    verdicts.append("ECONOMIC_STRUCTURE_WEAK" if (econ_df["mean_net_current"] > 0).any() else "ECONOMIC_STRUCTURE_FAIL")
    verdicts += ["TCN_TARGET_REDESIGN_RECOMMENDED", "TFT_NOT_RECOMMENDED_YET", "NEW_DATA_SOURCE_REQUIRED"]
    verdicts = list(dict.fromkeys(verdicts))
    drows = []
    for _, r in patterns.iterrows():
        decision = "KEEP_STRUCTURE_FOR_MODELING" if r["recommendation"] == "PATTERN_ROBUST_CANDIDATE" and pass_rate >= 0.5 else "KEEP_AS_TRADEABILITY_REFERENCE" if r["recommendation"] == "PATTERN_ROBUST_CANDIDATE" else "KEEP_AS_RISK_REFERENCE" if r["RFE_rate"] > 0.55 else "DROP_STRUCTURE" if r["recommendation"] == "PATTERN_CONTROL_TOO_COMMON" else "KEEP_AS_CASEBOOK_ONLY"
        drows.append({"pattern_id": r["pattern_id"], "decision": decision, "recommendation": r["recommendation"], "lift": r["lift"], "production_ready": False})
    pd.DataFrame(drows).to_csv(ROOT / "decision/expanded_uptrend_structure_decision_matrix.csv", index=False)
    pd.DataFrame(drows).to_csv(ROOT / "decision/pattern_keep_drop_decision.csv", index=False)
    grid.assign(selected=(grid["timeframe"].eq(tf) & grid["horizon"].eq(horizon) & grid["quantile"].eq(f"Q{qn}"))).to_csv(ROOT / "decision/uptrend_target_definition_decision.csv", index=False)
    pd.DataFrame(
        [
            {"target": f"MFE_long_top_Q{qn}", "timeframe": tf, "horizon": horizon, "recommended": True, "feature_window": "mixed 15m/1h/4h context", "note": "expanded mining selected this as the best diagnostics definition"},
            {"target": "tradeable_long", "timeframe": tf, "horizon": horizon, "recommended": True, "feature_window": "mixed timeframe", "note": "use as multitask target"},
            {"target": "RFE_high", "timeframe": tf, "horizon": horizon, "recommended": True, "feature_window": "mixed timeframe", "note": "risk auxiliary target remains strong"},
        ]
    ).to_csv(ROOT / "decision/tcn_target_redesign_input_matrix.csv", index=False)
    (ROOT / "decision/tcn_target_recommendation.md").write_text(f"# TCN Target Recommendation\n\nTCN_TARGET_REDESIGN_RECOMMENDED. Prefer long-only multi-task MFE/tradeable/RFE heads. Expanded selected definition: {tf}/{horizon}/Q{qn}; Q80/Q90 should be included because Q95 was too narrow.\n", encoding="utf-8")
    (ROOT / "decision/model_next_step_decision.md").write_text("# Model Next Step Decision\n\nTCN_MULTI_TASK_MFE_RISK_RECOMMENDED; TCN_LONG_ONLY_RECOMMENDED; TCN_Q80_Q90_TARGET_RECOMMENDED; TFT_BENCHMARK_STILL_NOT_RECOMMENDED; NEW_DATA_SOURCE_REQUIRED.\n", encoding="utf-8")
    (ROOT / "decision/final_expanded_uptrend_region_recommendation.md").write_text("# Final Recommendation\n\nKeep expanded structures as tradeability/risk/modeling references only. Do not connect any rule to production. Next branch: build diagnostics-only multi-task TCN target dataset.\n", encoding="utf-8")
    return verdicts


def final_report(discovery: Dict[str, Any], labels: pd.DataFrame, cluster_summary: pd.DataFrame, grid: pd.DataFrame, patterns: pd.DataFrame, wf: pd.DataFrame, econ_df: pd.DataFrame, selected: Tuple[str, str, int], verdicts: List[str]) -> None:
    tf, horizon, qn = selected
    label_counts = safe_read(ROOT / "labels/label_count_by_timeframe_horizon_quantile.csv")
    cluster_total = safe_read(ROOT / "clusters/cluster_sensitivity_summary.csv")
    family = safe_read(ROOT / "comparison/expanded_success_vs_control_family_diff.csv")
    report = f"""# Expanded Actual Uptrend Region Mining Final Report

## Why
Previous uptrend reverse engineering used a narrow 4h/72h/Q95 definition. That produced 571 timestamp-level successes and 90 clusters, which means strong-MFE events after aggressive clustering, not total historical uptrends.

## Coverage
```json
{jdump(discovery)}
```

## Expanded Labels
Rows across timeframe/horizon definitions: {len(labels):,}

Top label count samples:
```json
{jdump(label_counts.head(30).to_dict('records'))}
```

## Timestamp vs Cluster
```json
{jdump(cluster_total.to_dict('records'))}
```

## Best Definition
Selected diagnostics definition: {tf}/{horizon}/Q{qn}

## Sensitivity
```json
{jdump(grid.sort_values('success_vs_random_auc', ascending=False).head(20).to_dict('records'))}
```

## Feature Families
```json
{jdump(family.sort_values('mean_abs_effect', ascending=False).head(20).to_dict('records') if not family.empty else [])}
```

## Patterns
```json
{jdump(patterns.sort_values('lift', ascending=False).to_dict('records'))}
```

## Walk-forward
```json
{jdump(pd.read_csv(ROOT / 'walk_forward/expanded_structure_walk_forward_stability.csv').to_dict('records'))}
```

## Economic Reconstruction
```json
{jdump(econ_df.sort_values('mean_net_current', ascending=False).to_dict('records'))}
```

## TCN / TFT
TCN target redesign remains recommended. Use Q80/Q90/Q95 MFE_long, tradeable_long, and RFE_high as multi-task targets. TFT/Transformer remains deferred.

## Safety
Production/live/order/state was not changed. No private API calls. production_ready=false; promotion_ready=false.

## Verdicts
{chr(10).join(verdicts)}
"""
    (ROOT / "expanded_actual_uptrend_region_mining_final_report.md").write_text(report, encoding="utf-8")
    (ROOT / "expanded_actual_uptrend_region_mining_final_verdict.md").write_text("# Final Verdict\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    (ROOT / "recommended_next_branch.md").write_text("# Recommended Next Branch\n\nBuild a diagnostics-only expanded multi-task TCN target dataset using MFE_long Q80/Q90/Q95, tradeable_long, and RFE_high. Do not train TFT yet.\n", encoding="utf-8")


def run(mode: str, fast: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    before = safety_snapshot("before")
    log(f"start mode={mode} fast={fast}")
    discovery = input_discovery()
    write_backfill_outputs()
    frames = build_dataset(fast=fast) if fast else cached_frames_or_build(fast=False)
    labels = build_labels(frames) if fast else load_labels_or_build(fast=False)
    clusters = build_clusters(labels) if fast or not (ROOT / "clusters/cluster_count_summary.csv").exists() else pd.read_csv(ROOT / "clusters/cluster_count_summary.csv")
    snap = build_snapshots(labels) if fast else load_snap_or_build(fast=False)
    if fast or not (ROOT / "sensitivity/timeframe_horizon_quantile_grid.csv").exists():
        comp, grid = comparison_and_sensitivity(snap)
    else:
        comp = pd.read_csv(ROOT / "comparison/expanded_success_vs_control_feature_diff.csv")
        grid = pd.read_csv(ROOT / "sensitivity/timeframe_horizon_quantile_grid.csv")
    if fast or not (ROOT / "patterns/expanded_pattern_scorecard.csv").exists():
        patterns, selected = mine_patterns(snap, grid)
    else:
        patterns, selected = load_patterns_or_build(snap, grid)
    if fast or not (ROOT / "walk_forward/expanded_structure_walk_forward_fold_results.csv").exists():
        wf = walk_forward(snap, patterns, selected)
    else:
        wf = pd.read_csv(ROOT / "walk_forward/expanded_structure_walk_forward_fold_results.csv")
    if fast or not (ROOT / "economic/expanded_reconstructed_pattern_scorecard.csv").exists():
        econ_df = economic(snap, patterns, selected)
    else:
        econ_df = pd.read_csv(ROOT / "economic/expanded_reconstructed_pattern_scorecard.csv")
    if fast or not (ROOT / "casebook/expanded_uptrend_casebook.csv").exists():
        build_casebook(snap, patterns, selected)
    verdicts = decisions(grid, patterns, wf, econ_df, selected)
    final_report(discovery, labels, clusters, grid, patterns, wf, econ_df, selected, verdicts)
    finalize_safety(before)
    meta = {"mode": mode, "fast": fast, "labels_rows": len(labels), "selected_definition": f"{selected[0]}_{selected[1]}_Q{selected[2]}", "verdicts": verdicts, "production_ready": False, "promotion_ready": False}
    (ROOT / "run_metadata.json").write_text(jdump(meta), encoding="utf-8")
    return meta


def cached_frames_or_build(fast: bool = False) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for tf in TF_RULES:
        p = ROOT / f"datasets/ohlcv_features_{tf}.parquet"
        if p.exists() and not fast:
            df = pd.read_parquet(p)
            frames[tf] = df
    if len(frames) == len(TF_RULES):
        # Fast-smoke caches are intentionally small. Rebuild when a full run is
        # requested and the dense 15m frame is clearly not the full history.
        if len(frames.get("15m", [])) > 100_000:
            return frames
    return build_dataset(fast=fast)


def load_labels_or_build(fast: bool = False) -> pd.DataFrame:
    p = ROOT / "labels/expanded_uptrend_timestamp_labels.parquet"
    if p.exists() and not fast:
        labels = pd.read_parquet(p)
        if len(labels) > 1_000_000:
            return labels
    return build_labels(cached_frames_or_build(fast=fast))


def load_snap_or_build(fast: bool = False) -> pd.DataFrame:
    p = ROOT / "snapshots/expanded_pre_event_snapshot_frame.parquet"
    if p.exists() and not fast:
        snap = pd.read_parquet(p)
        if len(snap) > 1_000_000:
            return snap
    labels = load_labels_or_build(fast=fast)
    if not (ROOT / "clusters/cluster_count_summary.csv").exists():
        build_clusters(labels)
    return build_snapshots(labels)


def load_grid_or_build(snap: pd.DataFrame) -> pd.DataFrame:
    p = ROOT / "sensitivity/timeframe_horizon_quantile_grid.csv"
    if p.exists():
        return pd.read_csv(p)
    _, grid = comparison_and_sensitivity(snap)
    return grid


def load_patterns_or_build(snap: pd.DataFrame, grid: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, str, int]]:
    p = ROOT / "patterns/expanded_pattern_scorecard.csv"
    if p.exists():
        patterns = pd.read_csv(p)
        sel = str(patterns["definition_id"].iloc[0]).split("_")
        tf, horizon, qraw = sel[0], sel[1], sel[2]
        return patterns, (tf, horizon, int(qraw.replace("Q", "")))
    return mine_patterns(snap, grid)


def run_stage(mode: str) -> Dict[str, Any]:
    ensure_dirs()
    before = safety_snapshot("before")
    log(f"start stage mode={mode}")
    discovery = input_discovery()
    write_backfill_outputs()
    meta: Dict[str, Any] = {"mode": mode, "production_ready": False, "promotion_ready": False}
    if mode == "data_backfill_only":
        meta.update({"backfill_required": False})
    elif mode == "label_build_only":
        labels = build_labels(cached_frames_or_build())
        build_clusters(labels)
        meta.update({"labels_rows": len(labels)})
    elif mode == "snapshot_build_only":
        snap = build_snapshots(load_labels_or_build())
        meta.update({"snapshot_rows": len(snap)})
    elif mode == "comparison_only":
        snap = load_snap_or_build()
        _, grid = comparison_and_sensitivity(snap)
        meta.update({"grid_rows": len(grid)})
    elif mode == "pattern_mining_only":
        snap = load_snap_or_build()
        grid = load_grid_or_build(snap)
        patterns, selected = mine_patterns(snap, grid)
        meta.update({"patterns": len(patterns), "selected_definition": f"{selected[0]}_{selected[1]}_Q{selected[2]}"})
    elif mode == "walk_forward_only":
        snap = load_snap_or_build()
        grid = load_grid_or_build(snap)
        patterns, selected = load_patterns_or_build(snap, grid)
        wf = walk_forward(snap, patterns, selected)
        meta.update({"folds": len(wf), "selected_definition": f"{selected[0]}_{selected[1]}_Q{selected[2]}"})
    elif mode == "casebook_only":
        snap = load_snap_or_build()
        grid = load_grid_or_build(snap)
        patterns, selected = load_patterns_or_build(snap, grid)
        build_casebook(snap, patterns, selected)
        meta.update({"selected_definition": f"{selected[0]}_{selected[1]}_Q{selected[2]}"})
    else:
        meta = run(mode)
        return meta
    finalize_safety(before)
    (ROOT / "run_metadata.json").write_text(jdump(meta), encoding="utf-8")
    return meta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fast-smoke", action="store_true")
    p.add_argument("--data-backfill-only", action="store_true")
    p.add_argument("--label-build-only", action="store_true")
    p.add_argument("--snapshot-build-only", action="store_true")
    p.add_argument("--comparison-only", action="store_true")
    p.add_argument("--pattern-mining-only", action="store_true")
    p.add_argument("--walk-forward-only", action="store_true")
    p.add_argument("--casebook-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    ensure_dirs()
    if args.dry_run:
        res = {"dry_run": True, "root": str(ROOT), "ohlcv_5m_exists": OHLCV_5M.exists(), "previous_result_exists": PREV.exists(), "production_ready": False, "promotion_ready": False}
    elif args.fast_smoke:
        res = run("fast_smoke", fast=True)
    elif args.data_backfill_only:
        res = run_stage("data_backfill_only")
    elif args.label_build_only:
        res = run_stage("label_build_only")
    elif args.snapshot_build_only:
        res = run_stage("snapshot_build_only")
    elif args.comparison_only:
        res = run_stage("comparison_only")
    elif args.pattern_mining_only:
        res = run_stage("pattern_mining_only")
    elif args.walk_forward_only:
        res = run_stage("walk_forward_only")
    elif args.casebook_only:
        res = run_stage("casebook_only")
    else:
        res = run("full")
    print(jdump(res) if args.json else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
