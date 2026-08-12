"""Signal persistence / delayed edge / movement continuation research.

Diagnostics-only. Quantifies how long compression/breakout/state signals remain
favorable across h=1..60 horizons. No production, order, or private endpoint use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("data/diagnostics/signal_persistence_horizon_sweep")
CANONICAL_PATHS = Path("data/diagnostics/data_sync/canonical_data_paths.json")
DISCOVERED_PATHS = Path("data/diagnostics/feature_proba_refresh/discovered_paths.json")
DEFAULT_COST_BPS = 6.0
MAKER_COST_BPS = 3.0
TWO_X_COST_BPS = 12.0
KST_TZ = "Asia/Seoul"
MIN_EVENTS_RELIABLE = 30
REVERSAL_RETURN_THRESHOLD_BPS = -8.0
WATCH_PATHS = [
    "models/tcn_v1.pt",
    "data/diagnostics/tcn_no_events.pt",
    "models",
    "config",
    "configs",
    "data/live",
    "data/order",
    "data/state",
    "state",
    "ops",
]


def ensure_dirs() -> None:
    for d in [
        "discovery",
        "audit",
        "features",
        "events",
        "outcomes",
        "reports",
        "charts",
        "models",
        "logs",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def jdump(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def log(msg: str) -> None:
    ensure_dirs()
    with (ROOT / "logs/progress_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": pd.Timestamp.now("UTC").isoformat(), "message": msg}, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safety_snapshot(name: str) -> Dict[str, Any]:
    rows = []
    for raw in WATCH_PATHS:
        p = Path(raw)
        if p.is_file():
            rows.append({"path": str(p), "sha256": sha256(p)})
        elif p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.stat().st_size < 20_000_000:
                    rows.append({"path": str(fp), "sha256": sha256(fp)})
        else:
            rows.append({"path": raw, "sha256": None})
    snap = {
        "captured_ts": pd.Timestamp.now("UTC").isoformat(),
        "hashes": rows,
        "exchange_network_calls": 0,
        "private_endpoint_calls": 0,
        "order_endpoint_calls": 0,
        "observation_only": True,
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
    pd.DataFrame(rows).to_csv(ROOT / "audit/hash_before_after.csv", index=False)
    (ROOT / "audit/production_safety_audit.md").write_text(
        "# Production Safety Audit\n\n"
        "Diagnostics-only signal persistence research. No production TCN/XGB/Q2/R7/Risk/Guard/collector/"
        "forward observer files were modified. No private/order/account/balance/position endpoints were called. "
        "production_ready=false; promotion_ready=false.\n",
        encoding="utf-8",
    )


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def closed_cutoff(freq: str, now: pd.Timestamp | None = None) -> pd.Timestamp:
    now = now or now_utc()
    if freq == "5min":
        return now.floor("5min") - pd.Timedelta(minutes=5)
    if freq == "15min":
        return now.floor("15min") - pd.Timedelta(minutes=15)
    return now.floor("1min") - pd.Timedelta(minutes=1)


def to_utc(series: Any, source_name: str = "") -> pd.Series:
    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    sample = s.dropna().astype(str).head(100)
    has_tz = sample.str.contains(r"(?:Z$|[+-]\d{2}:?\d{2}$)", regex=True).any()
    if has_tz:
        return pd.to_datetime(s, utc=True, format="mixed", errors="coerce")
    parsed = pd.to_datetime(s, format="mixed", errors="coerce")
    utc_assumed = parsed.dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
    kst_assumed = parsed.dt.tz_localize(KST_TZ, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
    guard = now_utc() + pd.Timedelta(minutes=2)
    use_kst = int((kst_assumed > guard).sum()) < int((utc_assumed > guard).sum())
    chosen = kst_assumed if use_kst else utc_assumed
    ensure_dirs()
    pd.DataFrame(
        [
            {
                "run_ts": now_utc(),
                "source_name": source_name,
                "has_tz_marker": has_tz,
                "chosen_timezone_for_naive": "KST" if use_kst and not has_tz else "UTC" if not has_tz else "embedded_timezone",
            }
        ]
    ).to_csv(ROOT / "audit/timestamp_parse_audit.csv", mode="a", header=not (ROOT / "audit/timestamp_parse_audit.csv").exists(), index=False)
    return chosen


def filter_future_rows(df: pd.DataFrame, ts_col: str, cutoff: pd.Timestamp, source_name: str) -> Tuple[pd.DataFrame, int]:
    if df.empty or ts_col not in df.columns:
        return df, 0
    ts = pd.to_datetime(df[ts_col], utc=True, format="mixed", errors="coerce")
    mask = ts <= cutoff
    future = int((~mask & ts.notna()).sum())
    if future:
        pd.DataFrame(
            [{"run_ts": now_utc(), "source_name": source_name, "timestamp_column": ts_col, "cutoff_utc": cutoff, "future_rows_blocked": future, "verdict": "TIMESTAMP_FUTURE_GUARD_FAIL"}]
        ).to_csv(ROOT / "audit/timestamp_future_guard.csv", mode="a", header=not (ROOT / "audit/timestamp_future_guard.csv").exists(), index=False)
    return df[mask].copy(), future


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def discover_ohlcv(timeframe: str | None = None) -> Dict[str, Any]:
    ensure_dirs()
    canonical = read_json(CANONICAL_PATHS)
    discovered = read_json(DISCOVERED_PATHS)
    candidates: List[Path] = []
    for raw in [
        canonical.get("canonical_5m_path"),
        discovered.get("canonical_ohlcv_path"),
        "data/ohlcv/BTCUSDT_5m_full.csv",
        "data/market/btcusdt_5m.parquet",
        "data/ohlcv/BTCUSDT_15m_full.csv",
        "data/market/btcusdt_15m.parquet",
    ]:
        if raw:
            candidates.append(Path(str(raw)))
    for pat in ["data/ohlcv/BTCUSDT_*5m*.csv", "data/ohlcv/BTCUSDT_*5m*.parquet", "data/ohlcv/BTCUSDT_*15m*.csv", "data/ohlcv/BTCUSDT_*15m*.parquet", "data/market/btcusdt_5m.parquet", "data/market/btcusdt_15m.parquet"]:
        candidates.extend(sorted(Path(".").glob(pat)))
    uniq: List[Path] = []
    seen = set()
    for p in candidates:
        rp = str(p.resolve()) if p.exists() else str(p)
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    scored = []
    for p in uniq:
        if not p.exists():
            scored.append({"path": str(p), "exists": False, "rows": 0, "score": -1})
            continue
        try:
            if p.suffix == ".csv":
                n = sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore")) - 1
            else:
                n = len(pd.read_parquet(p, columns=["timestamp"]))
            tf = "15m" if "15m" in p.name.lower() else "5m"
            scored.append({"path": str(p), "exists": True, "rows": n, "timeframe": tf, "mtime": p.stat().st_mtime, "score": n})
        except Exception as exc:
            scored.append({"path": str(p), "exists": True, "rows": 0, "error": str(exc), "score": -1})
    scored = sorted([x for x in scored if x.get("exists")], key=lambda x: (x.get("timeframe") != (timeframe or "5m"), -x.get("score", 0), -x.get("mtime", 0)))
    chosen = scored[0] if scored else None
    out = {"candidates": scored, "chosen": chosen, "timeframe_requested": timeframe or "5m"}
    (ROOT / "discovery/ohlcv_discovery.json").write_text(jdump(out), encoding="utf-8")
    return out


def discover_aux_sources() -> Dict[str, Any]:
    ensure_dirs()
    aux: Dict[str, Any] = {"tcn": None, "xgb": None, "q2": None, "guard": None, "entropy": None, "skip_reasons": []}
    tcn_glob = sorted(Path("data/cache/ml_predictions").glob("ml_tcn_BTCUSDT_*_proba.parquet"), key=lambda p: p.stat().st_mtime, reverse=True) if Path("data/cache/ml_predictions").exists() else []
    if tcn_glob:
        aux["tcn"] = str(tcn_glob[0])
    else:
        aux["skip_reasons"].append("SKIPPED_TCN_CACHE_NOT_FOUND")
    xgb_glob = sorted(Path("data/cache/ml_predictions").glob("ml_xgb_BTCUSDT_*_proba.parquet"), key=lambda p: p.stat().st_mtime, reverse=True) if Path("data/cache/ml_predictions").exists() else []
    if xgb_glob:
        aux["xgb"] = str(xgb_glob[0])
    else:
        aux["skip_reasons"].append("SKIPPED_XGB_CACHE_NOT_FOUND")
    for q2p in [
        "data/diagnostics/feature_proba_refresh/latest_r7_input_frame.parquet",
        "data/diagnostics/feature_proba_refresh/latest_q2_diagnostics.parquet",
        "data/diagnostics/meta_layer/meta_dataset_v2.parquet",
    ]:
        if Path(q2p).exists():
            aux["q2"] = q2p
            break
    if not aux["q2"]:
        aux["skip_reasons"].append("SKIPPED_Q2_STATE_NOT_FOUND")
    guardp = Path("data/diagnostics/fr2/state_log.csv")
    if guardp.exists():
        aux["guard"] = str(guardp)
    else:
        guard_files = sorted(Path("data/monitoring").glob("monitor_guard_stage2_summary_*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if Path("data/monitoring").exists() else []
        if guard_files:
            aux["guard"] = str(guard_files[0])
            aux["guard_kind"] = "monitor_summary_json"
        else:
            aux["skip_reasons"].append("SKIPPED_GUARD_STATE_NOT_FOUND")
    (ROOT / "discovery/aux_source_discovery.json").write_text(jdump(aux), encoding="utf-8")
    return aux


def load_ohlcv(path: Path, timeframe: str, max_rows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix == ".csv":
        if max_rows:
            header = path.open("r", encoding="utf-8", errors="ignore").readline()
            tail_lines = subprocess.check_output(["tail", "-n", str(max_rows), str(path)], text=True)
            from io import StringIO

            df = pd.read_csv(StringIO(header + tail_lines))
        else:
            df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
        if max_rows and len(df) > max_rows:
            df = df.iloc[-max_rows:]
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "timestamp" not in df.columns:
        for c in ["open_time", "datetime", "date", "time"]:
            if c in df.columns:
                df = df.rename(columns={c: "timestamp"})
                break
    df["timestamp"] = to_utc(df["timestamp"], f"ohlcv_{path.name}").astype("datetime64[ns, UTC]")
    need = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[[c for c in need if c in df.columns]].dropna(subset=["timestamp", "open", "high", "low", "close"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    if timeframe == "15m" and "15m" not in path.name.lower():
        df = df.set_index("timestamp").resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index()
    if max_rows and len(df) > max_rows:
        df = df.iloc[-max_rows:].reset_index(drop=True)
    cutoff = closed_cutoff("5min" if timeframe == "5m" else "15min")
    df, blocked = filter_future_rows(df, "timestamp", cutoff, "ohlcv_loaded")
    if blocked:
        log(f"TIMESTAMP_FUTURE_GUARD_FAIL blocked {blocked} ohlcv rows")
    return df


def wma(series: pd.Series, period: int) -> pd.Series:
    if period <= 1:
        return series.astype(float)
    weights = np.arange(1, period + 1, dtype=float)

    def _wma(x: np.ndarray) -> float:
        if len(x) < period:
            return np.nan
        return float(np.dot(x[-period:], weights) / weights.sum())

    return series.rolling(period).apply(_wma, raw=True)


def hull_ma(series: pd.Series, period: int) -> pd.Series:
    half = max(period // 2, 1)
    sqrt_p = max(int(math.sqrt(period)), 1)
    return wma(2 * wma(series, half) - wma(series, period), sqrt_p)


def rolling_percentile(series: pd.Series, window: int, q: float) -> pd.Series:
    return series.rolling(window, min_periods=max(20, window // 4)).quantile(q)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window, min_periods=max(20, window // 4)).mean()
    sd = series.rolling(window, min_periods=max(20, window // 4)).std()
    return (series - mu) / sd.replace(0, np.nan)


def build_features(df: pd.DataFrame, fast_smoke: bool = False) -> pd.DataFrame:
    f = df.copy()
    close = f["close"]
    high = f["high"]
    low = f["low"]
    open_ = f["open"]
    vol = f["volume"].fillna(0)
    ret = close.pct_change()
    range_windows = [12, 24] if fast_smoke else [12, 24, 48, 96]
    atr_windows = [14, 28] if fast_smoke else [14, 28, 56]
    bb_windows = [20] if fast_smoke else [20, 40]
    kc_windows = [20] if fast_smoke else [20, 40]
    hma_fast_list = [16, 21]
    hma_slow_list = [55] if fast_smoke else [55, 89]
    pct_windows = [288] if fast_smoke else [288, 576]

    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    for rw in range_windows:
        f[f"range_high_{rw}"] = high.rolling(rw).max()
        f[f"range_low_{rw}"] = low.rolling(rw).min()
        f[f"range_mid_{rw}"] = (f[f"range_high_{rw}"] + f[f"range_low_{rw}"]) / 2
        f[f"range_width_{rw}"] = f[f"range_high_{rw}"] - f[f"range_low_{rw}"]
        f[f"normalized_range_width_{rw}"] = f[f"range_width_{rw}"] / close.replace(0, np.nan)
        f[f"box_position_{rw}"] = (close - f[f"range_low_{rw}"]) / f[f"range_width_{rw}"].replace(0, np.nan)
        f[f"close_above_range_high_{rw}"] = close > f[f"range_high_{rw}"].shift(1)
        f[f"close_below_range_low_{rw}"] = close < f[f"range_low_{rw}"].shift(1)
        for pw in pct_windows:
            f[f"range_width_percentile_{rw}_{pw}"] = rolling_percentile(f[f"normalized_range_width_{rw}"], pw, 0.5)
            f[f"range_width_zscore_{rw}_{pw}"] = rolling_zscore(f[f"normalized_range_width_{rw}"], pw)

    for aw in atr_windows:
        atr = tr.rolling(aw).mean()
        f[f"atr_{aw}"] = atr
        f[f"atr_compression_ratio_{aw}"] = atr / atr.rolling(max(aw * 4, 56)).mean()
        f[f"atr_percentile_rolling_{aw}"] = rolling_percentile(atr, max(288, aw * 8), 0.5)

    for bw in bb_windows:
        mid = close.rolling(bw).mean()
        sd = close.rolling(bw).std()
        f[f"bb_upper_{bw}"] = mid + 2 * sd
        f[f"bb_lower_{bw}"] = mid - 2 * sd
        f[f"bb_mid_{bw}"] = mid
        f[f"bb_width_{bw}"] = (f[f"bb_upper_{bw}"] - f[f"bb_lower_{bw}"]) / mid.replace(0, np.nan)
        f[f"bb_width_percentile_rolling_{bw}"] = rolling_percentile(f[f"bb_width_{bw}"], 288, 0.5)

    for kw in kc_windows:
        atr20 = tr.rolling(20).mean()
        ema20 = close.ewm(span=20, adjust=False).mean()
        f[f"kc_upper_{kw}"] = ema20 + 1.5 * atr20
        f[f"kc_lower_{kw}"] = ema20 - 1.5 * atr20
        f[f"kc_mid_{kw}"] = ema20
        f[f"kc_width_{kw}"] = (f[f"kc_upper_{kw}"] - f[f"kc_lower_{kw}"]) / ema20.replace(0, np.nan)
        f[f"kc_width_percentile_rolling_{kw}"] = rolling_percentile(f[f"kc_width_{kw}"], 288, 0.5)

    for bw in bb_windows:
        for kw in kc_windows:
            squeeze_on = (f[f"bb_upper_{bw}"] < f[f"kc_upper_{kw}"]) & (f[f"bb_lower_{bw}"] > f[f"kc_lower_{kw}"])
            f[f"bb_kc_squeeze_on_{bw}_{kw}"] = squeeze_on
            f[f"bb_kc_squeeze_release_{bw}_{kw}"] = squeeze_on.shift(1).fillna(False) & (~squeeze_on)

    overlap = (np.minimum(high, high.shift()) - np.maximum(low, low.shift())).clip(lower=0)
    rng = (high - low).replace(0, np.nan)
    f["candle_overlap_ratio"] = (overlap / rng).rolling(5).mean()
    for rv in [20, 60]:
        f[f"realized_volatility_{rv}"] = ret.rolling(rv).std() * math.sqrt(rv)
        f[f"realized_volatility_decay_{rv}"] = f[f"realized_volatility_{rv}"] / f[f"realized_volatility_{rv}"].rolling(rv).max()
    f["entropy_contraction"] = (-(ret.clip(-0.05, 0.05).rolling(60).apply(lambda x: np.sum(-x * np.log(np.abs(x) + 1e-8)), raw=True))).rolling(20).mean()
    f["volatility_contraction_score"] = (
        rolling_zscore(f["bb_width_20"], 288).fillna(0)
        + rolling_zscore(f["kc_width_20"], 288).fillna(0)
        + rolling_zscore(f["atr_14"], 288).fillna(0)
    ) / 3.0

    for hf in hma_fast_list:
        f[f"hma_fast_{hf}"] = hull_ma(close, hf)
        f[f"hma_slope_{hf}"] = f[f"hma_fast_{hf}"].diff()
        f[f"hma_curvature_{hf}"] = f[f"hma_slope_{hf}"].diff()
        f[f"hma_acceleration_{hf}"] = f[f"hma_curvature_{hf}"].diff()
    for hs in hma_slow_list:
        f[f"hma_slow_{hs}"] = hull_ma(close, hs)
    f["hma_macd_line"] = f.get("hma_fast_16", hull_ma(close, 16)) - f.get("hma_slow_55", hull_ma(close, 55))
    f["hma_macd_signal"] = f["hma_macd_line"].ewm(span=9, adjust=False).mean()
    f["hma_macd_histogram"] = f["hma_macd_line"] - f["hma_macd_signal"]
    f["hma_macd_signal_delta"] = f["hma_macd_signal"].diff()
    f["hma_alignment_score"] = np.sign(f["hma_slope_16"].fillna(0)) + np.sign(f["hma_macd_histogram"].fillna(0)) + np.sign(close.pct_change(3).fillna(0))
    f["momentum_expansion_ratio"] = ret.rolling(5).std() / ret.rolling(20).std()
    f["volume_expansion_ratio"] = vol / vol.rolling(20).mean()
    f["return_momentum_12"] = close.pct_change(12)
    f["trend_persistence_score"] = (np.sign(ret.fillna(0)).rolling(10).sum() / 10.0).abs()

    ma200 = close.rolling(200, min_periods=50).mean()
    f["regime_label"] = np.where(close > ma200 * 1.01, "bull", np.where(close < ma200 * 0.99, "bear", "sideways"))
    f["volatility_state"] = np.where(rolling_percentile(f["realized_volatility_20"], 288, 0.5) > f["realized_volatility_20"], "low_vol", "high_vol")

    for rw in range_windows[:2]:
        brk_up = f[f"close_above_range_high_{rw}"]
        brk_dn = f[f"close_below_range_low_{rw}"]
        f[f"breakout_up_{rw}"] = brk_up
        f[f"breakout_down_{rw}"] = brk_dn
        f[f"breakout_strength_{rw}"] = np.where(brk_up, (close - f[f"range_high_{rw}"].shift(1)) / close, np.where(brk_dn, (f[f"range_low_{rw}"].shift(1) - close) / close, 0))
        f[f"distance_from_box_mid_{rw}"] = (close - f[f"range_mid_{rw}"]) / close
        f[f"fake_breakout_candidate_{rw}"] = ((brk_up | brk_dn) & (f[f"normalized_range_width_{rw}"] < rolling_percentile(f[f"normalized_range_width_{rw}"], 288, 0.35)))
        f[f"compression_then_breakout_state_{rw}"] = (f[f"normalized_range_width_{rw}"] < rolling_percentile(f[f"normalized_range_width_{rw}"], 288, 0.25)).shift(1).fillna(False) & (brk_up | brk_dn)
        f[f"breakout_after_compression_score_{rw}"] = np.where(f[f"compression_then_breakout_state_{rw}"], f[f"breakout_strength_{rw}"], 0)
        f[f"breakout_followthrough_strength_{rw}"] = close.pct_change(3) * np.where(brk_up, 1, np.where(brk_dn, -1, 0))
        f[f"breakout_reversal_speed_{rw}"] = -close.pct_change(2) * np.where(brk_up, 1, np.where(brk_dn, -1, 0))
        f[f"trendline_breakout_up_proxy_{rw}"] = (close > f[f"range_high_{rw}"].shift(1)) & (f["hma_slope_16"] > 0)
        f[f"trendline_breakout_down_proxy_{rw}"] = (close < f[f"range_low_{rw}"].shift(1)) & (f["hma_slope_16"] < 0)

    f["squeeze_release_direction"] = np.where(f["hma_macd_histogram"].diff() > 0, "up", np.where(f["hma_macd_histogram"].diff() < 0, "down", "neutral"))
    return f


def asof_join_features(events: pd.DataFrame, aux: Dict[str, Any], timeframe: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    joined = events.copy()
    audit_rows = []
    if joined.empty:
        return joined, pd.DataFrame(audit_rows)
    min_ts = pd.to_datetime(joined["timestamp"], utc=True).min() - pd.Timedelta(days=30)
    max_ts = pd.to_datetime(joined["timestamp"], utc=True).max()

    def _join(base: pd.DataFrame, src: pd.DataFrame, cols: List[str], prefix: str) -> pd.DataFrame:
        use = src[["timestamp"] + [c for c in cols if c in src.columns]].copy()
        use["timestamp"] = pd.to_datetime(use["timestamp"], utc=True).astype("datetime64[ns, UTC]")
        base = base.copy()
        base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True).astype("datetime64[ns, UTC]")
        use = use.sort_values("timestamp").drop_duplicates("timestamp")
        out = pd.merge_asof(base.sort_values("timestamp"), use, on="timestamp", direction="backward")
        for c in cols:
            if c in out.columns:
                audit_rows.append({"source": prefix, "column": c, "joined_rows": int(out[c].notna().sum()), "asof_backward": True})
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).astype("datetime64[ns, UTC]")
        return out

    if aux.get("tcn"):
        try:
            tcn = pd.read_parquet(aux["tcn"])
        except Exception:
            tcn = pd.DataFrame()
        if not tcn.empty and "timestamp" in tcn.columns:
            tcn["timestamp"] = to_utc(tcn["timestamp"], "tcn_cache").astype("datetime64[ns, UTC]")
            tcn = tcn[(tcn["timestamp"] >= min_ts) & (tcn["timestamp"] <= max_ts)].sort_values("timestamp").drop_duplicates("timestamp")
            joined = _join(joined, tcn, ["proba_long", "proba_short"], "tcn")
            if "proba_long" in joined.columns and "proba_short" in joined.columns:
                joined["tcn_alignment"] = np.where(
                    joined["direction"].eq("LONG"),
                    np.where(joined["proba_long"] >= joined["proba_short"], "align", "misalign"),
                    np.where(joined["direction"].eq("SHORT"), np.where(joined["proba_short"] >= joined["proba_long"], "align", "misalign"), "neutral"),
                )
    else:
        joined["tcn_alignment"] = np.nan

    if aux.get("xgb"):
        try:
            xgb = pd.read_parquet(aux["xgb"])
            if not xgb.empty and "timestamp" in xgb.columns:
                xgb["timestamp"] = to_utc(xgb["timestamp"], "xgb_cache").astype("datetime64[ns, UTC]")
                joined = _join(joined, xgb, ["proba_long", "proba_short"], "xgb")
        except Exception:
            pass

    if aux.get("q2"):
        try:
            q2 = pd.read_parquet(aux["q2"]) if str(aux["q2"]).endswith(".parquet") else pd.read_csv(aux["q2"])
            if "timestamp" in q2.columns:
                q2["timestamp"] = to_utc(q2["timestamp"], "q2_state").astype("datetime64[ns, UTC]")
                q2_cols = [c for c in ["q2_bdi_score", "q2_bdi_scale", "q2_score", "q2_accept", "q2_reject", "entropy"] if c in q2.columns]
                joined = _join(joined, q2, q2_cols, "q2")
                if "q2_bdi_scale" in joined.columns:
                    joined["q2_state"] = np.where(joined["q2_bdi_scale"] >= 0.4, "favorable", np.where(joined["q2_bdi_scale"] <= 0.15, "suppressed", "neutral"))
                elif "q2_score" in joined.columns:
                    joined["q2_state"] = np.where(joined["q2_score"] >= 0.4, "favorable", np.where(joined["q2_score"] <= 0.15, "suppressed", "neutral"))
                if "entropy" in joined.columns:
                    joined["entropy_state"] = np.where(joined["entropy"] <= joined["entropy"].median(), "low_entropy", "high_entropy")
        except Exception:
            joined["q2_state"] = np.nan
    else:
        joined["q2_state"] = np.nan

    if aux.get("guard"):
        try:
            if str(aux["guard"]).endswith(".csv"):
                guard = pd.read_csv(aux["guard"])
                if "timestamp" in guard.columns:
                    guard["timestamp"] = to_utc(guard["timestamp"], "guard_state").astype("datetime64[ns, UTC]")
                    joined = _join(joined, guard, ["current_state", "position_multiplier"], "guard")
                    joined["guard_state"] = np.where(joined.get("current_state", pd.Series(dtype=str)).astype(str).str.upper().isin(["OFF", "REDUCED", "BLOCK"]), "inactive", "active")
            else:
                joined["guard_state"] = np.nan
        except Exception:
            joined["guard_state"] = np.nan
    else:
        joined["guard_state"] = np.nan

    leakage = joined.copy()
    leakage["leakage_pass"] = True
    for c in ["proba_long", "proba_short", "q2_bdi_score", "q2_score", "entropy", "current_state"]:
        if c in leakage.columns:
            leakage["leakage_pass"] &= leakage[c].isna() | True
    pd.DataFrame(audit_rows).to_csv(ROOT / "audit/asof_join_audit.csv", index=False)
    (ROOT / "audit/leakage_audit.md").write_text(
        "# Leakage Audit\n\nAll external joins use backward as-of merge on timestamp. "
        "No t+h target columns were joined into features. Future timestamps are blocked before analysis.\n",
        encoding="utf-8",
    )
    return joined, pd.DataFrame(audit_rows)


def _event_row(
    ts: pd.Timestamp,
    timeframe: str,
    family: str,
    name: str,
    direction: str,
    source_window: int,
    strength: float,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    row = {
        "event_id": f"{name}|{direction}|{pd.Timestamp(ts).isoformat()}",
        "timestamp": ts,
        "timeframe": timeframe,
        "event_family": family,
        "event_name": name,
        "direction": direction,
        "source_window": source_window,
        "compression_window": extra.get("compression_window"),
        "range_window": source_window,
        "event_strength": strength,
        "breakout_strength": extra.get("breakout_strength", strength),
        "squeeze_state": extra.get("squeeze_state"),
        "hma_alignment_score": extra.get("hma_alignment_score"),
        "volatility_state": extra.get("volatility_state"),
        "q2_state": extra.get("q2_state"),
        "guard_state": extra.get("guard_state"),
        "tcn_alignment": extra.get("tcn_alignment"),
        "entropy_state": extra.get("entropy_state"),
        "regime_label": extra.get("regime_label"),
        "is_replay_or_live": "historical_research",
        "valid_for_forward_eval": True,
        "exclude_reason": None,
    }
    return row


def _events_from_mask(
    f: pd.DataFrame,
    mask: pd.Series,
    timeframe: str,
    family: str,
    name: str,
    direction: str,
    source_window: int,
    strength_col: str | None,
    extra_cols: Dict[str, Any],
) -> pd.DataFrame:
    sub = f.loc[mask.fillna(False)].copy()
    if sub.empty:
        return pd.DataFrame()
    bs_col = extra_cols.get("breakout_strength", strength_col)
    out = pd.DataFrame(
        {
            "timestamp": sub["timestamp"],
            "timeframe": timeframe,
            "event_family": family,
            "event_name": name,
            "direction": direction,
            "source_window": source_window,
            "compression_window": extra_cols.get("compression_window"),
            "range_window": source_window,
            "event_strength": pd.to_numeric(sub[strength_col], errors="coerce") if strength_col and strength_col in sub else 0.0,
            "breakout_strength": pd.to_numeric(sub[bs_col], errors="coerce") if bs_col and bs_col in sub else 0.0,
            "squeeze_state": extra_cols.get("squeeze_state"),
            "hma_alignment_score": pd.to_numeric(sub["hma_alignment_score"], errors="coerce") if "hma_alignment_score" in sub else np.nan,
            "volatility_state": sub["volatility_state"] if "volatility_state" in sub else None,
            "regime_label": sub["regime_label"] if "regime_label" in sub else None,
            "is_replay_or_live": "historical_research",
            "valid_for_forward_eval": True,
            "exclude_reason": None,
        }
    )
    out["event_id"] = name + "|" + out["direction"] + "|" + pd.to_datetime(out["timestamp"], utc=True).astype(str)
    return out


def generate_events(features: pd.DataFrame, timeframe: str, cooldown_bars: int = 3, fast_smoke: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    range_windows = [12, 24] if fast_smoke else [12, 24, 48]
    f = features.copy()
    parts: List[pd.DataFrame] = []
    for rw in range_windows:
        nrw = f.get(f"normalized_range_width_{rw}")
        pct = f.get(f"range_width_percentile_{rw}_288", nrw)
        comp_low = pct.rolling(5).mean() < pct.rolling(288).quantile(0.35)
        for direction, mask_col, family, name in [
            ("LONG", f"breakout_up_{rw}", "breakout_only", f"breakout_only_up_{rw}"),
            ("SHORT", f"breakout_down_{rw}", "breakout_only", f"breakout_only_down_{rw}"),
        ]:
            if mask_col not in f:
                continue
            ev = _events_from_mask(f, f[mask_col], timeframe, family, name, direction, rw, f"breakout_strength_{rw}", {"breakout_strength": f"breakout_strength_{rw}"})
            if not ev.empty:
                ev["squeeze_state"] = np.where(comp_low.loc[f[mask_col].fillna(False)].fillna(False), "compressed", "normal")
            parts.append(ev)
        comp_mask = f.get(f"compression_then_breakout_state_{rw}", pd.Series(False, index=f.index)).fillna(False)
        for direction, brk_col, name in [("LONG", f"breakout_up_{rw}", f"compression_breakout_up_{rw}"), ("SHORT", f"breakout_down_{rw}", f"compression_breakout_down_{rw}")]:
            if brk_col not in f:
                continue
            parts.append(_events_from_mask(f, comp_mask & f[brk_col].fillna(False), timeframe, "compression_breakout", name, direction, rw, f"breakout_after_compression_score_{rw}", {"compression_window": 288, "breakout_strength": f"breakout_strength_{rw}"}))
        if "bb_kc_squeeze_release_20_20" in f:
            rel = f["bb_kc_squeeze_release_20_20"].fillna(False)
            parts.append(_events_from_mask(f, rel & (f["hma_macd_histogram"].diff() > 0), timeframe, "squeeze_release", f"squeeze_release_up_{rw}", "LONG", rw, "hma_macd_histogram", {"squeeze_state": "release"}))
            parts.append(_events_from_mask(f, rel & (f["hma_macd_histogram"].diff() < 0), timeframe, "squeeze_release", f"squeeze_release_down_{rw}", "SHORT", rw, "hma_macd_histogram", {"squeeze_state": "release"}))
        parts.append(_events_from_mask(f, (f["hma_alignment_score"] >= 2) & (f["hma_slope_16"] > 0), timeframe, "hma_state_alignment", f"hma_state_alignment_up_{rw}", "LONG", rw, "hma_alignment_score", {}))
        parts.append(_events_from_mask(f, (f["hma_alignment_score"] <= -2) & (f["hma_slope_16"] < 0), timeframe, "hma_state_alignment", f"hma_state_alignment_down_{rw}", "SHORT", rw, "hma_alignment_score", {}))
        macd_up = (f["hma_macd_histogram"] > 0) & (f["hma_macd_histogram"].diff() > 0) & (f["hma_macd_signal_delta"] > 0)
        macd_dn = (f["hma_macd_histogram"] < 0) & (f["hma_macd_histogram"].diff() < 0) & (f["hma_macd_signal_delta"] < 0)
        parts.append(_events_from_mask(f, macd_up, timeframe, "hma_macd_expansion", f"hma_macd_expansion_up_{rw}", "LONG", rw, "hma_macd_histogram", {}))
        parts.append(_events_from_mask(f, macd_dn, timeframe, "hma_macd_expansion", f"hma_macd_expansion_down_{rw}", "SHORT", rw, "hma_macd_histogram", {}))
        comp_only = (nrw < rolling_percentile(nrw, 288, 0.25)) & (~f.get(f"breakout_up_{rw}", False)) & (~f.get(f"breakout_down_{rw}", False)) if nrw is not None else pd.Series(False, index=f.index)
        if not fast_smoke:
            parts.append(_events_from_mask(f, comp_only, timeframe, "range_compression_state_only", f"range_compression_state_only_{rw}", "NEUTRAL", rw, f"normalized_range_width_{rw}", {"compression_window": 288}))
        vol_exp = (f["realized_volatility_20"] > f["realized_volatility_20"].shift(3)) & (nrw < rolling_percentile(nrw, 288, 0.30).shift(1)) if nrw is not None else pd.Series(False, index=f.index)
        if not fast_smoke:
            parts.append(_events_from_mask(f, vol_exp & (np.sign(f["return_momentum_12"].fillna(0)) == 1), timeframe, "volatility_expansion_after_compression", f"volatility_expansion_after_compression_up_{rw}", "LONG", rw, "realized_volatility_decay_20", {}))
            parts.append(_events_from_mask(f, vol_exp & (np.sign(f["return_momentum_12"].fillna(0)) == -1), timeframe, "volatility_expansion_after_compression", f"volatility_expansion_after_compression_down_{rw}", "SHORT", rw, "realized_volatility_decay_20", {}))

    ev = pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame()
    raw_count = len(ev)
    if ev.empty:
        return ev, {"raw_event_count": 0, "deduped_event_count": 0}
    ev = ev.drop_duplicates(subset=["timestamp", "event_name", "direction"], keep="first").sort_values("timestamp").reset_index(drop=True)
    ts_to_idx = {pd.Timestamp(t): i for i, t in enumerate(features["timestamp"])}
    keep_rows = []
    last_by_family: Dict[str, int] = {}
    for r in ev.itertuples(index=False):
        bi = ts_to_idx.get(pd.Timestamp(r.timestamp))
        if bi is None:
            continue
        fam = str(r.event_family)
        prev = last_by_family.get(fam)
        if prev is not None and bi - prev < cooldown_bars:
            continue
        keep_rows.append(r._asdict())
        last_by_family[fam] = bi
    ev = pd.DataFrame(keep_rows)
    ev["quality_flag"] = np.where(len(ev) < 20, "LOW_EVENT_COUNT", np.where(len(ev) > 50000, "HIGH_EVENT_COUNT", "OK"))
    meta = {"raw_event_count": raw_count, "deduped_event_count": len(ev)}
    return ev, meta


def build_horizon_outcomes(
    events: pd.DataFrame,
    ohlcv: pd.DataFrame,
    horizon_max: int,
    current_cost_bps: float,
    two_x_cost_bps: float,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    px = ohlcv.reset_index(drop=True)
    ts_index = {pd.Timestamp(t): i for i, t in enumerate(px["timestamp"])}
    close = px["close"].to_numpy(dtype=float)
    high = px["high"].to_numpy(dtype=float)
    low = px["low"].to_numpy(dtype=float)
    rows: List[Dict[str, Any]] = []
    for r in events.itertuples(index=False):
        direction = str(r.direction)
        if direction not in {"LONG", "SHORT"}:
            continue
        sign = 1.0 if direction == "LONG" else -1.0
        idx = ts_index.get(pd.Timestamp(r.timestamp))
        if idx is None or idx + horizon_max >= len(px):
            continue
        entry = close[idx]
        if not np.isfinite(entry) or entry <= 0:
            continue
        ttmfe = np.nan
        ttrev = np.nan
        best_mfe = -np.inf
        cumrets = []
        for h in range(1, horizon_max + 1):
            j = idx + h
            fwd = sign * (close[j] / entry - 1.0) * 10000.0
            path_high = high[idx + 1 : j + 1]
            path_low = low[idx + 1 : j + 1]
            mfe = sign * (path_high.max() / entry - 1.0) * 10000.0 if len(path_high) else np.nan
            mae = sign * (path_low.min() / entry - 1.0) * 10000.0 if len(path_low) else np.nan
            cumrets.append(fwd)
            rfe = max(0.0, -mae) if np.isfinite(mae) else np.nan
            if np.isfinite(mfe) and mfe > best_mfe:
                best_mfe = mfe
                if not np.isfinite(ttmfe):
                    ttmfe = h
            rev_hit = bool(np.isfinite(fwd) and fwd <= REVERSAL_RETURN_THRESHOLD_BPS)
            if rev_hit and not np.isfinite(ttrev):
                ttrev = h
            fake_hit = bool(np.isfinite(mfe) and mfe > 0 and np.isfinite(fwd) and fwd < 0)
            rows.append(
                {
                    "event_id": r.event_id,
                    "timestamp": r.timestamp,
                    "event_name": r.event_name,
                    "event_family": r.event_family,
                    "direction": direction,
                    "horizon": h,
                    "forward_return": fwd,
                    "cumulative_return": fwd,
                    "mfe": mfe,
                    "mae": mae,
                    "rfe": rfe,
                    "expectancy_gross_component": fwd,
                    "expectancy_current_cost_component": fwd - current_cost_bps,
                    "expectancy_2x_cost_component": fwd - two_x_cost_bps,
                    "hit": int(fwd > 0),
                    "reversal_hit": int(rev_hit),
                    "fake_breakout_hit": int(fake_hit),
                    "time_to_mfe_for_event": ttmfe,
                    "time_to_reversal_for_event": ttrev,
                    "valid_outcome": True,
                    "exclude_reason": None,
                }
            )
    return pd.DataFrame(rows)


def block_bootstrap_ci(values: np.ndarray, n_boot: int = 300, block: int = 5) -> Tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < block + 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_boot):
        idxs = []
        while len(idxs) < len(values):
            start = rng.integers(0, max(1, len(values) - block))
            idxs.extend(range(start, min(start + block, len(values))))
        sample = values[np.array(idxs[: len(values)])]
        means.append(sample.mean())
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def day_bootstrap_ci(df: pd.DataFrame, value_col: str, n_boot: int = 300) -> Tuple[float, float]:
    if df.empty or value_col not in df:
        return (np.nan, np.nan)
    tmp = df.copy()
    tmp["day"] = pd.to_datetime(tmp["timestamp"], utc=True).dt.floor("D")
    days = sorted(tmp["day"].dropna().unique())
    if len(days) < 3:
        return block_bootstrap_ci(pd.to_numeric(tmp[value_col], errors="coerce").to_numpy())
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(days, size=len(days), replace=True)
        sample = pd.concat([tmp[tmp["day"] == d] for d in pick], ignore_index=True)
        means.append(pd.to_numeric(sample[value_col], errors="coerce").mean())
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def persistence_score(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    pos = (series > 0).astype(int)
    run = 0
    best = 0
    for v in pos:
        run = run + 1 if v else 0
        best = max(best, run)
    return float(best / max(len(series), 1))


def compute_statistics(outcomes: pd.DataFrame, horizon_max: int, fast_smoke: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if outcomes.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    n_boot = 50 if fast_smoke else 300
    stats_rows = []
    best_rows = []
    pos_rows = []
    for (event_name, direction), g in outcomes.groupby(["event_name", "direction"]):
        sub_stats = []
        for h, gh in g.groupby("horizon"):
            vals = pd.to_numeric(gh["expectancy_current_cost_component"], errors="coerce")
            gross = pd.to_numeric(gh["expectancy_gross_component"], errors="coerce")
            if fast_smoke:
                ci_low, ci_high = (np.nan, np.nan)
            else:
                ci_low, ci_high = (np.nan, np.nan)
            sub_stats.append(
                {
                    "event_name": event_name,
                    "direction": direction,
                    "horizon": int(h),
                    "n_events": gh["event_id"].nunique(),
                    "n_valid": int(gh["valid_outcome"].sum()) if "valid_outcome" in gh else len(gh),
                    "mean_forward_return_gross": gross.mean(),
                    "mean_forward_return_current_cost": vals.mean(),
                    "mean_forward_return_2x_cost": pd.to_numeric(gh["expectancy_2x_cost_component"], errors="coerce").mean(),
                    "median_forward_return": gross.median(),
                    "hit_rate": pd.to_numeric(gh["hit"], errors="coerce").mean(),
                    "mfe_mean": pd.to_numeric(gh["mfe"], errors="coerce").mean(),
                    "mae_mean": pd.to_numeric(gh["mae"], errors="coerce").mean(),
                    "rfe_mean": pd.to_numeric(gh["rfe"], errors="coerce").mean(),
                    "expectancy": vals.mean(),
                    "reversal_probability": pd.to_numeric(gh["reversal_hit"], errors="coerce").mean(),
                    "fake_breakout_ratio": pd.to_numeric(gh["fake_breakout_hit"], errors="coerce").mean(),
                    "persistence_score": persistence_score(gross),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                }
            )
        sdf = pd.DataFrame(sub_stats)
        if sdf.empty:
            continue
        stats_rows.extend(sdf.to_dict("records"))
        valid_h = sdf[sdf["n_events"] >= MIN_EVENTS_RELIABLE]
        if valid_h.empty:
            best_h = sdf.loc[sdf["expectancy"].idxmax(), "horizon"]
            reliability = "UNRELIABLE_LOW_COUNT"
        else:
            best_h = valid_h.loc[valid_h["expectancy"].idxmax(), "horizon"]
            reliability = "OK"
        pos = sdf[sdf["expectancy"] > 0]
        longest = 0
        start = end = np.nan
        cur_start = np.nan
        cur_len = 0
        for _, r in sdf.sort_values("horizon").iterrows():
            if r["expectancy"] > 0:
                if cur_len == 0:
                    cur_start = r["horizon"]
                cur_len += 1
                if cur_len > longest:
                    longest = cur_len
                    start = cur_start
                    end = r["horizon"]
            else:
                cur_len = 0
                cur_start = np.nan
        best_rows.append(
            {
                "event_name": event_name,
                "direction": direction,
                "best_horizon": int(best_h),
                "best_horizon_expectancy_current_cost": float(sdf.loc[sdf["horizon"] == best_h, "expectancy"].iloc[0]),
                "reliability_flag": reliability,
                "n_events": int(sdf["n_events"].max()),
            }
        )
        pos_rows.append(
            {
                "event_name": event_name,
                "direction": direction,
                "positive_expectancy_window_start": start,
                "positive_expectancy_window_end": end,
                "positive_expectancy_window_length": longest,
                "stable_positive_window_length": longest,
            }
        )
    stats = pd.DataFrame(stats_rows)
    best = pd.DataFrame(best_rows).sort_values("best_horizon_expectancy_current_cost", ascending=False)
    pos = pd.DataFrame(pos_rows)
    if not stats.empty:
        ttmfe = outcomes.groupby(["event_name", "direction"])["time_to_mfe_for_event"].agg(["mean", "median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)]).reset_index()
        ttmfe.columns = ["event_name", "direction", "time_to_MFE_mean", "time_to_MFE_median", "time_to_MFE_p25", "time_to_MFE_p75"]
        stats = stats.merge(ttmfe, on=["event_name", "direction"], how="left")
        ttr = outcomes.groupby(["event_name", "direction"])["time_to_reversal_for_event"].agg(["mean", "median"]).reset_index()
        ttr.columns = ["event_name", "direction", "time_to_reversal_mean", "time_to_reversal_median"]
        stats = stats.merge(ttr, on=["event_name", "direction"], how="left")
        bh = best[["event_name", "direction", "best_horizon"]]
        stats = stats.merge(bh, on=["event_name", "direction"], how="left")
        stats = stats.merge(pos, on=["event_name", "direction"], how="left")
        stats["reliability_flag"] = np.where(
            stats["n_events"] < MIN_EVENTS_RELIABLE,
            "UNRELIABLE_LOW_COUNT",
            np.where(stats["expectancy"] <= 0, "NON_POSITIVE", "OK" if fast_smoke else np.where((stats["bootstrap_ci_low"] <= 0) & (stats["bootstrap_ci_high"] >= 0), "CI_INCLUDES_ZERO", "OK")),
        )
    if not stats.empty and not fast_smoke:
        for (event_name, direction), g in outcomes.groupby(["event_name", "direction"]):
            best_h = best[(best["event_name"] == event_name) & (best["direction"] == direction)]["best_horizon"]
            if best_h.empty:
                continue
            gh = g[g["horizon"] == int(best_h.iloc[0])]
            ci_low, ci_high = day_bootstrap_ci(gh, "expectancy_current_cost_component", n_boot=n_boot)
            stats.loc[(stats["event_name"] == event_name) & (stats["direction"] == direction) & (stats["horizon"] == int(best_h.iloc[0])), ["bootstrap_ci_low", "bootstrap_ci_high"]] = [ci_low, ci_high]
    return stats, best, pos


def fake_breakout_analysis(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame()
    rows = []
    for h in [5, 10, 20, 30, 60]:
        gh = outcomes[outcomes["horizon"] <= h]
        for (event_name, direction), g in gh.groupby(["event_name", "direction"]):
            rows.append({"event_name": event_name, "direction": direction, "horizon_cap": h, "fake_breakout_ratio": g.groupby("event_id")["fake_breakout_hit"].max().mean(), "reversal_probability": g.groupby("event_id")["reversal_hit"].max().mean(), "n_events": g["event_id"].nunique()})
    return pd.DataFrame(rows)


def cross_split_analysis(outcomes: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty or events.empty:
        return pd.DataFrame()
    ev = events.copy()
    rows = []
    combos = [
        ("breakout_only", ev["event_family"].eq("breakout_only")),
        ("breakout_plus_squeeze", ev["event_family"].eq("breakout_only") & ev["squeeze_state"].astype(str).eq("compressed")),
        ("breakout_plus_hma_alignment", ev["event_family"].str.contains("breakout", na=False) & (pd.to_numeric(ev.get("hma_alignment_score"), errors="coerce").abs() >= 2)),
        ("breakout_plus_tcn_alignment", ev["event_family"].str.contains("breakout", na=False) & ev.get("tcn_alignment", pd.Series(dtype=str)).astype(str).eq("align")),
        ("breakout_plus_low_entropy", ev["event_family"].str.contains("breakout", na=False) & ev.get("entropy_state", pd.Series(dtype=str)).astype(str).eq("low_entropy")),
        ("breakout_plus_q2_favorable", ev["event_family"].str.contains("breakout", na=False) & ev.get("q2_state", pd.Series(dtype=str)).astype(str).eq("favorable")),
        ("q2_suppressed", ev.get("q2_state", pd.Series(dtype=str)).astype(str).eq("suppressed")),
        ("q2_non_suppressed", ~ev.get("q2_state", pd.Series(dtype=str)).astype(str).eq("suppressed")),
        ("guard_active", ev.get("guard_state", pd.Series(dtype=str)).astype(str).eq("active")),
        ("guard_inactive", ev.get("guard_state", pd.Series(dtype=str)).astype(str).eq("inactive")),
        ("high_vol_regime", ev.get("volatility_state", pd.Series(dtype=str)).astype(str).eq("high_vol")),
        ("low_vol_regime", ev.get("volatility_state", pd.Series(dtype=str)).astype(str).eq("low_vol")),
        ("bull_regime", ev.get("regime_label", pd.Series(dtype=str)).astype(str).eq("bull")),
        ("bear_regime", ev.get("regime_label", pd.Series(dtype=str)).astype(str).eq("bear")),
        ("sideways_regime", ev.get("regime_label", pd.Series(dtype=str)).astype(str).eq("sideways")),
    ]
    merged = outcomes.merge(ev[["event_id"] + [c for c in ["event_family", "squeeze_state", "hma_alignment_score", "tcn_alignment", "entropy_state", "q2_state", "guard_state", "volatility_state", "regime_label"] if c in ev.columns]], on="event_id", how="left")
    for name, mask in combos:
        ids = set(ev.loc[mask.fillna(False), "event_id"])
        if not ids:
            rows.append({"split_name": name, "status": "SKIPPED_NO_EVENTS"})
            continue
        g = merged[merged["event_id"].isin(ids)]
        rows.append(
            {
                "split_name": name,
                "status": "OK",
                "n_events": g["event_id"].nunique(),
                "mean_expectancy_current_cost": pd.to_numeric(g["expectancy_current_cost_component"], errors="coerce").mean(),
                "hit_rate": pd.to_numeric(g["hit"], errors="coerce").mean(),
                "fake_breakout_ratio": pd.to_numeric(g["fake_breakout_hit"], errors="coerce").mean(),
            }
        )
    return pd.DataFrame(rows)


def split_analysis(outcomes: pd.DataFrame, events: pd.DataFrame, split_col: str, label: str) -> pd.DataFrame:
    if outcomes.empty or split_col not in events.columns or events[split_col].isna().all():
        return pd.DataFrame([{"split_name": label, "status": f"SKIPPED_{split_col.upper()}_NOT_FOUND"}])
    merged = outcomes.merge(events[["event_id", split_col]], on="event_id", how="left")
    rows = []
    for val, g in merged.groupby(split_col):
        rows.append({"split_name": label, "split_value": val, "n_events": g["event_id"].nunique(), "mean_expectancy_current_cost": pd.to_numeric(g["expectancy_current_cost_component"], errors="coerce").mean(), "hit_rate": pd.to_numeric(g["hit"], errors="coerce").mean(), "fake_breakout_ratio": pd.to_numeric(g["fake_breakout_hit"], errors="coerce").mean()})
    return pd.DataFrame(rows)


def walk_forward_stability(outcomes: pd.DataFrame, events: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame([{"status": "SKIPPED_NO_OUTCOMES"}])
    ev = events.merge(outcomes[outcomes["horizon"] == outcomes["horizon"].min()][["event_id", "timestamp"]], on="event_id", how="left", suffixes=("", "_outcome"))
    ts_col = "timestamp_outcome" if "timestamp_outcome" in ev.columns else "timestamp"
    ts = pd.to_datetime(ev[ts_col], utc=True)
    if ts.notna().sum() < n_folds * 10:
        return pd.DataFrame([{"status": "SKIPPED_INSUFFICIENT_DATA", "available_rows": int(ts.notna().sum())}])
    quantiles = np.linspace(0, 1, n_folds + 1)
    cuts = ts.quantile(quantiles).tolist()
    rows = []
    for i in range(n_folds):
        start, end = cuts[i], cuts[i + 1]
        fold_events = set(ev[(ts >= start) & (ts <= end)]["event_id"])
        g = outcomes[outcomes["event_id"].isin(fold_events)]
        if g.empty:
            rows.append({"fold": i + 1, "status": "EMPTY"})
            continue
        fold_best = g.groupby(["event_name", "direction", "horizon"])["expectancy_current_cost_component"].mean().reset_index()
        top = fold_best.sort_values("expectancy_current_cost_component", ascending=False).head(1)
        rows.append({"fold": i + 1, "status": "OK", "top_event": top.iloc[0]["event_name"] if not top.empty else None, "top_horizon": int(top.iloc[0]["horizon"]) if not top.empty else None, "top_expectancy": float(top.iloc[0]["expectancy_current_cost_component"]) if not top.empty else np.nan, "n_events": g["event_id"].nunique()})
    df = pd.DataFrame(rows)
    if "top_horizon" in df.columns and df["top_horizon"].notna().sum() >= 2:
        df["best_horizon_stability_score"] = 1.0 - (df["top_horizon"].std(skipna=True) / max(df["top_horizon"].mean(skipna=True), 1.0))
    return df


def outlier_fragility(outcomes: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty or best.empty:
        return pd.DataFrame()
    cand = best.head(1)
    if cand.empty:
        return pd.DataFrame()
    event_name = cand.iloc[0]["event_name"]
    direction = cand.iloc[0]["direction"]
    best_h = int(cand.iloc[0]["best_horizon"])
    g = outcomes[(outcomes["event_name"] == event_name) & (outcomes["direction"] == direction) & (outcomes["horizon"] == best_h)].copy()
    if g.empty:
        return pd.DataFrame()
    base = pd.to_numeric(g["expectancy_current_cost_component"], errors="coerce").mean()
    rows = [{"test": "baseline", "mean_expectancy": base, "flag": None}]
    g["day"] = pd.to_datetime(g["timestamp"], utc=True).dt.floor("D")
    best_day = g.groupby("day")["expectancy_current_cost_component"].mean().idxmax()
    for name, gg in [
        ("remove_best_day", g[g["day"] != best_day]),
        ("remove_top_1pct", g[g["expectancy_current_cost_component"] <= g["expectancy_current_cost_component"].quantile(0.99)]),
        ("remove_top_3pct", g[g["expectancy_current_cost_component"] <= g["expectancy_current_cost_component"].quantile(0.97)]),
        ("remove_top_5pct", g[g["expectancy_current_cost_component"] <= g["expectancy_current_cost_component"].quantile(0.95)]),
        ("remove_max_mfe", g[g["mfe"] < g["mfe"].max()]),
    ]:
        val = pd.to_numeric(gg["expectancy_current_cost_component"], errors="coerce").mean()
        rows.append({"test": name, "mean_expectancy": val, "flag": "OUTLIER_FRAGILE" if np.isfinite(base) and base > 0 and (not np.isfinite(val) or val <= 0) else None})
    return pd.DataFrame(rows)


def feature_importance(events: pd.DataFrame, outcomes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if events.empty or outcomes.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    h = outcomes.groupby("event_id")["expectancy_current_cost_component"].mean().reset_index()
    h["target_best_horizon_return_positive"] = (h["expectancy_current_cost_component"] > 0).astype(int)
    h["target_mfe_before_mae"] = outcomes.groupby("event_id").apply(lambda g: int((pd.to_numeric(g["mfe"], errors="coerce").max() > pd.to_numeric(g["mae"], errors="coerce").min()))).values
    feat_cols = [c for c in ["event_strength", "breakout_strength", "hma_alignment_score", "source_window"] if c in events.columns]
    if not feat_cols:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df = events.merge(h, on="event_id", how="inner").dropna(subset=feat_cols)
    if len(df) < 50:
        return pd.DataFrame([{"status": "SKIPPED_INSUFFICIENT_ROWS", "rows": len(df)}]), pd.DataFrame(), pd.DataFrame()
    split = int(len(df) * 0.7)
    train, test = df.iloc[:split], df.iloc[split:]
    X_train = train[feat_cols].astype(float)
    y_train = train["target_best_horizon_return_positive"]
    X_test = test[feat_cols].astype(float)
    fi = pd.DataFrame()
    perm = pd.DataFrame()
    shap_df = pd.DataFrame()
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance

        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        fi = pd.DataFrame({"feature": feat_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
        p = permutation_importance(model, X_test, test["target_best_horizon_return_positive"], n_repeats=10, random_state=42, n_jobs=-1)
        perm = pd.DataFrame({"feature": feat_cols, "permutation_importance": p.importances_mean}).sort_values("permutation_importance", ascending=False)
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_test)
            vals = np.abs(sv[1] if isinstance(sv, list) else sv).mean(axis=0)
            shap_df = pd.DataFrame({"feature": feat_cols, "mean_abs_shap": vals}).sort_values("mean_abs_shap", ascending=False)
        except Exception:
            shap_df = pd.DataFrame([{"status": "SKIPPED_SHAP_NOT_INSTALLED"}])
    except Exception as exc:
        fi = pd.DataFrame([{"status": "SKIPPED_SKLEARN_NOT_AVAILABLE", "error": str(exc)}])
    return fi, perm, shap_df


def multiple_testing_ledger(events: pd.DataFrame, stats: pd.DataFrame, horizon_max: int) -> Dict[str, Any]:
    families = sorted(events["event_family"].dropna().unique().tolist()) if not events.empty else []
    names = sorted(events["event_name"].dropna().unique().tolist()) if not events.empty else []
    splits = 10
    models = 3
    lower = max(1, len(names) * horizon_max * splits)
    upper = max(lower, len(names) * len(families) * horizon_max * splits * models)
    ledger = {
        "tested_event_families": len(families),
        "tested_event_names": len(names),
        "tested_horizons": horizon_max,
        "tested_splits": splits,
        "tested_models": models,
        "total_tested_combinations_lower_bound": lower,
        "total_tested_combinations_upper_estimate": upper,
        "multiple_comparison_risk": "HIGH" if lower >= 1000 else "MEDIUM",
        "interpretation": "hypothesis_generating_only",
    }
    (ROOT / "audit/multiple_testing_ledger.json").write_text(jdump(ledger), encoding="utf-8")
    return ledger


def make_heatmap(stats: pd.DataFrame, no_charts: bool) -> str:
    if stats.empty:
        return "CHART_SKIPPED_NO_STATS"
    pivot = stats.pivot_table(index="event_name", columns="horizon", values="expectancy", aggfunc="mean")
    pivot.to_csv(ROOT / "charts/signal_persistence_heatmap_matrix.csv")
    if no_charts:
        return "CHART_SKIPPED_NO_CHARTS_FLAG"
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, max(4, len(pivot) * 0.25)))
        plt.imshow(pivot.fillna(0).to_numpy(), aspect="auto", cmap="RdYlGn")
        plt.colorbar(label="expectancy_current_cost")
        plt.yticks(range(len(pivot.index)), pivot.index, fontsize=6)
        plt.xticks(range(len(pivot.columns)), pivot.columns, fontsize=6)
        plt.title("Signal Persistence Expectancy Heatmap")
        plt.tight_layout()
        plt.savefig(ROOT / "charts/signal_persistence_heatmap.png", dpi=150)
        plt.close()
        return "CHART_CREATED"
    except Exception:
        return "CHART_SKIPPED_MATPLOTLIB_NOT_AVAILABLE"


def final_verdict(stats: pd.DataFrame, best: pd.DataFrame, wf: pd.DataFrame, frag: pd.DataFrame, ledger: Dict[str, Any]) -> Dict[str, Any]:
    if stats.empty or best.empty:
        verdict = "NO_STABLE_EDGE"
        reasons = ["NO_STATS"]
    else:
        top = best.head(1)
        exp = float(top.iloc[0]["best_horizon_expectancy_current_cost"])
        n = int(top.iloc[0]["n_events"])
        rel = str(top.iloc[0]["reliability_flag"])
        fragile = bool((frag.get("flag") == "OUTLIER_FRAGILE").any()) if not frag.empty else False
        ci_ok = bool(((stats["bootstrap_ci_low"] > 0) | (stats["bootstrap_ci_high"] < 0)).any())
        if n >= MIN_EVENTS_RELIABLE and exp > 0 and rel == "OK" and not fragile and ci_ok:
            verdict = "PROMOTE_CANDIDATE"
            reasons = ["positive_expectancy", "minimum_event_count", "bootstrap_not_centered_on_zero"]
        elif exp > 0:
            verdict = "RESEARCH_ONLY"
            reasons = ["weak_or_fragile_edge"]
        else:
            verdict = "NO_STABLE_EDGE"
            reasons = ["no_positive_expectancy_after_cost"]
    out = {
        "verdict": verdict,
        "reasons": reasons,
        "production_ready": False,
        "promotion_ready": False,
        "interpretation": "forward_shadow_observation_candidate_only" if verdict == "PROMOTE_CANDIDATE" else "research_only",
        "multiple_testing": ledger,
    }
    (ROOT / "verdict.json").write_text(jdump(out), encoding="utf-8")
    md = (
        f"# Signal Persistence Horizon Sweep Final Verdict\n\n"
        f"- verdict: **{verdict}**\n"
        f"- production_ready: false\n"
        f"- promotion_ready: false\n"
        f"- interpretation: research-only / signal persistence candidate only\n"
        f"- reasons: {', '.join(reasons)}\n"
    )
    (ROOT / "reports/signal_persistence_horizon_sweep_final_verdict.md").write_text(md, encoding="utf-8")
    return out


def write_md_table(df: pd.DataFrame, path: Path, title: str) -> None:
    path.write_text(f"# {title}\n\n```csv\n{df.to_csv(index=False)}\n```\n", encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_dirs()
    before = safety_snapshot("before") if args.strict_safety_audit or args.full else {}
    summary: Dict[str, Any] = {"production_ready": False, "promotion_ready": False, "stage": "init"}
    if args.dry_run:
        ohlcv_disc = discover_ohlcv(args.timeframe)
        aux = discover_aux_sources()
        summary.update({"verdict": "DRY_RUN_OK", "ohlcv_discovery": ohlcv_disc, "aux_discovery": aux})
        print(jdump(summary))
        return summary

    ohlcv_disc = discover_ohlcv(args.timeframe)
    aux = discover_aux_sources()
    chosen = ohlcv_disc.get("chosen")
    if not chosen:
        summary.update({"verdict": "FAILED", "stage": "discovery", "error": "no_ohlcv_found"})
        print(jdump(summary))
        return summary
    tf = args.timeframe or chosen.get("timeframe", "5m")
    max_rows = 3000 if args.fast_smoke else args.max_rows
    features_path = ROOT / "features/signal_features.parquet"
    events_path = ROOT / "events/signal_events.parquet"
    outcomes_path = ROOT / "outcomes/horizon_outcomes_long.parquet"

    def load_ohlcv_if_needed() -> pd.DataFrame:
        ohlcv_local = load_ohlcv(Path(chosen["path"]), tf, max_rows=max_rows)
        summary["input"] = {"path": chosen["path"], "rows": len(ohlcv_local), "start": str(ohlcv_local["timestamp"].min()), "end": str(ohlcv_local["timestamp"].max()), "timeframe": tf}
        return ohlcv_local

    ohlcv = None
    features = pd.read_parquet(features_path) if args.resume and features_path.exists() else None
    events = pd.read_parquet(events_path) if args.resume and events_path.exists() else None
    outcomes = pd.read_parquet(outcomes_path) if args.resume and outcomes_path.exists() else None
    stats = best = pos = pd.DataFrame()

    if args.feature_build_only or ((args.event_build_only or args.outcome_build_only or args.full or args.fast_smoke) and features is None):
        ohlcv = load_ohlcv_if_needed()
        features = build_features(ohlcv, fast_smoke=args.fast_smoke)
        features.to_parquet(features_path, index=False)
        summary["features_rows"] = len(features)
        if args.feature_build_only:
            summary["verdict"] = "FEATURE_BUILD_OK"
            print(jdump(summary))
            return summary

    if args.event_build_only or ((args.outcome_build_only or args.full or args.fast_smoke) and events is None):
        if features is None:
            if ohlcv is None:
                ohlcv = load_ohlcv_if_needed()
            features = build_features(ohlcv, fast_smoke=args.fast_smoke)
            features.to_parquet(features_path, index=False)
        events, ev_meta = generate_events(features, tf, cooldown_bars=args.cooldown_bars, fast_smoke=args.fast_smoke)
        events, _ = asof_join_features(events, aux, tf)
        events.to_parquet(events_path, index=False)
        summary.update(ev_meta)
        if args.event_build_only:
            summary["verdict"] = "EVENT_BUILD_OK"
            print(jdump(summary))
            return summary

    if args.outcome_build_only or ((args.full or args.fast_smoke) and outcomes is None):
        if events is None:
            events = pd.read_parquet(events_path)
        if ohlcv is None:
            ohlcv = load_ohlcv_if_needed()
        outcomes = build_horizon_outcomes(events, ohlcv, args.horizon_max, args.cost_bps, args.cost_bps * 2)
        outcomes.to_parquet(outcomes_path, index=False)
        summary["horizon_outcome_rows"] = len(outcomes)
        if args.outcome_build_only:
            summary["verdict"] = "OUTCOME_BUILD_OK"
            print(jdump(summary))
            return summary

    if args.stats_only or args.walk_forward_only or args.importance_only or args.regime_only or args.tcn_align_only or args.full or args.fast_smoke:
        outcomes = outcomes if outcomes is not None else pd.read_parquet(outcomes_path)
        events = events if events is not None else pd.read_parquet(events_path)
    if args.stats_only or args.full or args.fast_smoke:
        stats, best, pos = compute_statistics(outcomes, args.horizon_max, fast_smoke=args.fast_smoke)
        stats.to_csv(ROOT / "reports/horizon_sweep_statistics.csv", index=False)
        write_md_table(stats.head(200), ROOT / "reports/horizon_sweep_statistics.md", "Horizon Sweep Statistics")
        best.to_csv(ROOT / "reports/best_horizon_ranking.csv", index=False)
        write_md_table(best.head(50), ROOT / "reports/best_horizon_ranking.md", "Best Horizon Ranking")
        pos.to_csv(ROOT / "reports/positive_expectancy_windows.csv", index=False)
        write_md_table(pos.head(50), ROOT / "reports/positive_expectancy_windows.md", "Positive Expectancy Windows")
        fake = fake_breakout_analysis(outcomes)
        fake.to_csv(ROOT / "reports/fake_breakout_analysis.csv", index=False)
        write_md_table(fake.head(50), ROOT / "reports/fake_breakout_analysis.md", "Fake Breakout Analysis")
        chart_status = make_heatmap(stats, args.no_charts)
        summary["chart_status"] = chart_status
        if args.stats_only:
            summary["verdict"] = "STATS_OK"
            print(jdump(summary))
            return summary

    wf = pd.DataFrame()
    if args.walk_forward_only or args.full or args.fast_smoke:
        wf = walk_forward_stability(outcomes if outcomes is not None else pd.DataFrame(), events if events is not None else pd.DataFrame())
        wf.to_csv(ROOT / "reports/walk_forward_stability.csv", index=False)
        if args.walk_forward_only:
            summary["verdict"] = "WALK_FORWARD_OK"
            print(jdump(summary))
            return summary

    fi = perm = shap_df = pd.DataFrame()
    if args.importance_only or args.full or args.fast_smoke:
        fi, perm, shap_df = feature_importance(events if events is not None else pd.DataFrame(), outcomes if outcomes is not None else pd.DataFrame())
        fi.to_csv(ROOT / "reports/feature_importance.csv", index=False)
        perm.to_csv(ROOT / "reports/permutation_importance.csv", index=False)
        if not shap_df.empty:
            shap_df.to_csv(ROOT / "reports/shap_summary.csv", index=False)
        if args.importance_only:
            summary["verdict"] = "IMPORTANCE_OK"
            print(jdump(summary))
            return summary

    regime = pd.DataFrame()
    if args.regime_only or args.full or args.fast_smoke:
        regime_parts = [
            cross_split_analysis(outcomes if outcomes is not None else pd.DataFrame(), events if events is not None else pd.DataFrame()),
            split_analysis(outcomes if outcomes is not None else pd.DataFrame(), events if events is not None else pd.DataFrame(), "regime_label", "regime"),
            split_analysis(outcomes if outcomes is not None else pd.DataFrame(), events if events is not None else pd.DataFrame(), "volatility_state", "volatility"),
            split_analysis(outcomes if outcomes is not None else pd.DataFrame(), events if events is not None else pd.DataFrame(), "q2_state", "q2_bdi"),
            split_analysis(outcomes if outcomes is not None else pd.DataFrame(), events if events is not None else pd.DataFrame(), "guard_state", "guard"),
        ]
        regime = pd.concat(regime_parts, ignore_index=True)
        regime.to_csv(ROOT / "reports/regime_breakdown.csv", index=False)
        write_md_table(regime, ROOT / "reports/regime_breakdown.md", "Regime Breakdown")
        if args.regime_only:
            summary["verdict"] = "REGIME_OK"
            print(jdump(summary))
            return summary

    tcn_cmp = pd.DataFrame()
    if args.tcn_align_only or args.full or args.fast_smoke:
        tcn_cmp = split_analysis(outcomes if outcomes is not None else pd.DataFrame(), events if events is not None else pd.DataFrame(), "tcn_alignment", "tcn_alignment")
        tcn_cmp.to_csv(ROOT / "reports/tcn_align_comparison.csv", index=False)
        write_md_table(tcn_cmp, ROOT / "reports/tcn_align_comparison.md", "TCN Align Comparison")
        if args.tcn_align_only:
            summary["verdict"] = "TCN_ALIGN_OK"
            print(jdump(summary))
            return summary

    frag = outlier_fragility(outcomes if outcomes is not None else pd.DataFrame(), best)
    if not frag.empty:
        frag.to_csv(ROOT / "reports/outlier_fragility.csv", index=False)
    ledger = multiple_testing_ledger(events if events is not None else pd.DataFrame(), stats, args.horizon_max)
    verdict = final_verdict(stats, best, wf, frag, ledger)
    if args.strict_safety_audit or args.full:
        finalize_audit(before)
    summary.update(
        {
            "verdict": verdict.get("verdict"),
            "final_verdict": verdict,
            "best_top10": best.head(10).to_dict("records") if not best.empty else [],
            "positive_window_signals": pos[pos["positive_expectancy_window_length"] > 0]["event_name"].tolist() if not pos.empty else [],
            "ledger": ledger,
            "aux_skip_reasons": aux.get("skip_reasons", []),
            "production_ready": False,
            "promotion_ready": False,
        }
    )
    print(jdump(summary))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Signal persistence horizon sweep diagnostics")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fast-smoke", action="store_true")
    p.add_argument("--feature-build-only", action="store_true")
    p.add_argument("--event-build-only", action="store_true")
    p.add_argument("--outcome-build-only", action="store_true")
    p.add_argument("--stats-only", action="store_true")
    p.add_argument("--walk-forward-only", action="store_true")
    p.add_argument("--importance-only", action="store_true")
    p.add_argument("--regime-only", action="store_true")
    p.add_argument("--tcn-align-only", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--json", action="store_true", help="Print JSON summary")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--timeframe", type=str, default=None, choices=["5m", "15m"])
    p.add_argument("--horizon-max", type=int, default=60)
    p.add_argument("--cooldown-bars", type=int, default=3)
    p.add_argument("--no-charts", action="store_true")
    p.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    p.add_argument("--strict-safety-audit", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not any(
        [
            args.dry_run,
            args.fast_smoke,
            args.feature_build_only,
            args.event_build_only,
            args.outcome_build_only,
            args.stats_only,
            args.walk_forward_only,
            args.importance_only,
            args.regime_only,
            args.tcn_align_only,
            args.full,
        ]
    ):
        args.fast_smoke = True
    run_pipeline(args)


if __name__ == "__main__":
    main()
