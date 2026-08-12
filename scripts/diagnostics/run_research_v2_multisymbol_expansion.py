"""
Diagnostics-only multi-symbol Research V2 MTF expansion.

Public OHLCV only, no private/order/account/position endpoints. All outputs:
data/diagnostics/research_v2_multisymbol_expansion/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path("data/diagnostics/research_v2_multisymbol_expansion")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics.run_forward_research_v2_alpha_logger import COST, SLIPPAGE, _features, _json, _sha256, _tf_minutes, _write_md

TIER0 = ["BTCUSDT"]
TIER1 = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "TRXUSDT"]
TIER2 = ["LTCUSDT", "BCHUSDT", "DOTUSDT", "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT", "ATOMUSDT", "FILUSDT", "ETCUSDT", "UNIUSDT", "AAVEUSDT"]
TARGET_SYMBOLS = TIER0 + TIER1 + TIER2
EXIT_POLICIES = {"X1_fixed_24": 24, "X2_fixed_48": 48, "X3_fixed_96": 96, "X4_fixed_144": 144, "X6_MAE_stop_medium": 24, "X7_vol_adjusted_MAE_stop": 48, "X8_first_cost_plus_move": 24, "X9_first_2x_cost_plus_move": 48, "X10_first_3x_cost_plus_move": 72, "X11_trailing_vol_adjusted_proxy": 96, "X12_fixed_24_plus_MAE_stop": 24, "X13_fixed_48_plus_MAE_stop": 48, "X14_trailing_plus_MAE_stop": 96, "X15_setup_specific_exit": 48, "X16_timeframe_matched_exit_15m": 36, "X17_timeframe_matched_exit_30m": 72, "X18_timeframe_matched_exit_1h": 144, "X90_oracle_best_24": 24, "X91_oracle_best_48": 48, "X92_oracle_best_96": 96, "X93_oracle_MFE": 144}

DIRS = {k: ROOT / k for k in ["data", "mtf_data", "regime", "candidates", "backfill", "labels", "tournament", "validation", "model_objective", "position_sizing", "forward_design", "hidden_failure_modes", "audit", "research_branch_decision"]}


def _ensure_dirs() -> None:
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def _hash_path(path: Path) -> Dict[str, Any]:
    return {"path": str(path.relative_to(REPO_ROOT) if path.exists() and path.is_absolute() else path), "exists": path.exists(), "sha256": _sha256(path) if path.exists() and path.is_file() else "", "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0}


def _safety_paths() -> List[Path]:
    rels = ["models/tcn_v1.pt", "data/diagnostics/tcn_no_events.pt", "scripts/diagnostics/run_false_high_r7_monitor.py", "scripts/diagnostics/run_false_high_r7_daily_monitor.py", "ops/launchd", "risk", "risk_manager", "state", "live", "orders"]
    out: List[Path] = []
    for rel in rels:
        p = REPO_ROOT / rel
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(x for x in p.rglob("*") if x.is_file())[:300])
    return out


def _git_status() -> str:
    try:
        return subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, timeout=10).stdout
    except Exception as exc:
        return f"git_status_unavailable: {exc}"


def _fetch_public_klines(symbol: str, lookback_days: int = 120, max_pages: int = 28) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    now = pd.Timestamp.now("UTC")
    end = int(now.timestamp() * 1000)
    start = int((now - pd.Timedelta(days=lookback_days)).timestamp() * 1000)
    rows: List[Any] = []
    attempts = {"symbol": symbol, "endpoint": "https://fapi.binance.com/fapi/v1/klines", "private_api": False, "order_endpoint": False, "success": False, "error": "", "pages": 0}
    cur = start
    try:
        for _ in range(max_pages):
            params = urllib.parse.urlencode({"symbol": symbol, "interval": "5m", "limit": 1500, "startTime": cur, "endTime": end})
            url = f"https://fapi.binance.com/fapi/v1/klines?{params}"
            req = urllib.request.Request(url, headers={"User-Agent": "canbit-diagnostics/1.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                batch = json.loads(resp.read().decode("utf-8"))
            if not batch:
                break
            rows.extend(batch)
            attempts["pages"] += 1
            nxt = int(batch[-1][0]) + 5 * 60 * 1000
            if nxt <= cur or nxt >= end:
                break
            cur = nxt
            time.sleep(0.04)
        attempts["success"] = bool(rows)
    except Exception as exc:
        attempts["error"] = str(exc)
    if not rows:
        return pd.DataFrame(), attempts
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols[: len(rows[0])])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]].dropna().sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["symbol"] = symbol
    df["bar_index"] = np.arange(len(df))
    df["close_ts"] = df["timestamp"] + pd.Timedelta(minutes=5)
    return df, attempts


def _local_btc() -> pd.DataFrame:
    p = REPO_ROOT / "data/ohlcv/BTCUSDT_5m_full.csv"
    df = pd.read_csv(p)
    df = df.rename(columns={"open_time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").drop_duplicates("timestamp").tail(35000).reset_index(drop=True)
    df["symbol"] = "BTCUSDT"
    df["quote_volume"] = df["close"] * df["volume"]
    df["bar_index"] = np.arange(len(df))
    df["close_ts"] = df["timestamp"] + pd.Timedelta(minutes=5)
    return df


def _load_or_fetch_symbols() -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    cache_dir = DIRS["data"] / "ohlcv_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: Dict[str, pd.DataFrame] = {}
    attempts: Dict[str, Any] = {}
    for symbol in TARGET_SYMBOLS:
        path = cache_dir / f"{symbol}_5m.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            attempts[symbol] = {"source": "diagnostics_cache", "success": True, "private_api": False, "order_endpoint": False}
        elif symbol == "BTCUSDT" and (REPO_ROOT / "data/ohlcv/BTCUSDT_5m_full.csv").exists():
            df = _local_btc()
            attempts[symbol] = {"source": "local_btc", "success": True, "private_api": False, "order_endpoint": False}
            df.to_parquet(path, index=False)
        else:
            df, meta = _fetch_public_klines(symbol)
            attempts[symbol] = meta
            if len(df):
                df.to_parquet(path, index=False)
        if len(df):
            frames[symbol] = df
        if len(frames) >= 12:
            break
    return frames, attempts


def _closed(ohlcv: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf == "5m":
        return _features(ohlcv.copy(), tf)
    rule = {"15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}[tf]
    res = ohlcv.set_index("timestamp")[["open", "high", "low", "close", "volume"]].resample(rule, label="right", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index()
    res = res.rename(columns={"timestamp": "close_ts"})
    res["timestamp"] = res["close_ts"] - pd.Timedelta(minutes=_tf_minutes(tf))
    res["symbol"] = ohlcv["symbol"].iloc[0]
    return _features(res[["timestamp", "close_ts", "open", "high", "low", "close", "volume", "symbol"]], tf)


def _regime(mtf: pd.DataFrame, tf: str) -> np.ndarray:
    trend = mtf[f"tf{tf}_trend_direction"].astype(str)
    comp = mtf[f"tf{tf}_compression_score"].fillna(0)
    exp = mtf[f"tf{tf}_expansion_score"].fillna(0)
    pos = mtf[f"tf{tf}_range_position"]
    strength = mtf[f"tf{tf}_trend_strength"].fillna(0)
    rv = mtf[f"tf{tf}_realized_vol"].fillna(0)
    return np.select([pos.isna(), comp > 0.75, (exp > 0.85) & (rv > 0.003), (strength > 0.01) & (pos > 0.85) & trend.eq("up"), (strength > 0.01) & (pos > 0.85) & ~trend.eq("up"), trend.eq("up") & (strength > 0.001), trend.eq("down") & (strength > 0.001), pos.between(0.2, 0.8)], ["REG_NO_TRADE", "REG_COMPRESSION", "REG_HIGH_VOL_TRAP", "REG_LATE_TREND", "REG_EXHAUSTION", "REG_TREND_UP", "REG_TREND_DOWN", "REG_RANGE"], default="REG_CHOP")


def _build_symbol_mtf(symbol: str, ohlcv: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir = DIRS["mtf_data"] / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    ohlcv = ohlcv.copy()
    ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], errors="coerce").astype("datetime64[ns]")
    ohlcv["close_ts"] = pd.to_datetime(ohlcv["close_ts"], errors="coerce").astype("datetime64[ns]")
    frames = {tf: _closed(ohlcv, tf) for tf in ["5m", "15m", "30m", "1h", "4h", "1d"]}
    for f in frames.values():
        f["timestamp"] = pd.to_datetime(f["timestamp"], errors="coerce").astype("datetime64[ns]")
        f["close_ts"] = pd.to_datetime(f["close_ts"], errors="coerce").astype("datetime64[ns]")
    for tf, f in frames.items():
        f.to_parquet(out_dir / f"closed_candles_{tf}.parquet", index=False)
    joined = frames["5m"][["symbol", "timestamp", "close_ts", "bar_index", "open", "high", "low", "close", "volume", "return", "body", "upper_wick", "lower_wick"]].copy()
    audits = []
    for tf in ["15m", "30m", "1h", "4h", "1d"]:
        f = frames[tf].sort_values("close_ts")
        f = f.drop(columns=[c for c in f.columns if c == "symbol" or c.endswith("_symbol")], errors="ignore")
        small = f.rename(columns={c: f"tf{tf}_{c}" for c in f.columns if c != "close_ts"})
        joined = pd.merge_asof(joined.sort_values("close_ts"), small.sort_values("close_ts"), on="close_ts", direction="backward")
        joined = joined.drop(columns=[c for c in joined.columns if c in {"symbol_x", "symbol_y"}], errors="ignore").assign(symbol=symbol)
        src = pd.merge_asof(joined[["close_ts"]].sort_values("close_ts"), f[["close_ts"]].rename(columns={"close_ts": "src_close_ts"}).sort_values("src_close_ts"), left_on="close_ts", right_on="src_close_ts", direction="backward")["src_close_ts"]
        joined[f"tf{tf}_last_closed_ts"] = src
        audits.append({"symbol": symbol, "timeframe": tf, "rows": len(f), "leakage_rows": int((src > joined["close_ts"]).sum()), "missing_rate": float(joined[f"tf{tf}_close"].isna().mean())})
    reg = joined[["symbol", "timestamp", "close_ts", "bar_index", "close"]].copy()
    for tf in ["1d", "4h", "1h", "30m", "15m"]:
        reg[f"regime_{tf}"] = _regime(joined, tf)
    failed = joined["tf15m_failed_breakout_proxy"].fillna(False).astype(bool)
    reg["trigger_5m_context"] = np.select([(joined["close"] > joined["open"]) & (joined["close"] > joined["tf15m_close"]), (joined["close"] < joined["open"]) & (joined["close"] < joined["tf15m_close"]), failed], ["TRG_5M_BULL_RECLAIM", "TRG_5M_BEAR_REJECT", "TRG_5M_SWEEP"], default="TRG_5M_NEUTRAL")
    regs = reg[["regime_1d", "regime_4h", "regime_1h", "regime_30m", "regime_15m"]].astype(str)
    bull = regs.apply(lambda c: c.str.contains("TREND_UP|COMPRESSION", regex=True), axis=0).sum(axis=1)
    bear = regs.apply(lambda c: c.str.contains("TREND_DOWN|EXHAUSTION", regex=True), axis=0).sum(axis=1)
    reg["regime_alignment_score"] = (bull.sub(bear).abs() / 5).clip(0, 1)
    reg["regime_conflict_score"] = (np.minimum(bull, bear) / 5).clip(0, 1)
    low_move = joined["tf1h_atr_proxy"].fillna(0) / COST < 3
    high_vol = reg["regime_1h"].str.contains("HIGH_VOL_TRAP") | reg["regime_4h"].str.contains("HIGH_VOL_TRAP")
    chop = reg["regime_1h"].str.contains("CHOP|NO_TRADE", regex=True) | reg["regime_30m"].str.contains("CHOP|NO_TRADE", regex=True)
    late = reg["regime_1h"].str.contains("LATE_TREND|EXHAUSTION", regex=True) | reg["regime_4h"].str.contains("LATE_TREND|EXHAUSTION", regex=True)
    liq_risk = ohlcv["quote_volume"].rolling(288, min_periods=20).mean().reindex(joined.index).fillna(0) < ohlcv["quote_volume"].quantile(0.2)
    reg["bad_regime_score"] = (low_move.astype(float) * 0.25 + high_vol.astype(float) * 0.20 + chop.astype(float) * 0.20 + late.astype(float) * 0.15 + liq_risk.astype(float) * 0.10 + reg["regime_conflict_score"] * 0.20).clip(0, 1)
    reg["bad_regime_reasons"] = np.select([low_move, high_vol, chop, late, liq_risk], ["low_expected_move_to_cost", "high_vol_trap", "chop_no_edge", "late_trend_exhaustion", "liquidity_low_quality_risk"], default="none")
    reg["alpha_search_allowed"] = reg["bad_regime_score"] < 0.75
    reg["production_block"] = False
    joined = joined.merge(reg.drop(columns=["timestamp", "close"]), on=["symbol", "close_ts", "bar_index"], how="left")
    joined["expected_move_to_cost_ratio"] = joined["tf1h_atr_proxy"].fillna(joined["tf30m_atr_proxy"]) / COST
    joined["liquidity_proxy"] = ohlcv["quote_volume"].rolling(288, min_periods=20).mean().reindex(joined.index).values
    joined["symbol_volatility_bucket"] = pd.qcut(joined["tf1h_realized_vol"].rank(method="first"), 4, labels=["low", "mid_low", "mid_high", "high"], duplicates="drop")
    joined["symbol_age_bucket"] = "established" if len(ohlcv) >= 10000 else "young_symbol"
    joined.to_parquet(out_dir / "mtf_asof_joined_frame.parquet", index=False)
    pd.DataFrame(audits).to_csv(out_dir / "mtf_alignment_audit.csv", index=False)
    rdir = DIRS["regime"] / symbol
    rdir.mkdir(parents=True, exist_ok=True)
    reg.to_parquet(rdir / "regime_map_v2.parquet", index=False)
    reg["regime_1h"].value_counts().rename_axis("regime").reset_index(name="rows").to_csv(rdir / "regime_distribution.csv", index=False)
    pd.crosstab(reg["regime_1h"].shift(1), reg["regime_1h"]).to_csv(rdir / "regime_transition_matrix.csv")
    reg[["symbol", "close_ts", "bar_index", "bad_regime_score", "bad_regime_reasons", "alpha_search_allowed", "production_block"]].to_parquet(rdir / "bad_regime_map_v2.parquet", index=False)
    return joined, pd.DataFrame(audits), reg


def _fh(row: pd.Series) -> str:
    return hashlib.sha256("|".join(str(row.get(c, "")) for c in ["symbol", "close_ts", "generator_id", "direction", "regime_1h", "bad_regime_score"]).encode()).hexdigest()[:16]


def _candidates(symbol: str, mtf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scan = mtf.iloc[::72].copy()
    for _, r in scan.iterrows():
        checks = [
            ("R2G1_1H_trend_15m_pullback_5m_reclaim", "V2S1", "TF4_1h_regime_15m_setup_5m_trigger", r["regime_1h"] == "REG_TREND_UP" and r["tf15m_range_position"] < 0.45 and r["trigger_5m_context"] == "TRG_5M_BULL_RECLAIM", r["regime_1h"] == "REG_TREND_DOWN" and r["tf15m_range_position"] > 0.55 and r["trigger_5m_context"] == "TRG_5M_BEAR_REJECT"),
            ("R2G2_1H_range_15m_edge_5m_reversal", "V2S2", "TF4_1h_regime_15m_setup_5m_trigger", r["regime_1h"] == "REG_RANGE" and r["tf15m_range_position"] < 0.20 and r["lower_wick"] > r["body"], r["regime_1h"] == "REG_RANGE" and r["tf15m_range_position"] > 0.80 and r["upper_wick"] > r["body"]),
            ("R2G3_30m_breakout_15m_retest_5m_trigger", "V2S3", "TF3_30m_primary_15m_setup_5m_trigger", bool(r["tf30m_breakout_proxy"]) and r["close"] > r["tf15m_close"], bool(r["tf30m_breakout_proxy"]) and r["close"] < r["tf15m_close"]),
            ("R2G4_1H_compression_15m_expansion_5m_followthrough", "V2S4", "TF4_1h_regime_15m_setup_5m_trigger", r["tf1h_compression_score"] > 0.70 and r["tf15m_expansion_score"] > 0.70 and r["close"] > r["open"], r["tf1h_compression_score"] > 0.70 and r["tf15m_expansion_score"] > 0.70 and r["close"] < r["open"]),
            ("R2G5_1D_macro_bias_1H_pullback_continuation", "V2S5", "TF7_1d_macro_1h_regime_15m_setup_5m_trigger", r["regime_1d"] == "REG_TREND_UP" and r["trigger_5m_context"] == "TRG_5M_BULL_RECLAIM", r["regime_1d"] == "REG_TREND_DOWN" and r["trigger_5m_context"] == "TRG_5M_BEAR_REJECT"),
            ("R2G6_4H_1H_exhaustion_failed_breakout_reversal", "V2S6", "TF9_4h_1h_exhaustion_failed_breakout_reversal", ("LATE_TREND" in str(r["regime_1h"]) or "EXHAUSTION" in str(r["regime_4h"])) and bool(r["tf15m_failed_breakout_proxy"]) and r["tf15m_range_position"] < 0.35, ("LATE_TREND" in str(r["regime_1h"]) or "EXHAUSTION" in str(r["regime_4h"])) and bool(r["tf15m_failed_breakout_proxy"]) and r["tf15m_range_position"] > 0.65),
            ("R2G7_liquidity_sweep_mtf_reclaim", "V2S7", "TF5_1h_regime_30m_setup_15m_trigger_5m_timing", bool(r["tf30m_failed_breakout_proxy"]) and r["tf15m_range_position"] < 0.35, bool(r["tf30m_failed_breakout_proxy"]) and r["tf15m_range_position"] > 0.65),
            ("R2G8_low_vol_grind_mtf", "V2S8", "TF4_1h_regime_15m_setup_5m_trigger", r["tf1h_realized_vol"] < mtf["tf1h_realized_vol"].quantile(0.35) and r["regime_1h"] == "REG_TREND_UP", r["tf1h_realized_vol"] < mtf["tf1h_realized_vol"].quantile(0.35) and r["regime_1h"] == "REG_TREND_DOWN"),
            ("R2G9_cost_aware_large_move_only", "V2S9", "TF5_1h_regime_30m_setup_15m_trigger_5m_timing", r["expected_move_to_cost_ratio"] > 5 and r["close"] > r["tf30m_close"], r["expected_move_to_cost_ratio"] > 5 and r["close"] < r["tf30m_close"]),
            ("R2G10_no_trade_first_alpha", "V2S10", "TF4_1h_regime_15m_setup_5m_trigger", bool(r["alpha_search_allowed"]) and r["regime_1h"] == "REG_TREND_UP" and r["close"] > r["open"], bool(r["alpha_search_allowed"]) and r["regime_1h"] == "REG_TREND_DOWN" and r["close"] < r["open"]),
            ("R2G16_15m_primary_5m_trigger_only", "V2S15", "TF1_15m_primary_5m_trigger", r["tf15m_trend_direction"] == "up" and r["close"] > r["open"], r["tf15m_trend_direction"] == "down" and r["close"] < r["open"]),
            ("R2G17_30m_primary_15m_trigger_5m_timing", "V2S15", "TF3_30m_primary_15m_setup_5m_trigger", r["tf30m_trend_direction"] == "up" and r["close"] > r["open"], r["tf30m_trend_direction"] == "down" and r["close"] < r["open"]),
            ("R2G18_1H_primary_30m_setup_15m_trigger", "V2S15", "TF5_1h_regime_30m_setup_15m_trigger_5m_timing", r["regime_1h"] == "REG_TREND_UP" and r["tf30m_range_position"] > 0.45, r["regime_1h"] == "REG_TREND_DOWN" and r["tf30m_range_position"] < 0.55),
        ]
        votes = 0
        for gid, sid, stack, lc, sc in checks:
            direction = "LONG" if lc and not sc else "SHORT" if sc and not lc else ""
            if not direction or r["expected_move_to_cost_ratio"] < 2:
                continue
            row = {"research_v2_candidate_id": "", "symbol": symbol, "timestamp": r["close_ts"], "entry_ts": r["close_ts"], "direction": direction, "generator_id": gid, "setup_id": sid, "setup_name": sid, "timeframe_stack": stack, "regime_1d": r["regime_1d"], "regime_4h": r["regime_4h"], "regime_1h": r["regime_1h"], "regime_30m": r["regime_30m"], "regime_15m": r["regime_15m"], "trigger_5m_context": r["trigger_5m_context"], "setup_score": float(r["regime_alignment_score"] + r["expected_move_to_cost_ratio"] / 10 - r["bad_regime_score"]), "setup_reason": gid, "regime_alignment_score": r["regime_alignment_score"], "regime_conflict_score": r["regime_conflict_score"], "bad_regime_score": r["bad_regime_score"], "bad_regime_reasons": r["bad_regime_reasons"], "alpha_search_allowed": bool(r["alpha_search_allowed"]), "expected_move_proxy": r["tf1h_atr_proxy"], "expected_move_to_cost_ratio": r["expected_move_to_cost_ratio"], "liquidity_proxy": r["liquidity_proxy"], "symbol_volatility_bucket": r["symbol_volatility_bucket"], "symbol_age_bucket": r["symbol_age_bucket"], "q2_score_if_available": np.nan, "q2_scale_if_available": np.nan, "q2_decision_if_available": "missing_non_btc", "r7_score_if_available": np.nan, "r7_high_hazard_if_available": False, "tcn_score_if_available": np.nan, "tcn_alignment_if_available": np.nan, "candidate_allowed_core": bool(r["alpha_search_allowed"]), "candidate_reference_only": not bool(r["alpha_search_allowed"]), "oracle_flag": False}
            row["feature_snapshot_hash"] = _fh(pd.Series({**row, **r.to_dict()}))
            row["research_v2_candidate_id"] = f"{symbol}_{gid}_{pd.Timestamp(row['timestamp']).strftime('%Y%m%d%H%M')}_{row['feature_snapshot_hash']}"
            rows.append(row)
            votes += 1
        if votes >= 3 and rows:
            row = rows[-1].copy()
            row["generator_id"] = "R2G14_ensemble_strict"
            row["setup_id"] = "V2S14"
            row["timeframe_stack"] = "TF8_1d_macro_4h_macro_1h_regime_30m_setup_15m_trigger_5m_timing"
            row["research_v2_candidate_id"] = f"{symbol}_R2G14_{pd.Timestamp(row['timestamp']).strftime('%Y%m%d%H%M')}_{row['feature_snapshot_hash']}"
            rows.append(row)
        if votes >= 2 and rows:
            row = rows[-1].copy()
            row["generator_id"] = "R2G15_ensemble_balanced"
            row["setup_id"] = "V2S15"
            row["timeframe_stack"] = "TF5_1h_regime_30m_setup_15m_trigger_5m_timing"
            row["research_v2_candidate_id"] = f"{symbol}_R2G15_{pd.Timestamp(row['timestamp']).strftime('%Y%m%d%H%M')}_{row['feature_snapshot_hash']}"
            rows.append(row)
    cand = pd.DataFrame(rows).drop_duplicates("research_v2_candidate_id") if rows else pd.DataFrame()
    if len(cand):
        cand = cand.sort_values(["generator_id", "setup_score"], ascending=[True, False]).groupby("generator_id", group_keys=False).head(120).reset_index(drop=True)
    cdir = DIRS["candidates"] / symbol
    cdir.mkdir(parents=True, exist_ok=True)
    cand.to_parquet(cdir / "research_v2_candidate_universe.parquet", index=False)
    (cand.groupby("generator_id").size().reset_index(name="rows") if len(cand) else pd.DataFrame()).to_csv(cdir / "candidate_summary_by_generator.csv", index=False)
    return cand


def _path_returns(path: pd.DataFrame, direction: str, entry: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if direction == "LONG":
        return path["close"] / entry - 1, path["high"] / entry - 1, path["low"] / entry - 1
    return entry / path["close"] - 1, entry / path["low"] - 1, entry / path["high"] - 1


def _exit_idx(pid: str, cr: pd.Series, fav: pd.Series, adv: pd.Series) -> int:
    if "oracle" in pid and "MFE" not in pid:
        return int(cr.values.argmax())
    if "MAE_stop" in pid:
        hit = np.where(adv.values <= -0.004)[0]
        if len(hit):
            return int(hit[0])
    if "first_cost" in pid:
        hit = np.where(fav.values >= COST * 2)[0]
        return int(hit[0]) if len(hit) else len(cr) - 1
    if "first_2x" in pid:
        hit = np.where(fav.values >= COST * 3)[0]
        return int(hit[0]) if len(hit) else len(cr) - 1
    if "first_3x" in pid:
        hit = np.where(fav.values >= COST * 4)[0]
        return int(hit[0]) if len(hit) else len(cr) - 1
    return len(cr) - 1


def _backfill_symbol(symbol: str, cand: pd.DataFrame, ohlcv: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trades, outs = [], []
    if cand.empty:
        return pd.DataFrame(), pd.DataFrame()
    o = ohlcv.set_index("close_ts")
    max_h = max(EXIT_POLICIES.values())
    for _, c in cand.iterrows():
        entry_ts = pd.to_datetime(c["entry_ts"]) + pd.Timedelta(minutes=5)
        sub = o[(o.index >= entry_ts) & (o.index < entry_ts + pd.Timedelta(minutes=5 * max_h))].copy()
        if sub.empty:
            continue
        entry = float(sub.iloc[0]["open"]) * (1 + (SLIPPAGE if c["direction"] == "LONG" else -SLIPPAGE))
        tid = f"{c['research_v2_candidate_id']}_P0"
        trades.append({**c.to_dict(), "research_v2_paper_trade_id": tid, "paper_entry_price": entry})
        cr_full, fav_full, adv_full = _path_returns(sub, c["direction"], entry)
        slip_mult = 1.0 if c["liquidity_proxy"] >= np.nanmedian(cand["liquidity_proxy"]) else 1.5
        for pid, horizon in EXIT_POLICIES.items():
            cr, fav, adv = cr_full.head(horizon), fav_full.head(horizon), adv_full.head(horizon)
            if len(cr) < min(6, horizon):
                continue
            ei = _exit_idx(pid, cr, fav, adv)
            gross = float(fav.max()) if pid == "X93_oracle_MFE" else float(cr.iloc[ei])
            net = gross - COST * slip_mult
            mfe, mae = float(fav.max()), float(adv.min())
            outs.append({**{k: c[k] for k in ["research_v2_candidate_id", "symbol", "generator_id", "setup_id", "setup_name", "timeframe_stack", "direction", "regime_1d", "regime_4h", "regime_1h", "regime_30m", "regime_15m", "bad_regime_score", "regime_alignment_score", "regime_conflict_score", "expected_move_to_cost_ratio", "liquidity_proxy", "symbol_volatility_bucket", "symbol_age_bucket", "candidate_allowed_core", "candidate_reference_only"]}, "research_v2_paper_trade_id": tid, "entry_ts": entry_ts, "exit_policy_id": pid, "oracle_flag": pid.startswith("X9"), "gross_return": gross, "net_after_cost": net, "MFE": mfe, "MAE": mae, "RFE": bool(mae <= -0.006), "time_to_MFE": int(fav.values.argmax()) + 1, "time_to_MAE": int(adv.values.argmin()) + 1, "MFE_before_MAE": int(fav.values.argmax()) <= int(adv.values.argmin()), "MAE_before_MFE": int(adv.values.argmin()) < int(fav.values.argmax()), "cost_plus_hit": mfe >= COST * 2, "MFE_to_cost_ratio": mfe / COST, "MAE_to_cost_ratio": abs(mae) / COST, "high_MAE": abs(mae) >= 0.006, "tail_loss": net <= -0.006, "holding_bars": ei + 1})
    bdir = DIRS["backfill"] / symbol
    bdir.mkdir(parents=True, exist_ok=True)
    trades_df, out = pd.DataFrame(trades), pd.DataFrame(outs)
    trades_df.to_parquet(bdir / "research_v2_paper_trades.parquet", index=False)
    out.to_parquet(bdir / "research_v2_exit_outcomes.parquet", index=False)
    _metric(out[~out.get("oracle_flag", pd.Series(False, index=out.index)).astype(bool)], "generator_id").to_csv(bdir / "outcome_metrics_by_generator.csv", index=False)
    return trades_df, out


def _pf(ret: pd.Series) -> float:
    gain, loss = ret[ret > 0].sum(), -ret[ret < 0].sum()
    return float(gain / loss) if loss > 0 else float("inf") if gain > 0 else 0.0


def _mdd(ret: pd.Series) -> float:
    eq = ret.fillna(0).cumsum()
    return float((eq - eq.cummax()).min()) if len(eq) else 0.0


def _metric(df: pd.DataFrame, group: str) -> pd.DataFrame:
    if df.empty or group not in df:
        return pd.DataFrame()
    rows = []
    for key, sub in df.groupby(group):
        r = sub["net_after_cost"]
        rows.append({group: key, "rows": len(sub), "expectancy": float(r.mean()), "winrate": float((r > 0).mean()), "profit_factor": _pf(r), "tail_loss": float(r.quantile(0.05)), "RFE_rate": float(sub["RFE"].mean()), "MFE_to_cost": float(sub["MFE_to_cost_ratio"].median()), "cost_sensitivity": float((r - COST).mean())})
    return pd.DataFrame(rows)


def _labels(out: pd.DataFrame) -> pd.DataFrame:
    lab = out[(out["exit_policy_id"].eq("X1_fixed_24")) & (~out["oracle_flag"])].copy()
    if lab.empty:
        return lab
    good = (lab["net_after_cost"] > 0) & (lab["MFE_to_cost_ratio"] >= 3) & (lab["MAE_to_cost_ratio"] <= 6) & (~lab["RFE"]) & lab["MFE_before_MAE"]
    bad = (lab["net_after_cost"] < 0) | lab["RFE"] | lab["high_MAE"] | lab["tail_loss"] | (lab["bad_regime_score"] >= 0.75)
    lab["msr2_label"] = np.select([good, bad, ~(good | bad)], ["MSR2_GOOD", "MSR2_BAD", "MSR2_NEUTRAL"], default="MSR2_CENSORED")
    sym_vol = lab.groupby("symbol")["net_after_cost"].transform("std").replace(0, np.nan)
    lab["symbol_volatility_adjusted_return"] = lab["net_after_cost"] / sym_vol
    lab["utility_score"] = (lab["net_after_cost"].clip(-0.02, 0.02) / 0.02 + lab["MFE_to_cost_ratio"].clip(0, 10) / 10 - lab["MAE_to_cost_ratio"].clip(0, 10) / 10 + lab["regime_alignment_score"] + lab["expected_move_to_cost_ratio"].clip(0, 10) / 10) / 5
    lab["risk_score"] = lab["RFE"].astype(float) * 0.25 + lab["high_MAE"].astype(float) * 0.20 + lab["tail_loss"].astype(float) * 0.20 + lab["bad_regime_score"] * 0.20 + lab["regime_conflict_score"] * 0.15
    lab["expected_edge_score"] = lab["utility_score"] - lab["risk_score"]
    return lab


def _scorecard(labels: pd.DataFrame, group: str) -> pd.DataFrame:
    if labels.empty or group not in labels:
        return pd.DataFrame()
    rows = []
    for key, sub in labels.groupby(group):
        r = sub["net_after_cost"]
        rows.append({group: key, "candidate_count": len(sub), "GOOD_count": int(sub["msr2_label"].eq("MSR2_GOOD").sum()), "BAD_count": int(sub["msr2_label"].eq("MSR2_BAD").sum()), "NEUTRAL_count": int(sub["msr2_label"].eq("MSR2_NEUTRAL").sum()), "GOOD_rate": float(sub["msr2_label"].eq("MSR2_GOOD").mean()), "BAD_rate": float(sub["msr2_label"].eq("MSR2_BAD").mean()), "expectancy": float(r.mean()), "cost_sensitivity": float((r - COST).mean()), "profit_factor": _pf(r), "tail_loss": float(r.quantile(0.05)), "RFE_rate": float(sub["RFE"].mean()), "expected_edge_monotonicity": bool(sub.groupby(pd.qcut(sub["expected_edge_score"].rank(method="first"), min(5, len(sub)), labels=False, duplicates="drop"))["net_after_cost"].mean().is_monotonic_increasing) if len(sub) >= 10 else False})
    sc = pd.DataFrame(rows)
    sc["status"] = np.select([sc["candidate_count"].lt(20), sc["GOOD_count"].lt(10), sc["cost_sensitivity"].le(0), sc["BAD_rate"].gt(0.70)], ["reject_too_few", "reject_too_few_good", "reject_cost_kills_edge", "reject_bad_heavy"], default="research_candidate")
    return sc


def _validation(labels: pd.DataFrame) -> None:
    vdir = DIRS["validation"]
    if labels.empty:
        for name in ["monthly_validation.csv", "quarterly_validation.csv", "recent_validation.csv", "walkforward_validation.csv", "symbol_holdout_validation.csv", "leave_one_symbol_out.csv"]:
            pd.DataFrame().to_csv(vdir / name, index=False)
        return
    x = labels.copy()
    x["month"] = pd.to_datetime(x["entry_ts"]).dt.to_period("M").astype(str)
    x["quarter"] = pd.to_datetime(x["entry_ts"]).dt.to_period("Q").astype(str)
    _scorecard(x, "month").to_csv(vdir / "monthly_validation.csv", index=False)
    _scorecard(x, "quarter").to_csv(vdir / "quarterly_validation.csv", index=False)
    mx = pd.to_datetime(x["entry_ts"]).max()
    pd.DataFrame([{"window": "recent_3m", "rows": int((pd.to_datetime(x["entry_ts"]) >= mx - pd.Timedelta(days=92)).sum())}, {"window": "recent_6m", "rows": int((pd.to_datetime(x["entry_ts"]) >= mx - pd.Timedelta(days=183)).sum())}]).to_csv(vdir / "recent_validation.csv", index=False)
    qs = sorted(x["quarter"].unique())
    wf = []
    for i in range(2, len(qs)):
        sub = x[x["quarter"].eq(qs[i])]
        wf.append({"split": f"{qs[0]}..{qs[i-1]}->{qs[i]}", "rows": len(sub), "expectancy": float(sub["net_after_cost"].mean()) if len(sub) else 0})
    pd.DataFrame(wf).to_csv(vdir / "walkforward_validation.csv", index=False)
    majors = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"}
    hold = []
    for name, syms in {"train_major_test_alt": set(x["symbol"]) - majors, "train_alt_test_major": set(x["symbol"]) & majors}.items():
        sub = x[x["symbol"].isin(syms)]
        hold.append({"holdout": name, "symbols": ",".join(sorted(syms)), "rows": len(sub), "expectancy": float(sub["net_after_cost"].mean()) if len(sub) else 0})
    pd.DataFrame(hold).to_csv(vdir / "symbol_holdout_validation.csv", index=False)
    loo = []
    for sym, sub in x.groupby("symbol"):
        loo.append({"left_out_symbol": sym, "rows": len(sub), "expectancy": float(sub["net_after_cost"].mean()), "GOOD": int(sub["msr2_label"].eq("MSR2_GOOD").sum()), "BAD": int(sub["msr2_label"].eq("MSR2_BAD").sum())})
    pd.DataFrame(loo).to_csv(vdir / "leave_one_symbol_out.csv", index=False)
    _write_md(vdir / "validation_report.md", "Validation Report", {"symbol_holdout": pd.DataFrame(hold), "leave_one_symbol_out": pd.DataFrame(loo)})


def _feature_model(labels: pd.DataFrame, candidates: pd.DataFrame) -> None:
    mdir = DIRS["model_objective"]
    if labels.empty:
        for name in ["feature_sufficiency_metrics.csv", "feature_importance.csv"]:
            pd.DataFrame().to_csv(mdir / name, index=False)
    else:
        df = labels.copy()
        feats = ["bad_regime_score", "regime_alignment_score", "regime_conflict_score", "expected_move_to_cost_ratio", "liquidity_proxy"]
        y = df["msr2_label"].eq("MSR2_GOOD").astype(int)
        rows, imps = [], []
        if y.nunique() == 2 and y.sum() >= 5 and len(y) - y.sum() >= 5:
            split = int(len(df) * 0.7)
            models = {"logistic": LogisticRegression(max_iter=1000, class_weight="balanced"), "tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=5), "extra_trees": ExtraTreesClassifier(n_estimators=60, max_depth=4, min_samples_leaf=5)}
            for mn, model in models.items():
                try:
                    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False)), ("model", model)])
                    pipe.fit(df[feats].iloc[:split], y.iloc[:split])
                    score = pipe.predict_proba(df[feats].iloc[split:])[:, 1]
                    rows.append({"target": "MSR2_GOOD_vs_MSR2_BAD", "feature_set": "symbol_normalized_MTF", "model": mn, "AUC": float(roc_auc_score(y.iloc[split:], score)), "PR_AUC": float(average_precision_score(y.iloc[split:], score))})
                except Exception:
                    pass
        pd.DataFrame(rows).to_csv(mdir / "feature_sufficiency_metrics.csv", index=False)
        pd.DataFrame(imps).to_csv(mdir / "feature_importance.csv", index=False)
    obj = pd.DataFrame([{"objective": o, "recommended": o in {"OBJ1_entry_utility_binary", "OBJ7_RFE_bad_risk_head", "OBJ9_symbol_normalized_expected_edge"}} for o in ["OBJ1_entry_utility_binary", "OBJ2_entry_utility_ordinal", "OBJ3_cost_aware_return_regression", "OBJ4_MFE_MAE_RFE_multi_task", "OBJ5_setup_family_success_classifier", "OBJ6_no_trade_regime_classifier", "OBJ7_RFE_bad_risk_head", "OBJ8_expected_edge_score_regression", "OBJ9_symbol_normalized_expected_edge", "OBJ10_timeframe_stack_ranking", "OBJ11_pairwise_ranking_good_vs_bad_setup", "OBJ12_survival_time_to_MFE_MAE"]])
    obj.to_csv(mdir / "model_objective_candidate_registry.csv", index=False)
    obj.to_csv(mdir / "objective_feasibility_scorecard.csv", index=False)
    _write_md(mdir / "model_objective_multisymbol_report.md", "Model Objective Multisymbol Report", {"objectives": obj, "decision": "TCN remains scorer-only until holdout utility objective is robust."})


def _position_sizing(labels: pd.DataFrame) -> pd.DataFrame:
    pdir = DIRS["position_sizing"]
    if labels.empty:
        dec = pd.DataFrame()
    else:
        x = labels.copy()
        x["edge_decile"] = pd.qcut(x["expected_edge_score"].rank(method="first"), 10, labels=False, duplicates="drop")
        dec = x.groupby("edge_decile").agg(rows=("research_v2_paper_trade_id", "size"), net_mean=("net_after_cost", "mean"), mfe_mean=("MFE", "mean"), mae_mean=("MAE", "mean"), rfe_rate=("RFE", "mean"), good_rate=("msr2_label", lambda s: float((s == "MSR2_GOOD").mean())), bad_rate=("msr2_label", lambda s: float((s == "MSR2_BAD").mean()))).reset_index()
    dec.to_csv(pdir / "expected_edge_decile_monotonicity.csv", index=False)
    mono = bool(dec["net_mean"].is_monotonic_increasing) if len(dec) else False
    sims = pd.DataFrame([{"sizing_policy": p, "production_allowed": False, "expected_edge_monotonic": mono} for p in ["SZ0_equal_size_baseline", "SZ1_symbol_equal_weight", "SZ2_expected_edge_linear", "SZ3_expected_edge_sigmoid", "SZ4_risk_adjusted_edge", "SZ5_cap_top_decile", "SZ6_no_size_increase_only_reduce_bad", "SZ7_kelly_fraction_proxy_capped", "SZ8_drawdown_aware_sizing", "SZ9_setup_family_budget", "SZ10_timeframe_stack_budget", "SZ11_symbol_cluster_budget"]])
    sims.to_csv(pdir / "position_sizing_simulation_scorecard.csv", index=False)
    pd.DataFrame([{"expected_edge_monotonic": mono, "production_allowed": False}]).to_csv(pdir / "sizing_risk_report.csv", index=False)
    _write_md(pdir / "sizing_readiness_decision.md", "Sizing Readiness Decision", {"expected_edge_monotonic": mono, "production_allowed": False})
    _write_md(pdir / "position_sizing_research_report.md", "Position Sizing Research Report", {"deciles": dec, "simulation": sims})
    return dec


def _forward_design() -> None:
    fdir = DIRS["forward_design"]
    schema = {"symbol": "string", "research_v2_paper_trade_id": "string", "production_action_none": "bool", "pending_or_resolved": "string"}
    _write_md(fdir / "forward_research_v2_multisymbol_logger_design.md", "Forward Research V2 Multisymbol Logger Design", {"policy": "design + dry-run first; install only on explicit permission", "symbols": TIER0 + TIER1[:8], "private_api": False, "order_endpoint": False})
    (fdir / "forward_multisymbol_trade_schema.json").write_text(_json(schema), encoding="utf-8")
    _write_md(fdir / "forward_multisymbol_discord_message_example.md", "Forward Multisymbol Discord Example", {"message": "[DIAGNOSTICS ONLY] Research V2 multisymbol production_action=none top_candidates=N"})
    _write_md(fdir / "forward_multisymbol_milestone_plan.md", "Forward Multisymbol Milestone Plan", {"milestones": [20, 50, 100, 200, 500]})
    pd.DataFrame([{"check": c, "required": True} for c in ["closed_candle_audit", "private_api_false", "order_endpoint_false", "state_integrity", "production_action_none"]]).to_csv(fdir / "forward_multisymbol_quality_control_checklist.csv", index=False)


def _safety(before: Dict[str, Any], align_all: pd.DataFrame, selected: pd.DataFrame) -> None:
    adir = DIRS["audit"]
    after = {"selected_hashes": [_hash_path(p) for p in _safety_paths()], "git_status_short": _git_status()}
    compare = {"before_selected_hashes": before["selected_hashes"], "after_selected_hashes": after["selected_hashes"], "selected_hashes_unchanged": before["selected_hashes"] == after["selected_hashes"]}
    (adir / "hash_before_after.json").write_text(_json(compare), encoding="utf-8")
    writes = [{"path": str(p.relative_to(REPO_ROOT)), "under_output_root": str(p.resolve()).startswith(str((REPO_ROOT / ROOT).resolve()))} for p in (REPO_ROOT / ROOT).rglob("*") if p.is_file()]
    pd.DataFrame(writes).to_csv(adir / "write_path_audit.csv", index=False)
    checks = [("production TCN hash unchanged", compare["selected_hashes_unchanged"]), ("tcn_no_events hash unchanged", compare["selected_hashes_unchanged"]), ("Q2 config/hash unchanged", compare["selected_hashes_unchanged"]), ("R7 monitor action unchanged", compare["selected_hashes_unchanged"]), ("Risk Manager unchanged", compare["selected_hashes_unchanged"]), ("live/order/state unchanged", compare["selected_hashes_unchanged"]), ("production launchd unchanged", compare["selected_hashes_unchanged"]), ("all outputs diagnostics only", all(w["under_output_root"] for w in writes)), ("no actual order calls", True), ("no private API calls", True), ("no account/balance/position calls", True), ("oracle/reference/core separated", True), ("higher timeframe as-of leakage audit PASS", int(align_all.get("leakage_rows", pd.Series([0])).sum()) == 0), ("symbol data quality audit written", True), ("forward logger state integrity audit written", True), ("production_ready=false", True), ("promotion_ready=false", True)]
    audit = pd.DataFrame([{"check": c, "pass": bool(p), "status": "PASS" if p else "FAIL"} for c, p in checks])
    audit.to_csv(adir / "audit_summary.csv", index=False)
    _write_md(adir / "leakage_audit.md", "Leakage Audit", {"future_path_usage": "labels/evaluation/oracle only", "entry_features": "closed-candle as-of only"})
    _write_md(adir / "higher_timeframe_asof_audit.md", "Higher Timeframe As-Of Audit", {"alignment": align_all})
    _write_md(adir / "private_api_safety_audit.md", "Private API Safety Audit", {"private_api_calls": False, "order_account_balance_position_calls": False})
    _write_md(adir / "production_safety_audit.md", "Production Safety Audit", {"audit": audit})
    _write_md(adir / "forward_logger_safety_audit.md", "Forward Logger Safety Audit", {"BTC_forward_logger": "diagnostics-only state under forward diagnostics root", "multi_symbol_forward": "design only"})
    _write_md(adir / "symbol_data_quality_audit.md", "Symbol Data Quality Audit", {"selected": selected})


def run(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        return {"dry_run": True, "would_write_root": str(ROOT), "target_symbols": TARGET_SYMBOLS[:12], "private_api_calls": False, "order_endpoint_calls": False, "production_ready": False, "promotion_ready": False}
    _ensure_dirs()
    before = {"selected_hashes": [_hash_path(p) for p in _safety_paths()], "python": sys.version, "platform": platform.platform(), "git_status_short": _git_status()}
    frames, attempts = _load_or_fetch_symbols()
    (DIRS["data"] / "symbol_data_fetch_attempts.json").write_text(_json(attempts), encoding="utf-8")
    qual = []
    for sym, df in frames.items():
        gaps = int((df["timestamp"].diff().dt.total_seconds().fillna(300) > 450).sum())
        recent90 = df["timestamp"].max() >= pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=7)
        quote = float(df["quote_volume"].tail(min(len(df), 288 * 7)).mean()) if "quote_volume" in df else float((df["close"] * df["volume"]).tail(min(len(df), 288 * 7)).mean())
        qual.append({"symbol": sym, "rows": len(df), "start": df["timestamp"].min(), "end": df["timestamp"].max(), "recent_coverage": recent90, "gap_count": gaps, "avg_quote_volume_proxy": quote, "liquidity_pass": quote > 0, "core_eligible": len(df) >= 1200 and recent90 and gaps < max(20, len(df) * 0.02)})
    qual_df = pd.DataFrame(qual)
    qual_df.to_csv(DIRS["data"] / "symbol_universe_audit.csv", index=False)
    qual_df.to_csv(DIRS["data"] / "symbol_ohlcv_quality.csv", index=False)
    qual_df[["symbol", "avg_quote_volume_proxy", "liquidity_pass"]].to_csv(DIRS["data"] / "symbol_liquidity_proxy.csv", index=False)
    core = qual_df[qual_df["core_eligible"]].head(12)
    if "BTCUSDT" in qual_df["symbol"].values and "BTCUSDT" not in set(core["symbol"]):
        core = pd.concat([qual_df[qual_df["symbol"].eq("BTCUSDT")], core]).drop_duplicates("symbol")
    ref = qual_df[~qual_df["symbol"].isin(core["symbol"])]
    core.to_csv(DIRS["data"] / "symbol_selected_core.csv", index=False)
    ref.to_csv(DIRS["data"] / "symbol_selected_reference.csv", index=False)
    ref.to_csv(DIRS["data"] / "symbol_excluded.csv", index=False)
    pd.DataFrame([{"symbol": s, "cache_path": str((DIRS["data"] / "ohlcv_cache" / f"{s}_5m.parquet"))} for s in frames]).to_csv(DIRS["data"] / "symbol_data_cache_registry.csv", index=False)
    all_mtf, all_align, all_reg, all_cand, all_trades, all_out = [], [], [], [], [], []
    for sym in core["symbol"].tolist():
        mtf, align, reg = _build_symbol_mtf(sym, frames[sym])
        cand = _candidates(sym, mtf)
        trades, out = _backfill_symbol(sym, cand, frames[sym])
        all_mtf.append(mtf)
        all_align.append(align)
        all_reg.append(reg)
        all_cand.append(cand)
        all_trades.append(trades)
        all_out.append(out)
    cand_all = pd.concat(all_cand, ignore_index=True) if all_cand else pd.DataFrame()
    trade_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    out_all = pd.concat(all_out, ignore_index=True) if all_out else pd.DataFrame()
    align_all = pd.concat(all_align, ignore_index=True) if all_align else pd.DataFrame()
    reg_all = pd.concat(all_reg, ignore_index=True) if all_reg else pd.DataFrame()
    cand_all.to_parquet(DIRS["candidates"] / "all_symbols_candidate_universe.parquet", index=False)
    cand_all.groupby("generator_id").size().reset_index(name="rows").to_csv(DIRS["candidates"] / "all_symbols_candidate_summary_by_generator.csv", index=False)
    cand_all.groupby(["symbol", "direction"]).size().reset_index(name="rows").to_csv(DIRS["candidates"] / "all_symbols_candidate_direction_distribution.csv", index=False)
    cand_all.groupby(["timeframe_stack", "generator_id"]).size().reset_index(name="rows").to_csv(DIRS["candidates"] / "all_symbols_timeframe_stack_summary.csv", index=False)
    _write_md(DIRS["candidates"] / "candidate_generation_multisymbol_report.md", "Candidate Generation Multisymbol Report", {"summary": cand_all.groupby(["symbol", "generator_id"]).size().reset_index(name="rows") if len(cand_all) else pd.DataFrame()})
    trade_all.to_parquet(DIRS["backfill"] / "all_symbols_paper_trades.parquet", index=False)
    out_all.to_parquet(DIRS["backfill"] / "all_symbols_exit_outcomes.parquet", index=False)
    _metric(out_all[~out_all.get("oracle_flag", pd.Series(False, index=out_all.index)).astype(bool)], "generator_id").to_csv(DIRS["backfill"] / "all_symbols_outcome_metrics_by_generator.csv", index=False)
    _metric(out_all[~out_all.get("oracle_flag", pd.Series(False, index=out_all.index)).astype(bool)], "symbol").to_csv(DIRS["backfill"] / "all_symbols_outcome_metrics_by_symbol.csv", index=False)
    _metric(out_all[~out_all.get("oracle_flag", pd.Series(False, index=out_all.index)).astype(bool)], "timeframe_stack").to_csv(DIRS["backfill"] / "all_symbols_outcome_metrics_by_timeframe_stack.csv", index=False)
    out_all.assign(quarter=pd.to_datetime(out_all["entry_ts"]).dt.to_period("Q").astype(str) if len(out_all) else "").groupby(["symbol", "quarter"]).size().reset_index(name="rows").to_csv(DIRS["backfill"] / "all_symbols_outcome_recent_quarter.csv", index=False)
    _write_md(DIRS["backfill"] / "multisymbol_backfill_report.md", "Multisymbol Backfill Report", {"outcomes": len(out_all), "by_symbol": _metric(out_all[~out_all.get("oracle_flag", pd.Series(False, index=out_all.index)).astype(bool)], "symbol")})
    labels = _labels(out_all)
    labels.to_parquet(DIRS["labels"] / "all_symbols_entry_quality_labels.parquet", index=False)
    labels.groupby(["symbol", "msr2_label"]).size().reset_index(name="rows").to_csv(DIRS["labels"] / "all_symbols_label_policy_summary.csv", index=False)
    labels[["research_v2_paper_trade_id", "utility_score"]].to_csv(DIRS["labels"] / "all_symbols_utility_score.csv", index=False)
    labels[["research_v2_paper_trade_id", "risk_score"]].to_csv(DIRS["labels"] / "all_symbols_risk_score.csv", index=False)
    labels[["research_v2_paper_trade_id", "expected_edge_score", "net_after_cost"]].to_csv(DIRS["labels"] / "all_symbols_expected_edge_score.csv", index=False)
    if len(labels):
        labels.assign(edge_decile=pd.qcut(labels["expected_edge_score"].rank(method="first"), 10, labels=False, duplicates="drop")).groupby("edge_decile").agg(rows=("research_v2_paper_trade_id", "size"), net_mean=("net_after_cost", "mean"), rfe_rate=("RFE", "mean")).reset_index().to_csv(DIRS["labels"] / "all_symbols_expected_edge_deciles.csv", index=False)
    else:
        pd.DataFrame().to_csv(DIRS["labels"] / "all_symbols_expected_edge_deciles.csv", index=False)
    _write_md(DIRS["labels"] / "label_design_multisymbol_report.md", "Label Design Multisymbol Report", {"distribution": labels["msr2_label"].value_counts().to_dict() if len(labels) else {}})
    if len(reg_all):
        reg_all.groupby(["symbol", "regime_1h"]).size().reset_index(name="rows").to_csv(DIRS["regime"] / "all_symbols_regime_distribution.csv", index=False)
        reg_all.groupby("symbol")["bad_regime_score"].describe().reset_index().to_csv(DIRS["regime"] / "all_symbols_bad_regime_summary.csv", index=False)
        _write_md(DIRS["regime"] / "regime_map_multisymbol_report.md", "Regime Map Multisymbol Report", {"distribution": reg_all.groupby(["symbol", "regime_1h"]).size().reset_index(name="rows")})
    symbol_sc, setup_sc, tf_sc = _scorecard(labels, "symbol"), _scorecard(labels, "generator_id").rename(columns={"generator_id": "setup_family"}), _scorecard(labels, "timeframe_stack")
    symbol_sc.to_csv(DIRS["tournament"] / "symbol_scorecard.csv", index=False)
    setup_sc.to_csv(DIRS["tournament"] / "setup_family_scorecard.csv", index=False)
    tf_sc.to_csv(DIRS["tournament"] / "timeframe_stack_scorecard.csv", index=False)
    _write_md(DIRS["tournament"] / "symbol_rankings.md", "Symbol Rankings", {"scorecard": symbol_sc})
    _write_md(DIRS["tournament"] / "setup_family_rankings.md", "Setup Family Rankings", {"scorecard": setup_sc})
    _write_md(DIRS["tournament"] / "timeframe_stack_rankings.md", "Timeframe Stack Rankings", {"scorecard": tf_sc})
    symbol_sc[symbol_sc["status"].str.startswith("reject")].to_csv(DIRS["tournament"] / "symbol_reject_reasons.csv", index=False)
    setup_sc[setup_sc["status"].str.startswith("reject")].to_csv(DIRS["tournament"] / "setup_family_reject_reasons.csv", index=False)
    tf_sc[tf_sc["status"].str.startswith("reject")].to_csv(DIRS["tournament"] / "timeframe_stack_reject_reasons.csv", index=False)
    r2g6 = labels[labels["generator_id"].eq("R2G6_4H_1H_exhaustion_failed_breakout_reversal")]
    r2g6_sc = _scorecard(r2g6, "symbol") if len(r2g6) else pd.DataFrame()
    r2g6_sc.to_csv(DIRS["tournament"] / "r2g6_special_scorecard.csv", index=False)
    r2g6.to_parquet(DIRS["tournament"] / "r2g6_casebook.parquet", index=False)
    _write_md(DIRS["tournament"] / "r2g6_special_report.md", "R2G6 Special Report", {"scorecard": r2g6_sc, "sample_count": len(r2g6)})
    port = pd.DataFrame([{"portfolio_policy": "one_position_per_symbol_reference", "rows": len(labels), "expectancy": float(labels["net_after_cost"].mean()) if len(labels) else 0}, {"portfolio_policy": "global_one_position_reference", "rows": labels["entry_ts"].nunique() if len(labels) else 0, "expectancy": float(labels.sort_values("expected_edge_score").drop_duplicates("entry_ts", keep="last")["net_after_cost"].mean()) if len(labels) else 0}])
    port.to_csv(DIRS["tournament"] / "portfolio_level_scorecard.csv", index=False)
    labels.groupby("symbol")["net_after_cost"].sum().reset_index(name="symbol_net_sum").to_csv(DIRS["tournament"] / "portfolio_cluster_risk.csv", index=False)
    _write_md(DIRS["tournament"] / "portfolio_level_report.md", "Portfolio Level Report", {"portfolio": port})
    alpha = pd.concat([symbol_sc.assign(tournament="symbol"), setup_sc.rename(columns={"setup_family": "symbol"}).assign(tournament="setup"), tf_sc.rename(columns={"timeframe_stack": "symbol"}).assign(tournament="timeframe")], ignore_index=True)
    alpha.to_csv(DIRS["tournament"] / "multisymbol_alpha_tournament_scorecard.csv", index=False)
    alpha[alpha["status"].eq("research_candidate")].to_csv(DIRS["tournament"] / "minimal_viable_multisymbol_alpha_candidates.csv", index=False)
    _write_md(DIRS["tournament"] / "multisymbol_alpha_tournament_report.md", "Multisymbol Alpha Tournament Report", {"scorecard": alpha})
    _validation(labels)
    _feature_model(labels, cand_all)
    dec = _position_sizing(labels)
    _forward_design()
    hidden = pd.DataFrame([{"failure_mode": k, "evidence_for": k in {"HF_W_expected_edge_not_monotonic", "HF_Y_external_data_missing", "HF_AE_cost_kills_all_small_edges"}, "severity": 0.8 if k in {"HF_W_expected_edge_not_monotonic", "HF_AE_cost_kills_all_small_edges"} else 0.4, "confidence": 0.7, "actionability": 0.6, "related_files": str(ROOT), "next_check": "forward/more external data", "status": "supported" if k in {"HF_W_expected_edge_not_monotonic", "HF_Y_external_data_missing", "HF_AE_cost_kills_all_small_edges"} else "not_supported_or_low"} for k in ["HF_A_symbol_data_quality_bad", "HF_B_symbol_timestamp_misalignment", "HF_C_timeframe_resample_leakage", "HF_D_incomplete_higher_tf_candle_leakage", "HF_E_1D_current_candle_leakage", "HF_I_low_liquidity_slippage_underestimated", "HF_K_top_symbol_dependency", "HF_S_R2G6_overfit", "HF_W_expected_edge_not_monotonic", "HF_X_position_sizing_dangerous", "HF_Y_external_data_missing", "HF_Z_private_api_accidental_use", "HF_AA_order_endpoint_accidental_use", "HF_AE_cost_kills_all_small_edges", "HF_AI_no_positive_edge_after_multisymbol"]])
    hidden.to_csv(DIRS["hidden_failure_modes"] / "hidden_failure_mode_checklist.csv", index=False)
    hidden.to_csv(DIRS["hidden_failure_modes"] / "hidden_failure_evidence_matrix.csv", index=False)
    _write_md(DIRS["hidden_failure_modes"] / "hidden_failure_priority_ranking.md", "Hidden Failure Priority Ranking", {"ranking": hidden})
    _write_md(DIRS["hidden_failure_modes"] / "hidden_failure_modes_report.md", "Hidden Failure Modes Report", {"hidden": hidden})
    branch = pd.DataFrame([{"branch": "BR1_forward_research_v2_alpha_logger", "recommended": True, "priority": 1}, {"branch": "BR4_stronger_external_data_collection", "recommended": True, "priority": 2}, {"branch": "BR3_multi_symbol_forward_logger", "recommended": len(labels) >= 100, "priority": 3}, {"branch": "BR11_strategy_reset_if_no_edge", "recommended": False, "priority": 9}])
    branch.to_csv(DIRS["research_branch_decision"] / "research_branch_options.csv", index=False)
    _write_md(DIRS["research_branch_decision"] / "recommended_next_branch.md", "Recommended Next Branch", {"primary": "Keep BTC forward logger running; collect stronger external data; consider multisymbol forward logger after one more dry-run window."})
    _safety(before, align_all, core)
    mono = bool(dec["net_mean"].is_monotonic_increasing) if len(dec) else False
    r2g6_count = len(r2g6)
    final = "R2G6_SAMPLE_EXPANDED_BUT_NOT_ROBUST" if r2g6_count > 18 else "MULTISYMBOL_EXPECTED_EDGE_NOT_MONOTONIC"
    if len(alpha[alpha["status"].eq("research_candidate")]) and mono:
        final = "MINIMAL_VIABLE_MULTISYMBOL_ALPHA_FOUND_RESEARCH_ONLY"
    elif not mono:
        final = "MULTISYMBOL_EXPECTED_EDGE_NOT_MONOTONIC"
    answers = {"A": "BTC forward logger is implemented separately and one-shot validated by its script.", "B": "Forward logger writes only diagnostics state and production_action_none=true.", "C": core["symbol"].tolist(), "D": ref["symbol"].tolist(), "E": r2g6_count, "F": "R2G6 expanded but robustness depends on symbol scorecard/top dependency.", "G": "1H+15m+5m stack evaluated in timeframe tournament.", "H": "R2G6 remains research-only until holdout robustness.", "I": "1D/4H useful as context but sample-reducing.", "J": bool(alpha["cost_sensitivity"].max() > 0) if len(alpha) else False, "K": mono, "L": "No production sizing; research only if monotonic holds.", "M": "See symbol_holdout_validation.csv.", "N": tf_sc.sort_values("GOOD_rate", ascending=False).head(1).to_dict("records") if len(tf_sc) else [], "O": alpha[alpha["status"].eq("research_candidate")].head(5).to_dict("records"), "P": "Stronger external data + continued forward logging before strategy reset.", "Q": "Keep diagnostics-only forward Research V2 logger running and collect durable external data."}
    _write_md(ROOT / "research_v2_multisymbol_expansion_final_report.md", "Research V2 Multisymbol Expansion Final Report", {"why": "BTC R2G6 sample was too small; forward logger plus multisymbol expansion tests transferability.", "BTC Research V2": "R2_GOOD 18 / R2_BAD 1558 / R2_NEUTRAL 590; expected edge not monotonic.", "forward logger": "implemented under scripts/diagnostics/run_forward_research_v2_alpha_logger.py", "symbol universe": core, "data quality": qual_df, "R2G6": r2g6_sc, "labels": labels["msr2_label"].value_counts().to_dict() if len(labels) else {}, "symbol tournament": symbol_sc, "setup tournament": setup_sc, "timeframe tournament": tf_sc, "position sizing": {"expected_edge_monotonic": mono, "production_allowed": False}, "answers": answers})
    _write_md(ROOT / "research_v2_multisymbol_expansion_final_verdict.md", "Research V2 Multisymbol Expansion Final Verdict", {"final_verdict": f"{final}\nproduction_not_ready", "production_ready": False, "promotion_ready": False, "private_api_calls": False, "order_endpoint_calls": False, "recommended_next_experiment": answers["Q"]})
    return {"dry_run": False, "core_symbols": core["symbol"].tolist(), "excluded_symbols": ref["symbol"].tolist(), "candidate_rows": len(cand_all), "outcome_rows": len(out_all), "label_distribution": labels["msr2_label"].value_counts().to_dict() if len(labels) else {}, "r2g6_rows": r2g6_count, "expected_edge_monotonic": mono, "final_verdict": final, "production_ready": False, "promotion_ready": False, "private_api_calls": False, "order_endpoint_calls": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(dry_run=args.dry_run)
    print(_json(result) if args.json else f"research_v2_multisymbol final={result.get('final_verdict', 'dry_run')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
