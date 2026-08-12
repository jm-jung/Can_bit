"""
Daily Shadow Ops: 하루치 OHLCV 바를 Shadow 엔진에 통과시켜
리포트(JSON+MD)를 생성하고, Discord로 전송한다.

Usage:
    python -m scripts.run_daily_shadow [--dry-run] [--no-discord]
"""
from __future__ import annotations

import json
import math
import logging
import sys
import traceback
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ops.state_manager import StateManager
from src.ops.risk_manager import RiskManager
from src.ops.execution_router import ExecutionRouter
from src.ops.monitor import Monitor
from src.ops.trading_engine import ENTROPY_THRESHOLD, TickContext, TradingEngine
from src.strategy_filters.regime_ablation import compute_regime_components

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("daily_shadow")

MONITORING_DIR = Path("data/monitoring")
STATE_DIR = Path("data/state")
SHADOW_STATE_FILE = STATE_DIR / "shadow_daily_state.json"

# ── Proba 캐시 / 정렬 기본값 ─────────────────────────────────────────────
PROBA_CACHE_DEFAULT_PATH = Path("data/cache/ml_predictions/ml_tcn_BTCUSDT_5m_proba.parquet")
ALLOW_LAG_MINUTES_DEFAULT = 5.0
RAW_SIGNAL_ENTROPY_THRESHOLD = 1.0  # 운영 엔진과 동일 임계값
FALLBACK_PROBA = (0.33, 0.33, 0.34)

# Shadow 리포트: 최근 윈도 / 가상 round-trip
RECENT_WINDOW_HOURS = 24.0
SHADOW_MAX_HOLDING_BARS = 12  # 가상 청산: 진입 후 고정 N바 종가 청산
SHADOW_POSITION_SIZE = 1.0  # 가상 포지션 크기 (equity *= 1 + r * size)


def load_ohlcv() -> Optional[pd.DataFrame]:
    """Load OHLCV data, preferring parquet then CSV."""
    candidates = [
        Path("data/btc_1h.parquet"),
        Path("data/ohlcv_1h.parquet"),
        Path("data/btc_data.parquet"),
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_parquet(p)
                if "close" in df.columns and len(df) > 200:
                    return df
            except Exception:
                continue

    csv_dir = Path("data/ohlcv")
    if csv_dir.exists():
        csvs = sorted(csv_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
        for p in csvs:
            try:
                df = pd.read_csv(p)
                if "close" in df.columns and len(df) > 200:
                    return df
            except Exception:
                continue

    for p in sorted(Path("data").glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            df = pd.read_csv(p)
            if "close" in df.columns and len(df) > 200:
                return df
        except Exception:
            continue

    return None


def diagnose_proba_source(
    df_ohlcv: Optional[pd.DataFrame], proba_meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    simulate_signals가 만든 proba_meta를 summary용 진단 dict로 변환.
    이중 로딩 제거 — _load_and_align_proba가 만든 결과를 그대로 표현.
    """
    SHADOW_EXPECTED_COLS = ["p_long", "p_short", "p_flat"]
    m = proba_meta or {}

    cols = m.get("cache_columns") or []
    has_3class = all(c in cols for c in SHADOW_EXPECTED_COLS) if cols else False
    has_2class = ("proba_long" in cols) and ("proba_short" in cols) if cols else False

    if m.get("schema") == "3class":
        match_status = "MATCH"
        note = "캐시에 p_long/p_short/p_flat 3-class 컬럼이 모두 존재"
    elif m.get("schema") == "2class":
        match_status = "DERIVED"
        note = "캐시는 2-class(proba_long, proba_short). p_flat = 1 - p_long - p_short로 유도"
    elif m.get("schema") == "unknown" and m.get("cache_exists"):
        match_status = "UNKNOWN_SCHEMA"
        note = m.get("fallback_reason") or "예상 컬럼이 없음"
    else:
        match_status = "UNKNOWN"
        note = m.get("fallback_reason") or ""

    info: Dict[str, Any] = {
        "expected_columns_in_shadow_code": SHADOW_EXPECTED_COLS,
        "daily_run_cache_path": m.get("cache_path", str(PROBA_CACHE_DEFAULT_PATH)),
        "shadow_cache_path": m.get("cache_path", "<unknown>"),
        "paths_identical": True,  # shadow와 daily_run이 같은 cache_path를 본다
        "proba_loaded": bool(m.get("proba_loaded", False)),
        "cache_exists": bool(m.get("cache_exists", False)),
        "cache_mtime": m.get("cache_mtime"),
        "cache_rows": m.get("cache_rows"),
        "cache_columns": cols,
        "cache_ts_min": m.get("cache_ts_min"),
        "cache_ts_max": m.get("cache_ts_max"),
        "schema": m.get("schema", "unknown"),
        "column_map": m.get("column_map", {}),
        "column_match": {
            "p_long_in_cache": "p_long" in (cols or []),
            "p_short_in_cache": "p_short" in (cols or []),
            "p_flat_in_cache": "p_flat" in (cols or []),
            "proba_long_in_cache": "proba_long" in (cols or []),
            "proba_short_in_cache": "proba_short" in (cols or []),
            "match_status": match_status,
            "note": note,
        },
        "alignment": {
            "ohlcv_ts_max": m.get("ohlcv_ts_max"),
            "proba_ts_max": m.get("cache_ts_max"),
            "allow_lag_minutes": m.get("allow_lag_minutes"),
            "aligned_count": m.get("aligned_count"),
            "aligned_ratio": m.get("aligned_ratio"),
            "aligned": bool(m.get("proba_loaded", False)),
        },
        "p_sum_check": {
            "p_sum_min": m.get("p_sum_min"),
            "p_sum_max": m.get("p_sum_max"),
            "normalize_applied": m.get("p_sum_normalize_applied", False),
        },
        "fallback": {
            "fallback_used": bool(m.get("fallback_used", True)),
            "fallback_reason": m.get("fallback_reason"),
            "fallback_ticks_count": m.get("fallback_ticks_count"),
            "fallback_ticks_ratio": m.get("fallback_ticks_ratio"),
        },
        "raw_signal_meta": {
            "raw_signal_long_emitted": m.get("raw_signal_long_emitted"),
            "raw_signal_short_emitted": m.get("raw_signal_short_emitted"),
            "raw_signal_candidate_total": m.get("raw_signal_candidate_total"),
            "raw_signal_blocked_by_local_entropy": m.get("raw_signal_blocked_by_local_entropy"),
        },
        "signal_rule": m.get("signal_rule", {}),
    }
    return info


def _load_and_align_proba(
    df_ohlcv: pd.DataFrame,
    cache_path: Path = PROBA_CACHE_DEFAULT_PATH,
    allow_lag_minutes: float = ALLOW_LAG_MINUTES_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Shadow 전용 proba cache 로더 + OHLCV timestamp 정렬.

    Returns:
        p_long_arr   : shape (N,) float, NaN이면 fallback
        p_short_arr  : shape (N,) float
        p_flat_arr   : shape (N,) float
        valid_mask   : shape (N,) bool — True면 실제 cache proba가 정렬됨
        meta         : dict (proba_loaded / schema / fallback_reason / lag / alignment 등)
    """
    n = len(df_ohlcv)
    p_long_arr = np.full(n, np.nan, dtype=float)
    p_short_arr = np.full(n, np.nan, dtype=float)
    p_flat_arr = np.full(n, np.nan, dtype=float)
    valid_mask = np.zeros(n, dtype=bool)

    meta: Dict[str, Any] = {
        "cache_path": str(cache_path),
        "cache_exists": False,
        "cache_mtime": None,
        "cache_rows": 0,
        "cache_columns": [],
        "cache_ts_min": None,
        "cache_ts_max": None,
        "ohlcv_ts_min": None,
        "ohlcv_ts_max": None,
        "schema": "unknown",
        "column_map": {},
        "p_sum_min": None,
        "p_sum_max": None,
        "p_sum_normalize_applied": False,
        "allow_lag_minutes": float(allow_lag_minutes),
        "aligned_count": 0,
        "aligned_ratio": 0.0,
        "proba_loaded": False,
        "fallback_used": True,
        "fallback_reason": None,
    }

    if "timestamp" not in df_ohlcv.columns:
        meta["fallback_reason"] = "ohlcv has no 'timestamp' column"
        return p_long_arr, p_short_arr, p_flat_arr, valid_mask, meta

    o_ts = pd.to_datetime(df_ohlcv["timestamp"])
    meta["ohlcv_ts_min"] = str(o_ts.min())
    meta["ohlcv_ts_max"] = str(o_ts.max())

    if not cache_path.exists():
        meta["fallback_reason"] = f"cache file not found: {cache_path}"
        return p_long_arr, p_short_arr, p_flat_arr, valid_mask, meta

    meta["cache_exists"] = True
    meta["cache_mtime"] = datetime.fromtimestamp(cache_path.stat().st_mtime).isoformat()

    try:
        cdf = pd.read_parquet(cache_path)
    except Exception as e:
        meta["fallback_reason"] = f"cache read failed: {e}"
        return p_long_arr, p_short_arr, p_flat_arr, valid_mask, meta

    meta["cache_rows"] = int(len(cdf))
    meta["cache_columns"] = list(cdf.columns)

    if "timestamp" not in cdf.columns:
        meta["fallback_reason"] = "cache has no 'timestamp' column"
        return p_long_arr, p_short_arr, p_flat_arr, valid_mask, meta

    cdf = cdf.copy()
    cdf["timestamp"] = pd.to_datetime(cdf["timestamp"])
    meta["cache_ts_min"] = str(cdf["timestamp"].min())
    meta["cache_ts_max"] = str(cdf["timestamp"].max())

    # 스키마 자동 인식: 3-class 우선, 없으면 2-class (proba_long/proba_short)
    if all(c in cdf.columns for c in ["p_long", "p_short", "p_flat"]):
        meta["schema"] = "3class"
        meta["column_map"] = {"p_long": "p_long", "p_short": "p_short", "p_flat": "p_flat"}
        proba_df = cdf[["timestamp", "p_long", "p_short", "p_flat"]].copy()
    elif all(c in cdf.columns for c in ["proba_long", "proba_short"]):
        meta["schema"] = "2class"
        meta["column_map"] = {
            "p_long": "proba_long",
            "p_short": "proba_short",
            "p_flat": "1 - proba_long - proba_short (derived)",
        }
        proba_df = cdf[["timestamp", "proba_long", "proba_short"]].copy()
        proba_df.columns = ["timestamp", "p_long", "p_short"]
        proba_df["p_flat"] = (1.0 - proba_df["p_long"] - proba_df["p_short"]).clip(lower=0.0)
    else:
        meta["fallback_reason"] = (
            f"unknown schema. cache columns={list(cdf.columns)[:20]}; "
            f"need (p_long,p_short,p_flat) or (proba_long,proba_short)"
        )
        return p_long_arr, p_short_arr, p_flat_arr, valid_mask, meta

    # 합 검증 및 normalize (편차 > 1% 이상일 때만)
    p_sum = proba_df["p_long"] + proba_df["p_short"] + proba_df["p_flat"]
    p_sum_safe = p_sum.replace(0.0, np.nan)
    meta["p_sum_min"] = round(float(np.nanmin(p_sum.values)), 6)
    meta["p_sum_max"] = round(float(np.nanmax(p_sum.values)), 6)
    if (np.abs(p_sum - 1.0) > 0.01).any():
        proba_df["p_long"] = proba_df["p_long"] / p_sum_safe
        proba_df["p_short"] = proba_df["p_short"] / p_sum_safe
        proba_df["p_flat"] = proba_df["p_flat"] / p_sum_safe
        meta["p_sum_normalize_applied"] = True

    # OHLCV timestamp 기준 nearest merge (tolerance)
    o_df = pd.DataFrame({"timestamp": o_ts, "_ohlcv_idx": np.arange(n)})
    o_df = o_df.sort_values("timestamp").reset_index(drop=True)
    proba_df = proba_df.sort_values("timestamp").reset_index(drop=True)

    try:
        merged = pd.merge_asof(
            o_df,
            proba_df,
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=allow_lag_minutes),
        )
    except Exception as e:
        meta["fallback_reason"] = f"merge_asof failed: {e}"
        return p_long_arr, p_short_arr, p_flat_arr, valid_mask, meta

    merged = merged.sort_values("_ohlcv_idx").reset_index(drop=True)
    pl = merged["p_long"].to_numpy(dtype=float)
    ps = merged["p_short"].to_numpy(dtype=float)
    pf = merged["p_flat"].to_numpy(dtype=float)
    mask = ~(np.isnan(pl) | np.isnan(ps) | np.isnan(pf))

    p_long_arr[mask] = pl[mask]
    p_short_arr[mask] = ps[mask]
    p_flat_arr[mask] = pf[mask]
    valid_mask = mask

    aligned_count = int(mask.sum())
    meta["aligned_count"] = aligned_count
    meta["aligned_ratio"] = round(float(aligned_count / max(n, 1)), 6)

    if aligned_count == 0:
        meta["fallback_reason"] = (
            f"merge_asof matched 0 rows within tolerance={allow_lag_minutes}min. "
            f"ohlcv_ts_max={meta['ohlcv_ts_max']} vs cache_ts_max={meta['cache_ts_max']}"
        )
        meta["proba_loaded"] = False
        meta["fallback_used"] = True
    else:
        meta["proba_loaded"] = True
        meta["fallback_used"] = False
        meta["fallback_reason"] = None

    return p_long_arr, p_short_arr, p_flat_arr, valid_mask, meta


def _entropy_nl(probs: List[float]) -> float:
    """Natural-log Shannon entropy."""
    e = 0.0
    for p in probs:
        if p > 1e-10:
            e -= p * math.log(p)
    return e


def _parse_tick_timestamp(t: Dict[str, Any]) -> Optional[pd.Timestamp]:
    raw = t.get("timestamp")
    if raw is None:
        return None
    try:
        ts = pd.Timestamp(raw)
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None


def _entropy_distribution_from_array(ent_arr: np.ndarray) -> Dict[str, Any]:
    if ent_arr.size == 0:
        return {
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "max": None,
            "le_1_00_count": 0,
            "le_1_05_count": 0,
            "le_1_10_count": 0,
        }
    return {
        "min": round(float(np.min(ent_arr)), 6),
        "p25": round(float(np.percentile(ent_arr, 25)), 6),
        "p50": round(float(np.percentile(ent_arr, 50)), 6),
        "p75": round(float(np.percentile(ent_arr, 75)), 6),
        "max": round(float(np.max(ent_arr)), 6),
        "le_1_00_count": int(np.sum(ent_arr <= 1.00)),
        "le_1_05_count": int(np.sum(ent_arr <= 1.05)),
        "le_1_10_count": int(np.sum(ent_arr <= 1.10)),
    }


def compute_recent_window_summary(
    ticks: List[Dict[str, Any]],
    decisions: List[str],
    reasons: List[str],
    strategies: List[str],
    entropies: List[float],
    hours: float = RECENT_WINDOW_HOURS,
) -> Dict[str, Any]:
    """
    OHLCV tick timestamp 기준 최근 `hours` 시간 구간 집계.
    timestamp 없는 tick은 윈도 집계에서 제외하고 개수만 기록.
    """
    n = len(ticks)
    if n == 0 or len(decisions) != n or len(reasons) != n or len(strategies) != n or len(entropies) != n:
        return {"ok": False, "note": "empty ticks or parallel list length mismatch"}

    ts_per_tick = [_parse_tick_timestamp(t) for t in ticks]
    missing_ts = sum(1 for ts in ts_per_tick if ts is None)
    valid_pairs = [(i, ts) for i, ts in enumerate(ts_per_tick) if ts is not None]
    if not valid_pairs:
        return {
            "ok": False,
            "window_hours": hours,
            "ticks_missing_timestamp": missing_ts,
            "note": "no parseable tick timestamps",
        }

    ts_max = max(ts for _, ts in valid_pairs)
    cutoff = ts_max - pd.Timedelta(hours=hours)
    window_idx = [i for i, ts in valid_pairs if ts >= cutoff]
    m = len(window_idx)
    if m == 0:
        return {
            "ok": True,
            "window_hours": hours,
            "reference_ts_max": str(ts_max),
            "cutoff_ts": str(cutoff),
            "total_ticks": 0,
            "ticks_missing_timestamp": missing_ts,
            "signal_count": 0,
            "activation_on_count": 0,
            "activation_off_count": 0,
            "activation_ratio": 0.0,
            "s1_count": 0,
            "s2_count": 0,
            "s1_ratio": 0.0,
            "s2_ratio": 0.0,
            "entropy_mean": 0.0,
            "entropy_std": 0.0,
            "entropy_distribution": _entropy_distribution_from_array(np.array([])),
            "signal_long_count": 0,
            "signal_short_count": 0,
            "note": "no ticks in window",
        }

    w_ent = np.array([entropies[i] for i in window_idx], dtype=float)
    w_dec = [decisions[i] for i in window_idx]
    w_rea = [reasons[i] for i in window_idx]
    w_str = [strategies[i] for i in window_idx]
    w_ticks = [ticks[i] for i in window_idx]

    activation_on = sum(1 for t in w_ticks if t.get("vol_bucket") in ("mid", "high"))
    s1_c = sum(1 for s in w_str if s == "S1")
    s2_c = sum(1 for s in w_str if s == "S2")
    s_tot = s1_c + s2_c

    sig_long = sum(
        1 for d, r in zip(w_dec, w_rea) if d == "enter" and (r or "").startswith("LONG")
    )
    sig_short = sum(
        1 for d, r in zip(w_dec, w_rea) if d == "enter" and (r or "").startswith("SHORT")
    )

    return {
        "ok": True,
        "window_hours": hours,
        "reference_ts_max": str(ts_max),
        "cutoff_ts": str(cutoff),
        "total_ticks": m,
        "ticks_missing_timestamp": missing_ts,
        "signal_count": sum(1 for d in w_dec if d == "enter"),
        "activation_on_count": activation_on,
        "activation_off_count": m - activation_on,
        "activation_ratio": round(float(activation_on / max(m, 1)), 6),
        "s1_count": s1_c,
        "s2_count": s2_c,
        "s1_ratio": round(float(s1_c / s_tot), 6) if s_tot > 0 else 0.0,
        "s2_ratio": round(float(s2_c / s_tot), 6) if s_tot > 0 else 0.0,
        "entropy_mean": round(float(np.mean(w_ent)), 6) if w_ent.size else 0.0,
        "entropy_std": round(float(np.std(w_ent)), 6) if w_ent.size else 0.0,
        "entropy_distribution": _entropy_distribution_from_array(w_ent),
        "signal_long_count": sig_long,
        "signal_short_count": sig_short,
    }


def _equity_curve_from_returns(
    returns: List[float],
    position_size: float = SHADOW_POSITION_SIZE,
) -> Dict[str, Any]:
    """
    equity *= (1 + trade_return * position_size) 반복.
    MDD: 각 시점에서 (running_peak - equity) / running_peak 의 최대.
    """
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    steps: List[Dict[str, Any]] = []
    for k, r in enumerate(returns):
        equity *= 1.0 + float(r) * float(position_size)
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
        steps.append({"after_trade": k + 1, "equity": round(float(equity), 10)})
    return {
        "shadow_final_equity": round(float(equity), 10),
        "shadow_equity_return": round(float(equity - 1.0), 10),
        "shadow_max_drawdown": round(float(max_dd), 8),
        "shadow_equity_curve_steps": steps,
    }


def compute_shadow_virtual_roundtrip(
    ticks: List[Dict[str, Any]],
    decisions: List[str],
    reasons: List[str],
    max_holding_bars: int = SHADOW_MAX_HOLDING_BARS,
    position_size: float = SHADOW_POSITION_SIZE,
    recent_window: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    엔진 entry 결정마다 가상 진입 → `max_holding_bars` 뒤 바의 종가로 청산.

    - shadow_pnl_proxy: simple return 합(기존)
    - equity curve: 매 트레이드 후 equity *= (1 + r * position_size)
    - recent_window: compute_recent_window_summary 결과를 넘기면 진입 시각이
      최근 24h 구간에 속한 트레이드만 별도 집계.
    """
    n = len(ticks)
    trades: List[Dict[str, Any]] = []
    if n == 0 or len(decisions) != n or len(reasons) != n:
        empty = {
            "shadow_pnl_proxy": 0.0,
            "shadow_trade_count": 0,
            "shadow_win_rate": 0.0,
            "shadow_avg_profit": 0.0,
            "shadow_max_holding_bars": max_holding_bars,
            "shadow_position_size": float(position_size),
            "shadow_final_equity": 1.0,
            "shadow_equity_return": 0.0,
            "shadow_max_drawdown": 0.0,
            "shadow_equity_curve_steps": [],
            "recent_24h_shadow_trade_count": 0,
            "recent_24h_shadow_pnl_proxy": 0.0,
            "recent_24h_shadow_win_rate": 0.0,
            "recent_24h_shadow_avg_profit": 0.0,
            "recent_24h_shadow_note": "no trades",
            "shadow_virtual_note": "empty",
        }
        return empty

    for i in range(n):
        if decisions[i] != "enter":
            continue
        rsn = reasons[i] or ""
        if rsn.startswith("LONG"):
            side = "LONG"
        elif rsn.startswith("SHORT"):
            side = "SHORT"
        else:
            continue
        entry_px = float(ticks[i]["price"])
        if entry_px <= 0:
            continue
        j = min(i + max_holding_bars, n - 1)
        exit_px = float(ticks[j]["price"])
        if side == "LONG":
            pnl = (exit_px - entry_px) / entry_px
        else:
            pnl = (entry_px - exit_px) / entry_px
        entry_ts = _parse_tick_timestamp(ticks[i])
        trades.append(
            {
                "tick_index": int(i),
                "side": side,
                "pnl": float(pnl),
                "entry_ts": entry_ts,
            }
        )

    pnls = [t["pnl"] for t in trades]
    cnt = len(pnls)
    wins = sum(1 for p in pnls if p > 0)

    eq_block = _equity_curve_from_returns(pnls, position_size=position_size)

    # ── 최근 24h 가상 트레이드 (진입 시각 기준) ─────────────────────────
    r24_note = ""
    r24_pnls: List[float] = []
    if recent_window and recent_window.get("ok"):
        try:
            ref = pd.Timestamp(recent_window["reference_ts_max"])
            cutoff = pd.Timestamp(recent_window["cutoff_ts"])
        except Exception:
            ref = None
            cutoff = None
        if ref is not None and cutoff is not None:
            for t in trades:
                ts = t.get("entry_ts")
                if ts is None or pd.isna(ts):
                    continue
                if cutoff <= ts <= ref:
                    r24_pnls.append(t["pnl"])
            r24_note = f"entry_ts in [{cutoff}, {ref}]"
        else:
            r24_note = "could not parse recent_window cutoff/reference"
    else:
        r24_note = "recent_window unavailable or not ok"

    r24_cnt = len(r24_pnls)
    r24_wins = sum(1 for p in r24_pnls if p > 0)

    return {
        "shadow_pnl_proxy": round(float(sum(pnls)), 8) if pnls else 0.0,
        "shadow_trade_count": int(cnt),
        "shadow_win_rate": round(float(wins / cnt), 6) if cnt else 0.0,
        "shadow_avg_profit": round(float(np.mean(pnls)), 8) if pnls else 0.0,
        "shadow_max_holding_bars": int(max_holding_bars),
        "shadow_position_size": float(position_size),
        "shadow_final_equity": eq_block["shadow_final_equity"],
        "shadow_equity_return": eq_block["shadow_equity_return"],
        "shadow_max_drawdown": eq_block["shadow_max_drawdown"],
        "shadow_equity_curve_steps": eq_block["shadow_equity_curve_steps"],
        "recent_24h_shadow_trade_count": int(r24_cnt),
        "recent_24h_shadow_pnl_proxy": round(float(sum(r24_pnls)), 8) if r24_pnls else 0.0,
        "recent_24h_shadow_win_rate": round(float(r24_wins / r24_cnt), 6) if r24_cnt else 0.0,
        "recent_24h_shadow_avg_profit": round(float(np.mean(r24_pnls)), 8) if r24_pnls else 0.0,
        "recent_24h_shadow_note": r24_note,
        "shadow_virtual_note": (
            f"independent fixed-horizon exit at +{max_holding_bars} bars; "
            f"sum_pnl=Σr; equity uses Π(1+r*{position_size})"
        ),
    }


def simulate_signals(
    df: pd.DataFrame,
    cache_path: Path = PROBA_CACHE_DEFAULT_PATH,
    allow_lag_minutes: float = ALLOW_LAG_MINUTES_DEFAULT,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    OHLCV 바마다 regime feature 계산 + proba cache 결합 + raw signal 생성.

    Raw signal 임시 기준 (사용자 명시):
        LONG  if p_long  > p_short AND entropy <= ENTROPY_THRESHOLD
        SHORT if p_short > p_long  AND entropy <= ENTROPY_THRESHOLD

    Cache 정렬 실패 / NaN 시: 상수 fallback (0.33, 0.33, 0.34) + signal=None.

    Returns:
        ticks: TickContext 호환 dict 리스트
        proba_meta: 진단 메타 (proba_loaded, fallback_*, raw_signal_*, signal_rule)
    """
    ema_above, is_sideways, is_high_vol, is_mid_vol, is_low_vol, trend_label, _ = (
        compute_regime_components(df)
    )

    p_long_arr, p_short_arr, p_flat_arr, valid_mask, proba_meta = _load_and_align_proba(
        df, cache_path=cache_path, allow_lag_minutes=allow_lag_minutes
    )

    # Raw signal 진단 (simulate_signals 내부 카운트)
    raw_long_emitted = 0
    raw_short_emitted = 0
    raw_candidate_total = 0  # entropy 무관 후보 (pl > ps OR ps > pl)
    raw_blocked_by_local_entropy = 0  # 후보였지만 entropy > 1.0이라 raw 생성 안 함
    fb_ticks = 0  # fallback proba를 쓴 tick 수

    ticks: List[Dict[str, Any]] = []
    for i in range(200, len(df)):
        if is_high_vol[i]:
            vb = "high"
        elif is_low_vol[i]:
            vb = "low"
        else:
            vb = "mid"
        trend = str(trend_label[i])

        if valid_mask[i] and not (
            np.isnan(p_long_arr[i]) or np.isnan(p_short_arr[i]) or np.isnan(p_flat_arr[i])
        ):
            pl = float(p_long_arr[i])
            ps = float(p_short_arr[i])
            pf = float(p_flat_arr[i])
            used_fallback_this_tick = False
        else:
            pl, ps, pf = FALLBACK_PROBA
            used_fallback_this_tick = True
            fb_ticks += 1

        ent = _entropy_nl([pl, ps, pf])

        signal: Optional[str] = None
        if not used_fallback_this_tick:
            if pl > ps:
                raw_candidate_total += 1
                if ent <= RAW_SIGNAL_ENTROPY_THRESHOLD:
                    signal = "LONG"
                    raw_long_emitted += 1
                else:
                    raw_blocked_by_local_entropy += 1
            elif ps > pl:
                raw_candidate_total += 1
                if ent <= RAW_SIGNAL_ENTROPY_THRESHOLD:
                    signal = "SHORT"
                    raw_short_emitted += 1
                else:
                    raw_blocked_by_local_entropy += 1

        if "timestamp" in df.columns:
            _row_ts = df["timestamp"].iloc[i]
            _ts_iso: Optional[str] = None
            if pd.notna(_row_ts):
                _ts_iso = pd.Timestamp(_row_ts).isoformat()
        else:
            _ts_iso = None

        ticks.append({
            "price": float(df["close"].iloc[i]),
            "vol_bucket": vb,
            "trend_label": trend,
            "p_long": pl,
            "p_short": ps,
            "p_flat": pf,
            "signal": signal,
            "exit_signal": False,
            "df_idx": int(i),
            "timestamp": _ts_iso,
        })

    proba_meta["fallback_ticks_count"] = int(fb_ticks)
    proba_meta["fallback_ticks_ratio"] = round(float(fb_ticks / max(len(ticks), 1)), 6)
    proba_meta["raw_signal_long_emitted"] = int(raw_long_emitted)
    proba_meta["raw_signal_short_emitted"] = int(raw_short_emitted)
    proba_meta["raw_signal_candidate_total"] = int(raw_candidate_total)
    proba_meta["raw_signal_blocked_by_local_entropy"] = int(raw_blocked_by_local_entropy)
    proba_meta["signal_rule"] = {
        "rule_source": "scripts/run_daily_shadow.py :: simulate_signals()",
        "raw_signal_logic": (
            "LONG if p_long>p_short AND entropy<=1.0; "
            "SHORT if p_short>p_long AND entropy<=1.0"
        ),
        "uses_columns": "p_long, p_short, p_flat (3-class, derived if 2-class cache)",
        "entropy_threshold_local": RAW_SIGNAL_ENTROPY_THRESHOLD,
        "engine_will_reapply_entropy_filter_on_S2": True,
    }
    return ticks, proba_meta


def run_shadow_batch(ticks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Shadow 엔진에 전체 tick 배치를 통과시키고 통계를 수집한다.
    """
    state_mgr = StateManager(state_path=SHADOW_STATE_FILE)
    risk_mgr = RiskManager(state_mgr)
    router = ExecutionRouter("shadow", state_mgr)
    monitor = Monitor(monitoring_dir=MONITORING_DIR, enable_discord=False)
    engine = TradingEngine(
        mode="shadow",
        state_mgr=state_mgr,
        risk_mgr=risk_mgr,
        router=router,
        monitor=monitor,
    )

    decisions: List[str] = []
    reasons: List[str] = []
    strategies: List[str] = []
    entropies: List[float] = []
    max_probas: List[float] = []
    margins: List[float] = []
    p_long_vals: List[float] = []
    p_short_vals: List[float] = []
    p_flat_vals: List[float] = []

    # signal diagnostics (집계만; 엔진 로직은 그대로)
    signal_before_entropy_count = 0
    signal_after_entropy_count = 0
    signal_blocked_by_entropy_count = 0
    activation_on_signal_candidate_count = 0

    for t in ticks:
        ctx = TickContext(
            price=t["price"],
            vol_bucket=t["vol_bucket"],
            trend_label=t["trend_label"],
            p_long=t["p_long"],
            p_short=t["p_short"],
            p_flat=t["p_flat"],
            signal=t["signal"],
            exit_signal=t["exit_signal"],
        )
        result = engine.process_tick(ctx)
        decisions.append(result["decision"])
        reasons.append(result.get("reason", ""))
        strategies.append(result.get("strategy", ""))

        probs_raw = [t["p_long"], t["p_short"], t["p_flat"]]
        ent = 0.0
        for p in probs_raw:
            if p > 1e-10:
                ent -= p * math.log(p)
        entropies.append(ent)

        # proba diagnostics: max_proba, margin = top1 - top2
        sorted_p = sorted(probs_raw, reverse=True)
        max_probas.append(float(sorted_p[0]))
        margins.append(float(sorted_p[0] - sorted_p[1]))
        p_long_vals.append(float(t["p_long"]))
        p_short_vals.append(float(t["p_short"]))
        p_flat_vals.append(float(t["p_flat"]))

        # signal diagnostics
        activation_on = t["vol_bucket"] in ("mid", "high")
        has_raw_signal = t.get("signal") is not None
        if activation_on:
            activation_on_signal_candidate_count += 1
            if has_raw_signal:
                signal_before_entropy_count += 1
                # rule_C_trend: trend != sideways → S2(엔트로피 필터), else S1(필터 없음)
                if t["trend_label"] != "sideways":
                    if ent <= ENTROPY_THRESHOLD:
                        signal_after_entropy_count += 1
                    else:
                        signal_blocked_by_entropy_count += 1
                else:
                    signal_after_entropy_count += 1

    engine.shutdown()

    st = state_mgr.state
    state_equity_delta = float(st.equity - st.initial_equity)

    # recent_stats는 virtual round-trip(24h 진입 필터)보다 먼저 계산되어야 함
    recent_stats: Dict[str, Any] = {}
    try:
        recent_stats = compute_recent_window_summary(
            ticks, decisions, reasons, strategies, entropies, hours=RECENT_WINDOW_HOURS
        )
    except Exception as e:
        logger.warning("compute_recent_window_summary failed (non-fatal): %s", e, exc_info=True)
        recent_stats = {
            "ok": False,
            "total_ticks": 0,
            "window_hours": RECENT_WINDOW_HOURS,
            "recent_24h_unavailable": True,
            "error": str(e),
            "note": f"compute_recent_window_summary failed: {e}",
        }
    if not isinstance(recent_stats, dict):
        recent_stats = {
            "ok": False,
            "total_ticks": 0,
            "window_hours": RECENT_WINDOW_HOURS,
            "recent_24h_unavailable": True,
            "note": "invalid recent_stats (non-dict)",
        }
    elif recent_stats.get("ok") is False and "recent_24h_unavailable" not in recent_stats:
        recent_stats = {**recent_stats, "recent_24h_unavailable": True}

    try:
        virtual_stats = compute_shadow_virtual_roundtrip(
            ticks, decisions, reasons, recent_window=recent_stats
        )
    except Exception as e:
        logger.exception("compute_shadow_virtual_roundtrip failed (non-fatal): %s", e)
        virtual_stats = compute_shadow_virtual_roundtrip([], [], [])
        virtual_stats["shadow_virtual_note"] = f"compute_shadow_virtual_roundtrip failed: {e}"

    decision_dist = dict(Counter(decisions))
    reason_dist = dict(Counter(r for r in reasons if r))
    strategy_dist = dict(Counter(s for s in strategies if s))

    total_ticks = len(ticks)
    activation_on = sum(1 for d in decisions if d != "skip" or True)
    activation_on_count = sum(
        1 for t in ticks if t["vol_bucket"] in ("mid", "high")
    )
    activation_off_count = total_ticks - activation_on_count

    s1_count = strategy_dist.get("S1", 0)
    s2_count = strategy_dist.get("S2", 0)
    s_total = s1_count + s2_count
    s1_ratio = s1_count / s_total if s_total > 0 else 0.0
    s2_ratio = s2_count / s_total if s_total > 0 else 0.0

    full_signal_long = sum(
        1 for d, r in zip(decisions, reasons) if d == "enter" and (r or "").startswith("LONG")
    )
    full_signal_short = sum(
        1 for d, r in zip(decisions, reasons) if d == "enter" and (r or "").startswith("SHORT")
    )

    ent_arr = np.array(entropies, dtype=float) if entropies else np.array([0.0])
    mp_arr = np.array(max_probas, dtype=float) if max_probas else np.array([0.0])
    mg_arr = np.array(margins, dtype=float) if margins else np.array([0.0])

    entropy_distribution = {
        "min": round(float(np.min(ent_arr)), 6),
        "p25": round(float(np.percentile(ent_arr, 25)), 6),
        "p50": round(float(np.percentile(ent_arr, 50)), 6),
        "p75": round(float(np.percentile(ent_arr, 75)), 6),
        "max": round(float(np.max(ent_arr)), 6),
        "le_1_00_count": int(np.sum(ent_arr <= 1.00)),
        "le_1_05_count": int(np.sum(ent_arr <= 1.05)),
        "le_1_10_count": int(np.sum(ent_arr <= 1.10)),
    }

    aggregate_full = {
        "scope": "full_batch",
        "total_ticks": total_ticks,
        "signal_count": decision_dist.get("enter", 0),
        "activation_on_count": activation_on_count,
        "activation_off_count": activation_off_count,
        "activation_ratio": round(float(activation_on_count / max(total_ticks, 1)), 6),
        "s1_count": s1_count,
        "s2_count": s2_count,
        "s1_ratio": round(float(s1_ratio), 6),
        "s2_ratio": round(float(s2_ratio), 6),
        "entropy_mean": round(float(np.mean(entropies)) if entropies else 0.0, 6),
        "entropy_std": round(float(np.std(entropies)) if entropies else 0.0, 6),
        "entropy_distribution": dict(entropy_distribution),
        "signal_long_count": full_signal_long,
        "signal_short_count": full_signal_short,
    }

    raw_signal_diagnostics = {
        "signal_before_entropy_count": signal_before_entropy_count,
        "signal_after_entropy_count": signal_after_entropy_count,
        "signal_blocked_by_entropy_count": signal_blocked_by_entropy_count,
        "activation_on_signal_candidate_count": activation_on_signal_candidate_count,
    }

    proba_diagnostics = {
        "max_proba_mean": round(float(np.mean(mp_arr)), 6),
        "max_proba_p75": round(float(np.percentile(mp_arr, 75)), 6),
        "max_proba_max": round(float(np.max(mp_arr)), 6),
        "margin_mean": round(float(np.mean(mg_arr)), 6),
        "margin_p75": round(float(np.percentile(mg_arr, 75)), 6),
        "margin_max": round(float(np.max(mg_arr)), 6),
    }

    # ── Proba Distribution by class ────────────────────────────────────
    def _cls_stats(name: str, arr: np.ndarray) -> Dict[str, float]:
        return {
            f"{name}_min": round(float(np.min(arr)), 6),
            f"{name}_p25": round(float(np.percentile(arr, 25)), 6),
            f"{name}_p50": round(float(np.percentile(arr, 50)), 6),
            f"{name}_p75": round(float(np.percentile(arr, 75)), 6),
            f"{name}_max": round(float(np.max(arr)), 6),
        }

    pl_arr = np.array(p_long_vals, dtype=float) if p_long_vals else np.array([0.0])
    ps_arr = np.array(p_short_vals, dtype=float) if p_short_vals else np.array([0.0])
    pf_arr = np.array(p_flat_vals, dtype=float) if p_flat_vals else np.array([0.0])
    proba_distribution_by_class: Dict[str, float] = {}
    proba_distribution_by_class.update(_cls_stats("p_long", pl_arr))
    proba_distribution_by_class.update(_cls_stats("p_short", ps_arr))
    proba_distribution_by_class.update(_cls_stats("p_flat", pf_arr))

    # ── Fallback / Default Proba 사용 여부 자동 판정 ───────────────────
    fallback_default_proba_used: Dict[str, Any] = {
        "is_fallback": False,
        "fallback_signature": None,
        "evidence": [],
    }
    if (
        float(np.std(pl_arr)) < 1e-9
        and float(np.std(ps_arr)) < 1e-9
        and float(np.std(pf_arr)) < 1e-9
    ):
        fallback_default_proba_used["is_fallback"] = True
        fallback_default_proba_used["fallback_signature"] = (
            f"p_long={pl_arr[0]:.4f}, p_short={ps_arr[0]:.4f}, p_flat={pf_arr[0]:.4f}"
        )
        fallback_default_proba_used["evidence"].append(
            "std(p_long)=std(p_short)=std(p_flat)≈0 → 전 구간 상수"
        )
    if abs(pl_arr[0] - 0.33) < 1e-3 and abs(ps_arr[0] - 0.33) < 1e-3 and abs(pf_arr[0] - 0.34) < 1e-3:
        fallback_default_proba_used["is_fallback"] = True
        fallback_default_proba_used["evidence"].append(
            "simulate_signals() 하드코딩 fallback (0.33, 0.33, 0.34)과 일치"
        )

    # Zero-trade 자동 원인 판정
    zero_trade_reason: Optional[str] = None
    if st.total_trades == 0:
        if activation_on_signal_candidate_count == 0:
            zero_trade_reason = "activation off"
        elif signal_before_entropy_count == 0:
            zero_trade_reason = "no raw signal"
        elif signal_after_entropy_count == 0:
            zero_trade_reason = "entropy too high"
        else:
            zero_trade_reason = "no market opportunity"

    summary = {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "shadow",
        "total_ticks": total_ticks,
        "activation_ratio": activation_on_count / max(total_ticks, 1),
        "activation_on_count": activation_on_count,
        "activation_off_count": activation_off_count,
        "signal_count": decision_dist.get("enter", 0),
        "shadow_signal_count": decision_dist.get("enter", 0) + decision_dist.get("exit", 0),
        "trade_count": st.total_trades,
        "s1_count": s1_count,
        "s2_count": s2_count,
        "s1_ratio": round(s1_ratio, 4),
        "s2_ratio": round(s2_ratio, 4),
        "entropy_mean": round(float(np.mean(entropies)) if entropies else 0.0, 6),
        "entropy_std": round(float(np.std(entropies)) if entropies else 0.0, 6),
        "decision_distribution": decision_dist,
        "skipped_reason_distribution": {k: v for k, v in reason_dist.items() if "skip" in decisions or True},
        "kill_switch_active": st.kill_switch_active,
        "kill_switch_reason": st.kill_switch_reason,
        "equity": st.equity,
        "daily_pnl": st.daily_pnl,
        "shadow_state_equity_delta": round(state_equity_delta, 8),
        "shadow_pnl_proxy": virtual_stats["shadow_pnl_proxy"],
        "shadow_trade_count": virtual_stats["shadow_trade_count"],
        "shadow_win_rate": virtual_stats["shadow_win_rate"],
        "shadow_avg_profit": virtual_stats["shadow_avg_profit"],
        "shadow_max_holding_bars": virtual_stats["shadow_max_holding_bars"],
        "shadow_position_size": virtual_stats["shadow_position_size"],
        "shadow_final_equity": virtual_stats["shadow_final_equity"],
        "shadow_equity_return": virtual_stats["shadow_equity_return"],
        "shadow_max_drawdown": virtual_stats["shadow_max_drawdown"],
        "shadow_equity_curve_steps": virtual_stats["shadow_equity_curve_steps"],
        "recent_24h_shadow_trade_count": virtual_stats["recent_24h_shadow_trade_count"],
        "recent_24h_shadow_pnl_proxy": virtual_stats["recent_24h_shadow_pnl_proxy"],
        "recent_24h_shadow_win_rate": virtual_stats["recent_24h_shadow_win_rate"],
        "recent_24h_shadow_avg_profit": virtual_stats["recent_24h_shadow_avg_profit"],
        "recent_24h_shadow_note": virtual_stats["recent_24h_shadow_note"],
        "shadow_virtual_note": virtual_stats["shadow_virtual_note"],
        "recent_window_24h": recent_stats,
        "aggregate_full": aggregate_full,
        "consecutive_losses": st.consecutive_losses,
        "errors": st.errors[-5:] if st.errors else [],
        "entropy_distribution": entropy_distribution,
        "raw_signal_diagnostics": raw_signal_diagnostics,
        "proba_diagnostics": proba_diagnostics,
        "proba_distribution_by_class": proba_distribution_by_class,
        "fallback_default_proba_used": fallback_default_proba_used,
        "zero_trade_reason": zero_trade_reason,
    }
    return summary


def generate_report_md(summary: Dict[str, Any]) -> str:
    """요약 dict → Markdown 리포트 문자열."""
    dd = summary.get("decision_distribution", {})
    sd = summary.get("skipped_reason_distribution", {})

    af = summary.get("aggregate_full") or {}
    rw = summary.get("recent_window_24h") or {}
    rw_ed = (rw.get("entropy_distribution") or {}) if isinstance(rw, dict) else {}

    lines = [
        f"# 【Shadow Ops】Shadow Daily Report",
        f"",
        f"**실행 시각**: {summary['run_timestamp']}",
        f"**Mode**: shadow",
        f"",
        f"## Pipeline 통계 — 전체 누적 (배치 전체)",
        f"",
        f"| 항목 | 값 |",
        f"|------|-----|",
        f"| Total Ticks | {summary['total_ticks']} |",
        f"| Activation Ratio | {summary['activation_ratio']:.2%} |",
        f"| Activation ON | {summary['activation_on_count']} |",
        f"| Activation OFF | {summary['activation_off_count']} |",
        f"| Signal Count | {summary['signal_count']} |",
        f"| Trade Count (state) | {summary['trade_count']} |",
        f"| LONG enter / SHORT enter | {af.get('signal_long_count', '—')} / {af.get('signal_short_count', '—')} |",
        f"",
        f"## Pipeline 통계 — 최근 {rw.get('window_hours', RECENT_WINDOW_HOURS):.0f}시간 윈도우",
        f"",
    ]

    if rw.get("ok"):
        lines += [
            f"- **기준**: 마지막 OHLCV 시각 `reference_ts_max` 기준 역산",
            f"- **reference_ts_max**: `{rw.get('reference_ts_max', '')}`",
            f"- **cutoff_ts**: `{rw.get('cutoff_ts', '')}`",
            f"- **timestamp 없는 tick (윈도 제외)**: {rw.get('ticks_missing_timestamp', 0)}건",
            f"",
            f"| 항목 | 값 |",
            f"|------|-----|",
            f"| Total Ticks (윈도 내) | {rw.get('total_ticks', 0)} |",
            f"| Signal Count | {rw.get('signal_count', 0)} |",
            f"| Activation Ratio | {float(rw.get('activation_ratio', 0) or 0):.2%} |",
            f"| Activation ON / OFF | {rw.get('activation_on_count', 0)} / {rw.get('activation_off_count', 0)} |",
            f"| S1 / S2 (건수) | {rw.get('s1_count', 0)} / {rw.get('s2_count', 0)} |",
            f"| S1 / S2 (비율) | {float(rw.get('s1_ratio', 0) or 0):.2%} / {float(rw.get('s2_ratio', 0) or 0):.2%} |",
            f"| Entropy mean / std | {rw.get('entropy_mean', 0):.6f} / {rw.get('entropy_std', 0):.6f} |",
            f"| LONG enter / SHORT enter | {rw.get('signal_long_count', 0)} / {rw.get('signal_short_count', 0)} |",
            f"",
            f"### 최근 윈도우 Entropy 분포",
            f"",
            f"| 통계 | 값 |",
            f"|------|-----|",
            f"| min | {rw_ed.get('min', 0) if rw_ed.get('min') is not None else '—'} |",
            f"| p25 | {rw_ed.get('p25', 0) if rw_ed.get('p25') is not None else '—'} |",
            f"| p50 | {rw_ed.get('p50', 0) if rw_ed.get('p50') is not None else '—'} |",
            f"| p75 | {rw_ed.get('p75', 0) if rw_ed.get('p75') is not None else '—'} |",
            f"| max | {rw_ed.get('max', 0) if rw_ed.get('max') is not None else '—'} |",
            f"| entropy ≤ 1.00 | {rw_ed.get('le_1_00_count', 0)}건 |",
            f"| entropy ≤ 1.05 | {rw_ed.get('le_1_05_count', 0)}건 |",
            f"| entropy ≤ 1.10 | {rw_ed.get('le_1_10_count', 0)}건 |",
            f"",
        ]
    else:
        lines += [
            f"*최근 윈도 집계 불가* (**recent_24h_unavailable**): "
            f"{rw.get('note', rw.get('error', 'unknown'))}",
            f"",
        ]

    lines += [
        f"## Strategy 사용 비율 (전체 누적)",
        f"",
        f"| 전략 | 횟수 | 비율 |",
        f"|------|------|------|",
        f"| S1 (baseline) | {summary['s1_count']} | {summary['s1_ratio']:.2%} |",
        f"| S2 (entropy+P7) | {summary['s2_count']} | {summary['s2_ratio']:.2%} |",
        f"",
        f"## Entropy 통계 (전체 누적)",
        f"",
        f"- **Mean**: {summary['entropy_mean']:.6f}",
        f"- **Std**: {summary['entropy_std']:.6f}",
        f"",
        f"## Decision 분포",
        f"",
    ]
    for k, v in sorted(dd.items(), key=lambda x: -x[1]):
        lines.append(f"- {k}: {v}")

    lines += ["", "## Skip Reason 분포", ""]
    for k, v in sorted(sd.items(), key=lambda x: -x[1])[:10]:
        lines.append(f"- {k}: {v}")

    lines += [
        "",
        "## Risk / Kill Switch",
        "",
        f"- **Kill Switch Active**: {'YES' if summary['kill_switch_active'] else 'NO'}",
    ]
    if summary["kill_switch_active"]:
        lines.append(f"- **Reason**: {summary['kill_switch_reason']}")

    # ── Signal Diagnostics 섹션 ─────────────────────────────────────────
    ed = summary.get("entropy_distribution", {})
    rs = summary.get("raw_signal_diagnostics", {})
    pd_ = summary.get("proba_diagnostics", {})

    lines += [
        "",
        "## Entropy 분포",
        "",
        "| 통계 | 값 |",
        "|------|-----|",
        f"| min | {ed.get('min', 0.0):.6f} |",
        f"| p25 | {ed.get('p25', 0.0):.6f} |",
        f"| p50 | {ed.get('p50', 0.0):.6f} |",
        f"| p75 | {ed.get('p75', 0.0):.6f} |",
        f"| max | {ed.get('max', 0.0):.6f} |",
        f"| entropy ≤ 1.00 | {ed.get('le_1_00_count', 0)}건 |",
        f"| entropy ≤ 1.05 | {ed.get('le_1_05_count', 0)}건 |",
        f"| entropy ≤ 1.10 | {ed.get('le_1_10_count', 0)}건 |",
        "",
        "## Raw Signal Diagnostics",
        "",
        f"- **signal_before_entropy_count**: {rs.get('signal_before_entropy_count', 0)}",
        f"- **signal_after_entropy_count**: {rs.get('signal_after_entropy_count', 0)}",
        f"- **signal_blocked_by_entropy_count**: {rs.get('signal_blocked_by_entropy_count', 0)}",
        f"- **activation_on_signal_candidate_count**: {rs.get('activation_on_signal_candidate_count', 0)}",
        "",
        "## Proba Diagnostics",
        "",
        "| 항목 | mean | p75 | max |",
        "|------|------|-----|-----|",
        f"| max_proba | {pd_.get('max_proba_mean', 0.0):.4f} | {pd_.get('max_proba_p75', 0.0):.4f} | {pd_.get('max_proba_max', 0.0):.4f} |",
        f"| margin    | {pd_.get('margin_mean', 0.0):.4f} | {pd_.get('margin_p75', 0.0):.4f} | {pd_.get('margin_max', 0.0):.4f} |",
        "",
    ]

    if summary.get("zero_trade_reason"):
        lines += [
            "## Zero-Trade 자동 진단",
            "",
            f"- **사유**: `{summary['zero_trade_reason']}`",
            "",
        ]

    # ── Proba Source / Alignment / Class-wise / Signal Rule / Fallback ──
    psd = summary.get("proba_source_diagnostics", {}) or {}
    cm = psd.get("column_match", {}) or {}
    al = psd.get("alignment", {}) or {}
    sr = psd.get("signal_rule", {}) or {}
    pdc = summary.get("proba_distribution_by_class", {}) or {}
    fb = summary.get("fallback_default_proba_used", {}) or {}

    psum = psd.get("p_sum_check", {}) or {}
    fb_block = psd.get("fallback", {}) or {}
    rsm = psd.get("raw_signal_meta", {}) or {}

    lines += [
        "",
        "## Proba Source",
        "",
        f"- **proba_loaded**: {'YES' if psd.get('proba_loaded') else 'NO'}",
        f"- **cache_path**: `{psd.get('daily_run_cache_path', '')}`",
        f"- **paths_identical (daily_run vs shadow)**: {'YES' if psd.get('paths_identical') else 'NO'}",
        f"- **cache_exists**: {'YES' if psd.get('cache_exists') else 'NO'}",
        f"- **mtime**: {psd.get('cache_mtime')}",
        f"- **rows**: {psd.get('cache_rows')}",
        f"- **columns**: `{psd.get('cache_columns')}`",
        f"- **schema**: `{psd.get('schema')}`",
        f"- **column_map**: `{psd.get('column_map')}`",
        f"- **컬럼 매칭 상태**: `{cm.get('match_status')}`",
        f"  - p_long/p_short/p_flat in cache: "
        f"{cm.get('p_long_in_cache')}/{cm.get('p_short_in_cache')}/{cm.get('p_flat_in_cache')}",
        f"  - proba_long/proba_short in cache: "
        f"{cm.get('proba_long_in_cache')}/{cm.get('proba_short_in_cache')}",
        f"  - 비고: {cm.get('note', '')}",
        f"- **p_sum**: min={psum.get('p_sum_min')}, max={psum.get('p_sum_max')}, "
        f"normalize_applied={psum.get('normalize_applied')}",
        "",
        "## Proba Timestamp Alignment",
        "",
        f"- **OHLCV ts_max**: {al.get('ohlcv_ts_max')}",
        f"- **Proba ts_max**: {al.get('proba_ts_max')}",
        f"- **allow_lag_minutes**: {al.get('allow_lag_minutes')}",
        f"- **aligned_count**: {al.get('aligned_count')}",
        f"- **aligned_ratio**: {al.get('aligned_ratio')}",
        f"- **정렬 OK**: {'YES' if al.get('aligned') else 'NO'}",
        "",
        "## Proba Distribution by class",
        "",
        "| 클래스 | min | p25 | p50 | p75 | max |",
        "|--------|-----|-----|-----|-----|-----|",
        f"| p_long  | {pdc.get('p_long_min', 0):.4f} | {pdc.get('p_long_p25', 0):.4f} | {pdc.get('p_long_p50', 0):.4f} | {pdc.get('p_long_p75', 0):.4f} | {pdc.get('p_long_max', 0):.4f} |",
        f"| p_short | {pdc.get('p_short_min', 0):.4f} | {pdc.get('p_short_p25', 0):.4f} | {pdc.get('p_short_p50', 0):.4f} | {pdc.get('p_short_p75', 0):.4f} | {pdc.get('p_short_max', 0):.4f} |",
        f"| p_flat  | {pdc.get('p_flat_min', 0):.4f} | {pdc.get('p_flat_p25', 0):.4f} | {pdc.get('p_flat_p50', 0):.4f} | {pdc.get('p_flat_p75', 0):.4f} | {pdc.get('p_flat_max', 0):.4f} |",
        "",
        "## Signal Rule Diagnostics",
        "",
        f"- **룰 소스**: `{sr.get('rule_source', '')}`",
        f"- **raw signal 로직**: {sr.get('raw_signal_logic', '')}",
        f"- **사용 컬럼**: {sr.get('uses_columns', '')}",
        f"- **entropy_threshold_local**: {sr.get('entropy_threshold_local')}",
        f"- **엔진 S2 entropy 재검사**: {sr.get('engine_will_reapply_entropy_filter_on_S2')}",
        f"- **raw LONG emitted**: {rsm.get('raw_signal_long_emitted')}",
        f"- **raw SHORT emitted**: {rsm.get('raw_signal_short_emitted')}",
        f"- **raw candidate total**: {rsm.get('raw_signal_candidate_total')}",
        f"- **raw blocked by local entropy**: {rsm.get('raw_signal_blocked_by_local_entropy')}",
        "",
        "## Fallback / Default Proba Used",
        "",
        f"- **Fallback 사용 여부**: {'YES' if fb.get('is_fallback') else 'NO'}",
    ]
    if fb.get("fallback_signature"):
        lines.append(f"- **Fallback signature**: `{fb['fallback_signature']}`")
    if fb.get("fallback_reason"):
        lines.append(f"- **fallback_reason**: {fb['fallback_reason']}")
    if fb_block:
        lines.append(
            f"- **fallback_ticks**: count={fb_block.get('fallback_ticks_count')}, "
            f"ratio={fb_block.get('fallback_ticks_ratio')}"
        )
    for ev in fb.get("evidence", []) or []:
        lines.append(f"- 근거: {ev}")
    lines.append("")

    steps = summary.get("shadow_equity_curve_steps") or []
    n_steps = len(steps)
    step_preview = ""
    if n_steps > 0:
        head = steps[:3]
        tail = steps[-2:] if n_steps > 5 else []
        step_preview = f"총 {n_steps}스텝 (JSON 전체). 앞 3개: `{head}`" + (
            f" … 끝 2개: `{tail}`" if tail else ""
        )
    else:
        step_preview = "(없음)"

    lines += [
        "",
        "## PnL Proxy",
        "",
        "### 가상 Round-trip (Shadow 전용, 엔진과 별도)",
        "",
        f"- **max_holding_bars**: {summary.get('shadow_max_holding_bars', SHADOW_MAX_HOLDING_BARS)}",
        f"- **position_size**: {summary.get('shadow_position_size', SHADOW_POSITION_SIZE)}",
        f"- **shadow_pnl_proxy** (Σ simple return): {summary.get('shadow_pnl_proxy', 0):.6f}",
        f"- **shadow_trade_count**: {summary.get('shadow_trade_count', 0)}",
        f"- **shadow_win_rate**: {float(summary.get('shadow_win_rate', 0) or 0):.2%}",
        f"- **shadow_avg_profit** (트레이드당 평균 return): {summary.get('shadow_avg_profit', 0):.6f}",
        "",
        "#### Equity curve (복리)",
        "",
        f"- **공식**: `equity *= (1 + trade_return * position_size)` (초기 equity=1)",
        f"- **shadow_final_equity**: {summary.get('shadow_final_equity', 1.0):.6f}",
        f"- **shadow_equity_return** (final − 1): {summary.get('shadow_equity_return', 0):.6f}",
        f"- **shadow_max_drawdown** (running-peak 대비): "
        f"{float(summary.get('shadow_max_drawdown', 0) or 0):.2%}",
        f"- **shadow_equity_curve_steps**: {step_preview}",
        f"- **비고**: {summary.get('shadow_virtual_note', '')}",
        "",
        "### 최근 24h — 가상 트레이드 (진입 시각 기준)",
        "",
        f"- **recent_24h_shadow_trade_count**: {summary.get('recent_24h_shadow_trade_count', 0)}",
        f"- **recent_24h_shadow_pnl_proxy** (Σ simple return): "
        f"{summary.get('recent_24h_shadow_pnl_proxy', 0):.6f}",
        f"- **recent_24h_shadow_win_rate**: "
        f"{float(summary.get('recent_24h_shadow_win_rate', 0) or 0):.2%}",
        f"- **recent_24h_shadow_avg_profit**: {summary.get('recent_24h_shadow_avg_profit', 0):.6f}",
        f"- **recent_24h_shadow_note**: {summary.get('recent_24h_shadow_note', '')}",
        "",
        "### State 기반 (운영 StateManager equity 변화)",
        "",
        f"- **Equity**: {summary['equity']:.6f}",
        f"- **Daily PnL**: {summary['daily_pnl']:.6f}",
        f"- **shadow_state_equity_delta**: {summary.get('shadow_state_equity_delta', 0):.6f}",
        "",
    ]

    if summary.get("errors"):
        lines += ["## Recent Errors", ""]
        for e in summary["errors"]:
            lines.append(f"- {e}")
        lines.append("")

    return "\n".join(lines)


def send_shadow_discord_report(summary: Dict[str, Any], dry_run: bool = False) -> bool:
    """
    Shadow 일일 리포트를 Discord embed로 전송한다.
    기존 notify_discord.send_discord_message 재사용.
    """
    # ── 기존 Discord embed 포맷 (주석 보존) ──
    # 기존에 사용하던 send_daily_report 형식:
    #   fields: [0] 공통 헤더, [1] 데이터 상태, [2] 전략 실험 결과,
    #           [3] 리스크 판단, [4] Decision, [5] 누적 성과,
    #           [6] 매매 밀도, [7] 리스크 제어 효과, [8] 생성된 결과물
    # 아래는 Shadow Ops 전용 신규 포맷이나 구조는 기존과 유사하게 유지.
    # ──────────────────────────────────────────

    import os

    try:
        from dotenv import load_dotenv
        try:
            load_dotenv()
        except Exception:
            pass
    except ImportError:
        pass

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url and not dry_run:
        logger.warning("[Discord] DISCORD_WEBHOOK_URL not set. Skipping.")
        return False

    dd = summary.get("decision_distribution", {})
    sd = summary.get("skipped_reason_distribution", {})

    # 상위 5개 skip reason
    top_reasons = sorted(sd.items(), key=lambda x: -x[1])[:5]
    reason_text = "\n".join(f"- {k}: {v}건" for k, v in top_reasons) if top_reasons else "- 없음"

    ks_status = "YES" if summary["kill_switch_active"] else "NO"
    ks_text = f"**활성 여부**: {ks_status}"
    if summary["kill_switch_active"]:
        ks_text += f"\n**사유**: {summary['kill_switch_reason']}"

    # ── 기존 [0]~[8] 섹션 구조를 참고한 Shadow 전용 구성 ──
    # (기존 send_daily_report의 embed fields 구조:
    #   [0] 공통 헤더 - 생성 시간, 파이프라인 상태
    #   [1] 데이터 상태 - 심볼, 타임프레임, 최신 데이터 시각
    #   [2] 전략 실험 결과 - 누적 수익률, 거래 수, 승률, MDD
    #   [3] 리스크 판단 - Guard 결과, Stage-2 CAP 분포
    #   [4] Decision - 최근 7일 집계
    #   [5] 최근 누적 성과 요약
    #   [6] 매매 밀도 및 과매매 체크
    #   [7] 리스크 제어 효과 요약
    #   [8] 생성된 결과물)

    fields = [
        {
            "name": "📋 [0] 실행 정보",
            "value": (
                f"**실행 시각**: {summary['run_timestamp']}\n"
                f"**Mode**: shadow\n"
                f"**Total Ticks**: {summary['total_ticks']}"
            ),
            "inline": False,
        },
        {
            "name": "📊 [1] Activation Filter",
            "value": (
                f"**Activation Ratio**: {summary['activation_ratio']:.2%}\n"
                f"**ON (mid/high vol)**: {summary['activation_on_count']}건\n"
                f"**OFF (low vol)**: {summary['activation_off_count']}건"
            ),
            "inline": True,
        },
        {
            "name": "🔀 [2] Strategy 사용 비율",
            "value": (
                f"**S1 (baseline)**: {summary['s1_count']}건 ({summary['s1_ratio']:.1%})\n"
                f"**S2 (entropy+P7)**: {summary['s2_count']}건 ({summary['s2_ratio']:.1%})"
            ),
            "inline": True,
        },
        {
            "name": "📈 [3] Signal / Trade",
            "value": (
                f"**Signal Count**: {summary['signal_count']}건\n"
                f"**Trade Count (state)**: {summary['trade_count']}건\n"
                f"**Virtual round-trips**: {summary.get('shadow_trade_count', 0)}건\n"
                f"**Entropy Mean**: {summary['entropy_mean']:.4f}"
            ),
            "inline": True,
        },
        {
            "name": "⏭️ [4] Skip Reason 분포 (Top 5)",
            "value": reason_text,
            "inline": False,
        },
        {
            "name": "🛡️ [5] Kill Switch",
            "value": ks_text,
            "inline": True,
        },
        {
            "name": "💰 [6] PnL (Σr + equity curve + 24h + state)",
            "value": (
                f"**Σ shadow_pnl_proxy**: {summary.get('shadow_pnl_proxy', 0):.4f}\n"
                f"**equity_return** (final−1): {summary.get('shadow_equity_return', 0):.4f}  "
                f"**final_eq**: {summary.get('shadow_final_equity', 1):.4f}\n"
                f"**maxDD**: {float(summary.get('shadow_max_drawdown', 0) or 0):.2%}  "
                f"**pos_size**: {summary.get('shadow_position_size', 1)}\n"
                f"**trades**: {summary.get('shadow_trade_count', 0)}  "
                f"win {float(summary.get('shadow_win_rate', 0) or 0):.1%}  "
                f"avg {summary.get('shadow_avg_profit', 0):.5f}\n"
                f"**24h virtual**: n={summary.get('recent_24h_shadow_trade_count', 0)}  "
                f"Σ={summary.get('recent_24h_shadow_pnl_proxy', 0):.4f}  "
                f"win {float(summary.get('recent_24h_shadow_win_rate', 0) or 0):.1%}  "
                f"avg {summary.get('recent_24h_shadow_avg_profit', 0):.5f}\n"
                f"**equity_steps**: {len(summary.get('shadow_equity_curve_steps') or [])} rows (see JSON)\n"
                f"**state Δ**: {summary.get('shadow_state_equity_delta', 0):.4f}"
            )[:1020],
            "inline": False,
        },
    ]

    # ── Signal Diagnostics ─────────────────────────────────────────────
    ed = summary.get("entropy_distribution", {})
    rs = summary.get("raw_signal_diagnostics", {})
    pd_ = summary.get("proba_diagnostics", {})

    fields.append({
        "name": "🧮 [7] Entropy 분포",
        "value": (
            f"**min/p25/p50/p75/max**: "
            f"{ed.get('min', 0.0):.4f} / {ed.get('p25', 0.0):.4f} / "
            f"{ed.get('p50', 0.0):.4f} / {ed.get('p75', 0.0):.4f} / "
            f"{ed.get('max', 0.0):.4f}\n"
            f"**≤1.00**: {ed.get('le_1_00_count', 0)}건  ·  "
            f"**≤1.05**: {ed.get('le_1_05_count', 0)}건  ·  "
            f"**≤1.10**: {ed.get('le_1_10_count', 0)}건"
        ),
        "inline": False,
    })

    fields.append({
        "name": "🧪 [8] Raw Signal Diagnostics",
        "value": (
            f"**activation_on_candidate**: {rs.get('activation_on_signal_candidate_count', 0)}\n"
            f"**before_entropy**: {rs.get('signal_before_entropy_count', 0)}\n"
            f"**after_entropy**: {rs.get('signal_after_entropy_count', 0)}\n"
            f"**blocked_by_entropy**: {rs.get('signal_blocked_by_entropy_count', 0)}"
        ),
        "inline": True,
    })

    fields.append({
        "name": "🎯 [9] Proba Diagnostics",
        "value": (
            f"**max_proba** mean/p75/max: "
            f"{pd_.get('max_proba_mean', 0.0):.3f} / "
            f"{pd_.get('max_proba_p75', 0.0):.3f} / "
            f"{pd_.get('max_proba_max', 0.0):.3f}\n"
            f"**margin** mean/p75/max: "
            f"{pd_.get('margin_mean', 0.0):.3f} / "
            f"{pd_.get('margin_p75', 0.0):.3f} / "
            f"{pd_.get('margin_max', 0.0):.3f}"
        ),
        "inline": True,
    })

    if summary.get("zero_trade_reason"):
        fields.append({
            "name": "🔍 [10] Zero-Trade 자동 진단",
            "value": f"**사유**: `{summary['zero_trade_reason']}`",
            "inline": False,
        })

    # ── Proba 진단 5종 ────────────────────────────────────────────────
    psd = summary.get("proba_source_diagnostics", {}) or {}
    cm = psd.get("column_match", {}) or {}
    al = psd.get("alignment", {}) or {}
    sr = psd.get("signal_rule", {}) or {}
    pdc = summary.get("proba_distribution_by_class", {}) or {}
    fb = summary.get("fallback_default_proba_used", {}) or {}

    def _short_cols(cols: Any, n: int = 6) -> str:
        if not isinstance(cols, list):
            return str(cols)
        if len(cols) <= n:
            return str(cols)
        return f"{cols[:n]} … (+{len(cols) - n})"

    proba_loaded = bool(psd.get("proba_loaded"))
    fallback_used = bool(fb.get("is_fallback"))
    badge = (
        "✅ 실제 proba 기반 shadow"
        if (proba_loaded and not fallback_used)
        else "⚠️ FALLBACK proba 사용 중"
    )

    fields.append({
        "name": "📦 [11] Proba Source",
        "value": (
            f"**상태**: {badge}\n"
            f"**proba_loaded**: {proba_loaded}\n"
            f"**path**: `{psd.get('daily_run_cache_path', '')}`\n"
            f"**exists**: {psd.get('cache_exists')}  ·  **rows**: {psd.get('cache_rows')}\n"
            f"**mtime**: {psd.get('cache_mtime')}\n"
            f"**schema**: `{psd.get('schema')}`  ·  **match**: `{cm.get('match_status')}`\n"
            f"**columns**: `{_short_cols(psd.get('cache_columns'))}`"
        )[:1020],
        "inline": False,
    })

    fields.append({
        "name": "🕒 [12] Proba TS Alignment",
        "value": (
            f"**ohlcv_ts_max**: {al.get('ohlcv_ts_max')}\n"
            f"**proba_ts_max**: {al.get('proba_ts_max')}\n"
            f"**allow_lag_min**: {al.get('allow_lag_minutes')}\n"
            f"**aligned_count**: {al.get('aligned_count')}\n"
            f"**aligned_ratio**: {al.get('aligned_ratio')}\n"
            f"**aligned**: {al.get('aligned')}"
        ),
        "inline": True,
    })

    fields.append({
        "name": "📐 [13] Proba Dist by class (min·p50·max)",
        "value": (
            f"**p_long**: {pdc.get('p_long_min', 0):.3f} / {pdc.get('p_long_p50', 0):.3f} / {pdc.get('p_long_max', 0):.3f}\n"
            f"**p_short**: {pdc.get('p_short_min', 0):.3f} / {pdc.get('p_short_p50', 0):.3f} / {pdc.get('p_short_max', 0):.3f}\n"
            f"**p_flat**: {pdc.get('p_flat_min', 0):.3f} / {pdc.get('p_flat_p50', 0):.3f} / {pdc.get('p_flat_max', 0):.3f}"
        ),
        "inline": True,
    })

    rsm = psd.get("raw_signal_meta", {}) or {}
    fields.append({
        "name": "🧷 [14] Signal Rule Diagnostics",
        "value": (
            f"**rule_source**: `{sr.get('rule_source', '')}`\n"
            f"**raw signal**: {sr.get('raw_signal_logic', '')}\n"
            f"**uses_columns**: {sr.get('uses_columns', '')}\n"
            f"**raw LONG/SHORT emitted**: "
            f"{rsm.get('raw_signal_long_emitted')} / {rsm.get('raw_signal_short_emitted')}\n"
            f"**raw candidate / locally blocked by entropy**: "
            f"{rsm.get('raw_signal_candidate_total')} / {rsm.get('raw_signal_blocked_by_local_entropy')}"
        )[:1020],
        "inline": False,
    })

    fb_evidence = " · ".join(fb.get("evidence", []) or []) or "—"
    fb_block = psd.get("fallback", {}) or {}
    fields.append({
        "name": "⚠️ [15] Fallback / Default Proba",
        "value": (
            f"**is_fallback**: `{fb.get('is_fallback')}`\n"
            f"**signature**: `{fb.get('fallback_signature')}`\n"
            f"**fallback_reason**: {fb.get('fallback_reason')}\n"
            f"**fallback_ticks**: count={fb_block.get('fallback_ticks_count')}, "
            f"ratio={fb_block.get('fallback_ticks_ratio')}\n"
            f"**evidence**: {fb_evidence}"
        )[:1020],
        "inline": False,
    })

    log_path = MONITORING_DIR / f"trading_log_{date.today().strftime('%Y%m%d')}.jsonl"
    fields.append({
        "name": "📁 [16] 로그 파일",
        "value": f"`{log_path}`",
        "inline": False,
    })

    rw = summary.get("recent_window_24h") or {}
    rw_ed = rw.get("entropy_distribution") or {}
    if rw.get("ok") and rw.get("total_ticks", 0) > 0:
        recent_field = {
            "name": f"🕐 [17] 최근 {rw.get('window_hours', RECENT_WINDOW_HOURS):.0f}h 윈도우",
            "value": (
                f"**ticks**: {rw.get('total_ticks')}  ·  **signals**: {rw.get('signal_count')}\n"
                f"**activation**: {float(rw.get('activation_ratio', 0) or 0):.1%} "
                f"(ON {rw.get('activation_on_count')} / OFF {rw.get('activation_off_count')})\n"
                f"**S1/S2**: {rw.get('s1_count')}/{rw.get('s2_count')} "
                f"({float(rw.get('s1_ratio', 0) or 0):.0%} / {float(rw.get('s2_ratio', 0) or 0):.0%})\n"
                f"**entropy** μ={rw.get('entropy_mean'):.4f} σ={rw.get('entropy_std'):.4f}  "
                f"min…max: {rw_ed.get('min')} … {rw_ed.get('max')}\n"
                f"**LONG/SHORT enter**: {rw.get('signal_long_count')} / {rw.get('signal_short_count')}\n"
                f"**ref_ts_max** → **cutoff**: `{rw.get('reference_ts_max')}` → `{rw.get('cutoff_ts')}`"
            )[:1020],
            "inline": False,
        }
    else:
        unavail_msg = rw.get("note") or rw.get("error") or "no ticks in window or missing timestamps"
        recent_field = {
            "name": "🕐 [17] 최근 24h 윈도우",
            "value": (
                "**recent_24h_unavailable** — Shadow 배치는 계속 완료되었습니다.\n"
                f"`{unavail_msg}`"
            )[:1020],
            "inline": False,
        }
    fields.append(recent_field)

    psd_top = summary.get("proba_source_diagnostics", {}) or {}
    fb_top = summary.get("fallback_default_proba_used", {}) or {}
    is_fb = bool(fb_top.get("is_fallback"))
    is_loaded = bool(psd_top.get("proba_loaded"))

    if summary["kill_switch_active"]:
        color = 0xe74c3c
    elif is_fb:
        color = 0xf1c40f  # 노랑 — fallback 사용 경고
    else:
        color = 0x2ecc71

    desc_lines = [
        "확정 파이프라인(activation → rule_C_trend → entropy) **배치 평가**입니다. "
        "**【일일 파이프라인】** 메시지와 데이터·목적이 다릅니다.",
    ]
    if is_loaded and not is_fb:
        desc_lines.append("**[OK] 실제 proba cache 기반 shadow 실행**")
    else:
        desc_lines.append(
            f"**[FALLBACK] 상수 proba 사용 중** — reason: `{fb_top.get('fallback_reason') or 'unknown'}`"
        )

    embed = {
        "title": "【Shadow Ops】Production Shadow 일일 요약",
        "description": "\n".join(desc_lines),
        "color": color,
        "fields": fields,
        "footer": {
            "text": f"Can_bit Shadow Ops (run_daily_shadow) | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        },
        "timestamp": datetime.utcnow().isoformat(),
    }
    payload = {"embeds": [embed]}

    if dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return True

    try:
        import requests
        resp = requests.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("[Discord] Shadow daily report sent successfully")
        return True
    except ImportError:
        logger.warning("[Discord] requests module not installed")
        return False
    except Exception as e:
        logger.error(f"[Discord] Failed to send: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Daily Shadow Ops")
    parser.add_argument("--dry-run", action="store_true", help="Discord payload만 출력")
    parser.add_argument("--no-discord", action="store_true", help="Discord 전송 안함")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_json_path = MONITORING_DIR / f"shadow_daily_report_{ts}.json"
    report_md_path = MONITORING_DIR / f"shadow_daily_report_{ts}.md"
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Shadow Daily Ops 시작 ===")

    try:
        df = load_ohlcv()
        if df is None:
            error_msg = "OHLCV 데이터를 찾을 수 없습니다."
            logger.error(error_msg)
            error_summary = {
                "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "shadow",
                "status": "FAILED",
                "error": error_msg,
            }
            report_json_path.write_text(json.dumps(error_summary, indent=2), encoding="utf-8")
            if not args.no_discord and not args.dry_run:
                from src.monitoring.notify_discord import send_discord_message
                send_discord_message(
                    "【Shadow Ops】Shadow Daily Ops FAILED",
                    f"**오류**: {error_msg}\n**시각**: {error_summary['run_timestamp']}",
                    "ERROR",
                )
            sys.exit(1)

        logger.info(f"OHLCV 로드 완료: {len(df)} rows")

        ticks, proba_meta = simulate_signals(df)
        logger.info(f"Tick 생성 완료: {len(ticks)}건")

        summary = run_shadow_batch(ticks)
        summary["status"] = "SUCCESS"
        summary["proba_source_diagnostics"] = diagnose_proba_source(df, proba_meta)

        # ── fallback_default_proba_used를 proba_meta 기준으로 재판정 ─────
        fb_meta_used = bool(proba_meta.get("fallback_used", True))
        fb_meta_reason = proba_meta.get("fallback_reason")
        evidence: List[str] = []
        if fb_meta_used:
            evidence.append(f"proba_meta.fallback_used=true ({fb_meta_reason})")
        else:
            ratio = proba_meta.get("fallback_ticks_ratio")
            if ratio and ratio > 0:
                evidence.append(f"일부 tick fallback (ratio={ratio:.4f})")
            else:
                evidence.append(
                    f"실제 proba cache 사용 (schema={proba_meta.get('schema')}, "
                    f"aligned_ratio={proba_meta.get('aligned_ratio')})"
                )
        summary["fallback_default_proba_used"] = {
            "is_fallback": fb_meta_used,
            "fallback_signature": (
                f"p_long={FALLBACK_PROBA[0]:.4f}, p_short={FALLBACK_PROBA[1]:.4f}, p_flat={FALLBACK_PROBA[2]:.4f}"
                if fb_meta_used else None
            ),
            "fallback_reason": fb_meta_reason,
            "evidence": evidence,
        }

        # ── zero-trade reason 재판정 (raw signal 후보 정보 활용) ──────────
        # 사용자 요구: signal_count==0 AND trade_count==0 일 때만 4종 자동판정.
        # signal이 있었는데 trade가 0인 경우는 청산 신호가 없어서이므로 별도 문구.
        rs = summary.get("raw_signal_diagnostics", {}) or {}
        sig_cnt = int(summary.get("signal_count", 0))
        trd_cnt = int(summary.get("trade_count", 0))
        if trd_cnt == 0:
            if sig_cnt == 0:
                if rs.get("activation_on_signal_candidate_count", 0) == 0:
                    summary["zero_trade_reason"] = "activation off"
                elif fb_meta_used:
                    summary["zero_trade_reason"] = "fallback proba (no real proba loaded)"
                elif proba_meta.get("raw_signal_candidate_total", 0) == 0:
                    summary["zero_trade_reason"] = "no raw signal"
                elif rs.get("signal_after_entropy_count", 0) == 0:
                    summary["zero_trade_reason"] = "entropy too high"
                else:
                    summary["zero_trade_reason"] = "no market opportunity"
            else:
                summary["zero_trade_reason"] = (
                    f"signals emitted ({sig_cnt}) but 0 round-trips — "
                    "no exit signal (shadow는 exit_signal=False 고정)"
                )
        summary["report_json"] = str(report_json_path)
        summary["report_md"] = str(report_md_path)

        report_json_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info(f"JSON 리포트 저장: {report_json_path}")

        md_content = generate_report_md(summary)
        report_md_path.write_text(md_content, encoding="utf-8")
        logger.info(f"MD 리포트 저장: {report_md_path}")

        if not args.no_discord:
            discord_ok = send_shadow_discord_report(summary, dry_run=args.dry_run)
            if discord_ok:
                logger.info("Discord 전송 성공")
            else:
                logger.warning("Discord 전송 실패 또는 생략")

        logger.info("=== Shadow Daily Ops 완료 ===")
        today_str = date.today().strftime("%Y%m%d")
        log_file = MONITORING_DIR / f"trading_log_{today_str}.jsonl"
        print(f"\n[결과 파일]")
        print(f"  JSON: {report_json_path}")
        print(f"  MD  : {report_md_path}")
        print(f"  Log : {log_file}")

    except Exception as e:
        logger.error(f"Shadow Daily Ops 실패: {traceback.format_exc()}")
        error_summary = {
            "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "shadow",
            "status": "FAILED",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        try:
            report_json_path.write_text(json.dumps(error_summary, indent=2), encoding="utf-8")
        except Exception:
            pass

        if not args.no_discord and not args.dry_run:
            try:
                from src.monitoring.notify_discord import send_discord_message
                send_discord_message(
                    "【Shadow Ops】Shadow Daily Ops FAILED",
                    f"**오류**: {str(e)[:500]}\n**시각**: {error_summary['run_timestamp']}",
                    "ERROR",
                )
            except Exception:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
