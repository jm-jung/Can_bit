#!/usr/bin/env python3
"""
FR2 Meta Layer용 rolling metrics 공급자.

목표:
- data/diagnostics/fr2/meta_metrics_source_latest.json
  (rolling metrics source snapshot)을 최신으로 갱신한다.
- 이후 Meta Layer가 next evaluate에서 이 소스를 읽고
  meta_metrics_snapshot.json/score/alpha_score/state_log를 일관되게 갱신한다.

운영 모드(default):
- scripts/run_fr2_meta_layer.py 로직을 재사용하되, eval_date를 "마지막 1개"만 계산해서
  비용을 줄이려는 방향(아직도 무거울 수 있음)으로 구현한다.

검증/파이프라인 테스트 모드(--use-legacy-last-row):
- pandas/torch 연산 없이, state_log_legacy_*.csv의 마지막 유효 row를 가져와
  meta_metrics_source_latest.json을 갱신(단 stale-duration 관측을 위해 timestamp는 now로 설정)한다.
- 운영에서는 사용하지 말고, 배선/헬스체크 검증에만 사용한다.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import csv

import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.trading.meta.meta_metrics import resolve_latest_fr2_offline_legacy_path

FR2_DIR = PROJECT_ROOT / "data" / "diagnostics" / "fr2"
STATE_LOG_PATH = FR2_DIR / "state_log.csv"
METRICS_SOURCE_LATEST_PATH = FR2_DIR / "meta_metrics_source_latest.json"
OHLCV_5M_PATH = PROJECT_ROOT / "data" / "ohlcv" / "BTCUSDT_5m_full.csv"
META_OHLCV_SYNC_SLACK_MINUTES = 5.0
LOG_PATH = PROJECT_ROOT / "logs" / "meta_metrics_refresh.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _safe_int(x: Any) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def _read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        for row in r:
            return [c.strip() for c in row if c is not None]
    return []


def _resolve_latest_legacy_state_log(parent_dir: Path) -> Optional[Path]:
    """오프라인 FR2 meta_layer CSV만 사용 (운영용 legacy export 파일 제외)."""
    return resolve_latest_fr2_offline_legacy_path(parent_dir)


def _read_last_valid_row_from_legacy() -> Optional[dict]:
    legacy = _resolve_latest_legacy_state_log(FR2_DIR)
    if not legacy:
        return None

    last: dict | None = None
    with legacy.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades = _safe_int(row.get("trades_60d"))
            if trades <= 0:
                continue
            cost60 = _safe_float(row.get("cost_on_60d"))
            cost90 = _safe_float(row.get("cost_on_90d"))
            a60 = _safe_float(row.get("alpha_60d"))
            a90 = _safe_float(row.get("alpha_90d"))
            # legacy 스키마는 alpha_60d/alpha_90d
            if any(v != v for v in [cost60, cost90, a60, a90]):
                continue
            last = row
    return last


def _json_sanitize(obj: Any) -> Any:
    """numpy scalar / NaN 등을 JSON 직렬화 가능 형태로."""
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        fv = float(obj)
        if isinstance(fv, float) and (math.isnan(fv) or math.isinf(fv)):
            return None
        return fv
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _parse_iso_to_utc_dt(s: str) -> datetime | None:
    if not s or not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def read_ohlcv_latest_timestamp_utc() -> datetime | None:
    """BTCUSDT 5m full CSV의 마지막 캔들 시각(UTC)."""
    if not OHLCV_5M_PATH.exists():
        return None
    import pandas as pd

    try:
        df = pd.read_csv(OHLCV_5M_PATH, usecols=["timestamp"])
    except Exception:
        return None
    if df.empty or "timestamp" not in df.columns:
        return None
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").max()
    if pd.isna(ts):
        return None
    dt = ts.to_pydatetime()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def meta_asof_covers_ohlcv_latest(
    *,
    slack_minutes: float = META_OHLCV_SYNC_SLACK_MINUTES,
) -> tuple[bool, str]:
    """
    meta_metrics_source_latest.json asof_ts >= OHLCV latest - slack.
    OHLCV를 읽을 수 없으면 검증 생략( True ).
    """
    ohlcv_latest = read_ohlcv_latest_timestamp_utc()
    if ohlcv_latest is None:
        return True, "ohlcv_latest_unavailable_skip_check"
    if not METRICS_SOURCE_LATEST_PATH.exists():
        return False, "meta_json_missing"
    try:
        payload = json.loads(METRICS_SOURCE_LATEST_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"meta_json_read_error:{e}"
    asof_raw = payload.get("asof_ts") or payload.get("timestamp")
    if not asof_raw:
        return False, "meta_asof_missing"
    meta_dt = _parse_iso_to_utc_dt(str(asof_raw))
    if meta_dt is None:
        return False, "meta_asof_unparseable"
    floor = ohlcv_latest - timedelta(minutes=slack_minutes)
    if meta_dt >= floor:
        return True, "ok"
    return (
        False,
        f"lag meta_asof={meta_dt.isoformat()} ohlcv_latest={ohlcv_latest.isoformat()} floor={floor.isoformat()}",
    )


def apply_asof_stamp_fallback_to_ohlcv_latest() -> None:
    """지표 값은 유지하고 asof_ts만 OHLCV 최신에 맞춰 freshness를 복구한다."""
    ohlcv_latest = read_ohlcv_latest_timestamp_utc()
    if ohlcv_latest is None or not METRICS_SOURCE_LATEST_PATH.exists():
        return
    try:
        payload = json.loads(METRICS_SOURCE_LATEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    iso = ohlcv_latest.isoformat()
    payload["asof_ts"] = iso
    payload["timestamp"] = iso
    payload["refresh_status"] = "stale_asof_fallback"
    si = payload.get("source_info")
    if not isinstance(si, dict):
        si = {}
    si["asof_fallback"] = "ohlcv_latest_stamp_after_sync_check"
    payload["source_info"] = si
    _atomic_write_json(METRICS_SOURCE_LATEST_PATH, payload)
    _log(f"apply_asof_stamp_fallback_to_ohlcv_latest: stamped asof_ts={iso}")


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _json_sanitize(payload)
    tmp.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _json_safe_float(x: float) -> float | None:
    """JSON에 NaN/Inf를 넣지 않는다 (Meta Layer 로더와 동일하게 유효 row만 유지)."""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def _write_source_from_values(
    *,
    asof_ts: str,
    cost_on_60d: float,
    cost_on_90d: float,
    alpha_fee_ratio_60d: float,
    alpha_fee_ratio_90d: float,
    trades_60d: int,
    source_info: dict,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "timestamp": asof_ts,
        "asof_ts": asof_ts,
        "generated_at": now,
        "cost_on_60d": _json_safe_float(cost_on_60d),
        "cost_on_90d": _json_safe_float(cost_on_90d),
        "alpha_fee_ratio_60d": _json_safe_float(alpha_fee_ratio_60d),
        "alpha_fee_ratio_90d": _json_safe_float(alpha_fee_ratio_90d),
        "trades_60d": trades_60d,
        "source_info": source_info,
        "refresh_status": "ok",
    }
    _atomic_write_json(METRICS_SOURCE_LATEST_PATH, payload)


def compute_latest_metrics_heavy(
    relaxed: bool = False,
    eval_mode: str = "research_step",
    operational_scan_max_days: int = 30,
) -> None:
    """
    운영용 heavy 계산(기존 scripts/run_fr2_meta_layer.py 로직에서 last 1 eval_date만 계산).
    """
    import pandas as pd

    # 아래 상수들은 scripts/run_fr2_meta_layer.py의 핵심 파라미터를 그대로 맞춘다.
    DAYS_FULL = 720
    TIMEFRAME = "5m"
    SYMBOL = "BTCUSDT"
    COMMISSION = 0.0009
    SLIPPAGE = 0.0001
    MAX_ENTROPY = 1.30
    TIME_STOP_BARS = 72
    EARLY_EXIT_BAD_K = 8
    MIN_HOLD = 24
    COOLDOWN = 24

    THRESHOLD_PRIMARY = 0.6
    REGIME_C4_PRIMARY = False

    W_60D = 0.7
    W_90D = 0.3
    WINDOW_60D = 60
    WINDOW_90D = 90
    EVAL_STEP_DAYS = 30
    TRADES_MIN = 50
    MIN_BARS_60_90_STRICT = 200
    MIN_BARS_60_90_RELAXED = 100

    # reuse
    from scripts.run_fr2_diagnostics import MODELS_DIR, get_ohlcv_and_proba
    from scripts.run_fr2_regime_conditioning import add_regime_columns
    from scripts.run_tcn_label_sweep_v2 import run_backtest_7d
    from src.strategies.ensemble_strategy import (
        EnsembleInputs,
        build_ensemble_proba,
        build_fr2_c4_mask,
    )

    BASELINE_PT = MODELS_DIR / "tcn_h15_t0p004.pt"
    FR2_PT = MODELS_DIR / "tcn_h15_micro_v1.pt"

    def _align_by_timestamp(df_base: pd.DataFrame, pl_base: np.ndarray, ps_base: np.ndarray, df_fr2: pd.DataFrame, pl_fr2: np.ndarray, ps_fr2: np.ndarray):
        d1 = df_base.copy()
        d2 = df_fr2.copy()
        for d in (d1, d2):
            d["timestamp"] = pd.to_datetime(d["timestamp"])
        d1 = d1.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        d2 = d2.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        d1["pl_base"] = pl_base[: len(d1)]
        d1["ps_base"] = ps_base[: len(d1)]
        d2["pl_fr2"] = pl_fr2[: len(d2)]
        d2["ps_fr2"] = ps_fr2[: len(d2)]
        joined = d1.merge(d2[["timestamp", "pl_fr2", "ps_fr2"]], on="timestamp", how="inner")
        joined = joined.sort_values("timestamp").reset_index(drop=True)
        if len(joined) < 500:
            raise RuntimeError("Aligned length too small")
        df_bt = joined[["timestamp", "close", "high", "low"]].copy()
        pl_b = joined["pl_base"].to_numpy(dtype=float)
        ps_b = joined["ps_base"].to_numpy(dtype=float)
        pl_f = joined["pl_fr2"].to_numpy(dtype=float)
        ps_f = joined["ps_fr2"].to_numpy(dtype=float)
        return df_bt, pl_b, ps_b, pl_f, ps_f

    def _run_backtest_detail(df_w: pd.DataFrame, pl: np.ndarray, ps: np.ndarray, threshold: float) -> dict[str, Any]:
        """
        trades_60d: res_on['total_trades']
        alpha_fee_ratio: (cost_off/trades) / ((cost_off-cost_on)/trades) when |fee_per_trade| > 1e-12 else NaN
        -> fee_per_trade ~ 0 이면 NaN (분모 0)
        -> trades==0 이면 NaN
        -> res_on is None 이면 trades=0, NaN
        """
        res_on, err_on = run_backtest_7d(
            SYMBOL, TIMEFRAME, df_w, pl, ps,
            commission_rate=COMMISSION, slippage_rate=SLIPPAGE,
            min_max_proba=threshold, max_entropy=MAX_ENTROPY, decision_mode="argmax",
            min_hold=MIN_HOLD, cooldown=COOLDOWN,
            time_stop_enabled=True, time_stop_bars=TIME_STOP_BARS,
            early_exit_enabled=True, early_exit_bad_k=EARLY_EXIT_BAD_K,
        )
        res_off, err_off = run_backtest_7d(
            SYMBOL, TIMEFRAME, df_w, pl, ps,
            commission_rate=0.0, slippage_rate=0.0,
            min_max_proba=threshold, max_entropy=MAX_ENTROPY, decision_mode="argmax",
            min_hold=MIN_HOLD, cooldown=COOLDOWN,
            time_stop_enabled=True, time_stop_bars=TIME_STOP_BARS,
            early_exit_enabled=True, early_exit_bad_k=EARLY_EXIT_BAD_K,
        )
        if res_on is None:
            return {
                "cost_on": float("nan"),
                "alpha_fee_ratio": float("nan"),
                "trades": 0,
                "fee_per_trade": float("nan"),
                "alpha_per_trade": float("nan"),
                "res_on_none": True,
                "err_on": err_on,
                "err_off": err_off,
            }
        cost_on = float(res_on.get("total_return", np.nan))
        trades = int(res_on.get("total_trades", 0))
        cost_off = float(res_off.get("total_return", np.nan)) if res_off else float("nan")
        alpha_per_trade = float("nan")
        fee_per_trade = float("nan")
        afr = float("nan")
        if trades > 0 and res_off is not None:
            alpha_per_trade = cost_off / trades
            fee_per_trade = (cost_off - cost_on) / trades
            afr = alpha_per_trade / fee_per_trade if abs(fee_per_trade) > 1e-12 else float("nan")
        return {
            "cost_on": cost_on,
            "alpha_fee_ratio": afr,
            "trades": trades,
            "fee_per_trade": fee_per_trade,
            "alpha_per_trade": alpha_per_trade,
            "res_on_none": False,
            "err_on": err_on,
            "err_off": err_off,
        }

    def _meta_layer_accepts_metrics(
        *,
        t60: int,
        c60: float,
        a60: float,
        c90: float,
        a90: float,
        min_trades_60: int,
    ) -> bool:
        """try_load_latest_metrics_from_state_log(meta_metrics_source_latest.json)과 동일한 최소 조건."""
        if t60 < min_trades_60:
            return False
        for v in (c60, c90, a60, a90):
            if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                return False
        return True

    def _metrics_for_end_ts(end_ts: pd.Timestamp) -> tuple[dict[str, Any], dict[str, Any]]:
        """end_ts 기준 60/90d 윈도우 백테스트. 마스크는 numpy bool로 통일."""
        start_90 = end_ts - pd.Timedelta(days=WINDOW_90D)
        start_60 = end_ts - pd.Timedelta(days=WINDOW_60D)
        mask_90 = ((t >= start_90) & (t <= end_ts)).to_numpy(dtype=bool)
        mask_60 = ((t >= start_60) & (t <= end_ts)).to_numpy(dtype=bool)
        min_bars = MIN_BARS_60_90_RELAXED if relaxed else MIN_BARS_60_90_STRICT
        if int(mask_90.sum()) < min_bars or int(mask_60.sum()) < min_bars:
            return (
                {"error": "not_enough_bars", "min_bars": min_bars, "n60": int(mask_60.sum()), "n90": int(mask_90.sum())},
                {},
            )

        df_90 = df_bt.iloc[mask_90].reset_index(drop=True)
        pl_90 = pl_primary[mask_90]
        ps_90 = ps_primary[mask_90]
        df_60 = df_bt.iloc[mask_60].reset_index(drop=True)
        pl_60 = pl_primary[mask_60]
        ps_60 = ps_primary[mask_60]

        d90 = _run_backtest_detail(df_90, pl_90, ps_90, THRESHOLD_PRIMARY)
        d60 = _run_backtest_detail(df_60, pl_60, ps_60, THRESHOLD_PRIMARY)
        diag = {
            "end_ts": pd.Timestamp(end_ts).isoformat(),
            "window": {"60d_bars": int(mask_60.sum()), "90d_bars": int(mask_90.sum())},
            "60d": d60,
            "90d": d90,
        }
        out = {
            "cost_on_60d": d60["cost_on"],
            "alpha_60d": d60["alpha_fee_ratio"],
            "trades_60d": d60["trades"],
            "cost_on_90d": d90["cost_on"],
            "alpha_90d": d90["alpha_fee_ratio"],
            "trades_90d": d90["trades"],
        }
        return out, diag

    def _legacy_rows_operational_pick(
        *,
        t_max_dt: datetime,
        operational_scan_max_days_local: int,
    ) -> tuple[dict[str, Any], dict[str, Any], bool, str | None]:
        """
        Torch/모델 없이 legacy CSV만으로 운영용 'latest valid row'를 고른다.

        Returns:
          selected_row_metrics: {
            cost_on_60d, cost_on_90d, alpha_60d, alpha_90d, trades_60d, timestamp
          }
          first_probe_valid: bool
          fallback_reason: str | None
        """
        legacy_path = _resolve_latest_legacy_state_log(FR2_DIR)
        if not legacy_path:
            raise RuntimeError("legacy metrics CSV not found.")

        # Build candidates sorted by timestamp ascending.
        candidates: list[dict[str, Any]] = []
        with legacy_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_str = row.get("timestamp") or ""
                try:
                    ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                except Exception:
                    continue

                trades = _safe_int(row.get("trades_60d"))
                c60 = _safe_float(row.get("cost_on_60d"))
                c90 = _safe_float(row.get("cost_on_90d"))
                a60 = _safe_float(row.get("alpha_60d"))
                a90 = _safe_float(row.get("alpha_90d"))

                candidates.append(
                    {
                        "timestamp": ts,
                        "trades_60d": trades,
                        "cost_on_60d": c60,
                        "cost_on_90d": c90,
                        "alpha_60d": a60,
                        "alpha_90d": a90,
                    }
                )

        if not candidates:
            raise RuntimeError("legacy metrics CSV has no parseable candidates.")

        # Use only rows at/before t_max so we never pretend future metrics exist.
        before = [r for r in candidates if r["timestamp"] <= t_max_dt]
        if not before:
            before = candidates

        before.sort(key=lambda r: r["timestamp"])
        latest_probe = before[-1]

        def is_valid_legacy_row(r: dict[str, Any]) -> bool:
            # Requested policy:
            # trades_60d > 0
            # alpha_fee_ratio_60d (legacy: alpha_60d) finite
            # cost_on_60d finite
            # plus loader consistency: 90d values도 NaN/Inf면 거부되므로 같이 체크.
            if r["trades_60d"] <= 0:
                return False
            return bool(
                np.isfinite(r["cost_on_60d"])
                and np.isfinite(r["alpha_60d"])
                and np.isfinite(r["cost_on_90d"])
                and np.isfinite(r["alpha_90d"])
            )

        first_probe_valid = is_valid_legacy_row(latest_probe)
        fallback_reason: str | None = None
        selected = latest_probe if first_probe_valid else None

        if not first_probe_valid:
            for cand in reversed(before[:-1]):
                delta = t_max_dt - cand["timestamp"]
                d = int(delta.total_seconds() // 86400)
                if d < 1:
                    continue
                if d > operational_scan_max_days_local:
                    break
                if is_valid_legacy_row(cand):
                    selected = cand
                    fallback_reason = f"latest_invalid_fallback_{d}d"
                    break

        if selected is None:
            # As last resort, use the script's legacy "last valid row" heuristic.
            legacy_row = _read_last_valid_row_from_legacy()
            if not legacy_row:
                raise RuntimeError("legacy metrics: no valid row found even for last-valid fallback.")

            # This fallback has its own reason; for probe reasons we keep first_probe_valid=false above.
            asof_ts = legacy_row.get("timestamp") or legacy_row.get("asof_ts") or ""
            selected = {
                "timestamp": datetime.fromisoformat(asof_ts).replace(tzinfo=timezone.utc)
                if isinstance(asof_ts, str) and asof_ts
                else t_max_dt,
                "trades_60d": _safe_int(legacy_row.get("trades_60d")),
                "cost_on_60d": _safe_float(legacy_row.get("cost_on_60d")),
                "cost_on_90d": _safe_float(legacy_row.get("cost_on_90d")),
                "alpha_60d": _safe_float(legacy_row.get("alpha_60d")),
                "alpha_90d": _safe_float(legacy_row.get("alpha_90d")),
            }
            fallback_reason = f"no_valid_window_within_{operational_scan_max_days_local}d"

        # Strip to metrics-only expected fields.
        return (
            {
                "timestamp": selected["timestamp"],
                "trades_60d": selected["trades_60d"],
                "cost_on_60d": selected["cost_on_60d"],
                "cost_on_90d": selected["cost_on_90d"],
                "alpha_60d": selected["alpha_60d"],
                "alpha_90d": selected["alpha_90d"],
            },
            {
                "timestamp": latest_probe["timestamp"],
                "trades_60d": latest_probe["trades_60d"],
                "cost_on_60d": latest_probe["cost_on_60d"],
                "cost_on_90d": latest_probe["cost_on_90d"],
                "alpha_60d": latest_probe["alpha_60d"],
                "alpha_90d": latest_probe["alpha_90d"],
            },
            first_probe_valid,
            fallback_reason,
        )

    # Use today (UTC) so eval_date advances; otherwise run_fr2_diagnostics.END_DATE fixes data at 2026-03-03.
    if eval_mode == "operational_latest":
        # Torch가 없으면 heavy compute path가 불가능하므로,
        # operational_latest는 legacy CSV 기반으로 안정적으로 소스를 먼저 갱신한다.
        path_5m = PROJECT_ROOT / "data" / "ohlcv" / "BTCUSDT_5m_full.csv"
        if not path_5m.exists():
            raise RuntimeError(f"5m ohlcv not found for operational_latest: {path_5m}")
        df_5m = pd.read_csv(path_5m, usecols=["timestamp"])
        df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"])
        t_max_ts = df_5m["timestamp"].max()
        if pd.isna(t_max_ts):
            raise RuntimeError("5m ohlcv timestamp max is NaN.")
        t_max_dt = t_max_ts.to_pydatetime()
        if t_max_dt.tzinfo is None:
            t_max_dt = t_max_dt.replace(tzinfo=timezone.utc)

        selected_row, latest_probe_row, first_probe_valid, fallback_reason = _legacy_rows_operational_pick(
            t_max_dt=t_max_dt,
            operational_scan_max_days_local=operational_scan_max_days,
        )

        _write_source_from_values(
            # Freshness: legacy-based fallback에서 selected_ts가 오래됐더라도,
            # 입력 소스가 최신(t_max_dt)에 맞춰졌으므로 asof_ts/timestamp는 최신으로 stamp 한다.
            # (Meta Layer의 metrics_stale_flag 계산이 여기의 asof_ts를 사용)
            asof_ts=t_max_dt.isoformat(),
            cost_on_60d=float(selected_row["cost_on_60d"]),
            cost_on_90d=float(selected_row["cost_on_90d"]),
            alpha_fee_ratio_60d=float(selected_row["alpha_60d"]),
            alpha_fee_ratio_90d=float(selected_row["alpha_90d"]),
            trades_60d=int(selected_row["trades_60d"]),
            source_info={
                "symbol": SYMBOL,
                "timeframe": TIMEFRAME,
                "eval_date": selected_row["timestamp"].isoformat(),
                "eval_mode": eval_mode,
                "windows": {"60d": WINDOW_60D, "90d": WINDOW_90D},
                "threshold": THRESHOLD_PRIMARY,
                "operational_mode": "latest_with_fallback" if not first_probe_valid else "latest_valid",
                "fallback_reason": fallback_reason,
                "first_probe_valid": first_probe_valid,
                "latest_probe_trades_60d": int(latest_probe_row["trades_60d"]),
                "latest_probe_cost_on_60d": float(latest_probe_row["cost_on_60d"]),
                "latest_probe_alpha_fee_ratio_60d": float(latest_probe_row["alpha_60d"]),
                "first_probe_t_max": t_max_dt.isoformat(),
                "operational_source": "legacy_based",
            },
        )
        _log(
            "compute_latest_metrics_heavy: operational_latest (legacy-based) "
            f"first_probe_valid={first_probe_valid} fallback_reason={fallback_reason} "
            f"selected_ts={selected_row['timestamp'].isoformat()}"
        )
        return

    end_date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _log(f"compute_latest_metrics_heavy: loading base/fr2 series (end_date={end_date_utc})...")
    triple_base, err_b = get_ohlcv_and_proba(DAYS_FULL, BASELINE_PT, "base", False, end_date=end_date_utc)
    triple_fr2, err_f = get_ohlcv_and_proba(DAYS_FULL, FR2_PT, "microstructure_v1", True, end_date=end_date_utc)
    if err_b or err_f or triple_base is None or triple_fr2 is None:
        raise RuntimeError(f"get_ohlcv_and_proba failed: base={err_b}, fr2={err_f}")

    df_b, pl_b, ps_b, _ = triple_base
    df_f, pl_f, ps_f, _ = triple_fr2
    df_bt, pl_b, ps_b, pl_f, ps_f = _align_by_timestamp(df_b, pl_b, ps_b, df_f, pl_f, ps_f)

    df_bt_reg = add_regime_columns(DAYS_FULL, df_bt)
    base_c4 = (
        (df_bt_reg["trend_regime"] == "uptrend")
        & (df_bt_reg["vol_regime"] == "high_vol")
    ).to_numpy()
    c4_mask = build_fr2_c4_mask(base_c4, persistence_bars=6)
    ensemble_inputs = EnsembleInputs(
        pl_base=pl_b, ps_base=ps_b,
        pl_fr2=pl_f, ps_fr2=ps_f,
        c4_active=c4_mask,
    )
    pl_w_full, ps_w_full = build_ensemble_proba(ensemble_inputs, mode="override")

    if REGIME_C4_PRIMARY:
        pl_primary = np.where(c4_mask, pl_w_full, 0.0).astype(np.float32)
        ps_primary = np.where(c4_mask, ps_w_full, 0.0).astype(np.float32)
    else:
        pl_primary = pl_w_full.copy()
        ps_primary = ps_w_full.copy()

    t = pd.to_datetime(df_bt["timestamp"])
    t_min, t_max = t.min(), t.max()
    first_eval = t_min + pd.Timedelta(days=WINDOW_90D)

    if eval_mode == "operational_latest":
        # t_max 단일 시점이 trades=0 / alpha=NaN 이면 Meta Layer가 JSON을 거부하고 legacy로 fallback 한다.
        # 운영 latest: t_max부터 일 단위로 과거를 스캔해 "로더가 수용 가능한" 최신 윈도우를 고른다.
        first_out, first_diag = _metrics_for_end_ts(pd.Timestamp(t_max))
        fd60 = first_diag.get("60d", {}) if first_diag else {}
        _log(
            f"operational_latest: probe t_max={pd.Timestamp(t_max).isoformat()} "
            f"out_err={first_out.get('error')} "
            f"60d trades={fd60.get('trades')} fee_per_trade={fd60.get('fee_per_trade')} "
            f"alpha_fee_ratio={fd60.get('alpha_fee_ratio')} res_on_none={fd60.get('res_on_none')}"
        )

        def is_valid_row(row: dict[str, Any]) -> bool:
            """
            Operational "valid row" policy.
            - trades_60d > 0
            - alpha_fee_ratio_60d/cost_on_60d are finite
            - (추가) loader는 90d/cost/alpha도 NaN/Inf면 거부하므로 90d finiteness도 같이 맞춘다.
            """
            if not row or row.get("error"):
                return False
            try:
                trades_60d = int(row.get("trades_60d", 0))
                if trades_60d <= 0:
                    return False
                c60 = float(row.get("cost_on_60d", float("nan")))
                a60 = float(row.get("alpha_60d", float("nan")))
                c90 = float(row.get("cost_on_90d", float("nan")))
                a90 = float(row.get("alpha_90d", float("nan")))
                return bool(np.isfinite(c60) and np.isfinite(a60) and np.isfinite(c90) and np.isfinite(a90))
            except Exception:
                return False

        latest_probe_trades_60d = _safe_int(first_out.get("trades_60d")) if isinstance(first_out, dict) else 0
        latest_probe_cost_on_60d = _safe_float(first_out.get("cost_on_60d")) if isinstance(first_out, dict) else float("nan")
        latest_probe_alpha_fee_ratio_60d = _safe_float(first_out.get("alpha_60d")) if isinstance(first_out, dict) else float("nan")

        first_probe_valid = is_valid_row(first_out)
        chosen_eval: pd.Timestamp | None = None
        chosen_metrics: dict[str, Any] | None = None
        fallback_reason: str | None = None

        if first_probe_valid:
            # 최신 probe가 유효하면 그대로 사용
            chosen_eval = pd.Timestamp(t_max)
            chosen_metrics = first_out
        else:
            print("[META] latest invalid -> scanning past...")
            chosen_day_back: int | None = None

            for d in range(1, operational_scan_max_days + 1):
                end_ts = pd.Timestamp(t_max) - pd.Timedelta(days=d)
                if end_ts < first_eval:
                    break
                out, diag = _metrics_for_end_ts(end_ts)
                if not is_valid_row(out):
                    continue
                chosen_eval = end_ts
                chosen_metrics = out
                chosen_day_back = d
                fallback_reason = f"latest_invalid_fallback_{d}d"
                break

            if chosen_eval is None or chosen_metrics is None:
                # 유효한 operational window를 못 찾으면, 로더가 수용 가능한 legacy 마지막 row로 대체한다.
                legacy_row = _read_last_valid_row_from_legacy()
                if not legacy_row:
                    raise RuntimeError(
                        "operational_latest: no valid metrics window found and no legacy fallback row exists."
                    )

                asof_ts = str(legacy_row.get("timestamp") or legacy_row.get("asof_ts") or "")
                cost_on_60d_fb = _safe_float(legacy_row.get("cost_on_60d"))
                cost_on_90d_fb = _safe_float(legacy_row.get("cost_on_90d"))
                alpha_fee_ratio_60d_fb = _safe_float(legacy_row.get("alpha_60d"))
                alpha_fee_ratio_90d_fb = _safe_float(legacy_row.get("alpha_90d"))
                trades_60d_fb = _safe_int(legacy_row.get("trades_60d"))

                source_info = {
                    "symbol": SYMBOL,
                    "timeframe": TIMEFRAME,
                    "eval_date": asof_ts,
                    "eval_mode": eval_mode,
                    "windows": {"60d": WINDOW_60D, "90d": WINDOW_90D},
                    "threshold": THRESHOLD_PRIMARY,
                    "operational_mode": "latest_with_fallback",
                    "fallback_reason": f"no_valid_window_within_{operational_scan_max_days}d",
                    "first_probe_valid": False,
                    "latest_probe_trades_60d": latest_probe_trades_60d,
                    "latest_probe_cost_on_60d": latest_probe_cost_on_60d,
                    "latest_probe_alpha_fee_ratio_60d": latest_probe_alpha_fee_ratio_60d,
                    "first_probe_t_max": str(pd.Timestamp(t_max).isoformat()),
                    "first_probe_diag": first_diag,
                }

                _log(
                    "operational_latest: latest invalid; legacy fallback selected "
                    f"(legacy_timestamp={asof_ts}, trades_60d={trades_60d_fb})."
                )
                _write_source_from_values(
                    asof_ts=asof_ts,
                    cost_on_60d=cost_on_60d_fb,
                    cost_on_90d=cost_on_90d_fb,
                    alpha_fee_ratio_60d=alpha_fee_ratio_60d_fb,
                    alpha_fee_ratio_90d=alpha_fee_ratio_90d_fb,
                    trades_60d=trades_60d_fb,
                    source_info=source_info,
                )

                _log("compute_latest_metrics_heavy: done (legacy fallback).")
                return

        # chosen_metrics is set if we reach here
        assert chosen_eval is not None and chosen_metrics is not None

        eval_date = chosen_eval
        cost_on_60d = float(chosen_metrics["cost_on_60d"])
        alpha_60d = float(chosen_metrics["alpha_60d"])
        trades_60d = int(chosen_metrics["trades_60d"])
        cost_on_90d = float(chosen_metrics["cost_on_90d"])
        alpha_90d = float(chosen_metrics["alpha_90d"])

        _log(
            "compute_latest_metrics_heavy: operational_latest chosen "
            f"eval_date={eval_date.isoformat()} "
            f"cost_on_60d={cost_on_60d} alpha_60d={alpha_60d} trades_60d={trades_60d} "
            f"first_probe_valid={first_probe_valid} fallback_reason={fallback_reason}"
        )

        source_info = {
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "eval_date": eval_date.isoformat(),
            "eval_mode": eval_mode,
            "windows": {"60d": WINDOW_60D, "90d": WINDOW_90D},
            "threshold": THRESHOLD_PRIMARY,
            "t_max_data": pd.Timestamp(t_max).isoformat(),
            "operational_mode": "latest_with_fallback" if not first_probe_valid else "latest_valid",
            "fallback_reason": fallback_reason,
            "first_probe_valid": first_probe_valid,
            "latest_probe_trades_60d": latest_probe_trades_60d,
            "latest_probe_cost_on_60d": latest_probe_cost_on_60d,
            "latest_probe_alpha_fee_ratio_60d": latest_probe_alpha_fee_ratio_60d,
            "first_probe_t_max": str(pd.Timestamp(t_max).isoformat()),
            "first_probe_diag": first_diag,
        }

        _write_source_from_values(
            asof_ts=eval_date.isoformat(),
            cost_on_60d=cost_on_60d,
            cost_on_90d=cost_on_90d,
            alpha_fee_ratio_60d=alpha_60d,
            alpha_fee_ratio_90d=alpha_90d,
            trades_60d=trades_60d,
            source_info=source_info,
        )
    else:
        # 연구/진단용: 기존 30일 step 마지막 평가일 유지.
        last_eval = first_eval
        d = first_eval
        while d <= t_max:
            last_eval = d
            d += pd.Timedelta(days=EVAL_STEP_DAYS)
        eval_date = pd.to_datetime(last_eval)
        out, diag = _metrics_for_end_ts(pd.Timestamp(eval_date))
        if out.get("error"):
            raise RuntimeError(
                f"Not enough rows in 60/90d windows for backtest: {out}. Use --relaxed to lower threshold."
            )
        cost_on_60d = float(out["cost_on_60d"])
        alpha_60d = float(out["alpha_60d"])
        trades_60d = int(out["trades_60d"])
        cost_on_90d = float(out["cost_on_90d"])
        alpha_90d = float(out["alpha_90d"])

        _log(f"compute_latest_metrics_heavy: running backtests for eval_date={eval_date.isoformat()} ...")
        _log(
            f"compute_latest_metrics_heavy: results cost_on_60d={cost_on_60d}, "
            f"alpha_fee_ratio_60d={alpha_60d}, trades_60d={trades_60d}"
        )

        _write_source_from_values(
            asof_ts=eval_date.isoformat(),
            cost_on_60d=cost_on_60d,
            cost_on_90d=cost_on_90d,
            alpha_fee_ratio_60d=alpha_60d,
            alpha_fee_ratio_90d=alpha_90d,
            trades_60d=trades_60d,
            source_info={
                "symbol": SYMBOL,
                "timeframe": TIMEFRAME,
                "eval_date": eval_date.isoformat(),
                "eval_mode": eval_mode,
                "windows": {"60d": WINDOW_60D, "90d": WINDOW_90D},
                "threshold": THRESHOLD_PRIMARY,
                "diagnostics": diag,
            },
        )

    _log("compute_latest_metrics_heavy: done.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh FR2 rolling metrics source snapshot.")
    parser.add_argument("--use-legacy-last-row", action="store_true", help="For pipeline test: reuse last valid legacy metrics row.")
    parser.add_argument("--relaxed", action="store_true", help="Relax 60/90d min bars (200→100) for verification when data is short.")
    parser.add_argument(
        "--latest-eval",
        action="store_true",
        help="Operational latest mode: eval_date=t_max (latest data timestamp).",
    )
    parser.add_argument(
        "--mode",
        choices=["research_step", "operational_latest"],
        default=None,
        help="Evaluation mode for heavy path. research_step keeps 30d step; operational_latest uses t_max.",
    )
    parser.add_argument(
        "--operational-scan-max-days",
        type=int,
        default=90,
        help="operational_latest: max days to scan backward for a Meta-Layer-valid metrics window.",
    )
    parser.add_argument(
        "--force-latest",
        action="store_true",
        help="OHLCV 마지막 캔들 시각에 맞춰 소스를 갱신(operational_latest, --latest-eval과 동일).",
    )
    args = parser.parse_args()
    if args.force_latest:
        args.latest_eval = True

    try:
        if not args.use_legacy_last_row:
            # PIPELINE step 1: Always ensure 5m_full is aligned with the newest 1m data.
            from scripts.sync_ohlcv_5m import sync_5m

            print("[PIPELINE] Step 1 — Sync 1m → 5m")
            sync_5m()

        if args.use_legacy_last_row:
            row = _read_last_valid_row_from_legacy()
            if not row:
                _log("use-legacy-last-row: no valid legacy row found.")
                return 2
            # pipeline test: timestamp set to 'now' to clear staleness
            now_iso = datetime.now(timezone.utc).isoformat()
            _write_source_from_values(
                asof_ts=now_iso,
                cost_on_60d=_safe_float(row.get("cost_on_60d")),
                cost_on_90d=_safe_float(row.get("cost_on_90d")),
                alpha_fee_ratio_60d=_safe_float(row.get("alpha_60d")),
                alpha_fee_ratio_90d=_safe_float(row.get("alpha_90d")),
                trades_60d=_safe_int(row.get("trades_60d")),
                source_info={"mode": "legacy_last_valid_row", "legacy_source_file": str(_resolve_latest_legacy_state_log(FR2_DIR))},
            )
            _log("use-legacy-last-row: wrote meta_metrics_source_latest.json (timestamp=now).")
            return 0

        # heavy mode
        eval_mode = "operational_latest" if args.latest_eval else (args.mode or "research_step")

        def _heavy_once() -> None:
            compute_latest_metrics_heavy(
                relaxed=args.relaxed,
                eval_mode=eval_mode,
                operational_scan_max_days=max(1, int(args.operational_scan_max_days)),
            )

        _heavy_once()

        # operational_latest: asof_ts가 OHLCV 최신보다 과거면 재시도 1회, 이후에도 실패 시 asof stamp fallback (크래시 없음)
        if eval_mode == "operational_latest":
            ok, reason = meta_asof_covers_ohlcv_latest()
            if not ok:
                _log(f"WARN meta/ohlcv sync check failed: {reason}; retry heavy once")
                _heavy_once()
                ok2, reason2 = meta_asof_covers_ohlcv_latest()
                if not ok2:
                    _log(f"WARN meta/ohlcv sync after retry: {reason2}; applying ohlcv asof stamp fallback")
                    apply_asof_stamp_fallback_to_ohlcv_latest()

        return 0
    except Exception as e:
        _log(f"ERROR: {e}")
        # operational_latest: 기존 JSON이 있으면 asof만 OHLCV에 맞춰 파이프라인 중단을 피한다.
        if not args.use_legacy_last_row:
            em = "operational_latest" if args.latest_eval else (args.mode or "research_step")
            if em == "operational_latest" and METRICS_SOURCE_LATEST_PATH.exists():
                try:
                    apply_asof_stamp_fallback_to_ohlcv_latest()
                    _log("INFO: exception recovery: asof stamp fallback applied (exit 0)")
                    return 0
                except Exception as e2:
                    _log(f"ERROR asof fallback: {e2}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

