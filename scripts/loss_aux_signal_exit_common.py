"""
후보1(signal_exit_th loss aux) 백테스트 공통: 데이터 로드 + run_backtest_7d 래퍼 + 지표.
threshold/모델 변경 없음 — kwargs만 전달.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def to_utc_ts(x: Any) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def cost_on(res: dict[str, Any]) -> float:
    ec = res.get("equity_curve") or []
    if isinstance(ec, list) and len(ec) >= 1:
        return float(ec[-1] - 1.0)
    return float(res.get("total_return", 0.0) or 0.0)


def deduped_trades(res: dict[str, Any]) -> list[Any]:
    from src.backtest.engine import dedupe_trades_round_trips

    trades = list(res.get("trades") or [])
    return dedupe_trades_round_trips(trades)


def mean_profit_roundtrip(res: dict[str, Any]) -> float | None:
    trades = deduped_trades(res)
    if not trades:
        return None
    vals = [float(t["profit"]) for t in trades if t.get("profit") is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def max_loss_trade(res: dict[str, Any]) -> float | None:
    trades = deduped_trades(res)
    if not trades:
        return None
    vals = [float(t["profit"]) for t in trades if t.get("profit") is not None]
    if not vals:
        return None
    return float(min(vals))


def run_loss_aux_window(
    *,
    df_bt: pd.DataFrame,
    pl_primary: np.ndarray,
    ps_primary: np.ndarray,
    t_end: pd.Timestamp,
    window_days: int,
    threshold: float,
    emit_trade_log: bool = False,
    signal_exit_th_loss_aux_delta_p: float | None = None,
    signal_exit_th_loss_aux_mae_cut: float | None = None,
    signal_exit_th_loss_aux_gate_delta_unreal_positive: bool = False,
    signal_exit_th_loss_aux_gate_mfe_min: float | None = None,
    entry_min_unique_round_trips: int | None = None,
    entry_urt_state_log_path: Path | str | None = None,
    entry_urt_state_log_snapshot_override: int | None = None,
) -> dict[str, Any]:
    from scripts.run_tcn_label_sweep_v2 import run_backtest_7d

    window_start = t_end - pd.Timedelta(days=window_days)
    t = pd.to_datetime(df_bt["timestamp"], utc=True)
    mask = (t >= window_start) & (t <= t_end)
    df_w = df_bt.loc[mask].reset_index(drop=True)
    pl_w = np.asarray(pl_primary, dtype=np.float32)[mask.to_numpy()]
    ps_w = np.asarray(ps_primary, dtype=np.float32)[mask.to_numpy()]

    COMMISSION = 0.0009
    SLIPPAGE = 0.0001
    MAX_ENTROPY = 1.30
    MIN_HOLD = 24
    COOLDOWN = 24
    TIME_STOP_BARS = 72
    EARLY_EXIT_BAD_K = 8

    res, err = run_backtest_7d(
        "BTCUSDT",
        "5m",
        df_w,
        pl_w,
        ps_w,
        commission_rate=COMMISSION,
        slippage_rate=SLIPPAGE,
        min_max_proba=float(threshold),
        max_entropy=MAX_ENTROPY,
        decision_mode="argmax",
        min_hold=MIN_HOLD,
        cooldown=COOLDOWN,
        time_stop_enabled=True,
        time_stop_bars=TIME_STOP_BARS,
        early_exit_enabled=True,
        early_exit_bad_k=EARLY_EXIT_BAD_K,
        emit_trade_log=emit_trade_log,
        signal_exit_th_trailing_window_bars=1,
        signal_exit_th_delta_from_entry=None,
        signal_exit_th_loss_aux_delta_p=signal_exit_th_loss_aux_delta_p,
        signal_exit_th_loss_aux_mae_cut=signal_exit_th_loss_aux_mae_cut,
        signal_exit_th_loss_aux_short_only=False,
        signal_exit_th_loss_aux_min_bars=None,
        signal_exit_th_loss_aux_gate_delta_unreal_positive=signal_exit_th_loss_aux_gate_delta_unreal_positive,
        signal_exit_th_loss_aux_gate_mfe_min=signal_exit_th_loss_aux_gate_mfe_min,
        entry_min_unique_round_trips=entry_min_unique_round_trips,
        entry_urt_state_log_path=(
            Path(entry_urt_state_log_path) if entry_urt_state_log_path is not None else None
        ),
        entry_urt_state_log_snapshot_override=entry_urt_state_log_snapshot_override,
    )
    if err:
        raise RuntimeError(f"run_backtest_7d failed: {err}")
    if res is None:
        res = {}
    return res


def metrics_row_step2(name: str, res: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    ut = res.get("unique_round_trips")
    wr = res.get("win_rate")
    row: dict[str, Any] = {
        "label": name,
        "unique_round_trips": int(ut) if ut is not None else None,
        "win_rate": float(wr) if wr is not None else None,
        "mean_profit_roundtrip": mean_profit_roundtrip(res),
        "cost_on": cost_on(res),
        "total_return": float(res.get("total_return", 0.0) or 0.0),
        "max_loss_trade": max_loss_trade(res),
        "signal_exit_th_total_profit": float(res.get("signal_exit_th_total_profit", 0.0) or 0.0),
    }
    if extra:
        row.update(extra)
    return row


def load_fr2_override_ensemble(
    days_full: int,
    t_max: pd.Timestamp,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Timestamp, pd.Timestamp]:
    """FR2 C4 override 앙상블 proba. df_bt 행 타임스탬프 범위 [ts_min, ts_max]."""
    from scripts.run_fr2_diagnostics import MODELS_DIR, get_ohlcv_and_proba
    from scripts.run_fr2_regime_conditioning import add_regime_columns
    from src.strategies.ensemble_strategy import EnsembleInputs, build_ensemble_proba, build_fr2_c4_mask

    end_date_utc_str = t_max.strftime("%Y-%m-%d")
    base_pt = MODELS_DIR / "tcn_h15_t0p004.pt"
    fr2_pt = MODELS_DIR / "tcn_h15_micro_v1.pt"

    triple_base, err_b = get_ohlcv_and_proba(days_full, base_pt, "base", False, end_date=end_date_utc_str)
    if err_b or triple_base is None:
        raise RuntimeError(f"get_ohlcv_and_proba(base) failed: {err_b}")
    df_b, pl_b, ps_b, _ = triple_base
    triple_fr2, err_f = get_ohlcv_and_proba(days_full, fr2_pt, "microstructure_v1", True, end_date=end_date_utc_str)
    if err_f or triple_fr2 is None:
        raise RuntimeError(f"get_ohlcv_and_proba(fr2) failed: {err_f}")
    df_f, pl_f, ps_f, _ = triple_fr2

    d1 = df_b.copy()
    d2 = df_f.copy()
    d1["timestamp"] = pd.to_datetime(d1["timestamp"])
    d2["timestamp"] = pd.to_datetime(d2["timestamp"])
    d1["pl_base"] = np.asarray(pl_b, dtype=np.float32)
    d1["ps_base"] = np.asarray(ps_b, dtype=np.float32)
    d2["pl_fr2"] = np.asarray(pl_f, dtype=np.float32)
    d2["ps_fr2"] = np.asarray(ps_f, dtype=np.float32)
    joined = d1.merge(d2[["timestamp", "pl_fr2", "ps_fr2"]], on="timestamp", how="inner", validate="one_to_one")
    joined = joined.sort_values("timestamp").reset_index(drop=True)
    df_bt = joined[["timestamp", "close", "high", "low"]].copy()

    pl_base_arr = joined["pl_base"].to_numpy(dtype=np.float32)
    ps_base_arr = joined["ps_base"].to_numpy(dtype=np.float32)
    pl_fr2_arr = joined["pl_fr2"].to_numpy(dtype=np.float32)
    ps_fr2_arr = joined["ps_fr2"].to_numpy(dtype=np.float32)

    df_bt_reg = add_regime_columns(days_full, df_bt.copy())
    base_c4 = ((df_bt_reg["trend_regime"] == "uptrend") & (df_bt_reg["vol_regime"] == "high_vol")).to_numpy()
    c4_mask = build_fr2_c4_mask(base_c4, persistence_bars=6)
    inputs = EnsembleInputs(
        pl_base=pl_base_arr,
        ps_base=ps_base_arr,
        pl_fr2=pl_fr2_arr,
        ps_fr2=ps_fr2_arr,
        c4_active=c4_mask,
    )
    pl_primary, ps_primary = build_ensemble_proba(inputs, mode="override")

    ts = pd.to_datetime(df_bt["timestamp"], utc=True)
    ts_min = pd.Timestamp(ts.min())
    ts_max = pd.Timestamp(ts.max())
    return df_bt, pl_primary, ps_primary, ts_min, ts_max
