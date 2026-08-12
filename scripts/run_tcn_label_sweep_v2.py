#!/usr/bin/env python3
"""
TCN 라벨 threshold 상향 스윕 v2: 12조합 학습 → 7일 신호품질 + 비용 on/off 백테스트 → 상위 선정 → (옵션) 필터 quick sweep.

사용법:
  source .venv/bin/activate
  python -m scripts.run_tcn_label_sweep_v2 --days 7 --epochs-first 3 --no-second-run --run-filter-sweep
  python -m scripts.run_tcn_label_sweep_v2 --days 7 --epochs-first 3 --epochs-second 10 --run-filter-sweep
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DIAG = PROJECT_ROOT / "data" / "diagnostics"
MODELS_DIR = DIAG / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 기본 12조합 = --horizons "5,15,30" × --thresholds "0.001,0.002,0.003,0.004".
# results가 4개인 경우: --horizons "5,15" --thresholds "0.001,0.002" (2×2) 또는
#   --eval-only 시 --sweep-ids로 4개 지정 또는 data/diagnostics/models/ 내 tcn_*.pt가 4개일 때.
HORIZONS = [5, 15, 30]
THRESHOLDS = [0.001, 0.002, 0.003, 0.004]


def _sweep_id(horizon: int, thr: float) -> str:
    return f"h{horizon}_t{thr:.3f}".replace(".", "p")


def _parse_sweep_id(sid: str) -> tuple[int, float] | None:
    """e.g. h5_t0p001 -> (5, 0.001); h30_t0p004 -> (30, 0.004)."""
    parts = sid.strip().split("_")
    if len(parts) != 2 or not parts[0].startswith("h") or not parts[1].startswith("t"):
        return None
    try:
        h = int(parts[0][1:])
        t_str = parts[1][1:].replace("p", ".")
        thr = float(t_str)
        return (h, thr)
    except (ValueError, IndexError):
        return None


def run_train(
    sweep_id: str,
    horizon: int,
    pos_threshold: float,
    neg_threshold: float,
    epochs: int,
) -> bool:
    out_pt = MODELS_DIR / f"tcn_{sweep_id}.pt"
    cmd = [
        sys.executable,
        "-m",
        "src.dl.train.train_tcn",
        "--epochs",
        str(epochs),
        "--horizon-bars",
        str(horizon),
        "--pos-threshold",
        str(pos_threshold),
        "--neg-threshold",
        str(neg_threshold),
        "--out-model",
        str(out_pt),
        "--seed",
        "42",
    ]
    print(f"[RUN] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=3600)
    return r.returncode == 0


def load_metrics(sweep_id: str) -> dict | None:
    p = MODELS_DIR / f"tcn_{sweep_id}_metrics.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def get_7d_ohlcv_and_proba(
    model_path: Path,
    symbol: str,
    timeframe: str,
    days: int,
    log_prefix: str = "",
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[tuple | None, str | None]:
    """
    Load OHLCV, filter by last `days` days (or by start_date/end_date when both given), build features, infer proba.
    Returns ((df_bt, pl, ps), None) on success or (None, error_reason) on failure.
    When start_date and end_date are both provided, they override the days-based cutoff (for ops pinning).
    """
    from src.services.ohlcv_service import load_ohlcv_df
    from src.indicators.basic import add_basic_indicators
    from src.ml.features import build_feature_frame
    from src.dl.tcn_model import TCNSignalModel
    import numpy as np

    def _log(msg: str) -> None:
        print(f"  [EVAL] {log_prefix}{msg}", flush=True)
    if not model_path.exists():
        return (None, "model_path does not exist")
    _log("loading OHLCV ...")
    df = load_ohlcv_df(timeframe=timeframe, symbol=symbol)
    df = add_basic_indicators(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if start_date is not None and end_date is not None:
        start_ts = pd.Timestamp(start_date).tz_localize("UTC") if pd.Timestamp(start_date).tz is None else pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date).tz_localize("UTC") if pd.Timestamp(end_date).tz is None else pd.Timestamp(end_date)
        end_ts = end_ts + pd.Timedelta(days=1)  # inclusive end day
        if df["timestamp"].dt.tz is None:
            start_ts = start_ts.tz_localize(None)
            end_ts = end_ts.tz_localize(None)
        df = df.loc[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()
    else:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        if df["timestamp"].dt.tz is None:
            cutoff = cutoff.tz_localize(None)
        df = df.loc[df["timestamp"] >= cutoff].copy()
    if len(df) < 100:
        return (None, f"after 7d slice rows={len(df)} < 100")
    _log("building features ...")
    features = build_feature_frame(df, symbol=symbol, timeframe=timeframe, use_events=True)
    features = features.dropna()
    if len(features) < 60:
        return (None, f"after dropna features rows={len(features)} < 60")
    _log("loading model + running inference ...")
    model = TCNSignalModel(model_path=model_path, use_events=True)
    if not model.is_loaded():
        return (None, "TCNSignalModel failed to load")
    pl_arr, ps_arr = model.predict_proba_batch(
        features=features, symbol=symbol, timeframe=timeframe, batch_size=512
    )
    pl = np.asarray(pl_arr, dtype=np.float32)
    ps = np.asarray(ps_arr, dtype=np.float32)
    if "close" in features.columns and "high" in features.columns and "low" in features.columns:
        df_bt = features[["close", "high", "low"]].copy()
        if isinstance(features.index, pd.DatetimeIndex):
            df_bt["timestamp"] = features.index
        elif "timestamp" in features.columns:
            df_bt["timestamp"] = features["timestamp"].values
    else:
        df_indexed = df.set_index("timestamp")
        df_bt = df_indexed.reindex(features.index)[["close", "high", "low"]].dropna(how="any")
        if len(df_bt) == 0:
            return (None, "reindex df_bt empty")
        common = df_bt.index
        features = features.loc[common]
        pos = [list(features.index).index(i) for i in common]
        pl = np.asarray([pl[i] for i in pos], dtype=np.float32)
        ps = np.asarray([ps[i] for i in pos], dtype=np.float32)
        df_bt = df_bt.reset_index()
        if df_bt.columns[0] != "timestamp":
            df_bt = df_bt.rename(columns={df_bt.columns[0]: "timestamp"})
    # Ensure timestamp column for backtest (first branch already has it)
    if "timestamp" not in df_bt.columns and isinstance(features.index, pd.DatetimeIndex):
        df_bt = df_bt.reset_index()
        if len(df_bt.columns) > 0 and df_bt.columns[0] != "timestamp":
            df_bt = df_bt.rename(columns={df_bt.columns[0]: "timestamp"})

    # Align df_bt and pl/ps by timestamp when lengths differ (TCN returns N - window_size predictions).
    # Use timestamp dedupe + one-to-one merge to prevent row duplication (joined > len(proba)).
    if len(df_bt) != len(pl):
        try:
            window_size = int(getattr(model, "window_size", 60))
            proba_len = len(pl)

            # (1) df_bt timestamp 정규화 + 중복 제거 (인덱스/컬럼 이중 사용 방지)
            df_bt = df_bt.copy()
            if "timestamp" not in df_bt.columns and isinstance(df_bt.index, pd.DatetimeIndex):
                df_bt["timestamp"] = df_bt.index
            df_bt = df_bt.reset_index(drop=True)
            df_bt["timestamp"] = pd.to_datetime(df_bt["timestamp"])
            df_bt = df_bt.sort_values("timestamp")
            dup_bt = int(df_bt["timestamp"].duplicated().sum())
            df_bt = df_bt.drop_duplicates(subset=["timestamp"], keep="last")

            # (2) features: align_ts는 모델 출력 위치와 1:1 대응되므로 원본 features index 기준으로만 생성 (dedupe 하지 않음)
            if "timestamp" in features.columns:
                feat_index = pd.DatetimeIndex(pd.to_datetime(features["timestamp"]))
            else:
                feat_index = pd.DatetimeIndex(pd.to_datetime(features.index))
            dup_feat = int(feat_index.duplicated().sum()) if hasattr(feat_index, "duplicated") else 0

            # (3) align_ts = features index [window_size : window_size + proba_len]
            align_ts = feat_index[window_size : window_size + proba_len]
            if len(align_ts) != proba_len:
                return (None, f"[ALIGN] len(align_ts)={len(align_ts)} != proba_len={proba_len}")

            # (4) proba_df with timestamp + dedupe
            proba_df = pd.DataFrame({
                "timestamp": pd.to_datetime(align_ts.values),
                "pl": pl,
                "ps": ps,
            })
            proba_df = proba_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

            # (5) df_bt_aligned: rows with timestamp in proba_df only, deduped
            df_bt_aligned = df_bt[df_bt["timestamp"].isin(proba_df["timestamp"])].copy()
            df_bt_aligned = df_bt_aligned.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

            ts_bt_min, ts_bt_max = (df_bt["timestamp"].min(), df_bt["timestamp"].max()) if len(df_bt) else (None, None)
            proba_ts_min = proba_df["timestamp"].min() if len(proba_df) else None
            proba_ts_max = proba_df["timestamp"].max() if len(proba_df) else None
            _log(
                f"[ALIGN] before: len(df_bt)={len(df_bt)}, len(proba)={proba_len}, "
                f"dup_ts_bt={dup_bt}, dup_ts_feat={dup_feat}, "
                f"df_bt ts_range=({ts_bt_min}, {ts_bt_max}), proba ts_range=({proba_ts_min}, {proba_ts_max})"
            )

            # (6) joined = merge(validate="one_to_one")
            joined = df_bt_aligned.merge(
                proba_df,
                on="timestamp",
                how="inner",
                validate="one_to_one",
            )

            # (7) 최종 길이 검증(증식 방지)
            if len(joined) != len(proba_df):
                raise RuntimeError(
                    f"[ALIGN] len(joined)={len(joined)} != len(proba_df)={len(proba_df)}; "
                    f"dup_bt={dup_bt}, dup_feat={dup_feat}, "
                    f"df_bt_ts_range=({ts_bt_min},{ts_bt_max}), "
                    f"proba_ts_range=({proba_ts_min},{proba_ts_max}), "
                    f"joined_ts_range=({joined['timestamp'].min()},{joined['timestamp'].max()})"
                )

            _log(
                f"[ALIGN] after: len(df_bt_aligned)={len(df_bt_aligned)}, len(proba_df)={len(proba_df)}, "
                f"len(joined)={len(joined)}, joined ts_range=({joined['timestamp'].min()}, {joined['timestamp'].max()})"
            )

            df_bt = joined[["timestamp", "close", "high", "low"]].copy()
            pl = joined["pl"].values.astype(np.float32)
            ps = joined["ps"].values.astype(np.float32)

            if len(df_bt) < 50:
                return (None, f"after align len(joined)={len(df_bt)} too small (min 50)")
        except Exception as e:
            import traceback
            return (None, f"{e}\n{traceback.format_exc()}")

    if len(df_bt) != len(pl):
        return (None, f"len(df_bt)={len(df_bt)} != len(pl)={len(pl)}")
    return ((df_bt, pl, ps), None)


def run_backtest_7d(
    symbol: str,
    timeframe: str,
    df_bt,
    proba_long_arr,
    proba_short_arr,
    commission_rate: float,
    slippage_rate: float,
    min_max_proba: float | None = None,
    max_entropy: float | None = None,
    min_proba_gap: float = 0.0,
    min_directional_gap: float | None = None,
    max_flat_entry_proba: float | None = None,
    # D12/D13: decision mode
    decision_mode: str = "argmax",
    min_directional_edge: float | None = None,
    require_direction_gt_flat: bool = True,
    min_side_flat_margin: float | None = None,
    min_confidence: float | None = None,
    use_legacy_entry_gates_with_directional: bool = True,
    min_hold: int | None = None,
    cooldown: int | None = None,
    regime_filter_enabled: bool = False,
    regime_ema_span: int = 200,
    regime_rule: str = "ema_only",
    regime_slope_lookback: int = 48,
    vol_window: int = 48,
    vol_threshold: float = 0.0010,
    breakout_lookback: int = 96,
    breakout_mode: str = "ema_high",
    slope_threshold: float = 1e-5,
    q_window: int = 8640,
    q: float = 0.90,
    p_floor: float = 0.55,
    position_scaling_enabled: bool = False,
    position_scaling_mode: str = "linear",
    position_p_floor: float = 0.55,
    position_p_full: float = 0.65,
    position_size_min: float = 0.25,
    position_size_max: float = 1.0,
    position_p_mid: float = 0.60,
    position_k: float = 25.0,
    early_exit_enabled: bool = False,
    early_exit_lookback: int = 12,
    early_exit_p_floor: float = 0.55,
    early_exit_bad_k: int = 8,
    entry_flat_gate_enabled: bool = False,
    max_flat_proba: float = 0.45,
    flat_exit_aware_enabled: bool = False,
    flat_exit_threshold: float = 0.30,
    flat_exit_badk_delta: int = 2,
    time_stop_enabled: bool = False,
    time_stop_bars: int = 96,
    partial_tp_enabled: bool = False,
    partial_tp_threshold: float = 0.0025,
    partial_tp_ratio: float = 0.5,
    break_even_stop_enabled: bool = False,
    be_threshold: float = 0.0,
    emit_trade_log: bool = False,
    signal_exit_th_trailing_window_bars: int = 1,
    signal_exit_th_delta_from_entry: float | None = None,
    signal_exit_th_loss_aux_delta_p: float | None = None,
    signal_exit_th_loss_aux_mae_cut: float | None = None,
    signal_exit_th_loss_aux_short_only: bool = False,
    signal_exit_th_loss_aux_min_bars: int | None = None,
    signal_exit_th_loss_aux_gate_delta_unreal_positive: bool = False,
    signal_exit_th_loss_aux_gate_mfe_min: float | None = None,
    entry_min_unique_round_trips: int | None = None,
    entry_urt_state_log_path: Path | str | None = None,
    entry_urt_state_log_snapshot_override: int | None = None,
) -> tuple[dict | None, str | None]:
    """Returns (result, None) on success or (None, error_str) on failure."""
    import traceback
    from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine
    from src.strategies.ml_thresholds import resolve_ml_thresholds

    try:
        engine = get_ml_backtest_engine(
            strategy_name="ml_tcn",
            symbol=symbol,
            timeframe=timeframe,
            feature_preset="base",
            tcn_preset=None,
        )
        long_th, short_th = resolve_ml_thresholds(
            strategy_name="ml_tcn",
            symbol=symbol,
            timeframe=timeframe,
            use_optimized_thresholds=True,
        )
        short_th = short_th if short_th is not None else 0.5
        result = engine.run_backtest(
            long_threshold=long_th,
            short_threshold=short_th,
            use_optimized_threshold=True,
            use_stage2=True,
            use_strategy_guard_v2=True,
            proba_long_cache=proba_long_arr,
            proba_short_cache=proba_short_arr,
            df_with_proba=df_bt,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            min_max_proba=min_max_proba,
            max_entropy=max_entropy,
            min_proba_gap=min_proba_gap,
            min_directional_gap=min_directional_gap,
            max_flat_entry_proba=max_flat_entry_proba,
            decision_mode=decision_mode,
            min_directional_edge=min_directional_edge,
            require_direction_gt_flat=require_direction_gt_flat,
            min_side_flat_margin=min_side_flat_margin,
            min_confidence=min_confidence,
            use_legacy_entry_gates_with_directional=use_legacy_entry_gates_with_directional,
            min_hold_bars_override=min_hold,
            cooldown_bars_override=cooldown,
            regime_filter_enabled=regime_filter_enabled,
            regime_ema_span=regime_ema_span,
            regime_rule=regime_rule,
            regime_slope_lookback=regime_slope_lookback,
            vol_window=vol_window,
            vol_threshold=vol_threshold,
            breakout_lookback=breakout_lookback,
            breakout_mode=breakout_mode,
            slope_threshold=slope_threshold,
            q_window=q_window,
            q=q,
            p_floor=p_floor,
            position_scaling_enabled=position_scaling_enabled,
            position_scaling_mode=position_scaling_mode,
            position_p_floor=position_p_floor,
            position_p_full=position_p_full,
            position_size_min=position_size_min,
            position_size_max=position_size_max,
            position_p_mid=position_p_mid,
            position_k=position_k,
            early_exit_enabled=early_exit_enabled,
            early_exit_lookback=early_exit_lookback,
            early_exit_p_floor=early_exit_p_floor,
            early_exit_bad_k=early_exit_bad_k,
            entry_flat_gate_enabled=entry_flat_gate_enabled,
            max_flat_proba=max_flat_proba,
            flat_exit_aware_enabled=flat_exit_aware_enabled,
            flat_exit_threshold=flat_exit_threshold,
            flat_exit_badk_delta=flat_exit_badk_delta,
            time_stop_enabled=time_stop_enabled,
            time_stop_bars=time_stop_bars,
            partial_tp_enabled=partial_tp_enabled,
            partial_tp_threshold=partial_tp_threshold,
            partial_tp_ratio=partial_tp_ratio,
            break_even_stop_enabled=break_even_stop_enabled,
            be_threshold=be_threshold,
            emit_trade_log=emit_trade_log,
            signal_exit_th_trailing_window_bars=signal_exit_th_trailing_window_bars,
            signal_exit_th_delta_from_entry=signal_exit_th_delta_from_entry,
            signal_exit_th_loss_aux_delta_p=signal_exit_th_loss_aux_delta_p,
            signal_exit_th_loss_aux_mae_cut=signal_exit_th_loss_aux_mae_cut,
            signal_exit_th_loss_aux_short_only=signal_exit_th_loss_aux_short_only,
            signal_exit_th_loss_aux_min_bars=signal_exit_th_loss_aux_min_bars,
            signal_exit_th_loss_aux_gate_delta_unreal_positive=signal_exit_th_loss_aux_gate_delta_unreal_positive,
            signal_exit_th_loss_aux_gate_mfe_min=signal_exit_th_loss_aux_gate_mfe_min,
            entry_min_unique_round_trips=entry_min_unique_round_trips,
            entry_urt_state_log_path=(
                Path(entry_urt_state_log_path) if entry_urt_state_log_path is not None else None
            ),
            entry_urt_state_log_snapshot_override=entry_urt_state_log_snapshot_override,
        )
        return (result, None)
    except Exception as e:
        err_msg = f"{e}\n{traceback.format_exc()}"
        print(f"[BT] Backtest failed: {e}", flush=True)
        return (None, err_msg)


def signal_quality_7d(proba_long_arr, proba_short_arr) -> dict:
    import numpy as np
    pl = np.asarray(proba_long_arr, dtype=float)
    ps = np.asarray(proba_short_arr, dtype=float)
    pf = np.clip(1.0 - pl - ps, 0.0, 1.0)
    max_proba = np.maximum(np.maximum(pl, ps), pf)
    entropy = np.zeros(len(pl))
    for i in range(len(pl)):
        for p in (pl[i], pf[i], ps[i]):
            if p > 1e-12:
                entropy[i] -= p * math.log2(p)
    # class = argmax(long, flat, short) -> 0=long, 1=flat, 2=short
    stack = np.stack([pl, pf, ps], axis=1)
    cls = np.argmax(stack, axis=1)
    n = len(cls)
    return {
        "n_rows": n,
        "avg_max_proba": float(max_proba.mean()),
        "entropy_mean": float(entropy.mean()),
        "max_proba_lt_055": float((max_proba < 0.55).mean()),
        "class_ratio": {
            "long": float((cls == 0).sum() / n),
            "flat": float((cls == 1).sum() / n),
            "short": float((cls == 2).sum() / n),
        },
    }


def score_row(r: dict) -> tuple:
    """Sort: 1) cost_on_return desc, 2) trades asc, 3) max_drawdown asc, 4) avg_max_proba desc, entropy_mean asc."""
    cost_on = r.get("cost_on_return")
    if cost_on is None:
        cost_on = -999.0
    trades = r.get("trades")
    if trades is None:
        trades = 999999
    mdd = r.get("max_drawdown")
    if mdd is None:
        mdd = 999.0
    insp = r.get("inspect_7d") or {}
    proba7 = insp.get("avg_max_proba") or 0.0
    ent7 = insp.get("entropy_mean") or 2.0
    return (-float(cost_on), int(trades), float(mdd), -proba7, ent7)


def run_filter_sweep_for_model(
    df_bt,
    proba_long_arr,
    proba_short_arr,
    symbol: str,
    timeframe: str,
    commission_default: float,
    slippage_default: float,
) -> list[dict]:
    min_max_proba_list = [0.45, 0.50, 0.55]
    max_entropy_list = [1.60, 1.55, 1.50]
    min_hold_list = [12, 24]
    cooldown_list = [6, 12]
    results = []
    for min_max_proba in min_max_proba_list:
        for max_entropy in max_entropy_list:
            for min_hold in min_hold_list:
                for cooldown in cooldown_list:
                    res_on, _ = run_backtest_7d(
                        symbol, timeframe, df_bt, proba_long_arr, proba_short_arr,
                        commission_default, slippage_default,
                        min_max_proba=min_max_proba, max_entropy=max_entropy,
                        min_hold=min_hold, cooldown=cooldown,
                    )
                    res_off, _ = run_backtest_7d(
                        symbol, timeframe, df_bt, proba_long_arr, proba_short_arr,
                        0.0, 0.0,
                        min_max_proba=min_max_proba, max_entropy=max_entropy,
                        min_hold=min_hold, cooldown=cooldown,
                    )
                    if res_on is None:
                        res_on = {}
                    if res_off is None:
                        res_off = {}
                    trades = int(res_on.get("total_trades", 0))
                    results.append({
                        "min_max_proba": min_max_proba,
                        "max_entropy": max_entropy,
                        "min_hold": min_hold,
                        "cooldown": cooldown,
                        "trades": trades,
                        "cost_on_return": float(res_on.get("total_return", 0.0)),
                        "cost_off_return": float(res_off.get("total_return", 0.0)),
                        "cost_on_win_rate": float(res_on.get("win_rate", 0.0)),
                        "cost_off_win_rate": float(res_off.get("win_rate", 0.0)),
                        "filter_skip_stats": res_on.get("filter_skip_stats") or {},
                        "final_balance": float(res_on.get("equity_curve", [1.0])[-1]) if res_on.get("equity_curve") else 1.0,
                        "max_drawdown": float(res_on.get("max_drawdown", 0.0)),
                    })
    def sort_key(x):
        enough = 0 if x["trades"] >= 10 else 1
        return (enough, -x["cost_on_return"], -x["cost_off_return"], -x["trades"])
    results_sorted = sorted(results, key=sort_key)
    return results_sorted[:5]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--epochs-first", type=int, default=3)
    parser.add_argument("--epochs-second", type=int, default=10)
    parser.add_argument("--no-second-run", action="store_true")
    parser.add_argument("--thresholds", type=str, default="0.001,0.002,0.003,0.004")
    parser.add_argument("--horizons", type=str, default="5,15,30")
    parser.add_argument("--commission", type=float, default=None)
    parser.add_argument("--slippage", type=float, default=None)
    parser.add_argument("--run-filter-sweep", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Skip train/backtest, write minimal md/json for verification")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; evaluate existing models only (use --sweep-ids or all tcn_*.pt in models dir)")
    parser.add_argument("--sweep-ids", type=str, default=None, help="Comma-separated sweep ids for --eval-only, e.g. h5_t0p001,h15_t0p002")
    args = parser.parse_args()

    if args.dry_run:
        date_str = datetime.now().strftime("%Y%m%d")
        out_md = DIAG / f"tcn_label_sweep_v2_{date_str}.md"
        out_json = DIAG / f"tcn_label_sweep_v2_{date_str}.json"
        lines = [
            "# TCN 라벨 스윕 v2 결과 (dry-run)",
            f"날짜: {date_str} | days={args.days} | DRY RUN",
            "",
            "## (1) 12조합 요약표",
            "| horizon | thr | val_macro_f1 | val_avg_max_proba | val_entropy | 7d_avg_max_proba | 7d_entropy | cost_on_return | cost_off_return | trades | mdd |",
            "|---------|-----|--------------|-------------------|-------------|------------------|------------|----------------|-----------------|--------|-----|",
        ]
        for h in [5, 15, 30]:
            for thr in [0.001, 0.002, 0.003, 0.004]:
                lines.append(f"| {h} | {thr} | - | - | - | - | - | - | - | - | - |")
        lines.extend([
            "",
            "## (2) 상위 2개 선정",
            "h5_t0p001, h5_t0p002 (dry-run)",
            "",
            "## (3) 상위 모델별 Quick Filter Sweep 상위 5개",
            "(dry-run 생략)",
            "",
            "## (4) 결론 및 다음 액션",
            "- **추천 horizon:** 5",
            "- **추천 threshold:** pos=0.001, neg=-0.001",
            "",
            f"저장: {out_md} | {out_json}",
        ])
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        payload = {
            "date": date_str,
            "days": args.days,
            "dry_run": True,
            "results": [],
            "top2_ids": ["h5_t0p001", "h5_t0p002"],
            "recommendation": {"horizon": 5, "pos_threshold": 0.001, "neg_threshold": -0.001},
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Dry-run 완료: {out_md}, {out_json}")
        return 0

    thresholds = [float(x.strip()) for x in args.thresholds.split(",")]
    horizons = [int(x.strip()) for x in args.horizons.split(",")]
    commission = args.commission if args.commission is not None else 0.0009
    slippage = args.slippage if args.slippage is not None else 0.0001

    sweep_configs = []
    if args.eval_only:
        if args.sweep_ids:
            for sid in [x.strip() for x in args.sweep_ids.split(",") if x.strip()]:
                parsed = _parse_sweep_id(sid)
                if parsed and (MODELS_DIR / f"tcn_{sid}.pt").exists():
                    h, thr = parsed
                    sweep_configs.append({
                        "id": sid,
                        "horizon": h,
                        "pos_threshold": thr,
                        "neg_threshold": -thr,
                    })
        else:
            for pt in sorted(MODELS_DIR.glob("tcn_*.pt")):
                sid = pt.stem.replace("tcn_", "")
                parsed = _parse_sweep_id(sid)
                if parsed:
                    h, thr = parsed
                    sweep_configs.append({
                        "id": sid,
                        "horizon": h,
                        "pos_threshold": thr,
                        "neg_threshold": -thr,
                    })
        if not sweep_configs:
            print("ERROR: --eval-only but no valid sweep configs (no models or invalid --sweep-ids)")
            return 1
    else:
        for h in horizons:
            for thr in thresholds:
                sweep_configs.append({
                    "id": _sweep_id(h, thr),
                    "horizon": h,
                    "pos_threshold": thr,
                    "neg_threshold": -thr,
                })

    date_str = datetime.now().strftime("%Y%m%d")
    out_md = DIAG / f"tcn_label_sweep_v2_{date_str}.md"
    out_json = DIAG / f"tcn_label_sweep_v2_{date_str}.json"
    print(f"Sweep configs: {len(sweep_configs)} (eval_only={args.eval_only}, symbol={args.symbol}, timeframe={args.timeframe}, days={args.days})", flush=True)

    results = []
    for cfg in sweep_configs:
        sid = cfg["id"]
        model_path = MODELS_DIR / f"tcn_{sid}.pt"
        print(f"[sweep] {sid} ...", flush=True)
        if args.eval_only:
            ok = model_path.exists()
        else:
            ok = run_train(
                sid, cfg["horizon"], cfg["pos_threshold"], cfg["neg_threshold"], args.epochs_first
            )
        metrics = load_metrics(sid) if ok else None
        inspect_7d = None
        cost_on_return = None
        cost_off_return = None
        trades = None
        mdd = None
        bars_used = None
        entries_attempted = None
        entries_executed = None
        cost_on_win_rate = None
        cost_off_win_rate = None
        eval_error = None
        backtest_error = None

        if ok and model_path.exists():
            print(f"  [7d] {sid} get_7d_ohlcv_and_proba ...", flush=True)
            try:
                triple, eval_error = get_7d_ohlcv_and_proba(
                    model_path, args.symbol, args.timeframe, args.days,
                    log_prefix=f"{sid} ",
                )
            except Exception as e:
                import traceback
                eval_error = f"{e}\n{traceback.format_exc()}"
                triple = None
                print(f"  [EVAL] {sid} exception: {e}", flush=True)
            if triple is not None:
                df_bt, pl, ps = triple
                print(f"  [BT] {sid} run_backtest cost_on ...", flush=True)
                inspect_7d = signal_quality_7d(pl, ps)
                res_on, err_on = run_backtest_7d(
                    args.symbol, args.timeframe, df_bt, pl, ps,
                    commission, slippage,
                )
                if err_on:
                    backtest_error = (backtest_error or "") + f"[cost_on] {err_on}\n"
                print(f"  [BT] {sid} run_backtest cost_off ...", flush=True)
                res_off, err_off = run_backtest_7d(
                    args.symbol, args.timeframe, df_bt, pl, ps,
                    0.0, 0.0,
                )
                if err_off:
                    backtest_error = (backtest_error or "") + f"[cost_off] {err_off}\n"
                if res_on:
                    cost_on_return = float(res_on.get("total_return", 0.0))
                    trades = int(res_on.get("total_trades", 0))
                    mdd = float(res_on.get("max_drawdown", 0.0))
                    bars_used = len(df_bt)
                    entries_attempted = res_on.get("entries_attempted")
                    cts = res_on.get("cap_trigger_stats") or {}
                    entries_executed = cts.get("entries_executed")
                if res_off:
                    cost_off_return = float(res_off.get("total_return", 0.0))
                cost_on_win_rate = float(res_on.get("win_rate", 0.0)) if res_on else None
                cost_off_win_rate = float(res_off.get("win_rate", 0.0)) if res_off else None

        results.append({
            "id": sid,
            "horizon": cfg["horizon"],
            "pos_threshold": cfg["pos_threshold"],
            "neg_threshold": cfg["neg_threshold"],
            "train_ok": ok,
            "metrics": metrics,
            "inspect_7d": inspect_7d,
            "cost_on_return": cost_on_return,
            "cost_off_return": cost_off_return,
            "trades": trades,
            "max_drawdown": mdd,
            "bars_used": bars_used,
            "entries_attempted": entries_attempted,
            "entries_executed": entries_executed,
            "cost_on_win_rate": cost_on_win_rate,
            "cost_off_win_rate": cost_off_win_rate,
            "eval_error": eval_error,
            "backtest_error": backtest_error,
        })

    if not args.no_second_run and top2_ids and not args.eval_only:
        for sid in top2_ids:
            cfg = next(c for c in sweep_configs if c["id"] == sid)
            run_train(
                sid, cfg["horizon"], cfg["pos_threshold"], cfg["neg_threshold"], args.epochs_second
            )
        for r in results:
            if r["id"] not in top2_ids:
                continue
            sid = r["id"]
            model_path = MODELS_DIR / f"tcn_{sid}.pt"
            triple, _ = get_7d_ohlcv_and_proba(
                model_path, args.symbol, args.timeframe, args.days, log_prefix=f"{sid} ",
            )
            if triple is not None:
                df_bt, pl, ps = triple
                r["inspect_7d"] = signal_quality_7d(pl, ps)
                res_on, _ = run_backtest_7d(
                    args.symbol, args.timeframe, df_bt, pl, ps, commission, slippage
                )
                res_off, _ = run_backtest_7d(
                    args.symbol, args.timeframe, df_bt, pl, ps, 0.0, 0.0
                )
                if res_on:
                    r["cost_on_return"] = float(res_on.get("total_return", 0.0))
                    r["trades"] = int(res_on.get("total_trades", 0))
                    r["max_drawdown"] = float(res_on.get("max_drawdown", 0.0))
                if res_off:
                    r["cost_off_return"] = float(res_off.get("total_return", 0.0))

    filter_sweep_top5_by_id = {}
    if args.run_filter_sweep and top2_ids:
        for sid in top2_ids[:2]:
            cfg = next(c for c in sweep_configs if c["id"] == sid)
            model_path = MODELS_DIR / f"tcn_{sid}.pt"
            triple, _ = get_7d_ohlcv_and_proba(
                model_path, args.symbol, args.timeframe, args.days, log_prefix=f"{sid} ",
            )
            if triple is not None:
                df_bt, pl, ps = triple
                top5 = run_filter_sweep_for_model(
                    df_bt, pl, ps, args.symbol, args.timeframe, commission, slippage
                )
                filter_sweep_top5_by_id[sid] = top5

    n_combos = len(results)
    sorted_results = sorted(results, key=score_row)
    top2_ids = [r["id"] for r in sorted_results[:2]]
    top2_detail = [(r["id"], r["horizon"], r["pos_threshold"]) for r in sorted_results[:2]]

    lines = [
        "# TCN 라벨 스윕 v2 결과",
        f"날짜: {date_str} | days={args.days} | 1차 epochs={args.epochs_first}" + (" | eval_only" if args.eval_only else ""),
        "",
        "## 상위 2개 후보 (추천)",
    ]
    for sid, h, thr in top2_detail:
        lines.append(f"- **{sid}**: horizon={h}, pos_threshold={thr}")
    lines.extend([
        "",
        f"## (1) 조합 요약표 ({n_combos}개)",
        "| horizon | thr | val_macro_f1 | val_avg_max_proba | val_entropy | 7d_avg_max_proba | 7d_entropy | cost_on_return | cost_off_return | trades | mdd |",
        "|---------|-----|--------------|-------------------|-------------|------------------|------------|----------------|-----------------|--------|-----|",
    ])
    for r in results:
        met = r.get("metrics") or {}
        insp = r.get("inspect_7d") or {}
        lines.append(
            "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} | {} | {} | {} |".format(
                r["horizon"],
                r["pos_threshold"],
                met.get("macro_f1") or 0.0,
                met.get("val_avg_max_proba") or 0.0,
                met.get("val_entropy_mean") or 0.0,
                insp.get("avg_max_proba") or 0.0,
                insp.get("entropy_mean") or 0.0,
                f"{r.get('cost_on_return') or 0:.4f}" if r.get("cost_on_return") is not None else "-",
                f"{r.get('cost_off_return') or 0:.4f}" if r.get("cost_off_return") is not None else "-",
                r.get("trades") if r.get("trades") is not None else "-",
                f"{r.get('max_drawdown') or 0:.4f}" if r.get("max_drawdown") is not None else "-",
            )
        )

    has_errors = any(r.get("eval_error") or r.get("backtest_error") for r in results)
    if has_errors:
        lines.append("")
        lines.append("## 에러 요약 (eval_error / backtest_error)")
        for r in results:
            if r.get("eval_error") or r.get("backtest_error"):
                lines.append(f"### {r['id']}")
                if r.get("eval_error"):
                    lines.append("**eval_error:**")
                    lines.append("```")
                    lines.append((r["eval_error"] or "").strip()[:2000])
                    lines.append("```")
                if r.get("backtest_error"):
                    lines.append("**backtest_error:**")
                    lines.append("```")
                    lines.append((r["backtest_error"] or "").strip()[:2000])
                    lines.append("```")
                lines.append("")

    lines.extend([
        "",
        "## (2) 상위 2개 선정",
        ", ".join(top2_ids),
        "",
    ])
    for r in sorted_results[:2]:
        lines.append(
            f"- **{r['id']}**: cost_on_return={r.get('cost_on_return')}, cost_off_return={r.get('cost_off_return')}, "
            f"trades={r.get('trades')}, mdd={r.get('max_drawdown')}, "
            f"cost_on_win_rate={r.get('cost_on_win_rate')}, cost_off_win_rate={r.get('cost_off_win_rate')}"
        )
    lines.append("")

    if filter_sweep_top5_by_id:
        lines.append("## (3) 상위 모델별 Quick Filter Sweep 상위 5개")
        for sid, top5 in filter_sweep_top5_by_id.items():
            lines.append(f"### {sid}")
            lines.append("| min_max_proba | max_entropy | min_hold | cooldown | trades | cost_on_return | cost_off_return | mdd |")
            lines.append("|---------------|-------------|----------|----------|--------|----------------|-----------------|-----|")
            for row in top5:
                t = row["trades"]
                tstr = str(t) if t > 0 else "0-trade"
                lines.append(
                    "| {:.2f} | {:.2f} | {} | {} | {} | {:.4f} | {:.4f} | {:.4f} |".format(
                        row["min_max_proba"], row["max_entropy"], row["min_hold"], row["cooldown"],
                        tstr, row["cost_on_return"], row["cost_off_return"], row["max_drawdown"]
                    )
                )
            lines.append("")
            lines.append("filter_skip_stats:")
            for i, row in enumerate(top5, 1):
                lines.append(f"  - #{i}: {row.get('filter_skip_stats', {})}")
            lines.append("")

    lines.append("## (4) 결론 및 다음 액션")
    best = sorted_results[0] if sorted_results else None
    if best:
        lines.append(f"- **추천 horizon:** {best['horizon']}")
        lines.append(f"- **추천 threshold:** pos={best['pos_threshold']}, neg={best['neg_threshold']}")
        cand_a = any(
            r.get("cost_on_return") is not None and r.get("cost_on_return") >= -0.01
            and r.get("cost_off_return") is not None and r.get("cost_off_return") > 0
            and 10 <= (r.get("trades") or 0) <= 200
            for r in results
        )
        cand_b = any(r.get("cost_on_return") is not None and r.get("cost_on_return") > 0 for r in results)
        cand_c = any(
            r.get("trades") is not None and r.get("trades") > 200 and (r.get("cost_on_return") or 0) < -0.02
            for r in results
        )
        if cand_a:
            lines.append("- **(A) 추가 개선 후보:** cost_on >= -0.01, cost_off > 0, trades 10~200 구간 존재.")
        if cand_b:
            lines.append("- **(B) 강력 후보:** cost_on > 0 인 조합 존재.")
        if cand_c:
            lines.append("- **(C) 과매매 억제 필요:** trades 과다로 cost_on 악화 구간 존재.")
    else:
        lines.append("(데이터 없음)")
    lines.append("")
    lines.append(f"저장: {out_md} | {out_json}")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    best = sorted_results[0] if sorted_results else None
    payload = {
        "date": date_str,
        "days": args.days,
        "epochs_first": args.epochs_first,
        "epochs_second": args.epochs_second,
        "results": results,
        "top2_ids": top2_ids,
        "filter_sweep_top5_by_id": {k: v for k, v in filter_sweep_top5_by_id.items()},
        "recommendation": {
            "horizon": best["horizon"],
            "pos_threshold": best["pos_threshold"],
            "neg_threshold": best["neg_threshold"],
        } if best else None,
    }
    def _json_default(obj):
        import numpy as np
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if not np.isnan(obj) else None
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)

    print("\n" + "=" * 80)
    print("TCN 라벨 스윕 v2 완료")
    print("=" * 80)
    print(f"저장: {out_md}, {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
