#!/usr/bin/env python3
"""
latest t_max=2026-03-20T07:40:00+00:00 고정 + 최근 60d window에서
threshold=0.60으로 ml_tcn backtest 1회 실행

dump_trades_path CSV를 만든 뒤,
요청 지표(평균 win/loss, PF, expectancy, gross/net mean, avg cost per trade)
및 closed trade(라운드트립)별 개별 분해를 출력한다.

※ res["trades"] 행 수 = 엔진이 반환한 체결 레코드 수(중복 제거 후에는 라운드트립 수와 일치).
  result["unique_round_trips"] / ["duplicate_trade_rows"]로 검증 가능.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _to_utc_ts(x: Any) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _analyze_trades_from_result(*, trades: list[dict[str, Any]], expected_trades: int) -> dict[str, Any]:
    if expected_trades is not None and len(trades) != expected_trades:
        print(
            f"[WARN] expected_trades={expected_trades} but got total_trades={len(trades)} from engine result",
            flush=True,
        )

    normalized: list[dict[str, Any]] = []
    for i, t in enumerate(trades, start=1):
        direction = str(t.get("direction", "")).upper()
        entry = float(t.get("entry_price"))
        exitp = float(t.get("exit_price"))
        net_profit = float(t.get("profit", 0.0))

        if direction == "LONG":
            gross_return = (exitp - entry) / entry
        elif direction == "SHORT":
            gross_return = (entry - exitp) / entry
        else:
            gross_return = np.nan

        cost_approx = float(gross_return - net_profit) if np.isfinite(gross_return) else np.nan

        normalized.append(
            {
                "id": i,
                "direction": direction,
                "entry_price": entry,
                "exit_price": exitp,
                "gross_return": float(gross_return),
                "net_profit": net_profit,
                "cost_approx": cost_approx,
                "is_win": net_profit > 0,
            }
        )

    pnls = np.asarray([t["net_profit"] for t in normalized], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0  # <= 0
    win_loss_ratio = float(avg_win / abs(avg_loss)) if len(wins) > 0 and len(losses) > 0 and avg_loss != 0.0 else np.nan

    profit_factor = float(wins.sum() / abs(losses.sum())) if len(wins) > 0 and len(losses) > 0 and abs(losses.sum()) > 1e-12 else np.nan
    expectancy = float(np.mean(pnls)) if len(pnls) > 0 else 0.0

    gross_rets = np.asarray([t["gross_return"] for t in normalized], dtype=float)
    gross_mean_return = float(np.mean(gross_rets)) if len(gross_rets) > 0 else 0.0
    net_mean_return = float(np.mean(pnls)) if len(pnls) > 0 else 0.0

    avg_cost_per_trade = float(np.mean([t["cost_approx"] for t in normalized])) if normalized else np.nan

    print("\n## Closed Trades Breakdown (threshold=0.60, recent 60d)", flush=True)
    print("| id | dir | entry | exit | gross_return | net_profit | cost_approx |", flush=True)
    print("|---:|---|---:|---:|---:|---:|---:|", flush=True)
    for t in normalized:
        print(
            f"| {t['id']} | {t['direction']} | {t['entry_price']:.2f} | {t['exit_price']:.2f} | "
            f"{t['gross_return']:.8f} | {t['net_profit']:.8f} | {t['cost_approx']:.8f} |",
            flush=True,
        )

    print("\n## Trade-level Metrics (threshold=0.60)", flush=True)
    print(f"average win: {avg_win:.8f}", flush=True)
    print(f"average loss: {avg_loss:.8f}", flush=True)
    print(f"win/loss ratio: {win_loss_ratio}", flush=True)
    print(f"profit factor: {profit_factor}", flush=True)
    print(f"expectancy (net mean profit): {expectancy:.8f}", flush=True)
    print(f"gross mean return: {gross_mean_return:.8f}", flush=True)
    print(f"net mean return: {net_mean_return:.8f}", flush=True)
    print(f"average cost per trade (approx): {avg_cost_per_trade}", flush=True)

    return {
        "trades": normalized,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "gross_mean_return": gross_mean_return,
        "net_mean_return": net_mean_return,
        "avg_cost_per_trade": avg_cost_per_trade,
        "n": len(normalized),
        "wins_count": int(len(wins)),
        "losses_count": int(len(losses)),
    }


def _dump_and_analyze_trades(*, trades_dump_csv: Path, expected_trades: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with open(trades_dump_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    entries: dict[str, dict[str, Any]] = {}
    exits: dict[str, dict[str, Any]] = {}

    for r in rows:
        event_type = (r.get("event_type") or "").strip()
        trade_index = (r.get("trade_index") or "").strip()
        direction = (r.get("direction") or "").strip()
        price = r.get("price") or ""
        pnl = r.get("pnl") or ""
        if trade_index == "":
            continue
        try:
            trade_index_i = int(trade_index)
        except ValueError:
            continue

        try:
            price_f = float(price) if price != "" else np.nan
        except Exception:
            price_f = np.nan

        try:
            pnl_f = float(pnl) if pnl != "" else np.nan
        except Exception:
            pnl_f = np.nan

        if event_type.startswith("ENTRY"):
            entries[str(trade_index_i)] = {
                "trade_index": trade_index_i,
                "direction": direction,
                "entry_price": price_f,
            }
        elif event_type.startswith("EXIT"):
            exits[str(trade_index_i)] = {
                "trade_index": trade_index_i,
                "direction": direction,
                "exit_price": price_f,
                "net_profit": pnl_f,
            }

    common = sorted(set(entries.keys()) & set(exits.keys()), key=lambda k: int(k))
    if expected_trades is not None and len(common) != expected_trades:
        print(f"[WARN] expected_trades={expected_trades} but found matched EXIT trades={len(common)}", flush=True)

    trades = []
    for k in common:
        e = entries[k]
        x = exits[k]
        entry = float(e["entry_price"])
        exitp = float(x["exit_price"])
        net = float(x["net_profit"])
        direction = e["direction"]

        if direction.upper() == "LONG":
            gross_return = (exitp - entry) / entry
        elif direction.upper() == "SHORT":
            gross_return = (entry - exitp) / entry
        else:
            gross_return = np.nan

        cost_approx = float(gross_return - net) if np.isfinite(gross_return) and np.isfinite(net) else np.nan
        trades.append(
            {
                "trade_index": int(e["trade_index"]),
                "direction": direction,
                "entry_price": entry,
                "exit_price": exitp,
                "gross_return": float(gross_return),
                "net_profit": net,
                "cost_approx": cost_approx,
                "is_win": bool(net > 0),
            }
        )

    trades = sorted(trades, key=lambda d: d["trade_index"])

    pnls = np.asarray([t["net_profit"] for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0  # <= 0
    win_loss_ratio = float(avg_win / abs(avg_loss)) if len(wins) > 0 and len(losses) > 0 and avg_loss != 0.0 else np.nan

    profit_factor = float(wins.sum() / abs(losses.sum())) if len(wins) > 0 and len(losses) > 0 and abs(losses.sum()) > 1e-12 else np.nan
    expectancy = float(np.mean(pnls)) if len(pnls) > 0 else 0.0

    gross_rets = np.asarray([t["gross_return"] for t in trades], dtype=float)
    gross_mean_return = float(np.mean(gross_rets)) if len(gross_rets) > 0 else 0.0
    net_mean_return = float(np.mean(pnls)) if len(pnls) > 0 else 0.0

    avg_cost_per_trade = float(np.mean([t["cost_approx"] for t in trades])) if trades else np.nan

    print("\n## Closed Trades Breakdown (threshold=0.60, recent 60d)", flush=True)
    print("| id | dir | entry | exit | gross_return | net_profit | cost_approx |", flush=True)
    print("|---:|---|---:|---:|---:|---:|---:|", flush=True)
    for t in trades:
        print(
            f"| {t['trade_index']} | {t['direction']} | {t['entry_price']:.2f} | {t['exit_price']:.2f} | "
            f"{t['gross_return']:.8f} | {t['net_profit']:.8f} | {t['cost_approx']:.8f} |",
            flush=True,
        )

    print("\n## Trade-level Metrics (threshold=0.60)", flush=True)
    print(f"average win: {avg_win:.8f}", flush=True)
    print(f"average loss: {avg_loss:.8f}", flush=True)
    print(f"win/loss ratio: {win_loss_ratio}", flush=True)
    print(f"profit factor: {profit_factor}", flush=True)
    print(f"expectancy (net mean profit): {expectancy:.8f}", flush=True)
    print(f"gross mean return: {gross_mean_return:.8f}", flush=True)
    print(f"net mean return: {net_mean_return:.8f}", flush=True)
    print(f"average cost per trade (approx): {avg_cost_per_trade}", flush=True)

    return {
        "trades": trades,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "gross_mean_return": gross_mean_return,
        "net_mean_return": net_mean_return,
        "avg_cost_per_trade": avg_cost_per_trade,
        "n": len(trades),
        "wins_count": int(len(wins)),
        "losses_count": int(len(losses)),
    }


def main() -> int:
    # fixed t_max
    t_max = _to_utc_ts("2026-03-20T07:40:00+00:00")
    window_days = 60
    window_start = t_max - pd.Timedelta(days=window_days)

    threshold = 0.60
    # None: 엔진 total_trades(라운드트립 수)와만 비교; 고정 숫자는 중복 제거 전후로 달라질 수 있음
    expected_trades: int | None = None

    # proba building (fast sweep와 동일한 pipeline)
    from scripts.run_fr2_diagnostics import MODELS_DIR, get_ohlcv_and_proba
    from scripts.run_fr2_regime_conditioning import add_regime_columns
    from src.strategies.ensemble_strategy import EnsembleInputs, build_ensemble_proba, build_fr2_c4_mask
    from src.strategies.ml_thresholds import resolve_ml_thresholds
    from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine

    BASELINE_PT = MODELS_DIR / "tcn_h15_t0p004.pt"
    FR2_PT = MODELS_DIR / "tcn_h15_micro_v1.pt"
    DAYS_FULL = 62
    end_date_utc_str = t_max.strftime("%Y-%m-%d")

    print(f"[RUN] t_max={t_max.isoformat()}", flush=True)
    print(f"[RUN] window=[{window_start.isoformat()} .. {t_max.isoformat()}]", flush=True)

    triple_base, err_b = get_ohlcv_and_proba(DAYS_FULL, BASELINE_PT, "base", False, end_date=end_date_utc_str)
    if err_b or triple_base is None:
        raise RuntimeError(f"get_ohlcv_and_proba(base) failed: {err_b}")
    df_b, pl_b, ps_b, _ = triple_base

    triple_fr2, err_f = get_ohlcv_and_proba(DAYS_FULL, FR2_PT, "microstructure_v1", True, end_date=end_date_utc_str)
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
    pl_base = joined["pl_base"].to_numpy(dtype=np.float32)
    ps_base = joined["ps_base"].to_numpy(dtype=np.float32)
    pl_fr2 = joined["pl_fr2"].to_numpy(dtype=np.float32)
    ps_fr2 = joined["ps_fr2"].to_numpy(dtype=np.float32)

    df_bt_reg = add_regime_columns(DAYS_FULL, df_bt.copy())
    base_c4 = ((df_bt_reg["trend_regime"] == "uptrend") & (df_bt_reg["vol_regime"] == "high_vol")).to_numpy()
    c4_mask = build_fr2_c4_mask(base_c4, persistence_bars=6)

    ensemble_inputs = EnsembleInputs(
        pl_base=pl_base,
        ps_base=ps_base,
        pl_fr2=pl_fr2,
        ps_fr2=ps_fr2,
        c4_active=c4_mask,
    )
    pl_primary, ps_primary = build_ensemble_proba(ensemble_inputs, mode="override")

    t = pd.to_datetime(df_bt["timestamp"], utc=True)
    mask = (t >= window_start) & (t <= t_max)
    df_60 = df_bt.loc[mask].reset_index(drop=True)

    pl_60 = np.asarray(pl_primary, dtype=np.float32)[mask.to_numpy()]
    ps_60 = np.asarray(ps_primary, dtype=np.float32)[mask.to_numpy()]

    print(f"[RUN] window rows={len(df_60)}", flush=True)

    pf_60 = np.clip(1.0 - pl_60 - ps_60, 0.0, 1.0)
    mp_60 = np.maximum(np.maximum(pl_60, ps_60), pf_60)
    print(f"[RUN] mp min={float(mp_60.min()):.6f} max={float(mp_60.max()):.6f}", flush=True)

    # backtest
    COMMISSION = 0.0009
    SLIPPAGE = 0.0001
    MAX_ENTROPY = 1.30
    MIN_HOLD = 24
    COOLDOWN = 24
    TIME_STOP_BARS = 72
    EARLY_EXIT_BAD_K = 8

    engine = get_ml_backtest_engine(
        strategy_name="ml_tcn",
        symbol="BTCUSDT",
        timeframe="5m",
        feature_preset="base",
        tcn_preset=None,
    )
    long_th, short_th = resolve_ml_thresholds(
        strategy_name="ml_tcn",
        symbol="BTCUSDT",
        timeframe="5m",
        use_optimized_thresholds=True,
    )
    short_th = short_th if short_th is not None else 0.5

    thr_str = f"{threshold:.2f}".replace(".", "p")
    dump_csv = PROJECT_ROOT / "data" / "backtest_dumps" / f"trades_boundary_validation_20260320_0740_thr_{thr_str}.csv"

    print(f"[TRADE DUMP] threshold={threshold:.2f} dump to: {dump_csv}", flush=True)

    res0 = engine.run_backtest(
        long_threshold=long_th,
        short_threshold=short_th,
        use_optimized_threshold=True,
        use_stage2=True,
        use_strategy_guard_v2=True,
        proba_long_cache=pl_60,
        proba_short_cache=ps_60,
        df_with_proba=df_60,
        commission_rate=COMMISSION,
        slippage_rate=SLIPPAGE,
        min_max_proba=threshold,
        max_entropy=MAX_ENTROPY,
        min_proba_gap=0.0,
        min_directional_gap=None,
        max_flat_entry_proba=None,
        decision_mode="argmax",
        min_directional_edge=None,
        require_direction_gt_flat=True,
        min_side_flat_margin=None,
        min_confidence=None,
        use_legacy_entry_gates_with_directional=True,
        min_hold_bars_override=MIN_HOLD,
        cooldown_bars_override=COOLDOWN,
        regime_filter_enabled=False,
        regime_ema_span=200,
        regime_rule="ema_only",
        regime_slope_lookback=48,
        vol_window=48,
        vol_threshold=0.0010,
        breakout_lookback=96,
        breakout_mode="ema_high",
        slope_threshold=1e-5,
        q_window=8640,
        q=0.90,
        p_floor=0.55,
        position_scaling_enabled=False,
        position_scaling_mode="linear",
        position_p_floor=0.55,
        position_p_full=0.65,
        position_size_min=0.25,
        position_size_max=1.0,
        position_p_mid=0.60,
        position_k=25.0,
        early_exit_enabled=True,
        early_exit_lookback=12,
        early_exit_p_floor=0.55,
        early_exit_bad_k=EARLY_EXIT_BAD_K,
        entry_flat_gate_enabled=False,
        max_flat_proba=0.45,
        flat_exit_aware_enabled=False,
        flat_exit_threshold=0.30,
        flat_exit_badk_delta=2,
        time_stop_enabled=True,
        time_stop_bars=TIME_STOP_BARS,
        partial_tp_enabled=False,
        partial_tp_threshold=0.0025,
        partial_tp_ratio=0.5,
        break_even_stop_enabled=False,
        be_threshold=0.0,
        emit_trade_log=False,
        dump_trades_path=str(dump_csv),
    )

    if isinstance(res0, dict):
        print(f"[CHECK] engine total_trades={res0.get('total_trades')} win_rate={res0.get('win_rate')}", flush=True)
        print(
            f"[CHECK] unique_round_trips={res0.get('unique_round_trips')} "
            f"duplicate_trade_rows={res0.get('duplicate_trade_rows')}",
            flush=True,
        )
        trades = res0.get("trades", []) or []
    else:
        trades = []

    _analyze_trades_from_result(trades=trades, expected_trades=expected_trades)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

