#!/usr/bin/env python3
"""
운영용 threshold 경계 검증 (latest t_max 고정 + 최근 60d window)

범위:
- threshold=0.61~0.64: pass_rate / entries_attempted / trades / win_rate / mean_return / cost_on
- threshold=0.60: closed trades 10개 손익 분해 + (avg win/loss, PF, expectancy, gross/net mean, avg cost) 계산

주의:
- Meta Layer 로직 변경 없음
- 학습 없음
- 입력 window 고정: t_max=2026-03-20T07:40:00+00:00, window=[t_max-60d, t_max]
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _mean_return_and_sharpe(equity_curve: list[float] | None) -> tuple[float, float]:
    if not equity_curve or len(equity_curve) < 2:
        return 0.0, 0.0
    arr = np.asarray(equity_curve, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0, 0.0
    rets = np.diff(arr) / arr[:-1]
    if len(rets) == 0:
        return 0.0, 0.0
    mean_ret = float(np.mean(rets))
    std_ret = float(np.std(rets))
    sharpe = float(mean_ret / std_ret) if std_ret != 0.0 else 0.0
    return mean_ret, sharpe


def _to_utc_ts(x: Any) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _calc_pass_rate(mp: np.ndarray, th: float) -> tuple[int, float]:
    cnt = int((mp >= th).sum())
    rate = float(cnt / len(mp)) if len(mp) > 0 else 0.0
    return cnt, rate


def _dump_and_analyze_trades(
    *,
    trades_dump_csv: Path,
    expected_trades: int,
):
    """
    trades_dump_csv: ml_backtest_engine_impl dump_trades_path output.
    """
    rows: list[dict[str, Any]] = []
    with open(trades_dump_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # event_type values: likely "ENTRY_LONG"/"ENTRY_SHORT"/"EXIT_LONG"/"EXIT_SHORT" or similar.
    # We use direction + trade_index + event_type.
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

        # Normalize types
        try:
            trade_index_i = int(trade_index)
        except ValueError:
            continue

        if event_type.startswith("ENTRY"):
            entries[str(trade_index_i)] = {
                "trade_index": trade_index_i,
                "direction": direction,
                "entry_price": float(price) if price != "" else np.nan,
            }
        elif event_type.startswith("EXIT"):
            exits[str(trade_index_i)] = {
                "trade_index": trade_index_i,
                "direction": direction,
                "exit_price": float(price) if price != "" else np.nan,
                "net_profit": float(pnl) if pnl != "" else np.nan,
            }

    common = sorted(set(entries.keys()) & set(exits.keys()), key=lambda k: int(k))
    if expected_trades is not None:
        # We tolerate mismatch; print it to help debugging.
        if len(common) != expected_trades:
            print(f"[TRADE DUMP] expected_trades={expected_trades} but found_exit_entries={len(common)}", flush=True)

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
        trades.append({
            "trade_index": int(e["trade_index"]),
            "direction": direction,
            "entry_price": entry,
            "exit_price": exitp,
            "gross_return": float(gross_return),
            "net_profit": float(net),
            "cost_approx": cost_approx,
            "is_win": net > 0,
        })

    trades = sorted(trades, key=lambda d: d["trade_index"])

    pnls = np.asarray([t["net_profit"] for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0  # <=0
    win_loss_ratio = float(avg_win / abs(avg_loss)) if len(wins) > 0 and len(losses) > 0 and avg_loss != 0.0 else np.nan

    profit_factor = float(wins.sum() / abs(losses.sum())) if len(wins) > 0 and len(losses) > 0 and abs(losses.sum()) > 1e-12 else np.nan
    expectancy = float(np.mean(pnls)) if len(pnls) > 0 else 0.0

    gross_rets = np.asarray([t["gross_return"] for t in trades], dtype=float)
    gross_mean_return = float(np.mean(gross_rets)) if len(gross_rets) > 0 else 0.0
    net_mean_return = float(np.mean(pnls)) if len(pnls) > 0 else 0.0

    avg_cost_per_trade = float(np.mean([t["cost_approx"] for t in trades])) if trades else np.nan

    # Print individual trade breakdown
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
        "wins_count": int(len(wins)),
        "losses_count": int(len(losses)),
        "n": int(len(trades)),
    }


def main() -> int:
    # 고정 t_max
    t_max = _to_utc_ts("2026-03-20T07:40:00+00:00")
    window_days = 60
    window_start = t_max - pd.Timedelta(days=window_days)

    thresholds_to_test = [0.61, 0.62, 0.63, 0.64]
    # pass_rate 비교에 참고용으로 0.60/0.65도 계산
    pass_rate_thresholds = [0.60, 0.61, 0.62, 0.63, 0.64, 0.65]

    # fast sweep와 동일 pipeline
    from scripts.run_fr2_diagnostics import MODELS_DIR, get_ohlcv_and_proba
    from scripts.run_fr2_regime_conditioning import add_regime_columns
    from scripts.run_fr2_regime_conditioning import add_regime_columns as _add_regime_columns  # noqa: F401
    from src.strategies.ensemble_strategy import EnsembleInputs, build_ensemble_proba, build_fr2_c4_mask
    from scripts.run_tcn_label_sweep_v2 import run_backtest_7d

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
    pf_60 = np.clip(1.0 - pl_60 - ps_60, 0.0, 1.0)
    mp_60 = np.maximum(np.maximum(pl_60, ps_60), pf_60)

    print(f"[RUN] window rows={len(df_60)} mp min={float(mp_60.min()):.6f} max={float(mp_60.max()):.6f}", flush=True)

    # pass rates
    pass_rates: dict[float, float] = {}
    for th in pass_rate_thresholds:
        _, rate = _calc_pass_rate(mp_60, th)
        pass_rates[th] = rate

    # backtest params (sweep와 동일)
    COMMISSION = 0.0009
    SLIPPAGE = 0.0001
    MAX_ENTROPY = 1.30
    MIN_HOLD = 24
    COOLDOWN = 24
    TIME_STOP_BARS = 72
    EARLY_EXIT_BAD_K = 8

    # STEP 1/2: run for boundary thresholds 0.61~0.64
    results: list[dict[str, Any]] = []
    for thr in thresholds_to_test:
        print(f"[BT] min_max_proba={thr:.2f} ...", flush=True)
        res, err = run_backtest_7d(
            "BTCUSDT",
            "5m",
            df_60,
            pl_60,
            ps_60,
            commission_rate=COMMISSION,
            slippage_rate=SLIPPAGE,
            min_max_proba=float(thr),
            max_entropy=MAX_ENTROPY,
            decision_mode="argmax",
            min_hold=MIN_HOLD,
            cooldown=COOLDOWN,
            time_stop_enabled=True,
            time_stop_bars=TIME_STOP_BARS,
            early_exit_enabled=True,
            early_exit_bad_k=EARLY_EXIT_BAD_K,
            emit_trade_log=False,
        )
        if err:
            raise RuntimeError(f"run_backtest_7d failed at thr={thr}: {err}")
        if res is None:
            res = {}

        equity_curve = res.get("equity_curve") or []
        total_trades = int(res.get("total_trades", 0) or 0)
        entries_attempted = int(res.get("entries_attempted", 0) or 0)
        win_rate = float(res.get("win_rate", 0.0) or 0.0)
        mean_return, _ = _mean_return_and_sharpe(equity_curve)
        cost_on = float(equity_curve[-1] - 1.0) if isinstance(equity_curve, list) and len(equity_curve) >= 1 else float(res.get("total_return", 0.0) or 0.0)

        results.append({
            "threshold": float(thr),
            "pass_rate": float(pass_rates[float(thr)]),
            "entries_attempted": entries_attempted,
            "trades": total_trades,
            "win_rate": win_rate,
            "mean_return": mean_return,
            "cost_on": cost_on,
        })

    # Print STEP 2 table in requested style
    print("\n## Boundary Threshold Table (0.61~0.64, fixed t_max + recent 60d)", flush=True)
    print("| threshold | pass_rate | entries_attempted | trades | win_rate | mean_return | cost_on |", flush=True)
    print("|---:|---:|---:|---:|---:|---:|---:|", flush=True)
    for r in results:
        print(
            f"| {r['threshold']:.2f} | {r['pass_rate']:.4f} | {r['entries_attempted']} | {r['trades']} | {r['win_rate']:.4f} | {r['mean_return']:.8f} | {r['cost_on']:.8f} |",
            flush=True,
        )

    # STEP 3: boundary decision (operational)
    alive = [r for r in results if int(r["trades"]) > 0]
    alive_thresholds = sorted([float(r["threshold"]) for r in alive])

    if not alive_thresholds:
        print("\nDecision:", flush=True)
        print("상향 불가: 0.60보다 큰 임시 threshold에서는 trades가 살아나지 않습니다.", flush=True)
        print("운영 임시 threshold: 0.60", flush=True)
        next_update_point = "경계 구간(0.60~0.65)에서 0.60 유지 + exit/손익 구조 개선 검토"
    else:
        last_alive = max(alive_thresholds)
        print("\nDecision:", flush=True)
        print(f"상향 가능: 0.60보다 큰 임시 threshold 중 trades가 살아있는 값들이 존재합니다.", flush=True)
        print(f"마지막 생존 상단 경계: {last_alive:.2f}", flush=True)
        print("운영 임시 threshold 후보:", flush=True)
        print(f"- 보수적 선택: 0.60", flush=True)
        print(f"- 경계 탐색 기반: {last_alive:.2f}", flush=True)
        next_update_point = "운영상 trade 수/성과 재검증 + 상위 경계 미세탐색(예: 0.60~마지막생존 사이)"

    print(f"next update point: {next_update_point}", flush=True)

    # STEP 4/5: trade breakdown for the band (use threshold=0.60)
    # Dump trades once for threshold=0.60.
    from src.backtest.ml_backtest_engines import LstmAttnBacktestEngine  # noqa: F401
    from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine
    from src.strategies.ml_thresholds import resolve_ml_thresholds

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

    dump_csv = PROJECT_ROOT / "data" / "backtest_dumps" / "trades_boundary_validation_20260320_0740_thr_0p60.csv"

    print(f"\n[TRADE DUMP] threshold=0.60 dump to: {dump_csv}", flush=True)
    res0, err0 = engine.run_backtest(
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
        min_max_proba=0.60,
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
    if err0:
        raise RuntimeError(f"trade dump backtest failed: {err0}")
    # 분석 출력
    _dump_and_analyze_trades(expected_trades=10, trades_dump_csv=dump_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

