#!/usr/bin/env python3
"""
빠른 버전: latest t_max 기준 최근 60일 구간 threshold 스윕(0.40~0.65).

기존 스윕이 오래 걸릴 수 있어 DAYS_FULL만 줄이고, 단계별 print(flush=True)로 진행상황을 보이게 합니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_meta_source_latest() -> dict[str, Any]:
    p = PROJECT_ROOT / "data" / "diagnostics" / "fr2" / "meta_metrics_source_latest.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _to_utc_ts(x: Any) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _mean_return_and_sharpe(equity_curve: list[float] | None) -> tuple[float, float]:
    """Legacy: adjacent equity-point simple returns; insufficient steps -> (0, 0)."""
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


def equity_step_metrics(
    equity_curve: list[float] | None,
) -> tuple[str, int, float | None]:
    """
    Returns (equity_step_status, equity_step_count, equity_step_mean_return or None).
    status: ok | insufficient_steps | empty
    """
    ec = equity_curve if isinstance(equity_curve, list) else []
    n = len(ec)
    equity_step_count = max(0, n - 1)
    if n == 0:
        return "empty", equity_step_count, None
    if n < 2:
        return "insufficient_steps", equity_step_count, None
    arr = np.asarray(ec, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return "insufficient_steps", equity_step_count, None
    rets = np.diff(arr) / arr[:-1]
    if len(rets) == 0:
        return "insufficient_steps", equity_step_count, None
    return "ok", equity_step_count, float(np.mean(rets))


def main() -> int:
    meta = _load_meta_source_latest()
    src_info = meta.get("source_info") or {}

    symbol = src_info.get("symbol", "BTCUSDT")
    timeframe = src_info.get("timeframe", "5m")

    # ✅ 고정 t_max (이전 지정값)
    # 사용자가 요구한 "t_max 변경 금지"를 위해 meta_metrics_source_latest에서 읽은 값과 무관하게 강제 사용.
    t_max = _to_utc_ts("2026-03-20T07:40:00+00:00")
    end_date_utc_str = t_max.strftime("%Y-%m-%d")

    window_days = 60
    window_start = t_max - pd.Timedelta(days=window_days)

    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    # Team policy: 운영 기본값은 자동탐지 결과와 별개로 고정 유지
    TEAM_RECOMMENDED_OPERATING_THRESHOLD = 0.60

    from scripts.run_fr2_diagnostics import MODELS_DIR, get_ohlcv_and_proba
    from scripts.run_fr2_regime_conditioning import add_regime_columns
    from src.strategies.ensemble_strategy import EnsembleInputs, build_ensemble_proba, build_fr2_c4_mask
    from scripts.run_tcn_label_sweep_v2 import run_backtest_7d
    from src.backtest.engine import dedupe_trades_round_trips

    BASELINE_PT = MODELS_DIR / "tcn_h15_t0p004.pt"
    FR2_PT = MODELS_DIR / "tcn_h15_micro_v1.pt"

    # 핵심 단축 포인트
    # 60d window 외에 TCN window_size/horizon 및 C4 regime 계산에 필요한 최소 여유만 확보
    DAYS_FULL = 62

    print(f"[FAST] t_max={t_max.isoformat()}", flush=True)
    print(f"[FAST] days_full={DAYS_FULL}, window=[{window_start.isoformat()} .. {t_max.isoformat()}]", flush=True)
    print("[FAST] loading base proba...", flush=True)
    triple_base, err_b = get_ohlcv_and_proba(DAYS_FULL, BASELINE_PT, "base", False, end_date=end_date_utc_str)
    if err_b or triple_base is None:
        raise RuntimeError(f"get_ohlcv_and_proba(base) failed: {err_b}")
    df_b, pl_b, ps_b, _ = triple_base
    print(f"[FAST] base done: rows={len(df_b)}", flush=True)

    print("[FAST] loading fr2 proba...", flush=True)
    triple_fr2, err_f = get_ohlcv_and_proba(DAYS_FULL, FR2_PT, "microstructure_v1", True, end_date=end_date_utc_str)
    if err_f or triple_fr2 is None:
        raise RuntimeError(f"get_ohlcv_and_proba(fr2) failed: {err_f}")
    df_f, pl_f, ps_f, _ = triple_fr2
    print(f"[FAST] fr2 done: rows={len(df_f)}", flush=True)

    if not (len(df_b) == len(pl_b) == len(ps_b)):
        raise RuntimeError("base length mismatch")
    if not (len(df_f) == len(pl_f) == len(ps_f)):
        raise RuntimeError("fr2 length mismatch")

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

    print(f"[FAST] joined aligned rows={len(df_bt)}", flush=True)

    print("[FAST] add regimes + build C4 mask...", flush=True)
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
    print("[FAST] build ensemble proba...", flush=True)
    pl_primary, ps_primary = build_ensemble_proba(ensemble_inputs, mode="override")

    # Slice window
    t = pd.to_datetime(df_bt["timestamp"], utc=True)
    t_max_utc = _to_utc_ts(t_max)
    mask = (t >= window_start) & (t <= t_max_utc)
    df_60 = df_bt.loc[mask].reset_index(drop=True)
    pl_60 = np.asarray(pl_primary, dtype=np.float32)[mask.to_numpy()]
    ps_60 = np.asarray(ps_primary, dtype=np.float32)[mask.to_numpy()]

    print(f"[FAST] window rows={len(df_60)}", flush=True)

    # Backtest fixed params
    COMMISSION = 0.0009
    SLIPPAGE = 0.0001
    MAX_ENTROPY = 1.30
    MIN_HOLD = 24
    COOLDOWN = 24
    TIME_STOP_BARS = 72
    EARLY_EXIT_BAD_K = 8

    rows: list[dict[str, Any]] = []
    for thr in thresholds:
        print(f"[FAST] backtest thr={thr} ...", flush=True)
        res, err = run_backtest_7d(
            symbol,
            timeframe,
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

        total_trades = int(res.get("total_trades", 0) or 0)
        entries_attempted = int(res.get("entries_attempted", 0) or 0)
        win_rate = float(res.get("win_rate", 0.0) or 0.0)
        mean_return, sharpe = _mean_return_and_sharpe(res.get("equity_curve"))

        # cost_on이 명시적으로 없을 수 있어, equity_curve 최종값 기준으로 총수익률을 대체값으로 사용
        # (equity_curve는 백테스트 누적 자산이며, total_return과 동일한 의미로 쓰이는 경우가 많음)
        equity_curve = res.get("equity_curve") or []
        if isinstance(equity_curve, list) and len(equity_curve) >= 1:
            cost_on = float(equity_curve[-1] - 1.0)
        else:
            cost_on = float(res.get("total_return", 0.0) or 0.0)

        equity_step_status, equity_step_count, equity_step_mean_return = equity_step_metrics(
            equity_curve if isinstance(equity_curve, list) else []
        )

        trades_list = list(res.get("trades") or [])
        deduped = dedupe_trades_round_trips(trades_list)
        unique_round_trips = int(res.get("unique_round_trips", len(deduped)))
        duplicate_trade_rows = int(res.get("duplicate_trade_rows", len(trades_list) - unique_round_trips))
        profits_u = [float(t["profit"]) for t in deduped if t.get("profit") is not None]
        mean_profit_roundtrip = float(np.mean(profits_u)) if profits_u else None

        rows.append({
            "threshold": float(thr),
            "entries_attempted": entries_attempted,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "mean_return": mean_return,
            "sharpe": sharpe,
            "cost_on": cost_on,
            "equity_step_status": equity_step_status,
            "equity_step_count": equity_step_count,
            "equity_step_mean_return": equity_step_mean_return,
            "unique_round_trips": unique_round_trips,
            "duplicate_trade_rows": duplicate_trade_rows,
            "mean_profit_roundtrip": mean_profit_roundtrip,
        })

    alive = [r for r in rows if int(r["total_trades"]) > 0]
    min_alive_thr = min(alive, key=lambda r: r["threshold"])["threshold"] if alive else None
    max_alive_thr = max(alive, key=lambda r: r["threshold"])["threshold"] if alive else None

    # STEP 4 sanity check
    all_trades_zero = all(int(r["total_trades"]) == 0 for r in rows)
    any_entries_attempted = any(int(r["entries_attempted"]) > 0 for r in rows)

    # STEP 3 CASE conclusion (수치 기반 규칙)
    if all_trades_zero or not any_entries_attempted:
        case = "CASE C"
        recommended_thr = None
        alive_count = 0
    else:
        case_alive_thresholds = sorted([r for r in rows if int(r["total_trades"]) > 0], key=lambda x: x["threshold"])
        min_row = min(case_alive_thresholds, key=lambda x: x["threshold"])
        min_alive_thr = float(min_row["threshold"])

        alive_count = len(case_alive_thresholds)
        recommended_thr = min_alive_thr

        # "정상 threshold 문제" 판정: trade가 살아난 뒤에도(상대적으로) 성과가 급격히 무너지는지로 판단
        best_mean = max(r["mean_return"] for r in case_alive_thresholds)
        mean_at_min = float(min_row["mean_return"])
        win_at_min = float(min_row["win_rate"])

        # trade가 여러 threshold에서 살아나면 CASE A에 더 가깝게
        if alive_count >= 2 and mean_at_min >= (best_mean * 0.6) and win_at_min >= 0.45:
            case = "CASE A"
        else:
            case = "CASE B"

    # STEP 2 출력: 레거시 한 줄 + 운영 해석용 컬럼
    print("\nthreshold | entries | trades | win_rate | mean_return | sharpe | cost_on", flush=True)
    print("--------------------------------------------------------------------------------", flush=True)
    for r in rows:
        esmr = r.get("equity_step_mean_return")
        esmr_s = f"{float(esmr):.8f}" if esmr is not None else "N/A"
        mpr = r.get("mean_profit_roundtrip")
        mpr_s = f"{float(mpr):.8f}" if mpr is not None else "N/A"
        print(
            f"{r['threshold']:.2f}      | {int(r['entries_attempted'])}   | {int(r['total_trades'])}   | "
            f"{float(r['win_rate']):.4f}     | {float(r['mean_return']):.8f} | {float(r['sharpe']):.8f} | {float(r['cost_on']):.8f}",
            flush=True,
        )
        print(
            f"  -> equity_step: status={r.get('equity_step_status')} count={int(r.get('equity_step_count', 0))} "
            f"step_mean={esmr_s} | unique_rt={int(r.get('unique_round_trips', 0))} dup_rows={int(r.get('duplicate_trade_rows', 0))} "
            f"mean_profit_rt={mpr_s}",
            flush=True,
        )

    # STEP 3 결론 출력 (자동 탐지 vs 운영 정책 분리)
    if case == "CASE C":
        print("\nCASE C — 완전 비활성", flush=True)
        print("자동 탐지 추천 threshold: NONE", flush=True)
        print(f"근거: total_trades=0 for all threshold (entries_attempted_any={any_entries_attempted})", flush=True)
    else:
        print(f"\n{case}", flush=True)
        print(f"trade가 살아나는 최소 threshold: {min_alive_thr:.2f}", flush=True)
        min_row = min([r for r in rows if int(r["total_trades"]) > 0], key=lambda x: x["threshold"])
        print(f"자동 탐지 추천 threshold: {float(min_row['threshold']):.2f}", flush=True)
        print(
            "근거: trade count="
            f"{int(min_row['total_trades'])}, win_rate={float(min_row['win_rate']):.4f}, "
            f"mean_return={float(min_row['mean_return']):.8f}, cost_on={float(min_row['cost_on']):.8f}",
            flush=True,
        )

    auto_detected_boundary_threshold = float(min_alive_thr) if min_alive_thr is not None else None
    lowest_alive_threshold = float(min_alive_thr) if min_alive_thr is not None else None
    highest_alive_threshold = float(max_alive_thr) if max_alive_thr is not None else None
    recommended_operating_threshold = TEAM_RECOMMENDED_OPERATING_THRESHOLD
    note = (
        "recommended_operating_threshold is team policy and intentionally decoupled from auto detection."
    )

    print("\n[OPERATING INTERPRETATION]", flush=True)
    print(
        f"auto_detected_boundary_threshold: {auto_detected_boundary_threshold if auto_detected_boundary_threshold is not None else 'N/A'}",
        flush=True,
    )
    print(
        f"lowest_alive_threshold: {lowest_alive_threshold if lowest_alive_threshold is not None else 'N/A'}",
        flush=True,
    )
    print(
        f"highest_alive_threshold: {highest_alive_threshold if highest_alive_threshold is not None else 'N/A'}",
        flush=True,
    )
    print(f"recommended_operating_threshold: {recommended_operating_threshold:.2f}", flush=True)
    print(f"note: {note}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

