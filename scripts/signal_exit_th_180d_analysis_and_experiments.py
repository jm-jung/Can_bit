#!/usr/bin/env python3
"""
180d / threshold=0.60 / emit_trade_log=True 기준 signal_exit_th 전용 분석 + EXP_A/B 최소 실험.

- STEP 1~2: signal_exit_th 청산만 추출, proba·가정 청산(hypothetical) 분석
- STEP 4: BASELINE(W=1, delta=None) vs EXP_A(W>1) vs EXP_B(Δproba)

주의: early_exit / time_stop / SHORT 전략 로직은 변경하지 않음 (엔진 기본 파라미터 동일).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _to_utc_ts(x: Any) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _cost_on(res: dict[str, Any]) -> float:
    ec = res.get("equity_curve") or []
    if isinstance(ec, list) and len(ec) >= 1:
        return float(ec[-1] - 1.0)
    return float(res.get("total_return", 0.0) or 0.0)


def _deduped_trades(res: dict[str, Any]) -> list[Any]:
    from src.backtest.engine import dedupe_trades_round_trips

    trades = list(res.get("trades") or [])
    return dedupe_trades_round_trips(trades)


def _mean_profit_roundtrip(res: dict[str, Any]) -> float | None:
    trades = _deduped_trades(res)
    if not trades:
        return None
    vals = [float(t["profit"]) for t in trades if t.get("profit") is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _max_loss_trade(res: dict[str, Any]) -> float | None:
    trades = _deduped_trades(res)
    if not trades:
        return None
    vals = [float(t["profit"]) for t in trades if t.get("profit") is not None]
    if not vals:
        return None
    return float(min(vals))


def _hypothetical_profit(
    *,
    entry_price: float,
    exit_price: float,
    direction: str,
    commission: float,
    slippage: float,
) -> float:
    fee_rate = commission + slippage
    entry_cost = entry_price * fee_rate
    exit_cost = exit_price * fee_rate
    if direction == "LONG":
        effective_entry = entry_price + entry_cost
        effective_exit = exit_price - exit_cost
        return (effective_exit - effective_entry) / effective_entry
    if direction == "SHORT":
        effective_entry = entry_price - entry_cost
        effective_exit = exit_price + exit_cost
        return (effective_entry - effective_exit) / effective_entry
    return 0.0


def _run_window(
    *,
    df_bt: pd.DataFrame,
    pl_primary: np.ndarray,
    ps_primary: np.ndarray,
    t_max: pd.Timestamp,
    window_days: int,
    threshold: float,
    emit_trade_log: bool,
    signal_exit_th_trailing_window_bars: int = 1,
    signal_exit_th_delta_from_entry: float | None = None,
) -> dict[str, Any]:
    from scripts.run_tcn_label_sweep_v2 import run_backtest_7d

    window_start = t_max - pd.Timedelta(days=window_days)
    t = pd.to_datetime(df_bt["timestamp"], utc=True)
    mask = (t >= window_start) & (t <= t_max)
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
        signal_exit_th_trailing_window_bars=signal_exit_th_trailing_window_bars,
        signal_exit_th_delta_from_entry=signal_exit_th_delta_from_entry,
    )
    if err:
        raise RuntimeError(f"run_backtest_7d failed (window={window_days}, thr={threshold}): {err}")
    if res is None:
        res = {}
    return {
        "window_days": window_days,
        "threshold": threshold,
        "rows": len(df_w),
        "df_w": df_w,
        "pl_w": pl_w,
        "ps_w": ps_w,
        "result": res,
    }


def _exit_events_df(res: dict[str, Any]) -> pd.DataFrame:
    events = list(res.get("trade_events") or [])
    exit_events = [e for e in events if str(e.get("event", "")).startswith("EXIT")]
    if not exit_events:
        return pd.DataFrame()
    return pd.DataFrame(exit_events)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-max", default="2026-03-20T07:40:00+00:00")
    parser.add_argument("--days-full", type=int, default=182)
    parser.add_argument("--window-days", type=int, default=180)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--exp-a-w", type=int, default=8, help="EXP_A: signal_exit_th_trailing_window_bars")
    parser.add_argument(
        "--exp-b-delta",
        type=float,
        default=0.08,
        help="EXP_B: signal_exit_th_delta_from_entry (Δproba < -x)",
    )
    parser.add_argument(
        "--out-md",
        default="data/diagnostics/fr2/SIGNAL_EXIT_TH_180D_ANALYSIS_REPORT.md",
    )
    args = parser.parse_args()

    from scripts.run_fr2_diagnostics import MODELS_DIR, get_ohlcv_and_proba
    from scripts.run_fr2_regime_conditioning import add_regime_columns
    from src.strategies.ensemble_strategy import EnsembleInputs, build_ensemble_proba, build_fr2_c4_mask

    t_max = _to_utc_ts(args.t_max)
    end_date_utc_str = t_max.strftime("%Y-%m-%d")
    DAYS_FULL = int(args.days_full)

    base_pt = MODELS_DIR / "tcn_h15_t0p004.pt"
    fr2_pt = MODELS_DIR / "tcn_h15_micro_v1.pt"

    print(f"[signal_exit_th] t_max={t_max.isoformat()} days_full={DAYS_FULL} window={args.window_days}", flush=True)
    triple_base, err_b = get_ohlcv_and_proba(DAYS_FULL, base_pt, "base", False, end_date=end_date_utc_str)
    if err_b or triple_base is None:
        raise RuntimeError(f"get_ohlcv_and_proba(base) failed: {err_b}")
    df_b, pl_b, ps_b, _ = triple_base
    triple_fr2, err_f = get_ohlcv_and_proba(DAYS_FULL, fr2_pt, "microstructure_v1", True, end_date=end_date_utc_str)
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

    df_bt_reg = add_regime_columns(DAYS_FULL, df_bt.copy())
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

    thr = float(args.threshold)
    out_base = _run_window(
        df_bt=df_bt,
        pl_primary=pl_primary,
        ps_primary=ps_primary,
        t_max=t_max,
        window_days=int(args.window_days),
        threshold=thr,
        emit_trade_log=True,
        signal_exit_th_trailing_window_bars=1,
        signal_exit_th_delta_from_entry=None,
    )
    res = out_base["result"]
    df_w = out_base["df_w"]
    pl_w = out_base["pl_w"]
    ps_w = out_base["ps_w"]

    exits = _exit_events_df(res)
    sig = exits[exits["exit_reason"].astype(str) == "signal_exit_th"].copy()
    n_sig = len(sig)

    lines: list[str] = []
    lines.append("# signal_exit_th 180d 분석·실험 리포트\n")
    lines.append(f"- t_max: `{t_max.isoformat()}`\n")
    lines.append(f"- window_days: {args.window_days}, min_max_proba(threshold): {thr}, emit_trade_log: True\n")
    lines.append("\n## STEP 1 — signal_exit_th 트레이드\n\n")

    if n_sig == 0:
        lines.append("signal_exit_th 청산이 없습니다.\n")
    else:
        prof = pd.to_numeric(sig["profit"], errors="coerce")
        lines.append(f"1. 개수: **{n_sig}**\n")
        lines.append(f"2. 평균 profit: {prof.mean():.6f}, 중앙값: {prof.median():.6f}, 합계: {prof.sum():.6f}\n")
        lines.append("\n3. 트레이드별 상세는 아래 표 및 CSV.\n")

    close_arr = df_w["close"].to_numpy(dtype=float)

    rows_csv: list[dict[str, Any]] = []
    hypo_summary: dict[int, dict[str, Any]] = {}

    COMMISSION = 0.0009
    SLIPPAGE = 0.0001

    for N in (4, 8):
        hypo_summary[N] = {"hypos": [], "actual": []}

    for _, r in sig.iterrows():
        exit_idx = int(r["idx"])
        bh = int(r["bars_held"]) if pd.notna(r.get("bars_held")) else 0
        entry_idx = exit_idx - bh
        direction = str(r.get("direction", ""))
        entry_ts = r.get("entry_ts")
        exit_ts = r.get("exit_ts")
        ep = float(r.get("entry_price", np.nan))
        xp = float(r.get("exit_price", np.nan))

        def _dir_proba(ii: int) -> float:
            ii = max(0, min(ii, len(pl_w) - 1))
            if direction == "LONG":
                return float(pl_w[ii])
            if direction == "SHORT":
                return float(ps_w[ii])
            return float("nan")

        proba_entry = _dir_proba(entry_idx)
        proba_exit = _dir_proba(exit_idx)
        proba_prev = _dir_proba(exit_idx - 1) if exit_idx > 0 else proba_exit

        row = {
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "direction": direction,
            "bars_held": bh,
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "proba_entry": proba_entry,
            "proba_exit_prev": proba_prev,
            "proba_exit": proba_exit,
            "profit_actual": float(r.get("profit", np.nan)),
            "entry_price": ep,
            "exit_price": xp,
        }
        actual_pf = float(r.get("profit", np.nan))
        for N in (4, 8):
            j = exit_idx - N
            if j < entry_idx:
                hypo = float("nan")
            else:
                early_px = float(close_arr[j])
                hypo = _hypothetical_profit(
                    entry_price=ep,
                    exit_price=early_px,
                    direction=direction,
                    commission=COMMISSION,
                    slippage=SLIPPAGE,
                )
            row[f"hypo_profit_N{N}"] = hypo
            if np.isfinite(hypo) and np.isfinite(actual_pf):
                hypo_summary[N]["hypos"].append(hypo)
                hypo_summary[N]["actual"].append(actual_pf)
        rows_csv.append(row)

    if rows_csv:
        df_out = pd.DataFrame(rows_csv)
        csv_path = PROJECT_ROOT / "data/diagnostics/fr2/signal_exit_th_trades_180d_detail.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(csv_path, index=False)
        lines.append(f"- 상세 CSV: `{csv_path.relative_to(PROJECT_ROOT)}`\n")
        show = df_out[
            [
                "entry_ts",
                "exit_ts",
                "direction",
                "bars_held",
                "proba_entry",
                "proba_exit_prev",
                "proba_exit",
                "profit_actual",
            ]
        ].copy()
        lines.append("\n### 트레이드별 표 (STEP 1)\n\n")
        try:
            lines.append(show.to_markdown(index=False))
        except ImportError:
            lines.append("```\n" + show.to_string(index=False) + "\n```\n")
        lines.append("\n")

    lines.append("\n## STEP 2 — N bars 일찍 청산 가정 (가격=해당 바 close)\n\n")
    for N in (4, 8):
        hs = hypo_summary[N]["hypos"]
        ac = hypo_summary[N]["actual"]
        if not hs:
            lines.append(f"### N={N}: 데이터 없음\n\n")
            continue
        h_mean = float(np.mean(hs))
        a_mean = float(np.mean(ac))
        improved = sum(1 for h, a in zip(hs, ac) if h > a)
        pct = 100.0 * improved / len(hs)
        lines.append(f"### N={N}\n\n")
        lines.append(f"- hypothetical profit 평균: **{h_mean:.6f}**\n")
        lines.append(f"- 실제(signal_exit) profit 평균: **{a_mean:.6f}**\n")
        lines.append(f"- 평균 기준 개선 여부: {'예' if h_mean > a_mean else '아니오'}\n")
        lines.append(f"- 개선된 트레이드 비율: **{pct:.1f}%** ({improved}/{len(hs)})\n\n")

    lines.append("## STEP 3 — 최소 실험 설계\n\n")
    lines.append(
        "- **EXP_A**: `signal_exit_th_trailing_window_bars = W` (W>1). "
        "최근 W바 중 방향 proba의 최소값이 exit 임계 미만이면 청산 → "
        "min_hold 직후 일시적 회복 구간에서도 약세 ‘기억’ 청산.\n"
    )
    lines.append(
        f"- **EXP_B**: `signal_exit_th_delta_from_entry = x` (예: {args.exp_b_delta}). "
        "진입 시점 방향 proba 대비 \\(\\Delta proba < -x\\) 이면 `signal_exit_th`.\n"
    )
    lines.append("- early_exit / time_stop / SHORT 로직·파라미터는 그대로.\n\n")

    lines.append("## STEP 4 — 180d BASELINE vs EXP_A vs EXP_B\n\n")

    exp_a = _run_window(
        df_bt=df_bt,
        pl_primary=pl_primary,
        ps_primary=ps_primary,
        t_max=t_max,
        window_days=int(args.window_days),
        threshold=thr,
        emit_trade_log=False,
        signal_exit_th_trailing_window_bars=int(args.exp_a_w),
        signal_exit_th_delta_from_entry=None,
    )["result"]
    exp_b = _run_window(
        df_bt=df_bt,
        pl_primary=pl_primary,
        ps_primary=ps_primary,
        t_max=t_max,
        window_days=int(args.window_days),
        threshold=thr,
        emit_trade_log=False,
        signal_exit_th_trailing_window_bars=1,
        signal_exit_th_delta_from_entry=float(args.exp_b_delta),
    )["result"]

    def _row(name: str, r: dict[str, Any]) -> str:
        tr = float(r.get("total_return", 0.0) or 0.0)
        mpr = _mean_profit_roundtrip(r)
        co = _cost_on(r)
        mlt = _max_loss_trade(r)
        mpr_s = f"{mpr:.6f}" if mpr is not None else "N/A"
        mlt_s = f"{mlt:.6f}" if mlt is not None else "N/A"
        return (
            f"| {name} | {tr:.6f} | {mpr_s} | {co:.6f} | {mlt_s} |\n"
        )

    lines.append("| 구분 | total_return | mean_profit_roundtrip | cost_on | max_loss_trade |\n")
    lines.append("|------|-------------:|------------------------:|--------:|---------------:|\n")
    lines.append(_row("BASELINE (signal_exit W=1, Δ=off)", res))
    lines.append(_row(f"EXP_A (signal_exit W={args.exp_a_w}, Δ=off)", exp_a))
    lines.append(_row(f"EXP_B (signal_exit W=1, Δ={args.exp_b_delta})", exp_b))

    lines.append("\n## STEP 5 — 결론 (데이터 기반)\n\n")
    for N in (4, 8):
        hs = hypo_summary[N]["hypos"]
        ac = hypo_summary[N]["actual"]
        if hs and float(np.mean(hs)) > float(np.mean(ac)):
            lines.append(
                f"- N={N}: 가정 청산이 평균적으로 유리 → **늦게 청산되는 케이스가 존재**할 수 있음.\n"
            )
        elif hs:
            lines.append(
                f"- N={N}: 가정 청산이 평균적으로 불리 또는 동일 → **단순 N바 앞당김**만으로는 개선 근거가 약함.\n"
            )
    lines.append(
        f"- **EXP_A (W={args.exp_a_w})** 가 total_return 기준 BASELINE 대비 개선되는지: "
        f"표 STEP 4 참고.\n"
    )
    lines.append(
        f"- **EXP_B (Δ={args.exp_b_delta})** 도 동일.\n"
    )

    out_path = PROJECT_ROOT / args.out_md
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"[signal_exit_th] wrote {out_path}", flush=True)
    print("".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
