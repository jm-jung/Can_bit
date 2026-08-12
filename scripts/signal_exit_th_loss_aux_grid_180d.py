#!/usr/bin/env python3
"""
180d / threshold=0.60 고정 — signal_exit_th 손실형 선별 보조(후보1) 그리드 실험.

보조 조건 (엔진): (proba_entry - proba_prev_bar) > δ_p AND MAE < mae_cut 일 때
임계/트레일링 기반 signal_exit_th 에서만 직전 바 close 로 청산가 조정.

baseline 대비 early_exit / time_stop / threshold / SHORT ON-OFF / delta_from_entry 는 변경하지 않음.
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


def _run_one(
    *,
    df_bt: pd.DataFrame,
    pl_primary: np.ndarray,
    ps_primary: np.ndarray,
    t_max: pd.Timestamp,
    window_days: int,
    threshold: float,
    emit_trade_log: bool,
    signal_exit_th_loss_aux_delta_p: float | None,
    signal_exit_th_loss_aux_mae_cut: float | None,
    signal_exit_th_loss_aux_short_only: bool = False,
    signal_exit_th_loss_aux_min_bars: int | None = None,
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
        signal_exit_th_trailing_window_bars=1,
        signal_exit_th_delta_from_entry=None,
        signal_exit_th_loss_aux_delta_p=signal_exit_th_loss_aux_delta_p,
        signal_exit_th_loss_aux_mae_cut=signal_exit_th_loss_aux_mae_cut,
        signal_exit_th_loss_aux_short_only=signal_exit_th_loss_aux_short_only,
        signal_exit_th_loss_aux_min_bars=signal_exit_th_loss_aux_min_bars,
    )
    if err:
        raise RuntimeError(f"run_backtest_7d failed: {err}")
    if res is None:
        res = {}
    return res


def _metrics_row(name: str, res: dict[str, Any]) -> dict[str, Any]:
    ut = res.get("unique_round_trips")
    wr = res.get("win_rate")
    return {
        "experiment": name,
        "unique_round_trips": int(ut) if ut is not None else None,
        "win_rate": float(wr) if wr is not None else None,
        "mean_profit_roundtrip": _mean_profit_roundtrip(res),
        "cost_on": _cost_on(res),
        "total_return": float(res.get("total_return", 0.0) or 0.0),
        "max_loss_trade": _max_loss_trade(res),
        "signal_exit_th_count": int(res.get("signal_exit_th_count", 0) or 0),
        "signal_exit_th_total_profit": float(res.get("signal_exit_th_total_profit", 0.0) or 0.0),
        "signal_exit_th_loss_aux_applied": int(res.get("signal_exit_th_loss_aux_applied", 0) or 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-max", default="2026-03-20T07:40:00+00:00")
    parser.add_argument("--days-full", type=int, default=182)
    parser.add_argument("--window-days", type=int, default=180)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument(
        "--out-md",
        default="data/diagnostics/fr2/SIGNAL_EXIT_TH_LOSS_AUX_GRID_180D_REPORT.md",
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

    print(f"[loss_aux_grid] t_max={t_max.isoformat()} days_full={DAYS_FULL} window={args.window_days}", flush=True)
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

    delta_ps = [0.05, 0.08, 0.10]
    mae_cuts = [-0.003, -0.005, -0.007]

    rows: list[dict[str, Any]] = []

    print("[loss_aux_grid] baseline...", flush=True)
    base_res = _run_one(
        df_bt=df_bt,
        pl_primary=pl_primary,
        ps_primary=ps_primary,
        t_max=t_max,
        window_days=args.window_days,
        threshold=args.threshold,
        emit_trade_log=False,
        signal_exit_th_loss_aux_delta_p=None,
        signal_exit_th_loss_aux_mae_cut=None,
    )
    rows.append(_metrics_row("baseline", base_res))

    for dp in delta_ps:
        for mc in mae_cuts:
            tag = f"δp={dp} mae_cut={mc}"
            print(f"[loss_aux_grid] {tag} ...", flush=True)
            r = _run_one(
                df_bt=df_bt,
                pl_primary=pl_primary,
                ps_primary=ps_primary,
                t_max=t_max,
                window_days=args.window_days,
                threshold=args.threshold,
                emit_trade_log=False,
                signal_exit_th_loss_aux_delta_p=dp,
                signal_exit_th_loss_aux_mae_cut=mc,
            )
            rows.append(_metrics_row(tag, r))

    base = rows[0]
    tr_base = float(base["total_return"])
    improved = [r for r in rows[1:] if float(r["total_return"]) > tr_base]

    # 정렬: total_return desc
    ranked = sorted(rows[1:], key=lambda x: float(x["total_return"] or 0.0), reverse=True)
    best_one = ranked[0] if ranked else None
    best_two = ranked[:2] if len(ranked) >= 2 else ranked

    ext_rows: list[dict[str, Any]] = []
    if best_two:
        for r in best_two:
            name = str(r["experiment"])
            # "δp=0.08 mae_cut=-0.005" parse
            parts = name.replace("δp=", "").split(" mae_cut=")
            if len(parts) == 2:
                dp_f = float(parts[0])
                mc_f = float(parts[1])
            else:
                continue
            # '|' 는 MD 표 셀 구분자와 충돌하므로 사용하지 않음
            ext_name = f"{name} · SHORT_only · bars>=24"
            print(f"[loss_aux_grid] extension {ext_name} ...", flush=True)
            er = _run_one(
                df_bt=df_bt,
                pl_primary=pl_primary,
                ps_primary=ps_primary,
                t_max=t_max,
                window_days=args.window_days,
                threshold=args.threshold,
                emit_trade_log=False,
                signal_exit_th_loss_aux_delta_p=dp_f,
                signal_exit_th_loss_aux_mae_cut=mc_f,
                signal_exit_th_loss_aux_short_only=True,
                signal_exit_th_loss_aux_min_bars=24,
            )
            ext_rows.append(_metrics_row(ext_name, er))

    # Markdown
    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, float):
            return f"{x:.6f}"
        return str(x)

    lines: list[str] = []
    lines.append("# SIGNAL_EXIT_TH 손실 보조(후보1) 180d 그리드 + 확장 실험")
    lines.append("")
    lines.append(f"- t_max: `{t_max.isoformat()}`")
    lines.append(f"- window_days: **{args.window_days}**")
    lines.append(f"- threshold (min_max_proba): **{args.threshold}**")
    lines.append(f"- δ_p ∈ {delta_ps}, mae_cut ∈ {mae_cuts}")
    lines.append("")
    lines.append("## A. 9개 조합 + baseline 비교 표")
    lines.append("")
    lines.append(
        "| experiment | unique_round_trips | win_rate | mean_profit_roundtrip | cost_on | total_return | "
        "max_loss_trade | signal_exit_th count | signal_exit_th total_profit | aux_applied |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        lines.append(
            f"| {r['experiment']} | {r['unique_round_trips']} | {fmt(r['win_rate'])} | "
            f"{fmt(r['mean_profit_roundtrip'])} | {fmt(r['cost_on'])} | {fmt(r['total_return'])} | "
            f"{fmt(r['max_loss_trade'])} | {r['signal_exit_th_count']} | {fmt(r['signal_exit_th_total_profit'])} | "
            f"{r['signal_exit_th_loss_aux_applied']} |"
        )
    lines.append("")
    lines.append("## B. 가장 좋은 조합 1~2개 (total_return 기준, 그리드만)")
    lines.append("")
    if best_one:
        lines.append(f"- **1위:** `{best_one['experiment']}` — total_return={fmt(best_one['total_return'])}")
    if len(best_two) > 1:
        lines.append(f"- **2위:** `{best_two[1]['experiment']}` — total_return={fmt(best_two[1]['total_return'])}")
    lines.append("")
    lines.append("## C. 확장 실험 (상위 1~2개 × SHORT 전용 × bars_held >= 24)")
    lines.append("")
    if not ext_rows:
        lines.append("(그리드 결과 없음 또는 파싱 실패)")
    else:
        lines.append(
            "| experiment | unique_round_trips | win_rate | mean_profit_roundtrip | cost_on | total_return | "
            "max_loss_trade | signal_exit_th count | signal_exit_th total_profit | aux_applied |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in ext_rows:
            lines.append(
                f"| {r['experiment']} | {r['unique_round_trips']} | {fmt(r['win_rate'])} | "
                f"{fmt(r['mean_profit_roundtrip'])} | {fmt(r['cost_on'])} | {fmt(r['total_return'])} | "
                f"{fmt(r['max_loss_trade'])} | {r['signal_exit_th_count']} | {fmt(r['signal_exit_th_total_profit'])} | "
                f"{r['signal_exit_th_loss_aux_applied']} |"
            )
    lines.append("")
    lines.append("## D. 운영 결론")
    lines.append("")
    lines.append("### 비교 질문")
    lines.append("")
    imp_list = ", ".join(f"`{x['experiment']}`" for x in improved) if improved else "없음"
    lines.append(
        f"1. baseline 대비 total_return 개선 조합: **{len(improved)}**개 ({imp_list})"
    )
    lines.append("")
    lines.append("2. 개선이 있을 때 max_loss_trade·signal_exit_th total_profit 동반 개선 여부는 위 표에서 행 단위 비교.")
    lines.append("")
    lines.append(
        "3. trade 수( unique_round_trips ) 훼솰: baseline과 ± 몇 % 이내인지 비교 권장."
    )
    lines.append("")
    lines.append("### 요약")
    lines.append("")
    if improved:
        lines.append(
            f"- **실제 개선 조합 존재:** 예 (total_return 기준 {len(improved)}개). "
            "운영 반영 전 OOS·샤프·드로다운 추가 확인 권장."
        )
    else:
        lines.append("- **실제 개선 조합 존재:** 이번 180d 윈도우 기준으로는 **없음** (total_return ≤ baseline).")
    lines.append(
        "- **코드 반영:** `signal_exit_th_loss_aux_*` 파라미터는 `run_backtest`/`run_backtest_7d`로 이미 전달 가능. "
        "개선이 확인되면 설정값만 기본값으로 옮기면 됨."
    )
    lines.append(
        "- **다음 단계:** (A) 다른 기간/윌크포워드로 재검증 (B) δ_p·mae_cut 미세 조정 (C) 확장(SHORT-only)이 베이스 그리드보다 나은지 확인."
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[loss_aux_grid] wrote {out_path}", flush=True)

    # CSV
    csv_path = out_path.with_suffix(".csv")
    pd.DataFrame(rows + ext_rows).to_csv(csv_path, index=False)
    print(f"[loss_aux_grid] wrote {csv_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
