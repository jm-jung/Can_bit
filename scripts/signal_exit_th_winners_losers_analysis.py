#!/usr/bin/env python3
"""
180d / threshold=0.60 / emit_trade_log=True 기준 signal_exit_th 청산을
profit으로 winners vs losers 분류·비교·해석 리포트 생성.

threshold / 모델 / time_stop / early_exit / SHORT OFF 변경 없음 (분석만).
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


def _run_window(
    *,
    df_bt: pd.DataFrame,
    pl_primary: np.ndarray,
    ps_primary: np.ndarray,
    t_max: pd.Timestamp,
    window_days: int,
    threshold: float,
    emit_trade_log: bool,
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
    )
    if err:
        raise RuntimeError(f"run_backtest_7d failed: {err}")
    if res is None:
        res = {}
    return {"df_w": df_w, "pl_w": pl_w, "ps_w": ps_w, "result": res}


def _exit_events_df(res: dict[str, Any]) -> pd.DataFrame:
    events = list(res.get("trade_events") or [])
    exit_events = [e for e in events if str(e.get("event", "")).startswith("EXIT")]
    if not exit_events:
        return pd.DataFrame()
    return pd.DataFrame(exit_events)


def _dir_proba(pl_w: np.ndarray, ps_w: np.ndarray, idx: int, direction: str) -> float:
    idx = max(0, min(idx, len(pl_w) - 1))
    if direction == "LONG":
        return float(pl_w[idx])
    if direction == "SHORT":
        return float(ps_w[idx])
    return float("nan")


def _unrealized_return(entry_price: float, close: float, direction: str) -> float:
    if direction == "LONG":
        return (close - entry_price) / entry_price
    if direction == "SHORT":
        return (entry_price - close) / entry_price
    return float("nan")


def _recompute_mfe_mae(
    *,
    high: np.ndarray,
    low: np.ndarray,
    entry_price: float,
    entry_idx: int,
    exit_idx: int,
    direction: str,
) -> tuple[float, float]:
    """엔진과 동일한 정의: bar별 high/low 기준 누적 MFE(최대 유리), MAE(최대 불리, 음수)."""
    mfe = -1e18
    mae = 1e18
    for j in range(entry_idx, exit_idx + 1):
        if j >= len(high):
            break
        ch, cl = float(high[j]), float(low[j])
        if direction == "LONG":
            rh = (ch - entry_price) / entry_price
            rl = (cl - entry_price) / entry_price
        else:
            rh = (entry_price - cl) / entry_price
            rl = (entry_price - ch) / entry_price
        mfe = max(mfe, rh)
        mae = min(mae, rl)
    return float(mfe), float(mae)


def _fmt_pct(x: float) -> str:
    if not np.isfinite(x):
        return "N/A"
    return f"{100.0 * x:.1f}%"


def _fmt_num(x: float, nd: int = 4) -> str:
    if not np.isfinite(x):
        return "N/A"
    return f"{float(x):.{nd}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-max", default="2026-03-20T07:40:00+00:00")
    parser.add_argument("--days-full", type=int, default=182)
    parser.add_argument("--window-days", type=int, default=180)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument(
        "--out-md",
        default="data/diagnostics/fr2/SIGNAL_EXIT_TH_WINNERS_LOSERS_REPORT.md",
    )
    args = parser.parse_args()

    from scripts.run_fr2_diagnostics import MODELS_DIR, get_ohlcv_and_proba
    from scripts.run_fr2_regime_conditioning import add_regime_columns
    from src.strategies.ensemble_strategy import EnsembleInputs, build_ensemble_proba, build_fr2_c4_mask

    t_max = _to_utc_ts(args.t_max)
    end_date_utc_str = t_max.strftime("%Y-%m-%d")
    DAYS_FULL = int(args.days_full)
    thr = float(args.threshold)

    base_pt = MODELS_DIR / "tcn_h15_t0p004.pt"
    fr2_pt = MODELS_DIR / "tcn_h15_micro_v1.pt"

    print(f"[wl] t_max={t_max.isoformat()} window={args.window_days} thr={thr}", flush=True)

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

    out = _run_window(
        df_bt=df_bt,
        pl_primary=pl_primary,
        ps_primary=ps_primary,
        t_max=t_max,
        window_days=int(args.window_days),
        threshold=thr,
        emit_trade_log=True,
    )
    res = out["result"]
    df_w = out["df_w"]
    pl_w = out["pl_w"]
    ps_w = out["ps_w"]

    window_start = t_max - pd.Timedelta(days=int(args.window_days))
    t = pd.to_datetime(df_bt["timestamp"], utc=True)
    mask = (t >= window_start) & (t <= t_max)
    df_w_reg = df_bt_reg.loc[mask].reset_index(drop=True)

    high_w = df_w["high"].to_numpy(dtype=float)
    low_w = df_w["low"].to_numpy(dtype=float)
    close_w = df_w["close"].to_numpy(dtype=float)

    exits = _exit_events_df(res)
    sig = exits[exits["exit_reason"].astype(str) == "signal_exit_th"].copy()
    if sig.empty:
        print("No signal_exit_th exits.", flush=True)
        return 1

    sig["profit"] = pd.to_numeric(sig["profit"], errors="coerce")
    sig["losers"] = sig["profit"] < 0
    sig["winners"] = sig["profit"] >= 0

    rows: list[dict[str, Any]] = []
    for _, r in sig.iterrows():
        exit_idx = int(r["idx"])
        bh = int(r["bars_held"]) if pd.notna(r.get("bars_held")) else 0
        entry_idx = exit_idx - bh
        direction = str(r.get("direction", ""))
        entry_price = float(r.get("entry_price", np.nan))
        ep = entry_idx
        if ep < 0 or exit_idx >= len(df_w):
            continue

        proba_entry = _dir_proba(pl_w, ps_w, entry_idx, direction)
        proba_exit_prev = _dir_proba(pl_w, ps_w, exit_idx - 1, direction)
        proba_exit = _dir_proba(pl_w, ps_w, exit_idx, direction)

        mfe_ev = r.get("max_favorable_excursion")
        mae_ev = r.get("max_adverse_excursion")
        if mfe_ev is None or mae_ev is None or (isinstance(mfe_ev, float) and np.isnan(mfe_ev)):
            mfe_v, mae_v = _recompute_mfe_mae(
                high=high_w,
                low=low_w,
                entry_price=entry_price,
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                direction=direction,
            )
        else:
            mfe_v, mae_v = float(mfe_ev), float(mae_ev)

        u_exit = _unrealized_return(entry_price, close_w[exit_idx], direction)
        j4 = exit_idx - 4
        if j4 < entry_idx:
            j4 = entry_idx
        u_prev4 = _unrealized_return(entry_price, close_w[j4], direction)
        d_unreal_4 = u_exit - u_prev4

        # exit 직전 4바 구간의 순가격 수익률 (방향 무관 비교용 보조)
        px_ret4 = (close_w[exit_idx] / close_w[max(entry_idx, exit_idx - 4)] - 1.0) if exit_idx > 0 else 0.0

        ts_ent = df_w_reg["timestamp"].iloc[entry_idx] if entry_idx < len(df_w_reg) else None
        trend_e = str(df_w_reg["trend_regime"].iloc[entry_idx]) if entry_idx < len(df_w_reg) else ""
        vol_e = str(df_w_reg["vol_regime"].iloc[entry_idx]) if entry_idx < len(df_w_reg) else ""
        cross_e = str(df_w_reg["cross_regime"].iloc[entry_idx]) if entry_idx < len(df_w_reg) else ""

        rows.append(
            {
                "profit": float(r["profit"]),
                "losers": bool(r["losers"]),
                "direction": direction,
                "bars_held": bh,
                "proba_entry": proba_entry,
                "proba_exit_prev": proba_exit_prev,
                "proba_exit": proba_exit,
                "mfe": mfe_v,
                "mae": mae_v,
                "delta_unreal_4bars": d_unreal_4,
                "px_ret_last4_to_exit": px_ret4,
                "trend_regime_entry": trend_e,
                "vol_regime_entry": vol_e,
                "cross_regime_entry": cross_e,
            }
        )

    dfa = pd.DataFrame(rows)
    if dfa.empty:
        print("No rows after alignment.", flush=True)
        return 1

    def _agg(name: str, sub: pd.DataFrame) -> dict[str, Any]:
        n = len(sub)
        if n == 0:
            return {
                "name": name,
                "count": 0,
                "share_short": float("nan"),
                "share_long": float("nan"),
                "mean_bars": float("nan"),
                "median_bars": float("nan"),
                "mean_proba_entry": float("nan"),
                "mean_proba_exit_prev": float("nan"),
                "mean_mfe": float("nan"),
                "mean_mae": float("nan"),
                "mean_delta_unreal_4": float("nan"),
                "mean_px_ret4": float("nan"),
            }
        short_share = float((sub["direction"] == "SHORT").mean())
        long_share = float((sub["direction"] == "LONG").mean())
        return {
            "name": name,
            "count": n,
            "share_short": short_share,
            "share_long": long_share,
            "mean_bars": float(sub["bars_held"].mean()),
            "median_bars": float(sub["bars_held"].median()),
            "mean_proba_entry": float(sub["proba_entry"].mean()),
            "mean_proba_exit_prev": float(sub["proba_exit_prev"].mean()),
            "mean_mfe": float(sub["mfe"].mean()),
            "mean_mae": float(sub["mae"].mean()),
            "mean_delta_unreal_4": float(sub["delta_unreal_4bars"].mean()),
            "mean_px_ret4": float(sub["px_ret_last4_to_exit"].mean()),
        }

    los = dfa[dfa["losers"]]
    win = dfa[~dfa["losers"]]
    A_l = _agg("losers", los)
    A_w = _agg("winners", win)

    # regime 분포 (entry 시점)
    def _regime_lines(sub: pd.DataFrame, title: str) -> list[str]:
        if sub.empty:
            return [f"- {title}: (empty)\n"]
        out = [f"- {title} — trend: {sub['trend_regime_entry'].value_counts().to_dict()}\n"]
        out.append(f"  - vol: {sub['vol_regime_entry'].value_counts().to_dict()}\n")
        out.append(f"  - cross_regime: {sub['cross_regime_entry'].value_counts().to_dict()}\n")
        return out

    lines: list[str] = []
    lines.append("# signal_exit_th — Winners vs Losers (180d 분류 분석)\n\n")
    lines.append(f"- 설정: window={args.window_days}d, min_max_proba={thr}, emit_trade_log=True\n\n")

    lines.append("## A. winners vs losers 비교 표\n\n")
    lines.append("| 지표 | losers (profit<0) | winners (profit>=0) |\n")
    lines.append("|---|---:|---:|\n")
    lines.append(f"| count | {A_l['count']} | {A_w['count']} |\n")
    lines.append(f"| direction SHORT 비율 | {_fmt_pct(A_l['share_short'])} | {_fmt_pct(A_w['share_short'])} |\n")
    lines.append(f"| direction LONG 비율 | {_fmt_pct(A_l['share_long'])} | {_fmt_pct(A_w['share_long'])} |\n")
    lines.append(f"| mean bars_held | {_fmt_num(A_l['mean_bars'], 2)} | {_fmt_num(A_w['mean_bars'], 2)} |\n")
    lines.append(f"| median bars_held | {_fmt_num(A_l['median_bars'], 1)} | {_fmt_num(A_w['median_bars'], 1)} |\n")
    lines.append(f"| mean proba_entry | {_fmt_num(A_l['mean_proba_entry'], 4)} | {_fmt_num(A_w['mean_proba_entry'], 4)} |\n")
    lines.append(f"| mean proba_exit_prev | {_fmt_num(A_l['mean_proba_exit_prev'], 4)} | {_fmt_num(A_w['mean_proba_exit_prev'], 4)} |\n")
    lines.append(f"| mean MFE (누적 유리) | {_fmt_num(A_l['mean_mfe'], 6)} | {_fmt_num(A_w['mean_mfe'], 6)} |\n")
    lines.append(f"| mean MAE (누적 불리, 음수) | {_fmt_num(A_l['mean_mae'], 6)} | {_fmt_num(A_w['mean_mae'], 6)} |\n")
    lines.append(
        "| mean Δunrealized (exit vs exit-4, 진입가 대비) | "
        f"{_fmt_num(A_l['mean_delta_unreal_4'], 6)} | {_fmt_num(A_w['mean_delta_unreal_4'], 6)} |\n"
    )
    lines.append(
        "| mean 순가격 수익률 (close[exit]/close[exit-4]-1) | "
        f"{_fmt_num(A_l['mean_px_ret4'], 6)} | {_fmt_num(A_w['mean_px_ret4'], 6)} |\n"
    )
    lines.append("\n### Regime (진입 시점)\n\n")
    lines.extend(_regime_lines(los, "losers"))
    lines.extend(_regime_lines(win, "winners"))

    # [3] 질문에 대한 수치 답변
    lines.append("\n## 질문에 대한 정량 답변\n\n")
    # SHORT 편중: losers vs winners short share
    lines.append(
        f"- **손실이 SHORT에 편중되는가?** "
        f"losers SHORT {_fmt_pct(A_l['share_short'])} vs winners SHORT {_fmt_pct(A_w['share_short'])}. "
    )
    if np.isfinite(A_l["share_short"]) and np.isfinite(A_w["share_short"]):
        lines[-1] += (
            "**예** (손실군 SHORT 비율이 더 큼)"
            if A_l["share_short"] > A_w["share_short"]
            else (
                "**아니오** (손실군 SHORT 비율이 더 작음)"
                if A_l["share_short"] < A_w["share_short"]
                else "**동일**."
            )
        )
    lines[-1] += "\n"
    lines.append(
        f"- **손실이 bars_held가 더 긴가?** "
        f"losers mean/median {_fmt_num(A_l['mean_bars'], 2)} / {_fmt_num(A_l['median_bars'], 1)} vs "
        f"winners {_fmt_num(A_w['mean_bars'], 2)} / {_fmt_num(A_w['median_bars'], 1)}. "
    )
    if np.isfinite(A_l["mean_bars"]) and np.isfinite(A_w["mean_bars"]):
        lines[-1] += "**예** (평균 기준 더 김)" if A_l["mean_bars"] > A_w["mean_bars"] else (
            "**아니오** (평균 기준 더 짧음)" if A_l["mean_bars"] < A_w["mean_bars"] else "**혼합** (평균 동일)."
        )
    lines[-1] += "\n"
    # MAE: 더 불리 = 더 음수
    lines.append(
        f"- **손실이 역행(MAE)이 더 큰가(더 음수)?** "
        f"mean MAE losers {_fmt_num(A_l['mean_mae'], 6)} vs winners {_fmt_num(A_w['mean_mae'], 6)}. "
    )
    if np.isfinite(A_l["mean_mae"]) and np.isfinite(A_w["mean_mae"]):
        lines[-1] += (
            "**예**" if A_l["mean_mae"] < A_w["mean_mae"] else ("**아니오**" if A_l["mean_mae"] > A_w["mean_mae"] else "**동일**")
        )
    lines[-1] += "\n"
    lines.append(
        "- **수익 그룹 공통 패턴(표 기준):** "
        f"MFE 평균 {_fmt_num(A_w['mean_mfe'], 6)} vs 손실 {_fmt_num(A_l['mean_mfe'], 6)}, "
        f"MAE 평균 {_fmt_num(A_w['mean_mae'], 6)} vs 손실 {_fmt_num(A_l['mean_mae'], 6)}, "
        f"exit 직전 4바 Δunrealized 평균 {_fmt_num(A_w['mean_delta_unreal_4'], 6)} (losers {_fmt_num(A_l['mean_delta_unreal_4'], 6)}).\n"
    )

    lines.append("\n## B. 손실형 signal_exit_th의 공통 패턴 (요약)\n\n")
    b_pts: list[str] = []
    if np.isfinite(A_l["share_short"]) and np.isfinite(A_w["share_short"]):
        if A_l["share_short"] > A_w["share_short"]:
            b_pts.append(
                f"- SHORT 비중이 수익군보다 **높음** (losers {_fmt_pct(A_l['share_short'])} vs winners {_fmt_pct(A_w['share_short'])}).\n"
            )
        elif A_l["share_short"] < A_w["share_short"]:
            b_pts.append(
                f"- SHORT 비중은 수익군보다 **낮음** (losers {_fmt_pct(A_l['share_short'])} vs winners {_fmt_pct(A_w['share_short'])}).\n"
            )
        else:
            b_pts.append("- SHORT 비중은 두 그룹에서 **유사**.\n")
    if A_l["mean_bars"] > A_w["mean_bars"]:
        b_pts.append(
            f"- 평균 보유 바가 더 김 (losers {A_l['mean_bars']:.1f} vs winners {A_w['mean_bars']:.1f}).\n"
        )
    elif A_l["mean_bars"] < A_w["mean_bars"]:
        b_pts.append(
            f"- 평균 보유 바가 더 짧음 (losers {A_l['mean_bars']:.1f} vs winners {A_w['mean_bars']:.1f}).\n"
        )
    if A_l["mean_mae"] < A_w["mean_mae"]:
        b_pts.append(
            f"- 평균 MAE가 더 **불리**(더 음수): {A_l['mean_mae']:.6f} vs {A_w['mean_mae']:.6f}.\n"
        )
    elif A_l["mean_mae"] > A_w["mean_mae"]:
        b_pts.append(
            f"- 평균 MAE는 수익군보다 **덜 불리**: {A_l['mean_mae']:.6f} vs {A_w['mean_mae']:.6f}.\n"
        )
    if A_l["mean_mfe"] < A_w["mean_mfe"]:
        b_pts.append(
            f"- 평균 MFE가 더 **작음**(유리한 움직임을 덜 누림): {A_l['mean_mfe']:.6f} vs {A_w['mean_mfe']:.6f}.\n"
        )
    if A_l["mean_delta_unreal_4"] < A_w["mean_delta_unreal_4"]:
        b_pts.append(
            f"- 청산 직전 4바 **미실현 개선폭(Δunreal)이 더 작음**(말기 흐름이 덜 유리): "
            f"losers {A_l['mean_delta_unreal_4']:.6f} vs winners {A_w['mean_delta_unreal_4']:.6f}.\n"
        )
    elif A_l["mean_delta_unreal_4"] > A_w["mean_delta_unreal_4"]:
        b_pts.append(
            f"- 청산 직전 4바 Δunreal 평균이 수익군보다 **큼**(말기 반등 성분 가능): "
            f"losers {A_l['mean_delta_unreal_4']:.6f} vs winners {A_w['mean_delta_unreal_4']:.6f}.\n"
        )
    lines.extend(b_pts or ["- (요약 생성용 차이가 미미하거나 표본 부족)\n"])

    lines.append("\n## C. 회복형(수익) signal_exit_th의 공통 패턴 (요약)\n\n")
    c_pts: list[str] = []
    if A_w["mean_mfe"] > A_l["mean_mfe"]:
        c_pts.append(
            f"- **MFE 평균이 더 큼** → 진입 후 유리한 움직임을 더 크게 경험: {A_w['mean_mfe']:.6f} vs {A_l['mean_mfe']:.6f}.\n"
        )
    if A_w["mean_mae"] > A_l["mean_mae"]:
        c_pts.append(
            f"- **MAE가 덜 깊음**(덜 불리): {A_w['mean_mae']:.6f} vs {A_l['mean_mae']:.6f}.\n"
        )
    if A_w["mean_delta_unreal_4"] > A_l["mean_delta_unreal_4"]:
        c_pts.append(
            f"- 청산 직전 4바에서 **미실현이 더 개선**되는 경향: Δunreal winners {A_w['mean_delta_unreal_4']:.6f} vs losers {A_l['mean_delta_unreal_4']:.6f}.\n"
        )
    if np.isfinite(A_w["share_short"]) and A_w["share_short"] < A_l["share_short"]:
        c_pts.append("- 수익군에서 SHORT 비중이 상대적으로 **낮을 수 있음** (표 A 참고).\n")
    lines.extend(c_pts or ["- (수익군 특징이 손실군과 명확히 분리되지 않음 — 표 A·세부 CSV로 추가 확인)\n"])

    lines.append("\n## D. 다음 최소 실험 후보 (손실형만 선별 조기 절단 — 전체 조기화 금지)\n\n")
    lines.append(
        "1. **후보 1 — 이중 필터 (신뢰 하락 + 역행 확대):** "
        "`signal_exit_th`가 이미 발동 직전, "
        "`(proba_entry - proba_exit_prev) > δ_p` **그리고** "
        "`MAE < mae_cut` (진입 이후 누적 불리가 이미 충분히 깊음) 일 때만 "
        "청산 시점을 한두 바 앞당기거나 동일 바에서 확정 손실을 줄이는 **exit-only 보조 규칙** "
        "(구현 시 기존 `signal_exit_th` 분기 내부에서 losers-proxy 조건만 추가).\n"
    )
    lines.append(
        "2. **후보 2 — SHORT·장보유·역행 결합:** "
        "`direction==SHORT` **이고** `bars_held >= 24` **이고** "
        "`delta_unreal_4bars < 0` (청산 직전 4바 동안 포지션 관점 손익이 악화) "
        "인 `signal_exit_th`에 한해 **추가 압력 청산** 후보 "
        "(LONG·단기·직전 4바 개선 케이스는 건드리지 않음).\n"
    )
    lines.append(
        "\n> 위 후보는 **분류 결과를 본 뒤** 임계값(δ_p, mae_cut)을 소수 점만 스윕하는 **최소 실험**으로 검증.\n"
    )

    out_path = PROJECT_ROOT / args.out_md
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path_p = out_path.parent / f"{out_path.stem}_detail.csv"
    dfa.to_csv(csv_path_p, index=False)

    out_text = "".join(lines)
    out_path.write_text(out_text, encoding="utf-8")
    print(f"[wl] wrote {out_path}", flush=True)
    print(f"[wl] detail {csv_path_p}", flush=True)
    print(out_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
