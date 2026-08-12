#!/usr/bin/env python3
"""
FR2 (h15_micro_v1) + Baseline (h15_t0p004) Ensemble Study.

목적:
- baseline_only / FR2_C4_only / override / confirmation / score blending 앙상블을
  동일 데이터·백테스트 파이프라인에서 비교한다.
- 평가 구간: 720d / recent 180d / recent 90d (필요 시 recent 60d).

출력:
- data/diagnostics/fr2/fr2_baseline_ensemble_results.csv
- data/diagnostics/fr2/fr2_baseline_ensemble_regime.csv
- data/diagnostics/fr2/FR2_BASELINE_ENSEMBLE_REPORT.md
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "fr2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"

from scripts.run_fr2_diagnostics import (  # type: ignore
    DAYS,
    HORIZON,
    MODELS_DIR,
    WINDOW_SIZE,
    get_ohlcv_and_proba,
)
from scripts.run_fr2_oos_diagnosis import (  # type: ignore
    NEG_THRESHOLD,
    POS_THRESHOLD,
    _metrics,
)
from scripts.run_fr2_regime_conditioning import (  # type: ignore
    add_regime_columns,
)
from scripts.run_tcn_label_sweep_v2 import run_backtest_7d  # type: ignore
from src.strategies.ensemble_strategy import (
    EnsembleInputs,
    build_ensemble_proba,
    build_fr2_c4_mask,
)


BASELINE_ID = "h15_t0p004"
FR2_ID = "h15_micro_v1"

BASELINE_PT = MODELS_DIR / "tcn_h15_t0p004.pt"
FR2_PT = MODELS_DIR / "tcn_h15_micro_v1.pt"


def _align_by_timestamp(
    df_base: pd.DataFrame,
    pl_base: np.ndarray,
    ps_base: np.ndarray,
    df_fr2: pd.DataFrame,
    pl_fr2: np.ndarray,
    ps_fr2: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    baseline / FR2 예측을 timestamp 기준으로 inner join 하여 완전히 정렬된 시계열을 만든다.
    (미래 데이터 누수 없음: 모든 연산은 단일 시계열 상에서 순방향 인덱스 기반.)
    """
    df_b = df_base.copy()
    df_f = df_fr2.copy()
    for df in (df_b, df_f):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_b = df_b.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df_f = df_f.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    df_b["pl_base"] = pl_base[: len(df_b)]
    df_b["ps_base"] = ps_base[: len(df_b)]
    df_f["pl_fr2"] = pl_fr2[: len(df_f)]
    df_f["ps_fr2"] = ps_fr2[: len(df_f)]

    joined = df_b.merge(df_f[["timestamp", "pl_fr2", "ps_fr2"]], on="timestamp", how="inner")
    joined = joined.sort_values("timestamp").reset_index(drop=True)
    if len(joined) < 500:
        raise RuntimeError(f"Aligned length too small: {len(joined)}")

    df_bt = joined[["timestamp", "close", "high", "low"]].copy()
    pl_b_aligned = joined["pl_base"].to_numpy(dtype=float)
    ps_b_aligned = joined["ps_base"].to_numpy(dtype=float)
    pl_f_aligned = joined["pl_fr2"].to_numpy(dtype=float)
    ps_f_aligned = joined["ps_fr2"].to_numpy(dtype=float)
    return df_bt, pl_b_aligned, ps_b_aligned, pl_f_aligned, ps_f_aligned


def _future_return_from_close(close: np.ndarray) -> np.ndarray:
    """
    WINDOW_SIZE/HORIZON 설정과 동일한 방식으로 future_ret 을 재구성한다.
    (FR2 진단과 동일한 정의를 사용.)
    """
    N = len(close)
    valid_len = N - WINDOW_SIZE - HORIZON
    if valid_len <= 0:
        raise RuntimeError(f"valid_len={valid_len}")
    base = close[WINDOW_SIZE : WINDOW_SIZE + valid_len]
    fut = close[WINDOW_SIZE + HORIZON : WINDOW_SIZE + valid_len + HORIZON]
    future_ret = fut / base - 1.0
    if len(future_ret) > valid_len:
        future_ret = future_ret[:valid_len]
    elif len(future_ret) < valid_len:
        future_ret = np.resize(future_ret, valid_len)
    return future_ret


def _window_mask(df_bt: pd.DataFrame, window_days: int) -> np.ndarray:
    t = pd.to_datetime(df_bt["timestamp"])
    t_max = t.max()
    if window_days == 720:
        return np.ones(len(df_bt), dtype=bool)
    start = t_max - pd.Timedelta(days=window_days)
    return (t >= start).to_numpy()


def _extract_trade_stats(trades: List[dict]) -> dict:
    if not trades:
        return {
            "avg_pnl_per_trade": np.nan,
            "median_pnl_per_trade": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_consecutive_losses": 0,
        }
    pnls = [float(t.get("profit", t.get("pnl", 0.0))) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    if losses:
        profit_factor = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else np.nan
    else:
        profit_factor = np.nan
    max_consec = 0
    cur = 0
    for p in pnls:
        if p < 0:
            cur += 1
            if cur > max_consec:
                max_consec = cur
        else:
            cur = 0
    return {
        "avg_pnl_per_trade": float(np.mean(pnls)),
        "median_pnl_per_trade": float(np.median(pnls)),
        "win_rate": float(len(wins) / len(pnls)),
        "profit_factor": float(profit_factor) if not np.isnan(profit_factor) else np.nan,
        "max_consecutive_losses": int(max_consec),
    }


def _active_ratio(pl: np.ndarray, ps: np.ndarray, threshold_signal: float = 0.60) -> float:
    p_flat = np.clip(1.0 - pl - ps, 0.0, 1.0)
    max_proba = np.maximum(np.maximum(pl, ps), p_flat)
    return float((max_proba >= threshold_signal).mean())


def run_ensemble_experiment() -> None:
    # 1) Load aligned baseline & FR2 predictions on 720d
    triple_base, err_b = get_ohlcv_and_proba(DAYS, BASELINE_PT, "base", False)
    triple_fr2, err_f = get_ohlcv_and_proba(DAYS, FR2_PT, "microstructure_v1", True)
    if err_b or err_f or triple_base is None or triple_fr2 is None:
        raise RuntimeError(f"get_ohlcv_and_proba failed: base={err_b}, fr2={err_f}")

    df_b, pl_b, ps_b, _ = triple_base
    df_f, pl_f, ps_f, _ = triple_fr2
    df_bt_raw, pl_b_aligned, ps_b_aligned, pl_f_aligned, ps_f_aligned = _align_by_timestamp(
        df_b, pl_b, ps_b, df_f, pl_f, ps_f
    )
    close = df_bt_raw["close"].to_numpy(dtype=float)
    future_ret_full = _future_return_from_close(close)

    # 동일한 유효 구간만 사용 (WINDOW_SIZE~WINDOW_SIZE+valid_len)
    valid_len = len(future_ret_full)
    df_bt = df_bt_raw.iloc[WINDOW_SIZE : WINDOW_SIZE + valid_len].reset_index(drop=True)
    pl_b = pl_b_aligned[:valid_len]
    ps_b = ps_b_aligned[:valid_len]
    pl_f = pl_f_aligned[:valid_len]
    ps_f = ps_f_aligned[:valid_len]
    future_ret = future_ret_full[:valid_len]

    # 2) Regime columns + FR2 C4 gate mask (uptrend & high_vol & persistence>=6)
    df_bt_reg = add_regime_columns(DAYS, df_bt)
    # entry_mask_uptrend_and_high_vol는 add_regime_columns 내부 정의에 기반해 있으므로
    # vol_regime == high_vol & trend_regime == uptrend 으로 C4 base mask 구성
    base_c4_mask = (
        (df_bt_reg["trend_regime"] == "uptrend")
        & (df_bt_reg["vol_regime"] == "high_vol")
    ).to_numpy()
    c4_mask = build_fr2_c4_mask(base_c4_mask, persistence_bars=6)

    ensemble_inputs = EnsembleInputs(
        pl_base=pl_b,
        ps_base=ps_b,
        pl_fr2=pl_f,
        ps_fr2=ps_f,
        c4_active=c4_mask,
    )

    strategies = [
        ("baseline_only", "baseline_only"),
        ("fr2_c4_only", "fr2_c4_only"),
        ("override", "override_ensemble"),
        ("confirmation", "confirmation_ensemble"),
        ("blend_0.7_0.3", "blend_0.7_0.3"),
        ("blend_0.5_0.5", "blend_0.5_0.5"),
        ("blend_0.3_0.7", "blend_0.3_0.7"),
    ]
    windows = [720, 180, 90]
    rows: List[Dict] = []

    for mode, name in strategies:
        pl_mode, ps_mode = build_ensemble_proba(ensemble_inputs, mode=mode)  # type: ignore[arg-type]

        for window_days in windows:
            mask = _window_mask(df_bt, window_days)
            if mask.sum() < 500:
                continue

            df_w = df_bt.loc[mask].reset_index(drop=True)
            pl_w = pl_mode[mask]
            ps_w = ps_mode[mask]
            fr_w = future_ret[mask]

            # alignment metrics (direction_accuracy, spearman, signal_density, mean_return_signal)
            met = _metrics(pl_w, ps_w, fr_w, threshold_signal=0.60)

            # backtest via existing ML backtest engine (no leakage, df_w/proba aligned bar-by-bar)
            res_bt, err_bt = run_backtest_7d(
                SYMBOL,
                TIMEFRAME,
                df_w,
                pl_w,
                ps_w,
                commission_rate=0.0009,
                slippage_rate=0.0001,
                min_max_proba=None,
                max_entropy=None,
            )
            if res_bt is None:
                cost_on = np.nan
                mdd = np.nan
                trades = 0
                win_rate = np.nan
                trade_stats = _extract_trade_stats([])
                sample_flag = "yes"
            else:
                cost_on = float(res_bt.get("total_return", np.nan))
                mdd = float(res_bt.get("max_drawdown", np.nan))
                trades = int(res_bt.get("total_trades", 0))
                win_rate = float(res_bt.get("win_rate", np.nan))
                trade_stats = _extract_trade_stats(res_bt.get("trades", []))
                sample_flag = "yes" if trades < 100 else "no"

            row = {
                "strategy": name,
                "mode": mode,
                "window_days": window_days,
                "trades": trades,
                "signal_density": met.get("signal_density", np.nan),
                "direction_accuracy": met.get("direction_accuracy", np.nan),
                "spearman": met.get("spearman", np.nan),
                "mean_return_signal": met.get("mean_return_signal", np.nan),
                "cost_on": cost_on,
                "MDD": mdd,
                "avg_pnl_per_trade": trade_stats["avg_pnl_per_trade"],
                "median_pnl_per_trade": trade_stats["median_pnl_per_trade"],
                "win_rate": win_rate,
                "profit_factor": trade_stats["profit_factor"],
                "max_consecutive_losses": trade_stats["max_consecutive_losses"],
                "active_ratio": _active_ratio(pl_w, ps_w, threshold_signal=0.60),
                "sample_too_small": sample_flag,
            }
            rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "fr2_baseline_ensemble_results.csv", index=False)

    # ---- Regime analysis (trend / volatility) for each strategy on 720d ----
    regime_rows: List[Dict] = []
    df_reg_full = add_regime_columns(DAYS, df_bt.copy())
    trend_vals = ["uptrend", "downtrend"]
    vol_vals = ["low_vol", "mid_vol", "high_vol"]

    for mode, name in strategies:
        pl_mode, ps_mode = build_ensemble_proba(ensemble_inputs, mode=mode)  # type: ignore[arg-type]

        # 720d only for regime split
        pl_w = pl_mode
        ps_w = ps_mode
        fr_w = future_ret

        # trend regimes
        for reg in trend_vals:
            mask = (df_reg_full["trend_regime"] == reg).to_numpy()
            if mask.sum() < 50:
                regime_rows.append(
                    {
                        "strategy": name,
                        "mode": mode,
                        "window_days": 720,
                        "decomposition": "trend",
                        "regime": reg,
                        "trades": np.nan,
                        "signal_density": np.nan,
                        "direction_accuracy": np.nan,
                        "spearman": np.nan,
                        "mean_return_signal": np.nan,
                        "cost_on": np.nan,
                        "MDD": np.nan,
                        "sample_too_small": "yes",
                    }
                )
                continue
            df_reg = df_reg_full.loc[mask].reset_index(drop=True)
            pl_r = pl_w[mask]
            ps_r = ps_w[mask]
            fr_r = fr_w[mask]
            met = _metrics(pl_r, ps_r, fr_r, threshold_signal=0.60)
            res_bt, _ = run_backtest_7d(
                SYMBOL,
                TIMEFRAME,
                df_reg,
                pl_r,
                ps_r,
                commission_rate=0.0009,
                slippage_rate=0.0001,
                min_max_proba=None,
                max_entropy=None,
            )
            if res_bt is None:
                trades = 0
                cost_on = np.nan
                mdd = np.nan
                sample_flag = "yes"
            else:
                trades = int(res_bt.get("total_trades", 0))
                cost_on = float(res_bt.get("total_return", np.nan))
                mdd = float(res_bt.get("max_drawdown", np.nan))
                sample_flag = "yes" if trades < 100 else "no"

            regime_rows.append(
                {
                    "strategy": name,
                    "mode": mode,
                    "window_days": 720,
                    "decomposition": "trend",
                    "regime": reg,
                    "trades": trades,
                    "signal_density": met.get("signal_density", np.nan),
                    "direction_accuracy": met.get("direction_accuracy", np.nan),
                    "spearman": met.get("spearman", np.nan),
                    "mean_return_signal": met.get("mean_return_signal", np.nan),
                    "cost_on": cost_on,
                    "MDD": mdd,
                    "sample_too_small": sample_flag,
                }
            )

        # volatility regimes
        for reg in vol_vals:
            mask = (df_reg_full["vol_regime"] == reg).to_numpy()
            if mask.sum() < 50:
                regime_rows.append(
                    {
                        "strategy": name,
                        "mode": mode,
                        "window_days": 720,
                        "decomposition": "volatility",
                        "regime": reg,
                        "trades": np.nan,
                        "signal_density": np.nan,
                        "direction_accuracy": np.nan,
                        "spearman": np.nan,
                        "mean_return_signal": np.nan,
                        "cost_on": np.nan,
                        "MDD": np.nan,
                        "sample_too_small": "yes",
                    }
                )
                continue
            df_reg = df_reg_full.loc[mask].reset_index(drop=True)
            pl_r = pl_w[mask]
            ps_r = ps_w[mask]
            fr_r = fr_w[mask]
            met = _metrics(pl_r, ps_r, fr_r, threshold_signal=0.60)
            res_bt, _ = run_backtest_7d(
                SYMBOL,
                TIMEFRAME,
                df_reg,
                pl_r,
                ps_r,
                commission_rate=0.0009,
                slippage_rate=0.0001,
                min_max_proba=None,
                max_entropy=None,
            )
            if res_bt is None:
                trades = 0
                cost_on = np.nan
                mdd = np.nan
                sample_flag = "yes"
            else:
                trades = int(res_bt.get("total_trades", 0))
                cost_on = float(res_bt.get("total_return", np.nan))
                mdd = float(res_bt.get("max_drawdown", np.nan))
                sample_flag = "yes" if trades < 100 else "no"

            regime_rows.append(
                {
                    "strategy": name,
                    "mode": mode,
                    "window_days": 720,
                    "decomposition": "volatility",
                    "regime": reg,
                    "trades": trades,
                    "signal_density": met.get("signal_density", np.nan),
                    "direction_accuracy": met.get("direction_accuracy", np.nan),
                    "spearman": met.get("spearman", np.nan),
                    "mean_return_signal": met.get("mean_return_signal", np.nan),
                    "cost_on": cost_on,
                    "MDD": mdd,
                    "sample_too_small": sample_flag,
                }
            )

    df_regime = pd.DataFrame(regime_rows)
    df_regime.to_csv(OUT_DIR / "fr2_baseline_ensemble_regime.csv", index=False)


def write_report() -> None:
    res_p = OUT_DIR / "fr2_baseline_ensemble_results.csv"
    reg_p = OUT_DIR / "fr2_baseline_ensemble_regime.csv"
    if not res_p.exists():
        return
    df = pd.read_csv(res_p)

    lines: List[str] = []
    lines.append("# FR2 + Baseline Ensemble Study\n")
    lines.append("## 1. Strategies compared\n")
    lines.append("- baseline_only\n- FR2_C4_only\n- override_ensemble\n- confirmation_ensemble\n- blend_0.7_0.3\n- blend_0.5_0.5\n- blend_0.3_0.7\n")

    # 720d / 180d / 90d performance summary
    lines.append("## 2. Performance summary (720d / 180d / 90d)\n")
    for window in [720, 180, 90]:
        sub = df[df["window_days"] == window].copy()
        if sub.empty:
            continue
        lines.append(f"### Window = {window}d\n")
        lines.append("| strategy | trades | cost_on | MDD | win_rate | avg_pnl | median_pnl | profit_factor | signal_density | sample_too_small |")
        lines.append("|----------|--------|---------|-----|----------|---------|------------|---------------|----------------|------------------|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['strategy']} | {int(r['trades'])} | {float(r['cost_on']):.4f} | {float(r['MDD']):.4f} | "
                f"{float(r['win_rate']) if not pd.isna(r['win_rate']) else 0:.4f} | "
                f"{float(r['avg_pnl_per_trade']):.6f} | {float(r['median_pnl_per_trade']):.6f} | "
                f"{float(r['profit_factor']) if not pd.isna(r['profit_factor']) else 0:.3f} | "
                f"{float(r['signal_density']):.4f} | {r['sample_too_small']} |"
            )
        lines.append("")

    # 3. Regime performance summary
    if reg_p.exists():
        df_reg = pd.read_csv(reg_p)
        lines.append("## 3. Regime performance (trend / volatility, 720d)\n")
        for decomp in ["trend", "volatility"]:
            sub = df_reg[df_reg["decomposition"] == decomp].copy()
            if sub.empty:
                continue
            lines.append(f"### {decomp.capitalize()} regimes\n")
            lines.append("| strategy | regime | trades | cost_on | MDD | direction_accuracy | spearman | signal_density | sample_too_small |")
            lines.append("|----------|--------|--------|---------|-----|--------------------|----------|----------------|------------------|")
            for _, r in sub.iterrows():
                lines.append(
                    f"| {r['strategy']} | {r['regime']} | {r['trades']} | {float(r['cost_on']):.4f} | "
                    f"{float(r['MDD']):.4f} | {float(r['direction_accuracy']):.4f} | "
                    f"{float(r['spearman']):.4f} | {float(r['signal_density']):.4f} | {r['sample_too_small']} |"
                )
            lines.append("")

    lines.append("## 4. Statistical cautions\n")
    lines.append("- trades < 100 인 window/regime 셀은 sample_too_small 로 표시되며 과해석을 피해야 한다.\n")
    lines.append("- FR2_C4_only 및 confirmation_ensemble 은 의도적으로 low-frequency high-quality 영역을 노리므로, cost_on 대비 MDD·trade quality 를 함께 봐야 한다.\n")

    # 5. Final recommendation placeholder (to be interpreted manually from CSV)
    lines.append("## 5. Final recommendation (template)\n")
    lines.append(
        "- ENSEMBLE_IMPROVES_BASELINE: baseline_only 대비 cost_on 증가 + MDD 감소 또는 동등 + "
        "trade quality (avg/median PnL, PF, win_rate) 개선.\n"
    )
    lines.append(
        "- FR2_ADDS_CONDITIONAL_ALPHA: 전 구간이 아니라 특정 regime(예: uptrend_high_vol)에서만 "
        "override / blending 계열이 baseline_only 대비 우수할 때.\n"
    )
    lines.append("- NO_MEANINGFUL_ENSEMBLE_GAIN: 위 두 조건이 모두 충족되지 않을 때.\n")

    (OUT_DIR / "FR2_BASELINE_ENSEMBLE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    run_ensemble_experiment()
    write_report()
    print("[FR2-ENSEMBLE] Done. Wrote fr2_baseline_ensemble_results.csv, fr2_baseline_ensemble_regime.csv, FR2_BASELINE_ENSEMBLE_REPORT.md", flush=True)


if __name__ == "__main__":
    main()

