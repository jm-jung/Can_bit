#!/usr/bin/env python3
"""
FR2 threshold sweep 실험.

가설: alpha_per_trade > fee_per_trade 가 되는 threshold가 존재하는가.
threshold ↑ → turnover ↓ → execution cost 극복 여부 검증.

Execution: run_fr2_fair_ensemble_comparison와 동일 (유일 변수 = probability threshold = min_max_proba).
전략: override_ensemble, confirmation_ensemble.
Threshold: 0.60, 0.65, 0.70, 0.75, 0.80, 0.85.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "fr2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"

COMMISSION = 0.0009
SLIPPAGE = 0.0001
MAX_ENTROPY = 1.30
MIN_HOLD = 36
COOLDOWN = 12
TIME_STOP_BARS = 72
EARLY_EXIT_BAD_K = 8

THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
STRATEGIES = [
    ("override", "override_ensemble"),
    ("confirmation", "confirmation_ensemble"),
]
WINDOW_DAYS = 720
MIN_TRADES_SAMPLE = 100

from scripts.run_fr2_diagnostics import (  # type: ignore
    DAYS,
    HORIZON,
    MODELS_DIR,
    WINDOW_SIZE,
    get_ohlcv_and_proba,
)
from scripts.run_fr2_regime_conditioning import add_regime_columns  # type: ignore
from scripts.run_tcn_label_sweep_v2 import run_backtest_7d  # type: ignore
from src.strategies.ensemble_strategy import (
    EnsembleInputs,
    build_ensemble_proba,
    build_fr2_c4_mask,
)

BASELINE_PT = MODELS_DIR / "tcn_h15_t0p004.pt"
FR2_PT = MODELS_DIR / "tcn_h15_micro_v1.pt"


def _align_by_timestamp(
    df_base: pd.DataFrame,
    pl_base: np.ndarray,
    ps_base: np.ndarray,
    df_fr2: pd.DataFrame,
    pl_fr2: np.ndarray,
    ps_fr2: np.ndarray,
) -> tuple:
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
    pl_b = joined["pl_base"].to_numpy(dtype=float)
    ps_b = joined["ps_base"].to_numpy(dtype=float)
    pl_f = joined["pl_fr2"].to_numpy(dtype=float)
    ps_f = joined["ps_fr2"].to_numpy(dtype=float)
    return df_bt, pl_b, ps_b, pl_f, ps_f


def _future_return_from_close(close: np.ndarray) -> np.ndarray:
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
    if window_days >= 720:
        return np.ones(len(df_bt), dtype=bool)
    start = t_max - pd.Timedelta(days=window_days)
    return (t >= start).to_numpy()


def _score_series(pl: np.ndarray, ps: np.ndarray) -> np.ndarray:
    p_flat = np.clip(1.0 - pl - ps, 0.0, 1.0)
    return np.maximum(np.maximum(pl, ps), p_flat)


def _extract_trade_stats(trades: List[Any]) -> Dict[str, float]:
    if not trades:
        return {"win_rate": np.nan, "profit_factor": np.nan}
    pnls = []
    for t in trades:
        p = t.get("profit") if isinstance(t, dict) else getattr(t, "profit", None)
        if p is None:
            p = t.get("net_return", t.get("pnl", 0.0))
        pnls.append(float(p))
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    if losses and sum(losses) != 0:
        profit_factor = sum(wins) / abs(sum(losses))
    else:
        profit_factor = np.nan
    return {
        "win_rate": float(len(wins) / len(pnls)),
        "profit_factor": float(profit_factor) if not np.isnan(profit_factor) else np.nan,
    }


def run_distribution_and_sweep() -> None:
    print("[FR2-THRESH] Loading baseline...", flush=True)
    triple_base, err_b = get_ohlcv_and_proba(DAYS, BASELINE_PT, "base", False)
    print("[FR2-THRESH] Loading FR2...", flush=True)
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
    valid_len = len(future_ret_full)
    df_bt = df_bt_raw.iloc[WINDOW_SIZE : WINDOW_SIZE + valid_len].reset_index(drop=True)
    pl_b = pl_b_aligned[:valid_len]
    ps_b = ps_b_aligned[:valid_len]
    pl_f = pl_f_aligned[:valid_len]
    ps_f = ps_f_aligned[:valid_len]

    df_bt_reg = add_regime_columns(DAYS, df_bt)
    base_c4_mask = (
        (df_bt_reg["trend_regime"] == "uptrend")
        & (df_bt_reg["vol_regime"] == "high_vol")
    ).to_numpy()
    c4_mask = build_fr2_c4_mask(base_c4_mask, persistence_bars=6)
    ensemble_inputs = EnsembleInputs(
        pl_base=pl_b, ps_base=ps_b, pl_fr2=pl_f, ps_fr2=ps_f, c4_active=c4_mask
    )

    mask = _window_mask(df_bt, WINDOW_DAYS)
    n_bars = int(mask.sum())
    df_w = df_bt.loc[mask].reset_index(drop=True)

    def run_bt(
        pl_w: np.ndarray,
        ps_w: np.ndarray,
        min_max_proba: float,
        commission: float,
        slippage: float,
    ):
        return run_backtest_7d(
            SYMBOL,
            TIMEFRAME,
            df_w,
            pl_w,
            ps_w,
            commission_rate=commission,
            slippage_rate=slippage,
            min_max_proba=min_max_proba,
            max_entropy=MAX_ENTROPY,
            decision_mode="argmax",
            min_hold=MIN_HOLD,
            cooldown=COOLDOWN,
            time_stop_enabled=True,
            time_stop_bars=TIME_STOP_BARS,
            early_exit_enabled=True,
            early_exit_bad_k=EARLY_EXIT_BAD_K,
        )

    # ----- Block 1: Score distribution (sweep 전에 실행) -----
    rows_dist: List[Dict] = []
    for mode, name in STRATEGIES:
        pl_mode, ps_mode = build_ensemble_proba(ensemble_inputs, mode=mode)  # type: ignore[arg-type]
        pl_w = pl_mode[mask]
        ps_w = ps_mode[mask]
        score = _score_series(pl_w, ps_w)
        for th in THRESHOLDS:
            above = (score >= th).astype(float)
            signal_density = float(above.mean())
            bar_count_above_threshold = int(above.sum())
            rows_dist.append({
                "strategy": name,
                "threshold": th,
                "signal_density": signal_density,
                "bar_count_above_threshold": bar_count_above_threshold,
            })
    pd.DataFrame(rows_dist).to_csv(OUT_DIR / "fr2_threshold_distribution.csv", index=False)
    print("[FR2-THRESH] Wrote fr2_threshold_distribution.csv", flush=True)

    # ----- Block 2: Threshold sweep -----
    rows_sweep: List[Dict] = []
    for mode, name in STRATEGIES:
        pl_mode, ps_mode = build_ensemble_proba(ensemble_inputs, mode=mode)  # type: ignore[arg-type]
        pl_w = pl_mode[mask]
        ps_w = ps_mode[mask]
        for th in THRESHOLDS:
            try:
                res_on, _ = run_bt(pl_w, ps_w, min_max_proba=th, commission=COMMISSION, slippage=SLIPPAGE)
            except Exception:
                res_on = None
            try:
                res_off, _ = run_bt(pl_w, ps_w, min_max_proba=th, commission=0.0, slippage=0.0)
            except Exception:
                res_off = None

            if res_on is None:
                cost_on = np.nan
                trades_on = 0
                win_rate = np.nan
                profit_factor = np.nan
            else:
                cost_on = float(res_on.get("total_return", np.nan))
                trades_on = int(res_on.get("total_trades", 0))
                stats = _extract_trade_stats(res_on.get("trades", []))
                win_rate = stats["win_rate"]
                profit_factor = stats["profit_factor"]
            if res_off is None:
                cost_off = np.nan
                trades_off = 0
            else:
                cost_off = float(res_off.get("total_return", np.nan))
                trades_off = int(res_off.get("total_trades", 0))

            trades = max(trades_on, trades_off)
            turnover_ratio = trades / n_bars if n_bars else np.nan

            if trades > 0:
                alpha_per_trade = cost_off / trades
                fee_per_trade = (cost_off - cost_on) / trades
                if fee_per_trade is not None and not np.isnan(fee_per_trade) and abs(fee_per_trade) > 1e-12:
                    alpha_fee_ratio = alpha_per_trade / fee_per_trade
                else:
                    alpha_fee_ratio = np.nan
            else:
                alpha_per_trade = np.nan
                fee_per_trade = np.nan
                alpha_fee_ratio = np.nan

            sample_too_small = trades < MIN_TRADES_SAMPLE

            rows_sweep.append({
                "strategy": name,
                "threshold": th,
                "trades": trades,
                "turnover_ratio": turnover_ratio,
                "cost_on": cost_on,
                "cost_off": cost_off,
                "profit_factor": profit_factor,
                "win_rate": win_rate,
                "alpha_per_trade": alpha_per_trade,
                "fee_per_trade": fee_per_trade,
                "alpha_fee_ratio": alpha_fee_ratio,
                "sample_too_small": sample_too_small,
            })
            co_s = f"{float(cost_on):.4f}" if cost_on is not None and not (isinstance(cost_on, float) and np.isnan(cost_on)) else "n/a"
            cf_s = f"{float(cost_off):.4f}" if cost_off is not None and not (isinstance(cost_off, float) and np.isnan(cost_off)) else "n/a"
            ar_s = f"{float(alpha_fee_ratio):.3f}" if alpha_fee_ratio is not None and not (isinstance(alpha_fee_ratio, float) and np.isnan(alpha_fee_ratio)) else "n/a"
            print(f"  {name} th={th} -> trades={trades} cost_on={co_s} cost_off={cf_s} alpha_fee_ratio={ar_s}", flush=True)

    pd.DataFrame(rows_sweep).to_csv(OUT_DIR / "fr2_threshold_sweep.csv", index=False)
    print("[FR2-THRESH] Wrote fr2_threshold_sweep.csv", flush=True)


def write_report_and_verdict() -> None:
    p_sweep = OUT_DIR / "fr2_threshold_sweep.csv"
    p_dist = OUT_DIR / "fr2_threshold_distribution.csv"
    if not p_sweep.exists():
        (OUT_DIR / "FR2_THRESHOLD_SWEEP_REPORT.md").write_text(
            "# FR2 Threshold Sweep Report\n\nNo data. Run: .venv/bin/python scripts/run_fr2_threshold_sweep.py\n",
            encoding="utf-8",
        )
        return

    df = pd.read_csv(p_sweep)
    df_dist = pd.read_csv(p_dist) if p_dist.exists() else pd.DataFrame()

    def _f(x, fmt=".4f"):
        if pd.isna(x):
            return "—"
        try:
            return format(float(x), fmt)
        except (TypeError, ValueError):
            return "—"

    lines: List[str] = [
        "# FR2 Threshold Sweep Report",
        "",
        "## 1. 실험 목적",
        "",
        "alpha_per_trade > fee_per_trade 가 되는 threshold 존재 여부 검증. threshold ↑ → turnover ↓ → execution cost 극복.",
        "",
        "## 2. Score distribution (feasibility)",
        "",
    ]
    if not df_dist.empty:
        lines.append("| strategy | threshold | signal_density | bar_count_above_threshold |")
        lines.append("|----------|-----------|----------------|---------------------------|")
        for _, r in df_dist.iterrows():
            lines.append(f"| {r['strategy']} | {r['threshold']} | {_f(r['signal_density'])} | {int(r['bar_count_above_threshold'])} |")
        lines.append("")

    lines.append("## 3. Threshold sweep 결과")
    lines.append("")
    lines.append("| strategy | threshold | trades | turnover_ratio | cost_on | cost_off | alpha_per_trade | fee_per_trade | alpha_fee_ratio | sample_too_small |")
    lines.append("|----------|-----------|--------|----------------|---------|----------|-----------------|---------------|-----------------|------------------|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['strategy']} | {r['threshold']} | {int(r['trades'])} | {_f(r['turnover_ratio'], '.6f')} | "
            f"{_f(r['cost_on'])} | {_f(r['cost_off'])} | {_f(r['alpha_per_trade'], '.6f')} | "
            f"{_f(r['fee_per_trade'], '.6f')} | {_f(r['alpha_fee_ratio'], '.3f')} | {r['sample_too_small']} |"
        )
    lines.append("")

    lines.append("## 4. 분석 질문")
    lines.append("")
    lines.append("1) threshold ↑ → turnover 감소? (위 표 turnover_ratio 열)")
    lines.append("2) threshold ↑ → alpha_per_trade 변화?")
    lines.append("3) alpha_per_trade > fee_per_trade 인 threshold 존재? (alpha_fee_ratio > 1)")
    lines.append("4) cost_on ≥ 0 인 threshold 존재?")
    lines.append("")

    # Verdict
    lines.append("## 5. 최종 판정")
    lines.append("")
    any_case1 = False
    any_case2 = False
    for _, r in df.iterrows():
        if r.get("sample_too_small") in (True, "True", "yes"):
            continue
        co = r["cost_on"]
        cf = r["cost_off"]
        if pd.notna(co) and float(co) >= 0 and int(r["trades"]) >= MIN_TRADES_SAMPLE:
            any_case1 = True
        if pd.notna(cf) and float(cf) > 0 and pd.notna(co) and float(co) < 0:
            any_case2 = True
    if any_case1:
        verdict = "EXECUTION_SOLVES_ALPHA"
        desc = "cost_on ≥ 0 이고 trades ≥ 100 인 (strategy, threshold) 존재."
    elif any_case2:
        verdict = "ALPHA_EXISTS_BUT_FEE_TOO_HIGH"
        desc = "cost_off > 0 이나 cost_on < 0. Execution engineering 필요."
    else:
        verdict = "SIGNAL_TOO_WEAK"
        desc = "cost_off ≤ 0. Signal 연구 재검토."
    lines.append(f"**판정: {verdict}**")
    lines.append("")
    lines.append(desc)
    lines.append("")

    (OUT_DIR / "FR2_THRESHOLD_SWEEP_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("[FR2-THRESH] Wrote FR2_THRESHOLD_SWEEP_REPORT.md", flush=True)


def main() -> None:
    global STRATEGIES, THRESHOLDS
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="1 strategy, 2 thresholds (0.60, 0.70) only")
    args = p.parse_args()
    if args.quick:
        STRATEGIES = [("override", "override_ensemble")]
        THRESHOLDS = [0.60, 0.70]
    print("[FR2-THRESH] Score distribution + threshold sweep (override, confirmation)...", flush=True)
    run_distribution_and_sweep()
    write_report_and_verdict()
    print("[FR2-THRESH] Done.", flush=True)


if __name__ == "__main__":
    main()
