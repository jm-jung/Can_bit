#!/usr/bin/env python3
"""
FR2 execution engineering 실험.

목적: 같은 signal을 더 적게·신중하게 거래하면 execution cost를 이길 수 있는가.
고정: strategy=override_ensemble, threshold=0.60.
변경: execution only — min_hold, cooldown (reentry/flip은 엔진 미지원으로 0 고정).
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
TIME_STOP_BARS = 72
EARLY_EXIT_BAD_K = 8
STRATEGY = "override_ensemble"
THRESHOLD = 0.60
WINDOW_DAYS = 720
MIN_TRADES_SAMPLE = 100

# Execution sweep (reentry/flip은 엔진에 파라미터 없음 → 0만 사용)
MIN_HOLD_VALUES = [0, 3, 6, 12, 24]
COOLDOWN_VALUES = [0, 3, 6, 12, 24]
REENTRY_SUPPRESSION_VALUES = [0]
FLIP_SUPPRESSION_VALUES = [0]

from scripts.run_fr2_diagnostics import (
    DAYS,
    HORIZON,
    MODELS_DIR,
    WINDOW_SIZE,
    get_ohlcv_and_proba,
)
from scripts.run_fr2_regime_conditioning import add_regime_columns
from scripts.run_tcn_label_sweep_v2 import run_backtest_7d
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
):
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
        raise RuntimeError("Aligned length too small")
    df_bt = joined[["timestamp", "close", "high", "low"]].copy()
    pl_b = joined["pl_base"].to_numpy(dtype=float)
    ps_b = joined["ps_base"].to_numpy(dtype=float)
    pl_f = joined["pl_fr2"].to_numpy(dtype=float)
    ps_f = joined["ps_fr2"].to_numpy(dtype=float)
    return df_bt, pl_b, ps_b, pl_f, ps_f


def _window_mask(df_bt: pd.DataFrame, window_days: int) -> np.ndarray:
    t = pd.to_datetime(df_bt["timestamp"])
    t_max = t.max()
    if window_days >= 720:
        return np.ones(len(df_bt), dtype=bool)
    start = t_max - pd.Timedelta(days=window_days)
    return (t >= start).to_numpy()


def _compute_holding_bars_from_times(
    df_bt: pd.DataFrame,
    entry_time: str | None,
    exit_time: str | None,
) -> float | None:
    if not entry_time or not exit_time:
        return None
    ts = pd.to_datetime(df_bt["timestamp"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    if len(ts) == 0:
        return None
    e = pd.to_datetime(entry_time, errors="coerce")
    x = pd.to_datetime(exit_time, errors="coerce")
    if pd.isna(e) or pd.isna(x):
        return None
    e64 = e.to_datetime64()
    x64 = x.to_datetime64()

    # Use searchsorted (timestamps are monotonic in df_w) and snap to nearest.
    def _nearest_idx(t64) -> int:
        i = int(np.searchsorted(ts, t64, side="left"))
        if i <= 0:
            return 0
        if i >= len(ts):
            return len(ts) - 1
        # choose closer between i-1 and i
        prev_dt = abs(ts[i - 1] - t64)
        next_dt = abs(ts[i] - t64)
        return i - 1 if prev_dt <= next_dt else i

    ei = _nearest_idx(e64)
    xi = _nearest_idx(x64)
    return float(max(0, xi - ei))


def _extract_trade_stats(trades: List[Any], df_bt: pd.DataFrame) -> Dict[str, float]:
    if not trades:
        return {"win_rate": np.nan, "profit_factor": np.nan, "mean_holding_bars": np.nan, "median_holding_bars": np.nan}
    pnls = []
    holdings = []
    for t in trades:
        p = t.get("profit") if isinstance(t, dict) else getattr(t, "profit", None)
        if p is None:
            p = t.get("net_return", t.get("pnl", 0.0))
        pnls.append(float(p))
        h = t.get("holding_bars") if isinstance(t, dict) else getattr(t, "holding_bars", None)
        if h is None:
            h = t.get("bars_held") if isinstance(t, dict) else getattr(t, "bars_held", None)
        if h is None:
            entry_time = t.get("entry_time") if isinstance(t, dict) else getattr(t, "entry_time", None)
            exit_time = t.get("exit_time") if isinstance(t, dict) else getattr(t, "exit_time", None)
            h = _compute_holding_bars_from_times(df_bt, entry_time, exit_time)
        if h is not None:
            holdings.append(float(h))
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else np.nan
    mean_hold = float(np.mean(holdings)) if holdings else np.nan
    median_hold = float(np.median(holdings)) if holdings else np.nan
    return {
        "win_rate": float(len(wins) / len(pnls)),
        "profit_factor": float(pf) if not np.isnan(pf) else np.nan,
        "mean_holding_bars": mean_hold,
        "median_holding_bars": median_hold,
    }


def main() -> None:
    print("[FR2-EXEC] Loading data...", flush=True)
    triple_base, err_b = get_ohlcv_and_proba(DAYS, BASELINE_PT, "base", False)
    triple_fr2, err_f = get_ohlcv_and_proba(DAYS, FR2_PT, "microstructure_v1", True)
    if err_b or err_f or triple_base is None or triple_fr2 is None:
        raise RuntimeError(f"get_ohlcv_and_proba failed: base={err_b}, fr2={err_f}")

    df_b, pl_b, ps_b, _ = triple_base
    df_f, pl_f, ps_f, _ = triple_fr2
    df_bt, pl_b, ps_b, pl_f, ps_f = _align_by_timestamp(df_b, pl_b, ps_b, df_f, pl_f, ps_f)
    df_bt_reg = add_regime_columns(DAYS, df_bt)
    base_c4 = (
        (df_bt_reg["trend_regime"] == "uptrend")
        & (df_bt_reg["vol_regime"] == "high_vol")
    ).to_numpy()
    c4_mask = build_fr2_c4_mask(base_c4, persistence_bars=6)
    ensemble_inputs = EnsembleInputs(
        pl_base=pl_b, ps_base=ps_b, pl_fr2=pl_f, ps_fr2=ps_f, c4_active=c4_mask
    )
    pl_w, ps_w = build_ensemble_proba(ensemble_inputs, mode="override")

    mask = _window_mask(df_bt, WINDOW_DAYS)
    n_bars = int(mask.sum())
    df_w = df_bt.loc[mask].reset_index(drop=True)
    pl_w = pl_w[mask]
    ps_w = ps_w[mask]

    rows: List[Dict] = []
    for min_hold in MIN_HOLD_VALUES:
        for cooldown in COOLDOWN_VALUES:
            for reentry in REENTRY_SUPPRESSION_VALUES:
                for flip in FLIP_SUPPRESSION_VALUES:
                    try:
                        res_on, _ = run_backtest_7d(
                            SYMBOL,
                            TIMEFRAME,
                            df_w,
                            pl_w,
                            ps_w,
                            commission_rate=COMMISSION,
                            slippage_rate=SLIPPAGE,
                            min_max_proba=THRESHOLD,
                            max_entropy=MAX_ENTROPY,
                            decision_mode="argmax",
                            min_hold=min_hold,
                            cooldown=cooldown,
                            time_stop_enabled=True,
                            time_stop_bars=TIME_STOP_BARS,
                            early_exit_enabled=True,
                            early_exit_bad_k=EARLY_EXIT_BAD_K,
                        )
                    except Exception:
                        res_on = None
                    try:
                        res_off, _ = run_backtest_7d(
                            SYMBOL,
                            TIMEFRAME,
                            df_w,
                            pl_w,
                            ps_w,
                            commission_rate=0.0,
                            slippage_rate=0.0,
                            min_max_proba=THRESHOLD,
                            max_entropy=MAX_ENTROPY,
                            decision_mode="argmax",
                            min_hold=min_hold,
                            cooldown=cooldown,
                            time_stop_enabled=True,
                            time_stop_bars=TIME_STOP_BARS,
                            early_exit_enabled=True,
                            early_exit_bad_k=EARLY_EXIT_BAD_K,
                        )
                    except Exception:
                        res_off = None

                    if res_on is None:
                        cost_on = np.nan
                        trades_count = 0
                        win_rate = profit_factor = np.nan
                        mean_holding_bars = median_holding_bars = np.nan
                    else:
                        cost_on = float(res_on.get("total_return", np.nan))
                        trades_count = int(res_on.get("total_trades", 0))
                        st = _extract_trade_stats(res_on.get("trades", []), df_w)
                        win_rate = st["win_rate"]
                        profit_factor = st["profit_factor"]
                        mean_holding_bars = st["mean_holding_bars"]
                        median_holding_bars = st["median_holding_bars"]
                    if res_off is None:
                        cost_off = np.nan
                    else:
                        cost_off = float(res_off.get("total_return", np.nan))

                    turnover_ratio = trades_count / n_bars if n_bars else np.nan
                    if trades_count > 0:
                        alpha_per_trade = cost_off / trades_count
                        fee_per_trade = (cost_off - cost_on) / trades_count
                        afr = alpha_per_trade / fee_per_trade if abs(fee_per_trade) > 1e-12 else np.nan
                        rtr = fee_per_trade / alpha_per_trade if alpha_per_trade > 1e-12 else np.nan
                    else:
                        alpha_per_trade = fee_per_trade = afr = rtr = np.nan

                    rows.append({
                        "strategy": STRATEGY,
                        "threshold": THRESHOLD,
                        "min_hold": min_hold,
                        "cooldown": cooldown,
                        "reentry_suppression": reentry,
                        "flip_suppression": flip,
                        "trades": trades_count,
                        "turnover_ratio": turnover_ratio,
                        "cost_on": cost_on,
                        "cost_off": cost_off,
                        "profit_factor": profit_factor,
                        "win_rate": win_rate,
                        "mean_holding_bars": mean_holding_bars,
                        "median_holding_bars": median_holding_bars,
                        "alpha_per_trade": alpha_per_trade,
                        "fee_per_trade": fee_per_trade,
                        "alpha_fee_ratio": afr,
                        "required_turnover_reduction": rtr,
                    })
                    print(f"  min_hold={min_hold} cooldown={cooldown} -> trades={trades_count} cost_on={cost_on:.4f} cost_off={cost_off:.4f}", flush=True)

    pd.DataFrame(rows).to_csv(OUT_DIR / "fr2_execution_engineering_results.csv", index=False)
    print("[FR2-EXEC] Wrote fr2_execution_engineering_results.csv", flush=True)
    write_report(rows, n_bars)


def write_report(rows: List[Dict], n_bars: int) -> None:
    df = pd.DataFrame(rows)
    baseline = df[(df["min_hold"] == 0) & (df["cooldown"] == 0)]
    base_row = baseline.iloc[0] if len(baseline) else None

    def _f(x, fmt=".4f"):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        try:
            return format(float(x), fmt)
        except (TypeError, ValueError):
            return "—"

    lines = [
        "# FR2 Execution Engineering Report",
        "",
        "## 1. 목적",
        "",
        "같은 signal(override_ensemble, threshold=0.60)을 더 적게·신중하게 거래하면 execution cost를 이길 수 있는가.",
        "",
        "## 2. 고정 설정",
        "",
        "- strategy: override_ensemble",
        "- threshold: 0.60",
        "- reentry_suppression / flip_suppression: 현재 엔진 미지원 → 0 고정. min_hold / cooldown만 sweep.",
        "",
        "## 3. Baseline (min_hold=0, cooldown=0)",
        "",
    ]
    if base_row is not None:
        lines.append(f"- trades: {int(base_row['trades'])}")
        lines.append(f"- cost_on: {_f(base_row['cost_on'])}")
        lines.append(f"- cost_off: {_f(base_row['cost_off'])}")
        lines.append(f"- alpha_fee_ratio: {_f(base_row['alpha_fee_ratio'], '.3f')}")
        lines.append("")
    lines.append("## 4. Sweep 결과 (min_hold × cooldown)")
    lines.append("")
    lines.append("| min_hold | cooldown | trades | turnover_ratio | cost_on | cost_off | alpha_fee_ratio | required_turnover_reduction |")
    lines.append("|----------|----------|--------|----------------|---------|----------|-----------------|-----------------------------|")
    for _, r in df.iterrows():
        lines.append(
            f"| {int(r['min_hold'])} | {int(r['cooldown'])} | {int(r['trades'])} | {_f(r['turnover_ratio'], '.6f')} | "
            f"{_f(r['cost_on'])} | {_f(r['cost_off'])} | {_f(r['alpha_fee_ratio'], '.3f')} | {_f(r['required_turnover_reduction'], '.2f')} |"
        )
    lines.append("")
    lines.append("## 5. 요약")
    lines.append("")
    best_cost_on = df.loc[df["cost_on"].idxmax()] if df["cost_on"].notna().any() else None
    if best_cost_on is not None and base_row is not None:
        lines.append(f"- Baseline cost_on: {_f(base_row['cost_on'])} (trades={int(base_row['trades'])})")
        lines.append(f"- Best cost_on: {_f(best_cost_on['cost_on'])} at min_hold={int(best_cost_on['min_hold'])}, cooldown={int(best_cost_on['cooldown'])} (trades={int(best_cost_on['trades'])})")
        if float(best_cost_on["cost_on"]) > float(base_row["cost_on"]):
            lines.append("- Execution tuning으로 cost_on이 개선됨.")
        else:
            lines.append("- min_hold/cooldown sweep만으로는 cost_on 개선이 제한적일 수 있음.")
    lines.append("")

    (OUT_DIR / "FR2_EXECUTION_ENGINEERING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("[FR2-EXEC] Wrote FR2_EXECUTION_ENGINEERING_REPORT.md", flush=True)


if __name__ == "__main__":
    main()
