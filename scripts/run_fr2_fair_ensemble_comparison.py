#!/usr/bin/env python3
"""
FR2 + Baseline 공정 비교 재실험.

목적:
1) Baseline vs Ensemble 비교가 execution 설정 차이로 왜곡되었는지 확인 (동일 execution 제약).
2) execution 제약이 실제 signal을 만든 것인지 (Scenario A vs B) 확인 → cost_on + cost_off + spearman + mean_return_signal + turnover.

Execution: run_fr2_diagnostics와 동일 (MIN_MAX_PROBA, MAX_ENTROPY, MIN_HOLD, COOLDOWN, time_stop, early_exit).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "fr2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"

# run_fr2_diagnostics와 동일 execution
COMMISSION = 0.0009
SLIPPAGE = 0.0001
MIN_MAX_PROBA = 0.575
MAX_ENTROPY = 1.30
MIN_HOLD = 36
COOLDOWN = 12
TIME_STOP_BARS = 72
EARLY_EXIT_BAD_K = 8

from scripts.run_fr2_diagnostics import (  # type: ignore
    DAYS,
    HORIZON,
    MODELS_DIR,
    WINDOW_SIZE,
    get_ohlcv_and_proba,
)
from scripts.run_fr2_oos_diagnosis import _metrics  # type: ignore
from scripts.run_fr2_regime_conditioning import add_regime_columns  # type: ignore
from scripts.run_tcn_label_sweep_v2 import run_backtest_7d  # type: ignore
from src.strategies.ensemble_strategy import (
    EnsembleInputs,
    build_ensemble_proba,
    build_fr2_c4_mask,
)

BASELINE_PT = MODELS_DIR / "tcn_h15_t0p004.pt"
FR2_PT = MODELS_DIR / "tcn_h15_micro_v1.pt"

STRATEGIES = [
    ("baseline_only", "baseline_only"),
    ("fr2_c4_only", "fr2_c4_only"),
    ("override", "override_ensemble"),
    ("confirmation", "confirmation_ensemble"),
    ("blend_0.7_0.3", "blend_0.7_0.3"),
    ("blend_0.5_0.5", "blend_0.5_0.5"),
    ("blend_0.3_0.7", "blend_0.3_0.7"),
]
WINDOWS = [720, 180, 90]


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
    if window_days == 720:
        return np.ones(len(df_bt), dtype=bool)
    start = t_max - pd.Timedelta(days=window_days)
    return (t >= start).to_numpy()


def _extract_trade_stats(trades: List[Any]) -> Dict[str, float]:
    if not trades:
        return {
            "avg_pnl_per_trade": np.nan,
            "median_pnl_per_trade": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_consecutive_losses": 0,
        }
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
    max_consec = 0
    cur = 0
    for p in pnls:
        if p < 0:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0
    return {
        "avg_pnl_per_trade": float(np.mean(pnls)),
        "median_pnl_per_trade": float(np.median(pnls)),
        "win_rate": float(len(wins) / len(pnls)),
        "profit_factor": float(profit_factor) if not np.isnan(profit_factor) else np.nan,
        "max_consecutive_losses": int(max_consec),
    }


def _avg_holding_bars(trades: List[Any]) -> float:
    holdings = []
    for t in trades:
        h = t.get("holding_bars") if isinstance(t, dict) else getattr(t, "holding_bars", None)
        if h is None:
            h = t.get("bars_held")
        if h is not None:
            holdings.append(int(h))
    if not holdings:
        return np.nan
    return float(np.mean(holdings))


def _active_ratio(pl: np.ndarray, ps: np.ndarray, threshold: float = 0.60) -> float:
    p_flat = np.clip(1.0 - pl - ps, 0.0, 1.0)
    mx = np.maximum(np.maximum(pl, ps), p_flat)
    return float((mx >= threshold).mean())


def run_fair_comparison() -> None:
    print("[FR2-FAIR] Loading baseline...", flush=True)
    triple_base, err_b = get_ohlcv_and_proba(DAYS, BASELINE_PT, "base", False)
    print("[FR2-FAIR] Loading FR2...", flush=True)
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
    future_ret = future_ret_full[:valid_len]

    df_bt_reg = add_regime_columns(DAYS, df_bt)
    base_c4_mask = (
        (df_bt_reg["trend_regime"] == "uptrend")
        & (df_bt_reg["vol_regime"] == "high_vol")
    ).to_numpy()
    c4_mask = build_fr2_c4_mask(base_c4_mask, persistence_bars=6)
    ensemble_inputs = EnsembleInputs(
        pl_base=pl_b, ps_base=ps_b, pl_fr2=pl_f, ps_fr2=ps_f, c4_active=c4_mask
    )

    strategies = STRATEGIES
    windows = WINDOWS

    def run_bt(df_w: pd.DataFrame, pl_w: np.ndarray, ps_w: np.ndarray, commission: float, slippage: float):
        return run_backtest_7d(
            SYMBOL,
            TIMEFRAME,
            df_w,
            pl_w,
            ps_w,
            commission_rate=commission,
            slippage_rate=slippage,
            min_max_proba=MIN_MAX_PROBA,
            max_entropy=MAX_ENTROPY,
            decision_mode="argmax",
            min_hold=MIN_HOLD,
            cooldown=COOLDOWN,
            time_stop_enabled=True,
            time_stop_bars=TIME_STOP_BARS,
            early_exit_enabled=True,
            early_exit_bad_k=EARLY_EXIT_BAD_K,
        )

    rows_main: List[Dict] = []
    rows_signal: List[Dict] = []
    rows_turnover: List[Dict] = []

    for mode, name in strategies:
        print(f"[FR2-FAIR] Strategy: {name}", flush=True)
        pl_mode, ps_mode = build_ensemble_proba(ensemble_inputs, mode=mode)  # type: ignore[arg-type]

        for window_days in windows:
            mask = _window_mask(df_bt, window_days)
            if mask.sum() < 500:
                continue
            n_bars = int(mask.sum())
            df_w = df_bt.loc[mask].reset_index(drop=True)
            pl_w = pl_mode[mask]
            ps_w = ps_mode[mask]
            fr_w = future_ret[mask]

            met = _metrics(pl_w, ps_w, fr_w, threshold_signal=0.60)
            rows_signal.append({
                "strategy": name,
                "mode": mode,
                "window_days": window_days,
                "spearman": met.get("spearman", np.nan),
                "mean_return_signal": met.get("mean_return_signal", np.nan),
                "direction_accuracy": met.get("direction_accuracy", np.nan),
                "signal_density": met.get("signal_density", np.nan),
            })

            try:
                res_on, err_on = run_bt(df_w, pl_w, ps_w, COMMISSION, SLIPPAGE)
            except Exception as e:
                res_on, err_on = None, str(e)
            try:
                res_off, err_off = run_bt(df_w, pl_w, ps_w, 0.0, 0.0)
            except Exception as e:
                res_off, err_off = None, str(e)

            def fill(res: Any, cost_key: str) -> Dict:
                if res is None:
                    return {
                        cost_key: np.nan,
                        "MDD": np.nan,
                        "trades": 0,
                        "win_rate": np.nan,
                        "avg_pnl_per_trade": np.nan,
                        "median_pnl_per_trade": np.nan,
                        "profit_factor": np.nan,
                        "max_consecutive_losses": 0,
                        "avg_holding_bars": np.nan,
                    }
                trades_list = res.get("trades", [])
                stats = _extract_trade_stats(trades_list)
                return {
                    cost_key: float(res.get("total_return", np.nan)),
                    "MDD": float(res.get("max_drawdown", np.nan)),
                    "trades": int(res.get("total_trades", 0)),
                    "win_rate": float(res.get("win_rate", np.nan)),
                    "avg_pnl_per_trade": stats["avg_pnl_per_trade"],
                    "median_pnl_per_trade": stats["median_pnl_per_trade"],
                    "profit_factor": stats["profit_factor"],
                    "max_consecutive_losses": stats["max_consecutive_losses"],
                    "avg_holding_bars": _avg_holding_bars(trades_list),
                }

            on = fill(res_on, "cost_on")
            off = fill(res_off, "cost_off")
            trades_on = on["trades"]
            trades_off = off["trades"]

            row = {
                "strategy": name,
                "mode": mode,
                "window_days": window_days,
                "cost_on": on["cost_on"],
                "cost_off": off["cost_off"],
                "MDD_on": on["MDD"],
                "MDD_off": off["MDD"],
                "trades_on": trades_on,
                "trades_off": trades_off,
                "win_rate_on": on["win_rate"],
                "win_rate_off": off["win_rate"],
                "avg_pnl_per_trade_on": on["avg_pnl_per_trade"],
                "median_pnl_per_trade_on": on["median_pnl_per_trade"],
                "profit_factor_on": on["profit_factor"],
                "max_consecutive_losses_on": on["max_consecutive_losses"],
                "active_ratio": _active_ratio(pl_w, ps_w, 0.60),
                "sample_too_small": "yes" if trades_on < 100 else "no",
            }
            rows_main.append(row)

            turnover_ratio_on = trades_on / n_bars if n_bars else np.nan
            turnover_ratio_off = trades_off / n_bars if n_bars else np.nan
            rows_turnover.append({
                "strategy": name,
                "mode": mode,
                "window_days": window_days,
                "n_bars": n_bars,
                "trades_on": trades_on,
                "trades_off": trades_off,
                "turnover_ratio_on": turnover_ratio_on,
                "turnover_ratio_off": turnover_ratio_off,
                "avg_holding_bars_on": on["avg_holding_bars"],
                "avg_holding_bars_off": off["avg_holding_bars"],
            })

            co = on["cost_on"]; cf = off["cost_off"]
            co_s = f"{float(co):.4f}" if co is not None and not (isinstance(co, float) and np.isnan(co)) else "n/a"
            cf_s = f"{float(cf):.4f}" if cf is not None and not (isinstance(cf, float) and np.isnan(cf)) else "n/a"
            print(f"  {name} {window_days}d -> cost_on={co_s} cost_off={cf_s} trades={trades_on}", flush=True)

    pd.DataFrame(rows_main).to_csv(OUT_DIR / "fr2_ensemble_fair_comparison.csv", index=False)
    pd.DataFrame(rows_signal).to_csv(OUT_DIR / "fr2_signal_edge_metrics.csv", index=False)
    pd.DataFrame(rows_turnover).to_csv(OUT_DIR / "fr2_turnover_summary.csv", index=False)
    print("[FR2-FAIR] Wrote fr2_ensemble_fair_comparison.csv, fr2_signal_edge_metrics.csv, fr2_turnover_summary.csv", flush=True)


def write_report_and_verdict() -> None:
    p_main = OUT_DIR / "fr2_ensemble_fair_comparison.csv"
    p_sig = OUT_DIR / "fr2_signal_edge_metrics.csv"
    p_turn = OUT_DIR / "fr2_turnover_summary.csv"
    if not p_main.exists():
        (OUT_DIR / "FR2_FAIR_ENSEMBLE_COMPARISON_REPORT.md").write_text(
            "# FR2 + Baseline 공정 비교 재실험 보고서\n\n"
            "No data. Run:\n  .venv/bin/python scripts/run_fr2_fair_ensemble_comparison.py\n"
            "(Full run: 30–60 min. Use --quick for 90d + 2 strategies only.)\n",
            encoding="utf-8",
        )
        return

    df = pd.read_csv(p_main)
    df_sig = pd.read_csv(p_sig) if p_sig.exists() else pd.DataFrame()
    df_turn = pd.read_csv(p_turn) if p_turn.exists() else pd.DataFrame()

    lines: List[str] = [
        "# FR2 + Baseline 공정 비교 재실험 보고서",
        "",
        "## 1. 실행 조건 (run_fr2_diagnostics와 동일)",
        "",
        "- MIN_MAX_PROBA=0.575, MAX_ENTROPY=1.30",
        "- MIN_HOLD=36, COOLDOWN=12",
        "- time_stop_enabled=True, time_stop_bars=72",
        "- early_exit_enabled=True, early_exit_bad_k=8",
        "",
        "## 2. cost_on vs cost_off (동일 execution)",
        "",
        "| strategy | window_days | cost_on | cost_off | trades_on | trades_off | win_rate_on | profit_factor_on | sample_too_small |",
        "|----------|-------------|---------|----------|-----------|------------|-------------|-------------------|------------------|",
    ]
    def _f(x, fmt=".4f"):
        if pd.isna(x):
            return "—"
        try:
            return format(float(x), fmt)
        except (TypeError, ValueError):
            return "—"
    for _, r in df.iterrows():
        lines.append(
            f"| {r['strategy']} | {r['window_days']} | {_f(r['cost_on'])} | {_f(r['cost_off'])} | "
            f"{int(r['trades_on'])} | {int(r['trades_off'])} | "
            f"{_f(r['win_rate_on'])} | "
            f"{_f(r['profit_factor_on'], '.3f')} | {r['sample_too_small']} |"
        )
    lines.append("")

    if not df_sig.empty:
        lines.append("## 3. Raw signal edge (spearman, mean_return_signal)")
        lines.append("")
        lines.append("| strategy | window_days | spearman | mean_return_signal | direction_accuracy | signal_density |")
        lines.append("|----------|-------------|----------|-------------------|--------------------|----------------|")
        for _, r in df_sig.iterrows():
            lines.append(
                f"| {r['strategy']} | {r['window_days']} | {_f(r['spearman'])} | "
                f"{_f(r['mean_return_signal'], '.6f')} | {_f(r['direction_accuracy'])} | {_f(r['signal_density'])} |"
            )
        lines.append("")

    if not df_turn.empty:
        lines.append("## 4. Turnover 분석 (720d 기준 요약)")
        lines.append("")
        sub = df_turn[df_turn["window_days"] == 720]
        lines.append("| strategy | n_bars | trades_on | turnover_ratio_on | avg_holding_bars_on |")
        lines.append("|----------|--------|-----------|-------------------|----------------------|")
        for _, r in sub.iterrows():
            ah = r.get("avg_holding_bars_on")
            ah_str = "—" if (pd.isna(ah) or (isinstance(ah, float) and np.isnan(ah))) else f"{float(ah):.1f}"
            lines.append(f"| {r['strategy']} | {int(r['n_bars'])} | {int(r['trades_on'])} | {float(r['turnover_ratio_on']):.6f} | {ah_str} |")
        lines.append("")

    lines.append("## 5. 핵심 비교 질문 요약")
    lines.append("")
    lines.append("1) **동일 execution 제약에서 baseline vs ensemble 성과**: 위 표 2 참조 (cost_on / cost_off, trades_on).")
    lines.append("2) **cost_off에서도 ensemble이 baseline보다 우위인가**: cost_off 열 비교 (baseline_only vs fr2_c4_only, override_ensemble 등).")
    lines.append("3) **baseline vs FR2 signal edge**: 위 표 3 (spearman, mean_return_signal) 참조.")
    lines.append("4) **execution 제약이 turnover에 미친 영향**: 표 4 (turnover_ratio_on, avg_holding_bars_on) 참조.")
    lines.append("5) **Scenario A vs B**: 아래 판정 참조.")
    lines.append("")

    # Scenario A vs B 판정
    lines.append("## 6. Scenario A vs B 판정")
    lines.append("")
    baseline_720 = df[(df["strategy"] == "baseline_only") & (df["window_days"] == 720)]
    if len(baseline_720):
        cost_on_b = float(baseline_720["cost_on"].iloc[0])
        cost_off_b = float(baseline_720["cost_off"].iloc[0])
        trades_b = int(baseline_720["trades_on"].iloc[0])
        if not df_sig.empty:
            sig_b = df_sig[(df_sig["strategy"] == "baseline_only") & (df_sig["window_days"] == 720)]
            spearman_b = float(sig_b["spearman"].iloc[0]) if len(sig_b) else np.nan
            mrs_b = float(sig_b["mean_return_signal"].iloc[0]) if len(sig_b) else np.nan
        else:
            spearman_b = mrs_b = np.nan
        if cost_off_b > 0 and spearman_b > 0:
            scenario = "SIGNAL_REAL_EXECUTION_NOISE"
            desc = "cost_off > 0 및 spearman > 0 → signal 존재, execution 제약이 noise 제거."
        elif cost_off_b <= 0 and cost_on_b > cost_off_b:
            scenario = "PSEUDO_ALPHA_FROM_EXECUTION"
            desc = "cost_off ≤ 0 인데 cost_on이 더 나음 → execution 제약이 trade selection으로 pseudo alpha 가능성."
        else:
            scenario = "ENSEMBLE_CONDITIONAL_ALPHA"
            desc = "FR2/ensemble이 conditional alpha 역할; baseline signal은 약함."
    else:
        scenario = "ENSEMBLE_CONDITIONAL_ALPHA"
        desc = "baseline 720d 데이터 없음; ensemble 결과 기준."
    lines.append(f"**판정: {scenario}**")
    lines.append("")
    lines.append(desc)
    lines.append("")
    lines.append("(SIGNAL_REAL_EXECUTION_NOISE | PSEUDO_ALPHA_FROM_EXECUTION | ENSEMBLE_CONDITIONAL_ALPHA)")
    lines.append("")

    (OUT_DIR / "FR2_FAIR_ENSEMBLE_COMPARISON_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("[FR2-FAIR] Wrote FR2_FAIR_ENSEMBLE_COMPARISON_REPORT.md", flush=True)


def main() -> None:
    import argparse
    global WINDOWS, STRATEGIES
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="90d only, baseline_only + fr2_c4_only (2 strategies)")
    args = p.parse_args()
    if args.quick:
        STRATEGIES = [("baseline_only", "baseline_only"), ("fr2_c4_only", "fr2_c4_only")]
        WINDOWS = [90]
    print("[FR2-FAIR] Loading data and running fair comparison (same execution, cost_on + cost_off)...", flush=True)
    run_fair_comparison()
    write_report_and_verdict()
    print("[FR2-FAIR] Done.", flush=True)


if __name__ == "__main__":
    main()
