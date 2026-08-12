#!/usr/bin/env python3
"""
FR2 calibration 실험.

목적: softmax 확률의 poor calibration 확인 및 calibration 후 threshold effectiveness 개선 검증.
- Time-based hold-out: fit = 앞 60%, eval = 뒤 40%. 성능 보고는 eval만.
- 방법: Temperature scaling, Platt scaling, Isotonic regression (max probability calibration).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
STRATEGIES = [("override", "override_ensemble"), ("confirmation", "confirmation_ensemble")]
WINDOW_DAYS = 720
FIT_RATIO = 0.6
MIN_TRADES_SAMPLE = 100
TEMPERATURE_CANDIDATES = [1.0, 1.5, 2.0, 2.5, 3.0]

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
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _score_series(pl: np.ndarray, ps: np.ndarray) -> np.ndarray:
    p_flat = np.clip(1.0 - pl - ps, 0.0, 1.0)
    return np.maximum(np.maximum(pl, ps), p_flat)


def _apply_calibrated_score_to_proba(
    pl: np.ndarray, ps: np.ndarray, score_cal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Rescale (pl, ps) so that max(pl_new, ps_new, p_flat_new) = score_cal and argmax is unchanged."""
    p_flat = np.clip(1.0 - pl - ps, 0.0, 1.0)
    stack = np.stack([pl, ps, p_flat], axis=1)
    idx = np.argmax(stack, axis=1)
    pl_new = np.copy(pl)
    ps_new = np.copy(ps)
    score_cal = np.clip(score_cal, 1e-6, 1.0 - 1e-6)
    rest = 1.0 - score_cal
    for k in (0, 1, 2):
        mask = idx == k
        if not np.any(mask):
            continue
        other_sum = rest[mask]
        if k == 0:
            pl_new[mask] = score_cal[mask]
            denom = ps[mask] + p_flat[mask]
            denom = np.where(denom < 1e-12, 1.0, denom)
            ps_new[mask] = other_sum * ps[mask] / denom
        elif k == 1:
            ps_new[mask] = score_cal[mask]
            denom = pl[mask] + p_flat[mask]
            denom = np.where(denom < 1e-12, 1.0, denom)
            pl_new[mask] = other_sum * pl[mask] / denom
        else:
            pl_new[mask] = other_sum * pl[mask] / (pl[mask] + ps[mask] + 1e-12)
            ps_new[mask] = other_sum * ps[mask] / (pl[mask] + ps[mask] + 1e-12)
    return pl_new, ps_new


def _ece(score: np.ndarray, outcome: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: mean over bins of |conf - acc|."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (score >= lo) & (score < hi) if i < n_bins - 1 else (score >= lo) & (score <= hi)
        if mask.sum() < 5:
            continue
        conf = score[mask].mean()
        acc = outcome[mask].mean()
        ece += mask.sum() * abs(conf - acc)
    return float(ece / len(score)) if len(score) else 0.0


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
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else np.nan
    return {"win_rate": float(len(wins) / len(pnls)), "profit_factor": float(pf) if not np.isnan(pf) else np.nan}


def run_backtest_for_sweep(
    df_w: pd.DataFrame,
    pl_w: np.ndarray,
    ps_w: np.ndarray,
    th: float,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    try:
        res_on, _ = run_backtest_7d(
            SYMBOL, TIMEFRAME, df_w, pl_w, ps_w,
            commission_rate=COMMISSION, slippage_rate=SLIPPAGE,
            min_max_proba=th, max_entropy=MAX_ENTROPY, decision_mode="argmax",
            min_hold=MIN_HOLD, cooldown=COOLDOWN,
            time_stop_enabled=True, time_stop_bars=TIME_STOP_BARS,
            early_exit_enabled=True, early_exit_bad_k=EARLY_EXIT_BAD_K,
        )
    except Exception:
        res_on = None
    try:
        res_off, _ = run_backtest_7d(
            SYMBOL, TIMEFRAME, df_w, pl_w, ps_w,
            commission_rate=0.0, slippage_rate=0.0,
            min_max_proba=th, max_entropy=MAX_ENTROPY, decision_mode="argmax",
            min_hold=MIN_HOLD, cooldown=COOLDOWN,
            time_stop_enabled=True, time_stop_bars=TIME_STOP_BARS,
            early_exit_enabled=True, early_exit_bad_k=EARLY_EXIT_BAD_K,
        )
    except Exception:
        res_off = None
    return res_on, res_off


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression

    print("[FR2-CAL] Loading data (temperature=1.0)...", flush=True)
    triple_base, err_b = get_ohlcv_and_proba(DAYS, BASELINE_PT, "base", False, temperature=1.0)
    triple_fr2, err_f = get_ohlcv_and_proba(DAYS, FR2_PT, "microstructure_v1", True, temperature=1.0)
    if err_b or err_f or triple_base is None or triple_fr2 is None:
        raise RuntimeError(f"get_ohlcv_and_proba failed: base={err_b}, fr2={err_f}")

    df_b, pl_b, ps_b, _ = triple_base
    df_f, pl_f, ps_f, _ = triple_fr2
    df_bt, pl_b, ps_b, pl_f, ps_f = _align_by_timestamp(df_b, pl_b, ps_b, df_f, pl_f, ps_f)
    n_total = len(df_bt)
    valid_len = n_total - WINDOW_SIZE - HORIZON
    if valid_len <= 0:
        raise RuntimeError(f"valid_len={valid_len}")
    fit_end = int(n_total * FIT_RATIO)
    fit_end_v = min(fit_end, valid_len)
    eval_start = fit_end
    n_eval = n_total - fit_end

    df_bt_reg = add_regime_columns(DAYS, df_bt)
    base_c4 = (
        (df_bt_reg["trend_regime"] == "uptrend")
        & (df_bt_reg["vol_regime"] == "high_vol")
    ).to_numpy()
    c4_mask = build_fr2_c4_mask(base_c4, persistence_bars=6)
    ensemble_inputs = EnsembleInputs(
        pl_base=pl_b, ps_base=ps_b, pl_fr2=pl_f, ps_fr2=ps_f, c4_active=c4_mask
    )

    df_eval = df_bt.iloc[eval_start:].reset_index(drop=True)
    n_bars_eval = len(df_eval)

    rows_dist: List[Dict] = []
    rows_sweep: List[Dict] = []
    rows_params: List[Dict] = []

    def run_sweep_on_proba(
        name: str,
        pl_eval: np.ndarray,
        ps_eval: np.ndarray,
        cal_method: str,
    ) -> None:
        score_eval = _score_series(pl_eval, ps_eval)
        for th in THRESHOLDS:
            above = (score_eval >= th).astype(float)
            signal_density = float(above.mean())
            bar_count = int(above.sum())
            rows_dist.append({
                "strategy": name,
                "calibration_method": cal_method,
                "threshold": th,
                "signal_density": signal_density,
                "bar_count_above_threshold": bar_count,
            })
            res_on, res_off = run_backtest_for_sweep(df_eval, pl_eval, ps_eval, th)
            cost_on = float(res_on["total_return"]) if res_on else np.nan
            cost_off = float(res_off["total_return"]) if res_off else np.nan
            trades = int(res_on["total_trades"]) if res_on else 0
            if res_on:
                st = _extract_trade_stats(res_on.get("trades", []))
                win_rate = st["win_rate"]
                profit_factor = st["profit_factor"]
            else:
                win_rate = profit_factor = np.nan
            turnover_ratio = trades / n_bars_eval if n_bars_eval else np.nan
            if trades > 0:
                alpha_per_trade = cost_off / trades
                fee_per_trade = (cost_off - cost_on) / trades
                afr = alpha_per_trade / fee_per_trade if abs(fee_per_trade) > 1e-12 else np.nan
                rtr = fee_per_trade / alpha_per_trade if alpha_per_trade > 1e-12 else np.nan
            else:
                alpha_per_trade = fee_per_trade = afr = rtr = np.nan
            rows_sweep.append({
                "strategy": name,
                "calibration_method": cal_method,
                "threshold": th,
                "trades": trades,
                "turnover_ratio": turnover_ratio,
                "cost_on": cost_on,
                "cost_off": cost_off,
                "profit_factor": profit_factor,
                "win_rate": win_rate,
                "alpha_per_trade": alpha_per_trade,
                "fee_per_trade": fee_per_trade,
                "alpha_fee_ratio": afr,
                "required_turnover_reduction": rtr,
                "sample_too_small": trades < MIN_TRADES_SAMPLE,
            })

    # ----- Uncalibrated baseline (eval only) -----
    print("[FR2-CAL] Uncalibrated baseline on eval window...", flush=True)
    for mode, name in STRATEGIES:
        pl_full, ps_full = build_ensemble_proba(ensemble_inputs, mode=mode)  # type: ignore[arg-type]
        pl_eval = pl_full[eval_start:]
        ps_eval = ps_full[eval_start:]
        run_sweep_on_proba(name, pl_eval, ps_eval, "uncalibrated")

    # ----- Temperature scaling: fit T on fit window -----
    print("[FR2-CAL] Temperature scaling: fitting T on fit window...", flush=True)
    best_T = 1.0
    best_ece = float("inf")
    close = df_bt["close"].to_numpy(dtype=float)
    base = close[WINDOW_SIZE : WINDOW_SIZE + valid_len]
    fut = close[WINDOW_SIZE + HORIZON : WINDOW_SIZE + valid_len + HORIZON]
    fit_future_ret = (fut / base - 1.0)[:valid_len]
    outcome_fit = (fit_future_ret[:fit_end_v] > 0).astype(float) if fit_end_v > 0 else np.zeros(fit_end_v)

    for T in TEMPERATURE_CANDIDATES:
        if T == 1.0:
            pl_b_t, ps_b_t = pl_b, ps_b
            pl_f_t, ps_f_t = pl_f, ps_f
        else:
            t_base, _ = get_ohlcv_and_proba(DAYS, BASELINE_PT, "base", False, temperature=T)
            t_fr2, _ = get_ohlcv_and_proba(DAYS, FR2_PT, "microstructure_v1", True, temperature=T)
            if t_base is None or t_fr2 is None:
                continue
            _, pl_b_t, ps_b_t, pl_f_t, ps_f_t = _align_by_timestamp(
                t_base[0], t_base[1], t_base[2], t_fr2[0], t_fr2[1], t_fr2[2]
            )
        inp_t = EnsembleInputs(pl_base=pl_b_t, ps_base=ps_b_t, pl_fr2=pl_f_t, ps_fr2=ps_f_t, c4_active=c4_mask)
        pl_ov, ps_ov = build_ensemble_proba(inp_t, mode="override")  # type: ignore[arg-type]
        score_fit = _score_series(pl_ov[:fit_end_v], ps_ov[:fit_end_v])
        oc = outcome_fit[: len(score_fit)] if len(outcome_fit) >= len(score_fit) else np.pad(outcome_fit, (0, max(0, len(score_fit) - len(outcome_fit))), constant_values=0)
        ece = _ece(score_fit, oc, n_bins=10)
        if ece < best_ece:
            best_ece = ece
            best_T = T
    print(f"[FR2-CAL] Best temperature T={best_T} (ECE={best_ece:.6f})", flush=True)
    rows_params.append({
        "calibration_method": "temperature_scaled",
        "fit_window_start": 0,
        "fit_window_end": fit_end,
        "eval_window_start": eval_start,
        "eval_window_end": n_total,
        "temperature_value": best_T,
        "platt_params": "",
        "isotonic_notes": "",
    })

    if best_T != 1.0:
        t_base, _ = get_ohlcv_and_proba(DAYS, BASELINE_PT, "base", False, temperature=best_T)
        t_fr2, _ = get_ohlcv_and_proba(DAYS, FR2_PT, "microstructure_v1", True, temperature=best_T)
        if t_base is not None and t_fr2 is not None:
            _, pl_b_t, ps_b_t, pl_f_t, ps_f_t = _align_by_timestamp(
                t_base[0], t_base[1], t_base[2], t_fr2[0], t_fr2[1], t_fr2[2]
            )
            inp_t = EnsembleInputs(pl_base=pl_b_t, ps_base=ps_b_t, pl_fr2=pl_f_t, ps_fr2=ps_f_t, c4_active=c4_mask)
            for mode, name in STRATEGIES:
                pl_full, ps_full = build_ensemble_proba(inp_t, mode=mode)  # type: ignore[arg-type]
                pl_eval = pl_full[eval_start:]
                ps_eval = ps_full[eval_start:]
                run_sweep_on_proba(name, pl_eval, ps_eval, "temperature_scaled")
    else:
        for mode, name in STRATEGIES:
            pl_full, ps_full = build_ensemble_proba(ensemble_inputs, mode=mode)  # type: ignore[arg-type]
            pl_eval = pl_full[eval_start:]
            ps_eval = ps_full[eval_start:]
            run_sweep_on_proba(name, pl_eval, ps_eval, "temperature_scaled")

    # ----- Platt scaling (max proba -> binary direction on fit) -----
    print("[FR2-CAL] Platt scaling on fit window...", flush=True)
    platt_override: Optional[LogisticRegression] = None
    platt_confirmation: Optional[LogisticRegression] = None
    for mode, name in STRATEGIES:
        pl_full, ps_full = build_ensemble_proba(ensemble_inputs, mode=mode)  # type: ignore[arg-type]
        score_fit = _score_series(pl_full[:fit_end_v], ps_full[:fit_end_v])
        outcome = (fit_future_ret[:fit_end_v] > 0).astype(float) if fit_end_v > 0 else np.zeros(fit_end_v)
        logit_score = np.clip(score_fit, 1e-6, 1.0 - 1e-6)
        logit_score = np.log(logit_score / (1 - logit_score)).reshape(-1, 1)
        lr = LogisticRegression(C=1e10, max_iter=500)
        lr.fit(logit_score, outcome)
        if mode == "override":
            platt_override = lr
        else:
            platt_confirmation = lr

    def platt_transform(score: np.ndarray, lr: LogisticRegression) -> np.ndarray:
        s = np.clip(score, 1e-6, 1.0 - 1e-6)
        logit = np.log(s / (1 - s)).reshape(-1, 1)
        return lr.predict_proba(logit)[:, 1]

    for mode, name in STRATEGIES:
        pl_full, ps_full = build_ensemble_proba(ensemble_inputs, mode=mode)  # type: ignore[arg-type]
        pl_eval = pl_full[eval_start:]
        ps_eval = ps_full[eval_start:]
        score_eval = _score_series(pl_eval, ps_eval)
        lr = platt_override if mode == "override" else platt_confirmation
        if lr is not None:
            score_cal = platt_transform(score_eval, lr)
            pl_cal, ps_cal = _apply_calibrated_score_to_proba(pl_eval, ps_eval, score_cal)
            run_sweep_on_proba(name, pl_cal, ps_cal, "platt_scaled")
        else:
            run_sweep_on_proba(name, pl_eval, ps_eval, "platt_scaled")

    rows_params.append({
        "calibration_method": "platt_scaled",
        "fit_window_start": 0,
        "fit_window_end": fit_end,
        "eval_window_start": eval_start,
        "eval_window_end": n_total,
        "temperature_value": "",
        "platt_params": "LogisticRegression(C=1e10) on logit(max_proba) vs direction",
        "isotonic_notes": "",
    })

    # ----- Isotonic regression -----
    print("[FR2-CAL] Isotonic regression on fit window...", flush=True)
    iso_override: Optional[IsotonicRegression] = None
    iso_confirmation: Optional[IsotonicRegression] = None
    for mode, name in STRATEGIES:
        pl_full, ps_full = build_ensemble_proba(ensemble_inputs, mode=mode)  # type: ignore[arg-type]
        score_fit = _score_series(pl_full[:fit_end_v], ps_full[:fit_end_v])
        outcome = (fit_future_ret[:fit_end_v] > 0).astype(float) if fit_end_v > 0 else np.zeros(fit_end_v)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(score_fit, outcome)
        if mode == "override":
            iso_override = iso
        else:
            iso_confirmation = iso

    for mode, name in STRATEGIES:
        pl_full, ps_full = build_ensemble_proba(ensemble_inputs, mode=mode)  # type: ignore[arg-type]
        pl_eval = pl_full[eval_start:]
        ps_eval = ps_full[eval_start:]
        score_eval = _score_series(pl_eval, ps_eval)
        iso = iso_override if mode == "override" else iso_confirmation
        if iso is not None:
            score_cal = iso.predict(score_eval)
            score_cal = np.clip(score_cal, 0.0, 1.0)
            pl_cal, ps_cal = _apply_calibrated_score_to_proba(pl_eval, ps_eval, score_cal)
            run_sweep_on_proba(name, pl_cal, ps_cal, "isotonic_scaled")
        else:
            run_sweep_on_proba(name, pl_eval, ps_eval, "isotonic_scaled")

    rows_params.append({
        "calibration_method": "isotonic_scaled",
        "fit_window_start": 0,
        "fit_window_end": fit_end,
        "eval_window_start": eval_start,
        "eval_window_end": n_total,
        "temperature_value": "",
        "platt_params": "",
        "isotonic_notes": "IsotonicRegression on (max_proba, future_return); out_of_bounds=clip",
    })

    pd.DataFrame(rows_dist).to_csv(OUT_DIR / "fr2_calibration_distribution.csv", index=False)
    pd.DataFrame(rows_sweep).to_csv(OUT_DIR / "fr2_calibration_threshold_sweep.csv", index=False)
    pd.DataFrame(rows_params).to_csv(OUT_DIR / "fr2_calibration_params.csv", index=False)
    print("[FR2-CAL] Wrote fr2_calibration_distribution.csv, fr2_calibration_threshold_sweep.csv, fr2_calibration_params.csv", flush=True)

    # ----- Report and verdict -----
    write_report(rows_dist, rows_sweep, rows_params, best_T)


def write_report(
    rows_dist: List[Dict],
    rows_sweep: List[Dict],
    rows_params: List[Dict],
    best_temperature: float,
) -> None:
    df_dist = pd.DataFrame(rows_dist)
    df_sweep = pd.DataFrame(rows_sweep)
    df_params = pd.DataFrame(rows_params)

    def _f(x, fmt=".4f"):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        try:
            return format(float(x), fmt)
        except (TypeError, ValueError):
            return "—"

    lines: List[str] = [
        "# FR2 Calibration Report",
        "",
        "## 1. What was calibrated",
        "",
        "- **Target**: max(pl, ps, p_flat) (max probability).",
        "- **Temperature scaling**: softmax(logits/T) in inference; T fitted on fit window (ECE).",
        "- **Platt scaling**: LogisticRegression on logit(max_proba) vs binary direction (future_ret>0) on fit.",
        "- **Isotonic regression**: IsotonicRegression on (max_proba, future_return) on fit; out_of_bounds=clip.",
        "",
        "## 2. Time split",
        "",
        "- Fit window = first 60% of bars (calibrator fitting only).",
        "- Eval window = last 40% of bars (all performance reported here only). No future leakage.",
        "",
        "## 3. Calibration parameters",
        "",
    ]
    for _, r in df_params.iterrows():
        lines.append(f"- **{r['calibration_method']}**: {r.get('temperature_value', '')} {r.get('platt_params', '')} {r.get('isotonic_notes', '')}")
    lines.append("")
    lines.append("## 4. Distribution comparison (eval window)")
    lines.append("")
    lines.append("| strategy | calibration_method | threshold | signal_density | bar_count_above_threshold |")
    lines.append("|----------|-------------------|-----------|----------------|---------------------------|")
    for _, r in df_dist.iterrows():
        lines.append(f"| {r['strategy']} | {r['calibration_method']} | {r['threshold']} | {_f(r['signal_density'])} | {int(r['bar_count_above_threshold'])} |")
    lines.append("")
    lines.append("## 5. Threshold sweep comparison (eval)")
    lines.append("")
    lines.append("| strategy | calibration_method | threshold | trades | cost_on | cost_off | alpha_fee_ratio | sample_too_small |")
    lines.append("|----------|-------------------|-----------|--------|---------|----------|-----------------|------------------|")
    for _, r in df_sweep.iterrows():
        lines.append(
            f"| {r['strategy']} | {r['calibration_method']} | {r['threshold']} | {int(r['trades'])} | "
            f"{_f(r['cost_on'])} | {_f(r['cost_off'])} | {_f(r['alpha_fee_ratio'], '.3f')} | {r['sample_too_small']} |"
        )
    lines.append("")
    lines.append("## 6. Statistical cautions")
    lines.append("")
    lines.append("- sample_too_small = True where trades < 100.")
    lines.append("- Isotonic can overfit on fit window; eval is held-out.")
    lines.append("- Calibration target is max probability only; direction (pl/ps) unchanged except rescaling.")
    lines.append("")
    lines.append("## 7. Final recommendation and verdict")
    lines.append("")

    uncal = df_sweep[df_sweep["calibration_method"] == "uncalibrated"]
    cal = df_sweep[df_sweep["calibration_method"] != "uncalibrated"]
    any_improve = False
    any_ratio_improve = False
    any_cost_improve = False
    for _, r in cal.iterrows():
        key = (r["strategy"], r["threshold"])
        u = uncal[(uncal["strategy"] == r["strategy"]) & (uncal["threshold"] == r["threshold"])]
        if len(u) == 0:
            continue
        u = u.iloc[0]
        if pd.notna(r["alpha_fee_ratio"]) and pd.notna(u["alpha_fee_ratio"]) and r["alpha_fee_ratio"] > u["alpha_fee_ratio"]:
            any_ratio_improve = True
        if pd.notna(r["cost_on"]) and pd.notna(u["cost_on"]) and r["cost_on"] > u["cost_on"]:
            any_cost_improve = True
    cost_on_positive = (df_sweep["cost_on"] >= 0) & (df_sweep["trades"] >= MIN_TRADES_SAMPLE)
    if cost_on_positive.any():
        verdict = "CALIBRATION_IMPROVES_THRESHOLDING"
        desc = "Some (strategy, calibration, threshold) achieve cost_on ≥ 0 with trades ≥ 100."
    elif any_ratio_improve or any_cost_improve:
        verdict = "CALIBRATION_HELPS_BUT_NOT_ENOUGH"
        desc = "Distribution/alpha_fee_ratio or cost_on improved but execution engineering still needed."
    else:
        verdict = "CALIBRATION_NO_MEANINGFUL_GAIN"
        desc = "No meaningful distribution or alpha_fee_ratio improvement from calibration."
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    lines.append(desc)
    lines.append("")

    (OUT_DIR / "FR2_CALIBRATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("[FR2-CAL] Wrote FR2_CALIBRATION_REPORT.md", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Skip temperature search; uncalibrated + platt only")
    args = p.parse_args()
    if args.quick:
        TEMPERATURE_CANDIDATES[:] = [1.0]
        # Optional: reduce to one strategy for speed
    main()
