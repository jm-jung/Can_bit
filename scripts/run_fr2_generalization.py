#!/usr/bin/env python3
"""
FR2 Generalization: 후보 전략(A,B,C,D)이 다른 기간·보수적 비용에서 유지되는지 검증.

- 2.1 Baseline: 720d 전체에서 A,B,C,D(,D2) 재실행
- 2.2 기간 분할: 720d, 최근 180d, 최근 90d, rolling 60d (step 30d)
- 2.4 보수적 비용: fee +20~30%, slippage 증가
- 2.3 Multi-symbol: 현재 BTCUSDT만 지원 (ETH/SOL은 데이터·모델 확장 시 추가)

출력: fr2_generalization_results.csv, FR2_GENERALIZATION_REPORT.md (구간별 테이블, PASS/FAIL, 추천)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
COMMISSION_CONSERVATIVE = 0.00115   # +~28%
SLIPPAGE_CONSERVATIVE = 0.0002
MAX_ENTROPY = 1.30
TIME_STOP_BARS = 72
EARLY_EXIT_BAD_K = 8
STRATEGY = "override_ensemble"
MIN_HOLD = 24
COOLDOWN = 24
DAYS_FULL = 720
ROLLING_WINDOW_DAYS = 60
ROLLING_STEP_DAYS = 30

# 후보: (id, threshold, regime_c4_only)
CANDIDATES = [
    ("A", 0.64, False),
    ("B", 0.66, False),
    ("C", 0.60, True),
    ("D", 0.64, True),
    ("D2", 0.66, True),
]

# PASS 기준
PASS_MIN_COST_ON = -0.2
PASS_MIN_TRADES = 1000
PASS_MAX_AVG_GAP = 150.0

from scripts.run_fr2_diagnostics import DAYS, MODELS_DIR, get_ohlcv_and_proba
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
    df_base: pd.DataFrame, pl_base: np.ndarray, ps_base: np.ndarray,
    df_fr2: pd.DataFrame, pl_fr2: np.ndarray, ps_fr2: np.ndarray,
):
    df_b = df_base.copy()
    df_f = df_fr2.copy()
    for d in (df_b, df_f):
        d["timestamp"] = pd.to_datetime(d["timestamp"])
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


def _bar_index_for_time(df_bt: pd.DataFrame, time_str: str | None) -> int | None:
    if not time_str:
        return None
    ts = pd.to_datetime(df_bt["timestamp"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    if len(ts) == 0:
        return None
    t = pd.to_datetime(time_str, errors="coerce")
    if pd.isna(t):
        return None
    t64 = t.to_datetime64()
    i = int(np.searchsorted(ts, t64, side="left"))
    if i <= 0:
        return 0
    if i >= len(ts):
        return len(ts) - 1
    prev_dt = abs(ts[i - 1] - t64)
    next_dt = abs(ts[i] - t64)
    return i - 1 if prev_dt <= next_dt else i


def _avg_gap_between_trades(trades: List[Any], df_bt: pd.DataFrame) -> float:
    if not trades or len(trades) < 2:
        return np.nan
    def _entry_ts(t):
        return (t.get("entry_time") if isinstance(t, dict) else getattr(t, "entry_time", None)) or ""
    sorted_trades = sorted(trades, key=_entry_ts)
    gaps = []
    for i in range(len(sorted_trades) - 1):
        t_cur, t_next = sorted_trades[i], sorted_trades[i + 1]
        exit_time = t_cur.get("exit_time") if isinstance(t_cur, dict) else getattr(t_cur, "exit_time", None)
        entry_next = t_next.get("entry_time") if isinstance(t_next, dict) else getattr(t_next, "entry_time", None)
        exit_bar = _bar_index_for_time(df_bt, exit_time)
        entry_bar = _bar_index_for_time(df_bt, entry_next)
        if exit_bar is not None and entry_bar is not None:
            gaps.append(float(max(0, entry_bar - exit_bar)))
    return float(np.mean(gaps)) if gaps else np.nan


def _run_one(
    df_w: pd.DataFrame,
    pl: np.ndarray,
    ps: np.ndarray,
    threshold: float,
    commission: float,
    slippage: float,
) -> tuple[dict | None, dict | None]:
    res_on, _ = run_backtest_7d(
        SYMBOL, TIMEFRAME, df_w, pl, ps,
        commission_rate=commission, slippage_rate=slippage,
        min_max_proba=threshold, max_entropy=MAX_ENTROPY, decision_mode="argmax",
        min_hold=MIN_HOLD, cooldown=COOLDOWN,
        time_stop_enabled=True, time_stop_bars=TIME_STOP_BARS,
        early_exit_enabled=True, early_exit_bad_k=EARLY_EXIT_BAD_K,
    )
    res_off, _ = run_backtest_7d(
        SYMBOL, TIMEFRAME, df_w, pl, ps,
        commission_rate=0.0, slippage_rate=0.0,
        min_max_proba=threshold, max_entropy=MAX_ENTROPY, decision_mode="argmax",
        min_hold=MIN_HOLD, cooldown=COOLDOWN,
        time_stop_enabled=True, time_stop_bars=TIME_STOP_BARS,
        early_exit_enabled=True, early_exit_bad_k=EARLY_EXIT_BAD_K,
    )
    return (res_on, res_off)


def _row_from_run(
    res_on: dict | None, res_off: dict | None, df_w: pd.DataFrame,
    candidate_id: str, period_label: str, symbol: str, cost_scenario: str,
) -> Dict[str, Any]:
    if res_on is None:
        cost_on, trades_count, alpha_fee_ratio, avg_gap = np.nan, 0, np.nan, np.nan
    else:
        cost_on = float(res_on.get("total_return", np.nan))
        trades_count = int(res_on.get("total_trades", 0))
        trades_list = res_on.get("trades", [])
        cost_off = float(res_off.get("total_return", np.nan)) if res_off else np.nan
        if trades_count > 0 and res_off is not None:
            alpha_per_trade = cost_off / trades_count
            fee_per_trade = (cost_off - cost_on) / trades_count
            alpha_fee_ratio = alpha_per_trade / fee_per_trade if abs(fee_per_trade) > 1e-12 else np.nan
        else:
            alpha_fee_ratio = np.nan
        avg_gap = _avg_gap_between_trades(trades_list, df_w)
    cost_off_val = float(res_off.get("total_return", np.nan)) if (res_on and res_off) else np.nan
    return {
        "candidate_id": candidate_id,
        "period_label": period_label,
        "symbol": symbol,
        "cost_scenario": cost_scenario,
        "trades": trades_count,
        "cost_on": cost_on,
        "cost_off": cost_off_val,
        "alpha_fee_ratio": alpha_fee_ratio,
        "avg_gap_between_trades": avg_gap,
    }


def _period_masks(df_bt: pd.DataFrame) -> List[Tuple[np.ndarray, str]]:
    """Returns [(mask, label), ...] for full 720d, last 180d, last 90d, rolling 60d step 30d."""
    t = pd.to_datetime(df_bt["timestamp"])
    t_max = t.max()
    out = []
    # full 720d
    start_720 = t_max - pd.Timedelta(days=DAYS_FULL)
    out.append(( (t >= start_720).to_numpy(), "720d" ))
    # last 180d, 90d
    for d in (180, 90):
        start = t_max - pd.Timedelta(days=d)
        out.append(( (t >= start).to_numpy(), f"last_{d}d" ))
    # rolling 60d step 30d
    n = 0
    end = t_max
    while True:
        start = end - pd.Timedelta(days=ROLLING_WINDOW_DAYS)
        if start < t.min():
            break
        mask = (t >= start) & (t <= end)
        if mask.sum() < 100:
            break
        out.append(( mask.to_numpy(), f"roll60d_{n}" ))
        n += 1
        end = end - pd.Timedelta(days=ROLLING_STEP_DAYS)
    return out


def main() -> None:
    print("[FR2-GEN] Loading data (BTC 720d)...", flush=True)
    triple_base, err_b = get_ohlcv_and_proba(DAYS_FULL, BASELINE_PT, "base", False)
    triple_fr2, err_f = get_ohlcv_and_proba(DAYS_FULL, FR2_PT, "microstructure_v1", True)
    if err_b or err_f or triple_base is None or triple_fr2 is None:
        raise RuntimeError(f"get_ohlcv_and_proba failed: base={err_b}, fr2={err_f}")

    df_b, pl_b, ps_b, _ = triple_base
    df_f, pl_f, ps_f, _ = triple_fr2
    df_bt, pl_b, ps_b, pl_f, ps_f = _align_by_timestamp(df_b, pl_b, ps_b, df_f, pl_f, ps_f)
    df_bt_reg = add_regime_columns(DAYS_FULL, df_bt)
    base_c4 = (
        (df_bt_reg["trend_regime"] == "uptrend")
        & (df_bt_reg["vol_regime"] == "high_vol")
    ).to_numpy()
    c4_mask = build_fr2_c4_mask(base_c4, persistence_bars=6)
    ensemble_inputs = EnsembleInputs(
        pl_base=pl_b, ps_base=ps_b, pl_fr2=pl_f, ps_fr2=ps_f, c4_active=c4_mask
    )
    pl_w_full, ps_w_full = build_ensemble_proba(ensemble_inputs, mode="override")

    period_masks = _period_masks(df_bt)
    print(f"[FR2-GEN] Periods: {[p[1] for p in period_masks]}", flush=True)

    rows: List[Dict] = []
    cost_scenarios = [("base", COMMISSION, SLIPPAGE), ("conservative", COMMISSION_CONSERVATIVE, SLIPPAGE_CONSERVATIVE)]

    for candidate_id, threshold, use_c4 in CANDIDATES:
        if use_c4:
            pl_base = np.where(c4_mask, pl_w_full, 0.0).astype(np.float32)
            ps_base = np.where(c4_mask, ps_w_full, 0.0).astype(np.float32)
        else:
            pl_base = pl_w_full.copy()
            ps_base = ps_w_full.copy()

        for (mask, period_label) in period_masks:
            df_w = df_bt.loc[mask].reset_index(drop=True)
            pl_w = pl_base[mask]
            ps_w = ps_base[mask]
            if len(df_w) < 200:
                continue
            for cost_name, comm, slip in cost_scenarios:
                res_on, res_off = _run_one(df_w, pl_w, ps_w, threshold, comm, slip)
                row = _row_from_run(res_on, res_off, df_w, candidate_id, period_label, SYMBOL, cost_name)
                rows.append(row)
                co = row["cost_on"]
                co_str = f"{co:.4f}" if isinstance(co, (int, float)) and not np.isnan(co) else str(co)
                print(f"  {candidate_id} {period_label} {cost_name} -> trades={row['trades']} cost_on={co_str}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "fr2_generalization_results.csv", index=False)
    print(f"[FR2-GEN] Wrote fr2_generalization_results.csv ({len(rows)} rows)", flush=True)

    # Aggregate per candidate (base cost only, all periods)
    df_base = df[df["cost_scenario"] == "base"]
    agg = []
    for cid in df_base["candidate_id"].unique():
        sub = df_base[df_base["candidate_id"] == cid]
        cost_ons = sub["cost_on"].dropna()
        afrs = sub["alpha_fee_ratio"].dropna()
        agg.append({
            "candidate_id": cid,
            "trades_mean": sub["trades"].mean(),
            "trades_min": sub["trades"].min(),
            "cost_on_mean": cost_ons.mean() if len(cost_ons) else np.nan,
            "cost_on_std": cost_ons.std() if len(cost_ons) > 1 else 0.0,
            "cost_on_min": cost_ons.min() if len(cost_ons) else np.nan,
            "alpha_fee_ratio_mean": afrs.mean() if len(afrs) else np.nan,
            "alpha_fee_ratio_std": afrs.std() if len(afrs) > 1 else 0.0,
            "avg_gap_mean": sub["avg_gap_between_trades"].mean(),
            "avg_gap_max": sub["avg_gap_between_trades"].max(),
        })
    agg_df = pd.DataFrame(agg)

    # PASS
    def pass_fail(r) -> str:
        if r["trades_min"] < PASS_MIN_TRADES:
            return "FAIL (trades)"
        if r["cost_on_min"] is not None and not np.isnan(r["cost_on_min"]) and r["cost_on_min"] < PASS_MIN_COST_ON:
            return "FAIL (min cost_on)"
        if r["avg_gap_max"] is not None and not np.isnan(r["avg_gap_max"]) and r["avg_gap_max"] > PASS_MAX_AVG_GAP:
            return "FAIL (avg_gap)"
        return "PASS"
    agg_df["pass_fail"] = agg_df.apply(pass_fail, axis=1)

    # Report
    lines = [
        "# FR2 Generalization Report",
        "",
        "## 1. 후보별 집계 (base cost, 모든 구간)",
        "",
        agg_df.to_string(),
        "",
        "## 2. PASS/FAIL",
        "",
    ]
    for _, r in agg_df.iterrows():
        lines.append(f"- **{r['candidate_id']}**: {r['pass_fail']}")
    lines.extend([
        "",
        "## 3. 구간별 성능 (base cost, 일부)",
        "",
    ])
    pivot = df_base.pivot_table(
        index=["candidate_id", "period_label"],
        values=["trades", "cost_on", "alpha_fee_ratio", "avg_gap_between_trades"],
        aggfunc="first",
    )
    lines.append(pivot.to_string())
    lines.extend([
        "",
        "## 4. 추천",
        "",
    ])
    passed = agg_df[agg_df["pass_fail"] == "PASS"]["candidate_id"].tolist()
    if passed:
        lines.append("PASS 후보: " + ", ".join(passed) + ". 이 중 cost_on_mean·alpha_fee_ratio_mean이 높고 cost_on_std가 낮은 1~2개를 운영안으로 추천.")
    else:
        lines.append("PASS한 후보 없음. 기준 완화 또는 파라미터 재검토 필요.")
    lines.append("")

    (OUT_DIR / "FR2_GENERALIZATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("[FR2-GEN] Wrote FR2_GENERALIZATION_REPORT.md", flush=True)


if __name__ == "__main__":
    main()
