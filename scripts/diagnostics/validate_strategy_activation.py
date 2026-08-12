"""
Strategy Activation Filter (전략 ON/OFF 제어) 검증.

전략을 항상 실행하지 않고, 좋은 구간에서만 ON / 나쁜 구간에서는 OFF.
Base strategy: rule_C_trend switching (S1/S2 동적 전환).
Activation filter가 OFF이면 어떤 전략도 entry하지 않음.

실행:
  python -m scripts.diagnostics.validate_strategy_activation --lookback-list 14,30,60,90,180
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "diagnostics" / "final"


def _get(result: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _mask_last_days(df, lookback_days: int):
    ts = pd.to_datetime(df["timestamp"])
    end = ts.max()
    start = end - pd.Timedelta(days=int(lookback_days))
    return (ts >= start).values


def _profit_factor(trades: List[Mapping[str, Any]]) -> Optional[float]:
    profits = [float(t["profit"]) for t in trades if t.get("profit") is not None]
    if not profits:
        return None
    wins = sum(p for p in profits if p > 0)
    losses = sum(p for p in profits if p < 0)
    if losses == 0:
        return None if wins == 0 else float("inf")
    return wins / abs(losses)


def _expectancy(trades: List[Mapping[str, Any]]) -> Optional[float]:
    profits = [float(t["profit"]) for t in trades if t.get("profit") is not None]
    if not profits:
        return None
    return float(np.mean(profits))


def _sharpe(trades: List[Mapping[str, Any]]) -> Optional[float]:
    profits = [float(t["profit"]) for t in trades if t.get("profit") is not None]
    if len(profits) < 2:
        return None
    arr = np.array(profits)
    std = float(np.std(arr, ddof=1))
    if std < 1e-15:
        return None
    return float(np.mean(arr) / std)


def _compute_metrics(trades: List[Mapping[str, Any]], period: int) -> Dict[str, Any]:
    profits = [float(t["profit"]) for t in trades if t.get("profit") is not None]
    if not profits:
        return {
            "total_return": 0.0, "win_rate": 0.0, "total_trades": 0,
            "avg_profit": 0.0, "median_profit": 0.0, "max_drawdown": 0.0,
            "profit_factor": None, "expectancy": None, "sharpe": None,
            "trades_per_day": 0.0,
        }
    cum = np.cumsum(profits)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    mdd = float(np.max(dd)) if len(dd) > 0 else 0.0
    wins = sum(1 for p in profits if p > 0)
    return {
        "total_return": float(cum[-1]),
        "win_rate": wins / len(profits) if profits else 0.0,
        "total_trades": len(profits),
        "avg_profit": float(np.mean(profits)),
        "median_profit": float(np.median(profits)),
        "max_drawdown": mdd,
        "profit_factor": _profit_factor(trades),
        "expectancy": _expectancy(trades),
        "sharpe": _sharpe(trades),
        "trades_per_day": len(profits) / period if period > 0 else 0.0,
    }


def _compute_activation_masks(
    df_period: pd.DataFrame,
    proba_long_period: np.ndarray,
    proba_short_period: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute per-bar activation filters.
    Returns dict of filter_name -> bool array (True = strategy ON).
    """
    from src.strategy_filters.regime_ablation import compute_regime_components

    n = len(df_period)

    # Regime components
    ema_above, is_sideways, is_high_vol, is_mid_vol, is_low_vol, trend_label, _ = compute_regime_components(df_period)
    vol_mid_high = is_mid_vol | is_high_vol
    tl = pd.Series(trend_label)
    trend_sw = (tl == "sideways").to_numpy(dtype=bool)
    not_sideways = ~trend_sw

    # Per-bar entropy (natural log, same as confidence_filter_mode)
    entropy_arr = np.zeros(n, dtype=float)
    for i in range(n):
        if i < len(proba_long_period) and i < len(proba_short_period):
            pl = float(proba_long_period[i])
            ps = float(proba_short_period[i])
            pf = max(0.0, min(1.0, 1.0 - pl - ps))
            h = 0.0
            for p in (pl, ps, pf):
                if p > 1e-12:
                    h -= p * math.log(p)
            entropy_arr[i] = h
    entropy_ok = entropy_arr <= 1.0

    # F1: no_sideways
    f1 = not_sideways.copy()

    # F2: vol mid/high
    f2 = vol_mid_high.copy()

    # F3: entropy <= 1.0
    f3 = entropy_ok.copy()

    # F4: trend + vol
    f4 = not_sideways & vol_mid_high

    # F5: trend + entropy
    f5 = not_sideways & entropy_ok

    # F6: vol + entropy
    f6 = vol_mid_high & entropy_ok

    # F7: strict (all three)
    f7 = not_sideways & vol_mid_high & entropy_ok

    return {
        "F1_no_sideways": f1,
        "F2_vol_mid_high": f2,
        "F3_entropy_100": f3,
        "F4_trend_vol": f4,
        "F5_trend_entropy": f5,
        "F6_vol_entropy": f6,
        "F7_strict_all": f7,
    }


def _apply_switching_rule_c(
    s1_trades: List[Mapping[str, Any]],
    s2_trades: List[Mapping[str, Any]],
    timestamps: pd.Series,
    not_sideways: np.ndarray,
) -> List[Mapping[str, Any]]:
    """Apply rule_C_trend switching: use S2 when not sideways, S1 when sideways."""
    ts_index = pd.to_datetime(timestamps)

    def find_bar_idx(entry_time_str):
        try:
            et = pd.Timestamp(entry_time_str)
            idx = ts_index.searchsorted(et, side="right") - 1
            return max(0, min(int(idx), len(ts_index) - 1))
        except Exception:
            return 0

    s1_kept = []
    for t in s1_trades:
        if t.get("entry_time") and t.get("profit") is not None:
            idx = find_bar_idx(t["entry_time"])
            if not not_sideways[idx]:
                s1_kept.append(t)

    s2_kept = []
    for t in s2_trades:
        if t.get("entry_time") and t.get("profit") is not None:
            idx = find_bar_idx(t["entry_time"])
            if not_sideways[idx]:
                s2_kept.append(t)

    all_trades = s1_kept + s2_kept
    all_trades.sort(key=lambda t: t.get("entry_time", ""))

    final_trades = []
    last_exit = None
    for t in all_trades:
        entry = t.get("entry_time", "")
        if last_exit is not None and entry < last_exit:
            continue
        final_trades.append(t)
        last_exit = t.get("exit_time") or entry

    return final_trades


def _apply_activation_filter(
    switched_trades: List[Mapping[str, Any]],
    timestamps: pd.Series,
    activation_mask: np.ndarray,
) -> List[Mapping[str, Any]]:
    """Keep only trades whose entry bar has activation ON."""
    ts_index = pd.to_datetime(timestamps)

    def find_bar_idx(entry_time_str):
        try:
            et = pd.Timestamp(entry_time_str)
            idx = ts_index.searchsorted(et, side="right") - 1
            return max(0, min(int(idx), len(ts_index) - 1))
        except Exception:
            return 0

    filtered = []
    for t in switched_trades:
        if t.get("entry_time") and t.get("profit") is not None:
            idx = find_bar_idx(t["entry_time"])
            if activation_mask[idx]:
                filtered.append(t)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-list", type=str, default="14,30,60,90,180")
    parser.add_argument("--cooldown-bars", type=int, default=12)
    parser.add_argument("--max-holding-bars", type=int, default=12)
    args = parser.parse_args()

    lookback_list = [int(x.strip()) for x in args.lookback_list.split(",") if x.strip()]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine
    from src.strategy_filters.regime_ablation import compute_regime_components

    engine = get_ml_backtest_engine("ml_lstm_attn", "BTCUSDT", "5m")
    proba_long, proba_short, df_full = engine.load_predictions()

    common_kw: Dict[str, Any] = dict(
        long_threshold=None,
        short_threshold=None,
        use_optimized_threshold=True,
        long_only=True,
        signal_confirmation_bars=1,
        min_hold_bars=12,
        cooldown_bars=args.cooldown_bars,
        use_strategy_guard=True,
        use_strategy_guard_v2=True,
        strategy_guard_v2_mode="soft",
        strategy_guard_v2_scale_floor=0.02,
        use_stage2=True,
        proba_long_cache=proba_long,
        proba_short_cache=proba_short,
        df_with_proba=df_full,
        emit_trade_log=False,
        max_holding_bars=args.max_holding_bars,
        enable_profit_lock_exit=False,
        enable_regime_filter=False,
        regime_filter_mode="none",
        enable_positive_regime_filter=False,
        enable_adaptive_positive_regime_filter=False,
        use_legacy_entry_gates_with_directional=True,
    )

    S1_EXTRA = dict(enable_confidence_filter=False, confidence_filter_mode="none")
    S2_EXTRA = dict(
        enable_confidence_filter=True,
        confidence_filter_mode="entropy_100",
        enable_positive_regime_filter=True,
        positive_regime_filter_mode="allow_ema_below_or_strong_up",
    )

    FILTERS = [
        "F1_no_sideways", "F2_vol_mid_high", "F3_entropy_100",
        "F4_trend_vol", "F5_trend_entropy", "F6_vol_entropy", "F7_strict_all",
    ]

    all_rows: List[Dict[str, Any]] = []

    for period in lookback_list:
        mask = _mask_last_days(df_full, period)
        if int(mask.sum()) <= 0:
            print(f"[WARN] period={period}d mask empty, skip")
            continue

        period_kw = {**common_kw, "index_mask": mask}
        df_period = df_full[mask].reset_index(drop=True)
        pl_period = proba_long[mask]
        ps_period = proba_short[mask]

        # Run S1 and S2
        res_s1 = engine.run_backtest(**{**period_kw, **S1_EXTRA})
        s1_trades = list(_get(res_s1, "trades") or [])

        res_s2 = engine.run_backtest(**{**period_kw, **S2_EXTRA})
        s2_trades = list(_get(res_s2, "trades") or [])

        # Compute regime for switching (rule_C: not_sideways → S2)
        _, is_sideways, _, _, _, trend_label, _ = compute_regime_components(df_period)
        tl = pd.Series(trend_label)
        not_sideways = (~(tl == "sideways")).to_numpy(dtype=bool)

        # Apply switching (rule_C_trend)
        switched_trades = _apply_switching_rule_c(s1_trades, s2_trades, df_period["timestamp"], not_sideways)
        sw_metrics = _compute_metrics(switched_trades, period)
        sw_metrics["case_name"] = "switching_C"
        sw_metrics["period"] = period
        sw_metrics["active_ratio"] = 1.0
        all_rows.append(sw_metrics)

        # Baseline (S1)
        bl_metrics = _compute_metrics(s1_trades, period)
        bl_metrics["case_name"] = "S1_baseline"
        bl_metrics["period"] = period
        bl_metrics["active_ratio"] = 1.0
        all_rows.append(bl_metrics)

        # Compute activation masks
        activation_masks = _compute_activation_masks(df_period, pl_period, ps_period)

        # Apply each activation filter on top of switching
        for filt_name in FILTERS:
            act_mask = activation_masks[filt_name]
            filtered_trades = _apply_activation_filter(switched_trades, df_period["timestamp"], act_mask)
            fm = _compute_metrics(filtered_trades, period)
            fm["case_name"] = filt_name
            fm["period"] = period
            fm["active_ratio"] = float(act_mask.sum()) / len(act_mask) if len(act_mask) > 0 else 0.0
            all_rows.append(fm)

    # ── Diffs ──
    sw_map: Dict[int, Dict[str, float]] = {}
    bl_map: Dict[int, Dict[str, float]] = {}
    for r in all_rows:
        if r["case_name"] == "switching_C":
            sw_map[r["period"]] = {"total_return": r["total_return"], "avg_profit": r["avg_profit"], "max_drawdown": r["max_drawdown"], "total_trades": r["total_trades"]}
        elif r["case_name"] == "S1_baseline":
            bl_map[r["period"]] = {"total_return": r["total_return"], "avg_profit": r["avg_profit"], "max_drawdown": r["max_drawdown"], "total_trades": r["total_trades"]}

    for r in all_rows:
        p = r["period"]
        sw = sw_map.get(p, {})
        bl = bl_map.get(p, {})
        r["imp_vs_switching"] = r["total_return"] - sw.get("total_return", 0.0)
        r["imp_vs_baseline"] = r["total_return"] - bl.get("total_return", 0.0)
        r["avg_profit_diff_sw"] = r["avg_profit"] - sw.get("avg_profit", 0.0)
        r["MDD_diff_sw"] = r["max_drawdown"] - sw.get("max_drawdown", 0.0)
        sw_trades = sw.get("total_trades", 0)
        r["trade_reduction"] = (1.0 - r["total_trades"] / sw_trades) if sw_trades > 0 else 0.0

    # ── Consistency ──
    consistency: Dict[str, Dict[str, Any]] = {}
    for filt in FILTERS:
        rows_f = [r for r in all_rows if r["case_name"] == filt]
        if not rows_f:
            continue
        imp_sw = [r["imp_vs_switching"] for r in rows_f]
        imp_bl = [r["imp_vs_baseline"] for r in rows_f]
        avg_diffs = [r["avg_profit_diff_sw"] for r in rows_f]
        mdd_diffs = [r["MDD_diff_sw"] for r in rows_f]
        n_periods = len(rows_f)

        imp_sw_count = sum(1 for x in imp_sw if x > 1e-9)
        imp_bl_count = sum(1 for x in imp_bl if x > 1e-9)
        avg_imp_sw = float(np.mean(imp_sw))
        avg_imp_bl = float(np.mean(imp_bl))
        worst_imp_sw = float(min(imp_sw))
        avg_avg_diff = float(np.mean(avg_diffs))
        avg_mdd_diff = float(np.mean(mdd_diffs))
        avg_trades = float(np.mean([r["total_trades"] for r in rows_f]))
        avg_active = float(np.mean([r["active_ratio"] for r in rows_f]))
        avg_trade_red = float(np.mean([r["trade_reduction"] for r in rows_f]))

        if imp_sw_count >= 4 and avg_imp_sw > 0 and avg_avg_diff > 0 and 0.2 <= avg_active <= 0.8:
            verdict = "A"
        elif imp_sw_count >= 4 and avg_imp_sw > 0:
            verdict = "A"
        elif imp_bl_count >= 4 and avg_imp_bl > 0:
            verdict = "A"
        elif imp_sw_count >= 3 and avg_imp_sw > 0:
            verdict = "B"
        elif imp_bl_count >= 3 and avg_imp_bl > 0:
            verdict = "B"
        else:
            verdict = "C"

        consistency[filt] = {
            "imp_sw": f"{imp_sw_count}/{n_periods}",
            "imp_bl": f"{imp_bl_count}/{n_periods}",
            "avg_imp_sw": round(avg_imp_sw, 8),
            "avg_imp_bl": round(avg_imp_bl, 8),
            "worst_imp_sw": round(worst_imp_sw, 8),
            "avg_avg_diff": round(avg_avg_diff, 8),
            "avg_mdd_diff": round(avg_mdd_diff, 8),
            "avg_trades": round(avg_trades, 1),
            "avg_active": round(avg_active, 4),
            "avg_trade_red": round(avg_trade_red, 4),
            "verdict": verdict,
        }

    best_filter = max(consistency.keys(), key=lambda f: consistency[f]["avg_imp_sw"])
    best_verdict = consistency[best_filter]["verdict"]

    # ── CSV ──
    csv_path = OUT_DIR / f"strategy_activation_{stamp}.csv"
    if all_rows:
        keys = sorted(set().union(*[r.keys() for r in all_rows]))
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in all_rows:
                w.writerow(r)

    # ── Markdown ──
    md_path = OUT_DIR / f"strategy_activation_{stamp}.md"
    md: List[str] = []

    md.append("# Strategy Activation Filter 검증")
    md.append("")
    md.append("## 1. 목적")
    md.append("전략을 항상 실행하지 않고, 나쁜 구간에서 OFF하면 성과 개선되는지 검증.")
    md.append("")
    md.append("## 2. 설정")
    md.append(f"- lookback: {lookback_list}")
    md.append("- Base: rule_C_trend switching (S1/S2 동적 전환)")
    md.append("- Activation OFF → 어떤 전략도 entry하지 않음")
    md.append("")

    md.append("## 3. Switching (Base) 결과")
    md.append("")
    md.append("| period | return | win_rate | trades | avg_profit | MDD |")
    md.append("|--------|--------|----------|--------|------------|-----|")
    for r in all_rows:
        if r["case_name"] == "switching_C":
            md.append(f"| {r['period']}d | {r['total_return']:.6f} | {r['win_rate']:.4f} | {r['total_trades']} | {r['avg_profit']:.8f} | {r['max_drawdown']:.6f} |")
    md.append("")

    md.append("## 4. Activation Filters 결과")
    md.append("")
    md.append("| filter | period | return | trades | avg_profit | MDD | imp_sw | imp_bl | active% | trade_red |")
    md.append("|--------|--------|--------|--------|------------|-----|--------|--------|---------|-----------|")
    for r in all_rows:
        if r["case_name"] in FILTERS:
            md.append(
                f"| {r['case_name']} | {r['period']}d | {r['total_return']:.6f} | "
                f"{r['total_trades']} | {r['avg_profit']:.8f} | {r['max_drawdown']:.6f} | "
                f"{r['imp_vs_switching']:.6f} | {r['imp_vs_baseline']:.6f} | "
                f"{r['active_ratio']:.2%} | {r['trade_reduction']:.2%} |"
            )
    md.append("")

    md.append("## 5. Consistency Summary")
    md.append("")
    md.append("| filter | imp_sw | imp_bl | avg_imp_sw | avg_imp_bl | avg_avgP | avg_MDD | active% | trade_red | verdict |")
    md.append("|--------|--------|--------|------------|------------|----------|---------|---------|-----------|---------|")
    for filt in sorted(consistency.keys(), key=lambda f: -consistency[f]["avg_imp_sw"]):
        cs = consistency[filt]
        md.append(
            f"| {filt} | {cs['imp_sw']} | {cs['imp_bl']} | "
            f"{cs['avg_imp_sw']} | {cs['avg_imp_bl']} | {cs['avg_avg_diff']} | "
            f"{cs['avg_mdd_diff']} | {cs['avg_active']:.2%} | {cs['avg_trade_red']:.2%} | **{cs['verdict']}** |"
        )
    md.append("")

    md.append("## 6. 최종 판정")
    md.append("")
    md.append(f"**Best filter: `{best_filter}`**")
    bc = consistency[best_filter]
    md.append(f"- Verdict: **{best_verdict}**")
    md.append(f"- Improved vs switching: {bc['imp_sw']}")
    md.append(f"- Improved vs baseline: {bc['imp_bl']}")
    md.append(f"- Avg improvement vs switching: {bc['avg_imp_sw']}")
    md.append(f"- Avg improvement vs baseline: {bc['avg_imp_bl']}")
    md.append(f"- Active ratio: {bc['avg_active']:.2%}")
    md.append(f"- Trade reduction: {bc['avg_trade_red']:.2%}")
    md.append("")

    if best_verdict == "A":
        md.append("### 결론: 실전 적용 가능")
        md.append(f"- `{best_filter}` activation filter 적용 권장")
    elif best_verdict == "B":
        md.append("### 결론: 부분 성공 (Shadow 테스트 필요)")
    else:
        md.append("### 결론: Activation filter 효과 없음")
    md.append("")

    md_path.write_text("\n".join(md), encoding="utf-8")

    # ── Console ──
    print("=" * 60)
    print("[ACTIVATION FILTER SUMMARY]")
    print("=" * 60)
    print()

    print("switching (base):")
    for r in all_rows:
        if r["case_name"] == "switching_C":
            print(f"  {r['period']}d: return={r['total_return']:.6f}, avg_profit={r['avg_profit']:.8f}, trades={r['total_trades']}, MDD={r['max_drawdown']:.6f}")
    print()

    print("baseline (S1):")
    for r in all_rows:
        if r["case_name"] == "S1_baseline":
            print(f"  {r['period']}d: return={r['total_return']:.6f}, trades={r['total_trades']}")
    print()

    print("activation filters:")
    for filt in sorted(consistency.keys(), key=lambda f: -consistency[f]["avg_imp_sw"]):
        cs = consistency[filt]
        print(
            f"  {filt}: imp_sw={cs['imp_sw']}, avg_imp={cs['avg_imp_sw']:.6f}, "
            f"imp_bl={cs['imp_bl']}, active={cs['avg_active']:.2%}, "
            f"trade_red={cs['avg_trade_red']:.2%}, verdict={cs['verdict']}"
        )
    print()

    print(f"best filter: {best_filter}")
    print(f"  verdict: {best_verdict}")
    print(f"  avg_imp_vs_switching: {bc['avg_imp_sw']}")
    print(f"  avg_imp_vs_baseline: {bc['avg_imp_bl']}")
    print(f"  active_ratio: {bc['avg_active']:.2%}")
    print(f"  trade_reduction: {bc['avg_trade_red']:.2%}")
    print()

    for r in all_rows:
        if r["case_name"] == best_filter:
            print(
                f"  {r['period']}d: return={r['total_return']:.6f}, trades={r['total_trades']}, "
                f"imp_sw={r['imp_vs_switching']:.6f}, imp_bl={r['imp_vs_baseline']:.6f}, "
                f"active={r['active_ratio']:.2%}"
            )
    print()

    print(f"final verdict: {best_verdict}")
    if best_verdict == "A":
        print(f"  → 실전 적용 가능 ({best_filter})")
    elif best_verdict == "B":
        print(f"  → 부분 성공, shadow 테스트 필요 ({best_filter})")
    else:
        print(f"  → activation filter 효과 미미")
    print()

    print(f"created:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")


if __name__ == "__main__":
    main()
