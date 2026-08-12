"""
Strategy Switching (전략 전환 로직) 검증.

단일 전략으로 전 구간 커버 불가 → 상황에 따라 S1(baseline) vs S2(entropy+P7) 전환.
각 bar에서 switching rule에 따라 어떤 전략을 쓸지 결정 후, 해당 전략의 trade만 유지.

실행:
  python -m scripts.diagnostics.validate_strategy_switching --lookback-list 14,30,60,90,180
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
    """Compute standard metrics from a trade list."""
    n_tr = len(trades)
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


def _compute_switching_signals(df_period: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Compute per-bar switching rule signals.
    Returns dict of rule_name -> bool array (True = use S2, False = use S1).
    """
    from src.strategy_filters.regime_ablation import compute_regime_components

    n = len(df_period)
    close = df_period["close"].astype(float).values

    # recent_14d return (rolling 14d = 14*24*12 = 4032 bars for 5m, use 4032 bars)
    bars_14d = 14 * 24 * 12
    rolling_ret = pd.Series(close).pct_change(bars_14d).values
    recent_neg = np.zeros(n, dtype=bool)
    for i in range(n):
        if np.isfinite(rolling_ret[i]):
            recent_neg[i] = rolling_ret[i] < 0
        else:
            recent_neg[i] = False

    # regime components
    ema_above, is_sideways, is_high_vol, is_mid_vol, is_low_vol, trend_label, _ = compute_regime_components(df_period)
    vol_mid_high = is_mid_vol | is_high_vol
    tl = pd.Series(trend_label)
    trend_up = (tl == "up").to_numpy(dtype=bool)
    trend_sw = (tl == "sideways").to_numpy(dtype=bool)
    not_sideways = ~trend_sw

    # Rule A: recent return < 0 → use S2
    rule_a = recent_neg.copy()

    # Rule B: vol mid/high → use S2
    rule_b = vol_mid_high.copy()

    # Rule C: not sideways → use S2
    rule_c = not_sideways.copy()

    # Rule D: recent_ret < 0 AND vol mid/high → use S2
    rule_d = recent_neg & vol_mid_high

    # Rule E: recent_ret < 0 OR trend_up → use S2
    rule_e = recent_neg | trend_up

    return {
        "rule_A_recent_return": rule_a,
        "rule_B_vol": rule_b,
        "rule_C_trend": rule_c,
        "rule_D_hybrid": rule_d,
        "rule_E_aggressive": rule_e,
    }


def _apply_switching(
    s1_trades: List[Mapping[str, Any]],
    s2_trades: List[Mapping[str, Any]],
    timestamps: pd.Series,
    use_s2_mask: np.ndarray,
) -> tuple[List[Mapping[str, Any]], Dict[str, Any]]:
    """
    Apply switching: keep S1 trades where rule says S1, S2 trades where rule says S2.
    Handle overlaps by entry_time ordering.
    """
    ts_index = pd.to_datetime(timestamps)
    ts_list = ts_index.tolist()

    def find_bar_idx(entry_time_str):
        try:
            et = pd.Timestamp(entry_time_str)
            idx = ts_index.searchsorted(et, side="right") - 1
            if idx < 0:
                idx = 0
            if idx >= len(ts_index):
                idx = len(ts_index) - 1
            return int(idx)
        except Exception:
            return 0

    # Filter S1 trades: keep where use_s2 is False at entry
    s1_kept = []
    for t in s1_trades:
        if t.get("entry_time") and t.get("profit") is not None:
            idx = find_bar_idx(t["entry_time"])
            if not use_s2_mask[idx]:
                s1_kept.append(t)

    # Filter S2 trades: keep where use_s2 is True at entry
    s2_kept = []
    for t in s2_trades:
        if t.get("entry_time") and t.get("profit") is not None:
            idx = find_bar_idx(t["entry_time"])
            if use_s2_mask[idx]:
                s2_kept.append(t)

    # Merge and sort by entry_time, resolve overlaps
    all_trades = s1_kept + s2_kept
    all_trades.sort(key=lambda t: t.get("entry_time", ""))

    # Remove overlapping trades (can't be in two positions at once)
    final_trades = []
    last_exit = None
    for t in all_trades:
        entry = t.get("entry_time", "")
        if last_exit is not None and entry < last_exit:
            continue
        final_trades.append(t)
        last_exit = t.get("exit_time") or entry

    # Usage stats
    n = len(use_s2_mask)
    s2_bars = int(use_s2_mask.sum())
    s1_bars = n - s2_bars
    switch_count = int(np.sum(np.diff(use_s2_mask.astype(int)) != 0))

    stats = {
        "s1_bars_pct": s1_bars / n if n > 0 else 0.0,
        "s2_bars_pct": s2_bars / n if n > 0 else 0.0,
        "switch_count": switch_count,
        "s1_trades_kept": len(s1_kept),
        "s2_trades_kept": len(s2_kept),
        "final_trades": len(final_trades),
        "overlap_removed": len(all_trades) - len(final_trades),
    }
    return final_trades, stats


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

    RULES = ["rule_A_recent_return", "rule_B_vol", "rule_C_trend", "rule_D_hybrid", "rule_E_aggressive"]

    all_rows: List[Dict[str, Any]] = []

    for period in lookback_list:
        mask = _mask_last_days(df_full, period)
        if int(mask.sum()) <= 0:
            print(f"[WARN] period={period}d mask empty, skip")
            continue

        period_kw = {**common_kw, "index_mask": mask}
        df_period = df_full[mask].reset_index(drop=True)

        # Run S1 (baseline)
        res_s1 = engine.run_backtest(**{**period_kw, **S1_EXTRA})
        s1_trades = list(_get(res_s1, "trades") or [])
        s1_metrics = _compute_metrics(s1_trades, period)
        s1_metrics["case_name"] = "S1_baseline"
        s1_metrics["period"] = period
        all_rows.append(s1_metrics)

        # Run S2 (entropy + P7)
        res_s2 = engine.run_backtest(**{**period_kw, **S2_EXTRA})
        s2_trades = list(_get(res_s2, "trades") or [])
        s2_metrics = _compute_metrics(s2_trades, period)
        s2_metrics["case_name"] = "S2_entropy_P7"
        s2_metrics["period"] = period
        all_rows.append(s2_metrics)

        # Compute switching signals
        switching_signals = _compute_switching_signals(df_period)

        # Apply each switching rule
        for rule_name in RULES:
            use_s2 = switching_signals[rule_name]
            switched_trades, stats = _apply_switching(
                s1_trades, s2_trades, df_period["timestamp"], use_s2,
            )
            sw_metrics = _compute_metrics(switched_trades, period)
            sw_metrics["case_name"] = rule_name
            sw_metrics["period"] = period
            sw_metrics["s1_bars_pct"] = stats["s1_bars_pct"]
            sw_metrics["s2_bars_pct"] = stats["s2_bars_pct"]
            sw_metrics["switch_count"] = stats["switch_count"]
            sw_metrics["s1_trades_kept"] = stats["s1_trades_kept"]
            sw_metrics["s2_trades_kept"] = stats["s2_trades_kept"]
            sw_metrics["overlap_removed"] = stats["overlap_removed"]
            all_rows.append(sw_metrics)

    # ── Compute diffs ──
    bl_map: Dict[int, Dict[str, float]] = {}
    ent_map: Dict[int, Dict[str, float]] = {}
    for r in all_rows:
        if r["case_name"] == "S1_baseline":
            bl_map[r["period"]] = {"total_return": r["total_return"], "avg_profit": r["avg_profit"], "max_drawdown": r["max_drawdown"], "total_trades": r["total_trades"]}
        elif r["case_name"] == "S2_entropy_P7":
            ent_map[r["period"]] = {"total_return": r["total_return"], "avg_profit": r["avg_profit"], "max_drawdown": r["max_drawdown"], "total_trades": r["total_trades"]}

    for r in all_rows:
        p = r["period"]
        bl = bl_map.get(p, {})
        ent = ent_map.get(p, {})
        r["imp_vs_baseline"] = r["total_return"] - bl.get("total_return", 0.0)
        r["imp_vs_entropy"] = r["total_return"] - ent.get("total_return", 0.0)
        r["avg_profit_diff_bl"] = r["avg_profit"] - bl.get("avg_profit", 0.0)
        r["MDD_diff_bl"] = r["max_drawdown"] - bl.get("max_drawdown", 0.0)

    # ── Consistency ──
    consistency: Dict[str, Dict[str, Any]] = {}
    for rule in RULES:
        rows_r = [r for r in all_rows if r["case_name"] == rule]
        if not rows_r:
            continue
        imp_bl = [r["imp_vs_baseline"] for r in rows_r]
        imp_ent = [r["imp_vs_entropy"] for r in rows_r]
        avg_diffs = [r["avg_profit_diff_bl"] for r in rows_r]
        mdd_diffs = [r["MDD_diff_bl"] for r in rows_r]
        n_periods = len(rows_r)
        imp_bl_count = sum(1 for x in imp_bl if x > 1e-9)
        imp_ent_count = sum(1 for x in imp_ent if x > 1e-9)
        better_than_both = sum(1 for i in range(n_periods) if imp_bl[i] > 1e-9 and imp_ent[i] > 1e-9)

        avg_imp_bl = float(np.mean(imp_bl))
        avg_imp_ent = float(np.mean(imp_ent))
        worst_imp_bl = float(min(imp_bl))
        avg_avg_diff = float(np.mean(avg_diffs))
        avg_mdd_diff = float(np.mean(mdd_diffs))
        avg_trades = float(np.mean([r["total_trades"] for r in rows_r]))
        avg_s2_pct = float(np.mean([r.get("s2_bars_pct", 0) for r in rows_r]))

        if better_than_both >= 4 and avg_imp_bl > 0 and avg_imp_ent > 0:
            verdict = "A"
        elif imp_bl_count >= 4 and avg_imp_bl > 0:
            verdict = "A"
        elif imp_bl_count >= 3 and avg_imp_bl > 0:
            verdict = "B"
        elif imp_bl_count >= 2:
            verdict = "B"
        else:
            verdict = "C"

        consistency[rule] = {
            "imp_bl": f"{imp_bl_count}/{n_periods}",
            "imp_ent": f"{imp_ent_count}/{n_periods}",
            "better_both": f"{better_than_both}/{n_periods}",
            "avg_imp_bl": round(avg_imp_bl, 8),
            "avg_imp_ent": round(avg_imp_ent, 8),
            "worst_imp_bl": round(worst_imp_bl, 8),
            "avg_avg_diff": round(avg_avg_diff, 8),
            "avg_mdd_diff": round(avg_mdd_diff, 8),
            "avg_trades": round(avg_trades, 1),
            "avg_s2_pct": round(avg_s2_pct, 4),
            "verdict": verdict,
        }

    best_rule = max(consistency.keys(), key=lambda r: consistency[r]["avg_imp_bl"])
    best_verdict = consistency[best_rule]["verdict"]

    # ── CSV ──
    csv_path = OUT_DIR / f"strategy_switching_{stamp}.csv"
    if all_rows:
        keys = sorted(set().union(*[r.keys() for r in all_rows]))
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in all_rows:
                w.writerow(r)

    # ── Markdown ──
    md_path = OUT_DIR / f"strategy_switching_{stamp}.md"
    md: List[str] = []

    md.append("# Strategy Switching 검증")
    md.append("")
    md.append("## 1. 목적")
    md.append("S1(baseline) vs S2(entropy+P7)를 상황에 따라 전환하면 전 구간 성과 개선 가능한지 검증.")
    md.append("")
    md.append("## 2. 설정")
    md.append(f"- lookback: {lookback_list}")
    md.append(f"- cooldown={args.cooldown_bars}, max_hold={args.max_holding_bars}")
    md.append("- S1: baseline (no filters)")
    md.append("- S2: entropy_100 + P7 (allow_ema_below_or_strong_up)")
    md.append("")

    md.append("## 3. S1 (Baseline) 결과")
    md.append("")
    md.append("| period | return | win_rate | trades | avg_profit | MDD |")
    md.append("|--------|--------|----------|--------|------------|-----|")
    for r in all_rows:
        if r["case_name"] == "S1_baseline":
            md.append(f"| {r['period']}d | {r['total_return']:.6f} | {r['win_rate']:.4f} | {r['total_trades']} | {r['avg_profit']:.8f} | {r['max_drawdown']:.6f} |")
    md.append("")

    md.append("## 4. S2 (Entropy+P7) 결과")
    md.append("")
    md.append("| period | return | win_rate | trades | avg_profit | MDD |")
    md.append("|--------|--------|----------|--------|------------|-----|")
    for r in all_rows:
        if r["case_name"] == "S2_entropy_P7":
            md.append(f"| {r['period']}d | {r['total_return']:.6f} | {r['win_rate']:.4f} | {r['total_trades']} | {r['avg_profit']:.8f} | {r['max_drawdown']:.6f} |")
    md.append("")

    md.append("## 5. Switching Rules 결과")
    md.append("")
    md.append("| rule | period | return | trades | avg_profit | MDD | imp_bl | imp_ent | S2% | switches |")
    md.append("|------|--------|--------|--------|------------|-----|--------|---------|-----|----------|")
    for r in all_rows:
        if r["case_name"] in RULES:
            md.append(
                f"| {r['case_name']} | {r['period']}d | {r['total_return']:.6f} | "
                f"{r['total_trades']} | {r['avg_profit']:.8f} | {r['max_drawdown']:.6f} | "
                f"{r['imp_vs_baseline']:.6f} | {r['imp_vs_entropy']:.6f} | "
                f"{r.get('s2_bars_pct', 0):.2%} | {r.get('switch_count', 0)} |"
            )
    md.append("")

    md.append("## 6. Consistency Summary")
    md.append("")
    md.append("| rule | imp_bl | imp_ent | both | avg_imp_bl | avg_imp_ent | avg_avgP | avg_MDD | S2% | verdict |")
    md.append("|------|--------|---------|------|------------|-------------|----------|---------|-----|---------|")
    for rule in sorted(consistency.keys(), key=lambda r: -consistency[r]["avg_imp_bl"]):
        cs = consistency[rule]
        md.append(
            f"| {rule} | {cs['imp_bl']} | {cs['imp_ent']} | {cs['better_both']} | "
            f"{cs['avg_imp_bl']} | {cs['avg_imp_ent']} | {cs['avg_avg_diff']} | "
            f"{cs['avg_mdd_diff']} | {cs['avg_s2_pct']:.2%} | **{cs['verdict']}** |"
        )
    md.append("")

    md.append("## 7. 최종 판정")
    md.append("")
    md.append(f"**Best rule: `{best_rule}`**")
    md.append(f"- Verdict: **{best_verdict}**")
    bc = consistency[best_rule]
    md.append(f"- Improved vs baseline: {bc['imp_bl']}")
    md.append(f"- Improved vs entropy: {bc['imp_ent']}")
    md.append(f"- Better than both: {bc['better_both']}")
    md.append(f"- Avg improvement vs baseline: {bc['avg_imp_bl']}")
    md.append(f"- Avg improvement vs entropy: {bc['avg_imp_ent']}")
    md.append(f"- Avg S2 usage: {bc['avg_s2_pct']:.2%}")
    md.append("")

    if best_verdict == "A":
        md.append("### 결론: 전략 전환 적용 가능")
        md.append(f"- `{best_rule}` 로직으로 S1/S2 동적 전환 권장")
    elif best_verdict == "B":
        md.append("### 결론: 부분 성공 (추가 검증 필요)")
        md.append(f"- `{best_rule}` shadow 테스트 후 재판단")
    else:
        md.append("### 결론: 전략 전환 불필요 또는 효과 없음")
    md.append("")

    md_path.write_text("\n".join(md), encoding="utf-8")

    # ── Console ──
    print("=" * 60)
    print("[STRATEGY SWITCHING SUMMARY]")
    print("=" * 60)
    print()

    print("S1 (baseline):")
    for r in all_rows:
        if r["case_name"] == "S1_baseline":
            print(f"  {r['period']}d: return={r['total_return']:.6f}, avg_profit={r['avg_profit']:.8f}, trades={r['total_trades']}, MDD={r['max_drawdown']:.6f}")
    print()

    print("S2 (entropy+P7):")
    for r in all_rows:
        if r["case_name"] == "S2_entropy_P7":
            print(f"  {r['period']}d: return={r['total_return']:.6f}, avg_profit={r['avg_profit']:.8f}, trades={r['total_trades']}, MDD={r['max_drawdown']:.6f}")
    print()

    print("switching rules:")
    for rule in sorted(consistency.keys(), key=lambda r: -consistency[r]["avg_imp_bl"]):
        cs = consistency[rule]
        print(
            f"  {rule}: imp_bl={cs['imp_bl']}, avg_imp={cs['avg_imp_bl']:.6f}, "
            f"imp_ent={cs['imp_ent']}, S2%={cs['avg_s2_pct']:.2%}, verdict={cs['verdict']}"
        )
    print()

    print(f"best rule: {best_rule}")
    print(f"  verdict: {best_verdict}")
    print(f"  avg_imp_vs_baseline: {bc['avg_imp_bl']}")
    print(f"  avg_imp_vs_entropy: {bc['avg_imp_ent']}")
    print(f"  avg_avg_profit_diff: {bc['avg_avg_diff']}")
    print()

    for r in all_rows:
        if r["case_name"] == best_rule:
            print(
                f"  {r['period']}d: return={r['total_return']:.6f}, trades={r['total_trades']}, "
                f"imp_bl={r['imp_vs_baseline']:.6f}, imp_ent={r['imp_vs_entropy']:.6f}, "
                f"S2%={r.get('s2_bars_pct', 0):.2%}"
            )
    print()

    print(f"final verdict: {best_verdict}")
    if best_verdict == "A":
        print(f"  → 실전 적용 가능 ({best_rule})")
    elif best_verdict == "B":
        print(f"  → 부분 성공, shadow 테스트 필요 ({best_rule})")
    else:
        print(f"  → 단일 전략 유지 권장")
    print()

    print(f"created:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")


if __name__ == "__main__":
    main()
