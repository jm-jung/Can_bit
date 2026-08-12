"""
Entropy-based Signal Quality Filter 최종 검증.

entropy ≤ 1.00 필터의 실전 적용 가능성 판단:
- 장기(180d)에서도 유지되는가?
- 단독 vs positive_filter(P7) 결합 시 어느 구조가 더 좋은가?
- trade set 분석: 제거된 trade가 손실 위주인가?

실행:
  python -m scripts.diagnostics.validate_entropy_final --lookback-list 14,30,60,90,180
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "diagnostics" / "final"


def _get(result: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _mask_last_days(df, lookback_days: int):
    import pandas as pd

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


def _calmar(total_return: float, max_drawdown: float) -> Optional[float]:
    if max_drawdown < 1e-15:
        return None if abs(total_return) < 1e-15 else float("inf")
    return total_return / max_drawdown


def _trade_set_analysis(
    bl_trades: List[Mapping[str, Any]],
    filtered_trades: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compare baseline vs filtered trade sets by entry_time."""
    bl_times = {t["entry_time"] for t in bl_trades if t.get("entry_time")}
    flt_times = {t["entry_time"] for t in filtered_trades if t.get("entry_time")}

    common_times = bl_times & flt_times
    removed_times = bl_times - flt_times
    new_times = flt_times - bl_times

    removed_profits = [
        float(t["profit"])
        for t in bl_trades
        if t.get("entry_time") in removed_times and t.get("profit") is not None
    ]
    new_profits = [
        float(t["profit"])
        for t in filtered_trades
        if t.get("entry_time") in new_times and t.get("profit") is not None
    ]
    common_bl_profits = [
        float(t["profit"])
        for t in bl_trades
        if t.get("entry_time") in common_times and t.get("profit") is not None
    ]

    removed_loss_count = sum(1 for p in removed_profits if p < 0)
    removed_win_count = sum(1 for p in removed_profits if p > 0)
    new_loss_count = sum(1 for p in new_profits if p < 0)
    new_win_count = sum(1 for p in new_profits if p > 0)

    return {
        "common_count": len(common_times),
        "removed_count": len(removed_times),
        "new_count": len(new_times),
        "removed_avg_profit": float(np.mean(removed_profits)) if removed_profits else None,
        "removed_sum_profit": float(np.sum(removed_profits)) if removed_profits else 0.0,
        "removed_loss_count": removed_loss_count,
        "removed_win_count": removed_win_count,
        "removed_loss_ratio": removed_loss_count / len(removed_profits) if removed_profits else None,
        "new_avg_profit": float(np.mean(new_profits)) if new_profits else None,
        "new_sum_profit": float(np.sum(new_profits)) if new_profits else 0.0,
        "new_loss_count": new_loss_count,
        "new_win_count": new_win_count,
        "common_avg_profit": float(np.mean(common_bl_profits)) if common_bl_profits else None,
    }


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

    CASES = [
        {
            "name": "baseline",
            "extra": dict(enable_confidence_filter=False, confidence_filter_mode="none"),
        },
        {
            "name": "entropy_100",
            "extra": dict(enable_confidence_filter=True, confidence_filter_mode="entropy_100"),
        },
        {
            "name": "entropy_100_P7",
            "extra": dict(
                enable_confidence_filter=True,
                confidence_filter_mode="entropy_100",
                enable_positive_regime_filter=True,
                positive_regime_filter_mode="allow_ema_below_or_strong_up",
            ),
        },
    ]

    all_rows: List[Dict[str, Any]] = []
    trade_sets: Dict[str, Dict[int, List[Mapping[str, Any]]]] = {c["name"]: {} for c in CASES}
    trade_analysis: List[Dict[str, Any]] = []

    for period in lookback_list:
        mask = _mask_last_days(df_full, period)
        if int(mask.sum()) <= 0:
            print(f"[WARN] period={period}d mask empty, skip")
            continue

        period_kw = {**common_kw, "index_mask": mask}

        for case in CASES:
            res = engine.run_backtest(**{**period_kw, **case["extra"]})
            trades = list(_get(res, "trades") or [])
            n_tr = int(_get(res, "total_trades") or 0)
            tr_ret = float(_get(res, "total_return") or 0.0)
            mdd = float(_get(res, "max_drawdown") or 0.0)

            trade_sets[case["name"]][period] = trades

            row = {
                "case_name": case["name"],
                "period": period,
                "total_return": tr_ret,
                "win_rate": float(_get(res, "win_rate") or 0.0),
                "total_trades": n_tr,
                "avg_profit": float(_get(res, "avg_profit") or 0.0),
                "median_profit": float(_get(res, "median_profit") or 0.0),
                "max_drawdown": mdd,
                "profit_factor": _profit_factor(trades),
                "expectancy": _expectancy(trades),
                "trades_per_day": float(n_tr) / float(period) if period > 0 else None,
                "sharpe": _sharpe(trades),
                "calmar": _calmar(tr_ret, mdd),
            }
            all_rows.append(row)

    # ── compute diffs vs baseline ──
    bl_map: Dict[int, Dict[str, float]] = {}
    for r in all_rows:
        if r["case_name"] == "baseline":
            bl_map[r["period"]] = {
                "total_return": r["total_return"],
                "avg_profit": r["avg_profit"],
                "win_rate": r["win_rate"],
                "max_drawdown": r["max_drawdown"],
                "total_trades": r["total_trades"],
            }

    entropy_map: Dict[int, Dict[str, float]] = {}
    for r in all_rows:
        if r["case_name"] == "entropy_100":
            entropy_map[r["period"]] = {
                "total_return": r["total_return"],
                "avg_profit": r["avg_profit"],
                "total_trades": r["total_trades"],
            }

    for r in all_rows:
        p = r["period"]
        bl = bl_map.get(p, {})
        ent = entropy_map.get(p, {})

        r["improvement_vs_baseline"] = r["total_return"] - bl.get("total_return", 0.0)
        r["avg_profit_diff"] = r["avg_profit"] - bl.get("avg_profit", 0.0)
        r["win_rate_diff"] = r["win_rate"] - bl.get("win_rate", 0.0)
        r["MDD_diff"] = r["max_drawdown"] - bl.get("max_drawdown", 0.0)
        bl_nt = bl.get("total_trades", 0)
        r["trade_reduction_ratio"] = (1.0 - r["total_trades"] / bl_nt) if bl_nt > 0 else 0.0

        r["improvement_vs_entropy"] = r["total_return"] - ent.get("total_return", 0.0)

    # ── trade set analysis ──
    for period in lookback_list:
        bl_trades = trade_sets["baseline"].get(period, [])
        for case_name in ["entropy_100", "entropy_100_P7"]:
            flt_trades = trade_sets[case_name].get(period, [])
            ta = _trade_set_analysis(bl_trades, flt_trades)
            ta["case_name"] = case_name
            ta["period"] = period
            trade_analysis.append(ta)

    # ── consistency per case ──
    consistency: Dict[str, Dict[str, Any]] = {}
    for case in CASES:
        cn = case["name"]
        if cn == "baseline":
            continue
        rows_cn = [r for r in all_rows if r["case_name"] == cn]
        if not rows_cn:
            continue
        imp_list = [r["improvement_vs_baseline"] for r in rows_cn]
        avg_diffs = [r["avg_profit_diff"] for r in rows_cn]
        mdd_diffs = [r["MDD_diff"] for r in rows_cn]
        trade_counts = [r["total_trades"] for r in rows_cn]
        trade_reds = [r["trade_reduction_ratio"] for r in rows_cn]

        improved_ret = sum(1 for x in imp_list if x > 1e-9)
        improved_avg = sum(1 for x in avg_diffs if x > 1e-9)
        n_periods = len(rows_cn)

        avg_imp = float(np.mean(imp_list))
        worst_imp = float(min(imp_list))
        best_imp = float(max(imp_list))
        avg_avg_diff = float(np.mean(avg_diffs))
        avg_mdd_diff = float(np.mean(mdd_diffs))
        avg_trade_red = float(np.mean(trade_reds))
        avg_trades = float(np.mean(trade_counts))

        if improved_ret >= 4 and avg_imp > 0 and avg_avg_diff > 0:
            verdict = "A"
        elif improved_ret >= 4 and avg_imp > 0:
            verdict = "A"
        elif improved_ret >= 3 and avg_imp > 0:
            verdict = "B"
        elif improved_ret >= 2:
            verdict = "B"
        else:
            verdict = "C"

        consistency[cn] = {
            "improved_periods": f"{improved_ret}/{n_periods}",
            "improved_avg_profit": f"{improved_avg}/{n_periods}",
            "avg_improvement": round(avg_imp, 8),
            "worst_improvement": round(worst_imp, 8),
            "best_improvement": round(best_imp, 8),
            "avg_avg_profit_diff": round(avg_avg_diff, 8),
            "avg_MDD_diff": round(avg_mdd_diff, 8),
            "avg_trade_reduction": round(avg_trade_red, 4),
            "avg_trades": round(avg_trades, 1),
            "verdict": verdict,
        }

    # ── determine best case ──
    best_case = max(consistency.keys(), key=lambda c: consistency[c]["avg_improvement"])
    best_verdict = consistency[best_case]["verdict"]

    # ── CSV ──
    csv_path = OUT_DIR / f"entropy_final_{stamp}.csv"
    if all_rows:
        keys = list(all_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)

    # ── Markdown ──
    md_path = OUT_DIR / f"entropy_final_{stamp}.md"
    md: List[str] = []

    md.append("# Entropy-based Signal Quality Filter 최종 검증")
    md.append("")
    md.append("## 1. 목적")
    md.append("entropy ≤ 1.00 필터의 실전 적용 가능성 최종 판단.")
    md.append("- 장기(180d) 유지 여부")
    md.append("- 단독 vs P7(allow_ema_below_or_strong_up) 결합 비교")
    md.append("- Trade set 분석: 제거 trade 품질 검증")
    md.append("")

    md.append("## 2. 설정")
    md.append(f"- lookback periods: {lookback_list}")
    md.append(f"- cooldown={args.cooldown_bars}, max_hold={args.max_holding_bars}")
    md.append("- Cases: baseline, entropy_100, entropy_100+P7")
    md.append("- Other filters: ALL OFF")
    md.append("")

    md.append("## 3. Baseline 결과")
    md.append("")
    md.append("| period | return | win_rate | trades | avg_profit | MDD | sharpe | calmar |")
    md.append("|--------|--------|----------|--------|------------|-----|--------|--------|")
    for r in all_rows:
        if r["case_name"] == "baseline":
            sh = f"{r['sharpe']:.4f}" if r['sharpe'] is not None else "N/A"
            ca = f"{r['calmar']:.4f}" if r['calmar'] is not None else "N/A"
            md.append(
                f"| {r['period']}d | {r['total_return']:.6f} | {r['win_rate']:.4f} | "
                f"{r['total_trades']} | {r['avg_profit']:.8f} | {r['max_drawdown']:.6f} | {sh} | {ca} |"
            )
    md.append("")

    md.append("## 4. Entropy 100 결과")
    md.append("")
    md.append("| period | return | win_rate | trades | avg_profit | MDD | imp_vs_bl | avg_p_diff | trade_red | sharpe |")
    md.append("|--------|--------|----------|--------|------------|-----|-----------|------------|-----------|--------|")
    for r in all_rows:
        if r["case_name"] == "entropy_100":
            sh = f"{r['sharpe']:.4f}" if r['sharpe'] is not None else "N/A"
            md.append(
                f"| {r['period']}d | {r['total_return']:.6f} | {r['win_rate']:.4f} | "
                f"{r['total_trades']} | {r['avg_profit']:.8f} | {r['max_drawdown']:.6f} | "
                f"{r['improvement_vs_baseline']:.6f} | {r['avg_profit_diff']:.8f} | "
                f"{r['trade_reduction_ratio']:.4f} | {sh} |"
            )
    md.append("")

    md.append("## 5. Entropy 100 + P7 결과")
    md.append("")
    md.append("| period | return | win_rate | trades | avg_profit | MDD | imp_vs_bl | imp_vs_ent | trade_red | sharpe |")
    md.append("|--------|--------|----------|--------|------------|-----|-----------|------------|-----------|--------|")
    for r in all_rows:
        if r["case_name"] == "entropy_100_P7":
            sh = f"{r['sharpe']:.4f}" if r['sharpe'] is not None else "N/A"
            md.append(
                f"| {r['period']}d | {r['total_return']:.6f} | {r['win_rate']:.4f} | "
                f"{r['total_trades']} | {r['avg_profit']:.8f} | {r['max_drawdown']:.6f} | "
                f"{r['improvement_vs_baseline']:.6f} | {r['improvement_vs_entropy']:.6f} | "
                f"{r['trade_reduction_ratio']:.4f} | {sh} |"
            )
    md.append("")

    md.append("## 6. Consistency Summary")
    md.append("")
    md.append("| case | imp_periods | imp_avg_p | avg_imp | worst | best | avg_avgP | avg_MDD | trade_red | avg_trades | verdict |")
    md.append("|------|-------------|-----------|---------|-------|------|----------|---------|-----------|------------|---------|")
    for cn, cs in consistency.items():
        md.append(
            f"| {cn} | {cs['improved_periods']} | {cs['improved_avg_profit']} | "
            f"{cs['avg_improvement']} | {cs['worst_improvement']} | {cs['best_improvement']} | "
            f"{cs['avg_avg_profit_diff']} | {cs['avg_MDD_diff']} | "
            f"{cs['avg_trade_reduction']} | {cs['avg_trades']} | **{cs['verdict']}** |"
        )
    md.append("")

    md.append("## 7. Trade Set 분석")
    md.append("")
    md.append("| case | period | common | removed | new | removed_avg_p | removed_loss% | new_avg_p | removed_sum |")
    md.append("|------|--------|--------|---------|-----|---------------|---------------|-----------|-------------|")
    for ta in trade_analysis:
        r_avg = f"{ta['removed_avg_profit']:.8f}" if ta['removed_avg_profit'] is not None else "N/A"
        r_loss = f"{ta['removed_loss_ratio']:.2%}" if ta['removed_loss_ratio'] is not None else "N/A"
        n_avg = f"{ta['new_avg_profit']:.8f}" if ta['new_avg_profit'] is not None else "N/A"
        md.append(
            f"| {ta['case_name']} | {ta['period']}d | {ta['common_count']} | "
            f"{ta['removed_count']} | {ta['new_count']} | {r_avg} | {r_loss} | "
            f"{n_avg} | {ta['removed_sum_profit']:.6f} |"
        )
    md.append("")

    md.append("## 8. Trade Set 핵심 진단")
    md.append("")
    entropy_ta = [ta for ta in trade_analysis if ta["case_name"] == "entropy_100"]
    if entropy_ta:
        total_removed = sum(ta["removed_count"] for ta in entropy_ta)
        total_removed_losses = sum(ta["removed_loss_count"] for ta in entropy_ta)
        total_removed_wins = sum(ta["removed_win_count"] for ta in entropy_ta)
        removed_profits_all = []
        for ta in entropy_ta:
            removed_profits_all.append(ta["removed_sum_profit"])
        avg_removed_sum = float(np.mean(removed_profits_all)) if removed_profits_all else 0.0
        md.append(f"- 총 제거된 trade: {total_removed}")
        md.append(f"- 제거 중 손실 trade: {total_removed_losses} ({total_removed_losses/total_removed*100:.1f}%)" if total_removed > 0 else "- 제거 trade 없음")
        md.append(f"- 제거 중 이익 trade: {total_removed_wins}")
        md.append(f"- 제거 trade 평균 합산 수익: {avg_removed_sum:.6f}")
        if total_removed > 0 and total_removed_losses / total_removed > 0.5:
            md.append("- **제거된 trade가 손실 위주** → 필터가 올바르게 동작")
        elif total_removed > 0:
            md.append("- 제거된 trade에 이익 trade도 포함 → 일부 기회 손실 존재")
        md.append("")

    md.append("## 9. 최종 판정")
    md.append("")
    md.append(f"**Best case: `{best_case}`**")
    md.append(f"- Verdict: **{best_verdict}**")
    md.append(f"- Consistency: {consistency[best_case]['improved_periods']}")
    md.append(f"- Avg improvement: {consistency[best_case]['avg_improvement']}")
    md.append(f"- Avg avg_profit diff: {consistency[best_case]['avg_avg_profit_diff']}")
    md.append(f"- Avg trade reduction: {consistency[best_case]['avg_trade_reduction']}")
    md.append("")

    if best_verdict == "A":
        md.append("### 결론: 실전 적용 가능")
        md.append(f"- `{best_case}` 필터를 shadow/paper → live 순차 배포 권장")
    elif best_verdict == "B":
        md.append("### 결론: 부분 적용 (추가 검증 필요)")
        md.append(f"- `{best_case}` 필터 shadow 테스트 후 재판단")
    else:
        md.append("### 결론: 보류")
        md.append("- 장기 성과 불안정. 추가 연구 필요.")
    md.append("")

    md.append("## 10. 운영 적용 권고")
    md.append("")
    if best_verdict == "A":
        md.append("**YES** – 실전 적용 권고")
        md.append("")
        md.append(f"적용 파라미터:")
        md.append(f"- `max_entropy = 1.00`")
        if best_case == "entropy_100_P7":
            md.append(f"- `enable_positive_regime_filter = True`")
            md.append(f"- `positive_regime_filter_mode = \"allow_ema_below_or_strong_up\"`")
    elif best_verdict == "B":
        md.append("**CONDITIONAL** – shadow 테스트 후 결정")
    else:
        md.append("**NO** – 현재 상태에서 적용 부적합")
    md.append("")

    md_path.write_text("\n".join(md), encoding="utf-8")

    # ── console output ──
    print("=" * 60)
    print("[ENTROPY FINAL SUMMARY]")
    print("=" * 60)
    print()

    print("baseline:")
    for p in lookback_list:
        bl = bl_map.get(p)
        if bl:
            print(f"  {p}d: return={bl['total_return']:.6f}, avg_profit={bl['avg_profit']:.8f}, trades={int(bl['total_trades'])}, MDD={bl['max_drawdown']:.6f}")
    print()

    print("entropy_100:")
    for r in all_rows:
        if r["case_name"] == "entropy_100":
            print(
                f"  {r['period']}d: return={r['total_return']:.6f}, avg_profit={r['avg_profit']:.8f}, "
                f"trades={r['total_trades']}, imp={r['improvement_vs_baseline']:.6f}"
            )
    print()

    print("entropy_100 + P7:")
    for r in all_rows:
        if r["case_name"] == "entropy_100_P7":
            print(
                f"  {r['period']}d: return={r['total_return']:.6f}, avg_profit={r['avg_profit']:.8f}, "
                f"trades={r['total_trades']}, imp_vs_bl={r['improvement_vs_baseline']:.6f}, "
                f"imp_vs_ent={r['improvement_vs_entropy']:.6f}"
            )
    print()

    print("consistency:")
    for cn, cs in consistency.items():
        print(
            f"  {cn}: imp={cs['improved_periods']}, avg_imp={cs['avg_improvement']}, "
            f"avg_avgP={cs['avg_avg_profit_diff']}, trade_red={cs['avg_trade_reduction']:.4f}, "
            f"verdict={cs['verdict']}"
        )
    print()

    print("trade set analysis (entropy_100):")
    for ta in trade_analysis:
        if ta["case_name"] == "entropy_100":
            r_loss = f"{ta['removed_loss_ratio']:.0%}" if ta['removed_loss_ratio'] is not None else "N/A"
            print(
                f"  {ta['period']}d: common={ta['common_count']}, removed={ta['removed_count']}, "
                f"new={ta['new_count']}, removed_loss%={r_loss}, removed_sum={ta['removed_sum_profit']:.6f}"
            )
    print()

    print(f"best case: {best_case}")
    print(f"  verdict: {best_verdict}")
    print(f"  avg_improvement: {consistency[best_case]['avg_improvement']}")
    print(f"  avg_avg_profit_diff: {consistency[best_case]['avg_avg_profit_diff']}")
    print()

    if best_verdict == "A":
        print(f"final verdict: A → 실전 적용 가능 ({best_case})")
    elif best_verdict == "B":
        print(f"final verdict: B → 부분 적용/추가 검증 ({best_case})")
    else:
        print(f"final verdict: C → 보류")
    print()

    print(f"created:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")


if __name__ == "__main__":
    main()
