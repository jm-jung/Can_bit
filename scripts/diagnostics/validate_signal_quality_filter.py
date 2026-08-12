"""
Signal Quality / Entry Confidence Ablation 다기간 검증.

baseline vs confidence filter modes 비교.
핵심: "confidence 높은 신호만 쓰면 전략이 살아나는가?"

실행:
  python -m scripts.diagnostics.validate_signal_quality_filter --lookback-list 14,30,60,90
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "diagnostics" / "final"

CONFIDENCE_MODES = [
    # A. max_proba threshold
    "max_proba_055",
    "max_proba_060",
    "max_proba_065",
    "max_proba_070",
    "max_proba_075",
    # B. margin threshold
    "margin_005",
    "margin_010",
    "margin_015",
    "margin_020",
    # C. entropy threshold
    "entropy_120",
    "entropy_110",
    "entropy_100",
    "entropy_090",
    # D. hybrid
    "hybrid_mp060_mg010",
    "hybrid_mp065_et110",
    "hybrid_mg010_et110",
]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-list", type=str, default="14,30,60,90")
    parser.add_argument("--cooldown-bars", type=int, default=12)
    parser.add_argument("--max-holding-bars", type=int, default=12)
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument(
        "--modes", type=str, default="",
        help="comma-separated confidence modes; empty = all",
    )
    args = parser.parse_args()

    lookback_list = [int(x.strip()) for x in args.lookback_list.split(",") if x.strip()]
    modes = (
        [m.strip() for m in args.modes.split(",") if m.strip()]
        if args.modes.strip()
        else list(CONFIDENCE_MODES)
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine

    engine = get_ml_backtest_engine("ml_lstm_attn", "BTCUSDT", "5m")
    proba_long, proba_short, df_full = engine.load_predictions()

    common_kw = dict(
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
        positive_regime_filter_mode="none",
        enable_adaptive_positive_regime_filter=False,
        adaptive_positive_regime_mode="none",
    )

    all_rows: List[Dict[str, Any]] = []

    for period in lookback_list:
        mask = _mask_last_days(df_full, period)
        if int(mask.sum()) <= 0:
            print(f"[WARN] period={period}d mask empty, skip")
            continue

        period_kw = {**common_kw, "index_mask": mask}

        def run_case(case_name: str, extra: Dict[str, Any]) -> Dict[str, Any]:
            res = engine.run_backtest(**{**period_kw, **extra})
            trades = list(_get(res, "trades") or [])
            n_tr = int(_get(res, "total_trades") or 0)
            tr_ret = float(_get(res, "total_return") or 0.0)
            mdd = float(_get(res, "max_drawdown") or 0.0)
            return {
                "case_name": case_name,
                "period": period,
                "total_return": tr_ret,
                "win_rate": _get(res, "win_rate"),
                "total_trades": n_tr,
                "avg_profit": _get(res, "avg_profit"),
                "median_profit": _get(res, "median_profit"),
                "max_drawdown": mdd,
                "profit_factor": _profit_factor(trades),
                "expectancy": _expectancy(trades),
                "trades_per_day": float(n_tr) / float(period) if period > 0 else None,
                "blocked_by_confidence": _get(res, "entries_blocked_by_confidence", 0),
                "score": tr_ret - mdd,
            }

        bl_row = run_case(
            "baseline",
            dict(enable_confidence_filter=False, confidence_filter_mode="none"),
        )
        all_rows.append(bl_row)

        for m in modes:
            row = run_case(
                m,
                dict(enable_confidence_filter=True, confidence_filter_mode=m),
            )
            all_rows.append(row)

    # ── compute deltas ──
    bl_map: Dict[int, Dict[str, float]] = {}
    for r in all_rows:
        if r["case_name"] == "baseline":
            bl_map[r["period"]] = {
                "return": float(r["total_return"]),
                "avg_profit": float(r["avg_profit"] or 0),
                "win_rate": float(r["win_rate"] or 0),
                "mdd": float(r["max_drawdown"]),
                "trades": int(r["total_trades"]),
            }

    for r in all_rows:
        p = r["period"]
        bl = bl_map.get(p, {})
        r_ret = float(r["total_return"])
        r["improvement_vs_baseline"] = r_ret - bl.get("return", 0)
        r["avg_profit_diff"] = float(r["avg_profit"] or 0) - bl.get("avg_profit", 0)
        r["MDD_diff"] = float(r["max_drawdown"]) - bl.get("mdd", 0)
        bl_n = bl.get("trades", 0)
        r["trade_reduction_ratio"] = (
            round(1.0 - int(r["total_trades"]) / bl_n, 4) if bl_n > 0 else None
        )
        r["win_rate_diff"] = float(r["win_rate"] or 0) - bl.get("win_rate", 0)

    # ── consistency per mode ──
    consistency: Dict[str, Dict[str, Any]] = {}
    mode_names = sorted(
        set(r["case_name"] for r in all_rows if r["case_name"] != "baseline")
    )
    for cn in mode_names:
        rows_cn = [r for r in all_rows if r["case_name"] == cn]
        if not rows_cn:
            continue
        imp = [float(r["improvement_vs_baseline"]) for r in rows_cn]
        apd = [float(r["avg_profit_diff"]) for r in rows_cn]
        mdd_d = [float(r["MDD_diff"]) for r in rows_cn]
        trd = [int(r["total_trades"]) for r in rows_cn]
        n_periods = len(rows_cn)
        improved_ret = sum(1 for x in imp if x > 1e-9)
        improved_ap = sum(1 for x in apd if x > 1e-9)
        avg_imp = float(np.mean(imp))
        avg_apd = float(np.mean(apd))
        avg_mdd = float(np.mean(mdd_d))
        avg_trades = float(np.mean(trd))
        avg_trd_ratio = float(np.mean(
            [float(r["trade_reduction_ratio"]) for r in rows_cn if r["trade_reduction_ratio"] is not None]
        )) if any(r["trade_reduction_ratio"] is not None for r in rows_cn) else None
        insuff = sum(1 for tc in trd if tc < args.min_trades)

        if insuff == n_periods:
            verdict = "D"
        elif (
            improved_ret >= 3
            and avg_imp > 0
            and avg_apd > 0
            and (avg_trd_ratio is None or avg_trd_ratio < 0.80)
        ):
            verdict = "A"
        elif improved_ret >= 3 and avg_imp > 0:
            verdict = "B+"
        elif improved_ret >= 2 and avg_imp > 0:
            verdict = "B"
        else:
            verdict = "C"

        consistency[cn] = {
            "improved_return_periods": f"{improved_ret}/{n_periods}",
            "improved_avgprofit_periods": f"{improved_ap}/{n_periods}",
            "avg_improvement": round(avg_imp, 8),
            "avg_avg_profit_diff": round(avg_apd, 8),
            "avg_MDD_diff": round(avg_mdd, 8),
            "avg_trade_count": round(avg_trades, 1),
            "avg_trade_reduction": round(avg_trd_ratio, 4) if avg_trd_ratio is not None else None,
            "verdict": verdict,
        }

    # best candidate by avg_improvement (with minimum trade threshold)
    eligible = {
        k: v for k, v in consistency.items()
        if v["verdict"] in ("A", "B+", "B") and v["avg_trade_count"] >= args.min_trades
    }
    best_name: Optional[str] = None
    if eligible:
        best_name = max(eligible, key=lambda k: eligible[k]["avg_improvement"])
    if not best_name:
        candidates = {k: v for k, v in consistency.items() if v["avg_improvement"] > 0}
        if candidates:
            best_name = max(candidates, key=lambda k: candidates[k]["avg_improvement"])

    # ── CSV ──
    csv_path = OUT_DIR / f"signal_quality_filter_{stamp}.csv"
    if all_rows:
        keys = list(all_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)

    # ── Markdown ──
    md_path = OUT_DIR / f"signal_quality_filter_{stamp}.md"
    md: List[str] = []

    md.append("# Signal Quality / Entry Confidence Ablation")
    md.append("")
    md.append("## 1. 목적")
    md.append("low-quality signal 제거 시 total_return, avg_profit, MDD 개선 여부 검증.")
    md.append("핵심: \"confidence 높은 신호만 쓰면 전략이 살아나는가?\"")
    md.append("")
    md.append("## 2. 실험 설정")
    md.append(f"- lookback periods: {lookback_list}")
    md.append(f"- cooldown={args.cooldown_bars}, max_hold={args.max_holding_bars}")
    md.append(f"- min_trades: {args.min_trades}")
    md.append(f"- confidence modes: {modes}")
    md.append("- 기타 필터: 전부 OFF (순수 signal quality만 평가)")
    md.append("")

    md.append("## 3. Baseline 결과")
    md.append("")
    md.append("| period | return | avg_profit | win_rate | trades | MDD | score |")
    md.append("|--------|--------|------------|----------|--------|-----|-------|")
    for r in all_rows:
        if r["case_name"] == "baseline":
            md.append(
                f"| {r['period']}d | {r['total_return']:.6f} | "
                f"{float(r['avg_profit'] or 0):.8f} | {r['win_rate']} | "
                f"{r['total_trades']} | {r['max_drawdown']:.6f} | {r['score']:.6f} |"
            )
    md.append("")

    md.append("## 4. 모드별 기간 결과")
    md.append("")
    md.append("| mode | period | return | avg_profit | trades | MDD | imp_ret | avg_profit_diff | trade_red | blocked |")
    md.append("|------|--------|--------|------------|--------|-----|---------|-----------------|-----------|---------|")
    for r in all_rows:
        if r["case_name"] != "baseline":
            md.append(
                f"| {r['case_name']} | {r['period']}d | {r['total_return']:.6f} | "
                f"{float(r['avg_profit'] or 0):.8f} | {r['total_trades']} | "
                f"{r['max_drawdown']:.6f} | {r['improvement_vs_baseline']:.6f} | "
                f"{r['avg_profit_diff']:.8f} | "
                f"{r['trade_reduction_ratio']} | {r['blocked_by_confidence']} |"
            )
    md.append("")

    md.append("## 5. Consistency Summary")
    md.append("")
    md.append("| mode | imp_ret_periods | imp_ap_periods | avg_imp | avg_ap_diff | avg_MDD_diff | avg_trades | trade_red | verdict |")
    md.append("|------|-----------------|----------------|---------|-------------|--------------|------------|-----------|---------|")
    for cn, cs in sorted(consistency.items(), key=lambda x: -x[1]["avg_improvement"]):
        md.append(
            f"| {cn} | {cs['improved_return_periods']} | "
            f"{cs['improved_avgprofit_periods']} | {cs['avg_improvement']} | "
            f"{cs['avg_avg_profit_diff']} | {cs['avg_MDD_diff']} | "
            f"{cs['avg_trade_count']} | {cs['avg_trade_reduction']} | {cs['verdict']} |"
        )
    md.append("")

    top5 = sorted(consistency.keys(), key=lambda c: -consistency[c]["avg_improvement"])[:5]
    md.append("## 6. Top 5")
    md.append("")
    for i, cn in enumerate(top5, 1):
        cs = consistency[cn]
        md.append(
            f"{i}. `{cn}` avg_imp={cs['avg_improvement']}, "
            f"avg_ap_diff={cs['avg_avg_profit_diff']}, verdict={cs['verdict']}"
        )
    md.append("")

    md.append("## 7. 핵심 분석")
    md.append("")
    if best_name:
        bc = consistency[best_name]
        best_rows = [r for r in all_rows if r["case_name"] == best_name]
        md.append(f"### Best candidate: `{best_name}` (verdict={bc['verdict']})")
        md.append("")
        md.append("| period | return | avg_profit | trades | imp_ret | avg_profit_diff | trade_red |")
        md.append("|--------|--------|------------|--------|---------|-----------------|-----------|")
        for r in best_rows:
            md.append(
                f"| {r['period']}d | {r['total_return']:.6f} | "
                f"{float(r['avg_profit'] or 0):.8f} | {r['total_trades']} | "
                f"{r['improvement_vs_baseline']:.6f} | {r['avg_profit_diff']:.8f} | "
                f"{r['trade_reduction_ratio']} |"
            )
        md.append("")
        md.append(f"- trade 줄었는데 avg_profit 올라갔는가? → "
                  f"avg_ap_diff={bc['avg_avg_profit_diff']}")
        md.append(f"- MDD 개선? → avg_MDD_diff={bc['avg_MDD_diff']}")
    else:
        md.append("Best candidate 없음 (모든 모드 avg_improvement ≤ 0)")
    md.append("")

    md.append("## 8. 최종 판정")
    md.append("")
    if best_name and consistency[best_name]["verdict"] == "A":
        md.append(f"**A (strong candidate)**: `{best_name}` – avg_profit 및 return 전면 개선")
    elif best_name and consistency[best_name]["verdict"] in ("B+", "B"):
        md.append(f"**{consistency[best_name]['verdict']} (partial candidate)**: `{best_name}` – 일부 기간 개선")
    else:
        md.append("**C (fail)** – confidence gating으로 전략 회복 불가")
    md.append("")

    md.append("## 9. 다음 액션")
    md.append("")
    if best_name and consistency[best_name]["verdict"] in ("A", "B+"):
        md.append(f"- `{best_name}` shadow/paper 테스트 고려")
        md.append("- OOS 추가 검증")
    else:
        md.append("- signal quality gating만으로는 불충분")
        md.append("- 모델 재학습 / feature engineering / 전략 구조 변경 검토 필요")
    md.append("")

    md_path.write_text("\n".join(md), encoding="utf-8")

    # ── console ──
    print("=" * 60)
    print("[SIGNAL QUALITY FILTER SUMMARY]")
    print("=" * 60)
    print()

    print("baseline:")
    for p in lookback_list:
        r = next((r for r in all_rows if r["case_name"] == "baseline" and r["period"] == p), None)
        if r:
            print(
                f"  {p}d: return={r['total_return']:.6f}, "
                f"avg_profit={float(r['avg_profit'] or 0):.8f}, "
                f"trades={r['total_trades']}, MDD={r['max_drawdown']:.6f}"
            )
    print()

    if best_name:
        bc = consistency[best_name]
        print(f"best candidate: {best_name}")
        for p in lookback_list:
            r = next(
                (r for r in all_rows if r["case_name"] == best_name and r["period"] == p),
                None,
            )
            if r:
                print(
                    f"  {p}d: return={r['total_return']:.6f}, "
                    f"avg_profit={float(r['avg_profit'] or 0):.8f}, "
                    f"trades={r['total_trades']}, "
                    f"imp_ret={r['improvement_vs_baseline']:.6f}, "
                    f"ap_diff={r['avg_profit_diff']:.8f}"
                )
        print(f"  consistency: imp_ret={bc['improved_return_periods']}, "
              f"imp_ap={bc['improved_avgprofit_periods']}")
        print(f"  avg_improvement: {bc['avg_improvement']}")
        print(f"  avg_avg_profit_diff: {bc['avg_avg_profit_diff']}")
        print(f"  avg_MDD_diff: {bc['avg_MDD_diff']}")
        print(f"  avg_trade_reduction: {bc['avg_trade_reduction']}")
        print(f"  verdict: {bc['verdict']}")
    else:
        print("best candidate: none")
    print()

    # top 5 summary
    print("top 5 modes:")
    for i, cn in enumerate(top5, 1):
        cs = consistency[cn]
        print(
            f"  {i}. {cn}: avg_imp={cs['avg_improvement']}, "
            f"avg_ap={cs['avg_avg_profit_diff']}, "
            f"trades={cs['avg_trade_count']}, v={cs['verdict']}"
        )
    print()

    final_v = consistency[best_name]["verdict"] if best_name else "C"
    print(f"final verdict: {final_v}")
    print()
    print(f"created:\n- {csv_path}\n- {md_path}")


if __name__ == "__main__":
    main()
