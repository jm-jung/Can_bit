"""
Adaptive positive regime filter 다기간 검증.

baseline / static positive / adaptive modes 비교.

실행:
  python -m scripts.diagnostics.validate_adaptive_positive_filter --lookback-list 14,30,60,90
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

ADAPTIVE_MODES = [
    "recent_return_gate_001",
    "recent_return_gate_0015",
    "recent_return_gate_002",
    "volatility_mid_high",
    "trend_strength_001",
    "trend_strength_0015",
    "trend_strength_002",
    "hybrid_regime",
    "hybrid_defensive",
    "combined_best",
]

POSITIVE_MODE = "allow_ema_below_or_strong_up"


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


def _calmar(total_return: float, mdd: float) -> Optional[float]:
    if mdd < 1e-15:
        return None
    return total_return / mdd


def _entry_indices(result: Mapping[str, Any]) -> set[int]:
    events = _get(result, "trade_events") or []
    idxs: set[int] = set()
    for ev in events:
        if isinstance(ev, dict) and str(ev.get("event", "")).startswith("ENTRY"):
            idx = ev.get("idx")
            if idx is not None:
                idxs.add(int(idx))
    trades = _get(result, "trades") or []
    for t in trades:
        if isinstance(t, dict):
            idx = t.get("entry_idx") or t.get("entry_bar")
            if idx is not None:
                idxs.add(int(idx))
    return idxs


def _trade_profit_map(result: Mapping[str, Any]) -> Dict[int, float]:
    trades = _get(result, "trades") or []
    out: Dict[int, float] = {}
    for t in trades:
        if isinstance(t, dict):
            idx = t.get("entry_idx") or t.get("entry_bar")
            profit = t.get("profit")
            if idx is not None and profit is not None:
                out[int(idx)] = float(profit)
    return out


def _trade_set_analysis(
    bl_result: Mapping[str, Any], cand_result: Mapping[str, Any]
) -> Dict[str, Any]:
    bl_pm = _trade_profit_map(bl_result)
    cand_pm = _trade_profit_map(cand_result)

    bl_keys = set(bl_pm.keys())
    cand_keys = set(cand_pm.keys())

    common_keys = bl_keys & cand_keys
    removed_keys = bl_keys - cand_keys
    new_keys = cand_keys - bl_keys

    def _group_stats(pm: Dict[int, float], keys: set[int]) -> Dict[str, Any]:
        vals = [pm[k] for k in keys if k in pm]
        if not vals:
            return {"count": 0, "win_rate": None, "total_profit": 0.0, "mean_profit": None}
        arr = np.array(vals)
        return {
            "count": len(vals),
            "win_rate": float(np.mean(arr > 0)),
            "total_profit": float(np.sum(arr)),
            "mean_profit": float(np.mean(arr)),
        }

    return {
        "common": _group_stats(bl_pm, common_keys),
        "removed": _group_stats(bl_pm, removed_keys),
        "new": _group_stats(cand_pm, new_keys),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-list", type=str, default="14,30,60,90")
    parser.add_argument("--cooldown-bars", type=int, default=12)
    parser.add_argument("--max-holding-bars", type=int, default=12)
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--modes", type=str, default="",
                        help="comma-separated adaptive modes; empty = all")
    args = parser.parse_args()

    lookback_list = [int(x.strip()) for x in args.lookback_list.split(",") if x.strip()]
    modes = (
        [m.strip() for m in args.modes.split(",") if m.strip()]
        if args.modes.strip()
        else list(ADAPTIVE_MODES)
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
        emit_trade_log=True,
        max_holding_bars=args.max_holding_bars,
        enable_profit_lock_exit=False,
        enable_regime_filter=False,
        regime_filter_mode="none",
    )

    all_rows: List[Dict[str, Any]] = []
    raw_results: Dict[str, Dict[int, Any]] = {}

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

            key = f"{case_name}_{period}"
            raw_results[key] = res

            row = {
                "case_name": case_name,
                "period": period,
                "total_return": _get(res, "total_return"),
                "win_rate": _get(res, "win_rate"),
                "total_trades": n_tr,
                "avg_profit": _get(res, "avg_profit"),
                "median_profit": _get(res, "median_profit"),
                "max_drawdown": mdd,
                "profit_factor": _profit_factor(trades),
                "expectancy": _expectancy(trades),
                "trades_per_day": float(n_tr) / float(period) if period > 0 else None,
                "blocked_entries": _get(res, "entries_blocked_by_adaptive_positive", 0),
                "adaptive_gate_on_count": _get(res, "adaptive_positive_gate_on_count", 0),
                "entries_blocked_by_adaptive_positive": _get(
                    res, "entries_blocked_by_adaptive_positive", 0
                ),
                "sharpe": _sharpe(trades),
                "calmar": _calmar(tr_ret, mdd),
                "score": tr_ret - mdd,
            }
            return row

        bl_row = run_case(
            "baseline",
            dict(
                enable_positive_regime_filter=False,
                positive_regime_filter_mode="none",
                enable_adaptive_positive_regime_filter=False,
                adaptive_positive_regime_mode="none",
            ),
        )
        all_rows.append(bl_row)

        static_row = run_case(
            "static_positive",
            dict(
                enable_positive_regime_filter=True,
                positive_regime_filter_mode=POSITIVE_MODE,
                enable_adaptive_positive_regime_filter=False,
                adaptive_positive_regime_mode="none",
            ),
        )
        all_rows.append(static_row)

        for m in modes:
            ad_row = run_case(
                f"adaptive_{m}",
                dict(
                    enable_positive_regime_filter=False,
                    positive_regime_filter_mode="none",
                    enable_adaptive_positive_regime_filter=True,
                    adaptive_positive_regime_mode=m,
                ),
            )
            all_rows.append(ad_row)

    bl_map: Dict[int, float] = {}
    static_map: Dict[int, float] = {}
    for r in all_rows:
        if r["case_name"] == "baseline":
            bl_map[r["period"]] = float(r["total_return"] or 0)
        elif r["case_name"] == "static_positive":
            static_map[r["period"]] = float(r["total_return"] or 0)

    bl_wr_map: Dict[int, float] = {}
    bl_mdd_map: Dict[int, float] = {}
    bl_ntrades_map: Dict[int, int] = {}
    for r in all_rows:
        if r["case_name"] == "baseline":
            bl_wr_map[r["period"]] = float(r["win_rate"] or 0)
            bl_mdd_map[r["period"]] = float(r["max_drawdown"] or 0)
            bl_ntrades_map[r["period"]] = int(r["total_trades"] or 0)

    for r in all_rows:
        p = r["period"]
        bl_ret = bl_map.get(p, 0.0)
        st_ret = static_map.get(p, 0.0)
        r_ret = float(r["total_return"] or 0)
        r["improvement_vs_baseline"] = r_ret - bl_ret
        r["improvement_vs_static"] = r_ret - st_ret
        r["win_rate_diff_vs_baseline"] = float(r["win_rate"] or 0) - bl_wr_map.get(p, 0.0)
        r["MDD_diff_vs_baseline"] = float(r["max_drawdown"] or 0) - bl_mdd_map.get(p, 0.0)
        r["trade_diff_vs_baseline"] = int(r["total_trades"] or 0) - bl_ntrades_map.get(p, 0)

    # ── consistency ──
    consistency: Dict[str, Dict[str, Any]] = {}
    case_names = sorted(
        set(r["case_name"] for r in all_rows if r["case_name"] not in ("baseline", "static_positive"))
    )
    for cn in case_names:
        rows_cn = [r for r in all_rows if r["case_name"] == cn]
        if not rows_cn:
            continue
        imp_bl = [float(r["improvement_vs_baseline"]) for r in rows_cn]
        imp_st = [float(r["improvement_vs_static"]) for r in rows_cn]
        mdd_diffs = [float(r["MDD_diff_vs_baseline"]) for r in rows_cn]
        trade_counts = [int(r["total_trades"]) for r in rows_cn]
        improved_bl = sum(1 for x in imp_bl if x > 1e-9)
        improved_st = sum(1 for x in imp_st if x > 1e-9)
        n_periods = len(rows_cn)
        avg_imp_bl = float(np.mean(imp_bl)) if imp_bl else 0.0
        worst_imp_bl = float(min(imp_bl)) if imp_bl else 0.0
        avg_mdd_diff = float(np.mean(mdd_diffs)) if mdd_diffs else 0.0
        avg_trade_count = float(np.mean(trade_counts)) if trade_counts else 0.0
        insuff = sum(1 for tc in trade_counts if tc < args.min_trades)

        if insuff == n_periods:
            verdict = "D"
        elif (
            improved_bl >= 3
            and improved_st >= 3
            and avg_imp_bl > 0
            and worst_imp_bl > (float(min(imp_st)) if imp_st else 0.0)
            and avg_mdd_diff <= 0
        ):
            verdict = "A"
        elif improved_bl >= 3 and avg_imp_bl > 0:
            verdict = "B"
        elif improved_bl >= 2:
            verdict = "C"
        else:
            verdict = "C"

        consistency[cn] = {
            "improved_periods_vs_baseline": improved_bl,
            "improved_periods_vs_static": improved_st,
            "improvement_ratio_vs_baseline": f"{improved_bl}/{n_periods}",
            "avg_improvement_vs_baseline": round(avg_imp_bl, 6),
            "worst_improvement_vs_baseline": round(worst_imp_bl, 6),
            "avg_MDD_diff": round(avg_mdd_diff, 6),
            "avg_trade_count": round(avg_trade_count, 1),
            "verdict": verdict,
        }

    best_adaptive_name: Optional[str] = None
    best_adaptive_score = -1e18
    for cn, cs in consistency.items():
        score = cs["avg_improvement_vs_baseline"] * cs["improved_periods_vs_baseline"]
        if score > best_adaptive_score:
            best_adaptive_score = score
            best_adaptive_name = cn

    # ── trade set analysis (baseline vs best adaptive, per period) ──
    trade_set_results: Dict[int, Dict[str, Any]] = {}
    if best_adaptive_name:
        for period in lookback_list:
            bl_key = f"baseline_{period}"
            ad_key = f"{best_adaptive_name}_{period}"
            if bl_key in raw_results and ad_key in raw_results:
                trade_set_results[period] = _trade_set_analysis(
                    raw_results[bl_key], raw_results[ad_key]
                )

    # ── CSV ──
    csv_path = OUT_DIR / f"adaptive_positive_filter_{stamp}.csv"
    if all_rows:
        keys = list(all_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)

    # ── Markdown ──
    md_path = OUT_DIR / f"adaptive_positive_filter_{stamp}.md"
    md: List[str] = []

    md.append("# Adaptive Positive Regime Filter Validation")
    md.append("")
    md.append("## 1. 목적")
    md.append("Positive filter를 조건부 (adaptive)로 ON/OFF 했을 때 baseline 대비 안정적 개선 검증.")
    md.append("")
    md.append("## 2. 실험 설정")
    md.append(f"- lookback periods: {lookback_list}")
    md.append(f"- cooldown={args.cooldown_bars}, max_hold={args.max_holding_bars}")
    md.append(f"- positive mode (fixed): {POSITIVE_MODE}")
    md.append(f"- adaptive modes: {modes}")
    md.append(f"- min_trades: {args.min_trades}")
    md.append("")

    md.append("## 3. Baseline 결과")
    md.append("")
    md.append("| period | return | win_rate | trades | MDD | score |")
    md.append("|--------|--------|----------|--------|-----|-------|")
    for r in all_rows:
        if r["case_name"] == "baseline":
            md.append(
                f"| {r['period']}d | {r['total_return']} | {r['win_rate']} | "
                f"{r['total_trades']} | {r['max_drawdown']} | {r['score']:.6f} |"
            )
    md.append("")

    md.append("## 4. Static Positive 결과")
    md.append("")
    md.append("| period | return | win_rate | trades | MDD | imp_vs_bl | score |")
    md.append("|--------|--------|----------|--------|-----|-----------|-------|")
    for r in all_rows:
        if r["case_name"] == "static_positive":
            md.append(
                f"| {r['period']}d | {r['total_return']} | {r['win_rate']} | "
                f"{r['total_trades']} | {r['max_drawdown']} | "
                f"{r['improvement_vs_baseline']:.6f} | {r['score']:.6f} |"
            )
    md.append("")

    md.append("## 5. Adaptive Mode별 기간 결과")
    md.append("")
    md.append("| case | period | return | trades | MDD | imp_vs_bl | imp_vs_static | score |")
    md.append("|------|--------|--------|--------|-----|-----------|---------------|-------|")
    for r in all_rows:
        if r["case_name"] not in ("baseline", "static_positive"):
            md.append(
                f"| {r['case_name']} | {r['period']}d | {r['total_return']} | "
                f"{r['total_trades']} | {r['max_drawdown']} | "
                f"{r['improvement_vs_baseline']:.6f} | {r['improvement_vs_static']:.6f} | "
                f"{r['score']:.6f} |"
            )
    md.append("")

    md.append("## 6. Consistency Summary")
    md.append("")
    md.append("| mode | improved_bl | improved_st | avg_imp_bl | worst_imp_bl | avg_MDD_diff | avg_trades | verdict |")
    md.append("|------|-------------|-------------|------------|--------------|--------------|------------|---------|")
    for cn, cs in sorted(consistency.items(), key=lambda x: -x[1]["avg_improvement_vs_baseline"]):
        md.append(
            f"| {cn} | {cs['improvement_ratio_vs_baseline']} | "
            f"{cs['improved_periods_vs_static']} | {cs['avg_improvement_vs_baseline']} | "
            f"{cs['worst_improvement_vs_baseline']} | {cs['avg_MDD_diff']} | "
            f"{cs['avg_trade_count']} | {cs['verdict']} |"
        )
    md.append("")

    top3_names = sorted(consistency.keys(), key=lambda c: -consistency[c]["avg_improvement_vs_baseline"])[:3]
    md.append("## 7. Top 3 Adaptive Modes")
    md.append("")
    for i, cn in enumerate(top3_names, 1):
        cs = consistency[cn]
        md.append(f"{i}. `{cn}` avg_imp={cs['avg_improvement_vs_baseline']}, verdict={cs['verdict']}")
    md.append("")

    md.append("## 8. 14d 악화 방어 여부")
    md.append("")
    static_14d = next((r for r in all_rows if r["case_name"] == "static_positive" and r["period"] == 14), None)
    if static_14d:
        static_14_imp = float(static_14d["improvement_vs_baseline"])
        md.append(f"- Static 14d imp_vs_baseline: {static_14_imp:.6f}")
    if best_adaptive_name:
        ad_14 = next(
            (r for r in all_rows if r["case_name"] == best_adaptive_name and r["period"] == 14),
            None,
        )
        if ad_14:
            ad_14_imp = float(ad_14["improvement_vs_baseline"])
            md.append(f"- Best adaptive ({best_adaptive_name}) 14d imp_vs_baseline: {ad_14_imp:.6f}")
            defended = ad_14_imp > static_14_imp if static_14d else False
            md.append(f"- 14d 방어 성공: {'YES' if defended else 'NO'}")
    md.append("")

    md.append("## 9. 30/60/90 개선 유지 여부")
    md.append("")
    if best_adaptive_name:
        for p in [30, 60, 90]:
            ad_r = next(
                (r for r in all_rows if r["case_name"] == best_adaptive_name and r["period"] == p),
                None,
            )
            if ad_r:
                md.append(f"- {p}d: imp_vs_bl={ad_r['improvement_vs_baseline']:.6f}, imp_vs_st={ad_r['improvement_vs_static']:.6f}")
    md.append("")

    md.append("## 10. Trade Set 분석 (baseline vs best adaptive)")
    md.append("")
    if trade_set_results:
        for period, ts in trade_set_results.items():
            md.append(f"### {period}d")
            for grp in ("common", "removed", "new"):
                g = ts[grp]
                md.append(
                    f"- {grp}: count={g['count']}, win_rate={g['win_rate']}, "
                    f"total_profit={g['total_profit']:.6f}, mean_profit={g['mean_profit']}"
                )
            md.append("")
    else:
        md.append("Trade set 분석 불가 (best adaptive 없음)")
        md.append("")

    static_cs = None
    for r in all_rows:
        if r["case_name"] == "static_positive":
            break
    static_imp_list = [
        float(r["improvement_vs_baseline"])
        for r in all_rows
        if r["case_name"] == "static_positive"
    ]
    static_improved = sum(1 for x in static_imp_list if x > 1e-9)

    md.append("## 11. 최종 판정")
    md.append("")
    if best_adaptive_name:
        bc = consistency[best_adaptive_name]
        if bc["verdict"] == "A":
            md.append(f"**ADOPT** adaptive candidate `{best_adaptive_name}`")
        elif bc["verdict"] == "B":
            md.append(f"**CONDITIONAL** adaptive candidate `{best_adaptive_name}` (부분 유효)")
        else:
            md.append(f"**NO CANDIDATE** – adaptive 개선 불충분")
    else:
        md.append("**NO CANDIDATE**")
    md.append("")

    md.append("## 12. 다음 액션")
    md.append("")
    if best_adaptive_name and consistency[best_adaptive_name]["verdict"] in ("A", "B"):
        md.append(f"- `{best_adaptive_name}` 모드를 shadow/paper 테스트 고려")
        md.append("- 추가 기간/데이터로 OOS 검증")
    else:
        md.append("- 현재 adaptive 후보 없음; 다른 접근 검토")
    md.append("")

    md_path.write_text("\n".join(md), encoding="utf-8")

    # ── console ──
    print("=" * 60)
    print("[ADAPTIVE POSITIVE FILTER SUMMARY]")
    print("=" * 60)
    print()
    print("baseline:")
    for p in lookback_list:
        r = next((r for r in all_rows if r["case_name"] == "baseline" and r["period"] == p), None)
        if r:
            print(f"  {p}d: return={r['total_return']}, trades={r['total_trades']}, MDD={r['max_drawdown']}")
    print()

    print("static positive:")
    improved_st_count = 0
    static_imps = []
    for p in lookback_list:
        r = next((r for r in all_rows if r["case_name"] == "static_positive" and r["period"] == p), None)
        if r:
            imp = float(r["improvement_vs_baseline"])
            static_imps.append(imp)
            if imp > 1e-9:
                improved_st_count += 1
            print(f"  {p}d: return={r['total_return']}, imp={imp:.6f}")
    print(f"  improved_periods: {improved_st_count}/{len(lookback_list)}")
    if static_imps:
        print(f"  avg_improvement: {np.mean(static_imps):.6f}")
        print(f"  worst_improvement: {min(static_imps):.6f}")
    print()

    if best_adaptive_name:
        bc = consistency[best_adaptive_name]
        print(f"best adaptive: {best_adaptive_name}")
        for p in lookback_list:
            r = next(
                (r for r in all_rows if r["case_name"] == best_adaptive_name and r["period"] == p),
                None,
            )
            if r:
                print(
                    f"  {p}d: return={r['total_return']}, imp_bl={r['improvement_vs_baseline']:.6f}, "
                    f"imp_st={r['improvement_vs_static']:.6f}"
                )
        print(f"  improved_periods_vs_baseline: {bc['improvement_ratio_vs_baseline']}")
        print(f"  improved_periods_vs_static: {bc['improved_periods_vs_static']}")
        print(f"  avg_improvement_vs_baseline: {bc['avg_improvement_vs_baseline']}")
        print(f"  worst_improvement_vs_baseline: {bc['worst_improvement_vs_baseline']}")
        print(f"  avg_MDD_diff: {bc['avg_MDD_diff']}")
        print(f"  verdict: {bc['verdict']}")
    print()

    if best_adaptive_name and consistency[best_adaptive_name]["verdict"] in ("A", "B"):
        print(f"final recommendation: adopt adaptive candidate ({best_adaptive_name})")
    elif best_adaptive_name:
        print("final recommendation: no candidate (insufficient improvement)")
    else:
        print("final recommendation: no candidate")
    print()
    print(f"created:\n- {csv_path}\n- {md_path}")


if __name__ == "__main__":
    main()
