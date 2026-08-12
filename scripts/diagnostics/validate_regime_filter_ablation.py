"""
Regime filter ablation: baseline vs F1–F7 (엔진 기준, profit lock OFF).

실행:
  python -m scripts.diagnostics.validate_regime_filter_ablation --lookback-days 14
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "diagnostics" / "regime"


def _mask_last_days(df: pd.DataFrame, lookback_days: int) -> np.ndarray:
    ts = pd.to_datetime(df["timestamp"])
    end = ts.max()
    start = end - pd.Timedelta(days=int(lookback_days))
    return (ts >= start).values


def _get(result: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


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


def _verdict(
    baseline: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> str:
    """A / B / C 스타일 단순 판정."""
    bl_ret = float(baseline.get("total_return") or 0.0)
    bl_trades = int(baseline.get("total_trades") or 0)
    candidates = [r for r in rows if r["case_id"] != "case0_baseline"]
    if not candidates:
        return "C"

    improved = [
        r
        for r in candidates
        if float(r.get("total_return") or 0) > bl_ret
        and int(r.get("total_trades") or 0) >= max(1, bl_trades // 10)
    ]
    if improved:
        best = max(improved, key=lambda x: float(x.get("total_return") or -1e18))
        if float(best.get("total_return") or 0) > 0:
            return "A"
        return "B"
    return "C"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--cooldown-bars", type=int, default=12)
    parser.add_argument("--max-holding-bars", type=int, default=12)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine

    engine = get_ml_backtest_engine("ml_lstm_attn", "BTCUSDT", "5m")
    proba_long, proba_short, df = engine.load_predictions()
    mask = _mask_last_days(df, args.lookback_days)
    if int(mask.sum()) <= 0:
        raise RuntimeError("빈 mask")

    common: Dict[str, Any] = dict(
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
        df_with_proba=df,
        index_mask=mask,
        emit_trade_log=False,
        max_holding_bars=args.max_holding_bars,
        enable_profit_lock_exit=False,
    )

    runs: List[tuple[str, Dict[str, Any]]] = [
        ("case0_baseline", dict(enable_regime_filter=False, regime_filter_mode="none")),
        ("case1_no_sideways", dict(enable_regime_filter=True, regime_filter_mode="no_sideways")),
        ("case2_no_high_vol", dict(enable_regime_filter=True, regime_filter_mode="no_high_vol")),
        ("case3_only_ema_below", dict(enable_regime_filter=True, regime_filter_mode="only_ema_below")),
        (
            "case4_no_sideways_high_vol",
            dict(enable_regime_filter=True, regime_filter_mode="no_sideways_high_vol"),
        ),
        (
            "case5_no_sideways_ema_above",
            dict(enable_regime_filter=True, regime_filter_mode="no_sideways_ema_above"),
        ),
        (
            "case6_no_high_vol_ema_above",
            dict(enable_regime_filter=True, regime_filter_mode="no_high_vol_ema_above"),
        ),
        ("case7_strict_all", dict(enable_regime_filter=True, regime_filter_mode="strict_all")),
    ]

    rows: List[Dict[str, Any]] = []
    for case_id, kw in runs:
        kw_full = {**common, **kw}
        res = engine.run_backtest(**kw_full)
        trades = list(_get(res, "trades") or [])
        n_tr = int(_get(res, "total_trades") or 0)
        skipped = int(_get(res, "entries_blocked_by_regime_ablation") or 0)
        row: Dict[str, Any] = {
            "case_id": case_id,
            "regime_filter_mode": _get(res, "regime_filter_mode"),
            "total_return": _get(res, "total_return"),
            "win_rate": _get(res, "win_rate"),
            "total_trades": n_tr,
            "avg_profit": _get(res, "avg_profit"),
            "median_profit": _get(res, "median_profit"),
            "max_drawdown": _get(res, "max_drawdown"),
            "profit_factor": _profit_factor(trades),
            "expectancy": _expectancy(trades),
            "trades_per_day": (
                float(n_tr) / float(args.lookback_days) if args.lookback_days > 0 else None
            ),
            "skipped_trade_count": skipped,
            "kept_trade_count": n_tr,
            "entries_blocked_by_regime_ablation": skipped,
        }
        rows.append(row)

    baseline_row = next(r for r in rows if r["case_id"] == "case0_baseline")
    bl_trades_raw = int(baseline_row.get("total_trades") or 0)
    bl_ret = float(baseline_row.get("total_return") or 0.0)

    for r in rows:
        kt = int(r.get("kept_trade_count") or 0)
        r["trade_reduction_ratio"] = (
            float(1.0 - kt / bl_trades_raw) if bl_trades_raw > 0 else None
        )
        r["return_vs_baseline"] = (
            float(r.get("total_return") or 0.0) - bl_ret if r["case_id"] != "case0_baseline" else 0.0
        )

    fieldnames = list(rows[0].keys())
    csv_path = OUT_DIR / f"regime_filter_ablation_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    non_baseline = [r for r in rows if r["case_id"] != "case0_baseline"]
    sorted_by_ret = sorted(
        non_baseline,
        key=lambda x: float(x.get("total_return") or -1e18),
        reverse=True,
    )
    top3 = sorted_by_ret[:3]
    best = sorted_by_ret[0] if sorted_by_ret else None

    verdict = _verdict(baseline_row, rows)

    md_lines = [
        "# Regime Filter Ablation",
        "",
        "## 1. 목적",
        "",
        "진입 전 레짐 필터(F1–F7)만 추가했을 때 엔진 기준 성과 변화를 비교한다. "
        "profit lock OFF, 전략·exit·Guard·Stage2는 변경하지 않는다.",
        "",
        "## 2. Baseline 결과",
        "",
        f"- total_return: {baseline_row.get('total_return')}",
        f"- win_rate: {baseline_row.get('win_rate')}",
        f"- total_trades: {baseline_row.get('total_trades')}",
        f"- max_drawdown: {baseline_row.get('max_drawdown')}",
        "",
        "## 3. 필터별 결과",
        "",
        "| case | mode | total_return | win_rate | trades | MDD | skipped | trade_reduction |",
        "|------|------|-------------|----------|--------|-----|---------|-----------------|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['case_id']} | {r.get('regime_filter_mode')} | "
            f"{r.get('total_return')} | {r.get('win_rate')} | {r.get('total_trades')} | "
            f"{r.get('max_drawdown')} | {r.get('skipped_trade_count')} | "
            f"{r.get('trade_reduction_ratio')} |"
        )
    md_lines.extend(
        [
            "",
            "## 4. Trade 감소 vs 성과",
            "",
            "trade_reduction_ratio는 baseline 대비 실행 트레이드 비율 감소(1 - kept/baseline_trades).",
            "",
            "## 5. Top 3 필터 (total_return 기준)",
            "",
        ]
    )
    for i, r in enumerate(top3, 1):
        md_lines.append(
            f"{i}. `{r['case_id']}` mode=`{r.get('regime_filter_mode')}` "
            f"return={r.get('total_return')} win_rate={r.get('win_rate')} "
            f"trades={r.get('total_trades')} reduction={r.get('trade_reduction_ratio')}"
        )
    md_lines.extend(
        [
            "",
            "## 6. 탈락 필터 후보",
            "",
            "total_return이 baseline 이하이거나 trade가 과도하게 줄어든 경우 후속 검토에서 제외 후보.",
            "",
            "## 7. 최종 판정",
            "",
            f"- Verdict: **{verdict}** (A=유효, B=부분·여전히 음수 등, C=무효/악화)",
            "",
        ]
    )

    md_path = OUT_DIR / f"regime_filter_ablation_{stamp}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    best_imp = (
        float(best.get("total_return") or 0.0) - bl_ret
        if best is not None
        else 0.0
    )
    print("[REGIME FILTER SUMMARY]")
    print("")
    print("baseline:")
    print(f"- return: {baseline_row.get('total_return')}")
    print(f"- win_rate: {baseline_row.get('win_rate')}")
    print(f"- trades: {baseline_row.get('total_trades')}")
    print("")
    if best is not None:
        print("best filter:")
        print(f"- name: {best.get('case_id')} ({best.get('regime_filter_mode')})")
        print(f"- return: {best.get('total_return')}")
        print(f"- win_rate: {best.get('win_rate')}")
        print(f"- trades: {best.get('total_trades')}")
        print(f"- improvement vs baseline (return): {best_imp:.6f}")
    else:
        print("best filter: (none)")
    print("")
    print(f"final verdict: {verdict}")
    print("")
    print(f"CSV: {csv_path}")
    print(f"MD:   {md_path}")


if __name__ == "__main__":
    main()
