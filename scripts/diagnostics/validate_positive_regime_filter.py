"""
Positive regime filter (allow-list) 단일 기간 엔진 검증 vs baseline.

실행:
  python -m scripts.diagnostics.validate_positive_regime_filter --lookback-days 90
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "diagnostics" / "regime"

DEFAULT_MODES = [
    "allow_up_only",
    "allow_up_mid_high_vol",
    "allow_up_mid_vol_only",
    "allow_up_above_mid_high",
    "allow_non_sideways_mid_high",
    "allow_down_or_up_mid_high",
    "allow_ema_below_or_strong_up",
    "allow_strict_quality",
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


def _verdict_single(rows: List[Dict[str, Any]], min_trades: int) -> str:
    eligible = [r for r in rows if r.get("case_name") != "baseline" and int(r.get("total_trades") or 0) >= min_trades]
    if not eligible:
        return "D"
    best = max(eligible, key=lambda x: float(x.get("total_return") or -1e18))
    bl = next(r for r in rows if r["case_name"] == "baseline")
    imp = float(best.get("improvement_vs_baseline") or 0)
    if int(best.get("total_trades") or 0) < min_trades:
        return "D"
    if imp > 1e-6 and float(best.get("avg_profit") or 0) >= float(bl.get("avg_profit") or 0):
        return "A"
    if imp > 0:
        return "B"
    return "C"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--cooldown-bars", type=int, default=12)
    parser.add_argument("--max-holding-bars", type=int, default=12)
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument(
        "--modes",
        type=str,
        default="",
        help="comma-separated modes; empty = all P1–P8",
    )
    args = parser.parse_args()

    modes = (
        [m.strip() for m in args.modes.split(",") if m.strip()]
        if args.modes.strip()
        else list(DEFAULT_MODES)
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine
    from src.strategy_filters.regime_ablation import POSITIVE_REGIME_FILTER_MODES

    for m in modes:
        if m not in POSITIVE_REGIME_FILTER_MODES:
            raise SystemExit(f"unknown mode {m!r}; allowed {sorted(POSITIVE_REGIME_FILTER_MODES)}")

    engine = get_ml_backtest_engine("ml_lstm_attn", "BTCUSDT", "5m")
    proba_long, proba_short, df_full = engine.load_predictions()
    mask = _mask_last_days(df_full, args.lookback_days)
    if int(mask.sum()) <= 0:
        raise RuntimeError("empty mask")

    common = dict(
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
        index_mask=mask,
        emit_trade_log=False,
        max_holding_bars=args.max_holding_bars,
        enable_profit_lock_exit=False,
        enable_regime_filter=False,
        regime_filter_mode="none",
    )

    rows: List[Dict[str, Any]] = []

    def run_case(case_name: str, kw_extra: Dict[str, Any]) -> None:
        res = engine.run_backtest(**{**common, **kw_extra})
        trades = list(_get(res, "trades") or [])
        n_tr = int(_get(res, "total_trades") or 0)
        tr_ret = float(_get(res, "total_return") or 0.0)
        mdd = float(_get(res, "max_drawdown") or 0.0)
        rows.append(
            {
                "case_name": case_name,
                "mode": kw_extra.get("positive_regime_filter_mode", "none"),
                "lookback_days": args.lookback_days,
                "total_return": _get(res, "total_return"),
                "win_rate": _get(res, "win_rate"),
                "total_trades": n_tr,
                "avg_profit": _get(res, "avg_profit"),
                "median_profit": _get(res, "median_profit"),
                "max_drawdown": _get(res, "max_drawdown"),
                "profit_factor": _profit_factor(trades),
                "expectancy": _expectancy(trades),
                "trades_per_day": float(n_tr) / float(args.lookback_days) if args.lookback_days > 0 else None,
                "blocked_entries": _get(res, "entries_blocked_by_positive_regime"),
                "allowed_entries": _get(res, "positive_regime_allowed_entries"),
                "score": tr_ret - mdd,
                "insufficient_trades": n_tr < args.min_trades,
            }
        )

    run_case(
        "baseline",
        dict(enable_positive_regime_filter=False, positive_regime_filter_mode="none"),
    )

    for m in modes:
        run_case(
            m,
            dict(enable_positive_regime_filter=True, positive_regime_filter_mode=m),
        )

    bl_ret = float(rows[0]["total_return"])
    bl_avg = float(rows[0]["avg_profit"] or 0)
    bl_ntr = int(rows[0]["total_trades"] or 0)
    for r in rows[1:]:
        r["improvement_vs_baseline"] = float(r["total_return"] or 0) - bl_ret
        r["avg_profit_diff_vs_baseline"] = float(r["avg_profit"] or 0) - bl_avg
        r["trade_reduction_ratio"] = (
            float(1.0 - int(r["total_trades"] or 0) / bl_ntr) if bl_ntr > 0 else None
        )
    rows[0]["improvement_vs_baseline"] = 0.0
    rows[0]["trade_reduction_ratio"] = 0.0
    rows[0]["avg_profit_diff_vs_baseline"] = 0.0

    non_bl = [r for r in rows if r["case_name"] != "baseline"]
    best = max(non_bl, key=lambda x: float(x.get("total_return") or -1e18)) if non_bl else None
    top3 = sorted(non_bl, key=lambda x: float(x.get("total_return") or -1e18), reverse=True)[:3]
    verdict = _verdict_single(rows, args.min_trades)

    csv_path = OUT_DIR / f"positive_regime_filter_{stamp}.csv"
    if rows:
        keys = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    md_path = OUT_DIR / f"positive_regime_filter_{stamp}.md"
    md_lines = [
        "# Positive Regime Filter (single window)",
        "",
        "## 목적",
        "허용 레짐만 통과시켰을 때 baseline 대비 성과 확인.",
        "",
        "## 설정",
        f"- lookback_days={args.lookback_days}, cooldown={args.cooldown_bars}, max_hold={args.max_holding_bars}",
        f"- min_trades_threshold={args.min_trades}",
        "",
        "## Baseline",
        "",
        f"- return={rows[0]['total_return']}, win_rate={rows[0]['win_rate']}, trades={rows[0]['total_trades']}, MDD={rows[0]['max_drawdown']}",
        "",
        "## 결과",
        "",
        "| mode | return | trades | win_rate | MDD | improvement | insufficient |",
        "|------|--------|--------|----------|-----|-------------|--------------|",
    ]
    for r in rows[1:]:
        md_lines.append(
            f"| {r['mode']} | {r['total_return']} | {r['total_trades']} | {r['win_rate']} | "
            f"{r['max_drawdown']} | {r['improvement_vs_baseline']} | {r['insufficient_trades']} |"
        )
    md_lines.extend(
        [
            "",
            "## Top 3 (total_return)",
            "",
            *[f"{i+1}. `{r['mode']}` return={r['total_return']}" for i, r in enumerate(top3)],
            "",
            "## 탈락 후보",
            "insufficient_trades=True 또는 return이 baseline 이하인 모드.",
            "",
            "## 판정",
            "",
            f"**{verdict}** (A/B/C/D)",
            "",
        ]
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("[POSITIVE REGIME FILTER SUMMARY]")
    print("")
    print("baseline:")
    print(f"- return: {rows[0]['total_return']}")
    print(f"- win_rate: {rows[0]['win_rate']}")
    print(f"- trades: {rows[0]['total_trades']}")
    print(f"- MDD: {rows[0]['max_drawdown']}")
    print("")
    if best:
        print("best mode:")
        print(f"- mode: {best['mode']}")
        print(f"- return: {best['total_return']}")
        print(f"- win_rate: {best['win_rate']}")
        print(f"- trades: {best['total_trades']}")
        print(f"- MDD: {best['max_drawdown']}")
        print(f"- improvement: {best['improvement_vs_baseline']}")
    print("")
    print(f"verdict: {verdict}")
    print("")
    print(f"created:\n- {csv_path}\n- {md_path}")


if __name__ == "__main__":
    main()
