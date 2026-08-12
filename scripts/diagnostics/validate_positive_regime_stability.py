"""
Positive regime 필터 다중 기간 안정성 + trade set 요약.

실행:
  python -m scripts.diagnostics.validate_positive_regime_stability --lookback-list 14,30,60,90
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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


def _entry_indices_from_trade_events(events: Sequence[Mapping[str, Any]], n_trades: int) -> List[int]:
    idxs: List[int] = []
    for e in events:
        ev = str(e.get("event", ""))
        if ev.startswith("EXIT") or ev == "FLIP":
            exit_idx = int(e["idx"])
            hb_raw = e.get("holding_bars")
            if hb_raw is None:
                hb_raw = e.get("bars_held")
            hb = int(hb_raw) if hb_raw is not None else 0
            idxs.append(exit_idx - hb)
    if len(idxs) != n_trades:
        idxs_alt = [
            int(e["idx"])
            for e in events
            if str(e.get("event", "")).startswith("ENTRY_") and e.get("idx") is not None
        ]
        if len(idxs_alt) == n_trades:
            return idxs_alt
        raise RuntimeError("trade_events vs trades length mismatch")
    return idxs


def _profit_map_by_entry_idx(entry_idxs: Sequence[int], trades: List[Mapping[str, Any]]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for i in range(min(len(entry_idxs), len(trades))):
        out[int(entry_idxs[i])] = (
            float(trades[i]["profit"]) if trades[i].get("profit") is not None else 0.0
        )
    return out


def _trade_subset_stats(profits: List[float]) -> Tuple[int, Optional[float], float]:
    n = len(profits)
    if n == 0:
        return 0, None, 0.0
    wr = sum(1 for p in profits if p > 0) / n
    return n, wr, float(sum(profits))


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


def _trade_set_metrics(res_bl: Mapping[str, Any], res_f: Mapping[str, Any]) -> Dict[str, Any]:
    tr_bl = list(_get(res_bl, "trades") or [])
    tr_f = list(_get(res_f, "trades") or [])
    ev_bl = list(_get(res_bl, "trade_events") or [])
    ev_f = list(_get(res_f, "trade_events") or [])
    ei_bl = _entry_indices_from_trade_events(ev_bl, len(tr_bl))
    ei_f = _entry_indices_from_trade_events(ev_f, len(tr_f))
    map_bl = _profit_map_by_entry_idx(ei_bl, tr_bl)
    map_f = _profit_map_by_entry_idx(ei_f, tr_f)
    set_bl = set(map_bl.keys())
    set_f = set(map_f.keys())
    common = set_bl & set_f
    removed = set_bl - set_f
    new_only = set_f - set_bl

    def agg(keys: set[int], dmap: Dict[int, float]) -> Tuple[int, Optional[float], float]:
        ps = [dmap[k] for k in sorted(keys)]
        return _trade_subset_stats(ps)

    c_n, c_wr, c_tp = agg(common, map_bl)
    r_n, r_wr, r_tp = agg(removed, map_bl)
    n_n, n_wr, n_tp = agg(new_only, map_f)
    return {
        "common_count": len(common),
        "removed_count": r_n,
        "removed_win_rate": r_wr,
        "removed_total_profit": r_tp,
        "new_count": n_n,
        "new_win_rate": n_wr,
        "new_total_profit": n_tp,
        "common_win_rate": c_wr,
        "common_total_profit_bl": c_tp,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-list", type=str, default="14,30,60,90")
    parser.add_argument("--cooldown-bars", type=int, default=12)
    parser.add_argument("--max-holding-bars", type=int, default=12)
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--modes", type=str, default="")
    args = parser.parse_args()

    lookbacks = [int(x.strip()) for x in args.lookback_list.split(",") if x.strip()]
    modes = (
        [m.strip() for m in args.modes.split(",") if m.strip()]
        if args.modes.strip()
        else list(DEFAULT_MODES)
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from scripts.diagnostics.analyze_regime_combinations import mask_last_days
    from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine
    from src.strategy_filters.regime_ablation import POSITIVE_REGIME_FILTER_MODES

    for m in modes:
        if m not in POSITIVE_REGIME_FILTER_MODES:
            raise SystemExit(f"unknown mode {m!r}")

    engine = get_ml_backtest_engine("ml_lstm_attn", "BTCUSDT", "5m")
    proba_long, proba_short, df_full = engine.load_predictions()

    common_bt = dict(
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

    detail_rows: List[Dict[str, Any]] = []
    by_mode_improvements: Dict[str, List[float]] = defaultdict(list)
    by_mode_trades: Dict[str, List[int]] = defaultdict(list)
    by_mode_wr_diff: Dict[str, List[float]] = defaultdict(list)

    for days in lookbacks:
        mask = mask_last_days(df_full, days)
        if int(mask.sum()) <= 0:
            continue
        kw_mask = {**common_bt, "index_mask": mask}

        res_bl = engine.run_backtest(
            **kw_mask,
            enable_positive_regime_filter=False,
            positive_regime_filter_mode="none",
        )

        for m in modes:
            res_f = engine.run_backtest(
                **kw_mask,
                enable_positive_regime_filter=True,
                positive_regime_filter_mode=m,
            )
            trades = list(_get(res_f, "trades") or [])
            bl_ret = float(_get(res_bl, "total_return") or 0)
            fr_ret = float(_get(res_f, "total_return") or 0)
            imp = fr_ret - bl_ret
            bl_wr = float(_get(res_bl, "win_rate") or 0)
            fr_wr = float(_get(res_f, "win_rate") or 0)
            n_tr = int(_get(res_f, "total_trades") or 0)
            bl_ntr = int(_get(res_bl, "total_trades") or 0)

            by_mode_improvements[m].append(imp)
            by_mode_trades[m].append(n_tr)
            by_mode_wr_diff[m].append(fr_wr - bl_wr)

            ts_metrics = _trade_set_metrics(res_bl, res_f)

            detail_rows.append(
                {
                    "lookback_days": days,
                    "mode": m,
                    "total_return": _get(res_f, "total_return"),
                    "win_rate": _get(res_f, "win_rate"),
                    "total_trades": n_tr,
                    "avg_profit": _get(res_f, "avg_profit"),
                    "median_profit": _get(res_f, "median_profit"),
                    "max_drawdown": _get(res_f, "max_drawdown"),
                    "profit_factor": _profit_factor(trades),
                    "expectancy": _expectancy(trades),
                    "trades_per_day": float(n_tr) / float(days) if days > 0 else None,
                    "blocked_entries": _get(res_f, "entries_blocked_by_positive_regime"),
                    "allowed_entries": _get(res_f, "positive_regime_allowed_entries"),
                    "improvement_vs_baseline": imp,
                    "win_rate_diff": fr_wr - bl_wr,
                    "trade_diff": n_tr - bl_ntr,
                    "baseline_total_return": bl_ret,
                    "baseline_total_trades": bl_ntr,
                    "trade_reduction_ratio": (
                        float(1.0 - n_tr / bl_ntr) if bl_ntr > 0 else None
                    ),
                    "score": fr_ret - float(_get(res_f, "max_drawdown") or 0),
                    "insufficient_trades": n_tr < args.min_trades,
                    **{f"ts_{k}": v for k, v in ts_metrics.items()},
                }
            )

    consistency_rows: List[Dict[str, Any]] = []
    for m in modes:
        imps = by_mode_improvements[m]
        if not imps:
            continue
        improved = sum(1 for x in imps if x > 1e-9)
        verdict_m = "C"
        if sum(1 for t in by_mode_trades[m] if t >= args.min_trades) >= max(1, len(lookbacks) // 2):
            if improved >= 3 and sum(imps) / len(imps) > 0:
                verdict_m = "A"
            elif improved >= 2:
                verdict_m = "B"
            elif improved >= 1:
                verdict_m = "B"
            else:
                verdict_m = "C"
        else:
            verdict_m = "D"

        consistency_rows.append(
            {
                "mode": m,
                "periods_tested": len(imps),
                "improved_periods": improved,
                "improvement_ratio": improved / len(imps) if imps else 0.0,
                "avg_improvement": float(sum(imps) / len(imps)) if imps else None,
                "worst_improvement": float(min(imps)) if imps else None,
                "avg_trade_count": float(sum(by_mode_trades[m]) / len(by_mode_trades[m]))
                if by_mode_trades[m]
                else None,
                "avg_win_rate_diff": float(sum(by_mode_wr_diff[m]) / len(by_mode_wr_diff[m]))
                if by_mode_wr_diff[m]
                else None,
                "verdict": verdict_m,
            }
        )

    consistency_rows.sort(key=lambda x: float(x.get("avg_improvement") or -1e18), reverse=True)

    csv_detail = OUT_DIR / f"positive_regime_stability_{stamp}.csv"
    if consistency_rows:
        cons_flat = []
        for r in consistency_rows:
            cons_flat.append({"row_kind": "consistency_summary", **r})
        all_rows = detail_rows + cons_flat if detail_rows else cons_flat
    else:
        all_rows = detail_rows

    if all_rows:
        keys = sorted({k for row in all_rows for k in row.keys()})
        with csv_detail.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k) for k in keys})

    md_path = OUT_DIR / f"positive_regime_stability_{stamp}.md"
    top3 = consistency_rows[:3]
    if top3 and top3[0].get("verdict") in ("A", "B"):
        rec = f"{top3[0]['mode']} (verdict {top3[0]['verdict']})"
    elif top3:
        rec = f"{top3[0]['mode']} (weak; verdict {top3[0]['verdict']})"
    else:
        rec = "no candidate"
    md_lines = [
        "# Positive Regime Stability",
        "",
        "## Consistency (요약)",
        "",
        "| mode | periods | improved | ratio | avg_imp | worst_imp | avg_trades | verdict |",
        "|------|---------|----------|-------|---------|-----------|------------|---------|",
    ]
    for r in consistency_rows:
        md_lines.append(
            f"| {r['mode']} | {r['periods_tested']} | {r['improved_periods']} | "
            f"{r['improvement_ratio']:.2f} | {r['avg_improvement']} | {r['worst_improvement']} | "
            f"{r['avg_trade_count']} | {r['verdict']} |"
        )
    md_lines.extend(
        [
            "",
            f"## 추천: `{rec}`",
            "",
            "## Trade set 요약",
            "상세는 CSV의 ts_* 컬럼 (removed/new/common).",
            "",
        ]
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("[POSITIVE REGIME STABILITY SUMMARY]")
    print("")
    print("top modes:")
    for r in top3:
        print(
            f"- {r['mode']}: improved={r['improved_periods']}/{r['periods_tested']} "
            f"avg_imp={r['avg_improvement']} worst={r['worst_improvement']} "
            f"avg_trades={r['avg_trade_count']} verdict={r['verdict']}"
        )
    print("")
    print(f"final recommendation: {rec}")
    print("")
    print(f"detail CSV: {csv_detail}")
    print(f"report MD: {md_path}")


if __name__ == "__main__":
    main()
