"""
레짐 조합 필터(F1–F5) 엔진 검증 vs baseline.

실행:
  python -m scripts.diagnostics.validate_regime_combination_filter

분석 CSV는 동일 실행에서 `run_combo_trade_aggregate`로 함께 저장한다.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "diagnostics" / "regime"


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
        raise RuntimeError(
            f"trade_events와 trades 불일치: exit_pairs={len(idxs)}, ENTRY={len(idxs_alt)}, trades={n_trades}"
        )
    return idxs


def _profit_map_by_entry_idx(entry_idxs: Sequence[int], trades: List[Mapping[str, Any]]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for i, ei in enumerate(entry_idxs):
        out[int(ei)] = float(trades[i]["profit"]) if trades[i].get("profit") is not None else 0.0
    return out


def _trade_subset_stats(profits: List[float]) -> Tuple[int, Optional[float], float]:
    n = len(profits)
    if n == 0:
        return 0, None, 0.0
    wr = sum(1 for p in profits if p > 0) / n
    return n, wr, float(sum(profits))


def _verdict(best_row: Dict[str, Any], baseline: Dict[str, Any]) -> str:
    imp = float(best_row.get("improvement_vs_baseline") or 0)
    rem_wr = best_row.get("removed_win_rate")
    rem_n = int(best_row.get("removed_trade_count") or 0)
    meaningful = imp > 5e-6
    if (
        meaningful
        and float(best_row.get("total_return") or 0) > float(baseline.get("total_return") or 0)
        and rem_n > 0
        and rem_wr is not None
        and float(rem_wr) < 0.45
    ):
        return "A"
    if imp > 0 or float(best_row.get("win_rate") or 0) > float(baseline.get("win_rate") or 0):
        return "B"
    return "C"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--cooldown-bars", type=int, default=12)
    parser.add_argument("--max-holding-bars", type=int, default=12)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from scripts.diagnostics.analyze_regime_combinations import mask_last_days, run_combo_trade_aggregate
    from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine

    rows_agg, worst5, f5_keys = run_combo_trade_aggregate(
        args.lookback_days,
        cooldown_bars=args.cooldown_bars,
        max_holding_bars=args.max_holding_bars,
    )

    analysis_csv = OUT_DIR / f"regime_combination_analysis_{stamp}.csv"
    if rows_agg:
        with analysis_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_agg[0].keys()))
            w.writeheader()
            for r in rows_agg:
                w.writerow(r)

    engine = get_ml_backtest_engine("ml_lstm_attn", "BTCUSDT", "5m")
    proba_long, proba_short, df_full = engine.load_predictions()

    mask = mask_last_days(df_full, args.lookback_days)

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
        emit_trade_log=True,
        max_holding_bars=args.max_holding_bars,
        enable_profit_lock_exit=False,
    )

    runs: List[Tuple[str, Dict[str, Any]]] = [
        ("case0_baseline", dict(enable_regime_filter=False, regime_filter_mode="none")),
        (
            "case1_f1_sideways_high",
            dict(enable_regime_filter=True, regime_filter_mode="combo_f1_sideways_high_vol"),
        ),
        (
            "case2_f2_sideways_ema_above",
            dict(enable_regime_filter=True, regime_filter_mode="combo_f2_sideways_ema_above"),
        ),
        (
            "case3_f3_high_ema_above",
            dict(enable_regime_filter=True, regime_filter_mode="combo_f3_high_vol_ema_above"),
        ),
        (
            "case4_f4_triple",
            dict(enable_regime_filter=True, regime_filter_mode="combo_f4_sideways_high_vol_ema_above"),
        ),
        (
            "case5_f5_worst3",
            dict(
                enable_regime_filter=True,
                regime_filter_mode="combo_f5_worst_keys",
                regime_combo_block_keys=f5_keys,
            ),
        ),
    ]

    val_rows: List[Dict[str, Any]] = []
    res_bl: Dict[str, Any] | None = None

    for case_id, kw in runs:
        kw_full = {**common, **kw}
        res = engine.run_backtest(**kw_full)
        if case_id == "case0_baseline":
            res_bl = res
        trades = list(_get(res, "trades") or [])
        ev = list(_get(res, "trade_events") or [])
        n_tr = int(_get(res, "total_trades") or 0)
        skipped = int(_get(res, "entries_blocked_by_regime_ablation") or 0)

        removed_wr = None
        removed_tp = None
        removed_n = 0
        if res_bl is not None and case_id != "case0_baseline":
            tr_bl = list(_get(res_bl, "trades") or [])
            ev_bl = list(_get(res_bl, "trade_events") or [])
            ei_bl = _entry_indices_from_trade_events(ev_bl, len(tr_bl))
            ei_f = _entry_indices_from_trade_events(ev, len(trades))
            map_bl = _profit_map_by_entry_idx(ei_bl, tr_bl)
            set_bl = set(map_bl.keys())
            set_f = set(_profit_map_by_entry_idx(ei_f, trades).keys())
            removed_idx = set_bl - set_f
            removed_profits = [map_bl[i] for i in sorted(removed_idx)]
            removed_n, removed_wr, removed_tp = _trade_subset_stats(removed_profits)

        bl_ret = float(_get(res_bl, "total_return") or 0.0) if res_bl else 0.0
        bl_wr = float(_get(res_bl, "win_rate") or 0.0) if res_bl else 0.0
        bl_ntr = int(_get(res_bl, "total_trades") or 0) if res_bl else 0

        row = {
            "case_id": case_id,
            "regime_filter_mode": _get(res, "regime_filter_mode"),
            "regime_combo_block_keys": str(_get(res, "regime_combo_block_keys")),
            "total_return": _get(res, "total_return"),
            "win_rate": _get(res, "win_rate"),
            "total_trades": n_tr,
            "avg_profit": _get(res, "avg_profit"),
            "max_drawdown": _get(res, "max_drawdown"),
            "trades_per_day": float(n_tr) / float(args.lookback_days) if args.lookback_days > 0 else None,
            "skipped_trades": skipped,
            "improvement_vs_baseline": float(_get(res, "total_return") or 0.0) - bl_ret,
            "win_rate_diff": float(_get(res, "win_rate") or 0.0) - bl_wr,
            "trade_diff": n_tr - bl_ntr,
            "removed_trade_count": removed_n,
            "removed_win_rate": removed_wr,
            "removed_total_profit": removed_tp,
        }
        val_rows.append(row)

    baseline_row = next(r for r in val_rows if r["case_id"] == "case0_baseline")
    non_base = [r for r in val_rows if r["case_id"] != "case0_baseline"]
    best = max(non_base, key=lambda x: float(x.get("total_return") or -1e18)) if non_base else None
    verdict = _verdict(best, baseline_row) if best else "C"

    val_csv = OUT_DIR / f"regime_combination_filter_validation_{stamp}.csv"
    if val_rows:
        with val_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(val_rows[0].keys()))
            w.writeheader()
            for r in val_rows:
                w.writerow(r)

    sorted_filters = sorted(
        non_base,
        key=lambda x: float(x.get("total_return") or -1e18),
        reverse=True,
    )
    top3 = sorted_filters[:3]

    md_path = OUT_DIR / f"regime_combination_report_{stamp}.md"
    combo_tbl = [
        "| regime_key | trades | win_rate | total_profit | mean_profit | contribution | loss_share |",
        "|------------|--------|----------|--------------|-------------|--------------|------------|",
    ]
    for r in rows_agg:
        combo_tbl.append(
            f"| {r['regime_key']} | {r['trade_count']} | {r['win_rate']} | "
            f"{r['total_profit']} | {r['mean_profit']} | {r['contribution_to_total']} | "
            f"{r['loss_contribution']} |"
        )

    lines = [
        "# Regime Combination Report",
        "",
        "## 1. 목적",
        "",
        "18개 레짐 조합별 손익 분해 후, 정의된 조합 필터(F1–F5)의 엔진 성과를 baseline과 비교한다.",
        "",
        "## 2. 설정",
        "",
        f"- lookback_days: {args.lookback_days}",
        f"- cooldown_bars: {args.cooldown_bars}",
        f"- F5 block keys: `{f5_keys}`",
        "",
        "## 3. 조합별 성과 테이블",
        "",
        *combo_tbl,
        "",
        "## 4. Worst regimes (total_profit 오름차순 상위 5)",
        "",
        *[f"- `{k}`" for k in worst5],
        "",
        "## 5. 필터별 결과",
        "",
        "| case | mode | return | win_rate | trades | MDD | improvement |",
        "|------|------|--------|----------|--------|-----|-------------|",
    ]
    for r in val_rows:
        lines.append(
            f"| {r['case_id']} | {r.get('regime_filter_mode')} | "
            f"{r.get('total_return')} | {r.get('win_rate')} | {r.get('total_trades')} | "
            f"{r.get('max_drawdown')} | {r.get('improvement_vs_baseline')} |"
        )
    lines.extend(
        [
            "",
            "## 6. 제거 trade (baseline 대비)",
            "",
            "| case | removed_n | removed_wr | removed_profit |",
            "|------|-----------|------------|----------------|",
        ]
    )
    for r in val_rows:
        if r["case_id"] == "case0_baseline":
            continue
        lines.append(
            f"| {r['case_id']} | {r.get('removed_trade_count')} | "
            f"{r.get('removed_win_rate')} | {r.get('removed_total_profit')} |"
        )
    lines.extend(
        [
            "",
            "## 7. Top 3 filters (total_return)",
            "",
            *[f"{i+1}. `{r['case_id']}` return={r.get('total_return')}" for i, r in enumerate(top3)],
            "",
            f"## 8. 판정: **{verdict}**",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("[REGIME COMBINATION SUMMARY]")
    print("")
    print("worst regimes:")
    for k in worst5:
        print(f"  - {k}")
    print("")
    if best:
        print("best filter:")
        print(f"  name: {best['case_id']} ({best.get('regime_filter_mode')})")
        print(f"  return: {best.get('total_return')}")
        print(f"  win_rate: {best.get('win_rate')}")
        print(f"  trades: {best.get('total_trades')}")
    print("")
    print(f"final verdict: {verdict}")
    print("")
    print(f"analysis CSV: {analysis_csv}")
    print(f"validation CSV: {val_csv}")
    print(f"report MD: {md_path}")


if __name__ == "__main__":
    main()
