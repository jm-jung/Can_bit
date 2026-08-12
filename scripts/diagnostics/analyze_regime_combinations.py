"""
레짐 튜플 (ema_side × trend × vol_bucket) 조합별 트레이드 성과 분해.

실행:
  python -m scripts.diagnostics.analyze_regime_combinations

  python -m scripts.diagnostics.analyze_regime_combinations --lookback-days 90
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "diagnostics" / "regime"


def mask_last_days(df: pd.DataFrame, lookback_days: int) -> np.ndarray:
    ts = pd.to_datetime(df["timestamp"])
    end = ts.max()
    start = end - pd.Timedelta(days=int(lookback_days))
    return (ts >= start).values


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


def run_combo_trade_aggregate(
    lookback_days: int,
    *,
    cooldown_bars: int = 12,
    max_holding_bars: int = 12,
) -> Tuple[List[Dict[str, Any]], Tuple[str, ...], Tuple[str, ...]]:
    """
    baseline 백테스트 1회 + 진입 바 레짐 키별 집계.

    Returns:
        rows (조합별 통계, total_profit 오름차순 정렬),
        worst5 regime_key 튜플,
        f5 후보 (trade_count>=min_trades, total_profit 최악 순 상위 3)
    """
    from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine
    from src.strategy_filters.regime_ablation import compute_regime_components

    engine = get_ml_backtest_engine("ml_lstm_attn", "BTCUSDT", "5m")
    proba_long, proba_short, df_full = engine.load_predictions()
    mask = mask_last_days(df_full, lookback_days)
    if int(mask.sum()) <= 0:
        raise RuntimeError("빈 mask")

    df_slice = df_full.loc[mask].reset_index(drop=True)
    *_, regime_key_arr = compute_regime_components(df_slice)

    common = dict(
        long_threshold=None,
        short_threshold=None,
        use_optimized_threshold=True,
        long_only=True,
        signal_confirmation_bars=1,
        min_hold_bars=12,
        cooldown_bars=cooldown_bars,
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
        max_holding_bars=max_holding_bars,
        enable_profit_lock_exit=False,
        enable_regime_filter=False,
        regime_filter_mode="none",
    )

    res = engine.run_backtest(**common)
    trades = list(_get(res, "trades") or [])
    events = list(_get(res, "trade_events") or [])
    entry_idxs = _entry_indices_from_trade_events(events, len(trades))

    by_key: dict[str, List[float]] = defaultdict(list)
    for ei, tr in zip(entry_idxs, trades):
        ei = int(ei)
        if ei < 0 or ei >= len(regime_key_arr):
            continue
        k = str(regime_key_arr[ei])
        p = float(tr["profit"]) if tr.get("profit") is not None else 0.0
        by_key[k].append(p)

    sum_total_profit = sum(sum(v) for v in by_key.values())
    neg_parts = [sum(v) for v in by_key.values() if sum(v) < 0]
    loss_pool = sum(abs(x) for x in neg_parts) if neg_parts else 0.0

    rows: List[Dict[str, Any]] = []
    for key, plist in sorted(by_key.items(), key=lambda kv: sum(kv[1])):
        tp = float(sum(plist))
        tc = len(plist)
        wins = sum(1 for p in plist if p > 0)
        wr = wins / tc if tc else None
        mean_p = tp / tc if tc else None
        contrib = (tp / sum_total_profit) if sum_total_profit != 0 else None
        if tp < 0 and loss_pool > 0:
            loss_ctrib = abs(tp) / loss_pool
        elif tp < 0:
            loss_ctrib = None
        else:
            loss_ctrib = 0.0

        rows.append(
            {
                "regime_key": key,
                "trade_count": tc,
                "win_rate": wr,
                "total_profit": tp,
                "mean_profit": mean_p,
                "contribution_to_total": contrib,
                "loss_contribution": loss_ctrib,
                "meets_min_sample_3": tc >= 3,
                "meets_min_sample_5": tc >= 5,
            }
        )

    worst5 = tuple(r["regime_key"] for r in rows[:5])

    min_trades_f5 = 3
    f5_eligible = [r for r in rows if int(r["trade_count"]) >= min_trades_f5]
    f5_eligible.sort(key=lambda x: float(x["total_profit"]))
    f5_keys = tuple(str(r["regime_key"]) for r in f5_eligible[:3])

    return rows, worst5, f5_keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--cooldown-bars", type=int, default=12)
    parser.add_argument("--max-holding-bars", type=int, default=12)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, worst5, f5_keys = run_combo_trade_aggregate(
        args.lookback_days,
        cooldown_bars=args.cooldown_bars,
        max_holding_bars=args.max_holding_bars,
    )

    csv_path = OUT_DIR / f"regime_combination_analysis_{stamp}.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    bad_a = [
        r
        for r in rows
        if r["meets_min_sample_3"]
        and (r["win_rate"] is not None and r["win_rate"] < 0.30)
        and r["total_profit"] < 0
    ]

    print("[REGIME COMBINATION ANALYSIS]")
    print(f"lookback_days={args.lookback_days} combos_with_trades={len(rows)}")
    print("worst 5 (total_profit asc):")
    for k in worst5:
        print(f"  - {k}")
    print(f"F5 worst-keys (min_trades>=3, top 3 loss): {f5_keys}")
    print(f"bad_sample (A: n>=3, wr<30%, profit<0): {len(bad_a)}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
