#!/usr/bin/env python3
"""
TCN 필터 파라미터 소규모 스윕: min_max_proba, max_entropy, min_hold, cooldown 조합으로
백테스트를 cost_on/cost_off 각 1회씩 실행하고, 상위 5개를 md/json로 저장.

사용법:
  source .venv/bin/activate
  python -m scripts.run_tcn_filter_sweep_quick --days 7 --symbol BTCUSDT --timeframe 5m --preset base
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DIAG_DIR = PROJECT_ROOT / "data" / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_filtered_cache(symbol: str, timeframe: str, preset: str, days: int):
    """Load proba cache for preset, filter to last `days` days. Returns (proba_long, proba_short, df) or None."""
    import pandas as pd
    from src.optimization.ml_proba_cache import _get_cache_path

    path = _get_cache_path(
        strategy_name="ml_tcn",
        symbol=symbol,
        timeframe=timeframe,
        feature_preset="base",
        tcn_preset=preset if preset != "base" else None,
    )
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        logger.warning("Cache not found for preset=%s: %s", preset, path)
        return None
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        return None
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is not None:
        from datetime import timezone
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    else:
        cutoff = datetime.utcnow() - timedelta(days=days)
    df = df[ts >= cutoff].copy()
    if df.empty:
        return None
    proba_long = df["proba_long"].values.astype("float32")
    proba_short = df["proba_short"].values.astype("float32")
    return proba_long, proba_short, df.reset_index(drop=True)


def _run_one(
    symbol: str,
    timeframe: str,
    preset: str,
    days: int,
    commission_rate: float,
    slippage_rate: float,
    min_max_proba: float | None,
    max_entropy: float | None,
    min_hold: int | None,
    cooldown: int | None,
    proba_long_arr,
    proba_short_arr,
    df_filtered,
):
    """Run a single backtest; return result dict or None."""
    from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine
    from src.strategies.ml_thresholds import resolve_ml_thresholds

    engine = get_ml_backtest_engine(
        strategy_name="ml_tcn",
        symbol=symbol,
        timeframe=timeframe,
        feature_preset="base",
        tcn_preset=preset,
    )
    long_th, short_th = resolve_ml_thresholds(
        strategy_name="ml_tcn",
        symbol=symbol,
        timeframe=timeframe,
        use_optimized_thresholds=True,
    )
    short_th = short_th if short_th is not None else 0.5
    try:
        result = engine.run_backtest(
            long_threshold=long_th,
            short_threshold=short_th,
            use_optimized_threshold=True,
            use_stage2=True,
            use_strategy_guard_v2=True,
            proba_long_cache=proba_long_arr,
            proba_short_cache=proba_short_arr,
            df_with_proba=df_filtered,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            min_max_proba=min_max_proba,
            max_entropy=max_entropy,
            min_hold_bars_override=min_hold,
            cooldown_bars_override=cooldown,
        )
    except Exception as e:
        logger.warning("Backtest failed: %s", e)
        return None
    return result


def main():
    parser = argparse.ArgumentParser(description="TCN filter parameter quick sweep")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--preset", type=str, default="base")
    args = parser.parse_args()

    cached = _load_filtered_cache(args.symbol, args.timeframe, args.preset, args.days)
    if cached is None:
        logger.error("Could not load cache for preset=%s", args.preset)
        return 1
    proba_long_arr, proba_short_arr, df_filtered = cached
    n_bars = len(df_filtered)
    logger.info("Sweep: preset=%s, bars=%s", args.preset, n_bars)

    min_max_proba_list = [0.45, 0.50, 0.55]
    max_entropy_list = [1.60, 1.55, 1.50]
    min_hold_list = [12, 24]
    cooldown_list = [6, 12]

    results = []
    total_runs = (
        len(min_max_proba_list)
        * len(max_entropy_list)
        * len(min_hold_list)
        * len(cooldown_list)
    )
    run_idx = 0
    for min_max_proba in min_max_proba_list:
        for max_entropy in max_entropy_list:
            for min_hold in min_hold_list:
                for cooldown in cooldown_list:
                    run_idx += 1
                    logger.info(
                        "[%s/%s] min_max_proba=%s max_entropy=%s min_hold=%s cooldown=%s",
                        run_idx,
                        total_runs,
                        min_max_proba,
                        max_entropy,
                        min_hold,
                        cooldown,
                    )
                    res_on = _run_one(
                        args.symbol,
                        args.timeframe,
                        args.preset,
                        args.days,
                        commission_rate=0.0009,
                        slippage_rate=0.0001,
                        min_max_proba=min_max_proba,
                        max_entropy=max_entropy,
                        min_hold=min_hold,
                        cooldown=cooldown,
                        proba_long_arr=proba_long_arr,
                        proba_short_arr=proba_short_arr,
                        df_filtered=df_filtered,
                    )
                    res_off = _run_one(
                        args.symbol,
                        args.timeframe,
                        args.preset,
                        args.days,
                        commission_rate=0.0,
                        slippage_rate=0.0,
                        min_max_proba=min_max_proba,
                        max_entropy=max_entropy,
                        min_hold=min_hold,
                        cooldown=cooldown,
                        proba_long_arr=proba_long_arr,
                        proba_short_arr=proba_short_arr,
                        df_filtered=df_filtered,
                    )
                    if res_on is None:
                        res_on = {}
                    if res_off is None:
                        res_off = {}
                    trades = int(res_on.get("total_trades", 0))
                    cost_on_return = float(res_on.get("total_return", 0.0))
                    cost_off_return = float(res_off.get("total_return", 0.0))
                    cost_on_win_rate = float(res_on.get("win_rate", 0.0))
                    cost_off_win_rate = float(res_off.get("win_rate", 0.0))
                    mdd = float(res_on.get("max_drawdown", 0.0))
                    eq = res_on.get("equity_curve")
                    final_balance = float(eq[-1]) if isinstance(eq, (list, tuple)) and len(eq) > 0 else 1.0
                    filter_skip_stats = res_on.get("filter_skip_stats") or {}

                    results.append({
                        "min_max_proba": min_max_proba,
                        "max_entropy": max_entropy,
                        "min_hold": min_hold,
                        "cooldown": cooldown,
                        "trades": trades,
                        "cost_on_return": cost_on_return,
                        "cost_off_return": cost_off_return,
                        "cost_on_win_rate": cost_on_win_rate,
                        "cost_off_win_rate": cost_off_win_rate,
                        "filter_skip_stats": filter_skip_stats,
                        "final_balance": final_balance,
                        "max_drawdown": mdd,
                    })

    # Sort: 1) trades >= 10 preferred, 2) cost_on_return desc, 3) cost_off_return desc
    def sort_key(r):
        # 0 = has enough trades (first), 1 = too few trades (last)
        enough = 0 if r["trades"] >= 10 else 1
        return (enough, -r["cost_on_return"], -r["cost_off_return"], -r["trades"])

    results_sorted = sorted(results, key=sort_key)
    top5 = results_sorted[:5]

    date_str = datetime.now().strftime("%Y%m%d")
    out_md = DIAG_DIR / f"tcn_filter_sweep_quick_{date_str}.md"
    out_json = DIAG_DIR / f"tcn_filter_sweep_quick_{date_str}.json"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "days": args.days,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "preset": args.preset,
        "top5": top5,
        "all_results": results,
    }

    lines = [
        "# TCN 필터 스윕 (Quick)",
        f"생성: {payload['generated_at']}",
        f"preset={args.preset}, days={args.days}, {args.symbol} {args.timeframe}",
        "",
        "## 상위 5개",
        "| min_max_proba | max_entropy | min_hold | cooldown | trades | cost_on_return | cost_off_return | cost_on_wr | cost_off_wr | mdd |",
        "|---------------|-------------|----------|----------|--------|----------------|-----------------|------------|-------------|-----|",
    ]
    for r in top5:
        lines.append(
            "| {:.2f} | {:.2f} | {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                r["min_max_proba"],
                r["max_entropy"],
                r["min_hold"],
                r["cooldown"],
                r["trades"],
                r["cost_on_return"],
                r["cost_off_return"],
                r["cost_on_win_rate"],
                r["cost_off_win_rate"],
                r["max_drawdown"],
            )
        )
    lines.extend(["", "## filter_skip_stats (상위 5개)", ""])
    for i, r in enumerate(top5, 1):
        lines.append(f"- #{i}: {r.get('filter_skip_stats', {})}")

    md_content = "\n".join(lines)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Saved: %s, %s", out_md, out_json)
    candidates_with_trades = [r for r in results if r["trades"] > 0]
    logger.info("Combos with trades>0: %s / %s", len(candidates_with_trades), len(results))
    print(md_content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
