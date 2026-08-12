#!/usr/bin/env python3
"""
TCN A/B 비교: preset A (base) vs preset B (calendar_e0).
신호 품질 + 7일 백테스트 결과 비교 후 자동 판정 및 md/json 산출.

사용법:
  source .venv/bin/activate
  python -m scripts.run_tcn_calendar_e0_ab --days 7 --symbol BTCUSDT --timeframe 5m --preset-a base --preset-b calendar_e0
  python -m scripts.run_tcn_calendar_e0_ab --days 7 --force-rebuild-cache
  python -m scripts.run_tcn_calendar_e0_ab --days 7 --skip-backtest
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
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


def _build_cache(symbol: str, timeframe: str, preset: str, force_rebuild: bool, timeout_sec: int = 600) -> bool:
    """Build proba cache for given preset. Returns True if success."""
    cmd = [
        sys.executable,
        "-m",
        "src.optimization.ml_proba_cache",
        "--strategy",
        "ml_tcn",
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--preset",
        preset,
    ]
    if force_rebuild:
        cmd.append("--force-rebuild")
    try:
        r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        logger.warning("Cache build timed out for preset=%s (calendar_e0 may need dedicated model).", preset)
        return False
    if r.returncode != 0:
        logger.warning("Cache build failed for preset=%s: %s", preset, (r.stderr or r.stdout or "")[:500])
        return False
    return True


def _signal_quality_dict(days: int, symbol: str, timeframe: str, preset: str) -> dict:
    """Return section_2 style metrics for given preset (from cache)."""
    from src.monitoring.inspect_tcn_signal_quality import section_2_proba_cache
    out = section_2_proba_cache(days, skip_load=False, preset=preset, symbol=symbol, timeframe=timeframe)
    return out


def _load_filtered_cache(
    symbol: str,
    timeframe: str,
    preset: str,
    days: int,
):
    """Load proba cache for preset and filter to last `days` days. Returns (proba_long, proba_short, df) or None."""
    import pandas as pd
    from src.optimization.ml_proba_cache import _get_cache_path, CACHE_DIR

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
        logger.warning("Cache has no timestamp column for preset=%s", preset)
        return None
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is not None:
        from datetime import timezone
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    else:
        cutoff = datetime.utcnow() - timedelta(days=days)
    df = df[ts >= cutoff].copy()
    if df.empty:
        logger.warning("No rows after filtering to last %s days for preset=%s", days, preset)
        return None
    proba_long = df["proba_long"].values.astype("float32")
    proba_short = df["proba_short"].values.astype("float32")
    logger.info(
        "[A/B] Backtest data filtered to last %s days: preset=%s, rows=%s, ts_range=%s ~ %s",
        days, preset, len(df), str(df["timestamp"].min()), str(df["timestamp"].max()),
    )
    return proba_long, proba_short, df.reset_index(drop=True)


def _run_backtest_for_preset(
    symbol: str,
    timeframe: str,
    preset: str,
    days: int,
    commission_rate: float | None = None,
    slippage_rate: float | None = None,
    min_max_proba: float | None = None,
    max_entropy: float | None = None,
    min_hold_bars_override: int | None = None,
    cooldown_bars_override: int | None = None,
) -> dict | None:
    """Run backtest for TCN with given preset (last `days` only); return summary dict."""
    from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine
    from src.strategies.ml_thresholds import resolve_ml_thresholds

    filtered = _load_filtered_cache(symbol, timeframe, preset, days)
    if filtered is None:
        return None
    proba_long_arr, proba_short_arr, df_filtered = filtered
    bars_used = len(df_filtered)

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
            min_hold_bars_override=min_hold_bars_override,
            cooldown_bars_override=cooldown_bars_override,
        )
    except Exception as e:
        logger.warning("Backtest failed for preset=%s: %s", preset, e)
        return None

    # [Validation] Log result keys and counter semantics (once per preset at INFO)
    logger.info(
        "[A/B][VALIDATION] result keys for preset=%s: %s",
        preset,
        list(result.keys()) if isinstance(result, dict) else type(result).__name__,
    )
    for k in ("total_trades", "entries_attempted", "entries_executed", "equity_curve"):
        if isinstance(result, dict) and k in result:
            v = result[k]
            if k == "equity_curve" and isinstance(v, (list, tuple)):
                logger.info("[A/B][VALIDATION] %s: len=%s, last=%.6f", k, len(v), float(v[-1]) if v else None)
            else:
                logger.info("[A/B][VALIDATION] %s=%s", k, v)

    # Extract summary
    summary = {
        "total_return": float(result.get("total_return", 0.0)),
        "win_rate": float(result.get("win_rate", 0.0)),
        "max_drawdown": float(result.get("max_drawdown", 0.0)),
        "total_trades": int(result.get("total_trades", 0)),
        "bars_used": bars_used,
    }
    if "entries_attempted" in result:
        summary["entries_attempted"] = int(result["entries_attempted"])
    if "filters_applied" in result:
        summary["filters_applied"] = result["filters_applied"]
    if "filter_skip_stats" in result:
        summary["filter_skip_stats"] = result["filter_skip_stats"]
    cts = result.get("cap_trigger_stats") or {}
    if "entries_executed" in cts:
        summary["entries_executed"] = int(cts["entries_executed"])
    eq = result.get("equity_curve")
    if isinstance(eq, (list, tuple)) and len(eq) > 0:
        summary["equity_curve_len"] = len(eq)
        summary["final_balance"] = float(eq[-1])
    sg = result.get("strategy_guard_stats") or {}
    total_checks = sg.get("total_checks") or 0
    block_count = sg.get("block_count") or 0
    if total_checks > 0:
        summary["guard_block_ratio"] = block_count / total_checks
        summary["total_checks"] = total_checks
    fs = result.get("final_scale_stats") or {}
    if "mean" in fs:
        summary["avg_final_scale"] = float(fs["mean"])
    return summary


def _verdict(signal_a: dict, signal_b: dict, backtest_a: dict | None, backtest_b: dict | None) -> str:
    """판정: calendar_e0 유효 가능성 vs 미미."""
    if not signal_b.get("available") or not signal_a.get("available"):
        return "calendar_e0 단독 효과 판정 불가 (신호 데이터 부족)"
    avg_mp_a = signal_a.get("avg_max_proba") or signal_a.get("max_proba_mean") or 0.0
    avg_mp_b = signal_b.get("avg_max_proba") or signal_b.get("max_proba_mean") or 0.0
    ent_a = signal_a.get("entropy_mean") or 0.0
    ent_b = signal_b.get("entropy_mean") or 0.0
    signal_ok = (avg_mp_b >= avg_mp_a + 0.03) or (ent_b <= ent_a - 0.03)
    ret_ok = False
    wr_ok = False
    if backtest_a is not None and backtest_b is not None:
        ret_b = backtest_b.get("total_return") or 0.0
        ret_a = backtest_a.get("total_return") or 0.0
        wr_b = backtest_b.get("win_rate") or 0.0
        wr_a = backtest_a.get("win_rate") or 0.0
        ret_ok = ret_b >= ret_a + 0.002
        wr_ok = wr_b >= wr_a + 0.03
    if signal_ok and (ret_ok or wr_ok):
        return "calendar_e0 유효 가능성"
    return "calendar_e0 단독 효과 미미 (추가 이벤트 확장/feature 수정 필요)"


def main():
    parser = argparse.ArgumentParser(description="TCN A/B: base vs calendar_e0")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--preset-a", type=str, default="base")
    parser.add_argument("--preset-b", type=str, default="calendar_e0")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--commission", type=float, default=None, help="Override commission rate (e.g. 0 for cost-off test)")
    parser.add_argument("--slippage", type=float, default=None, help="Override slippage rate (e.g. 0 for cost-off test)")
    parser.add_argument("--min-max-proba", type=float, default=None, help="Entry filter: skip if max(proba) < this")
    parser.add_argument("--max-entropy", type=float, default=None, help="Entry filter: skip if entropy > this")
    parser.add_argument("--min-hold", type=int, default=None, help="Override min_hold_bars")
    parser.add_argument("--cooldown", type=int, default=None, help="Override cooldown_bars")
    args = parser.parse_args()

    date_str = datetime.now().strftime("%Y%m%d")
    out_md = DIAG_DIR / f"tcn_calendar_e0_ab_{date_str}.md"
    out_json = DIAG_DIR / f"tcn_calendar_e0_ab_{date_str}.json"

    logger.info("A/B: preset_a=%s, preset_b=%s, days=%s, symbol=%s, timeframe=%s",
                args.preset_a, args.preset_b, args.days, args.symbol, args.timeframe)

    # 1) Cache A
    if not _build_cache(args.symbol, args.timeframe, args.preset_a, args.force_rebuild_cache, timeout_sec=600):
        logger.warning("Cache A (%s) build failed or skipped (reuse existing if any)", args.preset_a)
    # 2) Cache B (calendar_e0 can timeout if no dedicated model)
    timeout_b = 120 if args.preset_b == "calendar_e0" else 600
    if not _build_cache(args.symbol, args.timeframe, args.preset_b, args.force_rebuild_cache, timeout_sec=timeout_b):
        logger.warning("Cache B (%s) build failed or skipped. calendar_e0 may need a dedicated TCN model.", args.preset_b)

    # 3) Signal quality A & B
    signal_a = _signal_quality_dict(args.days, args.symbol, args.timeframe, args.preset_a)
    signal_b = _signal_quality_dict(args.days, args.symbol, args.timeframe, args.preset_b)

    # 4) Backtest A & B: cost_on (commission/slippage) + cost_off (0, 0) each
    backtest_a = None
    backtest_b = None
    filter_kw = {
        "min_max_proba": args.min_max_proba,
        "max_entropy": args.max_entropy,
        "min_hold_bars_override": args.min_hold,
        "cooldown_bars_override": args.cooldown,
    }
    if not args.skip_backtest:
        def _run_both_costs(preset: str):
            cost_on = _run_backtest_for_preset(
                args.symbol, args.timeframe, preset, args.days,
                commission_rate=args.commission, slippage_rate=args.slippage,
                **filter_kw,
            )
            cost_off = _run_backtest_for_preset(
                args.symbol, args.timeframe, preset, args.days,
                commission_rate=0.0, slippage_rate=0.0,
                **filter_kw,
            )
            attempted = (cost_on or {}).get("entries_attempted") or 0
            trades_on = (cost_on or {}).get("total_trades") or 0
            has_filter = any(
                filter_kw.get(k) is not None
                for k in ("min_max_proba", "max_entropy", "min_hold_bars_override", "cooldown_bars_override")
            )
            ratio = (attempted - trades_on) / attempted if (attempted > 0 and has_filter) else None
            return {
                "cost_on": cost_on,
                "cost_off": cost_off,
                "trade_reduction_ratio": ratio,
            }
        backtest_a = _run_both_costs(args.preset_a)
        backtest_b = _run_both_costs(args.preset_b)

    # 5) Compare & verdict (use cost_on for verdict)
    verdict = _verdict(
        signal_a, signal_b,
        backtest_a.get("cost_on") if backtest_a else None,
        backtest_b.get("cost_on") if backtest_b else None,
    )

    # 6) Build comparison table and payload
    def _signal_row(name: str, s: dict) -> dict:
        return {
            "n_rows": s.get("n_rows"),
            "avg_max_proba": s.get("avg_max_proba") or s.get("max_proba_mean"),
            "entropy_mean": s.get("entropy_mean"),
            "ratio_long": s.get("ratio_long"),
            "ratio_short": s.get("ratio_short"),
            "ratio_flat": s.get("ratio_flat"),
            "max_proba_lt_055": s.get("max_proba_lt_055"),
        }

    payload = {
        "generated_at": datetime.now().isoformat(),
        "days": args.days,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "preset_a": args.preset_a,
        "preset_b": args.preset_b,
        "signal": {
            args.preset_a: _signal_row(args.preset_a, signal_a),
            args.preset_b: _signal_row(args.preset_b, signal_b),
        },
        "backtest": {
            args.preset_a: backtest_a,
            args.preset_b: backtest_b,
        },
        "verdict": verdict,
    }

    # 7) Markdown report
    lines = [
        "# TCN calendar_e0 A/B 비교",
        f"생성: {payload['generated_at']}",
        f"기준: 최근 {args.days}일, {args.symbol} {args.timeframe}",
        "",
        "## 신호 품질",
        "| 지표 | " + args.preset_a + " | " + args.preset_b + " |",
        "|------|------|------|",
    ]
    for key in ["n_rows", "avg_max_proba", "entropy_mean", "max_proba_lt_055"]:
        va = signal_a.get(key) if isinstance(signal_a.get(key), (int, float)) else "-"
        vb = signal_b.get(key) if isinstance(signal_b.get(key), (int, float)) else "-"
        if isinstance(va, float):
            va = f"{va:.4f}"
        if isinstance(vb, float):
            vb = f"{vb:.4f}"
        lines.append(f"| {key} | {va} | {vb} |")
    lines.extend([
        "",
        "## 백테스트 (비용 포함)",
        "| 지표 | " + args.preset_a + " | " + args.preset_b + " |",
        "|------|------|------|",
    ])
    for key in ["total_return", "win_rate", "max_drawdown", "total_trades", "guard_block_ratio", "avg_final_scale"]:
        a_on = (backtest_a or {}).get("cost_on") or {}
        b_on = (backtest_b or {}).get("cost_on") or {}
        va = a_on.get(key) if a_on else "-"
        vb = b_on.get(key) if b_on else "-"
        if isinstance(va, float):
            va = f"{va:.4f}"
        if isinstance(vb, float):
            vb = f"{vb:.4f}"
        lines.append(f"| {key} | {va} | {vb} |")
    lines.extend([
        "",
        "## 백테스트 (비용 0)",
        "| 지표 | " + args.preset_a + " | " + args.preset_b + " |",
        "|------|------|------|",
    ])
    for key in ["total_return", "win_rate", "max_drawdown", "total_trades"]:
        a_off = (backtest_a or {}).get("cost_off") or {}
        b_off = (backtest_b or {}).get("cost_off") or {}
        va = a_off.get(key) if a_off else "-"
        vb = b_off.get(key) if b_off else "-"
        if isinstance(va, float):
            va = f"{va:.4f}"
        if isinstance(vb, float):
            vb = f"{vb:.4f}"
        lines.append(f"| {key} | {va} | {vb} |")
    lines.extend([
        "",
        "## Trade reduction ratio (필터 적용 시)",
        "",
    ])
    if backtest_a is not None and backtest_b is not None:
        rra = (backtest_a or {}).get("trade_reduction_ratio")
        rrb = (backtest_b or {}).get("trade_reduction_ratio")
        if rra is not None and rrb is not None:
            lines.append(f"- {args.preset_a}: {rra:.4f}")
            lines.append(f"- {args.preset_b}: {rrb:.4f}")
        else:
            lines.append("- (필터 미적용 시 N/A)")
    else:
        lines.append("- (백테스트 스킵)")
    lines.extend([
        "",
        "## 판정",
        f"**{verdict}**",
        "",
    ])
    md_content = "\n".join(lines)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(md_content)
    print(f"\nSaved: {out_md}, {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
