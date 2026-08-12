#!/usr/bin/env python3
"""
Phase D9: Cost Sensitivity Test.

Determine whether strategy fails due to trading costs or lack of predictive edge
by sweeping cost levels and comparing cost_on vs cost_off.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

# Fixed strategy parameters
SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"
END_DATE = "2026-03-03"
DAYS_LIST = [180, 365, 720]
MIN_MAX_PROBA = 0.575
MAX_ENTROPY = 1.30
MIN_HOLD = 36
COOLDOWN = 12
TIME_STOP_BARS = 72
EARLY_EXIT_BAD_K = 8
EARLY_EXIT_ON = True
REGIME_FILTER = "off"
POSITION_SCALING = "off"

# Cost sweep: (fee + slippage) per round-trip trade
COST_LEVELS = [0.0, 0.0002, 0.0004, 0.0006, 0.0008, 0.0010]

# Baseline check at cost=0.001
BASELINE_COST = 0.001
BASELINE_365D_COST_ON = -0.057
BASELINE_720D_COST_ON = -0.112
BASELINE_TOLERANCE = 0.05

DIAG = PROJECT_ROOT / "data" / "diagnostics"
MODELS_DIR = DIAG / "models"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
BACKTESTS_DIR = PROJECT_ROOT / "data" / "backtests"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)


def _get_ohlcv_and_proba(
    symbol: str,
    timeframe: str,
    days: int,
    end_date: str,
    feature_config,
    use_events: bool,
    model_path: Path,
) -> tuple[tuple[pd.DataFrame, np.ndarray, np.ndarray] | None, str | None]:
    """Load OHLCV, build features, run TCN inference, align df_bt and pl/ps."""
    from src.services.ohlcv_service import load_ohlcv_df
    from src.indicators.basic import add_basic_indicators
    from src.ml.features import build_feature_frame
    from src.dl.tcn_model import TCNSignalModel

    df = load_ohlcv_df(timeframe=timeframe, symbol=symbol)
    df = add_basic_indicators(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    start_ts = pd.Timestamp(end_date).tz_localize("UTC") - pd.Timedelta(days=days)
    end_ts = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1)
    if df["timestamp"].dt.tz is None:
        start_ts = start_ts.tz_localize(None)
        end_ts = end_ts.tz_localize(None)
    df = df.loc[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()
    if len(df) < 100:
        return (None, f"rows={len(df)} < 100")

    features = build_feature_frame(
        df, symbol=symbol, timeframe=timeframe, feature_config=feature_config
    )
    features = features.dropna()
    if len(features) < 60:
        return (None, f"features rows={len(features)} < 60")

    model = TCNSignalModel(model_path=model_path, use_events=use_events)
    if not model.is_loaded():
        return (None, "TCN model failed to load")
    pl_arr, ps_arr = model.predict_proba_batch(
        features=features, symbol=symbol, timeframe=timeframe, batch_size=512
    )
    pl = np.asarray(pl_arr, dtype=np.float32)
    ps = np.asarray(ps_arr, dtype=np.float32)

    if "close" in features.columns and "high" in features.columns and "low" in features.columns:
        df_bt = features[["close", "high", "low"]].copy()
        if isinstance(features.index, pd.DatetimeIndex):
            df_bt["timestamp"] = features.index
            df_bt = df_bt.reset_index(drop=True)
        elif "timestamp" in features.columns:
            df_bt["timestamp"] = features["timestamp"].values
    else:
        df_indexed = df.set_index("timestamp")
        df_bt = df_indexed.reindex(features.index)[["close", "high", "low"]].dropna(how="any")
        if len(df_bt) == 0:
            return (None, "reindex df_bt empty")
        common = df_bt.index
        features = features.loc[common]
        pos = [list(features.index).index(i) for i in common]
        pl = np.asarray([pl[i] for i in pos], dtype=np.float32)
        ps = np.asarray([ps[i] for i in pos], dtype=np.float32)
        df_bt = df_bt.reset_index()
        if df_bt.columns[0] != "timestamp":
            df_bt = df_bt.rename(columns={df_bt.columns[0]: "timestamp"})
    if "timestamp" not in df_bt.columns and isinstance(features.index, pd.DatetimeIndex):
        df_bt = df_bt.reset_index()
        if len(df_bt.columns) > 0 and df_bt.columns[0] != "timestamp":
            df_bt = df_bt.rename(columns={df_bt.columns[0]: "timestamp"})

    if isinstance(df_bt.index, pd.DatetimeIndex) and "timestamp" in df_bt.columns:
        df_bt = df_bt.reset_index(drop=True)
    elif isinstance(df_bt.index, pd.DatetimeIndex):
        df_bt = df_bt.reset_index()
        if df_bt.columns[0] != "timestamp":
            df_bt = df_bt.rename(columns={df_bt.columns[0]: "timestamp"})

    if len(df_bt) != len(pl):
        window_size = int(getattr(model, "window_size", 60))
        proba_len = len(pl)
        if "timestamp" in features.columns:
            feat_index = pd.DatetimeIndex(pd.to_datetime(features["timestamp"]))
        else:
            feat_index = pd.DatetimeIndex(pd.to_datetime(features.index))
        align_ts = feat_index[window_size : window_size + proba_len]
        if len(align_ts) != proba_len:
            return (None, f"align_ts len={len(align_ts)} != proba_len={proba_len}")
        proba_df = pd.DataFrame({
            "timestamp": pd.to_datetime(align_ts.values),
            "pl": pl,
            "ps": ps,
        })
        proba_df = proba_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        df_bt = df_bt.copy()
        if "timestamp" not in df_bt.columns and isinstance(df_bt.index, pd.DatetimeIndex):
            df_bt = df_bt.reset_index()
            if df_bt.columns[0] != "timestamp":
                df_bt = df_bt.rename(columns={df_bt.columns[0]: "timestamp"})
        df_bt["timestamp"] = pd.to_datetime(df_bt["timestamp"])
        df_bt = df_bt.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        df_bt_aligned = df_bt[df_bt["timestamp"].isin(proba_df["timestamp"])].copy()
        df_bt_aligned = df_bt_aligned.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        joined = df_bt_aligned.merge(proba_df, on="timestamp", how="inner", validate="one_to_one")
        df_bt = joined[["timestamp", "close", "high", "low"]].copy()
        pl = joined["pl"].values.astype(np.float32)
        ps = joined["ps"].values.astype(np.float32)
    if len(df_bt) != len(pl):
        return (None, f"len(df_bt)={len(df_bt)} != len(pl)={len(pl)}")
    return ((df_bt, pl, ps), None)


def _run_backtest(
    symbol: str,
    timeframe: str,
    df_bt: pd.DataFrame,
    pl: np.ndarray,
    ps: np.ndarray,
    commission_rate: float,
    slippage_rate: float,
    emit_trade_log: bool = False,
) -> tuple[dict | None, str | None]:
    """Run backtest with given cost (commission + slippage). Round-trip cost = 2*(commission_rate+slippage_rate)."""
    from scripts.run_tcn_label_sweep_v2 import run_backtest_7d

    return run_backtest_7d(
        symbol,
        timeframe,
        df_bt,
        pl,
        ps,
        commission_rate,
        slippage_rate,
        min_max_proba=MIN_MAX_PROBA,
        max_entropy=MAX_ENTROPY,
        min_hold=MIN_HOLD,
        cooldown=COOLDOWN,
        regime_filter_enabled=False,
        position_scaling_enabled=False,
        early_exit_enabled=EARLY_EXIT_ON,
        early_exit_bad_k=EARLY_EXIT_BAD_K,
        time_stop_enabled=True,
        time_stop_bars=TIME_STOP_BARS,
        emit_trade_log=emit_trade_log,
    )


def _mean_median_hold(trade_events: list) -> tuple[float, float]:
    if not trade_events:
        return 0.0, 0.0
    bars = [
        e.get("bars_held") or e.get("holding_bars") or 0
        for e in trade_events
        if e.get("event", "").startswith("EXIT")
    ]
    bars = [b for b in bars if b > 0]
    if not bars:
        return 0.0, 0.0
    return float(np.mean(bars)), float(np.median(bars))


def _export_trade_log(trade_events: list, out_path: Path) -> None:
    exits = [e for e in trade_events if e.get("event", "").startswith("EXIT") and "exit_reason" in e]
    if not exits:
        return
    rows = []
    for e in exits:
        rows.append({
            "entry_time": e.get("entry_ts") or e.get("ts"),
            "exit_time": e.get("exit_ts") or e.get("ts"),
            "pnl": e.get("net_return") if e.get("net_return") is not None else e.get("profit"),
            "exit_reason": e.get("exit_reason", ""),
            "holding_bars": e.get("holding_bars") if e.get("holding_bars") is not None else e.get("bars_held"),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[TRADE_LOG] {out_path} ({len(rows)} rows)", flush=True)


def main() -> int:
    from src.features.ml_feature_config import MLFeatureConfig

    config = MLFeatureConfig.from_preset("base")
    config.use_event_features = True
    model_id = os.environ.get("PHASE_D9_MODEL_ID", "h15_t0p004")
    model_path = MODELS_DIR / f"tcn_{model_id}.pt"
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1

    # Load data once per period
    period_data = {}
    for days in DAYS_LIST:
        print(f"[D9] {days}d: loading OHLCV + inference ...", flush=True)
        triple, err = _get_ohlcv_and_proba(
            SYMBOL, TIMEFRAME, days, END_DATE,
            config, True, model_path,
        )
        if err:
            print(f"[D9] {days}d: {err}", flush=True)
            return 1
        period_data[days] = triple

    # Precompute cost_off for each period (one backtest per period with 0 cost)
    print("[D9] Computing cost_off (0 cost) per period ...", flush=True)
    cost_off = {180: None, 365: None, 720: None}
    for days in DAYS_LIST:
        df_bt, pl, ps = period_data[days]
        res, err = _run_backtest(SYMBOL, TIMEFRAME, df_bt, pl, ps, 0.0, 0.0, emit_trade_log=False)
        if err:
            return 1
        cost_off[days] = res.get("total_return")

    # Cost sweep: round-trip cost = cost_level. One-way = cost_level/2. Use commission_rate = cost_level/2, slippage_rate = 0.
    results = []
    trade_log_365_cost0 = None
    trade_log_365_cost1 = None

    for cost_level in COST_LEVELS:
        commission_rate = cost_level / 2.0
        slippage_rate = 0.0
        row = {
            "cost_level": cost_level,
            "180d_cost_on": None,
            "365d_cost_on": None,
            "720d_cost_on": None,
            "720d_mdd": None,
            "720d_trades": None,
            "win_rate": None,
            "180d_cost_off": cost_off[180],
            "365d_cost_off": cost_off[365],
            "720d_cost_off": cost_off[720],
        }
        for days in DAYS_LIST:
            df_bt, pl, ps = period_data[days]
            emit_log = (days == 365 and (cost_level == 0.0 or cost_level == 0.001))
            print(f"[D9] cost={cost_level} {days}d: backtest ...", flush=True)
            res, err = _run_backtest(
                SYMBOL, TIMEFRAME, df_bt, pl, ps,
                commission_rate, slippage_rate,
                emit_trade_log=emit_log,
            )
            if err:
                print(f"[D9] backtest failed: {err}", flush=True)
                return 1
            cost_on = res.get("total_return")
            if days == 180:
                row["180d_cost_on"] = cost_on
            elif days == 365:
                row["365d_cost_on"] = cost_on
                row["win_rate"] = res.get("win_rate")
                if emit_log and res.get("trade_events"):
                    if cost_level == 0.0:
                        trade_log_365_cost0 = res["trade_events"]
                    elif cost_level == 0.001:
                        trade_log_365_cost1 = res["trade_events"]
            elif days == 720:
                row["720d_cost_on"] = cost_on
                row["720d_mdd"] = res.get("max_drawdown")
                row["720d_trades"] = res.get("total_trades", 0)
        results.append(row)

    # Baseline check at cost=0.001
    row_001 = next((r for r in results if r["cost_level"] == BASELINE_COST), None)
    if row_001:
        c365 = row_001.get("365d_cost_on")
        c720 = row_001.get("720d_cost_on")
        if c365 is not None:
            dev = abs(c365 - BASELINE_365D_COST_ON) / max(abs(BASELINE_365D_COST_ON), 1e-8)
            if dev > BASELINE_TOLERANCE:
                print(f"[D9] WARN: 365d cost_on deviation {dev:.2%} > 5%. Possible data drift.", file=sys.stderr)
        if c720 is not None:
            dev = abs(c720 - BASELINE_720D_COST_ON) / max(abs(BASELINE_720D_COST_ON), 1e-8)
            if dev > BASELINE_TOLERANCE:
                print(f"[D9] WARN: 720d cost_on deviation {dev:.2%} > 5%. Possible data drift.", file=sys.stderr)

    # Export trade logs
    if trade_log_365_cost0:
        _export_trade_log(trade_log_365_cost0, BACKTESTS_DIR / "phase_d9_cost0_trades.csv")
    if trade_log_365_cost1:
        _export_trade_log(trade_log_365_cost1, BACKTESTS_DIR / "phase_d9_cost1_trades.csv")

    # Break-even: max cost level where 720d cost_on >= 0 (None if never >= 0)
    cost_break_even = None
    for r in sorted(results, key=lambda x: x["cost_level"], reverse=True):
        if r.get("720d_cost_on") is not None and r["720d_cost_on"] >= 0:
            cost_break_even = r["cost_level"]
            break

    # Cost slope: (return at 0.001 - return at 0) / 0.001
    row_0 = next((r for r in results if r["cost_level"] == 0.0), None)
    row_1 = next((r for r in results if r["cost_level"] == 0.001), None)
    if row_0 is not None and row_1 is not None and row_0.get("720d_cost_on") is not None and row_1.get("720d_cost_on") is not None:
        cost_slope = (row_1["720d_cost_on"] - row_0["720d_cost_on"]) / 0.001
    else:
        cost_slope = None

    # Verdict: Case A = positive at cost <= 0.0004; Case B = improves but negative at cost=0; Case C = strongly negative
    positive_at_low_cost = any(
        r.get("720d_cost_on") is not None and r["720d_cost_on"] >= 0
        for r in results if r["cost_level"] <= 0.0004
    )
    cost_on_zero = row_0["720d_cost_on"] if (row_0 and row_0.get("720d_cost_on") is not None) else None
    if positive_at_low_cost:
        verdict = "EDGE_PRESENT_COST_SENSITIVE"
    elif cost_on_zero is not None and cost_on_zero > -0.10:
        verdict = "WEAK_EDGE"
    else:
        verdict = "NO_EDGE"

    # Report
    lines = [
        "# Phase D9: Cost Sensitivity Summary",
        "",
        "## Strategy parameters (fixed)",
        f"- symbol={SYMBOL}, timeframe={TIMEFRAME}, end_date={END_DATE}",
        f"- min_max_proba={MIN_MAX_PROBA}, max_entropy={MAX_ENTROPY}",
        f"- min_hold={MIN_HOLD}, cooldown={COOLDOWN}",
        f"- time_stop={TIME_STOP_BARS}, early_exit_bad_k={EARLY_EXIT_BAD_K}, early_exit=on",
        f"- regime_filter={REGIME_FILTER}, position_scaling={POSITION_SCALING}",
        f"- days_list={DAYS_LIST}",
        "",
        "## Cost sensitivity table",
        "",
        "| cost_level | 180d cost_on | 365d cost_on | 720d cost_on | 720d MDD | 720d trades | win_rate |",
        "|------------|--------------|--------------|--------------|----------|-------------|----------|",
    ]
    for r in results:
        lines.append(
            "| {:.4f} | {} | {} | {} | {} | {} | {} |".format(
                r["cost_level"],
                f"{r['180d_cost_on']:.4f}" if r.get("180d_cost_on") is not None else "-",
                f"{r['365d_cost_on']:.4f}" if r.get("365d_cost_on") is not None else "-",
                f"{r['720d_cost_on']:.4f}" if r.get("720d_cost_on") is not None else "-",
                f"{r['720d_mdd']:.4f}" if r.get("720d_mdd") is not None else "-",
                r.get("720d_trades") if r.get("720d_trades") is not None else "-",
                f"{r['win_rate']:.2%}" if r.get("win_rate") is not None else "-",
            )
        )
    lines.extend([
        "",
        "## Additional analysis",
        f"- **cost_break_even:** {cost_break_even if cost_break_even is not None else 'N/A'} (max cost where 720d cost_on >= 0)",
        f"- **cost_slope:** {cost_slope} (performance change per 0.001 cost increase)" if cost_slope is not None else "- **cost_slope:** N/A",
        "",
        "## Verdict",
        f"- **{verdict}**",
        "",
        "Export:",
        "- data/backtests/phase_d9_cost0_trades.csv",
        "- data/backtests/phase_d9_cost1_trades.csv",
        "",
    ])
    report_path = REPORTS_DIR / "phase_d9_cost_sensitivity_summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Console output
    print("", flush=True)
    print("=" * 60, flush=True)
    print("Phase D9 Cost Sensitivity", flush=True)
    print("=" * 60, flush=True)
    print("1) Cost sensitivity table:", flush=True)
    for r in results:
        print(f"   cost={r['cost_level']:.4f} 180d={r.get('180d_cost_on')} 365d={r.get('365d_cost_on')} 720d={r.get('720d_cost_on')} MDD={r.get('720d_mdd')} trades={r.get('720d_trades')} win_rate={r.get('win_rate')}", flush=True)
    print(f"2) cost_break_even: {cost_break_even}", flush=True)
    print(f"3) cost_slope: {cost_slope}", flush=True)
    print(f"4) strategy classification: {verdict}", flush=True)
    print(f"Report: {report_path}", flush=True)
    print("=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
