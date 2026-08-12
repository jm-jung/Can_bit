#!/usr/bin/env python3
"""
D10.1 — 지표 의미 확인 및 gap별 trade metrics 표 생성.

- 720d만 로드 후 각 min_proba_gap에 대해 백테스트 1회씩 실행.
- 수집: entries_attempted, entries_executed, total_trades, filtered_trade_count.
- 결과를 data/reports/phase_d10_trade_metrics_table.md 에 저장.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"
END_DATE = "2026-03-03"
COMMISSION = 0.0009
SLIPPAGE = 0.0001
DIAG = PROJECT_ROOT / "data" / "diagnostics"
MODELS_DIR = DIAG / "models"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

RUNS = [
    ("phase_d10_baseline", 0.00),
    ("phase_d10_gap003", 0.03),
    ("phase_d10_gap005", 0.05),
    ("phase_d10_gap007", 0.07),
    ("phase_d10_gap010", 0.10),
]


def _get_ohlcv_and_proba_720(symbol: str, timeframe: str, end_date: str, feature_config, use_events: bool, model_path: Path):
    from src.services.ohlcv_service import load_ohlcv_df
    from src.indicators.basic import add_basic_indicators
    from src.ml.features import build_feature_frame
    from src.dl.tcn_model import TCNSignalModel

    df = load_ohlcv_df(timeframe=timeframe, symbol=symbol)
    df = add_basic_indicators(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    start_ts = pd.Timestamp(end_date).tz_localize("UTC") - pd.Timedelta(days=720)
    end_ts = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1)
    if df["timestamp"].dt.tz is None:
        start_ts, end_ts = start_ts.tz_localize(None), end_ts.tz_localize(None)
    df = df.loc[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()
    if len(df) < 100:
        return (None, f"rows={len(df)} < 100")
    features = build_feature_frame(df, symbol=symbol, timeframe=timeframe, feature_config=feature_config)
    features = features.dropna()
    if len(features) < 60:
        return (None, f"features rows={len(features)} < 60")
    model = TCNSignalModel(model_path=model_path, use_events=use_events)
    if not model.is_loaded():
        return (None, "TCN model failed to load")
    pl_arr, ps_arr = model.predict_proba_batch(features=features, symbol=symbol, timeframe=timeframe, batch_size=512)
    pl = np.asarray(pl_arr, dtype=np.float32)
    ps = np.asarray(ps_arr, dtype=np.float32)
    if "close" in features.columns and "high" in features.columns and "low" in features.columns:
        df_bt = features[["close", "high", "low"]].copy()
        if isinstance(features.index, pd.DatetimeIndex):
            df_bt["timestamp"] = features.index
            df_bt = df_bt.reset_index(drop=True)
        else:
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
    if len(df_bt) != len(pl):
        window_size = int(getattr(model, "window_size", 60))
        proba_len = len(pl)
        feat_index = pd.DatetimeIndex(pd.to_datetime(features["timestamp"] if "timestamp" in features.columns else features.index))
        align_ts = feat_index[window_size : window_size + proba_len]
        if len(align_ts) != proba_len:
            return (None, "align_ts mismatch")
        proba_df = pd.DataFrame({"timestamp": pd.to_datetime(align_ts.values), "pl": pl, "ps": ps})
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
        return (None, "len mismatch")
    return ((df_bt, pl, ps), None)


def main() -> int:
    from src.features.ml_feature_config import MLFeatureConfig
    from scripts.run_tcn_label_sweep_v2 import run_backtest_7d

    config = MLFeatureConfig.from_preset("base")
    config.use_event_features = True
    model_id = os.environ.get("PHASE_D10_MODEL_ID", "h15_t0p004")
    model_path = MODELS_DIR / f"tcn_{model_id}.pt"
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1

    print("[D10.1] 720d load + inference (once) ...", flush=True)
    triple, err = _get_ohlcv_and_proba_720(SYMBOL, TIMEFRAME, END_DATE, config, True, model_path)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    df_bt, pl, ps = triple

    rows = []
    for run_id, min_proba_gap in RUNS:
        print(f"[D10.1] {run_id} min_proba_gap={min_proba_gap} ...", flush=True)
        res, err_bt = run_backtest_7d(
            SYMBOL, TIMEFRAME, df_bt, pl, ps, COMMISSION, SLIPPAGE,
            min_max_proba=0.575, max_entropy=1.30, min_proba_gap=min_proba_gap,
            min_hold=36, cooldown=12, time_stop_enabled=True, time_stop_bars=72,
            early_exit_enabled=True, early_exit_bad_k=8, emit_trade_log=False,
        )
        if err_bt:
            print(f"ERROR: {err_bt}", file=sys.stderr)
            return 1
        entries_attempted = res.get("entries_attempted")
        cap = res.get("cap_trigger_stats") or {}
        entries_executed = cap.get("entries_executed")
        total_trades = res.get("total_trades", 0)
        filtered_trade_count = res.get("filtered_trade_count", 0)
        rejected_by_gap = filtered_trade_count
        rejected_total = (entries_attempted - entries_executed) if entries_attempted is not None and entries_executed is not None else None
        rows.append({
            "run_id": run_id,
            "min_proba_gap": min_proba_gap,
            "entries_attempted": entries_attempted,
            "entries_executed": entries_executed,
            "total_trades": total_trades,
            "filtered_trade_count": rejected_by_gap,
            "rejected_entry_attempts_total": rejected_total,
        })

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "phase_d10_trade_metrics_table.md"

    lines = [
        "# D10.1 — Trade metrics: 지표 의미 및 gap별 표",
        "",
        "## 1) `trades` (total_trades) 의미",
        "",
        "- **정의:** **Closed trades count** (round-trip 기준, ENTRY+EXIT가 완료된 트레이드 수).",
        "- **코드:** `src/backtest/engine._compute_trade_stats(trades)` → `total_trades = len(trades)`.",
        "- **결론:** **실제로 청산된 포지션 수** = 신호 횟수나 진입 시도 횟수가 아님.",
        "",
        "## 2) 지표 정의 요약",
        "",
        "| 지표 | 의미 | 비고 |",
        "|------|------|------|",
        "| **entries_attempted** | LONG 또는 SHORT **신호가 난 bar 수** (진입 시도로 간주되는 횟수) | `execute_trades` 내에서 `signal in (LONG, SHORT)` 일 때 +1 |",
        "| **entries_executed** | **실제로 포지션을 연 횟수** (모든 필터·cooldown·hysteresis 통과 후 진입) | `cap_trigger_stats['entries_executed']` |",
        "| **total_trades** | **청산 완료된 거래 수** (round-trip) | 보통 `entries_executed` 와 동일 (기간 내 전부 청산 시) |",
        "| **filtered_trade_count** | **min_proba_gap 때문에만** 진입이 거절된 횟수 | Phase D10 spread 필터 |",
        "| **rejected entry attempts (total)** | 진입 시도 중 실제 포지션으로 이어지지 않은 전체 횟수 | `entries_attempted - entries_executed` |",
        "",
        "## 3) D10 gap별 720d 메트릭 표",
        "",
        "| run_id | min_proba_gap | entries_attempted | entries_executed | total_trades (closed) | filtered_trade_count | rejected_entry_attempts_total |",
        "|--------|---------------|-------------------|------------------|------------------------|----------------------|--------------------------------|",
    ]
    for r in rows:
        ea = r["entries_attempted"] if r["entries_attempted"] is not None else "-"
        ee = r["entries_executed"] if r["entries_executed"] is not None else "-"
        tt = r["total_trades"]
        fc = r["filtered_trade_count"]
        rej = r["rejected_entry_attempts_total"] if r["rejected_entry_attempts_total"] is not None else "-"
        lines.append(f"| {r['run_id']} | {r['min_proba_gap']:.2f} | {ea} | {ee} | {tt} | {fc} | {rej} |")

    lines.extend([
        "",
        "---",
        "Generated by `scripts/run_phase_d10_trade_metrics_table.py` (720d single period).",
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWritten: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
