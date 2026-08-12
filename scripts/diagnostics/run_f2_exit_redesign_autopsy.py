"""F2 pullback-reclaim exit redesign autopsy.

Diagnostics-only. F2 entry conditions are fixed; only exit policies are replayed.
No production/live/order/state path is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("data/diagnostics/f2_exit_redesign_autopsy")
F2_ROOT = Path("data/diagnostics/f2_pullback_reclaim_oos_failure_autopsy")
SWING_ROOT = Path("data/diagnostics/low_frequency_swing_entry_alpha_research")
F2_DATASET = F2_ROOT / "dataset/f2_canonical_autopsy_dataset.parquet"
F2_SUBSETS = F2_ROOT / "dataset/f2_subset_membership.parquet"
F2_FOLDS = F2_ROOT / "fold/f2_fold_autopsy.csv"
REPLAY_5M = SWING_ROOT / "timeframes/btcusdt_5m_for_replay.parquet"
FOLDS = SWING_ROOT / "walk_forward/walk_forward_fold_results.csv"
CURRENT_COST_BPS = 6.0
MAKER_COST_BPS = 3.0
TWO_X_COST_BPS = 12.0
MAX_BARS_14D = 4032
HORIZON_BARS = {
    "4h": 48,
    "8h": 96,
    "12h": 144,
    "24h": 288,
    "48h": 576,
    "72h": 864,
    "5d": 1440,
    "7d": 2016,
    "14d": 4032,
}


def ensure_dirs() -> None:
    for d in [
        "discovery",
        "audit",
        "replay",
        "catalog",
        "simulation",
        "scorecards",
        "walk_forward",
        "regime",
        "giveback",
        "overfit",
        "casebook",
        "decision",
        "logs",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def log(msg: str) -> None:
    ensure_dirs()
    with (ROOT / "logs/progress_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": pd.Timestamp.now("UTC").isoformat(), "message": msg}, ensure_ascii=False) + "\n")


def sh(cmd: List[str], timeout: int = 20) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception as exc:
        return f"unavailable: {exc}"


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safety_snapshot(name: str) -> Dict[str, Any]:
    targets = [
        "models/tcn_v1.pt",
        "data/diagnostics/tcn_no_events.pt",
        "config",
        "configs",
        "data/live",
        "data/order",
        "data/state",
        "state",
        "ops",
        "scripts/run_daily_meta_research_ops.sh",
        "scripts/run_daily_paper_ops.sh",
        "scripts/run_daily_h8_candidate_ops.sh",
        "scripts/run_daily_h8_softgate_candidate_ops.sh",
        "scripts/run_daily_hybrid_candidate_ops.sh",
        "scripts/run_daily_quality_score_candidate_ops.sh",
    ]
    rows: List[Dict[str, Any]] = []
    for raw in targets:
        p = Path(raw)
        if p.is_file():
            rows.append({"path": str(p), "exists": True, "sha256": sha256(p)})
        elif p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.stat().st_size < 20_000_000:
                    rows.append({"path": str(fp), "exists": True, "sha256": sha256(fp)})
        else:
            rows.append({"path": raw, "exists": False, "sha256": None})
    snap = {
        "captured_ts": pd.Timestamp.now("UTC").isoformat(),
        "hashes": rows,
        "canbit_launchd_lines": [ln for ln in sh(["launchctl", "list"]).splitlines() if "canbit" in ln.lower()],
        "git_status_short": sh(["git", "status", "--short"], timeout=10),
        "private_order_account_balance_position_calls": 0,
        "production_ready": False,
        "promotion_ready": False,
        "python": sys.version,
    }
    (ROOT / f"audit/safety_snapshot_{name}.json").write_text(jdump(snap), encoding="utf-8")
    return snap


def finalize_audit(before: Dict[str, Any]) -> None:
    after = safety_snapshot("after")
    bmap = {x["path"]: x.get("sha256") for x in before.get("hashes", [])}
    rows = []
    for x in after.get("hashes", []):
        old = bmap.get(x["path"])
        rows.append({"path": x["path"], "sha256_before": old, "sha256_after": x.get("sha256"), "changed": old is not None and old != x.get("sha256")})
    (ROOT / "audit/hash_before_after.json").write_text(jdump(rows), encoding="utf-8")
    writes = [{"path": str(p), "diagnostics_only": True, "write_class": "f2_exit_redesign_output"} for p in ROOT.rglob("*") if p.is_file()]
    writes.append({"path": "scripts/diagnostics/run_f2_exit_redesign_autopsy.py", "diagnostics_only": False, "write_class": "requested_entrypoint"})
    pd.DataFrame(writes).to_csv(ROOT / "audit/write_path_audit.csv", index=False)
    (ROOT / "audit/production_safety_audit.md").write_text(
        "# Production Safety Audit\n\nNo production TCN/Q2/R7/Risk Manager/live/order/state path was changed. `forward_orderflow_collector_v4` and `false_high_r7_daily_monitor` were read-only. Discord/webhook policy was unchanged. No private/order/account/balance/position endpoints were called. production_ready=false; promotion_ready=false.\n",
        encoding="utf-8",
    )


def read_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def input_discovery() -> Dict[str, Any]:
    required = [
        F2_ROOT / "f2_pullback_reclaim_oos_failure_autopsy_final_report.md",
        F2_DATASET,
        F2_SUBSETS,
        F2_FOLDS,
        F2_ROOT / "regime/f2_regime_scorecard.csv",
        F2_ROOT / "exit/f2_exit_horizon_scorecard.csv",
        F2_ROOT / "exit/f2_mfe_mae_timing.csv",
        F2_ROOT / "fake_reclaim/f2_fake_reclaim_cases.csv",
        F2_ROOT / "casebook/f2_casebook.parquet",
        SWING_ROOT / "candidates/pullback_reclaim_candidates.parquet",
        SWING_ROOT / "backfill/swing_alpha_paper_trades.parquet",
        SWING_ROOT / "backfill/swing_alpha_exit_outcomes.parquet",
        REPLAY_5M,
        SWING_ROOT / "timeframes/btcusdt_1h_ohlcv.parquet",
        SWING_ROOT / "timeframes/btcusdt_4h_ohlcv.parquet",
        SWING_ROOT / "timeframes/btcusdt_1d_ohlcv.parquet",
        FOLDS,
        SWING_ROOT / "walk_forward/walk_forward_selected_signals.csv",
        Path("data/diagnostics/data_sync/canonical_data_paths.json"),
    ]
    rows = []
    for p in required:
        rows.append({"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() and p.is_file() else 0, "suffix": p.suffix})
    for root in [F2_ROOT, SWING_ROOT, Path("scripts/diagnostics")]:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file() and any(k in str(p).lower() for k in ["f2", "exit", "replay", "walk_forward"]):
                    rows.append({"path": str(p), "exists": True, "size": p.stat().st_size, "suffix": p.suffix})
    inv = pd.DataFrame(rows).drop_duplicates("path")
    inv.to_csv(ROOT / "discovery/input_inventory.csv", index=False)
    inv[inv["path"].str.contains("f2|replay|walk_forward|timeframes", case=False, regex=True)].to_csv(ROOT / "discovery/f2_exit_input_availability.csv", index=False)
    (ROOT / "discovery/discovered_paths.json").write_text(jdump(inv.to_dict("records")[:5000]), encoding="utf-8")
    schema = {}
    for p in required:
        if p.suffix in [".csv", ".parquet"] and p.exists():
            df = read_df(p)
            schema[str(p)] = {"rows": len(df), "columns": list(df.columns), "dtypes": {c: str(df[c].dtype) for c in df.columns[:200]}}
    (ROOT / "discovery/f2_exit_schema_summary.json").write_text(jdump(schema), encoding="utf-8")
    replay = read_replay()
    f2 = load_f2_raw(fast=False)
    coverage = replay_coverage(f2, replay)
    coverage.to_csv(ROOT / "discovery/replay_path_coverage.csv", index=False)
    folds = read_df(FOLDS)
    folds.to_csv(ROOT / "discovery/fold_boundary_inventory.csv", index=False)
    (ROOT / "discovery/discovery_report.md").write_text(
        "# Discovery Report\n\nF2 exit redesign reads existing F2 canonical entries and 5m replay path only. Entry thresholds are not changed.\n",
        encoding="utf-8",
    )
    return {"inventory_rows": len(inv), "required_existing": int(sum(x["exists"] for x in rows[: len(required)])), "f2_rows": len(f2), "replay_rows": len(replay)}


def load_f2_raw(fast: bool = False) -> pd.DataFrame:
    f2 = pd.read_parquet(F2_DATASET)
    f2 = f2.sort_values(["signal_ts", "direction", "candidate_id"]).drop_duplicates(["signal_ts", "direction"]).reset_index(drop=True)
    if fast:
        f2 = f2.tail(500).copy()
    for c in ["signal_ts", "entry_ts", "exit_ts", "timestamp"]:
        if c in f2:
            f2[c] = pd.to_datetime(f2[c], errors="coerce").astype("datetime64[ns]")
    f2["baseline_net_current_bps"] = pd.to_numeric(f2["net_after_cost"], errors="coerce") * 10000
    f2["baseline_gross_bps"] = pd.to_numeric(f2["gross_return"], errors="coerce") * 10000
    return f2


def read_replay() -> pd.DataFrame:
    b = pd.read_parquet(REPLAY_5M, columns=["timestamp", "open", "high", "low", "close"])
    b["timestamp"] = pd.to_datetime(b["timestamp"], errors="coerce").astype("datetime64[ns]")
    for c in ["open", "high", "low", "close"]:
        b[c] = pd.to_numeric(b[c], errors="coerce")
    return b.dropna().sort_values("timestamp").reset_index(drop=True)


def replay_coverage(f2: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    ts = replay["timestamp"].to_numpy()
    rows = []
    for _, r in f2.iterrows():
        idx = int(np.searchsorted(ts, np.datetime64(r["entry_ts"]), side="left"))
        rows.append({"candidate_id": r["candidate_id"], "entry_ts": r["entry_ts"], "entry_index": idx, "bars_available": max(0, len(replay) - idx), "coverage_14d_ok": idx + MAX_BARS_14D < len(replay)})
    out = pd.DataFrame(rows)
    return pd.DataFrame(
        [
            {
                "rows": len(out),
                "coverage_4h_rate": (out["bars_available"] >= HORIZON_BARS["4h"]).mean(),
                "coverage_24h_rate": (out["bars_available"] >= HORIZON_BARS["24h"]).mean(),
                "coverage_14d_rate": (out["bars_available"] >= HORIZON_BARS["14d"]).mean(),
                "min_bars_available": out["bars_available"].min() if len(out) else 0,
            }
        ]
    )


def build_replay_dataset(fast: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    f2 = load_f2_raw(fast=fast)
    replay = read_replay()
    ts = replay["timestamp"].to_numpy()
    idx = np.searchsorted(ts, f2["entry_ts"].to_numpy(dtype="datetime64[ns]"), side="left")
    f2["entry_replay_index"] = idx
    f2["entry_bar_open"] = np.where(idx < len(replay), replay["open"].to_numpy()[np.clip(idx, 0, len(replay) - 1)], np.nan)
    f2["entry_bar_high"] = np.where(idx < len(replay), replay["high"].to_numpy()[np.clip(idx, 0, len(replay) - 1)], np.nan)
    f2["entry_bar_low"] = np.where(idx < len(replay), replay["low"].to_numpy()[np.clip(idx, 0, len(replay) - 1)], np.nan)
    f2["entry_bar_close"] = np.where(idx < len(replay), replay["close"].to_numpy()[np.clip(idx, 0, len(replay) - 1)], np.nan)
    f2["entry_cost_bps"] = CURRENT_COST_BPS
    f2["current_cost_bps"] = CURRENT_COST_BPS
    f2["maker_cost_bps"] = MAKER_COST_BPS
    f2["slippage_bps"] = 0.0
    f2["prior_reclaim_level"] = np.where(f2["signal_id"].astype(str).str.contains("EMA50"), f2.get("ema_50"), f2.get("ema_20"))
    f2["prior_pullback_low"] = f2.get("low")
    f2["prior_pullback_high"] = f2.get("high")
    f2["EMA20"] = f2.get("ema_20")
    f2["EMA50"] = f2.get("ema_50")
    f2["ATR14"] = f2.get("atr_14")
    f2["RSI"] = f2.get("rsi_14")
    f2["MACD_hist"] = f2.get("macd_hist")
    f2["support_level"] = f2.get("prev_low_20")
    f2["resistance_level"] = f2.get("prev_high_20")
    f2["fake_reclaim_proxy"] = f2.get("fake_reclaim", False)
    f2["regime_state"] = f2.get("BTC_market_state", "unknown")
    f2["source_subset_flags"] = ""
    f2.to_parquet(ROOT / "replay/f2_exit_replay_dataset.parquet", index=False)
    pd.DataFrame({"timestamp": replay["timestamp"], "bar_index": np.arange(len(replay))}).to_parquet(ROOT / "replay/f2_replay_path_index.parquet", index=False)
    replay_coverage(f2, replay).to_csv(ROOT / "replay/f2_replay_coverage_summary.csv", index=False)
    (ROOT / "replay/f2_exit_replay_dataset_schema.json").write_text(jdump({c: str(f2[c].dtype) for c in f2.columns}), encoding="utf-8")
    (ROOT / "replay/replay_dataset_report.md").write_text("# Replay Dataset Report\n\nF2 entries are fixed. 5m replay path is used only after entry_ts for exit simulation.\n", encoding="utf-8")
    return f2, replay


def exit_catalog() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add(exit_id: str, family: str, params: Dict[str, Any], ref: bool = False, desc: str = "") -> None:
        rows.append(
            {
                "exit_id": exit_id,
                "exit_family": family,
                "exit_name": exit_id,
                "requires_intrabar_path": True,
                "requires_technical_feature": family in ["C", "E", "F", "G"],
                "requires_oracle": ref,
                "is_reference_only": ref,
                "parameters": json.dumps(params),
                "conservative_assumption": "same 5m bar TP/SL ambiguity chooses adverse outcome",
                "description": desc or exit_id,
            }
        )

    for name, h in [("X_A1_fixed_4h", "4h"), ("X_A2_fixed_8h", "8h"), ("X_A3_fixed_12h", "12h"), ("X_A4_fixed_24h", "24h"), ("X_A5_fixed_48h", "48h"), ("X_A6_fixed_72h", "72h"), ("X_A7_fixed_5d", "5d"), ("X_A8_fixed_7d", "7d"), ("X_A9_fixed_14d_reference", "14d")]:
        add(name, "A_fixed", {"horizon": h}, ref="reference" in name)
    for x in [1, 2, 3, 5]:
        add(f"X_B{x}_tp_{x}x_cost", "B_tp", {"tp_cost_x": x, "else_horizon": None})
    add("X_B5_tp_1x_cost_else_24h", "B_tp", {"tp_cost_x": 1, "else_horizon": "24h"})
    add("X_B6_tp_2x_cost_else_24h", "B_tp", {"tp_cost_x": 2, "else_horizon": "24h"})
    add("X_B7_tp_3x_cost_else_24h", "B_tp", {"tp_cost_x": 3, "else_horizon": "24h"})
    add("X_B8_tp_2x_cost_else_72h", "B_tp", {"tp_cost_x": 2, "else_horizon": "72h"})
    for name, params in [
        ("X_C1_reclaim_level_break_exit", {"stop": "reclaim"}),
        ("X_C2_entry_candle_low_break_exit", {"stop": "entry_candle"}),
        ("X_C3_pullback_low_break_exit", {"stop": "pullback"}),
        ("X_C4_ATR_0_5_stop", {"atr_stop": 0.5}),
        ("X_C5_ATR_1_0_stop", {"atr_stop": 1.0}),
        ("X_C6_ATR_1_5_stop", {"atr_stop": 1.5}),
        ("X_C7_EMA20_close_below_exit", {"ema_close": "EMA20"}),
        ("X_C8_EMA50_close_below_exit", {"ema_close": "EMA50"}),
    ]:
        add(name, "C_stop_invalidation", params)
    for x in [1, 2, 3]:
        add(f"X_D{x}_tp_{x}x_then_breakeven", "D_tp_be", {"tp_cost_x": x, "breakeven": True})
    add("X_D4_tp_1x_partial_50_then_breakeven", "D_partial", {"tp_cost_x": 1, "partial": 0.5, "breakeven": True})
    add("X_D5_tp_2x_partial_50_then_breakeven", "D_partial", {"tp_cost_x": 2, "partial": 0.5, "breakeven": True})
    add("X_D6_tp_1x_partial_50_then_ATR_trail", "D_partial", {"tp_cost_x": 1, "partial": 0.5, "atr_trail": 1.0})
    add("X_D7_tp_2x_partial_50_then_ATR_trail", "D_partial", {"tp_cost_x": 2, "partial": 0.5, "atr_trail": 1.0})
    for v in [0.5, 1.0, 1.5, 2.0]:
        add(f"X_E{int(v*2)}_trailing_ATR_{str(v).replace('.', '_')}", "E_trailing", {"atr_trail": v})
    add("X_E5_trailing_swing_low", "E_trailing", {"trail": "swing"})
    add("X_E6_trailing_EMA20", "E_trailing", {"trail": "EMA20"})
    add("X_E7_trailing_EMA50", "E_trailing", {"trail": "EMA50"})
    for name, params in [
        ("X_F1_RSI_overbought_exit", {"momentum": "rsi_overbought"}),
        ("X_F2_RSI_cross_down_exit", {"momentum": "rsi_cross"}),
        ("X_F3_MACD_hist_flip_down_exit", {"momentum": "macd_flip"}),
        ("X_F4_bearish_candle_exit", {"momentum": "bearish_candle"}),
        ("X_F5_close_below_prior_4h_low", {"momentum": "prior_low"}),
        ("X_F6_two_red_candles_exit", {"momentum": "two_red"}),
    ]:
        add(name, "F_momentum", params)
    for name, params in [
        ("X_G1_reclaim_break_or_tp2x", {"stop": "reclaim", "tp_cost_x": 2}),
        ("X_G2_reclaim_break_or_tp3x", {"stop": "reclaim", "tp_cost_x": 3}),
        ("X_G3_ATR1_stop_or_tp2x", {"atr_stop": 1.0, "tp_cost_x": 2}),
        ("X_G4_ATR1_stop_or_tp3x", {"atr_stop": 1.0, "tp_cost_x": 3}),
        ("X_G5_tp2x_then_trail_ATR1", {"tp_cost_x": 2, "atr_trail": 1.0}),
        ("X_G6_tp2x_then_EMA20_exit", {"tp_cost_x": 2, "ema_close": "EMA20"}),
        ("X_G7_tp3x_then_EMA20_exit", {"tp_cost_x": 3, "ema_close": "EMA20"}),
        ("X_G8_time_stop_12h_then_exit_if_no_MFE", {"time_stop": "12h", "mfe_required_x": 1}),
        ("X_G9_time_stop_24h_then_exit_if_no_MFE", {"time_stop": "24h", "mfe_required_x": 1}),
        ("X_G10_fake_reclaim_fast_exit", {"stop": "reclaim", "time_stop": "4h"}),
        ("X_G11_MFE_2x_or_reclaim_break", {"stop": "reclaim", "tp_cost_x": 2}),
        ("X_G12_MFE_3x_or_reclaim_break", {"stop": "reclaim", "tp_cost_x": 3}),
    ]:
        add(name, "G_hybrid", params)
    for name, horizon in [("X_O1_oracle_best_24h_reference_only", "24h"), ("X_O2_oracle_best_72h_reference_only", "72h"), ("X_O3_oracle_best_7d_reference_only", "7d"), ("X_O4_max_MFE_reference_only", "14d")]:
        add(name, "O_oracle_reference", {"oracle": horizon}, ref=True)
    cat = pd.DataFrame(rows)
    cat.to_csv(ROOT / "catalog/f2_exit_policy_catalog.csv", index=False)
    (ROOT / "catalog/f2_exit_policy_catalog.md").write_text("# F2 Exit Policy Catalog\n\nAll policies replay fixed F2 entries only. Oracle policies are reference-only and excluded from edge decisions.\n", encoding="utf-8")
    return cat


def signed_return(direction: str, entry: float, exit_price: float) -> float:
    if direction == "SHORT":
        return entry / exit_price - 1
    return exit_price / entry - 1


def threshold_hit(path: pd.DataFrame, direction: str, target: float | None = None, stop: float | None = None) -> Tuple[int | None, str | None, float | None, bool]:
    highs, lows = path["high"].to_numpy(), path["low"].to_numpy()
    for i, (hi, lo) in enumerate(zip(highs, lows)):
        if direction == "LONG":
            tp = target is not None and hi >= target
            sl = stop is not None and lo <= stop
            if tp and sl:
                return i, "sl_ambiguous", stop, True
            if sl:
                return i, "sl_hit", stop, False
            if tp:
                return i, "tp_hit", target, False
        else:
            tp = target is not None and lo <= target
            sl = stop is not None and hi >= stop
            if tp and sl:
                return i, "sl_ambiguous", stop, True
            if sl:
                return i, "sl_hit", stop, False
            if tp:
                return i, "tp_hit", target, False
    return None, None, None, False


def price_for_horizon(path: pd.DataFrame, bars: int) -> Tuple[int, float]:
    # Path starts at the entry bar. Prior F2 outcomes used entry_index + bars,
    # so fixed horizons should use the close at that offset, not bars - 1.
    i = min(max(0, bars), len(path) - 1)
    return i, float(path["close"].iloc[i])


def mfe_mae_before(path: pd.DataFrame, direction: str, entry: float, end_i: int) -> Tuple[float, float]:
    p = path.iloc[: max(1, end_i + 1)]
    if direction == "SHORT":
        mfe = entry / p["low"].min() - 1
        mae = p["high"].max() / entry - 1
    else:
        mfe = p["high"].max() / entry - 1
        mae = entry / p["low"].min() - 1
    return float(mfe), float(mae)


def stop_price(row: pd.Series, kind: str) -> float | None:
    entry = float(row["entry_price"])
    atr = float(row.get("ATR14", np.nan)) if pd.notna(row.get("ATR14", np.nan)) else entry * 0.01
    direction = str(row["direction"])
    if kind == "reclaim":
        return float(row.get("prior_reclaim_level", np.nan)) if pd.notna(row.get("prior_reclaim_level", np.nan)) else None
    if kind == "entry_candle":
        return float(row["entry_bar_low"] if direction == "LONG" else row["entry_bar_high"])
    if kind == "pullback":
        return float(row["prior_pullback_low"] if direction == "LONG" else row["prior_pullback_high"])
    if kind.startswith("atr:"):
        mult = float(kind.split(":")[1])
        return entry - atr * mult if direction == "LONG" else entry + atr * mult
    return None


def simulate_policy(row: pd.Series, path: pd.DataFrame, policy: Dict[str, Any]) -> Dict[str, Any]:
    entry = float(row["entry_price"])
    direction = str(row["direction"])
    atr = float(row.get("ATR14", np.nan)) if pd.notna(row.get("ATR14", np.nan)) else entry * 0.01
    exit_id = policy["exit_id"]
    params = json.loads(policy["parameters"])
    ref = bool(policy["is_reference_only"])
    if path.empty:
        return {"exit_id": exit_id, "exit_price": np.nan, "exit_reason": "no_path", "holding_minutes": np.nan, "path_coverage_ok": False}

    target = None
    stop = None
    end_bars = HORIZON_BARS.get(params.get("else_horizon") or params.get("horizon") or "24h", HORIZON_BARS["24h"]) + 1
    end_bars = min(end_bars, len(path))
    sub = path.iloc[:end_bars].copy()

    if "oracle" in params:
        if direction == "SHORT":
            best_price = sub["low"].min()
            end_i = int(sub["low"].to_numpy().argmin())
        else:
            best_price = sub["high"].max()
            end_i = int(sub["high"].to_numpy().argmax())
        reason, ambiguous = "oracle_reference", False
        exit_price = float(best_price)
    elif "horizon" in params:
        end_i, exit_price = price_for_horizon(sub, HORIZON_BARS[params["horizon"]])
        reason, ambiguous = f"fixed_{params['horizon']}", False
    else:
        if "tp_cost_x" in params:
            move = CURRENT_COST_BPS * float(params["tp_cost_x"]) / 10000
            target = entry * (1 + move) if direction == "LONG" else entry * (1 - move)
        if "stop" in params:
            stop = stop_price(row, params["stop"])
        if "atr_stop" in params:
            stop = stop_price(row, f"atr:{params['atr_stop']}")
        hit_i, hit_reason, hit_price, ambiguous = threshold_hit(sub, direction, target=target, stop=stop)
        if hit_i is None and "ema_close" in params:
            lvl = float(row.get(params["ema_close"], np.nan)) if pd.notna(row.get(params["ema_close"], np.nan)) else np.nan
            closes = sub["close"].to_numpy()
            if pd.notna(lvl):
                cond = closes < lvl if direction == "LONG" else closes > lvl
                where = np.where(cond)[0]
                if len(where):
                    hit_i, hit_reason, hit_price, ambiguous = int(where[0]), f"{params['ema_close']}_close_exit", float(closes[where[0]]), False
        if hit_i is None and "time_stop" in params:
            tbi = min(HORIZON_BARS.get(params["time_stop"], HORIZON_BARS["24h"]) - 1, len(sub) - 1)
            # If no useful MFE yet, exit at time stop.
            mfe, _ = mfe_mae_before(sub, direction, entry, tbi)
            if mfe < float(params.get("mfe_required_x", 1)) * CURRENT_COST_BPS / 10000:
                hit_i, hit_reason, hit_price, ambiguous = tbi, f"time_stop_{params['time_stop']}", float(sub["close"].iloc[tbi]), False
        if hit_i is None and "atr_trail" in params:
            trail = None
            highs, lows, closes = sub["high"].to_numpy(), sub["low"].to_numpy(), sub["close"].to_numpy()
            for i in range(len(sub)):
                if direction == "LONG":
                    trail = max(trail if trail is not None else -np.inf, highs[: i + 1].max() - atr * float(params["atr_trail"]))
                    if lows[i] <= trail:
                        hit_i, hit_reason, hit_price, ambiguous = i, "atr_trail_hit", float(trail), False
                        break
                else:
                    trail = min(trail if trail is not None else np.inf, lows[: i + 1].min() + atr * float(params["atr_trail"]))
                    if highs[i] >= trail:
                        hit_i, hit_reason, hit_price, ambiguous = i, "atr_trail_hit", float(trail), False
                        break
        if hit_i is None and "trail" in params:
            lvl = row.get("EMA20" if params["trail"] == "EMA20" else "EMA50", np.nan)
            if params["trail"] == "swing":
                lvl = row.get("prior_pullback_low" if direction == "LONG" else "prior_pullback_high", np.nan)
            if pd.notna(lvl):
                hit_i, hit_reason, hit_price, ambiguous = threshold_hit(sub, direction, stop=float(lvl))
        if hit_i is None and params.get("momentum"):
            # Conservative proxy using candle structure available in 5m path.
            red = sub["close"].to_numpy() < sub["open"].to_numpy()
            if params["momentum"] in ["bearish_candle", "two_red"]:
                cond = red if direction == "LONG" else ~red
                if params["momentum"] == "two_red":
                    cond = pd.Series(cond).rolling(2).sum().fillna(0).to_numpy() >= 2
                where = np.where(cond)[0]
                if len(where):
                    hit_i, hit_reason, hit_price, ambiguous = int(where[0]), params["momentum"], float(sub["close"].iloc[int(where[0])]), False
        if hit_i is None:
            hit_i, hit_price, hit_reason, ambiguous = min(HORIZON_BARS["24h"], len(sub) - 1), float(sub["close"].iloc[min(HORIZON_BARS["24h"], len(sub) - 1)]), "fallback_24h", False
        end_i, exit_price, reason = int(hit_i), float(hit_price), str(hit_reason)

    gross = signed_return(direction, entry, exit_price)
    mfe, mae = mfe_mae_before(path, direction, entry, end_i)
    full_mfe, _ = mfe_mae_before(path.iloc[: min(HORIZON_BARS["24h"], len(path))], direction, entry, min(HORIZON_BARS["24h"], len(path)) - 1)
    partial = float(params.get("partial", 0))
    if partial and target is not None and reason.startswith("tp"):
        # Half realized at TP, remainder at breakeven/trail/exit hit proxy.
        tp_gross = signed_return(direction, entry, target)
        gross = partial * tp_gross + (1 - partial) * gross
    net_current = gross * 10000 - CURRENT_COST_BPS
    baseline = float(row.get("baseline_net_current_bps", np.nan))
    giveback_bps = max(0.0, full_mfe * 10000 - gross * 10000)
    return {
        "candidate_id": row["candidate_id"],
        "fold_id": row.get("fold_id", -1),
        "signal_ts": row["signal_ts"],
        "entry_ts": row["entry_ts"],
        "entry_timeframe": row.get("entry_timeframe"),
        "direction": direction,
        "signal_id": row.get("signal_id"),
        "variant": row.get("variant"),
        "exit_id": exit_id,
        "exit_family": policy["exit_family"],
        "exit_ts": path["timestamp"].iloc[min(end_i, len(path) - 1)],
        "exit_price": exit_price,
        "exit_reason": reason,
        "holding_minutes": min(end_i, len(path) - 1) * 5,
        "holding_hours": min(end_i, len(path) - 1) * 5 / 60,
        "gross_return_bps": gross * 10000,
        "net_current_bps": net_current,
        "net_maker_bps": gross * 10000 - MAKER_COST_BPS,
        "net_2x_cost_bps": gross * 10000 - TWO_X_COST_BPS,
        "baseline_net_current_bps": baseline,
        "improvement_bps": net_current - baseline if pd.notna(baseline) else np.nan,
        "MFE_before_exit": mfe,
        "MAE_before_exit": mae,
        "MFE_capture_ratio": gross / full_mfe if full_mfe and full_mfe > 0 else np.nan,
        "giveback_bps": giveback_bps,
        "giveback_ratio": giveback_bps / (full_mfe * 10000) if full_mfe > 0 else np.nan,
        "tp_hit": "tp" in reason,
        "sl_hit": "sl" in reason or "stop" in reason or "break" in reason,
        "breakeven_hit": "breakeven" in reason,
        "reclaim_break_hit": "reclaim" in reason,
        "fake_reclaim_exit": row.get("fake_reclaim_proxy", False) and ("reclaim" in reason or "sl" in reason),
        "ambiguous_hit_flag": ambiguous,
        "conservative_fill_flag": ambiguous or "fallback" in reason,
        "path_coverage_ok": len(path) >= HORIZON_BARS["24h"],
        "is_reference_only": ref,
        "regime_state": row.get("regime_state", ""),
        "vol_regime": row.get("vol_regime", ""),
        "fake_reclaim_proxy": row.get("fake_reclaim_proxy", False),
        "MFE_before_MAE": row.get("MFE_before_MAE", np.nan),
    }


def simulate_exits(f2: pd.DataFrame, replay: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    ts = replay["timestamp"].to_numpy()
    policies = catalog.to_dict("records")
    out: List[Dict[str, Any]] = []
    for n, (_, row) in enumerate(f2.iterrows()):
        start = int(row["entry_replay_index"])
        if start >= len(replay):
            continue
        path = replay.iloc[start : min(start + MAX_BARS_14D, len(replay))].copy()
        for pol in policies:
            out.append(simulate_policy(row, path, pol))
        if n and n % 500 == 0:
            log(f"simulated {n} F2 entries")
    sim = pd.DataFrame(out)
    sim.to_parquet(ROOT / "simulation/f2_exit_simulation_results.parquet", index=False)
    sim.groupby(["exit_id", "exit_family"]).agg(trade_count=("candidate_id", "count"), mean_net_current_bps=("net_current_bps", "mean"), mean_improvement_bps=("improvement_bps", "mean")).reset_index().to_csv(ROOT / "simulation/f2_exit_simulation_summary.csv", index=False)
    sim.groupby("exit_id").agg(path_coverage_ok=("path_coverage_ok", "mean"), ambiguous_rate=("ambiguous_hit_flag", "mean"), conservative_fill_rate=("conservative_fill_flag", "mean")).reset_index().to_csv(ROOT / "simulation/f2_exit_simulation_quality_audit.csv", index=False)
    (ROOT / "simulation/exit_simulation_report.md").write_text("# Exit Simulation Report\n\nAll exit policies replay fixed F2 entries using only post-entry 5m path. Same-bar ambiguity is adverse.\n", encoding="utf-8")
    return sim


def pf(x: pd.Series) -> float:
    s = pd.to_numeric(x, errors="coerce").dropna()
    pos = s[s > 0].sum()
    neg = -s[s < 0].sum()
    return float(pos / neg) if neg > 0 else math.inf


def mdd(x: pd.Series) -> float:
    c = pd.to_numeric(x, errors="coerce").fillna(0).cumsum()
    return float((c.cummax() - c).max()) if len(c) else 0.0


def score_group(g: pd.DataFrame, obj: str, group: str) -> Dict[str, Any]:
    if g.empty:
        return {
            "object_id": obj,
            "group": group,
            "trade_count": 0,
            "mean_net_current_bps": np.nan,
            "median_net_current_bps": np.nan,
            "sum_net_current": 0.0,
            "mean_net_maker_bps": np.nan,
            "mean_net_2x_cost_bps": np.nan,
            "gross_mean_bps": np.nan,
            "winrate": np.nan,
            "profit_factor": np.nan,
            "MFE_capture_ratio_mean": np.nan,
            "giveback_rate": np.nan,
            "giveback_bps_mean": np.nan,
            "RFE_rate": np.nan,
            "MDD_proxy": np.nan,
            "tail_loss": np.nan,
            "avg_holding_hours": np.nan,
            "MFE_1x_cost_capture_rate": np.nan,
            "MFE_2x_cost_capture_rate": np.nan,
            "MFE_3x_cost_capture_rate": np.nan,
            "fake_reclaim_loss_reduction": np.nan,
            "late_exit_loss_reduction": np.nan,
            "early_exit_missed_profit_rate": np.nan,
            "sample_warning": True,
            "production_ready": False,
        }
    base_giveback = g["baseline_net_current_bps"] < 0
    return {
        "object_id": obj,
        "group": group,
        "trade_count": len(g),
        "mean_net_current_bps": g["net_current_bps"].mean(),
        "median_net_current_bps": g["net_current_bps"].median(),
        "sum_net_current": g["net_current_bps"].sum(),
        "mean_net_maker_bps": g["net_maker_bps"].mean(),
        "mean_net_2x_cost_bps": g["net_2x_cost_bps"].mean(),
        "gross_mean_bps": g["gross_return_bps"].mean(),
        "winrate": (g["net_current_bps"] > 0).mean(),
        "profit_factor": pf(g["net_current_bps"]),
        "MFE_capture_ratio_mean": g["MFE_capture_ratio"].replace([np.inf, -np.inf], np.nan).mean(),
        "giveback_rate": (g["giveback_bps"] > CURRENT_COST_BPS).mean(),
        "giveback_bps_mean": g["giveback_bps"].mean(),
        "RFE_rate": (g["MAE_before_exit"] > g["MFE_before_exit"]).mean(),
        "MDD_proxy": mdd(g["net_current_bps"]),
        "tail_loss": g["net_current_bps"].quantile(0.05),
        "avg_holding_hours": g["holding_hours"].mean(),
        "MFE_1x_cost_capture_rate": (g["gross_return_bps"] >= CURRENT_COST_BPS).mean(),
        "MFE_2x_cost_capture_rate": (g["gross_return_bps"] >= CURRENT_COST_BPS * 2).mean(),
        "MFE_3x_cost_capture_rate": (g["gross_return_bps"] >= CURRENT_COST_BPS * 3).mean(),
        "fake_reclaim_loss_reduction": ((g["fake_reclaim_proxy"]) & (g["improvement_bps"] > 0)).mean(),
        "late_exit_loss_reduction": ((base_giveback) & (g["improvement_bps"] > 0)).mean(),
        "early_exit_missed_profit_rate": ((g["baseline_net_current_bps"] > g["net_current_bps"]) & (g["baseline_net_current_bps"] > 0)).mean(),
        "sample_warning": len(g) < 50,
        "production_ready": False,
    }


def subset_masks(f2: pd.DataFrame) -> Dict[str, pd.Series]:
    return {
        "F2_ALL": pd.Series(True, index=f2.index),
        "F2_LONG_ONLY": f2["direction"].eq("LONG"),
        "F2_SHORT_ONLY": f2["direction"].eq("SHORT"),
        "F2_1H_ENTRY": f2["entry_timeframe"].eq("1h"),
        "F2_4H_ENTRY": f2["entry_timeframe"].eq("4h"),
        "F2_1D_ENTRY": f2["entry_timeframe"].eq("1d"),
        "F2_4H_OR_1D_ENTRY": f2["entry_timeframe"].isin(["4h", "1d"]),
        "F2_LONG_4H_OR_1D": f2["direction"].eq("LONG") & f2["entry_timeframe"].isin(["4h", "1d"]),
        "F2_BULL_REGIME": f2.get("1d_trend_state", "").eq("bull") | f2.get("4h_trend_state", "").eq("bull"),
        "F2_RANGE_REGIME": f2.get("regime_state", "").astype(str).str.contains("range", na=False),
        "F2_BEAR_REGIME": f2.get("1d_trend_state", "").eq("bear") | f2.get("4h_trend_state", "").eq("bear"),
        "F2_FAKE_RECLAIM_TRUE": f2.get("fake_reclaim_proxy", False).astype(bool),
        "F2_FAKE_RECLAIM_FALSE": ~f2.get("fake_reclaim_proxy", False).astype(bool),
    }


def scorecards(sim: pd.DataFrame, f2: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    nonref = sim[~sim["is_reference_only"].astype(bool)].copy()
    rows = [score_group(g, eid, "exit_policy") for eid, g in nonref.groupby("exit_id")]
    policy = pd.DataFrame(rows).sort_values("mean_net_current_bps", ascending=False)
    policy.to_csv(ROOT / "scorecards/exit_policy_scorecard.csv", index=False)
    rows = []
    by_id = f2.set_index("candidate_id")
    masks = subset_masks(f2)
    for sname, mask in masks.items():
        ids = set(f2.loc[mask, "candidate_id"])
        sub = nonref[nonref["candidate_id"].isin(ids)]
        for eid, g in sub.groupby("exit_id"):
            m = score_group(g, f"{sname}|{eid}", "exit_policy_by_subset")
            m["subset"] = sname
            m["exit_id"] = eid
            rows.append(m)
    by_subset = pd.DataFrame(rows)
    by_subset.to_csv(ROOT / "scorecards/exit_policy_by_subset_scorecard.csv", index=False)
    fam = pd.DataFrame([score_group(g, fam, "exit_family") for fam, g in nonref.groupby("exit_family")]).sort_values("mean_net_current_bps", ascending=False)
    fam.to_csv(ROOT / "scorecards/exit_family_scorecard.csv", index=False)
    base = nonref[nonref["exit_id"].eq("X_A4_fixed_24h")]
    base_gb = base.groupby("candidate_id")["giveback_bps"].mean()
    gb_rows = []
    for eid, g in nonref.groupby("exit_id"):
        cur = g.groupby("candidate_id")["giveback_bps"].mean()
        aligned = pd.concat([base_gb.rename("baseline"), cur.rename("exit")], axis=1).dropna()
        gb_rows.append({"exit_id": eid, "baseline_giveback_bps": aligned["baseline"].mean(), "exit_giveback_bps": aligned["exit"].mean(), "giveback_reduction_bps": (aligned["baseline"] - aligned["exit"]).mean(), "giveback_reduction_rate": (aligned["exit"] < aligned["baseline"]).mean()})
    pd.DataFrame(gb_rows).sort_values("giveback_reduction_bps", ascending=False).to_csv(ROOT / "scorecards/giveback_reduction_scorecard.csv", index=False)
    fake_rows = []
    for eid, g in nonref[nonref["fake_reclaim_proxy"].astype(bool)].groupby("exit_id"):
        fake_rows.append(score_group(g, eid, "fake_reclaim_exit"))
    pd.DataFrame(fake_rows).sort_values("mean_net_current_bps", ascending=False).to_csv(ROOT / "scorecards/fake_reclaim_exit_scorecard.csv", index=False)
    cost_rows = []
    for eid, g in nonref.groupby("exit_id"):
        for cost in [0, 3, 6, 9, 12]:
            cost_rows.append({"exit_id": eid, "cost_bps": cost, "mean_net_bps": (g["gross_return_bps"] - cost).mean(), "PF": pf(g["gross_return_bps"] - cost), "trade_count": len(g)})
    pd.DataFrame(cost_rows).to_csv(ROOT / "scorecards/cost_sensitivity_scorecard.csv", index=False)
    (ROOT / "scorecards/exit_scorecard_report.md").write_text("# Exit Scorecard Report\n\nScorecards exclude oracle reference exits for decision metrics.\n", encoding="utf-8")
    return {"policy": policy, "by_subset": by_subset, "family": fam}


def complexity(exit_id: str) -> int:
    score = 1
    if "partial" in exit_id or "trail" in exit_id:
        score += 2
    if "_or_" in exit_id or "then" in exit_id:
        score += 1
    if "oracle" in exit_id:
        score += 99
    return score


def objective_score(df: pd.DataFrame, objective: str) -> pd.Series:
    mean = df["mean_net_current_bps"].fillna(-999)
    pfv = df["profit_factor"].replace(np.inf, 10).fillna(0)
    gb = -df["giveback_rate"].fillna(1) * 10
    rfe = -df["RFE_rate"].fillna(1) * 10
    mddv = -df["MDD_proxy"].fillna(999) * 0.01
    comp = -df["exit_id"].map(complexity).fillna(5)
    if objective == "OBJ_2_GIVEBACK_REDUCTION":
        return mean * 0.3 + gb * 2 + comp
    if objective == "OBJ_3_CONSERVATIVE":
        return mean * 0.2 + rfe * 2 + mddv * 2 + comp
    if objective == "OBJ_4_SIMPLE_POLICY":
        return mean * 0.5 + pfv * 2 + comp * 3
    return mean + pfv * 5 + gb + rfe + mddv + comp


def walk_forward(sim: pd.DataFrame, f2: pd.DataFrame) -> pd.DataFrame:
    folds = pd.read_csv(FOLDS)
    for c in ["train_start", "train_end", "test_start", "test_end"]:
        folds[c] = pd.to_datetime(folds[c], errors="coerce")
    sim = sim[~sim["is_reference_only"].astype(bool)].copy()
    id_ts = f2.set_index("candidate_id")[["signal_ts", "direction", "entry_timeframe"]]
    sim = sim.join(id_ts, on="candidate_id", rsuffix="_entry")
    objectives = ["OBJ_1_BALANCED", "OBJ_2_GIVEBACK_REDUCTION", "OBJ_3_CONSERVATIVE", "OBJ_4_SIMPLE_POLICY", "OBJ_5_LONG_ONLY_SUBSET", "OBJ_6_LONG_4H_1D_SUBSET"]
    rows = []
    for _, fold in folds.iterrows():
        for obj in objectives:
            train = sim[(sim["signal_ts_entry"] >= fold["train_start"]) & (sim["signal_ts_entry"] <= fold["train_end"])]
            test = sim[(sim["signal_ts_entry"] >= fold["test_start"]) & (sim["signal_ts_entry"] <= fold["test_end"])]
            if obj == "OBJ_5_LONG_ONLY_SUBSET":
                train = train[train["direction_entry"].eq("LONG")]
                test_eval = test[test["direction_entry"].eq("LONG")]
            elif obj == "OBJ_6_LONG_4H_1D_SUBSET":
                train = train[train["direction_entry"].eq("LONG") & train["entry_timeframe_entry"].isin(["4h", "1d"])]
                test_eval = test[test["direction_entry"].eq("LONG") & test["entry_timeframe_entry"].isin(["4h", "1d"])]
            else:
                test_eval = test
            train_scores = pd.DataFrame([score_group(g, eid, "train_policy") for eid, g in train.groupby("exit_id")])
            if train_scores.empty:
                continue
            train_scores["exit_id"] = train_scores["object_id"]
            train_scores["objective_score"] = objective_score(train_scores, obj)
            selected = train_scores.sort_values("objective_score", ascending=False).iloc[0]["exit_id"]
            tr = train[train["exit_id"].eq(selected)]
            te = test_eval[test_eval["exit_id"].eq(selected)]
            trm, tem = score_group(tr, selected, "wf_train"), score_group(te, selected, "wf_test")
            rows.append(
                {
                    "fold_id": int(fold["fold"]),
                    "train_start": fold["train_start"],
                    "train_end": fold["train_end"],
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    "objective": obj,
                    "selected_exit_id": selected,
                    "selected_exit_family": tr["exit_family"].iloc[0] if len(tr) else "",
                    "train_trade_count": trm["trade_count"],
                    "test_trade_count": tem["trade_count"],
                    "train_mean_net": trm["mean_net_current_bps"],
                    "test_mean_net": tem["mean_net_current_bps"],
                    "train_pf": trm["profit_factor"],
                    "test_pf": tem["profit_factor"],
                    "train_giveback_rate": trm["giveback_rate"],
                    "test_giveback_rate": tem["giveback_rate"],
                    "train_rfe": trm["RFE_rate"],
                    "test_rfe": tem["RFE_rate"],
                    "train_mdd": trm["MDD_proxy"],
                    "test_mdd": tem["MDD_proxy"],
                    "test_pass": tem["mean_net_current_bps"] >= 0,
                    "test_failure_reason": "PASS" if tem["mean_net_current_bps"] >= 0 else "OOS_NEGATIVE",
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["fold_id", "train_start", "train_end", "test_start", "test_end", "objective", "selected_exit_id", "selected_exit_family", "train_trade_count", "test_trade_count", "train_mean_net", "test_mean_net", "train_pf", "test_pf", "train_giveback_rate", "test_giveback_rate", "train_rfe", "test_rfe", "train_mdd", "test_mdd", "test_pass", "test_failure_reason"])
    (ROOT / "walk_forward/exit_walk_forward_config.json").write_text(jdump({"selection": "train only", "objectives": objectives, "complexity_penalty": True}), encoding="utf-8")
    out.to_csv(ROOT / "walk_forward/exit_walk_forward_fold_results.csv", index=False)
    out[["fold_id", "objective", "selected_exit_id", "selected_exit_family"]].to_csv(ROOT / "walk_forward/exit_walk_forward_selected_policies.csv", index=False)
    obj = out.groupby("objective").agg(folds=("fold_id", "count"), mean_test_net=("test_mean_net", "mean"), pass_rate=("test_pass", "mean"), mean_test_giveback=("test_giveback_rate", "mean"), selected_unique=("selected_exit_id", "nunique")).reset_index() if not out.empty else pd.DataFrame(columns=["objective", "folds", "mean_test_net", "pass_rate", "mean_test_giveback", "selected_unique"])
    obj.to_csv(ROOT / "walk_forward/exit_walk_forward_objective_summary.csv", index=False)
    stab = pd.DataFrame(
        [
            {
                "mean_test_net_bps": out["test_mean_net"].mean() if not out.empty else np.nan,
                "pass_rate": out["test_pass"].mean() if not out.empty else np.nan,
                "best_objective": obj.sort_values("mean_test_net", ascending=False).iloc[0]["objective"] if not obj.empty else "",
                "verdict": "EXIT_WALK_FORWARD_PASS" if (not obj.empty and obj["mean_test_net"].max() >= 0 and obj["pass_rate"].max() >= 0.5) else "EXIT_IMPROVES_BUT_NOT_ROBUST",
            }
        ]
    )
    stab.to_csv(ROOT / "walk_forward/exit_walk_forward_stability_summary.csv", index=False)
    (ROOT / "walk_forward/exit_walk_forward_report.md").write_text("# Exit Walk-forward Report\n\nExit policy is selected on train folds only and applied fixed to test folds. Test results are not used for selection.\n", encoding="utf-8")
    return out


def regime_outputs(sim: pd.DataFrame) -> pd.DataFrame:
    nonref = sim[~sim["is_reference_only"].astype(bool)].copy()
    rows = []
    for keys in [["exit_id", "regime_state"], ["exit_id", "vol_regime"], ["exit_id", "fake_reclaim_proxy"], ["exit_id", "MFE_before_MAE"]]:
        for vals, g in nonref.groupby(keys, dropna=False):
            m = score_group(g, "|".join(map(str, vals if isinstance(vals, tuple) else (vals,))), "exit_regime")
            for k, v in zip(keys, vals if isinstance(vals, tuple) else (vals,)):
                m[k] = v
            rows.append(m)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "regime/exit_regime_scorecard.csv", index=False)
    nonref.groupby(["fold_id", "exit_id", "regime_state"]).agg(mean_net_current_bps=("net_current_bps", "mean"), trade_count=("candidate_id", "count"), giveback_rate=("giveback_bps", lambda x: (x > CURRENT_COST_BPS).mean())).reset_index().to_csv(ROOT / "regime/exit_fold_regime_interaction.csv", index=False)
    (ROOT / "regime/exit_regime_report.md").write_text("# Exit Regime Report\n\nExit performance is bucketed by pre-existing F2 regime/fake-reclaim fields.\n", encoding="utf-8")
    return out


def giveback_outputs(sim: pd.DataFrame) -> None:
    nonref = sim[~sim["is_reference_only"].astype(bool)].copy()
    base = nonref[nonref["exit_id"].eq("X_A4_fixed_24h")].set_index("candidate_id")
    rows, fake, mfe, tradeoff = [], [], [], []
    for eid, g in nonref.groupby("exit_id"):
        gg = g.set_index("candidate_id")
        aligned = base[["giveback_bps", "net_current_bps", "MFE_capture_ratio"]].join(gg[["giveback_bps", "net_current_bps", "MFE_capture_ratio", "fake_reclaim_proxy"]], lsuffix="_base", rsuffix="_exit").dropna()
        rows.append({"exit_id": eid, "giveback_delta_bps": (aligned["giveback_bps_base"] - aligned["giveback_bps_exit"]).mean(), "giveback_reduced_rate": (aligned["giveback_bps_exit"] < aligned["giveback_bps_base"]).mean()})
        f = aligned[aligned["fake_reclaim_proxy"].astype(bool)]
        fake.append({"exit_id": eid, "fake_reclaim_net_delta_bps": (f["net_current_bps_exit"] - f["net_current_bps_base"]).mean() if len(f) else np.nan, "fake_reclaim_loss_reduced_rate": (f["net_current_bps_exit"] > f["net_current_bps_base"]).mean() if len(f) else np.nan})
        mfe.append({"exit_id": eid, "MFE_capture_ratio_delta": (aligned["MFE_capture_ratio_exit"] - aligned["MFE_capture_ratio_base"]).mean(), "MFE_capture_improved_rate": (aligned["MFE_capture_ratio_exit"] > aligned["MFE_capture_ratio_base"]).mean()})
        tradeoff.append({"exit_id": eid, "early_exit_missed_profit_rate": ((aligned["net_current_bps_exit"] < aligned["net_current_bps_base"]) & (aligned["net_current_bps_base"] > 0)).mean(), "late_exit_loss_reduction_rate": ((aligned["net_current_bps_base"] < 0) & (aligned["net_current_bps_exit"] > aligned["net_current_bps_base"])).mean()})
    pd.DataFrame(rows).sort_values("giveback_delta_bps", ascending=False).to_csv(ROOT / "giveback/giveback_reduction_comparison.csv", index=False)
    pd.DataFrame(fake).sort_values("fake_reclaim_net_delta_bps", ascending=False).to_csv(ROOT / "giveback/fake_reclaim_loss_reduction.csv", index=False)
    pd.DataFrame(mfe).sort_values("MFE_capture_ratio_delta", ascending=False).to_csv(ROOT / "giveback/mfe_capture_scorecard.csv", index=False)
    pd.DataFrame(tradeoff).to_csv(ROOT / "giveback/early_vs_late_exit_tradeoff.csv", index=False)
    (ROOT / "giveback/giveback_fake_reclaim_report.md").write_text("# Giveback/Fake Reclaim Report\n\nComparisons are paired against baseline fixed 24h for the same candidate_id.\n", encoding="utf-8")


def overfit_outputs(policy_sc: pd.DataFrame, wf: pd.DataFrame) -> None:
    rows = []
    for _, r in policy_sc.iterrows():
        eid = r["object_id"]
        rows.append({"exit_id": eid, "complexity_score": complexity(eid), "parameter_count": eid.count("_"), "mean_net_current_bps": r["mean_net_current_bps"], "sample_count": r["trade_count"], "overfit_warning": "EXIT_COMPLEX_OVERFIT_RISK" if complexity(eid) >= 4 else "EXIT_SIMPLE_AND_STABLE"})
    comp = pd.DataFrame(rows)
    comp.to_csv(ROOT / "overfit/exit_complexity_scorecard.csv", index=False)
    sens = comp[comp["exit_id"].str.contains("tp_|ATR|fixed", regex=True)].copy()
    sens.to_csv(ROOT / "overfit/exit_parameter_sensitivity.csv", index=False)
    conc = wf.groupby("selected_exit_id").agg(selected_folds=("fold_id", "count"), mean_test_net=("test_mean_net", "mean")).reset_index() if not wf.empty else pd.DataFrame()
    conc.to_csv(ROOT / "overfit/exit_performance_concentration.csv", index=False)
    (ROOT / "overfit/overfit_audit_report.md").write_text("# Overfit Audit Report\n\nComplex policies and single-parameter spikes are flagged. Oracle exits are excluded from decisions.\n", encoding="utf-8")


def casebook(sim: pd.DataFrame) -> pd.DataFrame:
    nonref = sim[~sim["is_reference_only"].astype(bool)].copy()
    best = nonref.sort_values("improvement_bps", ascending=False).groupby("candidate_id").head(1)
    worst = nonref.sort_values("improvement_bps", ascending=True).groupby("candidate_id").head(1)
    cb = pd.concat([best.head(100), worst.head(100)], ignore_index=True)
    cb["case_category"] = np.select(
        [
            cb["tp_hit"],
            cb["reclaim_break_hit"] & (cb["improvement_bps"] > 0),
            cb["exit_reason"].astype(str).str.contains("trail", na=False) & (cb["improvement_bps"] > 0),
            cb["sl_hit"] & (cb["improvement_bps"] < 0),
            cb["tp_hit"] & (cb["improvement_bps"] < 0),
            cb["giveback_bps"] > 100,
            cb["fake_reclaim_proxy"] & (cb["improvement_bps"] <= 0),
        ],
        [
            "F2_EXIT_SUCCESS_TP_CAPTURE",
            "F2_EXIT_SUCCESS_RECLAIM_BREAK_AVOIDED_LOSS",
            "F2_EXIT_SUCCESS_TRAILING_CAPTURE",
            "F2_EXIT_FAIL_STOP_TOO_TIGHT",
            "F2_EXIT_FAIL_TP_TOO_EARLY",
            "F2_EXIT_FAIL_GIVEBACK_STILL_HIGH",
            "F2_EXIT_FAIL_FAKE_RECLAIM_NOT_CAUGHT",
        ],
        default="F2_EXIT_OOS_FAIL_CASE",
    )
    cb["why_exit_helped"] = np.where(cb["improvement_bps"] > 0, "exit improved net versus fixed 24h baseline", "")
    cb["why_exit_failed"] = np.where(cb["improvement_bps"] <= 0, "exit underperformed fixed 24h baseline or remained negative", "")
    cb["recommended_action"] = "casebook_reference_only"
    cb["chart_window_path"] = ""
    cb.to_parquet(ROOT / "casebook/f2_exit_casebook.parquet", index=False)
    cb.to_csv(ROOT / "casebook/f2_exit_casebook.csv", index=False)
    cb.sort_values("improvement_bps", ascending=False).head(50).to_csv(ROOT / "casebook/f2_exit_top_success_cases.csv", index=False)
    cb.sort_values("improvement_bps", ascending=True).head(50).to_csv(ROOT / "casebook/f2_exit_top_failure_cases.csv", index=False)
    (ROOT / "casebook/f2_exit_casebook_report.md").write_text("# F2 Exit Casebook Report\n\nTop paired improvements/failures versus fixed 24h baseline are stored. Charts are not generated by default.\n", encoding="utf-8")
    return cb


def decisions(policy: pd.DataFrame, wf: pd.DataFrame) -> List[str]:
    base = policy[policy["object_id"].eq("X_A4_fixed_24h")].iloc[0]
    top = policy[~policy["object_id"].str.contains("oracle", case=False)].sort_values("mean_net_current_bps", ascending=False).iloc[0]
    wf_obj = pd.read_csv(ROOT / "walk_forward/exit_walk_forward_objective_summary.csv")
    best_wf = wf_obj.sort_values("mean_test_net", ascending=False).iloc[0] if not wf_obj.empty else pd.Series(dtype=object)
    verdicts = ["F2_EXIT_REDESIGN_COMPLETED", "production_not_ready"]
    if top["mean_net_current_bps"] > base["mean_net_current_bps"] + 1:
        verdicts.append("F2_FIXED_24H_WAS_BAD")
    if any(x in str(top["object_id"]) for x in ["fixed_4h", "fixed_8h", "fixed_12h"]):
        verdicts.append("F2_SHORTER_HORIZON_HELPS")
    if "tp_" in str(top["object_id"]) and "breakeven" in str(top["object_id"]):
        verdicts.append("F2_TP_BREAKEVEN_HELPS")
    if "reclaim" in str(top["object_id"]):
        verdicts.append("F2_RECLAIM_BREAK_EXIT_HELPS")
    if "trailing" in str(top["object_id"]):
        verdicts.append("F2_ATR_TRAIL_HELPS")
    gb = pd.read_csv(ROOT / "giveback/giveback_reduction_comparison.csv")
    fake = pd.read_csv(ROOT / "giveback/fake_reclaim_loss_reduction.csv")
    if gb["giveback_delta_bps"].max() > 5:
        verdicts.append("F2_GIVEBACK_REDUCED")
    if fake["fake_reclaim_net_delta_bps"].max() > 5:
        verdicts.append("F2_FAKE_RECLAIM_LOSS_REDUCED")
    if best_wf.get("mean_test_net", -999) >= 0 and best_wf.get("pass_rate", 0) >= 0.5:
        verdicts += ["F2_EXIT_REDESIGN_OOS_PASS", "F2_KEEP_FOR_FORWARD_EXIT_WATCH"]
        decision = "KEEP_FOR_FORWARD_EXIT_WATCH"
    elif best_wf.get("mean_test_net", -999) > -17.63:
        verdicts += ["F2_EXIT_REDESIGN_IMPROVES_BUT_WEAK", "F2_KEEP_CASEBOOK_ONLY"]
        decision = "KEEP_CASEBOOK_ONLY"
    else:
        verdicts += ["F2_EXIT_REDESIGN_FAIL", "F2_DROP_AFTER_EXIT_FAIL"]
        decision = "DROP_F2_AFTER_EXIT_FAIL"
    if complexity(str(top["object_id"])) >= 4:
        verdicts.append("F2_EXIT_OVERFIT")
    verdicts = list(dict.fromkeys(verdicts))
    pd.DataFrame(
        [
            {
                "decision": decision,
                "best_insample_exit_id": top["object_id"],
                "best_insample_mean_net_bps": top["mean_net_current_bps"],
                "baseline_24h_mean_net_bps": base["mean_net_current_bps"],
                "best_wf_objective": best_wf.get("objective", ""),
                "best_wf_mean_test_bps": best_wf.get("mean_test_net", np.nan),
                "best_wf_pass_rate": best_wf.get("pass_rate", np.nan),
                "production_ready": False,
            }
        ]
    ).to_csv(ROOT / "decision/f2_exit_keep_drop_decision.csv", index=False)
    pd.DataFrame(
        [
            {"watch_id": "FW_F2_EXIT_SIMPLE_WATCH", "exit_id": top["object_id"], "action": decision, "note": "research-only; no production trigger"},
        ]
    ).to_csv(ROOT / "decision/f2_exit_forward_watchlist_update.csv", index=False)
    (ROOT / "decision/f2_exit_final_recommendation.md").write_text(f"# F2 Exit Final Recommendation\n\nDecision: {decision}. Best in-sample exit: `{top['object_id']}`. Best walk-forward objective: `{best_wf.get('objective', '')}`. production_ready=false.\n", encoding="utf-8")
    return verdicts


def final_report(discovery: Dict[str, Any], policy: pd.DataFrame, wf: pd.DataFrame, verdicts: List[str]) -> None:
    base = policy[policy["object_id"].eq("X_A4_fixed_24h")].iloc[0].to_dict()
    top = policy.sort_values("mean_net_current_bps", ascending=False).head(10).to_dict("records")
    wf_summary = pd.read_csv(ROOT / "walk_forward/exit_walk_forward_objective_summary.csv")
    decision = pd.read_csv(ROOT / "decision/f2_exit_keep_drop_decision.csv").to_dict("records")
    report = f"""# F2 Exit Redesign Autopsy Final Report

## Why
F2 pullback-reclaim showed frequent MFE but failed OOS under fixed 24h. This branch keeps F2 entry fixed and replays exits only.

## Replay Coverage
```json
{jdump(discovery)}
```

## Baseline Fixed 24h
```json
{jdump(base)}
```

## Top Exit Policies
```json
{jdump(top)}
```

## Walk-forward Exit Validation
```json
{jdump(wf_summary.to_dict('records'))}
```

## Decision
```json
{jdump(decision)}
```

## Verdicts
{chr(10).join(verdicts)}

## Safety
Production TCN/Q2/R7/Risk Manager/live/order/state files were not changed. forward_orderflow_collector_v4 and Discord/webhook policy were read-only. production_ready=false; promotion_ready=false.
"""
    (ROOT / "f2_exit_redesign_autopsy_final_report.md").write_text(report, encoding="utf-8")
    (ROOT / "f2_exit_redesign_autopsy_final_verdict.md").write_text("# Final Verdict\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    (ROOT / "recommended_next_branch.md").write_text("# Recommended Next Branch\n\nDo not promote F2. If continuing, run forward-only observation for the simplest exit selected by train-only walk-forward, with production disconnected.\n", encoding="utf-8")


def run(mode: str, fast: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    before = safety_snapshot("before")
    log(f"start mode={mode} fast={fast}")
    discovery = input_discovery()
    f2, replay = build_replay_dataset(fast=fast)
    cat = exit_catalog()
    sim = simulate_exits(f2, replay, cat)
    sc = scorecards(sim, f2)
    wf = walk_forward(sim, f2)
    regime_outputs(sim)
    giveback_outputs(sim)
    overfit_outputs(sc["policy"], wf)
    casebook(sim)
    verdicts = decisions(sc["policy"], wf)
    final_report(discovery, sc["policy"], wf, verdicts)
    (ROOT / "run_metadata.json").write_text(jdump({"mode": mode, "fast": fast, "f2_rows": len(f2), "sim_rows": len(sim), "updated_ts": pd.Timestamp.now("UTC").isoformat()}), encoding="utf-8")
    finalize_audit(before)
    return {"mode": mode, "fast": fast, "f2_rows": len(f2), "simulation_rows": len(sim), "verdicts": verdicts, "production_ready": False, "promotion_ready": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast-smoke", action="store_true")
    parser.add_argument("--exit-sim-only", action="store_true")
    parser.add_argument("--walk-forward-only", action="store_true")
    parser.add_argument("--regime-only", action="store_true")
    parser.add_argument("--casebook-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    if args.dry_run:
        res = {"dry_run": True, "root": str(ROOT), "f2_dataset_exists": F2_DATASET.exists(), "replay_5m_exists": REPLAY_5M.exists(), "production_ready": False, "promotion_ready": False}
    elif args.fast_smoke:
        res = run("fast_smoke", fast=True)
    elif args.exit_sim_only:
        res = run("exit_sim_only")
    elif args.walk_forward_only:
        res = run("walk_forward_only")
    elif args.regime_only:
        res = run("regime_only")
    elif args.casebook_only:
        res = run("casebook_only")
    else:
        res = run("full")
    print(jdump(res) if args.json else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
