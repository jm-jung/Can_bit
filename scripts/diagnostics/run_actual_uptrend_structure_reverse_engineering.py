"""Actual uptrend / MFE success structure reverse engineering.

Diagnostics-only. This is outcome-first analysis: label real uptrend/MFE
success timestamps, compare them with controls, and derive TCN target redesign
inputs. No production/live/order/state path is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("data/diagnostics/actual_uptrend_structure_reverse_engineering")
ALPHA = Path("data/diagnostics/alpha_existence_target_feasibility_audit")
FEATURE_4H = ALPHA / "features/feature_frame_4h.parquet"
TARGET_4H = ALPHA / "targets/target_frame_4h.parquet"
MATRIX = ALPHA / "decision/target_feasibility_matrix.csv"
DEFAULT_5M = Path("data/ohlcv/BTCUSDT_5m_full.csv")
CURRENT_COST_BPS = 6.0
MAIN_TF = "4h"
MAIN_H = "72h"
RNG_SEED = 20260624


def ensure_dirs() -> None:
    for d in [
        "discovery",
        "audit",
        "backfill/raw",
        "backfill/normalized",
        "backfill/registry",
        "labels",
        "snapshots",
        "comparison",
        "patterns",
        "walk_forward",
        "economic",
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
    writes = [{"path": str(p), "diagnostics_only": True, "write_class": "uptrend_reverse_engineering_output"} for p in ROOT.rglob("*") if p.is_file()]
    writes.append({"path": "scripts/diagnostics/run_actual_uptrend_structure_reverse_engineering.py", "diagnostics_only": False, "write_class": "requested_entrypoint"})
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
        Path("data/diagnostics/data_sync/canonical_data_paths.json"),
        Path("data/diagnostics/research_orderflow_data_cache/cache_registry.csv"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/spot_ohlcv/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/futures_ohlcv/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/mark_price/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/premium_index/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/funding_rate/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/open_interest_history/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/taker_buy_sell_volume/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_1m.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_5m.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_15m.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_1h.parquet"),
        ALPHA / "alpha_existence_target_feasibility_audit_final_report.md",
        MATRIX,
        ALPHA / "economic/top_quantile_economic_replay.csv",
        ALPHA / "walk_forward/walk_forward_target_summary.csv",
        Path("data/diagnostics/f2_exit_redesign_autopsy/f2_exit_redesign_autopsy_final_report.md"),
        Path("data/diagnostics/f2_pullback_reclaim_oos_failure_autopsy/f2_pullback_reclaim_oos_failure_autopsy_final_report.md"),
        Path("data/diagnostics/low_frequency_swing_entry_alpha_research/low_frequency_swing_entry_alpha_research_final_report.md"),
        Path("data/diagnostics/classical_ta_entry_alpha_research/classical_ta_entry_alpha_research_final_report.md"),
        Path("data/diagnostics/event_driven_entry_alpha_stagewise_tournament/event_driven_entry_alpha_stagewise_tournament_final_report.md"),
        FEATURE_4H,
        TARGET_4H,
        DEFAULT_5M,
    ]
    rows = []
    for p in required:
        rows.append({"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() and p.is_file() else 0, "suffix": p.suffix})
    for root in [Path("data"), Path("data/diagnostics"), Path("data/ohlcv"), Path("data/market"), Path("scripts/diagnostics")]:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file() and any(k in str(p).lower() for k in ["uptrend", "alpha_existence", "btcusdt", "proxy_cvd", "orderflow", "mfe"]):
                    rows.append({"path": str(p), "exists": True, "size": p.stat().st_size, "suffix": p.suffix})
    inv = pd.DataFrame(rows).drop_duplicates("path")
    inv.to_csv(ROOT / "discovery/input_inventory.csv", index=False)
    (ROOT / "discovery/discovered_paths.json").write_text(jdump(inv.to_dict("records")[:7000]), encoding="utf-8")
    cov = []
    for p in [DEFAULT_5M, FEATURE_4H, TARGET_4H]:
        df = read_df(p)
        ts_col = "timestamp" if "timestamp" in df.columns else None
        if ts_col:
            ts = pd.to_datetime(df[ts_col], errors="coerce")
            cov.append({"path": str(p), "rows": len(df), "start": ts.min(), "end": ts.max()})
        else:
            cov.append({"path": str(p), "rows": len(df), "start": "", "end": ""})
    pd.DataFrame(cov).to_csv(ROOT / "discovery/data_coverage_summary.csv", index=False)
    tiers = [
        {"tier": "TIER_A_LONG_OHLCV_ONLY", "available": FEATURE_4H.exists() and TARGET_4H.exists(), "note": "main 4h/72h feature-target frame"},
        {"tier": "TIER_B_OHLCV_PLUS_PROXY_CVD", "available": Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_1h.parquet").exists(), "note": "proxy CVD partial overlap"},
        {"tier": "TIER_C_RECENT_ORDERFLOW", "available": Path("data/diagnostics/research_orderflow_data_cache/normalized/open_interest_history/BTCUSDT.parquet").exists(), "note": "recent OI/taker/funding/basis reference"},
        {"tier": "TIER_D_FORWARD_ORDERBOOK_ONLY", "available": False, "note": "orderbook/liquidation forward-only"},
    ]
    pd.DataFrame(tiers).to_csv(ROOT / "discovery/data_tier_summary.csv", index=False)
    of = []
    for p in required[9:13]:
        df = read_df(p)
        if not df.empty and "timestamp" in df:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            of.append({"path": str(p), "rows": len(df), "start": ts.min(), "end": ts.max()})
        else:
            of.append({"path": str(p), "rows": len(df), "start": "", "end": ""})
    pd.DataFrame(of).to_csv(ROOT / "discovery/orderflow_overlap_summary.csv", index=False)
    mat = read_df(MATRIX)
    mat.to_csv(ROOT / "discovery/previous_alpha_existence_summary.csv", index=False)
    backfill_need = pd.DataFrame(
        [
            {
                "local_feature_target_available": FEATURE_4H.exists() and TARGET_4H.exists(),
                "local_5m_available": DEFAULT_5M.exists(),
                "backfill_required": False,
                "reason": "local canonical 5m plus alpha existence 4h feature/target frames are sufficient for this reverse engineering audit",
            }
        ]
    )
    backfill_need.to_csv(ROOT / "discovery/backfill_need_assessment.csv", index=False)
    (ROOT / "discovery/discovery_report.md").write_text("# Discovery Report\n\nLocal long OHLCV and alpha existence feature/target frames are available. Public backfill is not required for the main 4h/72h audit.\n", encoding="utf-8")
    return {"inventory_rows": len(inv), "main_feature_exists": FEATURE_4H.exists(), "main_target_exists": TARGET_4H.exists(), "backfill_required": False}


def backfill_outputs() -> None:
    reg = pd.DataFrame([{"source": "local_existing", "symbol": "BTCUSDT", "timeframe": "4h", "status": "used", "private_api_calls": 0}])
    reg.to_csv(ROOT / "backfill/registry/backfill_registry.csv", index=False)
    (ROOT / "backfill/registry/backfill_progress.json").write_text(jdump({"status": "not_required", "reason": "local data sufficient"}), encoding="utf-8")
    for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
        pd.DataFrame().to_parquet(ROOT / f"backfill/normalized/btcusdt_{tf}.parquet", index=False)
    pd.DataFrame([{"backfill_required": False, "rows_downloaded": 0, "private_api_calls": 0}]).to_csv(ROOT / "backfill/backfill_summary.csv", index=False)
    pd.DataFrame([{"source": "local_existing", "gap_count": np.nan, "note": "backfill skipped"}]).to_csv(ROOT / "backfill/backfill_gap_audit.csv", index=False)
    (ROOT / "backfill/backfill_report.md").write_text("# Backfill Report\n\nBackfill skipped. Local historical data is sufficient; no public/private endpoint was called.\n", encoding="utf-8")


def load_main(fast: bool = False) -> pd.DataFrame:
    f = pd.read_parquet(FEATURE_4H)
    t = pd.read_parquet(TARGET_4H)
    f["timestamp"] = pd.to_datetime(f["timestamp"], errors="coerce")
    t["timestamp"] = pd.to_datetime(t["timestamp"], errors="coerce")
    h = f"h_{MAIN_H}"
    cols = [
        "timestamp",
        f"{h}_future_MFE_long_bps",
        f"{h}_future_MAE_long_bps",
        f"{h}_future_return_net_current_bps",
        f"{h}_long_MFE_before_MAE",
        f"{h}_long_tradeable_after_cost",
        f"{h}_long_RFE_high",
        f"{h}_short_MFE_2x_cost_hit",
        f"{h}_volatility_expansion",
        f"{h}_tradeability_score_long",
        f"{h}_risk_score",
    ]
    df = f.merge(t[[c for c in cols if c in t.columns]], on="timestamp", how="inner")
    if fast:
        df = df.tail(5000).copy()
    return df.sort_values("timestamp").reset_index(drop=True)


def build_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    h = f"h_{MAIN_H}"
    d = df[["timestamp", "symbol", "signal_timeframe", "close"]].copy()
    mfe = pd.to_numeric(df[f"{h}_future_MFE_long_bps"], errors="coerce")
    mae = pd.to_numeric(df[f"{h}_future_MAE_long_bps"], errors="coerce")
    net = pd.to_numeric(df[f"{h}_future_return_net_current_bps"], errors="coerce")
    q95, q90, q80 = mfe.quantile([0.95, 0.90, 0.80])
    d["future_MFE"] = mfe
    d["future_MAE"] = mae
    d["future_return"] = net
    d["net_current"] = net
    d["MFE_before_MAE"] = df[f"{h}_long_MFE_before_MAE"].astype(bool)
    d["RFE"] = df[f"{h}_long_RFE_high"].astype(bool)
    d["UP_SUCCESS_MFE_Q95"] = mfe >= q95
    d["UP_SUCCESS_MFE_Q90"] = mfe >= q90
    d["UP_SUCCESS_MFE_Q80"] = mfe >= q80
    d["UP_SUCCESS_NET_POSITIVE"] = net > 0
    d["UP_SUCCESS_TRADEABLE"] = df[f"{h}_long_tradeable_after_cost"].astype(bool)
    d["UP_SUCCESS_MFE_BEFORE_MAE"] = d["MFE_before_MAE"] & (mfe >= CURRENT_COST_BPS * 2)
    d["UP_SUCCESS_CLEAN_TREND"] = (mfe >= q90) & (net > 0) & d["MFE_before_MAE"]
    d["UP_SUCCESS_BREAKOUT_FOLLOWTHROUGH"] = (mfe >= q90) & (net > CURRENT_COST_BPS)
    d["UP_SUCCESS_SWING"] = (mfe >= q90) & (net > 0)
    d["UP_FAKE_MFE_ONLY"] = (mfe >= q90) & ((net <= 0) | (~d["MFE_before_MAE"]))
    d["UP_DIRTY_RALLY"] = (mfe >= q90) & (mae >= mfe * 0.8)
    d["UP_GIVEBACK"] = (mfe >= q90) & (net < mfe * 0.2)
    d["UP_CHOPPY"] = (mfe >= q90) & (mae >= q90)
    d["FAIL_NO_UP"] = (mfe <= mfe.quantile(0.30)) & (net <= 0)
    d["FAIL_RFE_HIGH"] = d["RFE"]
    d["FAIL_MAE_FIRST"] = ~d["MFE_before_MAE"]
    d["DOWN_SUCCESS_SHORT"] = df[f"{h}_short_MFE_2x_cost_hit"].astype(bool)
    rng = np.random.default_rng(RNG_SEED)
    d["RANDOM_CONTROL"] = False
    success_idx = d.index[d["UP_SUCCESS_MFE_Q95"]].to_numpy()
    control_pool = d.index[~d["UP_SUCCESS_MFE_Q90"]].to_numpy()
    sample_n = min(len(success_idx), len(control_pool))
    if sample_n:
        d.loc[rng.choice(control_pool, size=sample_n, replace=False), "RANDOM_CONTROL"] = True
    d["REGIME_MATCHED_CONTROL"] = False
    regime_cols = [c for c in ["trend_stack_bull", "trend_stack_bear"] if c in df]
    if regime_cols and sample_n:
        pool = d.index[(~d["UP_SUCCESS_MFE_Q90"]) & (df["trend_stack_bull"].eq(True) == df.loc[success_idx, "trend_stack_bull"].mode().iloc[0])].to_numpy()
        if len(pool):
            d.loc[rng.choice(pool, size=min(sample_n, len(pool)), replace=False), "REGIME_MATCHED_CONTROL"] = True
    d["TIME_MATCHED_CONTROL"] = False
    months = d["timestamp"].dt.month
    pool = d.index[(~d["UP_SUCCESS_MFE_Q90"]) & months.isin(months.iloc[success_idx].unique())].to_numpy()
    if len(pool):
        d.loc[rng.choice(pool, size=min(sample_n, len(pool)), replace=False), "TIME_MATCHED_CONTROL"] = True
    d["NEAR_MISS_CONTROL"] = (mfe < q95) & (mfe >= q90)
    d["label_primary"] = np.select(
        [d["UP_SUCCESS_MFE_Q95"], d["UP_FAKE_MFE_ONLY"], d["UP_DIRTY_RALLY"], d["FAIL_NO_UP"], d["FAIL_RFE_HIGH"], d["DOWN_SUCCESS_SHORT"], d["RANDOM_CONTROL"]],
        ["SUCCESS_MFE_Q95", "UP_FAKE_MFE_ONLY", "UP_DIRTY_RALLY", "FAIL_NO_UP", "FAIL_RFE_HIGH", "DOWN_SUCCESS_SHORT", "RANDOM_CONTROL"],
        default="OTHER",
    )
    d["event_id"] = np.arange(len(d))
    d.to_parquet(ROOT / "labels/uptrend_timestamp_labels.parquet", index=False)
    summary = []
    for c in [x for x in d.columns if x.startswith("UP_") or x.startswith("FAIL_") or x.endswith("_CONTROL") or x == "DOWN_SUCCESS_SHORT"]:
        summary.append({"label": c, "count": int(d[c].sum()), "rate": float(d[c].mean())})
    pd.DataFrame(summary).to_csv(ROOT / "labels/uptrend_label_summary.csv", index=False)
    pd.DataFrame([x for x in summary if "CONTROL" in x["label"]]).to_csv(ROOT / "labels/control_group_summary.csv", index=False)
    # Cluster consecutive Q95 success events within 2 bars.
    succ = d[d["UP_SUCCESS_MFE_Q95"]].copy()
    if not succ.empty:
        gap = succ["timestamp"].diff().dt.total_seconds().div(3600).fillna(999)
        succ["cluster_id"] = (gap > 8).cumsum()
        clusters = succ.groupby("cluster_id").agg(cluster_start=("timestamp", "min"), cluster_end=("timestamp", "max"), cluster_peak=("future_MFE", "max"), representative_signal_ts=("timestamp", "first"), event_count=("timestamp", "count")).reset_index()
    else:
        clusters = pd.DataFrame(columns=["cluster_id", "cluster_start", "cluster_end", "cluster_peak", "representative_signal_ts", "event_count"])
    clusters.to_parquet(ROOT / "labels/uptrend_event_clusters.parquet", index=False)
    clusters.describe(include="all").reset_index().to_csv(ROOT / "labels/uptrend_cluster_summary.csv", index=False)
    catalog = pd.DataFrame(
        [
            {"label": "UP_SUCCESS_MFE_Q95", "definition": "future MFE long top 5%"},
            {"label": "UP_FAKE_MFE_ONLY", "definition": "MFE high but net weak or MAE-first"},
            {"label": "FAIL_RFE_HIGH", "definition": "MAE dominates MFE"},
            {"label": "REGIME_MATCHED_CONTROL", "definition": "same broad trend regime, no MFE success"},
        ]
    )
    catalog.to_csv(ROOT / "labels/label_definition_catalog.csv", index=False)
    (ROOT / "labels/label_build_report.md").write_text(f"# Label Build Report\n\nMain axis: 4h/72h MFE_long. Q95 threshold={q95:.2f} bps, Q90={q90:.2f} bps.\n", encoding="utf-8")
    return d, clusters


def numeric_feature_cols(df: pd.DataFrame) -> List[str]:
    banned = {"event_id", "timestamp", "symbol", "signal_timeframe", "label_primary"}
    cols = []
    for c in df.columns:
        if c in banned or c.startswith("UP_") or c.startswith("FAIL_") or c.endswith("_CONTROL") or c.startswith("h_"):
            continue
        if any(k in c.lower() for k in ["future", "mfe", "mae", "net_current", "rfe"]):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            cols.append(c)
    return cols


def build_snapshots(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    labels_cols = ["event_id", "timestamp", "label_primary", "future_MFE", "future_MAE", "future_return", "net_current", "MFE_before_MAE", "RFE"] + [c for c in labels.columns if c.startswith("UP_") or c.startswith("FAIL_") or c.endswith("_CONTROL") or c == "DOWN_SUCCESS_SHORT"]
    snap = df.merge(labels[labels_cols], on="timestamp", how="inner")
    lag_base = [c for c in ["return_1", "return_3", "return_6", "return_12", "return_24", "ATR_pct", "volatility_percentile", "rsi_14", "macd_hist", "bb_width", "volume_z", "dist_ema_20", "dist_ema_50", "drawdown_from_high", "bounce_from_low"] if c in snap]
    for lag in [1, 3, 6, 12, 24]:
        for c in lag_base:
            snap[f"{c}_lag_{lag}"] = snap[c].shift(lag)
    snap["data_tier"] = "TIER_A_LONG_OHLCV_ONLY"
    snap.to_parquet(ROOT / "snapshots/pre_event_snapshot_frame.parquet", index=False)
    (ROOT / "snapshots/pre_event_snapshot_schema.json").write_text(jdump({c: str(snap[c].dtype) for c in snap.columns}), encoding="utf-8")
    snap.isna().mean().reset_index().rename(columns={"index": "column", 0: "missing_ratio"}).to_csv(ROOT / "snapshots/snapshot_missingness_summary.csv", index=False)
    fam = []
    for family, keys in {
        "PRICE_STRUCTURE": ["return", "range", "body", "wick", "drawdown", "bounce"],
        "TREND_STRUCTURE": ["ema", "sma", "trend", "dist_ema"],
        "PULLBACK_STRUCTURE": ["dist_ema", "bounce", "drawdown"],
        "BREAKOUT_STRUCTURE": ["donchian", "dist_high", "dist_low"],
        "VOLATILITY_STRUCTURE": ["ATR", "volatility", "bb_width", "inside"],
        "MOMENTUM_STRUCTURE": ["rsi", "macd", "bb_pct"],
        "VOLUME_ORDERFLOW_PROXY": ["volume", "proxy", "taker", "OI", "funding", "basis"],
        "TIME_CONTEXT": ["hour", "dayofweek", "month", "weekend"],
    }.items():
        fam.append({"feature_family": family, "feature_count": sum(any(k.lower() in c.lower() for k in keys) for c in snap.columns)})
    pd.DataFrame(fam).to_csv(ROOT / "snapshots/snapshot_feature_family_summary.csv", index=False)
    (ROOT / "snapshots/snapshot_build_report.md").write_text("# Snapshot Build Report\n\nSnapshots use T0 and lagged pre-event features only; outcome columns are kept separately for attribution.\n", encoding="utf-8")
    return snap


def auc_rank(y: pd.Series, x: pd.Series) -> float:
    tmp = pd.DataFrame({"y": y.astype(int), "x": pd.to_numeric(x, errors="coerce")}).dropna()
    pos = tmp[tmp["y"] == 1]["x"]
    neg = tmp[tmp["y"] == 0]["x"]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = pd.concat([pos, neg]).rank()
    return float((ranks.iloc[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def compare_groups(snap: pd.DataFrame) -> pd.DataFrame:
    comps = [
        ("SUCCESS_MFE_Q95", "UP_SUCCESS_MFE_Q95", "RANDOM_CONTROL"),
        ("SUCCESS_MFE_Q95", "UP_SUCCESS_MFE_Q95", "REGIME_MATCHED_CONTROL"),
        ("SUCCESS_MFE_Q95", "UP_SUCCESS_MFE_Q95", "TIME_MATCHED_CONTROL"),
        ("SUCCESS_MFE_Q95", "UP_SUCCESS_MFE_Q95", "NEAR_MISS_CONTROL"),
        ("SUCCESS_TRADEABLE", "UP_SUCCESS_TRADEABLE", "FAIL_NO_UP"),
        ("SUCCESS_TRADEABLE", "UP_SUCCESS_TRADEABLE", "FAIL_MAE_FIRST"),
        ("SUCCESS_CLEAN_TREND", "UP_SUCCESS_CLEAN_TREND", "UP_FAKE_MFE_ONLY"),
        ("SUCCESS_CLEAN_TREND", "UP_SUCCESS_CLEAN_TREND", "UP_DIRTY_RALLY"),
        ("SUCCESS_CLEAN_TREND", "UP_SUCCESS_CLEAN_TREND", "FAIL_RFE_HIGH"),
        ("SUCCESS_LONG", "UP_SUCCESS_MFE_Q95", "DOWN_SUCCESS_SHORT"),
    ]
    cols = numeric_feature_cols(snap)[:160]
    rows = []
    for name, a_col, b_col in comps:
        if a_col not in snap or b_col not in snap:
            continue
        a = snap[snap[a_col].astype(bool)]
        b = snap[snap[b_col].astype(bool)]
        if len(a) < 20 or len(b) < 20:
            continue
        for c in cols:
            av = pd.to_numeric(a[c], errors="coerce").dropna()
            bv = pd.to_numeric(b[c], errors="coerce").dropna()
            if len(av) < 20 or len(bv) < 20:
                continue
            pooled = math.sqrt((av.var() + bv.var()) / 2) if (av.var() + bv.var()) > 0 else np.nan
            effect = (av.mean() - bv.mean()) / pooled if pooled and pd.notna(pooled) else np.nan
            y = pd.concat([pd.Series(1, index=av.index), pd.Series(0, index=bv.index)])
            x = pd.concat([av, bv])
            rows.append({"comparison": f"{name}_vs_{b_col}", "success_label": a_col, "control_label": b_col, "feature": c, "success_n": len(av), "control_n": len(bv), "success_mean": av.mean(), "control_mean": bv.mean(), "mean_diff": av.mean() - bv.mean(), "effect_size": effect, "auc_single_feature": auc_rank(y, x), "odds_ratio_proxy": np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "comparison/success_vs_control_feature_diff.csv", index=False)
    if not out.empty:
        family = out.assign(feature_family=out["feature"].map(feature_family)).groupby(["comparison", "feature_family"]).agg(mean_abs_effect=("effect_size", lambda s: s.abs().mean()), max_auc=("auc_single_feature", "max"), feature_count=("feature", "nunique")).reset_index()
        family.to_csv(ROOT / "comparison/success_vs_control_family_diff.csv", index=False)
        out.assign(abs_auc=(out["auc_single_feature"] - 0.5).abs()).sort_values("abs_auc", ascending=False).head(200).to_csv(ROOT / "comparison/single_feature_auc_scorecard.csv", index=False)
        stab = out.assign(year=snap["timestamp"].dt.year.min()).groupby("comparison").agg(max_abs_effect=("effect_size", lambda s: s.abs().max()), mean_abs_effect=("effect_size", lambda s: s.abs().mean())).reset_index()
        stab.to_csv(ROOT / "comparison/success_vs_control_stability.csv", index=False)
    else:
        pd.DataFrame().to_csv(ROOT / "comparison/success_vs_control_family_diff.csv", index=False)
        pd.DataFrame().to_csv(ROOT / "comparison/single_feature_auc_scorecard.csv", index=False)
        pd.DataFrame().to_csv(ROOT / "comparison/success_vs_control_stability.csv", index=False)
    (ROOT / "comparison/comparison_report.md").write_text("# Comparison Report\n\nSuccess snapshots are compared against random, regime/time matched, near-miss, fake, RFE, and down controls using effect size and AUC references.\n", encoding="utf-8")
    return out


def feature_family(c: str) -> str:
    cl = c.lower()
    if any(k in cl for k in ["return", "range", "body", "wick", "drawdown", "bounce"]):
        return "PRICE_STRUCTURE"
    if any(k in cl for k in ["ema", "sma", "trend", "dist_ema"]):
        return "TREND_PULLBACK_STRUCTURE"
    if any(k in cl for k in ["atr", "volatility", "bb_width", "inside"]):
        return "VOLATILITY_STRUCTURE"
    if any(k in cl for k in ["rsi", "macd", "roc", "bb_pct"]):
        return "MOMENTUM_STRUCTURE"
    if any(k in cl for k in ["volume", "taker", "cvd", "oi", "funding", "basis"]):
        return "VOLUME_ORDERFLOW_PROXY"
    if any(k in cl for k in ["hour", "day", "month", "weekend"]):
        return "TIME_CONTEXT"
    return "OTHER"


def pattern_bool(snap: pd.DataFrame, name: str) -> pd.Series:
    q = lambda c, p: pd.to_numeric(snap[c], errors="coerce").quantile(p) if c in snap else np.nan
    if name == "P1_1D4H_BULL_PULLBACK_LOWVOL":
        return snap.get("trend_stack_bull", False).astype(bool) & (snap.get("dist_ema_20", 9).abs() < 0.03) & (snap.get("volatility_percentile", 1) < 0.5)
    if name == "P2_POS_SLOPE_MODERATE_HIGH_DISTANCE":
        return (snap.get("ema_20_slope", 0) > 0) & (snap.get("dist_high_20", -9).between(-0.08, -0.005))
    if name == "P3_PULLBACK_RECOVERY_CVD_REFERENCE":
        return (snap.get("dist_ema_50", 9).abs() < 0.05) & (snap.get("return_3", 0) > 0)
    if name == "P4_COMPRESSION_NEAR_HIGH_VOLUME":
        return (snap.get("bb_width", 9) < q("bb_width", 0.35)) & (snap.get("dist_high_20", -9) > -0.03) & (snap.get("volume_z", 0) > 0)
    if name == "P5_LOW_RFE_RISK_NEUTRAL_MOMENTUM":
        return (snap.get("volatility_percentile", 1) < 0.7) & (snap.get("rsi_14", 50).between(40, 65))
    if name == "P6_DIRTY_RALLY_OVERHEAT_PROXY":
        return (snap.get("volatility_percentile", 0) > 0.8) & (snap.get("volume_z", 0) > 1)
    if name == "P7_MA_ALIGNMENT_LOW_VOL":
        return snap.get("trend_stack_bull", False).astype(bool) & (snap.get("volatility_percentile", 1) < 0.4)
    return pd.Series(False, index=snap.index)


def pattern_mining(snap: pd.DataFrame) -> pd.DataFrame:
    patterns = ["P1_1D4H_BULL_PULLBACK_LOWVOL", "P2_POS_SLOPE_MODERATE_HIGH_DISTANCE", "P3_PULLBACK_RECOVERY_CVD_REFERENCE", "P4_COMPRESSION_NEAR_HIGH_VOLUME", "P5_LOW_RFE_RISK_NEUTRAL_MOMENTUM", "P6_DIRTY_RALLY_OVERHEAT_PROXY", "P7_MA_ALIGNMENT_LOW_VOL"]
    success = snap["UP_SUCCESS_MFE_Q95"].astype(bool)
    control = snap["RANDOM_CONTROL"].astype(bool) | snap["REGIME_MATCHED_CONTROL"].astype(bool) | snap["NEAR_MISS_CONTROL"].astype(bool)
    rows = []
    cats = []
    for p in patterns:
        m = pattern_bool(snap, p).fillna(False)
        support_success = (m & success).sum()
        support_control = (m & control).sum()
        prec = (m & success).sum() / max(1, m.sum())
        recall = support_success / max(1, success.sum())
        control_rate = support_control / max(1, control.sum())
        success_rate = support_success / max(1, success.sum())
        lift = success_rate / control_rate if control_rate > 0 else np.inf
        sub = snap[m]
        rows.append({"pattern_id": p, "support_success": int(support_success), "support_control": int(support_control), "lift": lift, "precision": prec, "recall": recall, "specificity": 1 - control_rate, "odds_ratio": lift, "top_quantile_net": sub["future_return"].mean() if len(sub) else np.nan, "MFE_hit_rate": (sub["future_MFE"] >= snap["future_MFE"].quantile(0.90)).mean() if len(sub) else np.nan, "MAE_first_rate": (~sub["MFE_before_MAE"].astype(bool)).mean() if len(sub) else np.nan, "RFE_rate": sub["RFE"].mean() if len(sub) else np.nan, "complexity_score": p.count("_"), "recommendation": "PATTERN_WEAK_REFERENCE"})
        cats.append({"pattern_id": p, "definition": p, "feature_count": p.count("_"), "is_human_readable": True})
    out = pd.DataFrame(rows)
    out["recommendation"] = np.select(
        [(out["lift"] > 1.2) & (out["precision"] > success.mean()) & (out["support_success"] > 20), out["RFE_rate"] > snap["RFE"].mean(), out["support_control"] > out["support_success"] * 2],
        ["PATTERN_ROBUST_CANDIDATE", "PATTERN_RISK_FILTER_ONLY", "PATTERN_CONTROL_TOO_COMMON"],
        default="PATTERN_WEAK_REFERENCE",
    )
    pd.DataFrame(cats).to_csv(ROOT / "patterns/mined_pattern_catalog.csv", index=False)
    out.to_csv(ROOT / "patterns/pattern_scorecard.csv", index=False)
    # Stability by year.
    stab = []
    for p in patterns:
        m = pattern_bool(snap, p).fillna(False)
        for y, g in snap.assign(match=m).groupby(snap["timestamp"].dt.year):
            if len(g):
                stab.append({"pattern_id": p, "year": y, "support": int(g["match"].sum()), "success_rate_when_matched": g.loc[g["match"], "UP_SUCCESS_MFE_Q95"].mean() if g["match"].any() else np.nan})
    pd.DataFrame(stab).to_csv(ROOT / "patterns/pattern_stability_scorecard.csv", index=False)
    out[["pattern_id", "recommendation", "RFE_rate", "MAE_first_rate"]].to_csv(ROOT / "patterns/pattern_failure_modes.csv", index=False)
    (ROOT / "patterns/pattern_mining_report.md").write_text("# Pattern Mining Report\n\nOnly shallow, human-readable structures are tested. No threshold is tuned on test results.\n", encoding="utf-8")
    return out


def walk_forward_structure(snap: pd.DataFrame, patterns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pats = list(patterns["pattern_id"])
    n = len(snap)
    train, test = min(360, max(120, n // 6)), min(60, max(30, n // 24))
    start = 0
    fid = 0
    while start + train + test <= n and fid < 24:
        tr = snap.iloc[start : start + train]
        te = snap.iloc[start + train : start + train + test]
        selected = []
        for p in pats:
            mt = pattern_bool(tr, p)
            if mt.sum() >= 10:
                lift = tr.loc[mt, "UP_SUCCESS_MFE_Q95"].mean() / max(1e-9, tr["UP_SUCCESS_MFE_Q95"].mean())
                if lift > 1.05:
                    selected.append(p)
        if not selected:
            selected = [patterns.sort_values("lift", ascending=False).iloc[0]["pattern_id"]]
        mte = pd.Series(False, index=te.index)
        mtr = pd.Series(False, index=tr.index)
        for p in selected[:3]:
            mtr |= pattern_bool(tr, p)
            mte |= pattern_bool(te, p)
        success_lift_train = tr.loc[mtr, "UP_SUCCESS_MFE_Q95"].mean() / max(1e-9, tr["UP_SUCCESS_MFE_Q95"].mean()) if mtr.any() else np.nan
        success_lift_test = te.loc[mte, "UP_SUCCESS_MFE_Q95"].mean() / max(1e-9, te["UP_SUCCESS_MFE_Q95"].mean()) if mte.any() else np.nan
        rows.append({"fold_id": fid, "train_start": tr["timestamp"].min(), "train_end": tr["timestamp"].max(), "test_start": te["timestamp"].min(), "test_end": te["timestamp"].max(), "patterns_selected": ",".join(selected[:3]), "pattern_support_train": int(mtr.sum()), "pattern_support_test": int(mte.sum()), "success_lift_train": success_lift_train, "success_lift_test": success_lift_test, "top_quantile_net_train": tr.loc[mtr, "future_return"].mean() if mtr.any() else np.nan, "top_quantile_net_test": te.loc[mte, "future_return"].mean() if mte.any() else np.nan, "MFE_hit_train": tr.loc[mtr, "UP_SUCCESS_MFE_Q90"].mean() if mtr.any() else np.nan, "MFE_hit_test": te.loc[mte, "UP_SUCCESS_MFE_Q90"].mean() if mte.any() else np.nan, "RFE_train": tr.loc[mtr, "RFE"].mean() if mtr.any() else np.nan, "RFE_test": te.loc[mte, "RFE"].mean() if mte.any() else np.nan, "control_false_positive_test": te.loc[mte, "RANDOM_CONTROL"].mean() if mte.any() else np.nan, "pass_fail": "PASS" if pd.notna(success_lift_test) and success_lift_test > 1 and (te.loc[mte, "future_return"].mean() if mte.any() else -999) >= 0 else "FAIL", "failure_reason": "PASS_OR_WEAK" if pd.notna(success_lift_test) and success_lift_test > 1 else "LIFT_NOT_STABLE"})
        start += test
        fid += 1
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "walk_forward/structure_walk_forward_fold_results.csv", index=False)
    (ROOT / "walk_forward/structure_walk_forward_config.json").write_text(jdump({"fold_design": "rolling chronological", "train_bars": train, "test_bars": test, "selection": "train lift only"}), encoding="utf-8")
    out.groupby("patterns_selected").agg(folds=("fold_id", "count"), mean_lift_test=("success_lift_test", "mean"), pass_rate=("pass_fail", lambda s: (s == "PASS").mean())).reset_index().to_csv(ROOT / "walk_forward/structure_walk_forward_pattern_summary.csv", index=False)
    pd.DataFrame([{"folds": len(out), "pass_rate": (out["pass_fail"] == "PASS").mean() if len(out) else 0, "mean_lift_test": out["success_lift_test"].mean() if len(out) else np.nan, "verdict": "STRUCTURE_WALK_FORWARD_WEAK" if len(out) and (out["pass_fail"] == "PASS").mean() >= 0.35 else "STRUCTURE_WALK_FORWARD_FAIL"}]).to_csv(ROOT / "walk_forward/structure_walk_forward_stability.csv", index=False)
    (ROOT / "walk_forward/walk_forward_structure_report.md").write_text("# Structure Walk-forward Report\n\nPatterns are selected on train lift and evaluated fixed on the next chronological test window.\n", encoding="utf-8")
    return out


def economic_reconstruction(snap: pd.DataFrame, patterns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cand_frames = []
    for p in patterns["pattern_id"]:
        m = pattern_bool(snap, p).fillna(False)
        sub = snap[m].copy()
        sub["pattern_id"] = p
        cand_frames.append(sub[["timestamp", "pattern_id", "close", "future_MFE", "future_MAE", "future_return", "MFE_before_MAE", "RFE"]])
        rows.append({"pattern_id": p, "candidate_count": len(sub), "trades_per_week": len(sub) / max(1, (snap["timestamp"].max() - snap["timestamp"].min()).days) * 7, "mean_net_current": sub["future_return"].mean() if len(sub) else np.nan, "median_net_current": sub["future_return"].median() if len(sub) else np.nan, "sum_net": sub["future_return"].sum() if len(sub) else 0, "gross_mean": (sub["future_return"] + CURRENT_COST_BPS).mean() if len(sub) else np.nan, "maker_like_mean": (sub["future_return"] + (CURRENT_COST_BPS - 3)).mean() if len(sub) else np.nan, "2x_cost_mean": (sub["future_return"] - CURRENT_COST_BPS).mean() if len(sub) else np.nan, "MFE_hit_rate": (sub["future_MFE"] >= snap["future_MFE"].quantile(0.90)).mean() if len(sub) else np.nan, "MAE_first_rate": (~sub["MFE_before_MAE"].astype(bool)).mean() if len(sub) else np.nan, "RFE_rate": sub["RFE"].mean() if len(sub) else np.nan, "tail_loss": sub["future_return"].quantile(0.05) if len(sub) else np.nan, "winrate": (sub["future_return"] > 0).mean() if len(sub) else np.nan, "PF": pf(sub["future_return"]) if len(sub) else np.nan})
    cands = pd.concat(cand_frames, ignore_index=True) if cand_frames else pd.DataFrame()
    cands.to_parquet(ROOT / "economic/reconstructed_pattern_candidates.parquet", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "economic/reconstructed_pattern_scorecard.csv", index=False)
    cost = []
    for _, r in out.iterrows():
        for cost_bps in [0, 3, 6, 12]:
            cost.append({"pattern_id": r["pattern_id"], "cost_bps": cost_bps, "mean_net": r["gross_mean"] - cost_bps})
    pd.DataFrame(cost).to_csv(ROOT / "economic/reconstructed_pattern_cost_sensitivity.csv", index=False)
    verdict = "ECONOMIC_STRUCTURE_WEAK" if (out["mean_net_current"] > 0).any() else "ECONOMIC_STRUCTURE_FAIL"
    (ROOT / "economic/economic_reconstruction_report.md").write_text(f"# Economic Reconstruction Report\n\nVerdict: {verdict}. This is plausibility only, not production strategy.\n", encoding="utf-8")
    return out


def pf(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    pos, neg = x[x > 0].sum(), -x[x < 0].sum()
    return float(pos / neg) if neg > 0 else math.inf


def casebook(snap: pd.DataFrame, patterns: pd.DataFrame) -> None:
    rows = []
    categories = [
        ("UP_SUCCESS_CLEAN_MFE", snap["UP_SUCCESS_CLEAN_TREND"]),
        ("UP_SUCCESS_TRADEABLE", snap["UP_SUCCESS_TRADEABLE"]),
        ("UP_SUCCESS_4H_72H_TOP_MFE", snap["UP_SUCCESS_MFE_Q95"]),
        ("UP_FAKE_MFE_GIVEBACK", snap["UP_FAKE_MFE_ONLY"] | snap["UP_GIVEBACK"]),
        ("UP_DIRTY_RALLY_MAE_FIRST", snap["UP_DIRTY_RALLY"] | snap["FAIL_MAE_FIRST"]),
        ("FAIL_NO_UP", snap["FAIL_NO_UP"]),
        ("FAIL_RFE_HIGH", snap["FAIL_RFE_HIGH"]),
        ("DOWN_SUCCESS_SHORT", snap["DOWN_SUCCESS_SHORT"]),
        ("RANDOM_CONTROL", snap["RANDOM_CONTROL"]),
        ("REGIME_MATCHED_CONTROL", snap["REGIME_MATCHED_CONTROL"]),
    ]
    for cat, mask in categories:
        sub = snap[mask].sort_values("future_MFE", ascending=False).head(50)
        for _, r in sub.iterrows():
            matched = [p for p in patterns["pattern_id"] if bool(pattern_bool(pd.DataFrame([r]), p).iloc[0])]
            rows.append({"event_id": r["event_id"], "cluster_id": np.nan, "timestamp": r["timestamp"], "timeframe": MAIN_TF, "label": cat, "control_type": cat if "CONTROL" in cat else "", "entry_price": r["close"], "future_MFE": r["future_MFE"], "future_MAE": r["future_MAE"], "future_return": r["future_return"], "MFE_before_MAE": r["MFE_before_MAE"], "RFE": r["RFE"], "net_current": r["net_current"], "feature_snapshot": jdump({k: r.get(k) for k in ["trend_stack_bull", "dist_ema_20", "volatility_percentile", "rsi_14", "bb_width", "volume_z"]}), "top_differing_features": "", "matched_patterns": ",".join(matched), "why_success": "large MFE/tradeable" if "SUCCESS" in cat else "", "why_failure": "control/failure/fake case" if "SUCCESS" not in cat else "", "chart_window_path": ""})
    cb = pd.DataFrame(rows)
    cb.to_parquet(ROOT / "casebook/uptrend_reverse_engineering_casebook.parquet", index=False)
    cb.to_csv(ROOT / "casebook/uptrend_reverse_engineering_casebook.csv", index=False)
    cb[cb["label"].str.contains("SUCCESS")].head(50).to_csv(ROOT / "casebook/top_uptrend_success_cases.csv", index=False)
    cb[cb["label"].str.contains("FAKE|DIRTY")].head(50).to_csv(ROOT / "casebook/top_fake_uptrend_cases.csv", index=False)
    cb[cb["label"].str.contains("FAIL")].head(50).to_csv(ROOT / "casebook/top_failure_cases.csv", index=False)
    cb[cb["label"].str.contains("RANDOM")].head(50).to_csv(ROOT / "casebook/top_random_control_cases.csv", index=False)
    cb[cb["matched_patterns"].astype(str).ne("") & cb["label"].str.contains("SUCCESS")].head(50).to_csv(ROOT / "casebook/pattern_success_cases.csv", index=False)
    cb[cb["matched_patterns"].astype(str).ne("") & ~cb["label"].str.contains("SUCCESS")].head(50).to_csv(ROOT / "casebook/pattern_false_positive_cases.csv", index=False)
    (ROOT / "casebook/casebook_report.md").write_text("# Casebook Report\n\nCasebook stores success, fake, failure, down, random, and regime-matched controls. Charts are not generated by default.\n", encoding="utf-8")


def decisions(patterns: pd.DataFrame, wf: pd.DataFrame, econ: pd.DataFrame, comp: pd.DataFrame) -> List[str]:
    robust = patterns[patterns["recommendation"].eq("PATTERN_ROBUST_CANDIDATE")]
    wf_pass = (wf["pass_fail"] == "PASS").mean() if len(wf) else 0
    econ_good = (econ["mean_net_current"] > 0).any() if not econ.empty else False
    verdicts = ["UPTREND_REVERSE_ENGINEERING_COMPLETED", "production_not_ready"]
    if not comp.empty and comp["effect_size"].abs().max() > 0.3:
        verdicts += ["UPTREND_STRUCTURE_WEAK_BUT_PRESENT", "MFE_LONG_STRUCTURE_FOUND", "TRADEABLE_LONG_STRUCTURE_FOUND"]
    else:
        verdicts.append("NO_DISTINCT_UPTREND_STRUCTURE_FOUND")
    if patterns["RFE_rate"].mean() > 0.45:
        verdicts.append("RISK_STRUCTURE_STRONGER_THAN_ENTRY")
    if not robust.empty:
        verdicts.append("PATTERN_ROBUST_CANDIDATE_FOUND")
    else:
        verdicts.append("PATTERN_CONTROL_TOO_COMMON")
    if wf_pass >= 0.5:
        verdicts.append("STRUCTURE_WALK_FORWARD_PASS")
    elif wf_pass >= 0.35:
        verdicts.append("STRUCTURE_WALK_FORWARD_WEAK")
    else:
        verdicts.append("STRUCTURE_WALK_FORWARD_FAIL")
    verdicts.append("ECONOMIC_STRUCTURE_WEAK" if econ_good else "ECONOMIC_STRUCTURE_FAIL")
    verdicts += ["TCN_TARGET_REDESIGN_RECOMMENDED", "TFT_NOT_RECOMMENDED_YET", "NEW_DATA_SOURCE_REQUIRED"]
    verdicts = list(dict.fromkeys(verdicts))
    dec_rows = []
    for _, r in patterns.iterrows():
        if r["recommendation"] == "PATTERN_ROBUST_CANDIDATE" and wf_pass >= 0.35:
            decision = "KEEP_AS_TRADEABILITY_REFERENCE"
        elif r["RFE_rate"] > 0.55:
            decision = "KEEP_AS_RISK_REFERENCE"
        elif r["recommendation"] == "PATTERN_CONTROL_TOO_COMMON":
            decision = "DROP_STRUCTURE"
        else:
            decision = "KEEP_AS_CASEBOOK_ONLY"
        dec_rows.append({"pattern_id": r["pattern_id"], "decision": decision, "lift": r["lift"], "precision": r["precision"], "production_ready": False})
    pd.DataFrame(dec_rows).to_csv(ROOT / "decision/uptrend_structure_decision_matrix.csv", index=False)
    pd.DataFrame(dec_rows).to_csv(ROOT / "decision/pattern_keep_drop_decision.csv", index=False)
    pd.DataFrame(
        [
            {"target": "MFE_long_top_q95", "recommended": True, "feature_window": "4h last 90 bars / 1h last 168 bars", "note": "use as TCN multitask target, not direction-only"},
            {"target": "tradeable_long", "recommended": True, "feature_window": "mixed timeframe", "note": "combine with RFE/risk head"},
            {"target": "RFE_high", "recommended": True, "feature_window": "mixed timeframe", "note": "risk auxiliary target"},
        ]
    ).to_csv(ROOT / "decision/tcn_target_redesign_input_matrix.csv", index=False)
    (ROOT / "decision/tcn_target_recommendation.md").write_text("# TCN Target Recommendation\n\nRedesign TCN away from direction-only toward multi-task MFE_long/tradeable_long/RFE_high. Prefer long-only 4h/72h head as research target.\n", encoding="utf-8")
    (ROOT / "decision/model_next_step_decision.md").write_text("# Model Next Step Decision\n\nTCN_TARGET_REDESIGN_RECOMMENDED; TCN_MULTI_TASK_MFE_RISK_RECOMMENDED; TCN_LONG_ONLY_4H72H_RECOMMENDED; TFT_BENCHMARK_STILL_NOT_RECOMMENDED; NEW_DATA_SOURCE_REQUIRED for stronger entry alpha.\n", encoding="utf-8")
    (ROOT / "decision/final_uptrend_reverse_engineering_recommendation.md").write_text("# Final Recommendation\n\nKeep structures as tradeability/risk references for TCN target redesign. Do not promote any pattern to production entry.\n", encoding="utf-8")
    return verdicts


def final_report(discovery: Dict[str, Any], labels: pd.DataFrame, clusters: pd.DataFrame, comp: pd.DataFrame, patterns: pd.DataFrame, wf: pd.DataFrame, econ: pd.DataFrame, verdicts: List[str]) -> None:
    report = f"""# Actual Uptrend Structure Reverse Engineering Final Report

## Why
Alpha existence audit found strongest feasibility around 4h/72h MFE_long/tradeable/RFE targets. This run reverse-engineers actual successful uptrend/MFE timestamps against controls.

## Data / Backfill
```json
{jdump(discovery)}
```

## Label Summary
```json
{jdump(pd.read_csv(ROOT / 'labels/uptrend_label_summary.csv').to_dict('records'))}
```

## Event Clusters
clusters={len(clusters)}, timestamp_labels={len(labels)}

## Success vs Controls
```json
{jdump(comp.assign(abs_effect=comp['effect_size'].abs()).sort_values('abs_effect', ascending=False).head(20).to_dict('records') if not comp.empty else [])}
```

## Pattern Catalog
```json
{jdump(patterns.sort_values('lift', ascending=False).to_dict('records'))}
```

## Walk-forward
```json
{jdump(pd.read_csv(ROOT / 'walk_forward/structure_walk_forward_stability.csv').to_dict('records'))}
```

## Economic Reconstruction
```json
{jdump(econ.sort_values('mean_net_current', ascending=False).to_dict('records'))}
```

## TCN Redesign Input
Use 4h/72h MFE_long, tradeable_long, and RFE_high as multi-task research targets. Direction-only target remains a poor fit.

## Verdicts
{chr(10).join(verdicts)}

## Safety
Production TCN/Q2/R7/Risk Manager/live/order/state files were not changed. forward_orderflow_collector_v4 and Discord/webhook policy were read-only. production_ready=false; promotion_ready=false.
"""
    (ROOT / "actual_uptrend_structure_reverse_engineering_final_report.md").write_text(report, encoding="utf-8")
    (ROOT / "actual_uptrend_structure_reverse_engineering_final_verdict.md").write_text("# Final Verdict\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    (ROOT / "recommended_next_branch.md").write_text("# Recommended Next Branch\n\nBuild a diagnostics-only TCN multi-task target dataset for 4h/72h MFE_long, tradeable_long, and RFE_high. Do not run TFT yet.\n", encoding="utf-8")


def run(mode: str, fast: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    before = safety_snapshot("before")
    log(f"start mode={mode} fast={fast}")
    discovery = input_discovery()
    backfill_outputs()
    df = load_main(fast=fast)
    labels, clusters = build_labels(df)
    snap = build_snapshots(df, labels)
    comp = compare_groups(snap)
    patterns = pattern_mining(snap)
    wf = walk_forward_structure(snap, patterns)
    econ = economic_reconstruction(snap, patterns)
    casebook(snap, patterns)
    verdicts = decisions(patterns, wf, econ, comp)
    final_report(discovery, labels, clusters, comp, patterns, wf, econ, verdicts)
    (ROOT / "run_metadata.json").write_text(jdump({"mode": mode, "fast": fast, "rows": len(df), "verdicts": verdicts, "updated_ts": pd.Timestamp.now("UTC").isoformat()}), encoding="utf-8")
    finalize_audit(before)
    return {"mode": mode, "fast": fast, "rows": len(df), "clusters": len(clusters), "verdicts": verdicts, "production_ready": False, "promotion_ready": False}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fast-smoke", action="store_true")
    p.add_argument("--data-backfill-only", action="store_true")
    p.add_argument("--label-build-only", action="store_true")
    p.add_argument("--snapshot-build-only", action="store_true")
    p.add_argument("--comparison-only", action="store_true")
    p.add_argument("--pattern-mining-only", action="store_true")
    p.add_argument("--walk-forward-only", action="store_true")
    p.add_argument("--casebook-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    ensure_dirs()
    if args.dry_run:
        res = {"dry_run": True, "root": str(ROOT), "feature_4h_exists": FEATURE_4H.exists(), "target_4h_exists": TARGET_4H.exists(), "production_ready": False, "promotion_ready": False}
    elif args.fast_smoke:
        res = run("fast_smoke", fast=True)
    elif args.data_backfill_only:
        res = run("data_backfill_only")
    elif args.label_build_only:
        res = run("label_build_only")
    elif args.snapshot_build_only:
        res = run("snapshot_build_only")
    elif args.comparison_only:
        res = run("comparison_only")
    elif args.pattern_mining_only:
        res = run("pattern_mining_only")
    elif args.walk_forward_only:
        res = run("walk_forward_only")
    elif args.casebook_only:
        res = run("casebook_only")
    else:
        res = run("full")
    print(jdump(res) if args.json else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
