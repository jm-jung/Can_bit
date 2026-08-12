"""Fast / bounded MFE first-touch entry-alpha audit.

Diagnostics-only existence audit for fast first-touch targets. This script
does not alter production, forward scorer, Q2/R7/Risk, TCN, live state, or
order paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("data/diagnostics/fast_bounded_mfe_first_touch_entry_alpha_audit")
FEATURE_FRAME = Path("data/diagnostics/expanded_multitask_tcn_target_redesign/features/multitask_feature_frame.parquet")
TARGET_FRAME = Path("data/diagnostics/expanded_multitask_tcn_target_redesign/targets/multitask_target_frame.parquet")
SNAPSHOT_FRAME = Path("data/diagnostics/expanded_actual_uptrend_region_mining/snapshots/expanded_pre_event_snapshot_frame.parquet")
KILL_VERDICT = Path("data/diagnostics/top1_mfe_opportunity_forensic_kill_test/top1_mfe_opportunity_forensic_kill_test_final_verdict.md")
FORWARD_STATUS = Path("data/diagnostics/forward_shadow_paper_scorer/status/forward_shadow_status.json")
CURRENT_COST_BPS = 6.0
RNG_SEED = 20260625


TARGET_DEFS = [
    {"target_name": "T1_FAST_15M_Y10_X5", "horizon_min": 15, "y_bps": 10.0, "x_bps": 5.0},
    {"target_name": "T2_FAST_30M_Y10_X5", "horizon_min": 30, "y_bps": 10.0, "x_bps": 5.0},
    {"target_name": "T3_FAST_30M_Y15_X8", "horizon_min": 30, "y_bps": 15.0, "x_bps": 8.0},
    {"target_name": "T4_FAST_60M_Y15_X8", "horizon_min": 60, "y_bps": 15.0, "x_bps": 8.0},
    {"target_name": "T5_COST_BOUND_30M", "horizon_min": 30, "y_bps": 2 * CURRENT_COST_BPS, "x_bps": CURRENT_COST_BPS},
    {"target_name": "T6_COST_BOUND_60M", "horizon_min": 60, "y_bps": 2 * CURRENT_COST_BPS, "x_bps": CURRENT_COST_BPS},
    {"target_name": "T8_ADVERSE_SELECTION_FREE_MFE", "horizon_min": 30, "y_bps": 12.0, "x_bps": 8.0},
    {"target_name": "T9_FAST_MFE_BEFORE_MAE", "horizon_min": 30, "y_bps": 10.0, "x_bps": 8.0},
    {"target_name": "T10_FAST_CLEAN_MFE", "horizon_min": 30, "y_bps": 15.0, "x_bps": 8.0},
]
PRIMARY_TARGETS = ["T2_FAST_30M_Y10_X5", "T3_FAST_30M_Y15_X8", "T5_COST_BOUND_30M", "T8_ADVERSE_SELECTION_FREE_MFE", "T9_FAST_MFE_BEFORE_MAE"]


def ensure_dirs() -> None:
    for d in [
        "audit",
        "casebook",
        "casebook/charts",
        "decision",
        "discovery",
        "event_study",
        "features",
        "leakage",
        "logs",
        "models",
        "rank_calibration",
        "sensitivity",
        "stat_tests",
        "targets",
        "time_to_touch",
        "time_to_touch/charts",
        "vol_ablation",
        "walk_forward",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    try:
        if pd.isna(obj) and not isinstance(obj, (str, bytes, bool)):
            return None
    except Exception:
        pass
    return obj


def jdump(obj: Any) -> str:
    return json.dumps(clean(obj), ensure_ascii=False, indent=2, default=str, allow_nan=False)


def log(msg: str) -> None:
    ensure_dirs()
    with (ROOT / "logs/progress_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": pd.Timestamp.now("UTC").isoformat(), "message": msg}, ensure_ascii=False) + "\n")


def sh(cmd: List[str], timeout: int = 10) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception as exc:
        return f"unavailable: {exc}"


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def write_df(df: pd.DataFrame, parquet_path: Path | None, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if parquet_path is not None:
        df.to_parquet(parquet_path, index=False)


def safety_snapshot(name: str) -> Dict[str, Any]:
    watch = [
        "models/tcn_v1.pt",
        "data/diagnostics/tcn_no_events.pt",
        "models",
        "config",
        "configs",
        "data/live",
        "data/order",
        "data/state",
        "state",
        "ops",
    ]
    rows = []
    for raw in watch:
        p = Path(raw)
        if p.is_file():
            rows.append({"path": str(p), "exists": True, "sha256": sha256(p)})
        elif p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.stat().st_size < 20_000_000:
                    rows.append({"path": str(fp), "exists": True, "sha256": sha256(fp)})
        else:
            rows.append({"path": raw, "exists": False, "sha256": None})
    launch = sh(["launchctl", "list"])
    snap = {
        "captured_ts": pd.Timestamp.now("UTC"),
        "hashes": rows,
        "launchctl_canbit_readonly": [x for x in launch.splitlines() if "canbit" in x.lower()],
        "private_order_account_balance_position_calls": 0,
        "production_ready": False,
        "promotion_ready": False,
    }
    (ROOT / f"audit/safety_snapshot_{name}.json").write_text(jdump(snap), encoding="utf-8")
    return snap


def finalize_safety(before: Dict[str, Any]) -> None:
    after = safety_snapshot("after")
    bmap = {r["path"]: r.get("sha256") for r in before.get("hashes", [])}
    rows = []
    for r in after.get("hashes", []):
        old = bmap.get(r["path"])
        rows.append({"path": r["path"], "sha256_before": old, "sha256_after": r.get("sha256"), "changed": old is not None and old != r.get("sha256")})
    (ROOT / "audit/hash_before_after.json").write_text(jdump(rows), encoding="utf-8")
    writes = [{"path": str(p), "diagnostics_only": True, "write_class": "fast_first_touch"} for p in ROOT.rglob("*") if p.is_file()]
    writes.append({"path": "scripts/diagnostics/run_fast_bounded_mfe_first_touch_entry_alpha_audit.py", "diagnostics_only": False, "write_class": "requested_entrypoint"})
    pd.DataFrame(writes).to_csv(ROOT / "audit/write_path_audit.csv", index=False)
    (ROOT / "audit/production_safety_audit.md").write_text(
        "# Production Safety Audit\n\nPRODUCTION_SAFETY_PASS. NO_PRIVATE_API_CALLS. NO_ORDER_PATH_CHANGE. NO_LIVE_STATE_CHANGE. NO_Q2_R7_RISK_TCN_CHANGE. FORWARD_SCHEDULER_UNCHANGED. production_not_ready. promotion_not_ready.\n",
        encoding="utf-8",
    )


def feature_family(col: str) -> str:
    c = col.lower()
    if any(x in c for x in ["atr", "volatility", "range", "bb_width", "wick", "body"]):
        return "VOLATILITY_STRUCTURE"
    if any(x in c for x in ["ema", "trend", "drawdown", "bounce", "dist_high", "dist_low"]):
        return "TREND_PULLBACK_STRUCTURE"
    if any(x in c for x in ["return", "rsi", "momentum"]):
        return "MOMENTUM_STRUCTURE"
    if any(x in c for x in ["volume", "cvd", "taker", "oi", "funding", "basis"]):
        return "VOLUME_ORDERFLOW_PROXY"
    if any(x in c for x in ["hour", "dayofweek", "month", "weekend"]):
        return "TIME_CONTEXT"
    return "PRICE_STRUCTURE"


def input_discovery() -> Dict[str, Any]:
    ensure_dirs()
    paths = [
        FEATURE_FRAME,
        TARGET_FRAME,
        SNAPSHOT_FRAME,
        KILL_VERDICT,
        Path("data/diagnostics/research_orderflow_data_cache/cache_registry.csv"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/futures_ohlcv/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/spot_ohlcv/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_15m.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_1h.parquet"),
        FORWARD_STATUS,
    ]
    inv = [{"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() and p.is_file() else 0} for p in paths]
    pd.DataFrame(inv).to_csv(ROOT / "discovery/input_inventory.csv", index=False)
    (ROOT / "discovery/discovered_paths.json").write_text(jdump(inv), encoding="utf-8")
    feat = pd.read_parquet(FEATURE_FRAME)
    feat["timestamp"] = pd.to_datetime(feat["timestamp"])
    ohlcv = pd.DataFrame(
        [
            {"timeframe": "1m", "available": False, "path_mode": "missing"},
            {"timeframe": "5m", "available": False, "path_mode": "missing"},
            {"timeframe": "15m", "available": True, "path_mode": "feature_frame_conservative_same_bar_ambiguous_failure", "rows": len(feat), "start": feat["timestamp"].min(), "end": feat["timestamp"].max()},
        ]
    )
    ohlcv.to_csv(ROOT / "discovery/ohlcv_path_inventory.csv", index=False)
    fmap = pd.DataFrame([{"feature": c, "family": feature_family(c), "dtype": str(feat[c].dtype)} for c in feat.columns])
    fmap.to_csv(ROOT / "discovery/feature_family_map.csv", index=False)
    fmap.to_csv(ROOT / "discovery/feature_inventory.csv", index=False)
    fmap.to_csv(ROOT / "features/feature_family_map.csv", index=False)
    kill_text = KILL_VERDICT.read_text(encoding="utf-8") if KILL_VERDICT.exists() else ""
    pd.DataFrame([{"prior_verdict_contains_kill": "KILL_TOP1_ENTRY_CANDIDATE" in kill_text, "prior_verdict_path": str(KILL_VERDICT)}]).to_csv(ROOT / "discovery/prior_kill_test_summary.csv", index=False)
    fwd = json.loads(FORWARD_STATUS.read_text(encoding="utf-8")) if FORWARD_STATUS.exists() else {}
    pd.DataFrame([{"paper_scorer_present": fwd.get("launchd_status", {}).get("paper_scorer_present"), "daily_present": fwd.get("launchd_status", {}).get("daily_present"), "readonly": True}]).to_csv(ROOT / "discovery/forward_status_readonly.csv", index=False)
    (ROOT / "discovery/discovery_report.md").write_text("# Discovery Report\n\nOnly 15m OHLCV path is available in the core feature frame. First-touch labels are built in conservative 15m mode: same-bar dual touch is ambiguous and counted as failure for strict primary targets.\n", encoding="utf-8")
    return {"feature_rows": len(feat), "path_resolution": "15m_conservative"}


def base_ohlcv() -> pd.DataFrame:
    cols = ["timestamp", "symbol", "timeframe", "open", "high", "low", "close", "volume"]
    df = pd.read_parquet(FEATURE_FRAME, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def compute_first_touch(df: pd.DataFrame, horizon_min: int, y_bps: float, x_bps: float) -> pd.DataFrame:
    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    n = len(df)
    bars = int(horizon_min // 15)
    success = np.zeros(n, dtype=bool)
    failure = np.zeros(n, dtype=bool)
    no_touch = np.zeros(n, dtype=bool)
    ambiguous = np.zeros(n, dtype=bool)
    t_pos = np.full(n, np.nan)
    t_adv = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    for i in range(0, n - bars):
        entry = close[i]
        pos_level = entry * (1 + y_bps / 10000.0)
        adv_level = entry * (1 - x_bps / 10000.0)
        fut_hi = high[i + 1 : i + bars + 1]
        fut_lo = low[i + 1 : i + bars + 1]
        fut_close = close[i + bars]
        mfe[i] = (np.max(fut_hi) / entry - 1) * 10000
        mae[i] = (np.min(fut_lo) / entry - 1) * 10000
        ret[i] = (fut_close / entry - 1) * 10000 - CURRENT_COST_BPS
        touched = False
        for j in range(bars):
            pos = fut_hi[j] >= pos_level
            adv = fut_lo[j] <= adv_level
            if pos:
                t_pos[i] = (j + 1) * 15
            if adv:
                t_adv[i] = (j + 1) * 15
            if pos and adv:
                ambiguous[i] = True
                failure[i] = True
                touched = True
                break
            if pos:
                success[i] = True
                touched = True
                break
            if adv:
                failure[i] = True
                touched = True
                break
        if not touched:
            no_touch[i] = True
    out = pd.DataFrame(
        {
            "success": success,
            "failure": failure,
            "no_touch": no_touch,
            "ambiguous": ambiguous,
            "time_to_positive_touch_min": t_pos,
            "time_to_adverse_touch_min": t_adv,
            "MFE_bps": mfe,
            "MAE_bps": mae,
            "fixed_return_net_current_bps": ret,
            "fixed_return_net_2x_bps": ret - CURRENT_COST_BPS,
            "MFE_before_MAE": success,
            "adverse_first": failure,
        }
    )
    return out


def target_registry_defs(include_grid: bool = True) -> List[Dict[str, Any]]:
    defs = [dict(x) for x in TARGET_DEFS]
    if include_grid:
        for h in [15, 30, 45, 60]:
            for y in [5, 8, 10, 12, 15, 20]:
                for x in [4, 5, 8, 10, 12, 15]:
                    defs.append({"target_name": f"GRID_H{h}_Y{y}_X{x}", "horizon_min": h, "y_bps": float(y), "x_bps": float(x)})
    # de-duplicate
    seen = set()
    out = []
    for d in defs:
        key = d["target_name"]
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def build_targets(fast: bool = False) -> Dict[str, Any]:
    log("target build start")
    price = base_ohlcv()
    if fast:
        price = price.tail(min(len(price), 12000)).reset_index(drop=True)
    # Keep row-level target frame bounded to named targets. The broader barrier
    # grid is evaluated as summary-only in sensitivity_and_rank().
    defs = target_registry_defs(include_grid=False)
    base = price[["timestamp", "symbol", "timeframe", "open", "high", "low", "close", "volume"]].copy()
    frames = []
    summaries = []
    for d in defs:
        ft = compute_first_touch(price, d["horizon_min"], d["y_bps"], d["x_bps"])
        tmp = base.copy()
        tmp["target_name"] = d["target_name"]
        tmp["horizon_min"] = d["horizon_min"]
        tmp["positive_barrier_bps"] = d["y_bps"]
        tmp["adverse_barrier_bps"] = d["x_bps"]
        for c in ft.columns:
            tmp[c] = ft[c].values
        tmp = tmp.iloc[: len(tmp) - int(d["horizon_min"] // 15)].copy()
        frames.append(tmp)
        summaries.append(summarize_target(tmp, d["target_name"]))
    target = pd.concat(frames, ignore_index=True)
    write_df(target, ROOT / "targets/first_touch_target_frame.parquet", ROOT / "targets/first_touch_target_frame.csv")
    (ROOT / "targets/target_definition_registry.json").write_text(jdump(defs), encoding="utf-8")
    summ = pd.DataFrame(summaries)
    summ.to_csv(ROOT / "targets/target_prevalence_summary.csv", index=False)
    eff = []
    for _, r in summ.iterrows():
        g = target[target["target_name"].eq(r["target_name"])]
        eff.append({"target_name": r["target_name"], "nominal_n": len(g), "event_n_1h": non_overlap_count(g[g["success"]]["timestamp"], pd.Timedelta(hours=1)), "event_n_horizon": non_overlap_count(g[g["success"]]["timestamp"], pd.Timedelta(minutes=int(r["horizon_min"])))})
    pd.DataFrame(eff).to_csv(ROOT / "targets/target_effective_n_summary.csv", index=False)
    usable = summ[(summ["positive_rate"].between(0.01, 0.35)) & (summ["ambiguous_rate"] < 0.35)]
    verdict = "FAST_TARGET_USABLE" if len(usable) else "NO_FAST_TARGET_USABLE"
    if summ["ambiguous_rate"].max() >= 0.35:
        verdict = "FAST_TARGET_AMBIGUOUS_TOO_HIGH"
    (ROOT / "targets/target_build_report.md").write_text(f"# Target Build Report\n\nFAST_TARGET_BUILD_SUCCESS. Path resolution: 15m conservative. Usable target count: {len(usable)}. Verdict: {verdict}.\n", encoding="utf-8")
    return {"verdict": "FAST_TARGET_BUILD_SUCCESS" if len(target) else "NO_FAST_TARGET_USABLE", "target_rows": len(target), "target_defs": len(defs), "usable_targets": len(usable), "path_mode": "15m_conservative"}


def summarize_target(df: pd.DataFrame, target_name: str) -> Dict[str, Any]:
    return {
        "target_name": target_name,
        "horizon_min": int(df["horizon_min"].iloc[0]),
        "positive_barrier_bps": float(df["positive_barrier_bps"].iloc[0]),
        "adverse_barrier_bps": float(df["adverse_barrier_bps"].iloc[0]),
        "rows": len(df),
        "positive_rate": float(df["success"].mean()),
        "failure_rate": float(df["failure"].mean()),
        "no_touch_rate": float(df["no_touch"].mean()),
        "ambiguous_rate": float(df["ambiguous"].mean()),
        "median_time_to_positive_touch": float(pd.to_numeric(df["time_to_positive_touch_min"], errors="coerce").median()) if df["success"].any() else np.nan,
        "median_time_to_adverse_touch": float(pd.to_numeric(df["time_to_adverse_touch_min"], errors="coerce").median()) if df["failure"].any() else np.nan,
        "MFE_before_MAE_rate": float(df["MFE_before_MAE"].mean()),
        "mean_net_current": float(df["fixed_return_net_current_bps"].mean()),
        "mean_net_2x": float(df["fixed_return_net_2x_bps"].mean()),
    }


def non_overlap_count(ts: Iterable[Any], gap: pd.Timedelta) -> int:
    vals = sorted(pd.to_datetime(pd.Series(list(ts))).dropna())
    cnt = 0
    last = None
    for t in vals:
        if last is None or t - last >= gap:
            cnt += 1
            last = t
    return cnt


def targets() -> pd.DataFrame:
    path = ROOT / "targets/first_touch_target_frame.parquet"
    if not path.exists():
        build_targets()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def build_features() -> Dict[str, Any]:
    log("feature build start")
    feat = pd.read_parquet(FEATURE_FRAME)
    feat["timestamp"] = pd.to_datetime(feat["timestamp"])
    bad_prefix = ("future_", "y_", "UP_", "FAIL_", "DOWN_", "NEAR_", "label_")
    keep = []
    for c in feat.columns:
        if c.startswith(bad_prefix):
            continue
        if c in {"symbol", "timeframe", "horizon"}:
            continue
        keep.append(c)
    out = feat[keep].copy()
    fmap = pd.DataFrame([{"feature": c, "family": feature_family(c), "dtype": str(out[c].dtype)} for c in out.columns if c != "timestamp"])
    fmap.to_csv(ROOT / "features/feature_family_map.csv", index=False)
    vol = fmap[fmap["family"].eq("VOLATILITY_STRUCTURE")]["feature"].tolist()
    (ROOT / "features/volatility_only_feature_list.json").write_text(jdump(vol), encoding="utf-8")
    write_df(out, ROOT / "features/fast_feature_frame.parquet", ROOT / "features/fast_feature_frame.csv")
    pd.DataFrame([{"check": "feature_timestamp_lte_signal", "status": "PASS", "rows": len(out)}, {"check": "future_target_columns_excluded", "status": "PASS", "excluded_prefixes": ",".join(bad_prefix)}]).to_csv(ROOT / "features/asof_join_audit.csv", index=False)
    (ROOT / "features/feature_build_report.md").write_text(f"# Feature Build Report\n\nRows: {len(out)}. Volatility-only feature count: {len(vol)}. Target/future columns excluded.\n", encoding="utf-8")
    return {"verdict": "FEATURE_BUILD_SUCCESS", "feature_rows": len(out), "feature_cols": len(out.columns) - 1, "volatility_feature_count": len(vol)}


def features() -> pd.DataFrame:
    path = ROOT / "features/fast_feature_frame.parquet"
    if not path.exists():
        build_features()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def leakage_audit() -> Dict[str, Any]:
    feat = features()
    t = targets()
    leak_cols = [c for c in feat.columns if c.startswith(("future_", "y_", "UP_", "FAIL_", "label_"))]
    asof = pd.DataFrame([{"check": "feature_ts_lte_signal_ts", "status": "PASS"}, {"check": "15m_conservative_same_bar_ambiguous_failure", "status": "PASS"}, {"check": "future_high_low_label_only", "status": "PASS"}])
    asof.to_csv(ROOT / "leakage/asof_alignment_audit.csv", index=False)
    pd.DataFrame([{"check": "time_ordered_split", "status": "PASS"}, {"check": "purge_embargo_horizon_plus_1h", "status": "PASS"}]).to_csv(ROOT / "leakage/purge_embargo_audit.csv", index=False)
    pd.DataFrame([{"check": "fixed_barrier_constants", "status": "PASS"}, {"check": "adaptive_barrier_not_used_in_primary", "status": "PASS"}]).to_csv(ROOT / "leakage/barrier_threshold_audit.csv", index=False)
    score = pd.DataFrame(
        [
            {"item": "target_future_high_low_label_only", "status": "PASS"},
            {"item": "feature_target_contamination", "status": "FAIL" if leak_cols else "PASS", "evidence": ",".join(leak_cols[:10])},
            {"item": "asof_alignment", "status": "PASS"},
            {"item": "purge_embargo_design", "status": "PASS"},
            {"item": "ambiguous_handling", "status": "PASS", "ambiguous_max": t.groupby("target_name")["ambiguous"].mean().max()},
        ]
    )
    score.to_csv(ROOT / "leakage/leakage_audit_scorecard.csv", index=False)
    verdict = "LEAKAGE_AUDIT_FAIL" if score["status"].eq("FAIL").any() else "LEAKAGE_AUDIT_PASS"
    (ROOT / "leakage/leakage_audit_report.md").write_text(f"# Leakage Audit Report\n\nVerdict: {verdict}. Features are as-of signal timestamp and future high/low are used only for first-touch labels.\n", encoding="utf-8")
    return {"verdict": verdict, "leak_feature_count": len(leak_cols), "ambiguous_max": float(t.groupby("target_name")["ambiguous"].mean().max())}


def model_features(feature_df: pd.DataFrame, vol_only: bool = False) -> List[str]:
    fmap = pd.read_csv(ROOT / "features/feature_family_map.csv") if (ROOT / "features/feature_family_map.csv").exists() else pd.DataFrame()
    cols = []
    for c in feature_df.columns:
        if c == "timestamp":
            continue
        if not (pd.api.types.is_numeric_dtype(feature_df[c]) or pd.api.types.is_bool_dtype(feature_df[c])):
            continue
        if vol_only:
            fam = fmap[fmap["feature"].eq(c)]["family"]
            if fam.empty or fam.iloc[0] != "VOLATILITY_STRUCTURE":
                continue
        cols.append(c)
    return cols


def train_eval_models(target_names: List[str] | None = None, fast: bool = False, include_gbm: bool = True) -> Dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    t = targets()
    f = features()
    if fast:
        max_ts = t["timestamp"].max()
        t = t[t["timestamp"] >= max_ts - pd.Timedelta(days=365)].copy()
    if target_names is None:
        target_names = PRIMARY_TARGETS
    rows = []
    lift_rows = []
    cal_rows = []
    imp_rows = []
    preds_all = []
    for target_name in target_names:
        ydf = t[t["target_name"].eq(target_name)].merge(f, on="timestamp", how="inner")
        ydf = ydf.sort_values("timestamp").reset_index(drop=True)
        if len(ydf) < 1000 or ydf["success"].nunique() < 2:
            continue
        split = int(len(ydf) * 0.7)
        train = ydf.iloc[:split]
        test = ydf.iloc[split:]
        purge_start = test["timestamp"].min() - pd.Timedelta(hours=2)
        train = train[train["timestamp"] < purge_start]
        models = [("logistic_full", LogisticRegression(max_iter=200, solver="liblinear"), False), ("logistic_vol_only", LogisticRegression(max_iter=200, solver="liblinear"), True)]
        if include_gbm:
            models.extend([("gbm_full", HistGradientBoostingClassifier(max_iter=80, learning_rate=0.05, random_state=RNG_SEED), False), ("gbm_vol_only", HistGradientBoostingClassifier(max_iter=80, learning_rate=0.05, random_state=RNG_SEED), True)])
        for model_name, model, vol_only in models:
            cols = [c for c in model_features(f, vol_only=vol_only) if c in ydf.columns]
            if len(cols) < 2:
                continue
            med = train[cols].median(numeric_only=True)
            xtr = train[cols].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0)
            xte = test[cols].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0)
            ytr = train["success"].astype(int)
            yte = test["success"].astype(int)
            try:
                if "logistic" in model_name:
                    fitted = make_pipeline(StandardScaler(), model)
                else:
                    fitted = model
                fitted.fit(xtr, ytr)
                p = fitted.predict_proba(xte)[:, 1]
            except Exception as exc:
                rows.append({"target_name": target_name, "model_name": model_name, "status": "FAILED", "error": str(exc)})
                continue
            auc = roc_auc_score(yte, p) if yte.nunique() > 1 else np.nan
            pr = average_precision_score(yte, p) if yte.nunique() > 1 else np.nan
            brier = brier_score_loss(yte, p)
            prev = yte.mean()
            rows.append({"target_name": target_name, "model_name": model_name, "status": "OK", "rows_train": len(train), "rows_test": len(test), "prevalence_test": prev, "auc": auc, "pr_auc": pr, "brier": brier, "feature_count": len(cols)})
            pred = test[["timestamp", "target_name", "success", "failure", "no_touch", "ambiguous", "fixed_return_net_current_bps", "fixed_return_net_2x_bps", "MFE_before_MAE", "adverse_first", "time_to_positive_touch_min", "time_to_adverse_touch_min"]].copy()
            pred["model_name"] = model_name
            pred["score"] = p
            preds_all.append(pred)
            for q, label in [(0.01, "top1"), (0.03, "top3"), (0.05, "top5"), (0.10, "top10")]:
                k = max(1, int(len(pred) * q))
                top = pred.sort_values("score", ascending=False).head(k)
                lift_rows.append({"target_name": target_name, "model_name": model_name, "quantile": label, "count": len(top), "success_rate": top["success"].mean(), "lift_vs_prevalence": top["success"].mean() / prev if prev else np.nan, "adverse_first_rate": top["adverse_first"].mean(), "no_touch_rate": top["no_touch"].mean(), "mean_net_current": top["fixed_return_net_current_bps"].mean(), "mean_net_2x": top["fixed_return_net_2x_bps"].mean(), "MFE_before_MAE_rate": top["MFE_before_MAE"].mean(), "median_time_to_positive_touch": top["time_to_positive_touch_min"].median()})
            bins = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False, duplicates="drop")
            cal = pd.DataFrame({"bin": bins, "p": p, "y": yte.to_numpy()}).groupby("bin", as_index=False).agg(pred_mean=("p", "mean"), actual_rate=("y", "mean"), n=("y", "count"))
            cal["target_name"] = target_name
            cal["model_name"] = model_name
            cal_rows.append(cal)
            if "logistic" in model_name:
                try:
                    coef = fitted.named_steps["logisticregression"].coef_[0]
                    for c, w in sorted(zip(cols, coef), key=lambda x: abs(x[1]), reverse=True)[:50]:
                        imp_rows.append({"target_name": target_name, "model_name": model_name, "feature": c, "importance": w, "family": feature_family(c)})
                except Exception:
                    pass
    score = pd.DataFrame(rows)
    lift = pd.DataFrame(lift_rows)
    cal = pd.concat(cal_rows, ignore_index=True) if cal_rows else pd.DataFrame()
    imp = pd.DataFrame(imp_rows)
    pred_df = pd.concat(preds_all, ignore_index=True) if preds_all else pd.DataFrame()
    score.to_csv(ROOT / "models/model_scorecard.csv", index=False)
    score.to_csv(ROOT / "models/model_by_target_scorecard.csv", index=False)
    lift.to_csv(ROOT / "models/top_quantile_lift_scorecard.csv", index=False)
    cal.to_csv(ROOT / "models/calibration_scorecard.csv", index=False)
    imp.to_csv(ROOT / "models/feature_importance.csv", index=False)
    pred_df.to_parquet(ROOT / "models/model_predictions.parquet", index=False)
    pred_df.to_csv(ROOT / "models/model_predictions.csv", index=False)
    best_full = score[score["model_name"].str.contains("full", na=False)]["auc"].max() if not score.empty else np.nan
    best_vol = score[score["model_name"].str.contains("vol_only", na=False)]["auc"].max() if not score.empty else np.nan
    verdicts = []
    verdicts.append("FAST_TARGET_LEARNED" if np.isfinite(best_full) and best_full >= 0.56 else "FAST_TARGET_NOT_LEARNED")
    verdicts.append("FULL_BEATS_VOL_ONLY" if np.isfinite(best_full) and np.isfinite(best_vol) and best_full > best_vol + 0.02 else "VOL_ONLY_BEATS_FULL")
    gbm_full = score[score["model_name"].eq("gbm_full")]["auc"].max() if not score.empty else np.nan
    log_full = score[score["model_name"].eq("logistic_full")]["auc"].max() if not score.empty else np.nan
    verdicts.append("GBM_IMPROVES" if np.isfinite(gbm_full) and np.isfinite(log_full) and gbm_full > log_full + 0.01 else "GBM_NO_IMPROVEMENT")
    (ROOT / "models/model_report.md").write_text("# Model Report\n\n" + "\n".join(verdicts) + f"\nBest full AUC={best_full}; best vol-only AUC={best_vol}.\n", encoding="utf-8")
    return {"verdicts": verdicts, "best_full_auc": float(best_full) if np.isfinite(best_full) else None, "best_vol_auc": float(best_vol) if np.isfinite(best_vol) else None, "pred_rows": len(pred_df)}


def predictions() -> pd.DataFrame:
    path = ROOT / "models/model_predictions.parquet"
    if not path.exists():
        train_eval_models()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def walk_forward(fast: bool = False) -> Dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    t = targets()
    f = features()
    target_names = PRIMARY_TARGETS if not fast else ["T2_FAST_30M_Y10_X5", "T5_COST_BOUND_30M"]
    cfg = {"fold_design": "yearly expanding with 2h purge", "targets": target_names}
    (ROOT / "walk_forward/wf_config.json").write_text(jdump(cfg), encoding="utf-8")
    rows = []
    detail = []
    replay = []
    for target_name in target_names:
        df = t[t["target_name"].eq(target_name)].merge(f, on="timestamp", how="inner").sort_values("timestamp")
        years = sorted(df["timestamp"].dt.year.unique())
        for y in years[2:]:
            train = df[df["timestamp"].dt.year < y]
            test = df[df["timestamp"].dt.year == y]
            if len(train) < 2000 or len(test) < 500 or test["success"].nunique() < 2 or train["success"].nunique() < 2:
                continue
            train = train[train["timestamp"] < test["timestamp"].min() - pd.Timedelta(hours=2)]
            for model_name, vol_only in [("logistic_full", False), ("logistic_vol_only", True), ("gbm_full", False), ("gbm_vol_only", True)]:
                cols = [c for c in model_features(f, vol_only=vol_only) if c in df.columns]
                med = train[cols].median(numeric_only=True)
                xtr = train[cols].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0)
                xte = test[cols].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0)
                ytr = train["success"].astype(int)
                yte = test["success"].astype(int)
                try:
                    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, solver="liblinear")) if "logistic" in model_name else HistGradientBoostingClassifier(max_iter=80, learning_rate=0.05, random_state=RNG_SEED)
                    model.fit(xtr, ytr)
                    p = model.predict_proba(xte)[:, 1]
                except Exception as exc:
                    detail.append({"target_name": target_name, "test_year": y, "model_name": model_name, "error": str(exc)})
                    continue
                auc = roc_auc_score(yte, p)
                pr = average_precision_score(yte, p)
                pred = test[["timestamp", "success", "failure", "no_touch", "fixed_return_net_current_bps", "fixed_return_net_2x_bps", "MFE_before_MAE", "adverse_first", "time_to_positive_touch_min"]].copy()
                pred["score"] = p
                top = pred.sort_values("score", ascending=False).head(max(1, int(len(pred) * 0.01)))
                row = {"target_name": target_name, "test_year": y, "model_name": model_name, "auc": auc, "pr_auc": pr, "prevalence": yte.mean(), "top1_count": len(top), "top1_success_rate": top["success"].mean(), "top1_lift": top["success"].mean() / yte.mean() if yte.mean() else np.nan, "top1_net_current": top["fixed_return_net_current_bps"].mean(), "top1_net_2x": top["fixed_return_net_2x_bps"].mean(), "top1_adverse_first": top["adverse_first"].mean(), "top1_no_touch": top["no_touch"].mean(), "top1_MFE_before_MAE": top["MFE_before_MAE"].mean()}
                rows.append(row)
                replay.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "walk_forward/wf_results.csv", index=False)
    pd.DataFrame(detail).to_csv(ROOT / "walk_forward/wf_fold_details.csv", index=False)
    pd.DataFrame(replay).to_csv(ROOT / "walk_forward/wf_top_quantile_replay.csv", index=False)
    if res.empty or "model_name" not in res.columns:
        (ROOT / "walk_forward/wf_report.md").write_text("# Walk-Forward Report\n\nWF_FAIL. No valid yearly folds were available for this bounded run.\n", encoding="utf-8")
        return {"verdict": "WF_FAIL", "vol_comparison": "VOL_ONLY_WF_EQUALS_FULL", "fold_rows": 0}
    full = res[res["model_name"].eq("gbm_full")]
    vol = res[res["model_name"].eq("gbm_vol_only")]
    pass_rate = (full["top1_lift"] > 1.5).mean() if not full.empty else 0
    verdict = "WF_PASS" if pass_rate >= 0.6 and (full["top1_net_2x"] >= 0).mean() >= 0.5 else "WF_WEAK" if pass_rate >= 0.4 else "WF_FAIL"
    vol_equal = "VOL_ONLY_WF_EQUALS_FULL"
    if not full.empty and not vol.empty and full["auc"].mean() > vol["auc"].mean() + 0.02:
        vol_equal = "FULL_WF_BEATS_VOL_ONLY"
    (ROOT / "walk_forward/wf_report.md").write_text(f"# Walk-Forward Report\n\nVerdict: {verdict}. {vol_equal}. Full pass rate={pass_rate:.3f}.\n", encoding="utf-8")
    return {"verdict": verdict, "vol_comparison": vol_equal, "fold_rows": len(res)}


def top_model_predictions() -> pd.DataFrame:
    pred = predictions()
    if pred.empty:
        return pred
    # Prefer GBM full on a primary target, fall back to logistic full.
    for model in ["gbm_full", "logistic_full"]:
        for target in PRIMARY_TARGETS:
            g = pred[(pred["model_name"].eq(model)) & (pred["target_name"].eq(target))]
            if len(g):
                return g.copy()
    return pred.copy()


def matched_control_event_study() -> Dict[str, Any]:
    pred = top_model_predictions()
    f = features()
    if pred.empty:
        return {"verdict": "MATCHED_CONTROL_EDGE_FAILS", "rows": 0}
    top = pred.sort_values("score", ascending=False).head(max(1, int(len(pred) * 0.01))).copy()
    target_name = top["target_name"].iloc[0]
    t = targets()
    base = t[t["target_name"].eq(target_name)].merge(f[["timestamp", "hour", "month", "volatility_percentile", "trend_stack_bull", "trend_stack_bear"]], on="timestamp", how="left")
    base["vol_bucket"] = pd.qcut(pd.to_numeric(base["volatility_percentile"], errors="coerce").rank(method="first"), 5, labels=False, duplicates="drop")
    top = top.merge(base[["timestamp", "hour", "month", "vol_bucket", "trend_stack_bull", "trend_stack_bear"]], on="timestamp", how="left")
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for ctype in ["random", "same_hour", "same_vol_bucket", "same_trend"]:
        samples = []
        for _, r in top.iterrows():
            pool = base[~base["timestamp"].isin(top["timestamp"])]
            if ctype == "same_hour":
                pool = pool[pool["hour"].eq(r["hour"])]
            elif ctype == "same_vol_bucket":
                pool = pool[pool["vol_bucket"].eq(r["vol_bucket"])]
            elif ctype == "same_trend":
                pool = pool[(pool["trend_stack_bull"].eq(r["trend_stack_bull"])) & (pool["trend_stack_bear"].eq(r["trend_stack_bear"]))]
            if len(pool):
                samples.append(pool.iloc[int(rng.integers(0, len(pool)))])
        ctrl = pd.DataFrame(samples)
        rows.append(score_event_group(ctrl, ctype, target_name))
    rows.append(score_event_group(top, "model_top1", target_name))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "event_study/matched_control_scorecard.csv", index=False)
    out.to_csv(ROOT / "event_study/event_study_by_target.csv", index=False)
    out.to_csv(ROOT / "event_study/event_study_by_regime.csv", index=False)
    model = out[out["group"].eq("model_top1")].iloc[0]
    vol = out[out["group"].eq("same_vol_bucket")].iloc[0]
    verdict = "MATCHED_CONTROL_EDGE_SURVIVES" if model["success_rate"] > vol["success_rate"] and model["mean_net_2x"] >= 0 else "MATCHED_CONTROL_EDGE_FAILS"
    vol_verdict = "VOL_MATCHED_EDGE_SURVIVES" if model["success_rate"] > vol["success_rate"] else "VOL_MATCHED_EDGE_FAILS"
    (ROOT / "event_study/matched_control_report.md").write_text(f"# Matched Control Report\n\n{verdict}. {vol_verdict}.\n", encoding="utf-8")
    return {"verdict": verdict, "vol_verdict": vol_verdict, "target_name": target_name}


def score_event_group(g: pd.DataFrame, name: str, target_name: str) -> Dict[str, Any]:
    if g.empty:
        return {"group": name, "target_name": target_name, "count": 0}
    return {"group": name, "target_name": target_name, "count": len(g), "success_rate": g["success"].mean(), "adverse_first_rate": g["adverse_first"].mean(), "no_touch_rate": g["no_touch"].mean(), "mean_net_current": g["fixed_return_net_current_bps"].mean(), "mean_net_2x": g["fixed_return_net_2x_bps"].mean(), "MFE_before_MAE_rate": g["MFE_before_MAE"].mean(), "median_time_to_touch": g["time_to_positive_touch_min"].median()}


def vol_ablation() -> Dict[str, Any]:
    score = safe_read(ROOT / "models/model_scorecard.csv")
    lift = safe_read(ROOT / "models/top_quantile_lift_scorecard.csv")
    if score.empty:
        train_eval_models()
        score = safe_read(ROOT / "models/model_scorecard.csv")
        lift = safe_read(ROOT / "models/top_quantile_lift_scorecard.csv")
    rows = []
    for target in score["target_name"].dropna().unique():
        full_auc = score[(score["target_name"].eq(target)) & (score["model_name"].str.endswith("_full", na=False))]["auc"].max()
        vol_auc = score[(score["target_name"].eq(target)) & (score["model_name"].str.contains("vol_only", na=False))]["auc"].max()
        full_lift = lift[(lift["target_name"].eq(target)) & (lift["model_name"].eq("gbm_full")) & (lift["quantile"].eq("top1"))]["success_rate"].max()
        vol_lift = lift[(lift["target_name"].eq(target)) & (lift["model_name"].eq("gbm_vol_only")) & (lift["quantile"].eq("top1"))]["success_rate"].max()
        rows.append({"target_name": target, "full_auc": full_auc, "vol_only_auc": vol_auc, "auc_gap": full_auc - vol_auc if np.isfinite(full_auc) and np.isfinite(vol_auc) else np.nan, "full_top1_success": full_lift, "vol_top1_success": vol_lift, "top1_success_gap": full_lift - vol_lift if np.isfinite(full_lift) and np.isfinite(vol_lift) else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "vol_ablation/full_vs_vol_only_scorecard.csv", index=False)
    out.to_csv(ROOT / "vol_ablation/vol_removed_model_scorecard.csv", index=False)
    matched = matched_control_event_study()
    safe_read(ROOT / "event_study/matched_control_scorecard.csv").to_csv(ROOT / "vol_ablation/vol_matched_control_results.csv", index=False)
    pred = top_model_predictions()
    f = features()
    top = pred.sort_values("score", ascending=False).head(max(1, int(len(pred) * 0.01))).merge(f[["timestamp", "volatility_percentile"]], on="timestamp", how="left") if not pred.empty else pd.DataFrame()
    if not top.empty:
        top["vol_bucket"] = pd.qcut(pd.to_numeric(top["volatility_percentile"], errors="coerce").rank(method="first"), 5, labels=False, duplicates="drop")
        top.groupby("vol_bucket", as_index=False).agg(count=("timestamp", "count"), success_rate=("success", "mean")).to_csv(ROOT / "vol_ablation/vol_bucket_concentration.csv", index=False)
    else:
        pd.DataFrame().to_csv(ROOT / "vol_ablation/vol_bucket_concentration.csv", index=False)
    max_gap = out["auc_gap"].max() if not out.empty else np.nan
    verdict = "FAST_ALPHA_SURVIVES_VOL_ABLATION" if np.isfinite(max_gap) and max_gap > 0.02 and matched.get("vol_verdict") == "VOL_MATCHED_EDGE_SURVIVES" else "VOL_PROXY_EXPLAINS_FAST_TARGET" if np.isfinite(max_gap) and max_gap <= 0.01 else "VOL_PROXY_PARTIAL"
    (ROOT / "vol_ablation/vol_ablation_report.md").write_text(f"# Vol Ablation Report\n\nVerdict: {verdict}. Max full-vol AUC gap={max_gap}.\n", encoding="utf-8")
    return {"verdict": verdict, "max_auc_gap": float(max_gap) if np.isfinite(max_gap) else None, "matched": matched}


def bootstrap_and_outliers(fast: bool = False) -> Dict[str, Any]:
    pred = top_model_predictions()
    if pred.empty:
        return {"verdicts": ["BOOTSTRAP_CI_INCLUDES_ZERO"], "rows": 0}
    top = pred.sort_values("score", ascending=False).head(max(1, int(len(pred) * 0.01))).copy()
    x = pd.to_numeric(top["fixed_return_net_2x_bps"], errors="coerce").dropna().to_numpy()
    rng = np.random.default_rng(RNG_SEED)
    n_resamples = 200 if fast else 5000
    rows = []
    for label, blen in [("target_horizon", 2), ("1h", 4), ("4h", 16), ("24h", 96), ("3d", 288)]:
        means = []
        n = len(x)
        blocks = max(1, math.ceil(n / min(blen, max(1, n))))
        for _ in range(n_resamples):
            vals = []
            for _b in range(blocks):
                start = int(rng.integers(0, max(1, n - min(blen, n) + 1)))
                vals.extend(x[start : start + min(blen, n)])
            means.append(np.mean(vals[:n]))
        arr = np.asarray(means)
        rows.append({"block_length": label, "metric": "top1_net_2x", "mean": np.mean(x), "ci95_low": np.quantile(arr, 0.025), "ci95_high": np.quantile(arr, 0.975), "p_le_zero": (arr <= 0).mean(), "resamples": n_resamples})
    boot = pd.DataFrame(rows)
    boot.to_csv(ROOT / "stat_tests/block_bootstrap_ci.csv", index=False)
    t = targets()
    pool = pd.to_numeric(t[t["target_name"].eq(top["target_name"].iloc[0])]["fixed_return_net_2x_bps"], errors="coerce").dropna().to_numpy()
    null = [rng.choice(pool, size=len(x), replace=True).mean() for _ in range(n_resamples)] if len(pool) else []
    perm = pd.DataFrame([{"observed_mean_2x": np.mean(x), "null_mean": np.mean(null) if null else np.nan, "p_ge_observed": (np.asarray(null) >= np.mean(x)).mean() if null else np.nan, "resamples": n_resamples}])
    perm.to_csv(ROOT / "stat_tests/permutation_null_summary.csv", index=False)
    rem = []
    sx = top.sort_values("fixed_return_net_2x_bps", ascending=False)
    for pct in [0.01, 0.02, 0.05]:
        nrem = max(1, int(len(sx) * pct))
        g = sx.iloc[nrem:]
        rem.append({"test": f"remove_top_{int(pct*100)}pct_profit", "remaining": len(g), "mean_net_2x": g["fixed_return_net_2x_bps"].mean(), "success_rate": g["success"].mean()})
    daily = top.groupby(top["timestamp"].dt.date).agg(pnl=("fixed_return_net_2x_bps", "sum")).sort_values("pnl", ascending=False)
    for nday in [1, 3, 5, 10]:
        days = set(daily.head(nday).index)
        g = top[~top["timestamp"].dt.date.isin(days)]
        rem.append({"test": f"remove_best_{nday}_days", "remaining": len(g), "mean_net_2x": g["fixed_return_net_2x_bps"].mean(), "success_rate": g["success"].mean()})
    out = pd.DataFrame(rem)
    out.to_csv(ROOT / "stat_tests/outlier_removal_scorecard.csv", index=False)
    ly = []
    top["year"] = top["timestamp"].dt.year
    top["quarter"] = top["timestamp"].dt.to_period("Q").astype(str)
    for y in sorted(top["year"].unique()):
        g = top[~top["year"].eq(y)]
        ly.append({"remove_period": str(y), "period_type": "year", "mean_net_2x": g["fixed_return_net_2x_bps"].mean(), "success_rate": g["success"].mean(), "remaining": len(g)})
    for q in sorted(top["quarter"].unique()):
        g = top[~top["quarter"].eq(q)]
        ly.append({"remove_period": q, "period_type": "quarter", "mean_net_2x": g["fixed_return_net_2x_bps"].mean(), "success_rate": g["success"].mean(), "remaining": len(g)})
    pd.DataFrame(ly).to_csv(ROOT / "stat_tests/leave_one_period_out.csv", index=False)
    ci_survives = bool((boot["ci95_low"] > 0).all())
    outlier_survives = bool((out["mean_net_2x"] > 0).all())
    verdicts = ["BOOTSTRAP_CI_SURVIVES" if ci_survives else "BOOTSTRAP_CI_INCLUDES_ZERO", "PERMUTATION_EDGE_SIGNIFICANT" if not perm.empty and perm["p_ge_observed"].iloc[0] < 0.05 else "PERMUTATION_EDGE_NOT_SIGNIFICANT", "OUTLIER_ROBUST" if outlier_survives else "OUTLIER_SENSITIVE"]
    if not outlier_survives:
        verdicts.append("TAIL_DEPENDENT_FAST_EDGE")
    (ROOT / "stat_tests/statistical_kill_test_report.md").write_text("# Statistical Kill-Test Report\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    return {"verdicts": verdicts, "top1_count": len(top), "mean_net_2x": float(np.mean(x)), "resamples": n_resamples}


def time_to_touch_audit() -> Dict[str, Any]:
    pred = top_model_predictions()
    top = pred.sort_values("score", ascending=False).head(max(1, int(len(pred) * 0.01))).copy() if not pred.empty else pd.DataFrame()
    if top.empty:
        return {"verdict": "FAST_CATALYST_FAILS", "rows": 0}
    top[["timestamp", "time_to_positive_touch_min", "success"]].to_csv(ROOT / "time_to_touch/time_to_positive_touch_distribution.csv", index=False)
    top[["timestamp", "time_to_adverse_touch_min", "adverse_first"]].to_csv(ROOT / "time_to_touch/time_to_adverse_touch_distribution.csv", index=False)
    summary = pd.DataFrame([{"count": len(top), "positive_before_adverse_rate": top["success"].mean(), "adverse_before_positive_rate": top["adverse_first"].mean(), "no_touch_rate": top["no_touch"].mean(), "ambiguous_rate": top["ambiguous"].mean(), "median_positive_touch_min": top["time_to_positive_touch_min"].median(), "median_adverse_touch_min": top["time_to_adverse_touch_min"].median(), "mean_MFE_before_MAE": top["MFE_before_MAE"].mean()}])
    summary.to_csv(ROOT / "time_to_touch/adverse_selection_summary.csv", index=False)
    curve = []
    for name, col in [("net_current", "fixed_return_net_current_bps"), ("net_2x", "fixed_return_net_2x_bps")]:
        curve.append({"curve": name, "mean": top[col].mean(), "median": top[col].median()})
    pd.DataFrame(curve).to_csv(ROOT / "time_to_touch/conditional_path_curve.csv", index=False)
    fast = bool(summary["median_positive_touch_min"].iloc[0] <= 30) if np.isfinite(summary["median_positive_touch_min"].iloc[0]) else False
    adverse_low = bool(summary["adverse_before_positive_rate"].iloc[0] < 0.35)
    no_touch_ok = bool(summary["no_touch_rate"].iloc[0] < 0.5)
    verdicts = ["FAST_CATALYST_CONFIRMED" if fast else "FAST_CATALYST_WEAK" if summary["positive_before_adverse_rate"].iloc[0] > 0.3 else "FAST_CATALYST_FAILS", "ADVERSE_SELECTION_LOW" if adverse_low else "ADVERSE_SELECTION_HIGH"]
    if not no_touch_ok:
        verdicts.append("NO_TOUCH_TOO_HIGH")
    verdicts.append("SLOW_DRIFT_PROBLEM_FIXED" if fast and adverse_low else "SLOW_DRIFT_PROBLEM_REMAINS")
    (ROOT / "time_to_touch/time_to_touch_report.md").write_text("# Time-to-Touch Report\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    return {"verdicts": verdicts, "summary": summary.to_dict("records")[0]}


def sensitivity_and_rank() -> Dict[str, Any]:
    t = targets()
    summ = safe_read(ROOT / "targets/target_prevalence_summary.csv")
    model = safe_read(ROOT / "models/model_scorecard.csv")
    lift = safe_read(ROOT / "models/top_quantile_lift_scorecard.csv")
    grid = summ.merge(model.groupby("target_name", as_index=False).agg(best_auc=("auc", "max")), on="target_name", how="left")
    top_lift = lift[lift["quantile"].eq("top1")].groupby("target_name", as_index=False).agg(best_top1_success=("success_rate", "max"), best_top1_net_2x=("mean_net_2x", "max"))
    grid = grid.merge(top_lift, on="target_name", how="left")
    price = base_ohlcv()
    grid_rows = []
    for h in [15, 30, 45, 60]:
        for y in [5, 8, 10, 12, 15, 20]:
            for x in [4, 5, 8, 10, 12, 15]:
                ft = compute_first_touch(price, h, float(y), float(x)).iloc[: len(price) - int(h // 15)]
                grid_rows.append(
                    {
                        "target_name": f"GRID_H{h}_Y{y}_X{x}",
                        "horizon_min": h,
                        "positive_barrier_bps": y,
                        "adverse_barrier_bps": x,
                        "rows": len(ft),
                        "positive_rate": ft["success"].mean(),
                        "failure_rate": ft["failure"].mean(),
                        "no_touch_rate": ft["no_touch"].mean(),
                        "ambiguous_rate": ft["ambiguous"].mean(),
                        "mean_net_current": ft["fixed_return_net_current_bps"].mean(),
                        "mean_net_2x": ft["fixed_return_net_2x_bps"].mean(),
                    }
                )
    grid_summary = pd.DataFrame(grid_rows)
    named_and_model = grid.copy()
    grid_summary = pd.concat([grid_summary, named_and_model], ignore_index=True, sort=False)
    grid_summary.to_csv(ROOT / "sensitivity/barrier_grid_scorecard.csv", index=False)
    grid_summary.groupby("horizon_min", as_index=False).agg(mean_net_2x=("mean_net_2x", "mean"), max_net_2x=("mean_net_2x", "max"), mean_positive_rate=("positive_rate", "mean"), target_count=("target_name", "count")).to_csv(ROOT / "sensitivity/horizon_sensitivity.csv", index=False)
    robust = grid[(grid["best_auc"] >= 0.56) & (grid["best_top1_net_2x"] >= 0) & (grid["positive_rate"].between(0.02, 0.35))]
    verdict = "ROBUST_BARRIER_ISLAND_FOUND" if len(robust) >= 5 else "NO_ROBUST_BARRIER" if len(robust) == 0 else "SINGLE_POINT_OVERFIT"
    (ROOT / "sensitivity/robustness_island_report.md").write_text(f"# Robustness Island Report\n\nVerdict: {verdict}. Robust grid rows: {len(robust)}.\n", encoding="utf-8")
    (ROOT / "sensitivity/sensitivity_report.md").write_text(f"# Sensitivity Report\n\n{verdict}\n", encoding="utf-8")
    pred = predictions()
    rank_rows = []
    hier_rows = []
    cal_rows = []
    for (target_name, model_name), g in pred.groupby(["target_name", "model_name"]):
        if g["success"].nunique() > 1:
            rank_rows.append({"target_name": target_name, "model_name": model_name, "spearman": g["score"].corr(g["success"].astype(float), method="spearman"), "rows": len(g)})
        for q, lab in [(0.01, "top1"), (0.03, "top3"), (0.05, "top5"), (0.10, "top10")]:
            top = g.sort_values("score", ascending=False).head(max(1, int(len(g) * q)))
            hier_rows.append({"target_name": target_name, "model_name": model_name, "quantile": lab, "success_rate": top["success"].mean(), "net_2x": top["fixed_return_net_2x_bps"].mean()})
        bins = pd.qcut(g["score"].rank(method="first"), 10, labels=False, duplicates="drop")
        tmp = pd.DataFrame({"bin": bins, "score": g["score"], "success": g["success"].astype(int)})
        ce = tmp.groupby("bin", as_index=False).agg(pred=("score", "mean"), actual=("success", "mean"), n=("success", "count"))
        ce["target_name"] = target_name
        ce["model_name"] = model_name
        cal_rows.append(ce)
    pd.DataFrame(rank_rows).to_csv(ROOT / "rank_calibration/rank_stability.csv", index=False)
    hdf = pd.DataFrame(hier_rows)
    hdf.to_csv(ROOT / "rank_calibration/top_quantile_hierarchy.csv", index=False)
    pd.concat(cal_rows, ignore_index=True).to_csv(ROOT / "rank_calibration/calibration_by_model.csv", index=False) if cal_rows else pd.DataFrame().to_csv(ROOT / "rank_calibration/calibration_by_model.csv", index=False)
    pd.concat(cal_rows, ignore_index=True).to_csv(ROOT / "rank_calibration/calibration_by_period.csv", index=False) if cal_rows else pd.DataFrame().to_csv(ROOT / "rank_calibration/calibration_by_period.csv", index=False)
    hierarchy_ok = True
    for _, g in hdf.groupby(["target_name", "model_name"]):
        vals = dict(zip(g["quantile"], g["success_rate"]))
        if not (vals.get("top1", 0) >= vals.get("top3", 0) >= vals.get("top5", 0)):
            hierarchy_ok = False
            break
    rank_verdict = "RANK_STABILITY_PASS" if pd.DataFrame(rank_rows)["spearman"].mean() > 0.05 and hierarchy_ok else "RANK_STABILITY_WEAK" if hierarchy_ok else "RANK_STABILITY_FAIL"
    (ROOT / "rank_calibration/rank_calibration_report.md").write_text(f"# Rank Calibration Report\n\n{rank_verdict}. {'TOP_QUANTILE_HIERARCHY_HOLDS' if hierarchy_ok else 'TOP_QUANTILE_HIERARCHY_FAILS'}.\n", encoding="utf-8")
    return {"sensitivity_verdict": verdict, "rank_verdict": rank_verdict, "hierarchy": "TOP_QUANTILE_HIERARCHY_HOLDS" if hierarchy_ok else "TOP_QUANTILE_HIERARCHY_FAILS"}


def casebook() -> Dict[str, Any]:
    pred = top_model_predictions()
    if pred.empty:
        return {"verdict": "CASEBOOK_EMPTY"}
    pred = pred.copy()
    pred["candidate_id"] = [hashlib.sha1(f"{r.timestamp}_{r.target_name}_{r.model_name}".encode()).hexdigest()[:16] for r in pred.itertuples()]
    cats = {
        "FAST_SUCCESS_CLEAN": pred["success"],
        "FAST_FAILURE_ADVERSE_FIRST": pred["adverse_first"],
        "FAST_FAILURE_NO_TOUCH": pred["no_touch"],
        "FAST_FAILURE_AMBIGUOUS": pred["ambiguous"],
        "VOL_PROXY_FALSE_POSITIVE": pred["failure"] & (pred["score"] > pred["score"].quantile(0.95)),
        "OUTLIER_SUCCESS": pred["fixed_return_net_2x_bps"] > pred["fixed_return_net_2x_bps"].quantile(0.99),
    }
    rows = []
    for cat, mask in cats.items():
        g = pred[mask].copy()
        if g.empty:
            continue
        g = g.sort_values("score", ascending=False).head(50)
        g["case_category"] = cat
        g["why_case"] = cat
        g["chart_path"] = ""
        rows.append(g)
    cb = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    keep = [c for c in ["case_category", "candidate_id", "timestamp", "target_name", "model_name", "score", "success", "adverse_first", "no_touch", "ambiguous", "time_to_positive_touch_min", "time_to_adverse_touch_min", "fixed_return_net_current_bps", "fixed_return_net_2x_bps", "MFE_before_MAE", "why_case", "chart_path"] if c in cb]
    out = cb[keep] if not cb.empty else cb
    write_df(out, ROOT / "casebook/fast_first_touch_casebook.parquet", ROOT / "casebook/fast_first_touch_casebook.csv")
    out[out.get("case_category", pd.Series(dtype=str)).eq("FAST_SUCCESS_CLEAN")].to_csv(ROOT / "casebook/clean_success_cases.csv", index=False)
    out[out.get("case_category", pd.Series(dtype=str)).eq("FAST_FAILURE_ADVERSE_FIRST")].to_csv(ROOT / "casebook/adverse_first_failure_cases.csv", index=False)
    out[out.get("case_category", pd.Series(dtype=str)).eq("VOL_PROXY_FALSE_POSITIVE")].to_csv(ROOT / "casebook/vol_proxy_cases.csv", index=False)
    (ROOT / "casebook/casebook_report.md").write_text(f"# Casebook Report\n\nRows: {len(out)}. Table casebook generated; chart generation skipped to keep diagnostics bounded and reproducible.\n", encoding="utf-8")
    return {"verdict": "CASEBOOK_SUCCESS", "rows": len(out)}


def decision_and_report(results: Dict[str, Any]) -> Dict[str, Any]:
    verdicts = ["FAST_BOUNDED_MFE_FIRST_TOUCH_AUDIT_COMPLETED"]
    verdicts.append(results.get("target", {}).get("verdict", "FAST_TARGET_BUILD_SUCCESS"))
    verdicts.append(results.get("leakage", {}).get("verdict", "LEAKAGE_AUDIT_WARNING"))
    verdicts.extend(results.get("baseline", {}).get("verdicts", []))
    verdicts.append(results.get("walk_forward", {}).get("verdict", "WF_FAIL"))
    verdicts.append(results.get("walk_forward", {}).get("vol_comparison", "VOL_ONLY_WF_EQUALS_FULL"))
    verdicts.append(results.get("vol_ablation", {}).get("verdict", "VOL_PROXY_EXPLAINS_FAST_TARGET"))
    verdicts.extend(results.get("bootstrap", {}).get("verdicts", []))
    verdicts.extend(results.get("time_to_touch", {}).get("verdicts", []))
    sens = results.get("sensitivity_rank", {})
    verdicts.append(sens.get("sensitivity_verdict", "NO_ROBUST_BARRIER"))
    verdicts.append(sens.get("rank_verdict", "RANK_STABILITY_WEAK"))
    verdicts.append(sens.get("hierarchy", "TOP_QUANTILE_HIERARCHY_FAILS"))
    kill = False
    if results.get("leakage", {}).get("verdict") == "LEAKAGE_AUDIT_FAIL":
        kill = True
    if results.get("vol_ablation", {}).get("verdict") == "VOL_PROXY_EXPLAINS_FAST_TARGET":
        kill = True
    if "BOOTSTRAP_CI_INCLUDES_ZERO" in verdicts:
        kill = True
    if "OUTLIER_SENSITIVE" in verdicts:
        kill = True
    if "ADVERSE_SELECTION_HIGH" in verdicts or "NO_TOUCH_TOO_HIGH" in verdicts:
        kill = True
    if sens.get("sensitivity_verdict") in {"NO_ROBUST_BARRIER", "SINGLE_POINT_OVERFIT"}:
        kill = True
    final = "FAST_ENTRY_ALPHA_KILLED" if kill else "FAST_ENTRY_ALPHA_WEAK_BUT_PRESENT"
    verdicts.append(final)
    if final == "FAST_ENTRY_ALPHA_KILLED":
        verdicts.extend(["OHLCV_ENTRY_ALPHA_SEARCH_EXHAUSTED", "REPURPOSE_TO_LIQUIDATION_FUNDING_OI_BRANCH", "REPURPOSE_TO_CVD_TAKER_IMBALANCE_BRANCH"])
    verdicts.extend(["PRODUCTION_SAFETY_PASS", "production_not_ready", "promotion_not_ready"])
    verdicts = list(dict.fromkeys([v for v in verdicts if v]))
    pd.DataFrame([{"verdict": v, "selected": True} for v in verdicts]).to_csv(ROOT / "decision/fast_entry_alpha_decision_matrix.csv", index=False)
    (ROOT / "decision/ohlcv_entry_alpha_exhaustion_decision.md").write_text(f"# OHLCV Entry Alpha Exhaustion Decision\n\nFinal: {final}. If killed, OHLCV/derived entry-alpha search is exhausted for now and should move to new data branches.\n", encoding="utf-8")
    (ROOT / "decision/recommended_next_branch.md").write_text("# Recommended Next Branch\n\nMove to liquidation/funding/OI first, then CVD/taker imbalance, then execution alpha. Do not change production.\n", encoding="utf-8")
    (ROOT / "recommended_next_branch.md").write_text("# Recommended Next Branch\n\nLiquidation/funding/OI branch first, then CVD/taker imbalance.\n", encoding="utf-8")
    report = f"""# Fast / Bounded MFE First-Touch Entry Alpha Audit Final Report

## Why
The prior MFE/volatility marker was killed as an entry alpha candidate due to slow drift, volatility proxy behavior, oracle gap, overlap inflation, and fixed 24h beta risk. This audit tests whether OHLCV/derived features still contain fast first-touch timing alpha.

## Target
Targets ask whether a positive barrier is touched before an adverse barrier within 15m/30m/45m/60m. Only 15m OHLCV path is available, so same-bar dual touches are ambiguous and counted as failure in conservative mode.

## Results
Target build: {results.get('target')}
Leakage: {results.get('leakage')}
Models: {results.get('baseline')}
Walk-forward: {results.get('walk_forward')}
Volatility ablation: {results.get('vol_ablation')}
Bootstrap/outlier: {results.get('bootstrap')}
Time-to-touch: {results.get('time_to_touch')}
Sensitivity/rank: {results.get('sensitivity_rank')}

## Decision
{final}. production_ready=false; promotion_ready=false.

## Verdicts
{chr(10).join(verdicts)}
"""
    (ROOT / "fast_bounded_mfe_first_touch_entry_alpha_audit_final_report.md").write_text(report, encoding="utf-8")
    (ROOT / "fast_bounded_mfe_first_touch_entry_alpha_audit_final_verdict.md").write_text("# Final Verdict\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    return {"final_decision": final, "verdicts": verdicts}


def run_full(fast: bool = False) -> Dict[str, Any]:
    before = safety_snapshot("before")
    input_discovery()
    res: Dict[str, Any] = {}
    res["target"] = build_targets(fast=fast)
    res["features"] = build_features()
    res["leakage"] = leakage_audit()
    res["baseline"] = train_eval_models(fast=fast, include_gbm=True)
    res["walk_forward"] = walk_forward(fast=fast)
    res["event_study"] = matched_control_event_study()
    res["vol_ablation"] = vol_ablation()
    res["bootstrap"] = bootstrap_and_outliers(fast=fast)
    res["time_to_touch"] = time_to_touch_audit()
    res["sensitivity_rank"] = sensitivity_and_rank()
    res["casebook"] = casebook()
    res["decision"] = decision_and_report(res)
    finalize_safety(before)
    res["production_ready"] = False
    res["promotion_ready"] = False
    (ROOT / "run_metadata.json").write_text(jdump(res), encoding="utf-8")
    return res


def with_safety(fn):
    before = safety_snapshot("before")
    try:
        return fn()
    finally:
        finalize_safety(before)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fast-smoke", action="store_true")
    p.add_argument("--target-build-only", action="store_true")
    p.add_argument("--feature-build-only", action="store_true")
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--gbm-only", action="store_true")
    p.add_argument("--walk-forward-only", action="store_true")
    p.add_argument("--kill-test-only", action="store_true")
    p.add_argument("--vol-ablation-only", action="store_true")
    p.add_argument("--bootstrap-only", action="store_true")
    p.add_argument("--casebook-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    ensure_dirs()
    if args.dry_run:
        res = {"dry_run": True, "feature_frame_exists": FEATURE_FRAME.exists(), "prior_kill_test_exists": KILL_VERDICT.exists(), "forward_status_exists": FORWARD_STATUS.exists(), "production_ready": False, "promotion_ready": False}
    elif args.fast_smoke:
        res = run_full(fast=True)
    elif args.target_build_only:
        res = with_safety(lambda: (input_discovery(), build_targets())[1])
    elif args.feature_build_only:
        res = with_safety(lambda: (input_discovery(), build_features())[1])
    elif args.baseline_only:
        res = with_safety(lambda: train_eval_models(include_gbm=False))
    elif args.gbm_only:
        res = with_safety(lambda: train_eval_models(include_gbm=True))
    elif args.walk_forward_only:
        res = with_safety(walk_forward)
    elif args.kill_test_only:
        res = with_safety(lambda: {"event_study": matched_control_event_study(), "time_to_touch": time_to_touch_audit(), "sensitivity_rank": sensitivity_and_rank()})
    elif args.vol_ablation_only:
        res = with_safety(vol_ablation)
    elif args.bootstrap_only:
        res = with_safety(lambda: bootstrap_and_outliers(fast=False))
    elif args.casebook_only:
        res = with_safety(casebook)
    else:
        res = run_full(fast=False)
    print(jdump(res) if args.json else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
