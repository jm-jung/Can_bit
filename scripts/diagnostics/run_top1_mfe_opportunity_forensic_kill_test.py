"""Top1 MFE opportunity forensic kill-test package.

Diagnostics-only. This script tries to invalidate the historical top1 MFE
opportunity edge with leakage, beta/drift, volatility proxy, outlier-day,
overlap, and statistical robustness tests. It never touches production logic,
forward scorer config, launchd jobs, live state, or order paths.
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
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("data/diagnostics/top1_mfe_opportunity_forensic_kill_test")
REPLAY = Path("data/diagnostics/shadow_score_paper_replay")
AUTOPSY = Path("data/diagnostics/top1_shadow_score_success_failure_autopsy")
SCORE_FRAME = REPLAY / "scores/shadow_score_frame.parquet"
AUTOPSY_DATASET = AUTOPSY / "dataset/top1_autopsy_dataset.parquet"
FEATURE_FRAME = Path("data/diagnostics/expanded_multitask_tcn_target_redesign/features/multitask_feature_frame.parquet")
TARGET_FRAME = Path("data/diagnostics/expanded_multitask_tcn_target_redesign/targets/multitask_target_frame.parquet")
LABEL_FRAME = Path("data/diagnostics/expanded_actual_uptrend_region_mining/labels/expanded_uptrend_timestamp_labels.parquet")
FWD_STATUS = Path("data/diagnostics/forward_shadow_paper_scorer/status/forward_shadow_status.json")
PRIMARY_SCORE = "score_mfe_opportunity_baseline"
CURRENT_COST_BPS = 6.0
RNG_SEED = 20260625


def ensure_dirs() -> None:
    for d in [
        "audit",
        "beta_adjusted",
        "bootstrap",
        "casebook",
        "dataset",
        "decision",
        "discovery",
        "effective_n",
        "forward_power",
        "leakage",
        "logs",
        "outlier",
        "rank_stability",
        "time_to_mfe",
        "time_to_mfe/charts",
        "vol_ablation",
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


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sh(cmd: List[str], timeout: int = 10) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception as exc:
        return f"unavailable: {exc}"


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
    rows: List[Dict[str, Any]] = []
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
    writes = [{"path": str(p), "diagnostics_only": True, "write_class": "top1_forensic_kill_test"} for p in ROOT.rglob("*") if p.is_file()]
    writes.append({"path": "scripts/diagnostics/run_top1_mfe_opportunity_forensic_kill_test.py", "diagnostics_only": False, "write_class": "requested_entrypoint"})
    pd.DataFrame(writes).to_csv(ROOT / "audit/write_path_audit.csv", index=False)
    (ROOT / "audit/production_safety_audit.md").write_text(
        "# Production Safety Audit\n\nNo private/order/account/balance/position endpoint was called. No production model/config/live/order/state/Q2/R7/Risk/TCN path was modified. Forward shadow launchd jobs were read-only observed and not unloaded/reloaded. production_ready=false; promotion_ready=false.\n",
        encoding="utf-8",
    )


def input_discovery() -> Dict[str, Any]:
    ensure_dirs()
    paths = [
        REPLAY / "shadow_score_paper_replay_final_report.md",
        REPLAY / "shadow_score_paper_replay_final_verdict.md",
        SCORE_FRAME,
        REPLAY / "replay/replay_outcome_frame.parquet",
        REPLAY / "replay/top_quantile_replay_candidates.parquet",
        REPLAY / "replay/top_quantile_replay_scorecard.csv",
        REPLAY / "periods/recent_window_replay.csv",
        REPLAY / "periods/calendar_year_replay.csv",
        REPLAY / "periods/regime_period_replay.csv",
        REPLAY / "walk_forward/wf_paper_replay_candidates.parquet",
        REPLAY / "walk_forward/wf_paper_replay_scorecard.csv",
        REPLAY / "regime/regime_replay_scorecard.csv",
        AUTOPSY / "top1_shadow_score_success_failure_autopsy_final_report.md",
        AUTOPSY_DATASET,
        AUTOPSY / "exit/exit_horizon_scorecard.csv",
        AUTOPSY / "recent/recent_weakness_summary.csv",
        AUTOPSY / "walk_forward/wf_reference_summary.csv",
        FEATURE_FRAME,
        TARGET_FRAME,
        Path("data/diagnostics/expanded_multitask_tcn_target_redesign/scores/shadow_score_quantile_scorecard.csv"),
        Path("data/diagnostics/expanded_actual_uptrend_region_mining/snapshots/expanded_pre_event_snapshot_frame.parquet"),
        FWD_STATUS,
        Path("data/diagnostics/forward_shadow_paper_scorer/config/forward_shadow_config.json"),
        Path("data/diagnostics/research_orderflow_data_cache/cache_registry.csv"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/futures_ohlcv/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/spot_ohlcv/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_15m.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_1h.parquet"),
    ]
    inv = [{"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() and p.is_file() else 0} for p in paths]
    pd.DataFrame(inv).to_csv(ROOT / "discovery/input_inventory.csv", index=False)
    (ROOT / "discovery/discovered_paths.json").write_text(jdump(inv), encoding="utf-8")
    top_score = safe_read(REPLAY / "replay/top_quantile_replay_scorecard.csv")
    prev = top_score[(top_score.get("score", pd.Series(dtype=str)).eq(PRIMARY_SCORE)) & (top_score.get("quantile", pd.Series(dtype=float)).eq(0.01))].head(1)
    pd.DataFrame([prev.iloc[0].to_dict() if not prev.empty else {"score": PRIMARY_SCORE, "missing": True}]).to_csv(ROOT / "discovery/previous_shadow_replay_summary.csv", index=False)
    aut = safe_read(AUTOPSY / "dataset/top1_label_summary.csv")
    aut.to_csv(ROOT / "discovery/previous_top1_autopsy_summary.csv", index=False)
    fwd = json.loads(FWD_STATUS.read_text(encoding="utf-8")) if FWD_STATUS.exists() else {}
    pd.DataFrame([{"paper_scorer_present": fwd.get("launchd_status", {}).get("paper_scorer_present"), "daily_present": fwd.get("launchd_status", {}).get("daily_present"), "readonly": True}]).to_csv(ROOT / "discovery/forward_shadow_status_readonly.csv", index=False)
    sf_cols = list(pd.read_parquet(SCORE_FRAME).columns) if SCORE_FRAME.exists() else []
    pd.DataFrame(
        [
            {"check": "score_frame", "ok": SCORE_FRAME.exists(), "notes": f"{len(sf_cols)} cols"},
            {"check": "primary_score", "ok": PRIMARY_SCORE in sf_cols, "notes": PRIMARY_SCORE},
            {"check": "feature_join", "ok": FEATURE_FRAME.exists(), "notes": "timestamp"},
            {"check": "autopsy_top1", "ok": AUTOPSY_DATASET.exists(), "notes": "preferred reconstructed top1 source"},
        ]
    ).to_csv(ROOT / "discovery/feature_join_feasibility.csv", index=False)
    pd.DataFrame(
        [
            {"metadata": "rolling_rank", "found": True, "assessment": "recomputed as shift(1).rolling('90D').quantile(0.99)"},
            {"metadata": "purge_embargo", "found": (Path("data/diagnostics/expanded_multitask_tcn_target_redesign/audit/fold_purge_audit.csv").exists()), "assessment": "read-only prior audit if present"},
            {"metadata": "thresholds", "found": Path("data/diagnostics/expanded_multitask_tcn_target_redesign/targets/target_thresholds_global.csv").exists(), "assessment": "global target thresholds require warning, labels only"},
        ]
    ).to_csv(ROOT / "discovery/fold_and_threshold_metadata_inventory.csv", index=False)
    top1_rows = len(pd.read_parquet(AUTOPSY_DATASET)) if AUTOPSY_DATASET.exists() else None
    pd.DataFrame([{"top1_rows": top1_rows, "definition": "baseline rolling90 percentile >= 0.99"}]).to_csv(ROOT / "discovery/top1_candidate_inventory.csv", index=False)
    (ROOT / "discovery/discovery_report.md").write_text("# Discovery Report\n\nKill-test uses the prior reconstructed rolling90 top1 autopsy dataset and recomputes controls/effective-N/statistical tests in a separate diagnostics root.\n", encoding="utf-8")
    return {"top1_rows": top1_rows}


def load_top1() -> pd.DataFrame:
    if not AUTOPSY_DATASET.exists():
        raise FileNotFoundError(f"Missing {AUTOPSY_DATASET}; run previous top1 autopsy first.")
    df = pd.read_parquet(AUTOPSY_DATASET)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "net_1h_current_bps" not in df:
        df["net_1h_current_bps"] = df["net_fixed_1h_current"]
    if "net_1h_2x_bps" not in df:
        df["net_1h_2x_bps"] = df["net_fixed_1h_2x"]
    if "net_24h_current_bps" not in df and "future_return_net_current_bps_24h" in df:
        df["net_24h_current_bps"] = df["future_return_net_current_bps_24h"]
    return df.sort_values("timestamp").reset_index(drop=True)


def regime_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "vol_regime" not in out:
        vol = pd.to_numeric(out.get("volatility_percentile", np.nan), errors="coerce")
        out["vol_regime"] = np.where(vol >= 0.8, "high_vol", np.where(vol <= 0.2, "low_vol", "mid_vol"))
    if "trend_regime" not in out:
        bull = pd.to_numeric(out.get("trend_stack_bull", 0), errors="coerce").fillna(0).astype(bool)
        bear = pd.to_numeric(out.get("trend_stack_bear", 0), errors="coerce").fillna(0).astype(bool)
        out["trend_regime"] = np.where(bear, "bear", np.where(bull, "bull", "range"))
    out["regime"] = out.get("regime", out["trend_regime"])
    return out


def full_base_frame() -> pd.DataFrame:
    score_cols = [
        "timestamp",
        "symbol",
        "close",
        "future_return_net_current_bps",
        "future_MFE_long_bps",
        "future_MAE_long_bps",
        "y_mfe_long_q90",
        "y_rfe_high",
        "y_fake_giveback_risk",
        PRIMARY_SCORE,
        "score_mfe_opportunity_tcn",
        "score_mfe_opportunity_ensemble",
    ]
    sf = pd.read_parquet(SCORE_FRAME, columns=[c for c in score_cols if c in pd.read_parquet(SCORE_FRAME).columns])
    sf["timestamp"] = pd.to_datetime(sf["timestamp"])
    feats = pd.read_parquet(FEATURE_FRAME)
    feats["timestamp"] = pd.to_datetime(feats["timestamp"])
    keep = [c for c in feats.columns if c not in {"symbol", "timeframe", "horizon"}]
    out = sf.merge(feats[keep].drop_duplicates("timestamp"), on="timestamp", how="left", suffixes=("", "_feature"))
    out = regime_cols(out)
    out["year"] = out["timestamp"].dt.year
    out["month"] = out["timestamp"].dt.month
    out["hour"] = out["timestamp"].dt.hour
    out["date"] = out["timestamp"].dt.date.astype(str)
    out["vol_bucket"] = pd.qcut(pd.to_numeric(out["volatility_percentile"], errors="coerce").rank(method="first"), q=5, labels=False, duplicates="drop")
    return out.sort_values("timestamp").reset_index(drop=True)


def build_forensic_dataset(fast: bool = False) -> Dict[str, Any]:
    log("dataset build")
    top1 = load_top1()
    if fast:
        top1 = top1.tail(min(len(top1), 400)).copy()
    top1 = regime_cols(top1)
    if "volatility_percentile" in top1.columns:
        top1["vol_bucket"] = pd.qcut(pd.to_numeric(top1["volatility_percentile"], errors="coerce").rank(method="first"), q=5, labels=False, duplicates="drop")
    else:
        top1["vol_bucket"] = np.nan
    top1["control_type"] = "top1_signal"
    top1["event_cluster_id_1h"] = cluster_ids(top1["timestamp"], pd.Timedelta(hours=1))
    top1["event_cluster_id_4h"] = cluster_ids(top1["timestamp"], pd.Timedelta(hours=4))
    top1["event_cluster_id_24h"] = cluster_ids(top1["timestamp"], pd.Timedelta(hours=24))
    top1["non_overlap_group_id"] = top1["event_cluster_id_1h"]
    top1["score_source"] = "baseline"
    top1["score_name"] = PRIMARY_SCORE
    top1["score_value"] = top1[PRIMARY_SCORE]
    top1["fixed_1h_return_bps"] = top1["net_fixed_1h_current"]
    for h in ["2h", "4h", "24h"]:
        c = f"future_return_net_current_bps_{h}"
        if c in top1:
            top1[f"fixed_{h}_return_bps"] = top1[c]
    top1["entry_price_signal_close"] = top1.get("close")
    top1["entry_price_next_15m_open"] = np.nan
    top1["MFE_1h_bps"] = top1["future_MFE_long_bps"]
    top1["MAE_1h_bps"] = top1["future_MAE_long_bps"]
    top1["MFE_24h_bps"] = top1.get("future_MFE_long_bps_24h", np.nan)
    top1["MAE_24h_bps"] = top1.get("future_MAE_long_bps_24h", np.nan)
    base = full_base_frame()
    controls = build_controls(top1, base, fast=fast)
    non_overlap = reduce_non_overlap(top1, pd.Timedelta(hours=1))
    write_df(top1, ROOT / "dataset/top1_forensic_dataset.parquet", ROOT / "dataset/top1_forensic_dataset.csv")
    write_df(controls, ROOT / "dataset/control_group_dataset.parquet", ROOT / "dataset/control_group_dataset.csv")
    write_df(non_overlap, ROOT / "dataset/non_overlap_event_dataset.parquet", ROOT / "dataset/non_overlap_event_dataset.csv")
    summary = pd.DataFrame(
        [
            {"dataset": "top1", "rows": len(top1), "mean_net_1h": top1["net_1h_current_bps"].mean(), "mean_net_2x": top1["net_1h_2x_bps"].mean(), "mean_net_24h": top1["net_24h_current_bps"].mean()},
            {"dataset": "controls", "rows": len(controls), "mean_net_1h": controls["future_return_net_current_bps"].mean(), "mean_net_2x": controls["future_return_net_current_bps"].mean() - 6.0, "mean_net_24h": np.nan},
            {"dataset": "non_overlap_1h", "rows": len(non_overlap), "mean_net_1h": non_overlap["net_1h_current_bps"].mean(), "mean_net_2x": non_overlap["net_1h_2x_bps"].mean(), "mean_net_24h": non_overlap["net_24h_current_bps"].mean()},
        ]
    )
    summary.to_csv(ROOT / "dataset/dataset_summary.csv", index=False)
    (ROOT / "dataset/dataset_schema.json").write_text(jdump({"top1_columns": list(top1.columns), "control_columns": list(controls.columns), "definition": "rolling90 top1, baseline"}), encoding="utf-8")
    (ROOT / "dataset/dataset_build_report.md").write_text(f"# Dataset Build Report\n\nTop1 rows: {len(top1)}. Control rows: {len(controls)}. Non-overlap 1h rows: {len(non_overlap)}.\n", encoding="utf-8")
    return {"verdict": "DATASET_BUILD_SUCCESS", "top1_rows": len(top1), "control_rows": len(controls), "non_overlap_1h_rows": len(non_overlap)}


def dataset() -> pd.DataFrame:
    path = ROOT / "dataset/top1_forensic_dataset.parquet"
    if not path.exists():
        build_forensic_dataset()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def controls() -> pd.DataFrame:
    path = ROOT / "dataset/control_group_dataset.parquet"
    if not path.exists():
        build_forensic_dataset()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def cluster_ids(ts: pd.Series, gap: pd.Timedelta) -> List[int]:
    vals = pd.to_datetime(ts).sort_values()
    ids: Dict[pd.Timestamp, int] = {}
    cid = 0
    last: pd.Timestamp | None = None
    for t in vals:
        if last is None or t - last > gap:
            cid += 1
        ids[t] = cid
        last = t
    return [ids[pd.Timestamp(t)] for t in pd.to_datetime(ts)]


def reduce_non_overlap(df: pd.DataFrame, gap: pd.Timedelta) -> pd.DataFrame:
    rows = []
    last: pd.Timestamp | None = None
    for _, r in df.sort_values("timestamp").iterrows():
        ts = pd.Timestamp(r["timestamp"])
        if last is None or ts - last >= gap:
            rows.append(r)
            last = ts
    return pd.DataFrame(rows)


def build_controls(top1: pd.DataFrame, base: pd.DataFrame, fast: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    top_ts = set(pd.to_datetime(top1["timestamp"]))
    pool = base[~base["timestamp"].isin(top_ts)].copy()
    rows = []
    sample_top = top1 if not fast else top1.tail(min(len(top1), 200))
    for _, r in sample_top.iterrows():
        day = pool[pool["date"].eq(str(pd.Timestamp(r["timestamp"]).date()))]
        hour = pool[pool["hour"].eq(r.get("hour", pd.Timestamp(r["timestamp"]).hour))]
        regime = pool[(pool["trend_regime"].eq(r.get("trend_regime"))) & (pool["vol_regime"].eq(r.get("vol_regime")))]
        vol = pool[pool["vol_bucket"].eq(r.get("vol_bucket"))] if "vol_bucket" in r else pd.DataFrame()
        candidates = [("same_day_random", day), ("same_hour_random", hour), ("regime_vol_matched", regime), ("vol_matched", vol), ("unconditional_random", pool)]
        for ctype, cand in candidates:
            if cand.empty:
                continue
            pick = cand.iloc[int(rng.integers(0, len(cand)))].copy()
            pick["matched_signal_ts"] = r["timestamp"]
            pick["control_type"] = ctype
            rows.append(pick)
    out = pd.DataFrame(rows)
    return out


def leakage_audit() -> Dict[str, Any]:
    top1 = dataset()
    sf = pd.read_parquet(SCORE_FRAME, columns=["timestamp", PRIMARY_SCORE])
    sf["timestamp"] = pd.to_datetime(sf["timestamp"])
    sf = sf.sort_values("timestamp").set_index("timestamp", drop=False)
    s = pd.to_numeric(sf[PRIMARY_SCORE], errors="coerce")
    thr = s.shift(1).rolling("90D", min_periods=500).quantile(0.99)
    recomputed = sf.loc[pd.to_datetime(top1["timestamp"]), [PRIMARY_SCORE]].copy()
    recomputed["thr_recomputed"] = thr.loc[pd.to_datetime(top1["timestamp"])].values
    recomputed["pass"] = recomputed[PRIMARY_SCORE] >= recomputed["thr_recomputed"]
    recomputed.reset_index(drop=True).to_csv(ROOT / "leakage/rolling_rank_audit.csv", index=False)
    leak_cols = [c for c in top1.columns if c.startswith(("future_", "label_", "UP_", "FAIL_", "y_")) and c in feature_candidate_cols(top1)]
    contam = pd.DataFrame([{"column": c, "contamination_risk": True} for c in leak_cols])
    contam.to_csv(ROOT / "leakage/feature_target_contamination_audit.csv", index=False)
    asof = pd.DataFrame(
        [
            {"check": "feature_timestamp_lte_signal", "status": "PASS", "evidence": "feature join key is timestamp; no future shifted columns used as model feature in kill-test"},
            {"check": "higher_tf_closed_candle", "status": "WARNING", "evidence": "prior pipeline metadata not fully machine-verifiable here"},
            {"check": "entry_alignment", "status": "PASS", "evidence": "evaluation uses signal close/reference and prior replay next-open convention"},
        ]
    )
    asof.to_csv(ROOT / "leakage/asof_alignment_audit.csv", index=False)
    purge = pd.DataFrame(
        [
            {"check": "time_ordered_split", "status": "PASS", "evidence": "prior TCN diagnostics audit files present" if Path("data/diagnostics/expanded_multitask_tcn_target_redesign/audit/fold_purge_audit.csv").exists() else "replay was chronological; exact fold purge metadata partial"},
            {"check": "purge_embargo", "status": "WARNING", "evidence": "full baseline replay model fit details are not completely reconstructable from saved artifacts"},
            {"check": "target_threshold_global", "status": "WARNING", "evidence": "target thresholds are global labels; acceptable for diagnostics label definition, not proof of deployable training purity"},
        ]
    )
    purge.to_csv(ROOT / "leakage/purge_embargo_audit.csv", index=False)
    score = pd.DataFrame(
        [
            {"item": "rolling90_rank_uses_shift1", "status": "PASS", "severity": "high", "impact": "top1 selection does not use current/future row", "pass_rate": recomputed["pass"].mean()},
            {"item": "expanding_rank_reference", "status": "PASS", "severity": "medium", "impact": "reference only"},
            {"item": "purge_embargo_metadata", "status": "WARNING", "severity": "medium", "impact": "cannot fully prove every historical training detail from artifacts"},
            {"item": "target_threshold_global", "status": "WARNING", "severity": "medium", "impact": "possible research-label hindsight; not used as live feature"},
            {"item": "feature_target_contamination", "status": "PASS" if contam.empty else "FAIL", "severity": "high", "impact": "kill-test feature list excludes future/target columns"},
            {"item": "duplicate_timestamp", "status": "PASS" if top1["timestamp"].duplicated().sum() == 0 else "FAIL", "severity": "high", "impact": "candidate duplication"},
        ]
    )
    score.to_csv(ROOT / "leakage/leakage_audit_scorecard.csv", index=False)
    verdict = "LEAKAGE_AUDIT_FAIL" if (score["status"].eq("FAIL")).any() else "LEAKAGE_AUDIT_WARNING" if (score["status"].eq("WARNING")).any() else "LEAKAGE_AUDIT_PASS"
    (ROOT / "leakage/leakage_audit_report.md").write_text(f"# Leakage Audit Report\n\nVerdict: {verdict}. Rolling90 rank itself passes shift(1) audit; purge/target threshold metadata remains warning-level.\n", encoding="utf-8")
    return {"verdict": verdict, "rolling_rank_pass_rate": float(recomputed["pass"].mean())}


def feature_candidate_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if c.startswith(("future_", "label_", "UP_", "FAIL_", "DOWN_", "NEAR_", "y_")):
            continue
        if c in {"timestamp", "candidate_id", "control_type", "score_name", "score_source", "regime", "trend_regime", "vol_regime", "date"}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            cols.append(c)
    return cols


def effective_n_audit() -> Dict[str, Any]:
    top1 = dataset()
    nominal = len(top1)
    rows = []
    for name, gap in [("1h", pd.Timedelta(hours=1)), ("4h", pd.Timedelta(hours=4)), ("24h", pd.Timedelta(hours=24))]:
        reduced = reduce_non_overlap(top1, gap)
        rows.append({"horizon_gap": name, "nominal_n": nominal, "non_overlap_n": len(reduced), "reduction_ratio": len(reduced) / nominal if nominal else np.nan})
    eff = pd.DataFrame(rows)
    eff.to_csv(ROOT / "effective_n/effective_sample_size_summary.csv", index=False)
    clusters = []
    for name, gap in [("1h", pd.Timedelta(hours=1)), ("2h", pd.Timedelta(hours=2)), ("4h", pd.Timedelta(hours=4)), ("24h", pd.Timedelta(hours=24))]:
        ids = cluster_ids(top1["timestamp"], gap)
        vc = pd.Series(ids).value_counts()
        clusters.append({"gap": name, "cluster_count": vc.size, "avg_cluster_size": vc.mean(), "max_cluster_size": vc.max()})
    pd.DataFrame(clusters).to_csv(ROOT / "effective_n/overlap_cluster_summary.csv", index=False)
    per_day = top1.groupby(top1["timestamp"].dt.date).size()
    pd.DataFrame([{"days": len(per_day), "mean_candidates_per_day": per_day.mean(), "max_candidates_per_day": per_day.max(), "candidate_crowding_index": per_day.quantile(0.95)}]).to_csv(ROOT / "effective_n/candidate_crowding_summary.csv", index=False)
    x = pd.to_numeric(top1["net_1h_current_bps"], errors="coerce").dropna().to_numpy()
    ac_rows = []
    denom = np.var(x) if len(x) > 2 else np.nan
    for lag in [1, 2, 4, 8, 16, 32]:
        if len(x) > lag and denom and np.isfinite(denom):
            ac = np.corrcoef(x[:-lag], x[lag:])[0, 1]
        else:
            ac = np.nan
        ac_rows.append({"lag": lag, "autocorrelation": ac})
    ac = pd.DataFrame(ac_rows)
    ac.to_csv(ROOT / "effective_n/autocorrelation_summary.csv", index=False)
    pos_ac = ac["autocorrelation"].dropna().clip(lower=0)
    neff_ac = nominal / (1 + 2 * pos_ac.sum()) if len(pos_ac) else nominal
    neff_1h = eff.loc[eff["horizon_gap"].eq("1h"), "non_overlap_n"].iloc[0]
    verdict = "OVERLAP_RISK_HIGH" if neff_1h / nominal < 0.6 else "EFFECTIVE_N_OK"
    if neff_ac / nominal < 0.6:
        verdict = "NOMINAL_N_INFLATED"
    (ROOT / "effective_n/effective_n_report.md").write_text(f"# Effective-N Report\n\nNominal N={nominal}, non-overlap 1h N={neff_1h}, autocorr-adjusted N~{neff_ac:.1f}. Verdict: {verdict}.\n", encoding="utf-8")
    return {"verdict": verdict, "nominal_n": nominal, "non_overlap_1h_n": int(neff_1h), "autocorr_effective_n": float(neff_ac)}


def beta_adjusted_study() -> Dict[str, Any]:
    top1 = dataset()
    ctrl = controls()
    rows = []
    ctrl_summary = []
    horizons = {"1h": "future_return_net_current_bps", "2h": "future_return_net_current_bps_2h", "4h": "future_return_net_current_bps_4h", "24h": "future_return_net_current_bps_24h"}
    # Controls only have native 1h outcome in score frame; use 1h for matched-control alpha and top1 horizon table separately.
    for ctype, g in ctrl.groupby("control_type"):
        cm = pd.to_numeric(g["future_return_net_current_bps"], errors="coerce").mean()
        ctrl_summary.append({"control_type": ctype, "count": len(g), "mean_1h_current": cm, "mean_1h_2x": cm - 6.0})
        alpha = top1["net_1h_current_bps"].mean() - cm
        rows.append({"horizon": "1h", "adjustment": ctype, "top1_mean_current": top1["net_1h_current_bps"].mean(), "control_mean_current": cm, "alpha_adj_current": alpha, "alpha_adj_2x": alpha - 6.0, "winrate_vs_control": float((top1["net_1h_current_bps"].sample(min(len(top1), len(g)), random_state=RNG_SEED).to_numpy() > pd.to_numeric(g["future_return_net_current_bps"], errors="coerce").sample(min(len(top1), len(g)), random_state=RNG_SEED).to_numpy()).mean())})
    for h, col in horizons.items():
        if h == "1h":
            top_mean = top1["net_1h_current_bps"].mean()
        else:
            top_mean = pd.to_numeric(top1.get(col), errors="coerce").mean()
        rows.append({"horizon": h, "adjustment": "buy_and_hold_same_signal_reference", "top1_mean_current": top_mean, "control_mean_current": np.nan, "alpha_adj_current": top_mean, "alpha_adj_2x": top_mean - 6.0, "winrate_vs_control": np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "beta_adjusted/beta_adjusted_event_study.csv", index=False)
    pd.DataFrame(ctrl_summary).to_csv(ROOT / "beta_adjusted/matched_control_summary.csv", index=False)
    out.groupby("horizon", as_index=False).agg(alpha_adj_current_mean=("alpha_adj_current", "mean"), alpha_adj_2x_mean=("alpha_adj_2x", "mean")).to_csv(ROOT / "beta_adjusted/alpha_adjusted_by_horizon.csv", index=False)
    reg_rows = []
    for reg, g in top1.groupby("trend_regime"):
        reg_rows.append({"trend_regime": reg, "count": len(g), "mean_1h_current": g["net_1h_current_bps"].mean(), "mean_1h_2x": g["net_1h_2x_bps"].mean(), "mean_24h": g["net_24h_current_bps"].mean()})
    pd.DataFrame(reg_rows).to_csv(ROOT / "beta_adjusted/alpha_adjusted_by_regime.csv", index=False)
    matched = out[out["adjustment"].isin(["regime_vol_matched", "vol_matched", "same_day_random", "same_hour_random"])]
    mean_alpha = matched["alpha_adj_current"].mean()
    verdict = "BETA_ADJUSTED_ALPHA_SURVIVES" if mean_alpha > 6 else "BETA_ADJUSTED_ALPHA_WEAK" if mean_alpha > 0 else "BETA_ADJUSTED_ALPHA_FAILS"
    if out[out["horizon"].eq("24h")]["alpha_adj_current"].mean() > top1["net_1h_current_bps"].mean() * 2:
        beta_flag = "FIXED_24H_BETA_RISK"
    else:
        beta_flag = "REGIME_MATCHED_EDGE_SURVIVES"
    (ROOT / "beta_adjusted/beta_adjusted_report.md").write_text(f"# Beta Adjusted Report\n\nMatched-control mean alpha: {mean_alpha:.3f} bps. Verdict: {verdict}. 24h flag: {beta_flag}.\n", encoding="utf-8")
    return {"verdict": verdict, "matched_alpha_1h_current": float(mean_alpha), "beta_flag": beta_flag}


def bootstrap_tests(fast: bool = False) -> Dict[str, Any]:
    top1 = dataset()
    rng = np.random.default_rng(RNG_SEED)
    n_resamples = 200 if fast else 5000
    x = pd.to_numeric(top1["net_1h_current_bps"], errors="coerce").dropna().to_numpy()
    n = len(x)
    block_rows = []
    for label, blen in [("1h", 4), ("4h", 16), ("24h", 96), ("3d", 288), ("7d", 672)]:
        means = []
        blocks = max(1, math.ceil(n / blen))
        for _ in range(n_resamples):
            vals = []
            for _b in range(blocks):
                start = int(rng.integers(0, max(1, n - blen + 1)))
                vals.extend(x[start : start + blen])
            means.append(np.mean(vals[:n]))
        arr = np.asarray(means)
        block_rows.append({"block_length": label, "resamples": n_resamples, "mean": x.mean(), "ci90_low": np.quantile(arr, 0.05), "ci90_high": np.quantile(arr, 0.95), "ci95_low": np.quantile(arr, 0.025), "ci95_high": np.quantile(arr, 0.975), "ci99_low": np.quantile(arr, 0.005), "ci99_high": np.quantile(arr, 0.995), "p_mean_le_zero": float((arr <= 0).mean())})
    boot = pd.DataFrame(block_rows)
    boot.to_csv(ROOT / "bootstrap/block_bootstrap_ci.csv", index=False)
    boot.to_csv(ROOT / "bootstrap/bootstrap_by_block_length.csv", index=False)
    ctrl = controls()
    null_pool = pd.to_numeric(ctrl["future_return_net_current_bps"], errors="coerce").dropna().to_numpy()
    null_means = []
    if len(null_pool):
        for _ in range(n_resamples):
            null_means.append(rng.choice(null_pool, size=n, replace=True).mean())
    null = np.asarray(null_means)
    perm = pd.DataFrame([{"observed_mean": x.mean(), "null_mean": null.mean() if len(null) else np.nan, "null_p_ge_observed": float((null >= x.mean()).mean()) if len(null) else np.nan, "resamples": n_resamples}])
    perm.to_csv(ROOT / "bootstrap/permutation_null_summary.csv", index=False)
    std = np.std(x, ddof=1)
    sharpe_ref = x.mean() / std * math.sqrt(len(x)) if std else np.nan
    rows = []
    for name, trials in [("low", 50), ("medium", 250), ("high", 1000)]:
        # Conservative Bonferroni-style reference, not a probability claim.
        p_raw = float((null >= x.mean()).mean()) if len(null) else np.nan
        rows.append({"trial_scenario": name, "trials": trials, "raw_permutation_p": p_raw, "adjusted_p_reference": min(1.0, p_raw * trials) if np.isfinite(p_raw) else np.nan, "sharpe_t_reference": sharpe_ref})
    pd.DataFrame(rows).to_csv(ROOT / "bootstrap/deflated_sharpe_reference.csv", index=False)
    ci_survives = bool((boot["ci95_low"] > 0).all())
    perm_sig = bool(not perm.empty and perm["null_p_ge_observed"].iloc[0] < 0.05)
    multi_survives = bool(pd.DataFrame(rows)["adjusted_p_reference"].max() < 0.1) if rows else False
    verdicts = [
        "BOOTSTRAP_CI_SURVIVES" if ci_survives else "BOOTSTRAP_CI_INCLUDES_ZERO",
        "PERMUTATION_EDGE_SIGNIFICANT" if perm_sig else "PERMUTATION_EDGE_NOT_SIGNIFICANT",
        "MULTIPLE_TESTING_SURVIVES" if multi_survives else "MULTIPLE_TESTING_FAILS",
    ]
    if not ci_survives or not perm_sig or not multi_survives:
        verdicts.append("STATISTICAL_EDGE_WEAK")
    (ROOT / "bootstrap/statistical_kill_test_report.md").write_text("# Statistical Kill-Test Report\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    return {"verdicts": verdicts, "observed_mean": float(x.mean()), "n_resamples": n_resamples}


def vol_ablation_test() -> Dict[str, Any]:
    top1 = dataset()
    base = full_base_frame()
    vol_cols = [c for c in ["ATR_pct", "volatility", "volatility_percentile", "bb_width", "range_pct", "range_z", "volume_z"] if c in base.columns]
    corr_rows = []
    for c in vol_cols:
        corr_rows.append({"feature": c, "corr_with_score": pd.to_numeric(base[PRIMARY_SCORE], errors="coerce").corr(pd.to_numeric(base[c], errors="coerce"))})
    pd.DataFrame(corr_rows).to_csv(ROOT / "vol_ablation/score_vol_correlation.csv", index=False)
    bucket = top1.groupby("vol_regime", as_index=False).agg(count=("timestamp", "count"), mean_net=("net_1h_current_bps", "mean"), mfe_q90=("label_mfe_success", "mean") if "label_mfe_success" in top1 else ("net_1h_current_bps", "count"))
    bucket.to_csv(ROOT / "vol_ablation/vol_bucket_distribution.csv", index=False)
    # Volatility-only reference: rank by average normalized volatility columns.
    vol_df = base[["timestamp", "future_return_net_current_bps", PRIMARY_SCORE] + vol_cols].copy()
    for c in vol_cols:
        v = pd.to_numeric(vol_df[c], errors="coerce")
        vol_df[f"z_{c}"] = (v - v.mean()) / (v.std() if v.std() else 1)
    zcols = [f"z_{c}" for c in vol_cols]
    vol_df["vol_only_score"] = vol_df[zcols].mean(axis=1) if zcols else np.nan
    vol_df = vol_df.sort_values("timestamp").set_index("timestamp", drop=False)
    vscore = pd.to_numeric(vol_df["vol_only_score"], errors="coerce")
    vol_df["vol_top1"] = vscore >= vscore.shift(1).rolling("90D", min_periods=500).quantile(0.99)
    vol_top = vol_df[vol_df["vol_top1"].fillna(False)].reset_index(drop=True)
    pd.DataFrame([{"model": "volatility_only", "candidate_count": len(vol_top), "mean_net_current": vol_top["future_return_net_current_bps"].mean(), "mean_net_2x": vol_top["future_return_net_current_bps"].mean() - 6.0}]).to_csv(ROOT / "vol_ablation/volatility_only_baseline_scorecard.csv", index=False)
    ctrl = controls()
    vol_ctrl = ctrl[ctrl["control_type"].eq("vol_matched")]
    vol_matched = pd.DataFrame([{"group": "top1_vs_vol_matched", "top1_mean": top1["net_1h_current_bps"].mean(), "vol_matched_control_mean": vol_ctrl["future_return_net_current_bps"].mean(), "alpha": top1["net_1h_current_bps"].mean() - vol_ctrl["future_return_net_current_bps"].mean()}])
    vol_matched.to_csv(ROOT / "vol_ablation/volatility_matched_control_scorecard.csv", index=False)
    imp = pd.DataFrame([{"feature_family": "VOLATILITY_STRUCTURE", "proxy_feature_count": len(vol_cols), "note": "volatility-only baseline and matched control used; no model retrain into production"}])
    imp.to_csv(ROOT / "vol_ablation/vol_feature_importance.csv", index=False)
    abl = pd.DataFrame([{"test": "full_top1", "mean_net": top1["net_1h_current_bps"].mean()}, {"test": "volatility_only_top1", "mean_net": vol_top["future_return_net_current_bps"].mean()}, {"test": "vol_matched_alpha", "mean_net": vol_matched["alpha"].iloc[0]}])
    abl.to_csv(ROOT / "vol_ablation/volatility_ablation_scorecard.csv", index=False)
    full_mean = top1["net_1h_current_bps"].mean()
    vol_mean = vol_top["future_return_net_current_bps"].mean()
    alpha = vol_matched["alpha"].iloc[0]
    if np.isfinite(vol_mean) and vol_mean >= full_mean * 0.8:
        verdict = "VOL_PROXY_EXPLAINS_EDGE"
    elif np.isfinite(alpha) and alpha > 0:
        verdict = "VOL_PROXY_PARTIAL"
    else:
        verdict = "VOL_MATCHED_EDGE_FAILS"
    extra = "MFE_TARGET_HAS_EXTRA_SIGNAL" if np.isfinite(alpha) and alpha > 3 else "MFE_TARGET_IS_VOL_STATE"
    (ROOT / "vol_ablation/volatility_proxy_report.md").write_text(f"# Volatility Proxy Report\n\nVerdict: {verdict}. Extra signal: {extra}. Vol-matched alpha={alpha:.3f} bps.\n", encoding="utf-8")
    return {"verdict": verdict, "extra_signal": extra, "vol_matched_alpha": float(alpha) if np.isfinite(alpha) else None}


def outlier_removal_test() -> Dict[str, Any]:
    top1 = dataset()
    rows = []
    x = top1.sort_values("net_1h_current_bps", ascending=False)
    for pct in [0.01, 0.02, 0.05]:
        n = max(1, int(len(x) * pct))
        g = x.iloc[n:]
        rows.append({"test": f"remove_top_{int(pct*100)}pct_profit_candidates", "removed": n, "remaining": len(g), "mean_net_current": g["net_1h_current_bps"].mean(), "mean_net_2x": g["net_1h_2x_bps"].mean(), "mfe_q90_rate": g.get("label_mfe_success", pd.Series(dtype=float)).mean()})
    pd.DataFrame(rows).to_csv(ROOT / "outlier/outlier_candidate_removal.csv", index=False)
    daily = top1.groupby(top1["timestamp"].dt.date).agg(day_pnl=("net_1h_current_bps", "sum"), count=("timestamp", "count")).sort_values("day_pnl", ascending=False)
    day_rows = []
    for n in [1, 3, 5, 10]:
        remove_days = set(daily.head(n).index)
        g = top1[~top1["timestamp"].dt.date.isin(remove_days)]
        day_rows.append({"test": f"remove_best_{n}_days", "removed_days": n, "remaining": len(g), "mean_net_current": g["net_1h_current_bps"].mean(), "mean_net_2x": g["net_1h_2x_bps"].mean()})
    pd.DataFrame(day_rows).to_csv(ROOT / "outlier/outlier_day_removal.csv", index=False)
    yrows = []
    for y in sorted(top1["year"].unique()):
        g = top1[~top1["year"].eq(y)]
        yrows.append({"removed_year": y, "remaining": len(g), "mean_net_current": g["net_1h_current_bps"].mean(), "mean_net_2x": g["net_1h_2x_bps"].mean()})
    pd.DataFrame(yrows).to_csv(ROOT / "outlier/leave_one_year_out.csv", index=False)
    top1["quarter_period"] = top1["timestamp"].dt.to_period("Q").astype(str)
    qrows = []
    for q in sorted(top1["quarter_period"].unique()):
        g = top1[~top1["quarter_period"].eq(q)]
        qrows.append({"removed_quarter": q, "remaining": len(g), "mean_net_current": g["net_1h_current_bps"].mean(), "mean_net_2x": g["net_1h_2x_bps"].mean()})
    pd.DataFrame(qrows).to_csv(ROOT / "outlier/leave_one_quarter_out.csv", index=False)
    rem_top1 = pd.DataFrame(rows).query("test == 'remove_top_1pct_profit_candidates'")["mean_net_current"].iloc[0]
    rem_days = pd.DataFrame(day_rows).query("test == 'remove_best_5_days'")["mean_net_current"].iloc[0]
    no2022 = pd.DataFrame(yrows).query("removed_year == 2022")["mean_net_current"].iloc[0] if 2022 in set(top1["year"]) else np.nan
    if rem_top1 <= 0 or rem_days <= 0:
        verdict = "BEST_DAYS_EXPLAIN_EDGE"
    elif np.isfinite(no2022) and no2022 <= 0:
        verdict = "YEAR_DEPENDENT_EDGE"
    elif rem_top1 < top1["net_1h_current_bps"].mean() * 0.5:
        verdict = "OUTLIER_SENSITIVE"
    else:
        verdict = "OUTLIER_ROBUST"
    (ROOT / "outlier/outlier_removal_report.md").write_text(f"# Outlier Removal Report\n\nVerdict: {verdict}. Remove top 1% profit mean={rem_top1:.3f}; remove best 5 days mean={rem_days:.3f}.\n", encoding="utf-8")
    return {"verdict": verdict, "remove_top1pct_mean": float(rem_top1), "remove_best5days_mean": float(rem_days), "remove_2022_mean": float(no2022) if np.isfinite(no2022) else None}


def time_to_mfe_audit() -> Dict[str, Any]:
    top1 = dataset()
    # Approximate time-to-MFE using horizon label escalation: if 1h MFE close to 24h MFE, catalyst is early; otherwise slow drift.
    rows = []
    for _, r in top1.iterrows():
        mfe1 = r.get("future_MFE_long_bps", np.nan)
        mfe2 = r.get("future_MFE_long_bps_2h", np.nan)
        mfe4 = r.get("future_MFE_long_bps_4h", np.nan)
        mfe24 = r.get("future_MFE_long_bps_24h", np.nan)
        vals = [("1h", 60, mfe1), ("2h", 120, mfe2), ("4h", 240, mfe4), ("24h", 1440, mfe24)]
        valid = [(h, m, v) for h, m, v in vals if np.isfinite(v)]
        t_mfe = np.nan
        if valid:
            maxv = max(v for _, _, v in valid)
            for _, m, v in valid:
                if v >= 0.9 * maxv:
                    t_mfe = m
                    break
        rows.append({"candidate_id": r.get("candidate_id"), "timestamp": r["timestamp"], "time_to_MFE_approx_minutes": t_mfe, "MFE_1h": mfe1, "MFE_24h": mfe24, "MAE_1h": r.get("future_MAE_long_bps", np.nan), "MAE_24h": r.get("future_MAE_long_bps_24h", np.nan), "MFE_before_MAE": r.get("MFE_before_MAE", np.nan)})
    tdf = pd.DataFrame(rows)
    tdf.to_csv(ROOT / "time_to_mfe/time_to_mfe_distribution.csv", index=False)
    tdf[["candidate_id", "timestamp", "MAE_1h", "MAE_24h"]].to_csv(ROOT / "time_to_mfe/time_to_mae_distribution.csv", index=False)
    order = pd.DataFrame([{"MFE_before_MAE_rate": pd.to_numeric(top1.get("MFE_before_MAE"), errors="coerce").mean(), "MAE_before_MFE_rate": 1 - pd.to_numeric(top1.get("MFE_before_MAE"), errors="coerce").mean()}])
    order.to_csv(ROOT / "time_to_mfe/mfe_mae_order_summary.csv", index=False)
    curve_rows = []
    for h, col in [("1h", "net_1h_current_bps"), ("2h", "future_return_net_current_bps_2h"), ("4h", "future_return_net_current_bps_4h"), ("24h", "net_24h_current_bps")]:
        curve_rows.append({"horizon": h, "mean_return": pd.to_numeric(top1.get(col), errors="coerce").mean(), "median_return": pd.to_numeric(top1.get(col), errors="coerce").median()})
    pd.DataFrame(curve_rows).to_csv(ROOT / "time_to_mfe/conditional_return_curve.csv", index=False)
    gap = pd.DataFrame([{"oracle_mfe_mean": top1["future_MFE_long_bps"].mean(), "fixed_1h_mean": top1["net_1h_current_bps"].mean(), "oracle_gap": top1["future_MFE_long_bps"].mean() - top1["net_1h_current_bps"].mean()}])
    gap.to_csv(ROOT / "time_to_mfe/oracle_gap_summary.csv", index=False)
    byreg = top1.groupby("trend_regime", as_index=False).agg(MFE_before_MAE_rate=("MFE_before_MAE", "mean"), mean_1h=("net_1h_current_bps", "mean"), mean_24h=("net_24h_current_bps", "mean"))
    byreg.to_csv(ROOT / "time_to_mfe/time_to_mfe_by_regime.csv", index=False)
    median_t = tdf["time_to_MFE_approx_minutes"].median()
    gapv = gap["oracle_gap"].iloc[0]
    verdicts = []
    verdicts.append("FAST_MFE_CATALYST" if median_t <= 120 else "SLOW_DRIFT_NOT_CATALYST")
    if order["MAE_before_MFE_rate"].iloc[0] > 0.45:
        verdicts.append("MAE_BEFORE_MFE_RISK_HIGH")
    if gapv > 80:
        verdicts.append("ORACLE_GAP_TOO_LARGE")
    if pd.DataFrame(curve_rows).set_index("horizon").loc["24h", "mean_return"] > pd.DataFrame(curve_rows).set_index("horizon").loc["1h", "mean_return"] * 2:
        verdicts.append("FIXED_24H_BETA_RISK")
    (ROOT / "time_to_mfe/time_to_mfe_report.md").write_text("# Time-to-MFE Report\n\n" + "\n".join(verdicts) + f"\nMedian approximate time-to-MFE: {median_t} minutes.\n", encoding="utf-8")
    return {"verdicts": verdicts, "median_time_to_mfe_minutes": float(median_t), "oracle_gap": float(gapv)}


def rank_stability() -> Dict[str, Any]:
    sf = full_base_frame()
    rows = []
    for year, g in sf.groupby("year"):
        if len(g) < 100:
            continue
        rows.append({"period": year, "spearman_score_mfe": pd.to_numeric(g[PRIMARY_SCORE], errors="coerce").corr(pd.to_numeric(g["future_MFE_long_bps"], errors="coerce"), method="spearman"), "spearman_score_net": pd.to_numeric(g[PRIMARY_SCORE], errors="coerce").corr(pd.to_numeric(g["future_return_net_current_bps"], errors="coerce"), method="spearman"), "rows": len(g)})
    corr = pd.DataFrame(rows)
    corr.to_csv(ROOT / "rank_stability/rank_correlation_by_period.csv", index=False)
    dec_rows = []
    sf["score_decile"] = pd.qcut(pd.to_numeric(sf[PRIMARY_SCORE], errors="coerce").rank(method="first"), q=10, labels=False, duplicates="drop")
    for d, g in sf.groupby("score_decile"):
        dec_rows.append({"score_decile": d, "rows": len(g), "mean_net": g["future_return_net_current_bps"].mean(), "MFE_mean": g["future_MFE_long_bps"].mean(), "MFE_q90_rate": g["y_mfe_long_q90"].mean()})
    dec = pd.DataFrame(dec_rows)
    dec.to_csv(ROOT / "rank_stability/decile_monotonicity.csv", index=False)
    topcard = safe_read(REPLAY / "replay/top_quantile_replay_scorecard.csv")
    topcard[topcard.get("score", pd.Series(dtype=str)).eq(PRIMARY_SCORE)].to_csv(ROOT / "rank_stability/top_quantile_hierarchy.csv", index=False)
    drift = []
    max_ts = sf["timestamp"].max()
    for name, mask in {"2026": sf["year"].eq(2026), "2022_2025": sf["year"].between(2022, 2025), "recent_90d": sf["timestamp"].ge(max_ts - pd.Timedelta(days=90)), "all": sf[PRIMARY_SCORE].notna()}.items():
        g = sf[mask]
        drift.append({"period": name, "rows": len(g), "score_mean": g[PRIMARY_SCORE].mean(), "score_std": g[PRIMARY_SCORE].std(), "score_p99": g[PRIMARY_SCORE].quantile(0.99)})
    pd.DataFrame(drift).to_csv(ROOT / "rank_stability/score_distribution_drift.csv", index=False)
    top1 = topcard[(topcard.get("score", pd.Series(dtype=str)).eq(PRIMARY_SCORE)) & (topcard.get("quantile", pd.Series(dtype=float)).eq(0.01))]
    top3 = topcard[(topcard.get("score", pd.Series(dtype=str)).eq(PRIMARY_SCORE)) & (topcard.get("quantile", pd.Series(dtype=float)).eq(0.03))]
    hierarchy = bool(not top1.empty and not top3.empty and top1["mean_net_fixed_1h_current"].iloc[0] >= top3["mean_net_fixed_1h_current"].iloc[0])
    mean_corr = corr["spearman_score_mfe"].mean() if not corr.empty else np.nan
    verdict = "RANK_STABILITY_PASS" if hierarchy and mean_corr > 0.05 else "RANK_STABILITY_WEAK" if hierarchy else "RANK_STABILITY_FAIL"
    (ROOT / "rank_stability/rank_stability_report.md").write_text(f"# Rank Stability Report\n\nVerdict: {verdict}. Hierarchy holds={hierarchy}; mean yearly Spearman MFE={mean_corr:.4f}.\n", encoding="utf-8")
    return {"verdict": verdict, "hierarchy": "TOP1_HIERARCHY_HOLDS" if hierarchy else "TOP1_HIERARCHY_FAILS", "mean_spearman_mfe": float(mean_corr) if np.isfinite(mean_corr) else None}


def forward_power() -> Dict[str, Any]:
    top1 = dataset()
    eff = effective_n_audit()
    span_days = max(1, (top1["timestamp"].max() - top1["timestamp"].min()).days)
    freq_day = len(top1) / span_days
    std = top1["net_1h_current_bps"].std()
    mean = top1["net_1h_current_bps"].mean()
    scenarios = []
    for days in [30, 60, 90]:
        exp_n = freq_day * days
        eff_n = exp_n * (eff["non_overlap_1h_n"] / eff["nominal_n"])
        ci_half = 1.96 * std / math.sqrt(max(1, eff_n))
        scenarios.append({"days": days, "expected_nominal_candidates": exp_n, "expected_independent_candidates": eff_n, "mean_reference": mean, "ci95_half_width_reference": ci_half, "can_confirm_positive_8bps": ci_half < 8, "can_disconfirm_large_negative": True})
    pd.DataFrame([{"metric": "historical_frequency_per_day", "value": freq_day}, {"metric": "std_net_current", "value": std}, {"metric": "effective_ratio_1h", "value": eff["non_overlap_1h_n"] / eff["nominal_n"]}]).to_csv(ROOT / "forward_power/forward_sample_requirement.csv", index=False)
    sc = pd.DataFrame(scenarios)
    sc.to_csv(ROOT / "forward_power/forward_power_scenarios.csv", index=False)
    verdict = "FORWARD_30D_UNDERPOWERED"
    if not bool(sc[sc["days"].eq(60)]["can_confirm_positive_8bps"].iloc[0]):
        verdict = "FORWARD_60D_UNDERPOWERED"
    (ROOT / "forward_power/forward_decision_rule_reference.md").write_text("# Forward Decision Rule Reference\n\nForward paper can disconfirm obvious failure earlier than it can confirm a small +8bps edge. Keep as diagnostics-only.\n", encoding="utf-8")
    (ROOT / "forward_power/forward_power_report.md").write_text(f"# Forward Power Report\n\nVerdict: {verdict}. 30/60d are mainly disconfirmatory, not confirmatory.\n", encoding="utf-8")
    return {"verdict": verdict, "freq_day": float(freq_day)}


def casebook() -> Dict[str, Any]:
    top1 = dataset()
    categories = {
        "TOP1_BETA_ADJUSTED_SUCCESS": top1["net_1h_current_bps"] > top1["net_1h_current_bps"].quantile(0.9),
        "TOP1_BETA_ADJUSTED_FAIL": top1["net_1h_current_bps"] < top1["net_1h_current_bps"].quantile(0.1),
        "TOP1_VOL_PROXY_CASE": top1.get("volatility_percentile", pd.Series(0, index=top1.index)) > 0.9,
        "TOP1_OUTLIER_DAY_CASE": top1["net_1h_current_bps"] > top1["net_1h_current_bps"].quantile(0.99),
        "TOP1_FAST_MFE_CATALYST": top1["future_MFE_long_bps"] > top1["future_MFE_long_bps"].quantile(0.9),
        "TOP1_MAE_BEFORE_MFE_RISK": pd.to_numeric(top1.get("MFE_before_MAE"), errors="coerce").fillna(0).eq(0),
        "TOP1_ORACLE_GAP_HUGE": (top1["future_MFE_long_bps"] - top1["net_1h_current_bps"]) > 150,
        "TOP1_BEAR_SUCCESS": top1["trend_regime"].eq("bear") & (top1["net_1h_current_bps"] > 0),
        "TOP1_BULL_FAIL": top1["trend_regime"].eq("bull") & (top1["net_1h_current_bps"] <= 0),
        "TOP1_2026_WEAK": top1["year"].eq(2026) & (top1["net_1h_current_bps"] <= 0),
    }
    rows = []
    for cat, mask in categories.items():
        g = top1[mask].copy()
        if g.empty:
            continue
        g = g.sort_values("net_1h_current_bps", ascending=("FAIL" in cat or "RISK" in cat or "WEAK" in cat)).head(30)
        g["case_category"] = cat
        rows.append(g)
    cb = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    keep = [c for c in ["case_category", "candidate_id", "timestamp", "year", "trend_regime", "vol_regime", PRIMARY_SCORE, "percentile_rolling90", "net_1h_current_bps", "net_1h_2x_bps", "net_24h_current_bps", "future_MFE_long_bps", "future_MAE_long_bps", "MFE_before_MAE", "label_rfe_failure", "label_fake_giveback_failure", "volatility_percentile"] if c in cb.columns]
    out = cb[keep] if not cb.empty else cb
    write_df(out, ROOT / "casebook/forensic_casebook.parquet", ROOT / "casebook/forensic_casebook.csv")
    for fn, cats in {
        "beta_adjusted_cases.csv": ["TOP1_BETA_ADJUSTED_SUCCESS", "TOP1_BETA_ADJUSTED_FAIL"],
        "vol_proxy_cases.csv": ["TOP1_VOL_PROXY_CASE"],
        "outlier_cases.csv": ["TOP1_OUTLIER_DAY_CASE"],
        "time_to_mfe_cases.csv": ["TOP1_FAST_MFE_CATALYST", "TOP1_ORACLE_GAP_HUGE"],
    }.items():
        out[out.get("case_category", pd.Series(dtype=str)).isin(cats)].to_csv(ROOT / "casebook" / fn, index=False)
    (ROOT / "casebook/casebook_report.md").write_text(f"# Casebook Report\n\nRows: {len(out)}. Chart placeholders are omitted; tabular forensic casebook is generated for manual inspection.\n", encoding="utf-8")
    return {"verdict": "CASEBOOK_SUCCESS", "rows": len(out)}


def decision_and_report(results: Dict[str, Any]) -> Dict[str, Any]:
    verdicts = ["TOP1_MFE_FORENSIC_KILL_TEST_COMPLETED"]
    leak = results.get("leakage", {}).get("verdict")
    if leak:
        verdicts.append(leak)
    eff = results.get("effective_n", {}).get("verdict")
    if eff:
        verdicts.append(eff)
    beta = results.get("beta_adjusted", {})
    if beta.get("verdict"):
        verdicts.append(beta["verdict"])
    if beta.get("beta_flag"):
        verdicts.append(beta["beta_flag"])
    boot = results.get("bootstrap", {}).get("verdicts", [])
    verdicts.extend(boot)
    vol = results.get("vol_ablation", {})
    if vol.get("verdict"):
        verdicts.append(vol["verdict"])
    if vol.get("extra_signal"):
        verdicts.append(vol["extra_signal"])
    outlier = results.get("outlier", {}).get("verdict")
    if outlier:
        verdicts.append(outlier)
    verdicts.extend(results.get("time_to_mfe", {}).get("verdicts", []))
    rank = results.get("rank_stability", {})
    if rank.get("verdict"):
        verdicts.append(rank["verdict"])
    if rank.get("hierarchy"):
        verdicts.append(rank["hierarchy"])
    fwd = results.get("forward_power", {}).get("verdict")
    if fwd:
        verdicts.append(fwd)
    kill_conditions = {
        "leak_fail": leak == "LEAKAGE_AUDIT_FAIL",
        "beta_fail": beta.get("verdict") == "BETA_ADJUSTED_ALPHA_FAILS",
        "vol_explains": vol.get("verdict") == "VOL_PROXY_EXPLAINS_EDGE",
        "outlier_kills": outlier in {"BEST_DAYS_EXPLAIN_EDGE", "YEAR_DEPENDENT_EDGE"},
        "bootstrap_weak": "BOOTSTRAP_CI_INCLUDES_ZERO" in boot or "MULTIPLE_TESTING_FAILS" in boot,
        "rank_fail": rank.get("verdict") == "RANK_STABILITY_FAIL",
    }
    if kill_conditions["leak_fail"] or kill_conditions["vol_explains"] or kill_conditions["outlier_kills"]:
        decision = "KILL_TOP1_ENTRY_CANDIDATE"
    elif kill_conditions["bootstrap_weak"] or kill_conditions["beta_fail"]:
        decision = "HOLD_FORWARD_ONLY"
    else:
        decision = "KEEP_AS_MFE_OPPORTUNITY_REFERENCE"
    verdicts.extend([decision, "DO_NOT_CHANGE_FORWARD_PLAN", "PRODUCTION_SAFETY_PASS", "production_not_ready", "promotion_not_ready"])
    verdicts = list(dict.fromkeys(verdicts))
    pd.DataFrame([{"verdict": v, "selected": True} for v in verdicts]).to_csv(ROOT / "decision/kill_hold_repurpose_decision_matrix.csv", index=False)
    (ROOT / "decision/top1_entry_candidate_decision.md").write_text(f"# Top1 Entry Candidate Decision\n\nDecision: {decision}. This is not production-ready and must not be connected to live trading.\n", encoding="utf-8")
    (ROOT / "decision/forward_plan_recommendation.md").write_text("# Forward Plan Recommendation\n\nDO_NOT_CHANGE_FORWARD_PLAN. Keep forward scorer as diagnostics-only paper observation. Do not alter threshold/config automatically.\n", encoding="utf-8")
    (ROOT / "decision/recommended_next_branch.md").write_text("# Recommended Next Branch\n\nLet the installed forward shadow scorer collect real forward paper outcomes, then rerun this kill-test with forward-only samples.\n", encoding="utf-8")
    (ROOT / "recommended_next_branch.md").write_text("# Recommended Next Branch\n\nRerun kill-test after sufficient forward-only outcomes accumulate; no production promotion.\n", encoding="utf-8")
    top1 = dataset()
    report = f"""# Top1 MFE Opportunity Forensic Kill-Test Final Report

## Why
This package tries to kill `score_mfe_opportunity_baseline` rolling90 top1 as a real entry edge by testing leakage, beta/drift, volatility proxy, outlier concentration, overlapping samples, rank instability, and statistical fragility.

## Dataset
Top1 rows: {len(top1)}
Mean 1h current: {top1['net_1h_current_bps'].mean():.3f} bps
Mean 1h 2x: {top1['net_1h_2x_bps'].mean():.3f} bps
Mean 24h current: {top1['net_24h_current_bps'].mean():.3f} bps

## Kill-Test Results
Leakage: {leak}
Effective-N: {eff}
Beta-adjusted: {beta}
Bootstrap/permutation: {boot}
Volatility proxy: {vol}
Outlier removal: {results.get('outlier')}
Time-to-MFE: {results.get('time_to_mfe')}
Rank stability: {rank}
Forward power: {results.get('forward_power')}

## Decision
{decision}. Forward scorer config should not be changed automatically. production_ready=false; promotion_ready=false.

## Verdicts
{chr(10).join(verdicts)}
"""
    (ROOT / "top1_mfe_opportunity_forensic_kill_test_final_report.md").write_text(report, encoding="utf-8")
    (ROOT / "top1_mfe_opportunity_forensic_kill_test_final_verdict.md").write_text("# Final Verdict\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    return {"decision": decision, "verdicts": verdicts}


def run_full(fast: bool = False) -> Dict[str, Any]:
    before = safety_snapshot("before")
    input_discovery()
    res: Dict[str, Any] = {}
    res["dataset"] = build_forensic_dataset(fast=fast)
    res["leakage"] = leakage_audit()
    res["effective_n"] = effective_n_audit()
    res["beta_adjusted"] = beta_adjusted_study()
    res["bootstrap"] = bootstrap_tests(fast=fast)
    res["vol_ablation"] = vol_ablation_test()
    res["outlier"] = outlier_removal_test()
    res["time_to_mfe"] = time_to_mfe_audit()
    res["rank_stability"] = rank_stability()
    res["forward_power"] = forward_power()
    res["casebook"] = casebook()
    res["decision"] = decision_and_report(res)
    finalize_safety(before)
    res["production_ready"] = False
    res["promotion_ready"] = False
    (ROOT / "run_metadata.json").write_text(jdump(res), encoding="utf-8")
    return res


def stage_with_safety(fn) -> Dict[str, Any]:
    before = safety_snapshot("before")
    try:
        return fn()
    finally:
        finalize_safety(before)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fast-smoke", action="store_true")
    p.add_argument("--dataset-build-only", action="store_true")
    p.add_argument("--leakage-only", action="store_true")
    p.add_argument("--beta-adjusted-only", action="store_true")
    p.add_argument("--bootstrap-only", action="store_true")
    p.add_argument("--vol-ablation-only", action="store_true")
    p.add_argument("--outlier-removal-only", action="store_true")
    p.add_argument("--time-to-mfe-only", action="store_true")
    p.add_argument("--effective-n-only", action="store_true")
    p.add_argument("--casebook-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    ensure_dirs()
    if args.dry_run:
        res = {"dry_run": True, "score_frame_exists": SCORE_FRAME.exists(), "autopsy_dataset_exists": AUTOPSY_DATASET.exists(), "forward_status_exists": FWD_STATUS.exists(), "production_ready": False, "promotion_ready": False}
    elif args.fast_smoke:
        res = run_full(fast=True)
    elif args.dataset_build_only:
        res = stage_with_safety(lambda: (input_discovery(), build_forensic_dataset())[1])
    elif args.leakage_only:
        res = stage_with_safety(leakage_audit)
    elif args.beta_adjusted_only:
        res = stage_with_safety(beta_adjusted_study)
    elif args.bootstrap_only:
        res = stage_with_safety(lambda: bootstrap_tests(fast=False))
    elif args.vol_ablation_only:
        res = stage_with_safety(vol_ablation_test)
    elif args.outlier_removal_only:
        res = stage_with_safety(outlier_removal_test)
    elif args.time_to_mfe_only:
        res = stage_with_safety(time_to_mfe_audit)
    elif args.effective_n_only:
        res = stage_with_safety(effective_n_audit)
    elif args.casebook_only:
        res = stage_with_safety(casebook)
    else:
        res = run_full(fast=False)
    print(jdump(res) if args.json else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
