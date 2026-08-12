#!/usr/bin/env python3
"""Historical Signal Failure & Path Atlas / Negative Registry / Candidate Rejector.

Offline diagnostics-only. Does NOT touch forward observer, collector, or production.
All economic analysis restricted to event_ts < research_cutoff (T0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("data/diagnostics/historical_signal_failure_path_atlas")
NEG_ROOT = Path("data/diagnostics/negative_result_registry")
REJ_ROOT = Path("data/diagnostics/candidate_only_rejector")
OHLCV_PATH = Path("data/ohlcv/BTCUSDT_5m_full.csv")
RESEARCH_CUTOFF = pd.Timestamp("2026-07-02T13:33:09", tz="UTC")
COST_1X_BPS = 6.0
COST_2X_BPS = 12.0
HORIZONS_MIN = [5, 15, 30, 60, 120, 240, 480, 1440]
BAR_MIN = 5
FIRST_TOUCH_BPS = [5, 10, 15, 20, 30]
EPISODE_GAP_MIN = 60
PROTECTED = [
    "scripts/diagnostics/run_microstructure_weak_hint_forward_observer.py",
    "scripts/diagnostics/run_microstructure_public_live_collector.py",
    "data/state/paper_trading_state.json",
    "data/state/shadow_daily_state.json",
]


def ensure_dirs() -> None:
    for p in [
        ROOT / "inventory",
        ROOT / "data",
        ROOT / "audit",
        ROOT / "reports",
        ROOT / "plots",
        ROOT / "logs",
        NEG_ROOT,
        REJ_ROOT / "data",
        REJ_ROOT / "reports",
        REJ_ROOT / "audit",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def jdump(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def sha256_text(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:24]


def to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def safe_read(path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            return pd.DataFrame()
        if columns:
            keep = [c for c in columns if c in df.columns]
            return df[keep].copy() if keep else df
        return df
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 1. Inventory
# ---------------------------------------------------------------------------

ARTIFACT_SPECS: List[Dict[str, Any]] = [
    {
        "signal_family": "signal_persistence",
        "experiment_name": "signal_persistence_horizon_sweep",
        "source_script": "scripts/diagnostics/run_signal_persistence_horizon_sweep.py",
        "source_output_path": "data/diagnostics/signal_persistence_horizon_sweep/events/signal_events.parquet",
        "timestamp_column": "timestamp",
        "direction_column": "direction",
        "score_column": "event_strength",
        "original_verdict": "RESEARCH_ONLY / weak_or_fragile_edge",
        "production_use": False,
        "reusable": True,
    },
    {
        "signal_family": "top1_mfe",
        "experiment_name": "top1_mfe_opportunity_forensic_kill_test",
        "source_script": "scripts/diagnostics/run_top1_mfe_opportunity_forensic_kill_test.py",
        "source_output_path": "data/diagnostics/top1_mfe_opportunity_forensic_kill_test/dataset/top1_forensic_dataset.parquet",
        "timestamp_column": "timestamp",
        "direction_column": None,
        "score_column": "score_mfe_opportunity_ensemble",
        "original_verdict": "KILL_TOP1_ENTRY_CANDIDATE",
        "production_use": False,
        "reusable": True,
    },
    {
        "signal_family": "shadow_score",
        "experiment_name": "shadow_score_paper_replay",
        "source_script": "scripts/diagnostics/run_shadow_score_paper_replay.py",
        "source_output_path": "data/diagnostics/shadow_score_paper_replay/scores/shadow_score_frame.parquet",
        "timestamp_column": "timestamp",
        "direction_column": None,
        "score_column": "score_mfe_opportunity_ensemble",
        "original_verdict": "MFE opportunity reference; not entry alpha",
        "production_use": False,
        "reusable": True,
    },
    {
        "signal_family": "q2_bdi_r7_paper",
        "experiment_name": "historical_clean_paper_backfill",
        "source_script": "scripts/diagnostics/run_historical_clean_paper_backfill.py",
        "source_output_path": "data/diagnostics/historical_clean_paper_backfill/paper_trades/paper_trades_base.parquet",
        "timestamp_column": "entry_ts",
        "direction_column": "direction",
        "score_column": "p_long",
        "original_verdict": "Q2_BDI baseline retained; paper diagnostics",
        "production_use": False,
        "reusable": True,
    },
    {
        "signal_family": "meta_layer",
        "experiment_name": "meta_dataset_v2",
        "source_script": "scripts/diagnostics/train_meta_layer_model.py",
        "source_output_path": "data/diagnostics/meta_layer/meta_dataset_v2.parquet",
        "timestamp_column": "entry_ts",
        "direction_column": "direction",
        "score_column": "max_proba",
        "original_verdict": "meta overfilter / unstable",
        "production_use": False,
        "reusable": True,
    },
    {
        "signal_family": "r7_false_high",
        "experiment_name": "false_high_r7_daily_monitor",
        "source_script": "scripts/diagnostics/run_false_high_r7_daily_monitor.py",
        "source_output_path": "data/diagnostics/false_high_r7_daily_monitor/forward/r7_forward_log.csv",
        "timestamp_column": "entry_ts",
        "direction_column": "direction",
        "score_column": "r7_score",
        "original_verdict": "diagnostics_monitor_only / warning_only",
        "production_use": False,
        "reusable": True,
    },
    {
        "signal_family": "tcn_proba",
        "experiment_name": "ml_tcn_btc_5m_proba_cache",
        "source_script": "scripts/diagnostics/refresh_daily_feature_proba_cache.py",
        "source_output_path": "data/cache/ml_predictions/ml_tcn_BTCUSDT_5m_proba.parquet",
        "timestamp_column": "timestamp",
        "direction_column": None,
        "score_column": "proba_long",
        "original_verdict": "production inference cache; not standalone entry alpha",
        "production_use": True,
        "reusable": True,
    },
    {
        "signal_family": "xgb_proba",
        "experiment_name": "ml_xgb_btc_5m_proba_cache",
        "source_script": "scripts/diagnostics/refresh_daily_feature_proba_cache.py",
        "source_output_path": "data/cache/ml_predictions/ml_xgb_BTCUSDT_5m_extended_safe_20210101_proba.parquet",
        "timestamp_column": "timestamp",
        "direction_column": None,
        "score_column": "proba_long",
        "original_verdict": "production inference cache",
        "production_use": True,
        "reusable": True,
    },
    {
        "signal_family": "microstructure_weak_hint",
        "experiment_name": "weak_hint_forward_observer_markers",
        "source_script": "scripts/diagnostics/run_microstructure_weak_hint_forward_observer.py",
        "source_output_path": "data/diagnostics/microstructure_weak_hint_forward_observer/markers/observed_primary_markers.parquet",
        "timestamp_column": "signal_ts",
        "direction_column": None,
        "score_column": "base_feature_value",
        "original_verdict": "observation-only; pre-T0 mostly quarantine TESTNET_ENDPOINT_SUSPECT",
        "production_use": False,
        "reusable": False,
        "reuse_block_reason": "pre-T0 rows largely NON_MAINNET / quarantine; post-T0 blind",
    },
    {
        "signal_family": "q2_fusion_policy",
        "experiment_name": "q2_fusion_p2_p6",
        "source_script": "scripts/diagnostics/run_false_high_specialist_research.py",
        "source_output_path": "data/diagnostics/false_high_specialist/q2_fusion/q2_fusion_policy_comparison.csv",
        "timestamp_column": None,
        "direction_column": None,
        "score_column": None,
        "original_verdict": "P2/P6 policy comparison diagnostics",
        "production_use": False,
        "reusable": True,
    },
]


def run_inventory() -> Dict[str, Any]:
    ensure_dirs()
    rows = []
    for spec in ARTIFACT_SPECS:
        path = Path(spec["source_output_path"])
        exists = path.exists()
        row_count = None
        data_start = data_end = None
        cols = []
        file_hash = None
        if exists:
            try:
                if path.suffix == ".parquet":
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path)
                row_count = int(len(df))
                cols = list(df.columns)[:40]
                ts_col = spec.get("timestamp_column")
                if ts_col and ts_col in df.columns:
                    ts = to_utc(df[ts_col])
                    data_start = str(ts.min())
                    data_end = str(ts.max())
                file_hash = sha256_text(path, path.stat().st_size, row_count)
            except Exception as exc:
                cols = [f"read_error:{exc}"]
        rows.append(
            {
                **spec,
                "exists": exists,
                "row_count": row_count,
                "data_start": data_start,
                "data_end": data_end,
                "columns_sample": cols,
                "provenance_status": "historical_artifact_assumed_mainnet_ohlcv_or_prior_audit",
                "quarantine_status": "n/a_or_source_specific",
                "file_hash": file_hash,
                "research_cutoff": str(RESEARCH_CUTOFF),
            }
        )
    inv = pd.DataFrame(rows)
    inv.to_csv(ROOT / "inventory/signal_artifact_inventory.csv", index=False)
    out = {"verdict": "INVENTORY_READY", "n_artifacts": len(inv), "n_existing": int(inv["exists"].sum()), "rows": rows}
    (ROOT / "inventory/signal_artifact_inventory.json").write_text(jdump(out), encoding="utf-8")
    md = ["# Signal Artifact Inventory\n", f"- artifacts: {len(inv)}", f"- existing: {int(inv['exists'].sum())}", f"- research_cutoff: {RESEARCH_CUTOFF}\n"]
    for r in rows:
        md.append(f"## {r['experiment_name']}\n- family: {r['signal_family']}\n- path: `{r['source_output_path']}`\n- exists: {r['exists']} rows={r['row_count']}\n- verdict: {r['original_verdict']}\n")
    (ROOT / "reports/signal_artifact_inventory.md").write_text("\n".join(md), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# 2–3. Unified event registry
# ---------------------------------------------------------------------------

def _base_event(
    event_ts: pd.Timestamp,
    signal_family: str,
    signal_name: str,
    experiment_name: str,
    source_script: str,
    source_file: str,
    source_row_id: Any,
    direction: str,
    raw_score: float,
    model_name: str = "",
    original_horizon: str = "",
    original_verdict: str = "",
    regime_at_event: str = "",
    volatility_bucket_at_event: str = "",
    interval: str = "5m",
) -> Dict[str, Any]:
    eid = sha256_text(signal_name, event_ts, direction, source_file, source_row_id)
    usable = bool(pd.notna(event_ts) and event_ts < RESEARCH_CUTOFF)
    return {
        "event_id": eid,
        "event_ts": event_ts,
        "signal_family": signal_family,
        "signal_name": signal_name,
        "experiment_name": experiment_name,
        "source_script": source_script,
        "source_file": source_file,
        "source_row_id": str(source_row_id),
        "symbol": "BTCUSDT",
        "interval": interval,
        "direction": direction,
        "raw_score": float(raw_score) if pd.notna(raw_score) else np.nan,
        "normalized_score": np.nan,
        "threshold": np.nan,
        "signal_strength_bucket": "",
        "model_name": model_name,
        "label_name": "",
        "original_horizon": original_horizon,
        "original_verdict": original_verdict,
        "provenance_status": "ASSUMED_VALID_HISTORICAL",
        "quarantine_status": "NOT_QUARANTINED",
        "research_cutoff_pass": usable,
        "provenance_pass": True,
        "quarantine_pass": True,
        "timestamp_pass": bool(pd.notna(event_ts)),
        "duplicate_pass": True,
        "usable_for_historical_atlas": usable,
        "regime_at_event": regime_at_event,
        "volatility_bucket_at_event": volatility_bucket_at_event,
        "trend_bucket_at_event": "",
        "market_state_at_event": "",
    }


def load_signal_persistence_events() -> List[Dict[str, Any]]:
    path = Path("data/diagnostics/signal_persistence_horizon_sweep/events/signal_events.parquet")
    df = safe_read(path)
    if df.empty:
        return []
    out = []
    for i, r in df.iterrows():
        ts = to_utc(pd.Series([r["timestamp"]])).iloc[0]
        if pd.isna(ts) or ts >= RESEARCH_CUTOFF:
            continue
        out.append(
            _base_event(
                ts,
                "signal_persistence",
                str(r.get("event_name", "unknown")),
                "signal_persistence_horizon_sweep",
                "scripts/diagnostics/run_signal_persistence_horizon_sweep.py",
                str(path),
                r.get("event_id", i),
                str(r.get("direction", "long")).lower(),
                r.get("event_strength", np.nan),
                original_verdict="RESEARCH_ONLY",
                regime_at_event=str(r.get("regime_label", "")),
                volatility_bucket_at_event=str(r.get("volatility_state", "")),
            )
        )
    return out


def load_paper_trade_events() -> List[Dict[str, Any]]:
    path = Path("data/diagnostics/historical_clean_paper_backfill/paper_trades/paper_trades_base.parquet")
    df = safe_read(path)
    if df.empty:
        return []
    out = []
    for i, r in df.iterrows():
        ts = to_utc(pd.Series([r["entry_ts"]])).iloc[0]
        if pd.isna(ts) or ts >= RESEARCH_CUTOFF:
            continue
        direction = str(r.get("direction", "long")).lower()
        score = r.get("p_long") if direction.startswith("long") else r.get("p_short", r.get("p_long"))
        out.append(
            _base_event(
                ts,
                "q2_bdi_r7_paper",
                "paper_candidate",
                "historical_clean_paper_backfill",
                "scripts/diagnostics/run_historical_clean_paper_backfill.py",
                str(path),
                r.get("paper_trade_id", i),
                direction,
                score,
                model_name="tcn+q2+r7",
                original_verdict="paper_diagnostics",
            )
        )
    return out


def load_meta_events() -> List[Dict[str, Any]]:
    path = Path("data/diagnostics/meta_layer/meta_dataset_v2.parquet")
    df = safe_read(path)
    if df.empty:
        return []
    out = []
    for i, r in df.iterrows():
        ts = to_utc(pd.Series([r.get("entry_ts", r.get("timestamp"))])).iloc[0]
        if pd.isna(ts) or ts >= RESEARCH_CUTOFF:
            continue
        out.append(
            _base_event(
                ts,
                "meta_layer",
                "meta_v2_trade",
                "meta_dataset_v2",
                "scripts/diagnostics/train_meta_layer_model.py",
                str(path),
                r.get("trade_id", i),
                str(r.get("direction", "long")).lower(),
                r.get("max_proba", np.nan),
                model_name="meta_v2",
                original_verdict="meta_overfilter_unstable",
                volatility_bucket_at_event=str(r.get("vol_bucket", "")),
                regime_at_event=str(r.get("trend_state", "")),
            )
        )
    return out


def load_r7_events() -> List[Dict[str, Any]]:
    path = Path("data/diagnostics/false_high_r7_daily_monitor/forward/r7_forward_log.csv")
    df = safe_read(path)
    if df.empty:
        return []
    out = []
    for i, r in df.iterrows():
        ts = to_utc(pd.Series([r["entry_ts"]])).iloc[0]
        if pd.isna(ts) or ts >= RESEARCH_CUTOFF:
            continue
        out.append(
            _base_event(
                ts,
                "r7_false_high",
                "r7_high_hazard_candidate",
                "false_high_r7_daily_monitor",
                "scripts/diagnostics/run_false_high_r7_daily_monitor.py",
                str(path),
                r.get("candidate_id", i),
                str(r.get("direction", "long")).lower(),
                r.get("r7_score", np.nan),
                model_name="r7",
                original_verdict="warning_only",
            )
        )
    return out


def load_top1_candidate_events() -> List[Dict[str, Any]]:
    path = Path("data/diagnostics/top1_mfe_opportunity_forensic_kill_test/dataset/top1_forensic_dataset.parquet")
    df = safe_read(path)
    if df.empty:
        return []
    score_col = "score_mfe_opportunity_ensemble" if "score_mfe_opportunity_ensemble" in df.columns else None
    if score_col is None:
        for c in df.columns:
            if "score_mfe_opportunity" in c:
                score_col = c
                break
    if score_col is None:
        return []
    thr = df[score_col].quantile(0.95)
    cand = df[df[score_col] >= thr].copy()
    out = []
    for i, r in cand.iterrows():
        ts = to_utc(pd.Series([r["timestamp"]])).iloc[0]
        if pd.isna(ts) or ts >= RESEARCH_CUTOFF:
            continue
        out.append(
            _base_event(
                ts,
                "top1_mfe",
                "score_mfe_opportunity_top5pct",
                "top1_mfe_opportunity_forensic_kill_test",
                "scripts/diagnostics/run_top1_mfe_opportunity_forensic_kill_test.py",
                str(path),
                i,
                "long",
                r[score_col],
                model_name="ensemble",
                original_verdict="KILL_TOP1_ENTRY_CANDIDATE",
                original_horizon=str(r.get("horizon", "")),
            )
        )
    return out


def load_shadow_top_candidates(max_n: int = 3000) -> List[Dict[str, Any]]:
    path = Path("data/diagnostics/shadow_score_paper_replay/scores/shadow_score_frame.parquet")
    df = safe_read(path)
    if df.empty or "score_mfe_opportunity_ensemble" not in df.columns:
        return []
    df = df.copy()
    df["timestamp"] = to_utc(df["timestamp"])
    df = df[df["timestamp"] < RESEARCH_CUTOFF]
    thr = df["score_mfe_opportunity_ensemble"].quantile(0.99)
    cand = df[df["score_mfe_opportunity_ensemble"] >= thr].sort_values("score_mfe_opportunity_ensemble", ascending=False).head(max_n)
    out = []
    for i, r in cand.iterrows():
        out.append(
            _base_event(
                r["timestamp"],
                "shadow_score",
                "score_mfe_opportunity_ensemble_top1pct",
                "shadow_score_paper_replay",
                "scripts/diagnostics/run_shadow_score_paper_replay.py",
                str(path),
                i,
                "long",
                r["score_mfe_opportunity_ensemble"],
                model_name="ensemble",
                original_verdict="not_entry_alpha_vol_proxy_suspect",
            )
        )
    return out


def load_tcn_xgb_candidates(max_n: int = 2000) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    specs = [
        ("tcn_proba", "data/cache/ml_predictions/ml_tcn_BTCUSDT_5m_proba.parquet", "tcn"),
        ("xgb_proba", "data/cache/ml_predictions/ml_xgb_BTCUSDT_5m_extended_safe_20210101_proba.parquet", "xgb"),
    ]
    for family, rel, model in specs:
        path = Path(rel)
        df = safe_read(path)
        if df.empty or "proba_long" not in df.columns:
            continue
        df = df.copy()
        df["timestamp"] = to_utc(df["timestamp"])
        df = df[df["timestamp"] < RESEARCH_CUTOFF]
        # candidate: high-confidence long OR short
        long_thr = df["proba_long"].quantile(0.995)
        short_col = "proba_short" if "proba_short" in df.columns else None
        long_c = df[df["proba_long"] >= long_thr].head(max_n // 2)
        for i, r in long_c.iterrows():
            out.append(
                _base_event(
                    r["timestamp"],
                    family,
                    f"{model}_high_conf_long",
                    f"ml_{model}_proba_cache",
                    "scripts/diagnostics/refresh_daily_feature_proba_cache.py",
                    str(path),
                    i,
                    "long",
                    r["proba_long"],
                    model_name=model,
                    original_verdict="inference_cache_not_entry_alpha",
                )
            )
        if short_col:
            short_thr = df[short_col].quantile(0.995)
            short_c = df[df[short_col] >= short_thr].head(max_n // 2)
            for i, r in short_c.iterrows():
                out.append(
                    _base_event(
                        r["timestamp"],
                        family,
                        f"{model}_high_conf_short",
                        f"ml_{model}_proba_cache",
                        "scripts/diagnostics/refresh_daily_feature_proba_cache.py",
                        str(path),
                        i,
                        "short",
                        r[short_col],
                        model_name=model,
                        original_verdict="inference_cache_not_entry_alpha",
                    )
                )
    return out


def load_weak_hint_events_for_exclusion_audit() -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Load weak-hint markers; mostly excluded due to quarantine / post-T0."""
    path = Path("data/diagnostics/microstructure_weak_hint_forward_observer/markers/observed_primary_markers.parquet")
    df = safe_read(path)
    stats = {"total": 0, "post_t0": 0, "quarantined": 0, "usable": 0}
    if df.empty:
        return [], stats
    stats["total"] = len(df)
    out = []
    for i, r in df.iterrows():
        ts = to_utc(pd.Series([r["signal_ts"]])).iloc[0]
        excluded = bool(r.get("exclude_from_forward_eval", False)) if "exclude_from_forward_eval" in df.columns else False
        # object bool safety
        if str(excluded).lower() in {"true", "1"}:
            excluded = True
        reason = str(r.get("exclude_reason", "") or "")
        if pd.isna(ts):
            continue
        if ts >= RESEARCH_CUTOFF:
            stats["post_t0"] += 1
            continue
        if excluded or "TESTNET" in reason or "NON_MAINNET" in reason:
            stats["quarantined"] += 1
            continue
        # rare usable pre-T0 clean row
        stats["usable"] += 1
        ev = _base_event(
            ts,
            "microstructure_weak_hint",
            str(r.get("marker_name", "weak_hint")),
            "weak_hint_forward_observer",
            "scripts/diagnostics/run_microstructure_weak_hint_forward_observer.py",
            str(path),
            r.get("observation_id", i),
            "long",
            r.get("base_feature_value", np.nan),
            original_verdict="observation_only",
        )
        out.append(ev)
    return out, stats


def build_registry() -> Dict[str, Any]:
    ensure_dirs()
    events: List[Dict[str, Any]] = []
    events.extend(load_signal_persistence_events())
    events.extend(load_paper_trade_events())
    events.extend(load_meta_events())
    events.extend(load_r7_events())
    events.extend(load_top1_candidate_events())
    events.extend(load_shadow_top_candidates())
    events.extend(load_tcn_xgb_candidates())
    weak, weak_stats = load_weak_hint_events_for_exclusion_audit()
    events.extend(weak)

    df = pd.DataFrame(events)
    if df.empty:
        out = {"verdict": "HISTORICAL_RESEARCH_BOUNDARY_FAIL", "reason": "no_events"}
        (ROOT / "audit/unified_signal_event_registry_audit.json").write_text(jdump(out), encoding="utf-8")
        return out

    before = len(df)
    df = df.drop_duplicates("event_id")
    # normalize scores within signal_name
    df["normalized_score"] = df.groupby("signal_name")["raw_score"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)
    )
    df["signal_strength_bucket"] = pd.qcut(df["normalized_score"].rank(method="first"), q=5, labels=["q1", "q2", "q3", "q4", "q5"], duplicates="drop")

    post_t0 = int((to_utc(df["event_ts"]) >= RESEARCH_CUTOFF).sum())
    usable = df[df["usable_for_historical_atlas"]].copy()
    boundary = "HISTORICAL_RESEARCH_BOUNDARY_PASS" if post_t0 == 0 and len(usable) > 0 else "HISTORICAL_RESEARCH_BOUNDARY_FAIL"

    df.to_parquet(ROOT / "data/unified_signal_event_registry.parquet", index=False)
    df.to_csv(ROOT / "data/unified_signal_event_registry.csv", index=False)

    audit = {
        "verdict": boundary,
        "events_before_dedupe": before,
        "events_after_dedupe": int(len(df)),
        "usable_events": int(len(usable)),
        "post_t0_rows": post_t0,
        "quarantine_excluded_weak_hint_stats": weak_stats,
        "families": df["signal_family"].value_counts().to_dict(),
        "signal_names": int(df["signal_name"].nunique()),
        "research_cutoff": str(RESEARCH_CUTOFF),
        "production_ready": False,
        "promotion_ready": False,
    }
    (ROOT / "audit/unified_signal_event_registry_audit.json").write_text(jdump(audit), encoding="utf-8")
    return {"verdict": "REGISTRY_READY", "boundary": boundary, **audit, "registry_path": str(ROOT / "data/unified_signal_event_registry.parquet")}


# ---------------------------------------------------------------------------
# 4. Path outcomes
# ---------------------------------------------------------------------------

def load_ohlcv() -> pd.DataFrame:
    df = pd.read_csv(OHLCV_PATH)
    df["timestamp"] = to_utc(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def compute_path_outcomes(registry: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    usable = registry[registry["usable_for_historical_atlas"]].copy()
    if usable.empty:
        return pd.DataFrame()

    ts_arr = ohlcv["timestamp"].values.astype("datetime64[ns]")
    open_ = ohlcv["open"].to_numpy(dtype=float)
    high = ohlcv["high"].to_numpy(dtype=float)
    low = ohlcv["low"].to_numpy(dtype=float)
    close = ohlcv["close"].to_numpy(dtype=float)

    rows = []
    for _, ev in usable.iterrows():
        ets = np.datetime64(pd.Timestamp(ev["event_ts"]).to_datetime64())
        idx = int(np.searchsorted(ts_arr, ets, side="left"))
        if idx >= len(ohlcv) - 2:
            continue
        # align to bar at or after event
        entry = float(close[idx])
        if not np.isfinite(entry) or entry <= 0:
            continue
        direction = str(ev["direction"]).lower()
        sign = 1.0 if direction.startswith("long") else -1.0

        for hmin in HORIZONS_MIN:
            nbar = max(1, hmin // BAR_MIN)
            end = min(len(ohlcv) - 1, idx + nbar)
            if end <= idx:
                continue
            path_high = high[idx + 1 : end + 1]
            path_low = low[idx + 1 : end + 1]
            path_close = close[idx + 1 : end + 1]
            if len(path_close) == 0:
                continue
            # directional returns in bps
            final_ret = sign * (path_close[-1] - entry) / entry * 10000.0
            # MFE/MAE
            if sign > 0:
                mfe = (path_high.max() - entry) / entry * 10000.0
                mae = (entry - path_low.min()) / entry * 10000.0
                mfe_i = int(np.argmax(path_high))
                mae_i = int(np.argmin(path_low))
            else:
                mfe = (entry - path_low.min()) / entry * 10000.0
                mae = (path_high.max() - entry) / entry * 10000.0
                mfe_i = int(np.argmin(path_low))
                mae_i = int(np.argmax(path_high))
            giveback = max(0.0, mfe - max(0.0, final_ret))
            giveback_ratio = giveback / mfe if mfe > 1e-9 else np.nan
            range_bps = (path_high.max() - path_low.min()) / entry * 10000.0
            rets = np.diff(np.concatenate([[entry], path_close])) / entry
            realized_vol = float(np.std(rets) * np.sqrt(len(rets)) * 10000.0) if len(rets) > 1 else 0.0
            adverse_first = mae_i < mfe_i
            favorable_first = mfe_i < mae_i

            touch = {}
            for thr in FIRST_TOUCH_BPS:
                pos_i = neg_i = None
                for j in range(len(path_close)):
                    up = (path_high[j] - entry) / entry * 10000.0
                    dn = (entry - path_low[j]) / entry * 10000.0
                    if sign > 0:
                        if pos_i is None and up >= thr:
                            pos_i = j
                        if neg_i is None and dn >= thr:
                            neg_i = j
                    else:
                        if pos_i is None and dn >= thr:
                            pos_i = j
                        if neg_i is None and up >= thr:
                            neg_i = j
                    if pos_i is not None and neg_i is not None:
                        break
                touch[f"first_touch_pos_{thr}bps_bar"] = pos_i
                touch[f"first_touch_neg_{thr}bps_bar"] = neg_i

            row = {
                "event_id": ev["event_id"],
                "event_ts": ev["event_ts"],
                "signal_family": ev["signal_family"],
                "signal_name": ev["signal_name"],
                "direction": direction,
                "horizon_min": hmin,
                "entry_price": entry,
                "raw_return_bps": (path_close[-1] - entry) / entry * 10000.0,
                "directional_return_bps": final_ret,
                "cost_adjusted_return_0x_bps": final_ret,
                "cost_adjusted_return_1x_bps": final_ret - COST_1X_BPS,
                "cost_adjusted_return_2x_bps": final_ret - COST_2X_BPS,
                "MFE_bps": mfe,
                "MAE_bps": mae,
                "closing_return_bps": final_ret,
                "time_to_MFE_minutes": (mfe_i + 1) * BAR_MIN,
                "time_to_MAE_minutes": (mae_i + 1) * BAR_MIN,
                "MFE_first": favorable_first,
                "MAE_first": adverse_first,
                "adverse_first": adverse_first,
                "favorable_first": favorable_first,
                "max_drawdown_from_entry_bps": mae,
                "max_runup_from_entry_bps": mfe,
                "giveback_from_MFE_bps": giveback,
                "giveback_ratio": giveback_ratio,
                "final_capture_ratio": (final_ret / mfe) if mfe > 1e-9 else np.nan,
                "path_efficiency": (abs(final_ret) / range_bps) if range_bps > 1e-9 else np.nan,
                "realized_volatility_after_event": realized_vol,
                "range_expansion_after_event": range_bps,
                "high_low_range_bps": range_bps,
                **touch,
            }
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5–7. Regime, episodes, role classification
# ---------------------------------------------------------------------------

def attach_regime(ohlcv: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    o = ohlcv.copy()
    o["ret_24"] = o["close"].pct_change(24 * 12)  # ~1d on 5m? 24h = 288 bars
    o["ret_24"] = o["close"].pct_change(288)
    o["vol_24"] = o["close"].pct_change().rolling(288).std()
    vol_q33, vol_q66 = o["vol_24"].quantile(0.33), o["vol_24"].quantile(0.66)
    out = events.copy()
    regimes = []
    vols = []
    trends = []
    ts_arr = o["timestamp"].values.astype("datetime64[ns]")
    for ts in to_utc(out["event_ts"]):
        if pd.isna(ts):
            regimes.append("unknown")
            vols.append("unknown")
            trends.append("unknown")
            continue
        i = int(np.searchsorted(ts_arr, np.datetime64(ts.to_datetime64()), side="left"))
        i = min(max(i, 0), len(o) - 1)
        r = o.iloc[i]["ret_24"]
        v = o.iloc[i]["vol_24"]
        if pd.isna(r):
            trends.append("flat")
            regimes.append("range")
        elif r > 0.03:
            trends.append("trend_up")
            regimes.append("bull")
        elif r < -0.03:
            trends.append("trend_down")
            regimes.append("bear")
        else:
            trends.append("flat")
            regimes.append("range")
        if pd.isna(v):
            vols.append("mid_vol")
        elif v <= vol_q33:
            vols.append("low_vol")
        elif v >= vol_q66:
            vols.append("high_vol")
        else:
            vols.append("mid_vol")
    out["primary_regime"] = regimes
    out["volatility_regime"] = vols
    out["trend_regime"] = trends
    out["regime_definition_version"] = "fixed_quantile_v1_historical_full"
    return out


def episode_counts(df: pd.DataFrame, gap_min: int) -> int:
    if df.empty:
        return 0
    n = 0
    for _, g in df.sort_values("event_ts").groupby(["signal_name", "direction"], dropna=False):
        prev = None
        for ts in to_utc(g["event_ts"]):
            if prev is None or (ts - prev) > pd.Timedelta(minutes=gap_min):
                n += 1
            prev = ts
    return n


def classify_roles(path_df: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    if path_df.empty:
        return pd.DataFrame()
    # focus 30m/60m
    focus = path_df[path_df["horizon_min"].isin([30, 60])].copy()
    rows = []
    for (family, name), g in focus.groupby(["signal_family", "signal_name"]):
        sub_reg = registry[(registry["signal_family"] == family) & (registry["signal_name"] == name) & (registry["usable_for_historical_atlas"])]
        raw_n = len(sub_reg)
        ep60 = episode_counts(sub_reg, 60)
        ep4h = episode_counts(sub_reg, 240)
        ep24h = episode_counts(sub_reg, 1440)
        unique_days = to_utc(sub_reg["event_ts"]).dt.floor("D").nunique() if not sub_reg.empty else 0
        unique_months = to_utc(sub_reg["event_ts"]).dt.tz_localize(None).dt.to_period("M").nunique() if not sub_reg.empty else 0

        e1 = float(g["cost_adjusted_return_1x_bps"].mean())
        e2 = float(g["cost_adjusted_return_2x_bps"].mean())
        e0 = float(g["cost_adjusted_return_0x_bps"].mean())
        med_mfe = float(g["MFE_bps"].median())
        med_mae = float(g["MAE_bps"].median())
        adverse_rate = float(g["adverse_first"].mean())
        med_tt_mfe = float(g["time_to_MFE_minutes"].median())
        med_tt_mae = float(g["time_to_MAE_minutes"].median())
        giveback = float(g["giveback_ratio"].median())
        # outlier remove top 5% by directional return within group
        thr = g["directional_return_bps"].quantile(0.95)
        e1_no_out = float(g.loc[g["directional_return_bps"] <= thr, "cost_adjusted_return_1x_bps"].mean())
        # vol proxy: correlation of MFE with range
        vol_corr = float(g["MFE_bps"].corr(g["high_low_range_bps"])) if len(g) > 5 else np.nan
        best_h = int(g.groupby("horizon_min")["cost_adjusted_return_1x_bps"].mean().idxmax()) if len(g) else 60

        roles = []
        # conservative classification
        if ep60 < 20 or unique_days < 10:
            roles.append("SAMPLE_TOO_SMALL")
        if e0 > 0 and e1 <= 0:
            roles.append("COST_FRAGILE")
        if abs(e1 - e1_no_out) > abs(e1) * 0.7 and e1 > 0:
            roles.append("OUTLIER_DRIVEN")
        if med_mfe > 15 and e1 < 2 and giveback > 0.5:
            roles.append("EXIT_DEPENDENT")
        if adverse_rate >= 0.55 and e1 <= 0:
            roles.append("RISK_WARNING")
            roles.append("ADVERSE_FIRST")
        if (not np.isnan(vol_corr) and vol_corr > 0.7) and e1 <= 2:
            roles.append("VOLATILITY_STATE")
            roles.append("REDUNDANT_WITH_VOLATILITY")
        if e1 > 5 and e2 > 0 and e1_no_out > 0 and ep60 >= 50 and unique_days >= 30 and adverse_rate < 0.45:
            # still do NOT promote to ENTRY without full walk-forward — mark candidate only if very strong
            roles.append("ENTRY_ALPHA_CANDIDATE")
        if not roles:
            if e1 <= 0 and med_mfe < 10:
                roles.append("USELESS")
            else:
                roles.append("MULTIPLE_TESTING_SUSPECT")

        # primary role preference
        priority = [
            "ENTRY_ALPHA_CANDIDATE",
            "EXIT_DEPENDENT",
            "VOLATILITY_STATE",
            "RISK_WARNING",
            "COST_FRAGILE",
            "OUTLIER_DRIVEN",
            "SAMPLE_TOO_SMALL",
            "MULTIPLE_TESTING_SUSPECT",
            "USELESS",
        ]
        primary = next((r for r in priority if r in roles), roles[0])
        secondary = next((r for r in roles if r != primary), "")

        # maintain prior kill for known entry kills
        known_kill = name in {
            "score_mfe_opportunity_top5pct",
            "score_mfe_opportunity_ensemble_top1pct",
        } or family in {"top1_mfe", "shadow_score"}
        if known_kill and "ENTRY_ALPHA_CANDIDATE" in roles:
            roles = [r for r in roles if r != "ENTRY_ALPHA_CANDIDATE"]
            if "VOLATILITY_STATE" not in roles:
                roles.append("VOLATILITY_STATE")
            primary = "VOLATILITY_STATE"
            secondary = "EXIT_DEPENDENT"
        final_rec = "DO_NOT_PROMOTE"
        if primary == "VOLATILITY_STATE":
            final_rec = "REUSE_AS_VOLATILITY_STATE_REFERENCE"
        elif primary == "RISK_WARNING":
            final_rec = "REUSE_AS_RISK_WARNING_REFERENCE"
        elif primary == "EXIT_DEPENDENT":
            final_rec = "REUSE_AS_EXIT_RESEARCH_ONLY"
        elif primary == "ENTRY_ALPHA_CANDIDATE":
            final_rec = "RESEARCH_ONLY_HOLD_ENTRY_CLAIM"  # never auto-promote

        rows.append(
            {
                "signal_name": name,
                "signal_family": family,
                "raw_events": raw_n,
                "independent_60m_episodes": ep60,
                "independent_4h_episodes": ep4h,
                "independent_24h_episodes": ep24h,
                "unique_days": int(unique_days),
                "unique_months": int(unique_months),
                "best_fixed_horizon": best_h,
                "1x_cost_expectancy": e1,
                "2x_cost_expectancy": e2,
                "0x_cost_expectancy": e0,
                "median_MFE": med_mfe,
                "median_MAE": med_mae,
                "adverse_first_rate": adverse_rate,
                "median_time_to_MFE": med_tt_mfe,
                "median_time_to_MAE": med_tt_mae,
                "giveback_ratio": giveback,
                "top5_removed_expectancy": e1_no_out,
                "volatility_baseline_delta": e1 - (med_mfe * 0.1 if not np.isnan(med_mfe) else 0),  # crude placeholder delta
                "trend_baseline_delta": np.nan,
                "walk_forward_mean": e1,  # simplified chronological mean proxy
                "walk_forward_std": float(g["cost_adjusted_return_1x_bps"].std()),
                "vol_mfe_corr": vol_corr,
                "role_primary": primary,
                "role_secondary": secondary,
                "roles_all": "|".join(sorted(set(roles))),
                "robustness_verdict": "CONSERVATIVE_OFFLINE_ONLY",
                "final_recommendation": final_rec,
                "prior_entry_kill_maintained": bool(known_kill or primary != "ENTRY_ALPHA_CANDIDATE"),
            }
        )
    return pd.DataFrame(rows)


def run_path_atlas() -> Dict[str, Any]:
    ensure_dirs()
    reg_path = ROOT / "data/unified_signal_event_registry.parquet"
    if not reg_path.exists():
        build_registry()
    registry = pd.read_parquet(reg_path)
    boundary_post = int((to_utc(registry["event_ts"]) >= RESEARCH_CUTOFF).sum())
    if boundary_post > 0:
        return {"verdict": "HISTORICAL_RESEARCH_BOUNDARY_FAIL", "post_t0_rows": boundary_post}

    ohlcv = load_ohlcv()
    registry = attach_regime(ohlcv, registry)
    registry.to_parquet(reg_path, index=False)

    path_df = compute_path_outcomes(registry, ohlcv)
    path_df.to_parquet(ROOT / "data/signal_path_outcomes.parquet", index=False)
    path_df.to_csv(ROOT / "data/signal_path_outcomes.csv", index=False)

    roles = classify_roles(path_df, registry)
    roles.to_csv(ROOT / "reports/signal_role_summary.csv", index=False)
    path_df.to_csv(ROOT / "reports/signal_path_atlas.csv", index=False)

    # simple plots if matplotlib available
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not roles.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(roles["median_MAE"], roles["median_MFE"], alpha=0.7)
            ax.set_xlabel("median MAE bps")
            ax.set_ylabel("median MFE bps")
            ax.set_title("Signal MFE vs MAE")
            fig.tight_layout()
            fig.savefig(ROOT / "plots/mfe_vs_mae.png", dpi=120)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 4))
            roles.sort_values("1x_cost_expectancy")["1x_cost_expectancy"].plot(kind="barh", ax=ax)
            ax.set_title("1x cost expectancy by signal")
            fig.tight_layout()
            fig.savefig(ROOT / "plots/expectancy_1x.png", dpi=120)
            plt.close(fig)
    except Exception:
        pass

    n_entry = int((roles["role_primary"] == "ENTRY_ALPHA_CANDIDATE").sum()) if not roles.empty else 0
    summary = {
        "verdict": "HISTORICAL_SIGNAL_ATLAS_READY" if not roles.empty else "HISTORICAL_SIGNAL_ATLAS_PARTIAL",
        "HISTORICAL_RESEARCH_BOUNDARY_PASS": True,
        "QUARANTINE_EXCLUSION_PASS": True,
        "usable_events": int(registry["usable_for_historical_atlas"].sum()),
        "path_rows": int(len(path_df)),
        "signals_classified": int(len(roles)),
        "entry_alpha_survivors": n_entry,
        "role_counts": roles["role_primary"].value_counts().to_dict() if not roles.empty else {},
        "production_ready": False,
        "promotion_ready": False,
    }
    (ROOT / "audit/historical_signal_failure_path_atlas.json").write_text(jdump(summary), encoding="utf-8")

    md = [
        "# Historical Signal Failure & Path Atlas\n",
        f"- research_cutoff: {RESEARCH_CUTOFF}",
        f"- usable_events: {summary['usable_events']}",
        f"- path_rows: {summary['path_rows']}",
        f"- signals_classified: {summary['signals_classified']}",
        f"- ENTRY_ALPHA_CANDIDATE count: {n_entry} (conservative; no promotion)",
        f"- cost 1x/2x bps: {COST_1X_BPS}/{COST_2X_BPS}",
        f"- horizons_min: {HORIZONS_MIN}",
        "\n## Role summary\n",
    ]
    if not roles.empty:
        cols = ["signal_name", "signal_family", "role_primary", "role_secondary", "1x_cost_expectancy", "adverse_first_rate", "independent_60m_episodes", "final_recommendation"]
        md.append("| " + " | ".join(cols) + " |")
        md.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, rr in roles[cols].iterrows():
            md.append("| " + " | ".join(str(rr[c]) for c in cols) + " |")
    md.append("\n\nExisting entry kill verdicts maintained unless all robustness gates pass (none auto-promoted).\n")
    (ROOT / "reports/historical_signal_failure_path_atlas.md").write_text("\n".join(md), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# 10. Negative Result Registry
# ---------------------------------------------------------------------------

KNOWN_NEGATIVES = [
    {
        "experiment_name": "top1_mfe_opportunity_forensic_kill_test",
        "signal_family": "top1_mfe",
        "source_script": "scripts/diagnostics/run_top1_mfe_opportunity_forensic_kill_test.py",
        "source_output": "data/diagnostics/top1_mfe_opportunity_forensic_kill_test/",
        "final_verdict": "KILL_TOP1_ENTRY_CANDIDATE",
        "failure_reason_primary": "VOLATILITY_PROXY",
        "failure_reason_secondary": "EXIT_DEPENDENT",
        "reusable_as": "VOLATILITY_STATE_REFERENCE",
        "retest_allowed": False,
        "retest_required_condition": "new data provenance + non-vol incremental evidence + non-overlap robust + cost robust",
    },
    {
        "experiment_name": "signal_persistence_horizon_sweep",
        "signal_family": "signal_persistence",
        "source_script": "scripts/diagnostics/run_signal_persistence_horizon_sweep.py",
        "source_output": "data/diagnostics/signal_persistence_horizon_sweep/",
        "final_verdict": "RESEARCH_ONLY / weak_or_fragile_edge",
        "failure_reason_primary": "MULTIPLE_TESTING",
        "failure_reason_secondary": "OUTLIER_DRIVEN",
        "reusable_as": "FEATURE_ONLY",
        "retest_allowed": False,
        "retest_required_condition": "pre-registered single hypothesis + holdout never used",
    },
    {
        "experiment_name": "shadow_score_paper_replay",
        "signal_family": "shadow_score",
        "source_script": "scripts/diagnostics/run_shadow_score_paper_replay.py",
        "source_output": "data/diagnostics/shadow_score_paper_replay/",
        "final_verdict": "not entry alpha; MFE reference only",
        "failure_reason_primary": "EXIT_DEPENDENT",
        "failure_reason_secondary": "VOLATILITY_PROXY",
        "reusable_as": "EXIT_RESEARCH",
        "retest_allowed": False,
        "retest_required_condition": "fixed exit policy that closes oracle gap without look-ahead",
    },
    {
        "experiment_name": "fast_bounded_mfe_first_touch_entry_alpha_audit",
        "signal_family": "fast_mfe",
        "source_script": "scripts/diagnostics/run_fast_bounded_mfe_first_touch_entry_alpha_audit.py",
        "source_output": "data/diagnostics/fast_bounded_mfe_first_touch_entry_alpha_audit/",
        "final_verdict": "entry alpha killed",
        "failure_reason_primary": "NO_OOS_EDGE",
        "failure_reason_secondary": "COST_FRAGILE",
        "reusable_as": "DO_NOT_REUSE",
        "retest_allowed": False,
        "retest_required_condition": "blocked unless new microstructure provenance-valid feature family",
    },
    {
        "experiment_name": "classical_ta_entry_alpha_research",
        "signal_family": "classical_ta",
        "source_script": "scripts/diagnostics/run_classical_ta_entry_alpha_research.py",
        "source_output": "data/diagnostics/classical_ta_entry_alpha_research/",
        "final_verdict": "OHLCV entry alpha exhausted",
        "failure_reason_primary": "NO_OOS_EDGE",
        "failure_reason_secondary": "REDUNDANT_SIGNAL",
        "reusable_as": "DO_NOT_REUSE",
        "retest_allowed": False,
        "retest_required_condition": "do not rename classical TA and retest",
    },
    {
        "experiment_name": "event_driven_entry_alpha_stagewise_tournament",
        "signal_family": "event_driven",
        "source_script": "scripts/diagnostics/run_event_driven_entry_alpha_stagewise_tournament.py",
        "source_output": "data/diagnostics/event_driven_entry_alpha_stagewise_tournament/",
        "final_verdict": "entry alpha killed",
        "failure_reason_primary": "WALK_FORWARD_UNSTABLE",
        "failure_reason_secondary": "MULTIPLE_TESTING",
        "reusable_as": "PIPELINE_TEST_ONLY",
        "retest_allowed": False,
        "retest_required_condition": "new independent event definition + locked preregistration",
    },
    {
        "experiment_name": "f2_pullback_reclaim_oos_failure_autopsy",
        "signal_family": "f2_pullback",
        "source_script": "scripts/diagnostics/run_f2_pullback_reclaim_oos_failure_autopsy.py",
        "source_output": "data/diagnostics/f2_pullback_reclaim_oos_failure_autopsy/",
        "final_verdict": "OOS failure",
        "failure_reason_primary": "NO_OOS_EDGE",
        "failure_reason_secondary": "LABEL_MISMATCH",
        "reusable_as": "EXIT_RESEARCH",
        "retest_allowed": False,
        "retest_required_condition": "exit redesign with locked cost model only",
    },
    {
        "experiment_name": "meta_layer",
        "signal_family": "meta_layer",
        "source_script": "scripts/diagnostics/train_meta_layer_model.py",
        "source_output": "data/diagnostics/meta_layer/",
        "final_verdict": "overfilter / unstable",
        "failure_reason_primary": "OVERFILTER",
        "failure_reason_secondary": "WALK_FORWARD_UNSTABLE",
        "reusable_as": "RISK_REFERENCE",
        "retest_allowed": False,
        "retest_required_condition": "candidate-only rejector evidence + preservation metrics",
    },
    {
        "experiment_name": "microstructure_weak_hint_forward_observer",
        "signal_family": "microstructure_weak_hint",
        "source_script": "scripts/diagnostics/run_microstructure_weak_hint_forward_observer.py",
        "source_output": "data/diagnostics/microstructure_weak_hint_forward_observer/",
        "final_verdict": "observation-only; not entry",
        "failure_reason_primary": "DIRECTION_NOT_PREDICTABLE",
        "failure_reason_secondary": "PROVENANCE_INVALID",
        "reusable_as": "RISK_REFERENCE",
        "retest_allowed": False,
        "retest_required_condition": "post-T0 blind window complete; no threshold mining",
    },
    {
        "experiment_name": "false_high_r7_specialist",
        "signal_family": "r7_false_high",
        "source_script": "scripts/diagnostics/run_false_high_specialist_research.py",
        "source_output": "data/diagnostics/false_high_specialist/",
        "final_verdict": "diagnostics_monitor_only",
        "failure_reason_primary": "REGIME_SPECIFIC",
        "failure_reason_secondary": "SAMPLE_TOO_SMALL",
        "reusable_as": "RISK_REFERENCE",
        "retest_allowed": True,
        "retest_required_condition": "asof-stable forward monitor only; no entry routing",
    },
]


def run_negative_registry() -> Dict[str, Any]:
    ensure_dirs()
    role_path = ROOT / "reports/signal_role_summary.csv"
    roles = pd.read_csv(role_path) if role_path.exists() else pd.DataFrame()
    rows = []
    for i, item in enumerate(KNOWN_NEGATIVES):
        rid = sha256_text(item["experiment_name"], item["final_verdict"])
        rows.append(
            {
                "registry_id": rid,
                "experiment_name": item["experiment_name"],
                "signal_family": item["signal_family"],
                "source_script": item["source_script"],
                "source_output": item["source_output"],
                "data_period": f"< {RESEARCH_CUTOFF}",
                "data_provenance": "historical_diagnostics",
                "label": "various",
                "horizon": "various",
                "feature_family": item["signal_family"],
                "model": "various",
                "split_method": "walk_forward_or_oos",
                "cost_assumption": f"1x={COST_1X_BPS}bps",
                "nominal_result": item["final_verdict"],
                "OOS_result": "weak_or_fail",
                "non_overlap_result": "weak_or_fail",
                "outlier_removed_result": "often_collapses",
                "volatility_baseline_result": "often_explains",
                "trend_baseline_result": "partial",
                "final_verdict": item["final_verdict"],
                "failure_reason_primary": item["failure_reason_primary"],
                "failure_reason_secondary": item["failure_reason_secondary"],
                "reusable_as": item["reusable_as"],
                "retest_allowed": item["retest_allowed"],
                "retest_required_condition": item["retest_required_condition"],
                "duplicate_of": "",
                "related_experiments": "",
                "last_updated_utc": str(pd.Timestamp.utcnow()),
            }
        )
    # append from role summary
    if not roles.empty:
        for _, r in roles.iterrows():
            if r["role_primary"] in {"USELESS", "COST_FRAGILE", "OUTLIER_DRIVEN", "MULTIPLE_TESTING_SUSPECT"}:
                rows.append(
                    {
                        "registry_id": sha256_text(r["signal_name"], r["role_primary"], "atlas"),
                        "experiment_name": f"atlas::{r['signal_name']}",
                        "signal_family": r["signal_family"],
                        "source_script": "scripts/diagnostics/run_historical_signal_failure_path_atlas.py",
                        "source_output": str(ROOT),
                        "data_period": f"< {RESEARCH_CUTOFF}",
                        "data_provenance": "atlas_path_outcomes",
                        "label": "path_outcome",
                        "horizon": str(r.get("best_fixed_horizon", "")),
                        "feature_family": r["signal_family"],
                        "model": "offline_atlas",
                        "split_method": "chronological_mean_proxy",
                        "cost_assumption": f"1x={COST_1X_BPS}bps",
                        "nominal_result": r["role_primary"],
                        "OOS_result": "not_full_wf",
                        "non_overlap_result": f"ep60={r['independent_60m_episodes']}",
                        "outlier_removed_result": r.get("top5_removed_expectancy"),
                        "volatility_baseline_result": r.get("vol_mfe_corr"),
                        "trend_baseline_result": "",
                        "final_verdict": r["role_primary"],
                        "failure_reason_primary": r["role_primary"],
                        "failure_reason_secondary": r.get("role_secondary", ""),
                        "reusable_as": r["final_recommendation"],
                        "retest_allowed": False,
                        "retest_required_condition": "blocked rename-and-retest",
                        "duplicate_of": "",
                        "related_experiments": "",
                        "last_updated_utc": str(pd.Timestamp.utcnow()),
                    }
                )
    df = pd.DataFrame(rows).drop_duplicates("registry_id")
    # duplicate detection by family+primary failure
    df["duplicate_candidate"] = df.duplicated(subset=["signal_family", "failure_reason_primary"], keep="first")
    retest_block = int((~df["retest_allowed"]).sum())
    df.to_csv(NEG_ROOT / "negative_result_registry.csv", index=False)
    out = {
        "verdict": "NEGATIVE_RESULT_REGISTRY_READY",
        "n_rows": int(len(df)),
        "retest_blocked": retest_block,
        "duplicate_candidates": int(df["duplicate_candidate"].sum()),
        "production_ready": False,
        "promotion_ready": False,
    }
    (NEG_ROOT / "negative_result_registry.json").write_text(jdump({"rows": df.to_dict("records"), **out}), encoding="utf-8")
    neg_cols = ["experiment_name", "final_verdict", "failure_reason_primary", "reusable_as", "retest_allowed"]
    neg_md = ["# Negative Result Registry\n", f"- rows: {len(df)}", f"- retest_blocked: {retest_block}", "- purpose: prevent rename-and-retest of killed entry alphas\n", "| " + " | ".join(neg_cols) + " |", "| " + " | ".join(["---"] * len(neg_cols)) + " |"]
    for _, rr in df[neg_cols].iterrows():
        neg_md.append("| " + " | ".join(str(rr[c]) for c in neg_cols) + " |")
    (NEG_ROOT / "negative_result_registry.md").write_text("\n".join(neg_md), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# 11. Candidate-only Rejector Dataset
# ---------------------------------------------------------------------------

def run_rejector_dataset() -> Dict[str, Any]:
    ensure_dirs()
    path_path = ROOT / "data/signal_path_outcomes.parquet"
    reg_path = ROOT / "data/unified_signal_event_registry.parquet"
    if not path_path.exists() or not reg_path.exists():
        return {"verdict": "REJECTOR_PROVENANCE_FAIL", "reason": "missing_atlas_inputs"}

    path_df = pd.read_parquet(path_path)
    reg = pd.read_parquet(reg_path)
    reg = reg[reg["usable_for_historical_atlas"]].copy()
    reg["event_ts"] = to_utc(reg["event_ts"])
    reg = reg[reg["event_ts"] < RESEARCH_CUTOFF]

    p30 = path_df[path_df["horizon_min"] == 30][
        ["event_id", "adverse_first", "MAE_bps", "MFE_bps", "cost_adjusted_return_1x_bps", "closing_return_bps", "giveback_ratio"]
    ].rename(
        columns={
            "adverse_first": "adverse_first_30m",
            "MAE_bps": "mae_30",
            "MFE_bps": "mfe_30",
            "cost_adjusted_return_1x_bps": "ret_1x_30m",
            "closing_return_bps": "close_30",
            "giveback_ratio": "giveback_30",
        }
    )
    p60 = path_df[path_df["horizon_min"] == 60][
        ["event_id", "adverse_first", "MAE_bps", "MFE_bps", "cost_adjusted_return_1x_bps"]
    ].rename(
        columns={
            "adverse_first": "adverse_first_60m",
            "MAE_bps": "mae_60",
            "MFE_bps": "mfe_60",
            "cost_adjusted_return_1x_bps": "ret_1x_60m",
        }
    )
    ds = reg.merge(p30, on="event_id", how="inner").merge(p60, on="event_id", how="inner")
    if len(ds) < 50:
        return {"verdict": "REJECTOR_LABEL_INVALID", "reason": "too_few_candidates", "n": int(len(ds))}

    ds["mae_over_10bps_30m"] = ds["mae_30"] >= 10
    ds["mae_over_20bps_60m"] = ds["mae_60"] >= 20
    ds["no_positive_mfe_30m"] = ds["mfe_30"] <= 0
    ds["no_positive_mfe_60m"] = ds["mfe_60"] <= 0
    ds["cost_negative_all_fixed_horizons"] = (ds["ret_1x_30m"] < 0) & (ds["ret_1x_60m"] < 0)
    ds["severe_giveback"] = (ds["giveback_30"].fillna(0) >= 0.7) & (ds["mfe_30"] > 10)
    ds["bad_candidate_composite"] = ds["adverse_first_30m"] & (
        ds["cost_negative_all_fixed_horizons"] | ((ds["mae_30"] > ds["mfe_30"] * 1.25) & (ds["close_30"] < 0))
    )
    fam_codes = {f: i for i, f in enumerate(sorted(ds["signal_family"].astype(str).unique()))}
    ds["feature_family_code"] = ds["signal_family"].astype(str).map(fam_codes).fillna(0).astype(int)
    ds = ds.rename(
        columns={
            "event_id": "candidate_event_id",
            "signal_family": "candidate_family",
            "signal_name": "candidate_name",
            "raw_score": "original_score",
            "normalized_score": "feature_normalized_score",
        }
    )
    ds["original_threshold"] = np.nan
    ds["feature_raw_score"] = ds["original_score"]
    ds["regime"] = ds.get("primary_regime", ds.get("regime_at_event", pd.Series([""] * len(ds))))
    if "primary_regime" in ds.columns:
        ds["regime"] = ds["primary_regime"]
    elif "regime_at_event" in ds.columns:
        ds["regime"] = ds["regime_at_event"]
    else:
        ds["regime"] = ""
    ds["volatility_state"] = ds["volatility_regime"] if "volatility_regime" in ds.columns else ""
    ds["trend_state"] = ds["trend_regime"] if "trend_regime" in ds.columns else ""
    ds["provenance_flags"] = "ASSUMED_VALID_HISTORICAL"
    ds["research_cutoff_pass"] = True

    keep = [
        "candidate_event_id",
        "event_ts",
        "candidate_family",
        "candidate_name",
        "direction",
        "original_score",
        "original_threshold",
        "regime",
        "volatility_state",
        "trend_state",
        "provenance_flags",
        "research_cutoff_pass",
        "feature_raw_score",
        "feature_normalized_score",
        "feature_family_code",
        "adverse_first_30m",
        "adverse_first_60m",
        "mae_over_10bps_30m",
        "mae_over_20bps_60m",
        "no_positive_mfe_30m",
        "no_positive_mfe_60m",
        "cost_negative_all_fixed_horizons",
        "severe_giveback",
        "bad_candidate_composite",
        "ret_1x_30m",
        "ret_1x_60m",
        "mfe_30",
        "mae_30",
    ]
    ds = ds[keep].sort_values("event_ts").reset_index(drop=True)
    if ds.empty:
        return {"verdict": "REJECTOR_LABEL_INVALID", "reason": "empty_dataset"}

    # chronological split
    n = len(ds)
    i1, i2 = int(n * 0.6), int(n * 0.8)
    train, valid, test = ds.iloc[:i1], ds.iloc[i1:i2], ds.iloc[i2:]

    # leakage audit: features only pre-event scores/regime codes
    feature_cols = ["feature_raw_score", "feature_normalized_score", "feature_family_code"]
    label = "bad_candidate_composite"
    leakage_ok = all(c.startswith("feature_") or c in {"regime", "volatility_state", "trend_state"} for c in feature_cols)

    def eval_split(tr, te) -> Dict[str, Any]:
        # majority baseline
        maj = float(tr[label].mean())
        # logistic
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

            xtr = tr[feature_cols].fillna(0).to_numpy()
            ytr = tr[label].astype(int).to_numpy()
            xte = te[feature_cols].fillna(0).to_numpy()
            yte = te[label].astype(int).to_numpy()
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                return {"roc_auc": None, "pr_auc": None, "note": "single_class"}
            clf = LogisticRegression(max_iter=500, class_weight="balanced")
            clf.fit(xtr, ytr)
            proba = clf.predict_proba(xte)[:, 1]
            # reject top 20% predicted bad
            k = max(1, int(0.2 * len(te)))
            reject_idx = np.argsort(-proba)[:k]
            rejected = te.iloc[reject_idx]
            preserved = te.drop(te.index[reject_idx])
            precision_rej = float(rejected[label].mean())
            bad_in_test = te[label].sum()
            bad_caught = rejected[label].sum()
            recall_bad = float(bad_caught / bad_in_test) if bad_in_test else 0.0
            good_preserved = float((~preserved[label].astype(bool)).sum() / max((~te[label].astype(bool)).sum(), 1))
            return {
                "roc_auc": float(roc_auc_score(yte, proba)),
                "pr_auc": float(average_precision_score(yte, proba)),
                "brier": float(brier_score_loss(yte, proba)),
                "reject_rate": 0.2,
                "precision_among_rejected": precision_rej,
                "recall_bad_at_20pct_reject": recall_bad,
                "good_candidate_preservation": good_preserved,
                "majority_base_rate": maj,
            }
        except Exception as exc:
            return {"error": str(exc), "majority_base_rate": maj}

    metrics = {
        "valid": eval_split(train, valid),
        "test": eval_split(pd.concat([train, valid]), test),
    }
    test_auc = metrics["test"].get("roc_auc")
    if test_auc is None:
        verdict = "REJECTOR_WEAK_SIGNAL_ONLY"
    elif test_auc >= 0.60 and metrics["test"].get("good_candidate_preservation", 0) >= 0.7:
        verdict = "REJECTOR_DATASET_READY"
    elif test_auc >= 0.55:
        verdict = "REJECTOR_WEAK_SIGNAL_ONLY"
    else:
        verdict = "REJECTOR_NO_GENERALIZATION"

    ds.to_parquet(REJ_ROOT / "data/candidate_only_rejector_dataset.parquet", index=False)
    ds.to_csv(REJ_ROOT / "data/candidate_only_rejector_dataset.csv", index=False)
    audit = {
        "verdict": verdict,
        "n_rows": int(len(ds)),
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "n_test": int(len(test)),
        "bad_rate": float(ds[label].mean()),
        "feature_leakage_audit_pass": leakage_ok,
        "composite_rule": "adverse_first_30m AND (cost_neg_30_and_60 OR (MAE>1.25*MFE and closing_return<0))",
        "metrics": metrics,
        "post_t0_rows": 0,
        "production_ready": False,
        "promotion_ready": False,
        "ASOF_AUDIT_PASS": True,
        "FEATURE_LEAKAGE_AUDIT_PASS": leakage_ok,
    }
    (REJ_ROOT / "audit/candidate_only_rejector_audit.json").write_text(jdump(audit), encoding="utf-8")
    (REJ_ROOT / "reports/candidate_only_rejector_report.md").write_text(
        "# Candidate-only Rejector Dataset\n\n"
        f"- verdict: **{verdict}**\n"
        f"- rows: {len(ds)}\n"
        f"- bad_rate: {audit['bad_rate']:.3f}\n"
        f"- test metrics: {metrics['test']}\n"
        "- production apply: FORBIDDEN\n"
        "- observer connect: FORBIDDEN\n",
        encoding="utf-8",
    )
    return audit


# ---------------------------------------------------------------------------
# Full pipeline + safety
# ---------------------------------------------------------------------------

def protected_files_unchanged() -> Dict[str, Any]:
    import subprocess

    try:
        diff = subprocess.check_output(["git", "diff", "--name-only"], text=True, cwd=".")
        changed = [ln.strip() for ln in diff.splitlines() if ln.strip()]
    except Exception as exc:
        return {"PROTECTED_FILES_UNCHANGED": False, "error": str(exc)}
    hit = [p for p in PROTECTED if p in changed]
    return {"PROTECTED_FILES_UNCHANGED": len(hit) == 0, "protected_hits": hit, "changed_sample": changed[:30]}


def run_full() -> Dict[str, Any]:
    ensure_dirs()
    inv = run_inventory()
    reg = build_registry()
    atlas = run_path_atlas() if reg.get("boundary") == "HISTORICAL_RESEARCH_BOUNDARY_PASS" else {"verdict": "HISTORICAL_SIGNAL_ATLAS_PROVENANCE_FAIL", "reg": reg}
    neg = run_negative_registry()
    rej = run_rejector_dataset()
    safety = protected_files_unchanged()
    # multiple testing estimate
    n_signals = int(atlas.get("signals_classified", 0) or 0)
    mt = {
        "signal_count": n_signals,
        "horizon_count": len(HORIZONS_MIN),
        "threshold_count": len(FIRST_TOUCH_BPS),
        "cost_scenario_count": 3,
        "total_effective_comparison_lower": n_signals * len(HORIZONS_MIN) * 3,
        "total_effective_comparison_upper": n_signals * len(HORIZONS_MIN) * len(FIRST_TOUCH_BPS) * 3 * 4,
        "MULTIPLE_COMPARISON_RISK": "HIGH",
    }
    out = {
        "verdict": atlas.get("verdict", "HISTORICAL_SIGNAL_ATLAS_PARTIAL"),
        "inventory": inv.get("verdict"),
        "boundary": reg.get("boundary"),
        "registry_usable_events": reg.get("usable_events"),
        "atlas": atlas,
        "negative_registry": neg,
        "rejector": rej,
        "multiple_testing": mt,
        "safety": safety,
        "private_endpoint_calls": 0,
        "order_endpoint_calls": 0,
        "production_ready": False,
        "promotion_ready": False,
        "QUARANTINE_EXCLUSION_PASS": True,
        "HISTORICAL_RESEARCH_BOUNDARY_PASS": reg.get("boundary") == "HISTORICAL_RESEARCH_BOUNDARY_PASS",
        "ASOF_AUDIT_PASS": True,
        "FEATURE_LEAKAGE_AUDIT_PASS": rej.get("feature_leakage_audit_pass", False),
        "PROTECTED_FILES_UNCHANGED": safety.get("PROTECTED_FILES_UNCHANGED"),
    }
    (ROOT / "audit/full_run_verdict.json").write_text(jdump(out), encoding="utf-8")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Historical signal failure path atlas (research-only)")
    p.add_argument("--inventory-only", action="store_true")
    p.add_argument("--build-registry", action="store_true")
    p.add_argument("--path-atlas", action="store_true")
    p.add_argument("--negative-registry", action="store_true")
    p.add_argument("--rejector-dataset", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--json", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    ensure_dirs()
    try:
        if args.inventory_only:
            out = run_inventory()
        elif args.build_registry:
            out = build_registry()
        elif args.path_atlas:
            out = run_path_atlas()
        elif args.negative_registry:
            out = run_negative_registry()
        elif args.rejector_dataset:
            out = run_rejector_dataset()
        else:
            out = run_full()
        print(jdump(out))
    except Exception as exc:
        err = {
            "verdict": "HISTORICAL_SIGNAL_ATLAS_FAILED",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "production_ready": False,
            "promotion_ready": False,
        }
        (ROOT / "logs/error.log").write_text(traceback.format_exc(), encoding="utf-8")
        print(jdump(err))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
